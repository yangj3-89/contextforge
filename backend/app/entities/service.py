"""Entity layer orchestration: known-name pass -> resolution -> relationships -> persist."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from app.core.config import EntityRetrievalSettings
from app.db.repository import Repository
from app.entities.extraction import derive_known_names, known_name_mentions
from app.entities.gazetteer import ORG_HEAD_WORDS
from app.entities.graph import EntityGraph
from app.entities.linking import QueryEntityLinker
from app.entities.relationships import extract_relationships
from app.entities.resolution import EntityResolver
from app.models.domain import Chunk, Mention, Relationship

DERIVED_EXTRACTORS = frozenset({"known_name", "header"})

logger = logging.getLogger(__name__)


@dataclass
class RebuildReport:
    stats: dict[str, int]
    elapsed_ms: float


class EntityService:
    def __init__(self, repo: Repository, resolver: EntityResolver, header_context: bool = True) -> None:
        self.repo = repo
        self.resolver = resolver
        self.header_context = header_context

    def rebuild(self) -> tuple[EntityGraph, RebuildReport]:
        t0 = time.perf_counter()
        base = [m for m in self.repo.all_mentions() if m.extractor not in DERIVED_EXTRACTORS]
        chunk_ids = self.repo.all_chunk_ids()
        chunks = self.repo.get_chunks(chunk_ids)

        known = self._known_name_pass(base, chunks)
        mentions = base + known
        embeddings = self.repo.get_embeddings(sorted({m.chunk_id for m in mentions}))
        result = self.resolver.resolve(mentions, embeddings)
        for m in mentions:
            m.entity_id = result.assignments.get(m.id)

        header = self._header_pass(chunks, EntityGraph(result.entities, [], [])) if self.header_context else []
        assignments = dict(result.assignments)
        assignments.update({m.id: m.entity_id for m in header if m.entity_id})
        context_by_chunk: dict[str, list[str]] = defaultdict(list)
        for m in header:
            context_by_chunk[m.chunk_id].append(m.entity_id)

        entity_types = {e.id: e.entity_type for e in result.entities}
        by_chunk: dict[str, list[Mention]] = defaultdict(list)
        for m in mentions:
            if m.entity_id:
                by_chunk[m.chunk_id].append(m)
        relationships: list[Relationship] = []
        for cid, chunk in chunks.items():
            if by_chunk.get(cid):
                relationships.extend(
                    extract_relationships(chunk, by_chunk[cid], entity_types, context_by_chunk.get(cid))
                )

        derived = known + header
        self.repo.replace_entity_graph(result.entities, assignments, relationships, result.decisions, derived)
        graph = EntityGraph(result.entities, mentions + header, relationships)
        stats = {
            **result.stats,
            "known_name_mentions": len(known),
            "header_context_mentions": len(header),
            "relationships": len(relationships),
        }
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Entity rebuild: %s in %.0f ms", stats, elapsed)
        return graph, RebuildReport(stats, round(elapsed, 1))

    @staticmethod
    def _header_pass(chunks: dict[str, Chunk], graph: EntityGraph) -> list[Mention]:
        """Entities named in a chunk's contextual header (document title / section path).

        A section such as "RFC: Project Lodestar > Implementation" is about Lodestar even
        if its paragraphs never repeat the name; those entities become context mentions
        of the chunk (exact alias matches only, unambiguous only).
        """
        linker = QueryEntityLinker(graph, EntityRetrievalSettings(min_link_confidence=1.0))
        out: list[Mention] = []
        for cid, chunk in chunks.items():
            header = chunk.search_text[: len(chunk.search_text) - len(chunk.text)].strip()
            if not header:
                continue
            seen: set[str] = set()
            for link in linker.link(header, exact_only=True).linked:
                if link.entity_id in seen or link.entity_type == "DATE":
                    continue
                seen.add(link.entity_id)
                out.append(
                    Mention(
                        id=f"{cid}:h{len(seen):02d}",
                        chunk_id=cid,
                        document_id=chunk.document_id,
                        surface=link.surface,
                        entity_type=link.entity_type,
                        char_start=0,
                        char_end=0,
                        extractor="header",
                        confidence=link.confidence,
                        entity_id=link.entity_id,
                    )
                )
        return out

    def load_graph(self) -> EntityGraph:
        return EntityGraph(self.repo.list_entities(), self.repo.all_mentions(), self.repo.all_relationships())

    @staticmethod
    def _known_name_pass(base: list[Mention], chunks: dict) -> list[Mention]:
        seeds = []
        for m in base:
            if m.entity_type == "PROJECT" and m.extractor == "rules":
                seeds.append((m.surface, m.entity_type))
            elif m.entity_type == "ORGANIZATION" and m.surface.split()[-1].rstrip(".,") in ORG_HEAD_WORDS:
                seeds.append((m.surface, m.entity_type))
            elif m.entity_type == "PERSON" and len(m.surface.split()) >= 2:
                seeds.append((m.surface, m.entity_type))
        names = derive_known_names(seeds)
        if not names:
            return []
        existing: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for m in base:
            existing[m.chunk_id].append((m.char_start, m.char_end))
        derived: list[Mention] = []
        for cid, chunk in chunks.items():
            for n, span in enumerate(known_name_mentions(chunk.text, names, existing.get(cid, []))):
                derived.append(
                    Mention(
                        id=f"{cid}:k{n:03d}",
                        chunk_id=cid,
                        document_id=chunk.document_id,
                        surface=span.surface,
                        entity_type=span.entity_type,
                        char_start=span.start,
                        char_end=span.end,
                        extractor="known_name",
                        confidence=span.confidence,
                    )
                )
        return derived
