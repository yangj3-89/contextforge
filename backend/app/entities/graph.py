"""In-process view of the entity layer used at query time.

The authoritative data lives in the repository (``entities``,
``entity_mentions``, ``entity_relationships`` tables). This index is rebuilt
after every ingestion / resolution run and answers the hot-path questions:
which entities does a chunk mention, which chunks mention an entity, and which
entities are one relationship away. At this corpus scale an in-memory
adjacency map is simpler and faster than recursive SQL; for a large graph the
same lookups map to indexed joins on the relationship table.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.entities.normalization import basic_normalize, compact_form, name_tokens
from app.models.domain import Entity, Mention, Relationship

MAX_EDGE_WEIGHT = 0.95


@dataclass
class Edge:
    neighbor_id: str
    weight: float
    relation_types: set[str] = field(default_factory=set)
    supporting_chunks: set[str] = field(default_factory=set)


class EntityGraph:
    def __init__(self, entities: list[Entity], mentions: list[Mention], relationships: list[Relationship]) -> None:
        self.entities: dict[str, Entity] = {e.id: e for e in entities}
        self.chunk_entities: dict[str, set[str]] = defaultdict(set)
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)
        for m in mentions:
            if m.entity_id and m.entity_id in self.entities:
                self.chunk_entities[m.chunk_id].add(m.entity_id)
                self.entity_chunks[m.entity_id].add(m.chunk_id)

        # Aggregate per-chunk relationship rows into weighted undirected edges
        # (noisy-OR over independent supporting evidence).
        miss: dict[tuple[str, str], float] = defaultdict(lambda: 1.0)
        types: dict[tuple[str, str], set[str]] = defaultdict(set)
        chunks: dict[tuple[str, str], set[str]] = defaultdict(set)
        for r in relationships:
            if r.source_entity_id not in self.entities or r.target_entity_id not in self.entities:
                continue
            for key in ((r.source_entity_id, r.target_entity_id), (r.target_entity_id, r.source_entity_id)):
                miss[key] *= 1.0 - r.confidence
                types[key].add(r.relationship_type)
                chunks[key].add(r.supporting_chunk_id)
        self.adjacency: dict[str, dict[str, Edge]] = defaultdict(dict)
        for (src, dst), m in miss.items():
            self.adjacency[src][dst] = Edge(dst, min(MAX_EDGE_WEIGHT, 1.0 - m), types[(src, dst)], chunks[(src, dst)])
        self.relationship_count = len(relationships)

        # Alias indexes for query-time linking.
        self.exact_index: dict[str, set[str]] = defaultdict(set)
        self.compact_index: dict[str, set[str]] = defaultdict(set)
        self.token_index: dict[str, set[str]] = defaultdict(set)
        for entity in entities:
            for alias in [entity.canonical_name, *entity.aliases]:
                key = basic_normalize(alias)
                if not key:
                    continue
                self.exact_index[key].add(entity.id)
                normalized = " ".join(name_tokens(alias, entity.entity_type))
                if normalized:
                    self.exact_index[normalized].add(entity.id)
                compact = compact_form(alias, entity.entity_type)
                if len(compact) >= 4:
                    self.compact_index[compact].add(entity.id)
                if entity.entity_type in ("PERSON", "ORGANIZATION", "PROJECT"):
                    for token in name_tokens(alias, entity.entity_type):
                        if len(token) >= 3:
                            self.token_index[token].add(entity.id)

    def neighbors(self, entity_id: str) -> dict[str, Edge]:
        return self.adjacency.get(entity_id, {})

    def is_empty(self) -> bool:
        return not self.entities
