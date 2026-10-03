"""In-memory repository: numpy cosine similarity + Python BM25.

Indexes are rebuilt lazily after writes. Optionally the full state is pickled
to ``snapshot_path`` so a dev server keeps its index across restarts.
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path

import numpy as np

from app.core.text import lexical_terms, query_terms
from app.models.domain import Chunk, Document, Entity, Mention, Relationship, ResolutionDecision
from app.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


class InMemoryRepository:
    backend_name = "memory"

    def __init__(
        self,
        dim: int = 384,
        k1: float = 1.2,
        b: float = 0.75,
        snapshot_path: Path | None = None,
    ) -> None:
        self.dim = dim
        self._k1, self._b = k1, b
        self._snapshot_path = snapshot_path
        self._lock = threading.RLock()
        self._clear()
        if snapshot_path and snapshot_path.exists():
            self._load_snapshot(snapshot_path)

    # ------------------------------------------------------------------ state
    def _clear(self) -> None:
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        self._doc_chunks: dict[str, list[str]] = {}
        self._mentions: dict[str, Mention] = {}
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []
        self._decisions: list[ResolutionDecision] = []
        self._dirty = True
        self._bm25 = BM25Index(self._k1, self._b)
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._matrix_ids: list[str] = []
        self._row_of: dict[str, int] = {}

    def _ensure_indexes(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            ids = [cid for cid, c in self._chunks.items() if c.embedding is not None]
            self._matrix_ids = ids
            self._row_of = {cid: i for i, cid in enumerate(ids)}
            self._matrix = (
                np.vstack([self._chunks[cid].embedding for cid in ids]).astype(np.float32)
                if ids
                else np.zeros((0, self.dim), dtype=np.float32)
            )
            self._bm25.build([(cid, lexical_terms(c.search_text)) for cid, c in self._chunks.items()])
            self._dirty = False

    def _snapshot(self) -> None:
        if not self._snapshot_path:
            return
        state = {
            "documents": self._documents,
            "chunks": self._chunks,
            "doc_chunks": self._doc_chunks,
            "mentions": self._mentions,
            "entities": self._entities,
            "relationships": self._relationships,
            "decisions": self._decisions,
        }
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._snapshot_path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(state, fh)
        tmp.replace(self._snapshot_path)

    def _load_snapshot(self, path: Path) -> None:
        # Only load snapshots this process wrote itself (pickle is not safe for untrusted input).
        with path.open("rb") as fh:
            state = pickle.load(fh)  # noqa: S301
        self._documents = state["documents"]
        self._chunks = state["chunks"]
        self._doc_chunks = state["doc_chunks"]
        self._mentions = state["mentions"]
        self._entities = state["entities"]
        self._relationships = state["relationships"]
        self._decisions = state["decisions"]
        self._dirty = True
        logger.info("Loaded in-memory snapshot with %d documents", len(self._documents))

    # ------------------------------------------------------- documents/chunks
    def add_document(self, document: Document, chunks: list[Chunk], mentions: list[Mention]) -> None:
        with self._lock:
            self._documents[document.id] = document
            self._doc_chunks[document.id] = [c.id for c in chunks]
            for chunk in chunks:
                self._chunks[chunk.id] = chunk
            for mention in mentions:
                self._mentions[mention.id] = mention
            self._dirty = True

    def find_duplicate(self, filename: str, content_hash: str) -> Document | None:
        for doc in self._documents.values():
            if doc.filename == filename and doc.content_hash == content_hash:
                return doc
        return None

    def get_document(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def list_documents(self) -> list[Document]:
        return sorted(self._documents.values(), key=lambda d: (d.created_at, d.filename))

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self._documents:
                return False
            chunk_ids = set(self._doc_chunks.pop(document_id, []))
            for cid in chunk_ids:
                self._chunks.pop(cid, None)
            self._mentions = {k: m for k, m in self._mentions.items() if m.document_id != document_id}
            self._relationships = [
                r for r in self._relationships if r.supporting_chunk_id not in chunk_ids
            ]
            del self._documents[document_id]
            self._dirty = True
            return True

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        return {cid: self._chunks[cid] for cid in chunk_ids if cid in self._chunks}

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        return [self._chunks[cid] for cid in self._doc_chunks.get(document_id, [])]

    def get_embeddings(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        return {
            cid: self._chunks[cid].embedding
            for cid in chunk_ids
            if cid in self._chunks and self._chunks[cid].embedding is not None
        }

    def all_chunk_ids(self) -> list[str]:
        return list(self._chunks)

    def counts(self) -> dict[str, int]:
        return {
            "documents": len(self._documents),
            "chunks": len(self._chunks),
            "mentions": len(self._mentions),
            "entities": len(self._entities),
            "relationships": len(self._relationships),
        }

    def finalize_ingest(self) -> None:
        with self._lock:
            self._ensure_indexes()
            self._snapshot()

    def reset(self) -> None:
        with self._lock:
            self._clear()
            if self._snapshot_path and self._snapshot_path.exists():
                self._snapshot_path.unlink()

    # -------------------------------------------------------------- retrieval
    def lexical_search(self, query: str, k: int) -> list[tuple[str, float]]:
        self._ensure_indexes()
        return self._bm25.search(query_terms(query), k)

    def lexical_scores(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        self._ensure_indexes()
        return self._bm25.scores_for(query_terms(query), chunk_ids)

    def vector_search(self, embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        self._ensure_indexes()
        if not self._matrix_ids:
            return []
        sims = self._matrix @ embedding.astype(np.float32)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(self._matrix_ids[i], float(sims[i])) for i in top]

    def vector_scores(self, embedding: np.ndarray, chunk_ids: list[str]) -> dict[str, float]:
        self._ensure_indexes()
        rows = [(cid, self._row_of[cid]) for cid in chunk_ids if cid in self._row_of]
        if not rows:
            return {}
        sims = self._matrix[[r for _, r in rows]] @ embedding.astype(np.float32)
        return {cid: float(s) for (cid, _), s in zip(rows, sims, strict=True)}

    def term_document_frequencies(self, terms: list[str]) -> tuple[dict[str, int], int]:
        self._ensure_indexes()
        return {t: self._bm25.df.get(t, 0) for t in terms}, self._bm25.n_docs

    def unknown_terms(self, text: str) -> list[str]:
        self._ensure_indexes()
        return [t for t in dict.fromkeys(lexical_terms(text)) if self._bm25.df.get(t, 0) == 0]

    # ----------------------------------------------------------- entity layer
    def all_mentions(self) -> list[Mention]:
        return list(self._mentions.values())

    def replace_entity_graph(
        self,
        entities: list[Entity],
        assignments: dict[str, str],
        relationships: list[Relationship],
        decisions: list[ResolutionDecision],
        derived_mentions: list[Mention],
    ) -> None:
        with self._lock:
            self._mentions = {
                k: m for k, m in self._mentions.items() if m.extractor not in ("known_name", "header")
            }
            for mention in derived_mentions:
                self._mentions[mention.id] = mention
            self._entities = {e.id: e for e in entities}
            for mention in self._mentions.values():
                mention.entity_id = assignments.get(mention.id)
            self._relationships = list(relationships)
            self._decisions = list(decisions)
            self._snapshot()

    def list_entities(self, entity_type: str | None = None) -> list[Entity]:
        items = [e for e in self._entities.values() if entity_type is None or e.entity_type == entity_type]
        return sorted(items, key=lambda e: (-e.mention_count, e.canonical_name))

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def all_relationships(self) -> list[Relationship]:
        return list(self._relationships)

    def resolution_decisions(self) -> list[ResolutionDecision]:
        return list(self._decisions)
