"""PostgreSQL + pgvector repository.

* Vector search: ``embedding <=> query`` (cosine distance) served by an HNSW index.
* Lexical search: a GIN-indexed ``tsvector`` prefilter (OR over query lexemes)
  followed by BM25 scoring computed in SQL from per-lexeme term frequencies
  (``unnest(tsvector)``) and document frequencies (``lexeme_stats`` view).
  ``lexical_scorer="ts_rank"`` switches to PostgreSQL's native ``ts_rank_cd``,
  which has no IDF component and is kept as an ablation.
"""

from __future__ import annotations

import json
import logging
from importlib import resources
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.models.domain import Chunk, Document, Entity, Mention, Relationship, ResolutionDecision

logger = logging.getLogger(__name__)

# Query lexemes -> OR tsquery. Casting to ::tsquery keeps lexemes verbatim (no re-stemming).
_QUERY_CTE = """
q AS (
    SELECT DISTINCT lexeme FROM unnest(tsvector_to_array(to_tsvector('english', %(query)s))) AS lexeme
),
qt AS (
    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS tsq FROM q
)
"""

_BM25_SQL = f"""
WITH {_QUERY_CTE},
qidf AS MATERIALIZED (
    -- IDF of each query term, computed once (Lucene-style non-negative BM25 idf).
    SELECT q.lexeme,
           ln(1 + (cs.n_chunks - COALESCE(ls.df, 0) + 0.5) / (COALESCE(ls.df, 0) + 0.5)) AS idf
    FROM q CROSS JOIN corpus_stats cs LEFT JOIN lexeme_stats ls ON ls.lexeme = q.lexeme
),
cand AS MATERIALIZED (
    -- GIN-indexed prefilter: chunks containing at least one query lexeme.
    SELECT c.id, c.tsv FROM chunks c CROSS JOIN qt WHERE c.tsv @@ qt.tsq {{id_filter}}
),
matched AS (
    SELECT cand.id, length(cand.tsv)::float8 AS dl, u.lexeme,
           COALESCE(array_length(u.positions, 1), 1)::float8 AS tf
    FROM cand, LATERAL unnest(cand.tsv) AS u
)
SELECT m.id,
       SUM(qi.idf * (m.tf * (%(k1)s + 1)) / (m.tf + %(k1)s * (1 - %(b)s + %(b)s * m.dl / cs.avgdl))) AS score
FROM matched m
JOIN qidf qi ON qi.lexeme = m.lexeme
CROSS JOIN corpus_stats cs
GROUP BY m.id
ORDER BY score DESC, m.id
{{limit}}
"""

_TSRANK_SQL = f"""
WITH {_QUERY_CTE}
SELECT c.id, ts_rank_cd(c.tsv, qt.tsq) AS score
FROM chunks c CROSS JOIN qt
WHERE c.tsv @@ qt.tsq {{id_filter}}
ORDER BY score DESC, c.id
{{limit}}
"""


def _to_numpy(value: Any) -> np.ndarray:
    """pgvector's adapter returns ``Vector`` objects (newer versions) or arrays."""
    if hasattr(value, "to_numpy"):
        return value.to_numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _load_schema(dim: int) -> str:
    return resources.files("app.db").joinpath("schema.sql").read_text().format(dim=dim)


class PostgresRepository:
    backend_name = "postgres"

    def __init__(
        self,
        database_url: str,
        dim: int = 384,
        k1: float = 1.2,
        b: float = 0.75,
        lexical_scorer: str = "bm25",
        ef_search: int = 100,
        pool_size: int = 8,
    ) -> None:
        self.dim = dim
        self.k1, self.b = k1, b
        self.lexical_scorer = lexical_scorer
        self.ef_search = ef_search
        self.backend_name = f"postgres+pgvector ({lexical_scorer})"
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute(_load_schema(dim))
        self._pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=pool_size,
            configure=self._configure,
            kwargs={"autocommit": True},
            open=True,
        )

    def _configure(self, conn: psycopg.Connection) -> None:
        register_vector(conn)
        conn.execute(f"SET hnsw.ef_search = {int(self.ef_search)}")

    def close(self) -> None:
        self._pool.close()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _row_to_document(row: dict[str, Any]) -> Document:
        return Document(
            id=row["id"],
            filename=row["filename"],
            source_type=row["source_type"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            text=row["text"],
            title=row["title"],
            metadata=row["metadata"] or {},
            chunk_count=row["chunk_count"],
        )

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> Chunk:
        emb = row.get("embedding")
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            search_text=row["search_text"],
            token_count=row["token_count"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            metadata=row["metadata"] or {},
            embedding=_to_numpy(emb) if emb is not None else None,
        )

    # ------------------------------------------------------- documents/chunks
    def add_document(self, document: Document, chunks: list[Chunk], mentions: list[Mention]) -> None:
        with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """INSERT INTO documents (id, filename, source_type, content_hash, title, text,
                                          metadata, chunk_count, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    document.id,
                    document.filename,
                    document.source_type,
                    document.content_hash,
                    document.title,
                    document.text,
                    json.dumps(document.metadata),
                    document.chunk_count,
                    document.created_at,
                ),
            )
            cur.executemany(
                """INSERT INTO chunks (id, document_id, chunk_index, text, search_text, token_count,
                                       char_start, char_end, metadata, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (
                        c.id,
                        c.document_id,
                        c.chunk_index,
                        c.text,
                        c.search_text,
                        c.token_count,
                        c.char_start,
                        c.char_end,
                        json.dumps(c.metadata),
                        c.embedding,
                    )
                    for c in chunks
                ],
            )
            if mentions:
                cur.executemany(
                    """INSERT INTO entity_mentions (id, chunk_id, document_id, surface, entity_type,
                                                    char_start, char_end, extractor, confidence)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (
                            m.id,
                            m.chunk_id,
                            m.document_id,
                            m.surface,
                            m.entity_type,
                            m.char_start,
                            m.char_end,
                            m.extractor,
                            m.confidence,
                        )
                        for m in mentions
                    ],
                )

    def find_duplicate(self, filename: str, content_hash: str) -> Document | None:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM documents WHERE filename = %s AND content_hash = %s",
                (filename, content_hash),
            )
            row = cur.fetchone()
        return self._row_to_document(row) if row else None

    def get_document(self, document_id: str) -> Document | None:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s", (document_id,))
            row = cur.fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM documents ORDER BY created_at, filename")
            return [self._row_to_document(r) for r in cur.fetchall()]

    def delete_document(self, document_id: str) -> bool:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            return cur.rowcount > 0

    _CHUNK_COLUMNS = (
        "id, document_id, chunk_index, text, search_text, token_count, char_start, char_end, metadata"
    )

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT {self._CHUNK_COLUMNS} FROM chunks WHERE id = ANY(%s)", (list(chunk_ids),)
            )
            return {r["id"]: self._row_to_chunk(r) for r in cur.fetchall()}

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT {self._CHUNK_COLUMNS} FROM chunks WHERE document_id = %s ORDER BY chunk_index",
                (document_id,),
            )
            return [self._row_to_chunk(r) for r in cur.fetchall()]

    def get_embeddings(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        if not chunk_ids:
            return {}
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, embedding FROM chunks WHERE id = ANY(%s) AND embedding IS NOT NULL",
                (list(chunk_ids),),
            )
            return {cid: _to_numpy(emb) for cid, emb in cur.fetchall()}

    def all_chunk_ids(self) -> list[str]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM chunks ORDER BY document_id, chunk_index")
            return [r[0] for r in cur.fetchall()]

    def counts(self) -> dict[str, int]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks),
                          (SELECT count(*) FROM entity_mentions), (SELECT count(*) FROM entities),
                          (SELECT count(*) FROM entity_relationships)"""
            )
            d, c, m, e, r = cur.fetchone()
        return {"documents": d, "chunks": c, "mentions": m, "entities": e, "relationships": r}

    def finalize_ingest(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("REFRESH MATERIALIZED VIEW lexeme_stats")
            conn.execute("REFRESH MATERIALIZED VIEW corpus_stats")
            conn.execute("ANALYZE chunks")

    def reset(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "TRUNCATE resolution_decisions, entity_relationships, entity_mentions, entities, "
                "chunks, documents"
            )
        self.finalize_ingest()

    # -------------------------------------------------------------- retrieval
    def _lexical(self, query: str, ids: list[str] | None, k: int | None) -> list[tuple[str, float]]:
        template = _BM25_SQL if self.lexical_scorer == "bm25" else _TSRANK_SQL
        sql = template.format(
            id_filter="AND c.id = ANY(%(ids)s)" if ids is not None else "",
            limit="LIMIT %(k)s" if k is not None else "",
        )
        params = {"query": query, "k1": self.k1, "b": self.b, "ids": ids, "k": k}
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return [(cid, float(score)) for cid, score in cur.fetchall()]

    def lexical_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._lexical(query, None, k)

    def lexical_scores(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        if not chunk_ids:
            return {}
        found = dict(self._lexical(query, list(chunk_ids), None))
        return {cid: found.get(cid, 0.0) for cid in chunk_ids}

    def vector_search(self, embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        vec = embedding.astype(np.float32)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, 1 - (embedding <=> %(v)s) AS sim FROM chunks
                   WHERE embedding IS NOT NULL ORDER BY embedding <=> %(v)s LIMIT %(k)s""",
                {"v": vec, "k": k},
            )
            return [(cid, float(sim)) for cid, sim in cur.fetchall()]

    def vector_scores(self, embedding: np.ndarray, chunk_ids: list[str]) -> dict[str, float]:
        if not chunk_ids:
            return {}
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, 1 - (embedding <=> %s) FROM chunks WHERE id = ANY(%s)",
                (embedding.astype(np.float32), list(chunk_ids)),
            )
            return {cid: float(sim) for cid, sim in cur.fetchall()}

    def term_document_frequencies(self, terms: list[str]) -> tuple[dict[str, int], int]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT lexeme, df FROM lexeme_stats WHERE lexeme = ANY(%s)", (list(terms),))
            found = {lex: int(df) for lex, df in cur.fetchall()}
            cur.execute("SELECT n_chunks FROM corpus_stats")
            row = cur.fetchone()
        return {t: found.get(t, 0) for t in terms}, int(row[0]) if row else 0

    def unknown_terms(self, text: str) -> list[str]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT q.term FROM unnest(tsvector_to_array(to_tsvector('english', %s))) AS q(term)
                   WHERE NOT EXISTS (SELECT 1 FROM lexeme_stats ls WHERE ls.lexeme = q.term)""",
                (text,),
            )
            return [r[0] for r in cur.fetchall()]

    # ----------------------------------------------------------- entity layer
    def all_mentions(self) -> list[Mention]:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM entity_mentions ORDER BY document_id, chunk_id, char_start")
            return [Mention(**row) for row in cur.fetchall()]

    def replace_entity_graph(
        self,
        entities: list[Entity],
        assignments: dict[str, str],
        relationships: list[Relationship],
        decisions: list[ResolutionDecision],
        derived_mentions: list[Mention],
    ) -> None:
        with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute("DELETE FROM entity_mentions WHERE extractor IN ('known_name', 'header')")
            if derived_mentions:
                cur.executemany(
                    """INSERT INTO entity_mentions (id, chunk_id, document_id, surface, entity_type,
                                                    char_start, char_end, extractor, confidence)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (m.id, m.chunk_id, m.document_id, m.surface, m.entity_type,
                         m.char_start, m.char_end, m.extractor, m.confidence)
                        for m in derived_mentions
                    ],
                )
            cur.execute("UPDATE entity_mentions SET entity_id = NULL")
            cur.execute("DELETE FROM entity_relationships")
            cur.execute("DELETE FROM resolution_decisions")
            cur.execute("DELETE FROM entities")
            cur.executemany(
                """INSERT INTO entities (id, canonical_name, entity_type, aliases, mention_count,
                                         document_count, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                [
                    (
                        e.id,
                        e.canonical_name,
                        e.entity_type,
                        e.aliases,
                        e.mention_count,
                        e.document_count,
                        json.dumps(e.metadata),
                    )
                    for e in entities
                ],
            )
            if assignments:
                cur.execute(
                    """UPDATE entity_mentions m SET entity_id = a.entity_id
                       FROM unnest(%s::text[], %s::text[]) AS a(mention_id, entity_id)
                       WHERE m.id = a.mention_id""",
                    (list(assignments.keys()), list(assignments.values())),
                )
            cur.executemany(
                """INSERT INTO entity_relationships (id, source_entity_id, target_entity_id,
                       relationship_type, confidence, supporting_chunk_id, evidence, method)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (
                        r.id,
                        r.source_entity_id,
                        r.target_entity_id,
                        r.relationship_type,
                        r.confidence,
                        r.supporting_chunk_id,
                        r.evidence,
                        r.method,
                    )
                    for r in relationships
                ],
            )
            cur.executemany(
                """INSERT INTO resolution_decisions (left_surface, right_surface, entity_type, score,
                                                     decision, features, reason)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                [
                    (d.left, d.right, d.entity_type, d.score, d.decision, json.dumps(d.features), d.reason)
                    for d in decisions
                ],
            )

    @staticmethod
    def _row_to_entity(row: dict[str, Any]) -> Entity:
        return Entity(
            id=row["id"],
            canonical_name=row["canonical_name"],
            entity_type=row["entity_type"],
            aliases=list(row["aliases"]),
            mention_count=row["mention_count"],
            document_count=row["document_count"],
            metadata=row["metadata"] or {},
        )

    def list_entities(self, entity_type: str | None = None) -> list[Entity]:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            if entity_type:
                cur.execute(
                    "SELECT * FROM entities WHERE entity_type = %s ORDER BY mention_count DESC, canonical_name",
                    (entity_type,),
                )
            else:
                cur.execute("SELECT * FROM entities ORDER BY mention_count DESC, canonical_name")
            return [self._row_to_entity(r) for r in cur.fetchall()]

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
            row = cur.fetchone()
        return self._row_to_entity(row) if row else None

    def all_relationships(self) -> list[Relationship]:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM entity_relationships")
            return [Relationship(**row) for row in cur.fetchall()]

    def resolution_decisions(self) -> list[ResolutionDecision]:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT left_surface, right_surface, entity_type, score, decision, features, reason "
                "FROM resolution_decisions ORDER BY id"
            )
            return [
                ResolutionDecision(
                    left=r["left_surface"],
                    right=r["right_surface"],
                    entity_type=r["entity_type"],
                    score=r["score"],
                    decision=r["decision"],
                    features=r["features"],
                    reason=r["reason"],
                )
                for r in cur.fetchall()
            ]
