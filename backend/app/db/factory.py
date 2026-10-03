from __future__ import annotations

from app.core.config import Settings
from app.db.repository import Repository


def build_repository(settings: Settings) -> Repository:
    r = settings.retrieval
    if settings.storage_backend == "postgres":
        from app.db.postgres import PostgresRepository

        return PostgresRepository(
            settings.database_url,
            dim=settings.embedding_dim,
            k1=r.bm25_k1,
            b=r.bm25_b,
            lexical_scorer=r.lexical_scorer,
            ef_search=r.hnsw_ef_search,
        )
    from app.db.memory import InMemoryRepository

    return InMemoryRepository(
        dim=settings.embedding_dim, k1=r.bm25_k1, b=r.bm25_b, snapshot_path=settings.memory_snapshot_path
    )
