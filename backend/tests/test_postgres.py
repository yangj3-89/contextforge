"""PostgreSQL + pgvector backend tests.

Run with a disposable database, e.g.:
    CF_TEST_DATABASE_URL=postgresql://contextforge:contextforge@localhost:5432/contextforge_test pytest -m postgres
"""

import pytest

from app.core.container import Container
from app.embeddings.hashing import HashingEmbedder
from app.entities.extraction import build_extractor
from tests.conftest import CORPUS, make_settings, postgres_url

pytestmark = pytest.mark.postgres


@pytest.fixture
def pg_container():
    url = postgres_url()
    if not url:
        pytest.skip("CF_TEST_DATABASE_URL not set")
    try:
        import psycopg

        psycopg.connect(url, connect_timeout=3).close()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    c = Container(make_settings(storage_backend="postgres", database_url=url), embedder=HashingEmbedder(), extractor=build_extractor("rules"))
    c.reset()
    c.ingest_files(list(CORPUS.items()))
    yield c
    c.reset()
    c.repo.close()


def test_documents_and_chunks_roundtrip(pg_container: Container):
    counts = pg_container.repo.counts()
    assert counts["documents"] == len(CORPUS) and counts["chunks"] > 0 and counts["entities"] > 0
    doc = next(d for d in pg_container.repo.list_documents() if d.filename == "halcyon.md")
    chunks = pg_container.repo.get_document_chunks(doc.id)
    assert all(doc.text[c.char_start : c.char_end] == c.text for c in chunks)
    assert pg_container.repo.find_duplicate("halcyon.md", doc.content_hash).id == doc.id


def test_sql_bm25_ranks_identifier_first(pg_container: Container):
    hits = pg_container.repo.lexical_search("error QL-5031", 5)
    chunk = pg_container.repo.get_chunks([hits[0][0]])[hits[0][0]]
    assert "QL-5031" in chunk.text
    scores = pg_container.repo.lexical_scores("error QL-5031", [h[0] for h in hits] + ["missing"])
    assert scores["missing"] == 0.0 and scores[hits[0][0]] == pytest.approx(hits[0][1])


def test_pgvector_cosine_search(pg_container: Container):
    emb = pg_container.embedder.embed(["Kafka consumer lag rebalance"])[0]
    hits = pg_container.repo.vector_search(emb, 3)
    assert len(hits) == 3 and hits[0][1] >= hits[1][1]
    again = pg_container.repo.vector_scores(emb, [hits[0][0]])
    assert again[hits[0][0]] == pytest.approx(hits[0][1], abs=1e-5)


def test_unknown_terms_regression(pg_container: Container):
    # Regression: a column-scoping bug once made every term look known.
    unknown = set(pg_container.repo.unknown_terms("Bob Smith Halcyon"))
    assert {"bob", "smith"} <= unknown and "halcyon" not in unknown


def test_entity_graph_persisted_and_delete_cascades(pg_container: Container):
    entities = {e.canonical_name: e for e in pg_container.repo.list_entities()}
    assert "Alice Chen" in entities and pg_container.repo.all_relationships()
    doc = next(d for d in pg_container.repo.list_documents() if d.filename == "incident.txt")
    assert pg_container.delete_document(doc.id)
    assert pg_container.repo.get_document(doc.id) is None
    assert all(m.document_id != doc.id for m in pg_container.repo.all_mentions())


def test_search_modes_on_postgres(pg_container: Container):
    for mode in ("lexical_only", "vector_only", "hybrid", "hybrid_entity"):
        out = pg_container.engine.search("Which database stores Halcyon embeddings?", mode)
        assert out.results, mode
