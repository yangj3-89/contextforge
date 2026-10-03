import pytest

from app.core.config import SignalWeights
from app.core.container import Container


def test_lexical_only_finds_exact_identifier(indexed: Container):
    out = indexed.engine.search("QL-5031", "lexical_only", top_k=3)
    assert out.results[0].filename == "incident.txt"
    top = out.results[0]
    assert top.lexical_score == 1.0 and top.vector_score is None and top.entity_score is None
    assert out.weights.lexical == 1.0


def test_vector_only_returns_cosine_similarity(indexed: Container):
    out = indexed.engine.search("pgvector HNSW embeddings storage", "vector_only", top_k=3)
    assert out.results
    top = out.results[0]
    assert top.vector_similarity is not None and -1.0 <= top.vector_similarity <= 1.0
    assert top.lexical_score is None
    assert "embed_ms" in out.timings_ms and "vector_ms" in out.timings_ms


def test_hybrid_scores_every_candidate_on_every_signal(indexed: Container):
    out = indexed.engine.search("Which database stores embeddings?", "hybrid", top_k=10)
    for r in out.results:
        assert r.vector_score is not None and r.lexical_score is not None
        expected = 0.5 * r.vector_score + 0.5 * r.lexical_score
        assert r.final_score == pytest.approx(expected, abs=1e-3)


def test_results_are_sorted_and_carry_provenance(indexed: Container):
    out = indexed.engine.search("Kafka consumer lag", "hybrid_entity", top_k=5)
    scores = [r.final_score for r in out.results]
    assert scores == sorted(scores, reverse=True)
    doc_text = {d.id: d.text for d in indexed.repo.list_documents()}
    for r in out.results:
        assert doc_text[r.document_id][r.char_start : r.char_end] == r.text
        assert r.filename and r.chunk_id.startswith(r.document_id)


def test_hybrid_entity_links_query_entities_and_scores_them(indexed: Container):
    out = indexed.engine.search("What does Alice Chen lead?", "hybrid_entity", top_k=5)
    assert "Alice Chen" in [e.canonical_name for e in out.query_entities]
    assert any(r.entity_score and r.entity_score > 0 for r in out.results)
    assert any("Alice Chen" in m for r in out.results for m in r.matched_entities)


def test_weight_override_changes_mix(indexed: Container):
    out = indexed.engine.search("Kafka", "hybrid_entity", weights=SignalWeights(vector=0, lexical=1, entity=0))
    for r in out.results:
        assert r.final_score == pytest.approx(r.lexical_score, abs=1e-6)


def test_mode_masks_irrelevant_weights(indexed: Container):
    out = indexed.engine.search("Kafka", "hybrid", weights=SignalWeights(vector=1, lexical=1, entity=5))
    assert out.weights.entity == 0.0


def test_rrf_fusion_is_supported(settings):
    from app.embeddings.hashing import HashingEmbedder
    from app.entities.extraction import build_extractor
    from tests.conftest import CORPUS, make_settings

    s = make_settings()
    s.retrieval.fusion = "rrf"
    c = Container(s, embedder=HashingEmbedder(), extractor=build_extractor("rules"))
    c.ingest_files(list(CORPUS.items()))
    out = c.engine.search("Kafka consumer lag", "hybrid")
    assert out.results and out.results[0].final_score < 0.05  # RRF scores are ~1/(k+rank)


@pytest.mark.parametrize("query,mode", [("", "hybrid"), ("ok", "nonsense_mode")])
def test_invalid_inputs_raise(indexed: Container, query, mode):
    with pytest.raises(ValueError):
        indexed.engine.search(query, mode)


def test_search_on_empty_index_abstains(container: Container):
    out = container.engine.search("anything at all", "hybrid")
    assert out.abstained and out.results == []
