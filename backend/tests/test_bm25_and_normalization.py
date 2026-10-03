import pytest

from app.core.text import lexical_terms
from app.retrieval.bm25 import BM25Index
from app.retrieval.normalization import max_scale, min_max, normalize, ranks, reciprocal_rank


@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index()
    idx.build(
        [
            ("a", lexical_terms("postgres vector index hnsw")),
            ("b", lexical_terms("postgres full text search")),
            ("c", lexical_terms("kafka consumer lag incident incident")),
        ]
    )
    return idx


def test_bm25_ranks_rare_terms_higher(index: BM25Index):
    hits = index.search(lexical_terms("hnsw postgres"), k=3)
    assert hits[0][0] == "a"
    assert {h[0] for h in hits} == {"a", "b"}


def test_bm25_idf_is_non_negative(index: BM25Index):
    assert index.idf("postgr") > 0  # appears in 2 of 3 docs
    assert index.idf("hnsw") > index.idf("postgr")
    assert index.idf("unseen") > index.idf("hnsw")


def test_bm25_scores_for_subset_includes_zero_scores(index: BM25Index):
    scores = index.scores_for(lexical_terms("kafka"), ["a", "c", "missing"])
    assert scores["a"] == 0.0 and scores["c"] > 0
    assert "missing" not in scores


def test_bm25_empty_query(index: BM25Index):
    assert index.search([], k=5) == []


def test_min_max_edge_cases():
    assert min_max({}) == {}
    assert min_max({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}
    assert min_max({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    assert min_max({"a": 1.0, "b": 3.0}) == {"a": 0.0, "b": 1.0}


def test_max_scale_and_none():
    assert max_scale({"a": 2.0, "b": 1.0}) == {"a": 1.0, "b": 0.5}
    assert max_scale({"a": 0.0}) == {"a": 0.0}
    assert normalize({"a": 1.5, "b": -0.2}, "none") == {"a": 1.0, "b": 0.0}
    with pytest.raises(ValueError):
        normalize({"a": 1.0}, "zscore")


def test_rank_helpers_are_deterministic():
    assert ranks({"b": 1.0, "a": 1.0, "c": 2.0}) == {"c": 1, "a": 2, "b": 3}
    assert reciprocal_rank(1, 60) == pytest.approx(1 / 61)
