import json

import pytest
from pydantic import ValidationError

from app.core.container import Container
from app.evaluation.dataset import (
    DatasetIntegrityError,
    EvalDataset,
    EvalQuery,
    load_dataset,
    resolve_relevance,
)
from app.evaluation.metrics import (
    abstention_metrics,
    hit_at_k,
    latency_stats,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.report import render_markdown
from app.evaluation.runner import run_evaluation
from tests.conftest import REPO_ROOT


def test_rank_metrics():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"b", "d"}, 1) == 0.0
    assert recall_at_k(ranked, {"b", "d"}, 2) == 0.5
    assert recall_at_k(ranked, {"b", "d"}, 4) == 1.0
    assert hit_at_k(ranked, {"c"}, 2) == 0.0 and hit_at_k(ranked, {"c"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / 1.5849625, rel=1e-6)
    assert ndcg_at_k(["a"], {"a"}, 10) == 1.0


def test_abstention_metrics():
    m = abstention_metrics(abstained=[True, True, False, False], answerable=[False, True, False, True])
    assert m["abstention_precision"] == 0.5 and m["abstention_recall"] == 0.5
    assert m["false_abstention_rate"] == 0.5
    empty = abstention_metrics([False], [True])
    assert empty["abstention_precision"] is None


def test_latency_stats():
    s = latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["p50_ms"] == 3.0 and s["p95_ms"] > 50


def test_dataset_validation():
    with pytest.raises(ValidationError):
        EvalQuery(id="x", query="q", category="c", answerable=True)
    with pytest.raises(ValidationError):
        EvalQuery(id="x", query="q", category="c", answerable=False, evidence=["e"])
    with pytest.raises(ValidationError):
        EvalDataset(name="d", queries=[EvalQuery(id="x", query="q", category="u", answerable=False)] * 2)


def test_relevance_resolution_and_integrity_errors(indexed: Container):
    ds = EvalDataset(
        name="d",
        queries=[EvalQuery(id="q1", query="q", category="c", relevant_sources=["incident.txt"], evidence=["error   QL-5031"])],
    )
    rel = resolve_relevance(ds, indexed.repo)["q1"]
    assert len(rel.chunk_ids) == 1 and len(rel.document_ids) == 1
    bad = EvalDataset(
        name="d",
        queries=[EvalQuery(id="q1", query="q", category="c", relevant_sources=["incident.txt"], evidence=["not in the corpus"])],
    )
    with pytest.raises(DatasetIntegrityError):
        resolve_relevance(bad, indexed.repo)


def test_run_evaluation_end_to_end(indexed: Container):
    ds = EvalDataset(
        name="tiny",
        queries=[
            EvalQuery(id="q1", query="QL-5031", category="lexical", relevant_sources=["incident.txt"], evidence=["error QL-5031"]),
            EvalQuery(id="q2", query="Who sponsors Project Ember?", category="entity", relevant_sources=["ember.md"], evidence=["Samuel Okafor sponsors"]),
            EvalQuery(id="q3", query="What is the capital of Australia?", category="unanswerable", answerable=False),
        ],
    )
    report = run_evaluation(indexed, ds, ["lexical_only", "hybrid_entity"], warmup=1)
    s = report["summary"]["lexical_only"]
    assert s["hit@1"] == 1.0 or s["hit@5"] == 1.0
    assert set(report["by_category"]["hybrid_entity"]) == {"lexical", "entity", "unanswerable"}
    assert report["per_query"][0]["relevant_chunk_ids"]
    assert report["meta"]["dataset"]["unanswerable"] == 1
    md = render_markdown(report)
    assert "Hybrid + Entity" in md and "Abstention" in md


def test_bundled_dataset_is_valid():
    ds = load_dataset(REPO_ROOT / "eval" / "datasets" / "queries.json")
    assert len(ds.queries) >= 100
    gold = json.loads((REPO_ROOT / "eval" / "datasets" / "entity_resolution_gold.json").read_text())
    assert gold["entities"]
