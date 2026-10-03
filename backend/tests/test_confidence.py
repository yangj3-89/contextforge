from app.core.config import ConfidenceSettings
from app.retrieval.confidence import ConfidenceEstimator, ConfidenceInputs


def _inputs(**kw) -> ConfidenceInputs:
    base = dict(
        mode="hybrid",
        final_scores=[0.9, 0.5, 0.4],
        top_vector_raw=[0.62, 0.40, 0.35],
        top_coverage=[0.9, 0.5, None],
        top_documents=["d1", "d2", "d3"],
        lexical_top=["c1", "c2", "c3", "c4", "c5"],
        vector_top=["c1", "c2", "c9", "c8", "c7"],
        top_chunk_ids=["c1", "c2", "c3"],
    )
    base.update(kw)
    return ConfidenceInputs(**base)


def test_strong_evidence_succeeds_with_all_signals():
    report = ConfidenceEstimator(ConfidenceSettings()).evaluate(_inputs())
    assert report.status == "success" and not report.reasons
    assert {"relevance", "margin", "agreement", "support", "vector_similarity", "lexical_coverage"} <= set(report.signals)
    assert 0.0 <= report.confidence <= 1.0


def test_weak_evidence_gate_abstains_with_reason():
    report = ConfidenceEstimator(ConfidenceSettings()).evaluate(
        _inputs(top_vector_raw=[0.12, 0.1, 0.05], top_coverage=[0.1, 0.0, None])
    )
    assert report.status == "insufficient_evidence"
    assert "weak_evidence" in report.gates_triggered
    assert any("cosine similarity 0.12" in r for r in report.reasons)


def test_one_strong_signal_avoids_the_gate():
    # Exact identifier lookups have low cosine but full lexical coverage.
    report = ConfidenceEstimator(ConfidenceSettings()).evaluate(_inputs(top_vector_raw=[0.15, 0.1, 0.1], top_coverage=[1.0, 0.2, None]))
    assert "weak_evidence" not in report.gates_triggered


def test_unknown_entity_gate_only_in_entity_mode():
    est = ConfidenceEstimator(ConfidenceSettings())
    entity_mode = est.evaluate(_inputs(mode="hybrid_entity", unknown_entities=["Project Nimbus"]))
    assert entity_mode.status == "insufficient_evidence" and "unknown_entity" in entity_mode.gates_triggered
    assert "Project Nimbus" in entity_mode.reasons[0]
    hybrid = est.evaluate(_inputs(mode="hybrid", unknown_entities=["Project Nimbus"]))
    assert hybrid.status == "success"


def test_threshold_controls_abstention():
    est = ConfidenceEstimator(ConfidenceSettings())
    base = est.evaluate(_inputs())
    assert est.evaluate(_inputs(), threshold=base.confidence + 0.01).status == "insufficient_evidence"
    assert est.evaluate(_inputs(), threshold=base.confidence - 0.01).status == "success"


def test_entity_consistency_signal():
    est = ConfidenceEstimator(ConfidenceSettings())
    hit = est.evaluate(_inputs(mode="hybrid_entity", linked_entities={"e1": 1.0}, entities_in_top={"e1"}))
    miss = est.evaluate(_inputs(mode="hybrid_entity", linked_entities={"e1": 1.0}, entities_in_top=set()))
    assert hit.signals["entity_consistency"] == 1.0 and miss.signals["entity_consistency"] == 0.0
    assert hit.confidence > miss.confidence


def test_single_signal_modes_drop_agreement():
    report = ConfidenceEstimator(ConfidenceSettings()).evaluate(
        _inputs(mode="vector_only", top_coverage=[None, None, None], lexical_top=[], vector_top=[])
    )
    assert "agreement" not in report.signals and "lexical_coverage" not in report.signals


def test_no_results_abstains():
    report = ConfidenceEstimator(ConfidenceSettings()).evaluate(_inputs(final_scores=[], top_vector_raw=[], top_coverage=[], top_documents=[]))
    assert report.status == "insufficient_evidence" and report.gates_triggered == ["no_results"]
