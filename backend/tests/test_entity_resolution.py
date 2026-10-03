import numpy as np

from app.core.config import ResolutionSettings
from app.entities.resolution import EntityResolver, person_compatible
from app.models.domain import Mention


def _mentions(rows: list[tuple[str, str, str]]) -> list[Mention]:
    return [
        Mention(f"m{i}", f"chunk_{doc}", doc, surface, etype, 0, len(surface), "rules", 0.9)
        for i, (surface, etype, doc) in enumerate(rows)
    ]


def _clusters(rows, embeddings=None) -> dict[str, set[str]]:
    mentions = _mentions(rows)
    result = EntityResolver(ResolutionSettings()).resolve(mentions, embeddings)
    out: dict[str, set[str]] = {}
    for m in mentions:
        out.setdefault(result.assignments[m.id], set()).add(f"{m.surface}@{m.document_id}")
    return {frozenset(v): v for v in out.values()}, result


def _same(result_rows, a: str, b: str) -> bool:
    clusters, _ = _clusters(result_rows)
    return any(a in c and b in c for c in clusters)


def test_company_name_variants_merge():
    rows = [("OpenAI", "ORGANIZATION", "d1"), ("Open AI", "ORGANIZATION", "d2"), ("OpenAI Inc.", "ORGANIZATION", "d3")]
    clusters, result = _clusters(rows)
    assert len(clusters) == 1
    assert result.entities[0].canonical_name in {"OpenAI", "Open AI"}  # suffix-free form preferred


def test_person_initial_and_nickname_variants_merge():
    rows = [("Samuel Okafor", "PERSON", "d1"), ("Sam Okafor", "PERSON", "d2"), ("S. Okafor", "PERSON", "d3")]
    clusters, _ = _clusters(rows)
    assert len(clusters) == 1


def test_ambiguous_first_name_stays_separate():
    rows = [
        ("Alice Chen", "PERSON", "d1"),
        ("Alice Moreno", "PERSON", "d2"),
        ("Alice", "PERSON", "d3"),  # no full name in d3: ambiguous
    ]
    clusters, result = _clusters(rows)
    assert len(clusters) == 3
    assert {e.canonical_name for e in result.entities} == {"Alice Chen", "Alice Moreno", "Alice"}


def test_within_document_coreference_resolves_first_name():
    rows = [("Alice Chen", "PERSON", "d1"), ("Alice", "PERSON", "d1"), ("Alice Moreno", "PERSON", "d2"), ("Alice", "PERSON", "d2")]
    assert _same(rows, "Alice@d1", "Alice Chen@d1")
    assert _same(rows, "Alice@d2", "Alice Moreno@d2")
    assert not _same(rows, "Alice@d1", "Alice@d2")


def test_different_surnames_never_merge():
    rows = [("Alice Chen", "PERSON", "d1"), ("Alice Chan", "PERSON", "d2")]
    clusters, result = _clusters(rows)
    assert len(clusters) == 2


def test_org_with_different_distinctive_token_is_blocked():
    rows = [("Quillon Labs", "ORGANIZATION", "d1"), ("Quillon Health", "ORGANIZATION", "d2"), ("Quillon", "ORGANIZATION", "d3")]
    clusters, _ = _clusters(rows)
    # "Quillon" is ambiguous between two organizations, so nothing merges.
    assert len(clusters) == 3


def test_unique_short_org_form_merges():
    rows = [("Brightwater Health", "ORGANIZATION", "d1"), ("Brightwater", "ORGANIZATION", "d1"), ("Brightwater", "ORGANIZATION", "d2")]
    clusters, _ = _clusters(rows)
    assert len(clusters) == 1


def test_known_technology_aliases_merge_without_string_similarity():
    rows = [("k8s", "TECHNOLOGY", "d1"), ("Kubernetes", "TECHNOLOGY", "d2"), ("Postgres", "TECHNOLOGY", "d1"), ("PostgreSQL", "TECHNOLOGY", "d3")]
    clusters, result = _clusters(rows)
    assert len(clusters) == 2
    assert {e.canonical_name for e in result.entities} == {"Kubernetes", "PostgreSQL"}


def test_dates_only_merge_on_exact_normalized_match():
    rows = [("Q3 2025", "DATE", "d1"), ("Q3 2026", "DATE", "d1"), ("Q3 2025", "DATE", "d2")]
    clusters, _ = _clusters(rows)
    assert len(clusters) == 2


def test_type_harmonization_prefers_confident_rule_labels():
    mentions = [
        Mention("m0", "c1", "d1", "Project Halcyon", "PROJECT", 0, 15, "rules", 0.95),
        Mention("m1", "c2", "d2", "Halcyon", "ORGANIZATION", 0, 7, "spacy", 0.6),
    ]
    result = EntityResolver(ResolutionSettings()).resolve(mentions)
    assert len(result.entities) == 1 and result.entities[0].entity_type == "PROJECT"
    assert any(d.decision == "type_harmonized" for d in result.decisions)


def test_decisions_expose_features_for_inspection():
    rows = [("Sam Okafor", "PERSON", "d1"), ("Samuel Okafor", "PERSON", "d2")]
    _, result = _clusters(rows)
    merge = [d for d in result.decisions if d.decision == "merge"][0]
    assert {"string", "surname_match", "context", "conflict"} <= set(merge.features)
    assert 0.0 <= merge.score <= 1.0


def test_context_similarity_contributes():
    rows = [("Brightwater", "ORGANIZATION", "d1"), ("Brightwater Health System", "ORGANIZATION", "d2")]
    mentions = _mentions(rows)
    resolver = EntityResolver(ResolutionSettings())
    same = np.ones(4, dtype=np.float32) / 2
    with_ctx = resolver.resolve(mentions, {"chunk_d1": same, "chunk_d2": same})
    without_ctx = resolver.resolve(mentions, {})
    score = lambda r: max((d.score for d in r.decisions if d.decision in ("merge", "ambiguous", "distinct")), default=0)  # noqa: E731
    assert score(with_ctx) > score(without_ctx)


def test_person_compatible_rules():
    assert person_compatible(["alice"], ["alice", "chen"])
    assert person_compatible(["chen"], ["alice", "chen"])
    assert person_compatible(["a", "chen"], ["alice", "chen"])
    assert not person_compatible(["b", "chen"], ["alice", "chen"])
    assert not person_compatible(["moreno"], ["alice", "chen"])
