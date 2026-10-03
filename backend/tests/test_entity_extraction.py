from app.entities.extraction import (
    ExtractedSpan,
    RuleBasedExtractor,
    derive_known_names,
    known_name_mentions,
    resolve_overlaps,
)


def _types(text: str) -> dict[str, str]:
    return {s.surface: s.entity_type for s in RuleBasedExtractor().extract(text)}


def test_rules_extract_core_types():
    found = _types(
        "At Quillon Labs, S. Okafor leads Project Halcyon on Kubernetes with Postgres since 2025-03-14 and Q3 2025. "
        "The Quillon Offsite 2026 is in Lisbon."
    )
    assert found["Quillon Labs"] == "ORGANIZATION"
    assert found["S. Okafor"] == "PERSON"
    assert found["Project Halcyon"] == "PROJECT"
    assert found["Kubernetes"] == "TECHNOLOGY" and found["Postgres"] == "TECHNOLOGY"
    assert found["2025-03-14"] == "DATE" and found["Q3 2025"] == "DATE"
    assert found["Quillon Offsite 2026"] == "EVENT"
    assert found["Lisbon"] == "LOCATION"


def test_rules_trim_sentence_initial_words():
    found = _types("At Kestrel Logistics the pilot started.")
    assert "Kestrel Logistics" in found and "At Kestrel Logistics" not in found


def test_json_fields_yield_people_and_orgs():
    found = _types("name: Ravi Narayan\norganization: Brightwater Health\nproject: Lodestar")
    assert found == {"Ravi Narayan": "PERSON", "Brightwater Health": "ORGANIZATION", "Lodestar": "PROJECT"}


def test_month_names_are_not_projects():
    assert "January" not in _types("During the January pilot we measured accuracy.")


def test_overlap_resolution_prefers_priority_then_length():
    spans = [
        ExtractedSpan("Halcyon", "ORGANIZATION", 8, 15, "spacy", 0.6, 50),
        ExtractedSpan("Project Halcyon", "PROJECT", 0, 15, "rules", 0.95, 90),
        ExtractedSpan("Kafka", "TECHNOLOGY", 20, 25, "rules", 0.95, 100),
    ]
    kept = resolve_overlaps(spans)
    assert [(s.surface, s.entity_type) for s in kept] == [("Project Halcyon", "PROJECT"), ("Kafka", "TECHNOLOGY")]


def test_known_names_recover_bare_mentions():
    names = derive_known_names([("Project Halcyon", "PROJECT"), ("Quillon Labs", "ORGANIZATION"), ("Marta Kowalski", "PERSON")])
    surfaces = {(n.surface, n.entity_type) for n in names}
    assert ("Halcyon", "PROJECT") in surfaces and ("Quillon", "ORGANIZATION") in surfaces
    assert ("Marta", "PERSON") in surfaces
    spans = known_name_mentions("Marta said Halcyon at Quillon is fine.", names, existing=[])
    assert {s.surface for s in spans} == {"Marta", "Halcyon", "Quillon"}
    # Already-covered spans are not duplicated.
    assert known_name_mentions("Halcyon", names, existing=[(0, 7)]) == []


def test_spacy_span_cleanup_keeps_name_after_heading():
    from app.entities.extraction import SpacyExtractor

    surface, start = SpacyExtractor._clean("Team\n\nSamantha Reyes", 10)
    assert surface == "Samantha Reyes" and start == 16
    assert SpacyExtractor._clean("the Quillon Labs's", 0) == ("Quillon Labs", 4)


def test_ner_segments_skip_headings_and_keep_offsets():
    from app.entities.extraction import _ner_segments

    text = "## Team\nSamantha Reyes leads Project Wren.\nSecond line.\n\n# H\n\nLast para"
    segments = _ner_segments(text)
    assert [seg for _, seg in segments] == ["Samantha Reyes leads Project Wren.\nSecond line.", "Last para"]
    assert all(text[offset : offset + len(seg)] == seg for offset, seg in segments)
