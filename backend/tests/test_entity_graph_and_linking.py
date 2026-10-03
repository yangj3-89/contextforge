from app.core.config import EntityRetrievalSettings
from app.entities.extraction import build_extractor
from app.entities.graph import EntityGraph
from app.entities.linking import QueryEntityLinker
from app.entities.relationships import extract_relationships
from app.models.domain import Chunk, Entity, Mention, Relationship
from app.retrieval.entity_scorer import EntityScorer, hinted_types


def _entity(eid, name, etype, aliases=()):
    return Entity(eid, name, etype, list(aliases), 1, 1)


ENTITIES = [
    _entity("p_alice", "Alice Chen", "PERSON", ["A. Chen"]),
    _entity("p_moreno", "Alice Moreno", "PERSON"),
    _entity("proj_h", "Project Halcyon", "PROJECT", ["Halcyon"]),
    _entity("org_q", "Quillon Labs", "ORGANIZATION", ["Quillon"]),
    _entity("org_o", "OpenAI", "ORGANIZATION", ["OpenAI Inc."]),
    _entity("tech_pg", "PostgreSQL", "TECHNOLOGY", ["Postgres"]),
]


def _graph(relationships=()):
    mentions = [
        Mention("m1", "c_bio", "d1", "Alice Chen", "PERSON", 0, 10, "rules", 0.9, "p_alice"),
        Mention("m2", "c_store", "d1", "Halcyon", "PROJECT", 0, 7, "rules", 0.9, "proj_h"),
        Mention("m3", "c_store", "d1", "Postgres", "TECHNOLOGY", 20, 28, "rules", 0.9, "tech_pg"),
        Mention("m4", "c_other", "d2", "Alice Moreno", "PERSON", 0, 12, "rules", 0.9, "p_moreno"),
    ]
    return EntityGraph(ENTITIES, mentions, list(relationships))


def test_linker_exact_compact_and_partial_matches():
    linker = QueryEntityLinker(_graph(), EntityRetrievalSettings())
    names = lambda q: {link.canonical_name: link for link in linker.link(q).linked}  # noqa: E731
    assert "Alice Chen" in names("What does A. Chen work on?")
    assert names("Is Open AI a customer?")["OpenAI"].method == "exact"  # compact form
    linked = names("what about chen")
    assert linked["Alice Chen"].method == "partial"


def test_linker_splits_confidence_for_ambiguous_names():
    linker = QueryEntityLinker(_graph(), EntityRetrievalSettings())
    links = linker.link("What does Alice do?").linked
    assert {link.canonical_name for link in links} == {"Alice Chen", "Alice Moreno"}
    assert all(abs(link.confidence - 0.35) < 1e-6 for link in links)  # 0.7 partial confidence / 2


def test_linker_fuzzy_match_for_typos():
    linker = QueryEntityLinker(_graph(), EntityRetrievalSettings())
    links = linker.link("Who founded Quillon Lab?").linked
    assert [link.canonical_name for link in links] == ["Quillon Labs"]


def test_linker_reports_unmatched_named_entities():
    linker = QueryEntityLinker(_graph(), EntityRetrievalSettings(), build_extractor("rules"))
    result = linker.link("What is Project Nimbus at Quillon Labs?")
    assert "Project Nimbus" in result.unmatched
    assert "Quillon Labs" in {link.canonical_name for link in result.linked}


def test_relationship_extraction_pattern_record_and_context():
    chunk = Chunk("c1", "d1", 0, "Alice Chen leads Project Halcyon. Postgres stores the vectors.", "", 10, 0, 62, {})
    mentions = [
        Mention("a", "c1", "d1", "Alice Chen", "PERSON", 0, 10, "rules", 0.9, "p_alice"),
        Mention("b", "c1", "d1", "Project Halcyon", "PROJECT", 17, 32, "rules", 0.9, "proj_h"),
        Mention("c", "c1", "d1", "Postgres", "TECHNOLOGY", 34, 42, "rules", 0.9, "tech_pg"),
    ]
    types = {e.id: e.entity_type for e in ENTITIES}
    rels = extract_relationships(chunk, mentions, types, context_entities=["proj_h"])
    by_type = {(r.source_entity_id, r.relationship_type, r.target_entity_id): r for r in rels}
    works_on = by_type[("p_alice", "works_on", "proj_h")]
    assert works_on.method == "pattern_between" and works_on.supporting_chunk_id == "c1"
    # "stores" is a 'uses' trigger; Halcyon is only in the header context for that sentence.
    assert ("proj_h", "uses", "tech_pg") in by_type and by_type[("proj_h", "uses", "tech_pg")].method == "context"

    record = Chunk("c2", "d1", 1, "name: Alice Chen\norganization: Quillon Labs", "", 8, 0, 44, {"kind": "record"})
    rec_mentions = [
        Mention("d", "c2", "d1", "Alice Chen", "PERSON", 6, 16, "rules", 0.9, "p_alice"),
        Mention("e", "c2", "d1", "Quillon Labs", "ORGANIZATION", 31, 43, "rules", 0.9, "org_q"),
    ]
    rec = extract_relationships(record, rec_mentions, types)
    assert [(r.relationship_type, r.method) for r in rec] == [("affiliated_with", "record")]


def test_graph_edges_aggregate_with_noisy_or():
    rels = [
        Relationship("r1", "p_alice", "proj_h", "works_on", 0.5, "c1", "", "pattern_sentence"),
        Relationship("r2", "p_alice", "proj_h", "works_on", 0.5, "c2", "", "pattern_sentence"),
    ]
    g = _graph(rels)
    edge = g.neighbors("p_alice")["proj_h"]
    assert abs(edge.weight - 0.75) < 1e-9
    assert g.neighbors("proj_h")["p_alice"].supporting_chunks == {"c1", "c2"}


def test_entity_scorer_direct_and_type_hinted_expansion():
    rels = [Relationship("r1", "p_alice", "proj_h", "works_on", 0.85, "c_bio", "", "pattern_between")]
    g = _graph(rels)
    linker = QueryEntityLinker(g, EntityRetrievalSettings())
    scorer = EntityScorer(g, EntityRetrievalSettings(hop_decay=0.5, hinted_hop_decay=1.0))

    plain = scorer.plan(linker.link("Alice Chen"), "Alice Chen")
    hinted = scorer.plan(linker.link("the project Alice Chen leads"), "the project Alice Chen leads")
    assert plain.direct == {"p_alice": 1.0}
    assert abs(plain.expanded["proj_h"] - 0.425) < 1e-9
    assert abs(hinted.expanded["proj_h"] - 0.85) < 1e-9

    scores = scorer.score_chunks(hinted, ["c_bio", "c_store", "c_other"])
    assert scores["c_bio"][0] == 1.0
    assert abs(scores["c_store"][0] - 0.85) < 1e-9
    assert "Project Halcyon (via Alice Chen)" in scores["c_store"][1]
    assert scores["c_other"][0] == 0.0
    assert [cid for cid, _ in scorer.candidates(hinted, 10)] == ["c_bio", "c_store"]


def test_type_hints():
    assert hinted_types("Which database does the project led by Alice use?") == {"TECHNOLOGY", "PROJECT"}
    assert hinted_types("Who runs the company?") == {"PERSON", "ORGANIZATION"}
