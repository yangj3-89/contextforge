import json

from fastapi.testclient import TestClient

from tests.conftest import CORPUS


def test_health_reports_configuration(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["storage_backend"] == "memory"
    assert body["counts"]["documents"] == 0


def test_upload_list_get_delete_roundtrip(client: TestClient):
    files = [("files", (name, data)) for name, data in CORPUS.items()]
    files.append(("files", ("bad.docx", b"binary")))
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 200
    items = {i["filename"]: i for i in resp.json()["items"]}
    assert items["halcyon.md"]["status"] == "indexed" and items["halcyon.md"]["chunks"] > 0
    assert items["bad.docx"]["status"] == "error" and "Unsupported" in items["bad.docx"]["error"]
    assert resp.json()["entity_stats"]["entities"] > 0

    again = client.post("/documents/upload", files=[("files", ("halcyon.md", CORPUS["halcyon.md"]))]).json()
    assert again["items"][0]["status"] == "duplicate"

    docs = client.get("/documents").json()
    assert {d["filename"] for d in docs} == set(CORPUS)
    doc_id = next(d["document_id"] for d in docs if d["filename"] == "halcyon.md")
    detail = client.get(f"/documents/{doc_id}").json()
    assert detail["chunks"] and detail["chunks"][0]["chunk_id"].startswith(doc_id)

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get(f"/documents/{doc_id}").status_code == 404
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_upload_rejects_empty_file(client: TestClient):
    resp = client.post("/documents/upload", files=[("files", ("empty.txt", b""))])
    assert resp.json()["items"][0]["status"] == "error"


def test_search_returns_scores_and_provenance(indexed_client: TestClient):
    resp = indexed_client.post("/search", json={"query": "What does error QL-5031 mean?", "mode": "hybrid", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success" and body["message"] is None
    top = body["results"][0]
    assert top["source"] == "incident.txt"
    for key in ("vector_score", "lexical_score", "entity_score", "final_score", "chunk_id", "document_id", "char_start"):
        assert key in top
    assert top["retrieval_mode"] == "hybrid"
    assert body["weights"] == {"alpha_vector": 0.5, "beta_lexical": 0.5, "gamma_entity": 0.0}
    assert "total_ms" in body["timings_ms"]


def test_search_abstains_and_hides_results(indexed_client: TestClient):
    resp = indexed_client.post("/search", json={"query": "What did Bob Smith present about Project Nimbus?", "mode": "hybrid_entity"})
    body = resp.json()
    assert body["status"] == "insufficient_evidence"
    assert body["message"] == "ContextForge could not find sufficiently strong supporting evidence."
    assert body["results"] == []
    assert body["confidence_detail"]["reasons"]
    debug = indexed_client.post(
        "/search",
        json={"query": "What did Bob Smith present about Project Nimbus?", "mode": "hybrid_entity", "include_evidence_on_abstain": True},
    ).json()
    assert debug["status"] == "insufficient_evidence" and debug["results"]


def test_search_validation_errors(indexed_client: TestClient):
    assert indexed_client.post("/search", json={"query": "x", "mode": "bogus"}).status_code == 422
    assert indexed_client.post("/search", json={"query": "   "}).status_code == 422
    assert indexed_client.post("/search", json={"query": "x", "weights": {"vector": 0, "lexical": 0, "entity": 0}}).status_code == 422


def test_search_on_empty_index_is_a_conflict(client: TestClient):
    assert client.post("/search", json={"query": "anything"}).status_code == 409


def test_entities_endpoints(indexed_client: TestClient):
    entities = indexed_client.get("/entities", params={"entity_type": "person"}).json()
    names = {e["canonical_name"] for e in entities}
    assert "Alice Chen" in names
    alice = next(e for e in entities if e["canonical_name"] == "Alice Chen")
    assert "A. Chen" in alice["aliases"]
    detail = indexed_client.get(f"/entities/{alice['entity_id']}").json()
    assert detail["mentions"] and detail["mentions"][0]["source"]
    assert any(r["relationship_type"] == "works_on" for r in detail["relationships"])
    assert indexed_client.get("/entities/ent_missing").status_code == 404
    graph = indexed_client.get("/entities/graph", params={"min_weight": 0.3}).json()
    assert graph["nodes"] and graph["edges"]


def test_resolve_endpoint_adhoc_and_full(indexed_client: TestClient):
    adhoc = indexed_client.post(
        "/entities/resolve",
        json={"mentions": [
            {"surface": "OpenAI", "entity_type": "ORGANIZATION", "document": "a"},
            {"surface": "Open AI", "entity_type": "organization", "document": "b"},
            {"surface": "Initech", "entity_type": "ORGANIZATION", "document": "c"},
        ]},
    ).json()
    clusters = sorted(sorted(c["members"]) for c in adhoc["clusters"])
    assert clusters == [["Initech"], ["Open AI", "OpenAI"]]
    assert any(d["decision"] == "merge" for d in adhoc["decisions"])
    bad = indexed_client.post("/entities/resolve", json={"mentions": [{"surface": "x", "entity_type": "ALIEN"}]})
    assert bad.status_code == 422
    full = indexed_client.post("/entities/resolve", json={}).json()
    assert full["stats"]["entities"] > 0


def test_answer_endpoint_cites_evidence_and_respects_gate(indexed_client: TestClient):
    ok = indexed_client.post("/answer", json={"query": "Which database stores Halcyon embeddings?", "mode": "hybrid"}).json()
    assert ok["status"] == "success" and ok["generator"] == "extractive"
    assert ok["citations"] and "[" in ok["answer"]
    refused = indexed_client.post("/answer", json={"query": "Who won the 2018 FIFA World Cup?", "mode": "hybrid_entity"}).json()
    assert refused["status"] == "insufficient_evidence" and refused["answer"] is None


def test_evaluate_endpoint(indexed_client: TestClient, tmp_path, monkeypatch):
    from app.api import evaluation as eval_api

    dataset = {
        "name": "tiny",
        "queries": [
            {"id": "t1", "query": "error QL-5031", "category": "lexical", "relevant_sources": ["incident.txt"], "evidence": ["error QL-5031"]},
            {"id": "t2", "query": "capital of Australia", "category": "unanswerable", "answerable": False},
        ],
    }
    (tmp_path / "tiny.json").write_text(json.dumps(dataset))
    monkeypatch.setattr(eval_api, "DATASET_DIR", tmp_path)
    body = indexed_client.post("/evaluate", json={"dataset": "tiny.json", "modes": ["lexical_only", "hybrid"]}).json()
    assert body["num_queries"] == 2
    assert body["summary"]["lexical_only"]["hit@5"] == 1.0
    assert "| Method |" in body["markdown"]
    assert indexed_client.post("/evaluate", json={"dataset": "../secrets.json"}).status_code == 404
