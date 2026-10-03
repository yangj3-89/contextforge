"""Shared fixtures.

Unit tests use the deterministic hashing embedder and the rule-based extractor
so they are fast and independent of model downloads. Tests that need the real
MiniLM model or a PostgreSQL server are marked and skipped when unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.container import Container
from app.embeddings.hashing import HashingEmbedder
from app.entities.extraction import build_extractor
from app.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA = REPO_ROOT / "sample_data"

CORPUS: dict[str, bytes] = {
    "halcyon.md": b"""# Project Halcyon

Owner: Alice Chen (Quillon Labs)

## Summary

Project Halcyon is the document search platform of Quillon Labs. Alice Chen leads Project Halcyon.

## Storage

Halcyon stores embeddings in PostgreSQL with pgvector and uses an HNSW index for vector search.
""",
    "incident.txt": b"""Incident report for the indexing backlog.

The ingestion API returned error QL-5031 while Kafka consumer lag grew after a rebalance.

A. Chen will fix the offset commit logic. Jonas Lindqvist was the incident commander.
""",
    "lodestar.json": b"""{
  "title": "Lodestar team",
  "members": [
    {"name": "Ravi Narayan", "organization": "Brightwater Health", "project": "Lodestar", "stack": "Rust, DuckDB"},
    {"name": "Wei Zhang", "organization": "Brightwater Health", "project": "Lodestar", "stack": "Apache Kafka"}
  ]
}""",
    "ember.md": b"""# Project Ember

Samuel Okafor sponsors Project Ember at Kestrel Logistics.

## Routing

Project Ember computes delivery routes with Google OR-Tools and caches traffic estimates in Redis.
""",
}


def make_settings(**overrides) -> Settings:
    base = {
        "storage_backend": "memory",
        "embedding_backend": "hashing",
        "entity_extractor": "rules",
        "memory_snapshot_path": None,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def container(settings: Settings) -> Container:
    return Container(settings, embedder=HashingEmbedder(384), extractor=build_extractor("rules"))


@pytest.fixture
def indexed(container: Container) -> Container:
    results = container.ingest_files(list(CORPUS.items()))
    assert all(r.status == "indexed" for r in results), results
    return container


@pytest.fixture
def client(container: Container) -> TestClient:
    with TestClient(create_app(container)) as c:
        yield c


@pytest.fixture
def indexed_client(indexed: Container) -> TestClient:
    with TestClient(create_app(indexed)) as c:
        yield c


def postgres_url() -> str | None:
    return os.environ.get("CF_TEST_DATABASE_URL")
