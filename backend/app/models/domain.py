"""Domain objects shared by every layer.

These are plain dataclasses (not Pydantic models) because they are created in
hot loops during ingestion and retrieval; the API layer converts them into
Pydantic response schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

ENTITY_TYPES: tuple[str, ...] = (
    "PERSON",
    "ORGANIZATION",
    "PROJECT",
    "LOCATION",
    "DATE",
    "TECHNOLOGY",
    "EVENT",
)


@dataclass(slots=True)
class Document:
    id: str
    filename: str
    source_type: str  # "txt" | "markdown" | "pdf" | "json"
    content_hash: str
    created_at: datetime
    text: str
    """Normalized full text; chunk character offsets index into this string."""
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    """Exact slice of ``Document.text`` (provenance)."""
    search_text: str
    """``text`` prefixed with a contextual header (title / section); what gets embedded and indexed."""
    token_count: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass(slots=True)
class Mention:
    id: str
    chunk_id: str
    document_id: str
    surface: str
    entity_type: str
    char_start: int
    """Offset within the chunk text."""
    char_end: int
    extractor: str
    confidence: float
    entity_id: str | None = None


@dataclass(slots=True)
class Entity:
    id: str
    canonical_name: str
    entity_type: str
    aliases: list[str]
    mention_count: int
    document_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Relationship:
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    supporting_chunk_id: str
    evidence: str
    method: str  # "pattern" | "cooccurrence" | "record"


@dataclass(slots=True)
class ResolutionDecision:
    """One inspected pair from entity resolution (kept for explainability)."""

    left: str
    right: str
    entity_type: str
    score: float
    decision: str  # "merge" | "ambiguous" | "distinct" | "blocked_conflict"
    features: dict[str, float]
    reason: str


@dataclass(slots=True)
class ScoredCandidate:
    chunk_id: str
    vector_raw: float | None = None
    lexical_raw: float | None = None
    entity_raw: float | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    entity_score: float | None = None
    fused_score: float = 0.0
    final_score: float = 0.0
    sources: set[str] = field(default_factory=set)
    """Which retrievers proposed this candidate: lexical / vector / entity."""
    matched_entities: list[str] = field(default_factory=list)
