"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import RetrievalMode

# ----------------------------------------------------------------- health


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    storage_backend: str
    embedding_model: str
    entity_extractor: str
    generation_backend: str
    counts: dict[str, int]


# -------------------------------------------------------------- documents


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    title: str
    source_type: str
    content_hash: str
    created_at: datetime
    chunk_count: int
    metadata: dict[str, Any]


class ChunkOut(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    metadata: dict[str, Any]


class DocumentDetail(DocumentSummary):
    chunks: list[ChunkOut]


class UploadItem(BaseModel):
    filename: str
    status: Literal["indexed", "duplicate", "error"]
    document_id: str | None = None
    chunks: int = 0
    mentions: int = 0
    error: str | None = None


class UploadResponse(BaseModel):
    items: list[UploadItem]
    entity_stats: dict[str, int] | None = None


# ----------------------------------------------------------------- search


class WeightsIn(BaseModel):
    vector: float = Field(0.0, ge=0)
    lexical: float = Field(0.0, ge=0)
    entity: float = Field(0.0, ge=0)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: RetrievalMode = "hybrid_entity"
    top_k: int = Field(5, ge=1, le=50)
    weights: WeightsIn | None = Field(
        None, description="Override alpha (vector), beta (lexical), gamma (entity) for this request."
    )
    threshold: float | None = Field(None, ge=0, le=1, description="Override the confidence threshold.")
    include_evidence_on_abstain: bool = Field(
        False, description="Return the (rejected) candidates even when abstaining, for debugging."
    )

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class SearchResultOut(BaseModel):
    rank: int
    text: str
    source: str = Field(..., description="Source filename")
    document_id: str
    chunk_id: str
    chunk_index: int
    title: str
    source_type: str
    char_start: int
    char_end: int
    location: dict[str, Any] = Field(default_factory=dict, description="Section / page / JSON path")
    retrieval_mode: str
    vector_score: float | None
    lexical_score: float | None
    entity_score: float | None
    final_score: float
    vector_similarity: float | None = Field(None, description="Raw cosine similarity")
    lexical_raw: float | None = Field(None, description="Raw BM25 / ts_rank score")
    retrieved_by: list[str]
    matched_entities: list[str]


class LinkedEntityOut(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    surface: str
    confidence: float
    method: str


class ConfidenceOut(BaseModel):
    score: float
    threshold: float
    signals: dict[str, float]
    reasons: list[str]
    gates_triggered: list[str]


class SearchResponse(BaseModel):
    query: str
    mode: str
    status: Literal["success", "insufficient_evidence"]
    confidence: float
    message: str | None = None
    results: list[SearchResultOut]
    confidence_detail: ConfidenceOut
    weights: dict[str, float]
    query_entities: list[LinkedEntityOut]
    unknown_entities: list[str]
    candidates_considered: int
    timings_ms: dict[str, float]


# ----------------------------------------------------------------- answer


class AnswerRequest(SearchRequest):
    pass


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    document_id: str
    source: str
    text: str


class AnswerResponse(BaseModel):
    query: str
    status: Literal["success", "insufficient_evidence", "generation_error"]
    answer: str | None
    message: str | None = None
    confidence: float
    generator: str
    citations: list[CitationOut]
    unsupported_citations: list[int] = Field(
        default_factory=list, description="Citation markers emitted by the model that match no evidence"
    )
    search: SearchResponse


# --------------------------------------------------------------- entities


class EntitySummary(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str]
    mention_count: int
    document_count: int


class MentionOut(BaseModel):
    mention_id: str
    surface: str
    chunk_id: str
    document_id: str
    source: str | None
    extractor: str
    context: str


class RelationshipOut(BaseModel):
    relationship_id: str
    source_entity_id: str
    source_name: str
    target_entity_id: str
    target_name: str
    relationship_type: str
    confidence: float
    supporting_chunk_id: str
    evidence: str
    method: str


class EntityDetail(EntitySummary):
    mentions: list[MentionOut]
    relationships: list[RelationshipOut]


class ResolveMentionIn(BaseModel):
    surface: str = Field(..., min_length=1)
    entity_type: str
    document: str = Field("adhoc", description="Document scope used for co-occurrence / coreference")


class ResolveRequest(BaseModel):
    mentions: list[ResolveMentionIn] | None = Field(
        None,
        description="If given, resolve only these mentions (stateless what-if). "
        "Otherwise re-run resolution over the whole index.",
    )


class DecisionOut(BaseModel):
    left: str
    right: str
    entity_type: str
    score: float
    decision: str
    features: dict[str, float]
    reason: str


class ResolvedClusterOut(BaseModel):
    canonical_name: str
    entity_type: str
    members: list[str]


class ResolveResponse(BaseModel):
    stats: dict[str, int | float]
    clusters: list[ResolvedClusterOut]
    decisions: list[DecisionOut]


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    mentions: int


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    weight: float


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# ------------------------------------------------------------- evaluation


class EvaluateRequest(BaseModel):
    dataset: str = Field("queries.json", description="File name inside eval/datasets/")
    modes: list[RetrievalMode] = Field(
        default_factory=lambda: ["lexical_only", "vector_only", "hybrid", "hybrid_entity"]
    )
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10])


class EvaluateResponse(BaseModel):
    dataset: str
    num_queries: int
    summary: dict[str, dict[str, Any]]
    markdown: str
