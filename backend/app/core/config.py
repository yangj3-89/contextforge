"""Central configuration.

Every tunable number in the system lives here (retrieval weights, normalization,
confidence thresholds, entity-resolution weights, chunk sizes) so that nothing
is scattered as a magic constant. Values can be overridden through environment
variables prefixed with ``CF_`` using ``__`` for nesting, e.g.
``CF_RETRIEVAL__CANDIDATE_POOL=50`` or ``CF_CONFIDENCE__THRESHOLD=0.5``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["lexical_only", "vector_only", "hybrid", "hybrid_entity"]
RETRIEVAL_MODES: tuple[str, ...] = ("lexical_only", "vector_only", "hybrid", "hybrid_entity")


class ChunkingSettings(BaseModel):
    max_tokens: int = 180
    """Soft upper bound on (approximate) tokens per chunk. MiniLM truncates at 256 word pieces."""
    min_tokens: int = 40
    """Chunks smaller than this are merged with the following block when possible."""
    overlap_sentences: int = 1
    """Sentences repeated at the start of a chunk when a long section is split."""


class SignalWeights(BaseModel):
    vector: float = 0.0
    lexical: float = 0.0
    entity: float = 0.0

    @model_validator(mode="after")
    def _non_negative(self) -> SignalWeights:
        if min(self.vector, self.lexical, self.entity) < 0:
            raise ValueError("signal weights must be non-negative")
        if self.vector + self.lexical + self.entity <= 0:
            raise ValueError("at least one signal weight must be positive")
        return self

    def normalized(self) -> SignalWeights:
        total = self.vector + self.lexical + self.entity
        return SignalWeights(
            vector=self.vector / total, lexical=self.lexical / total, entity=self.entity / total
        )


class RetrievalSettings(BaseModel):
    candidate_pool: int = 30
    """Candidates pulled from *each* retriever before fusion and reranking."""
    default_top_k: int = 5
    fusion: Literal["weighted", "rrf"] = "weighted"
    normalization: Literal["minmax", "max", "none"] = "minmax"
    """Per-query normalization for vector/lexical scores. Entity scores are already absolute in [0, 1]."""
    rrf_k: int = 60
    # alpha = vector, beta = lexical, gamma = entity. Chosen a priori (equal
    # lexical/vector split, a smaller entity share), not tuned on the test split.
    hybrid_weights: SignalWeights = SignalWeights(vector=0.5, lexical=0.5, entity=0.0)
    hybrid_entity_weights: SignalWeights = SignalWeights(vector=0.4, lexical=0.4, entity=0.2)
    lexical_scorer: Literal["bm25", "ts_rank"] = "bm25"
    """Postgres only: BM25 computed in SQL over tsvector, or native ts_rank_cd (no IDF)."""
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    hnsw_ef_search: int = 100


class EntityRetrievalSettings(BaseModel):
    hop_decay: float = 0.6
    """Multiplier applied to an entity reached through one relationship hop."""
    hinted_hop_decay: float = 1.0
    """Hop multiplier when the query names the neighbor's type ("the project led by ...").
    Setting it equal to ``hop_decay`` disables type-hinted expansion (ablation)."""
    header_context: bool = True
    """Propagate entities named in a chunk's title/section header to the chunk (context mentions)."""
    max_hops: int = 1
    min_link_confidence: float = 0.2
    """Query-entity links below this confidence are ignored (ambiguous aliases split their confidence)."""
    max_neighbors: int = 15
    """Expansion considers at most this many strongest neighbors per query."""
    partial_name_confidence: float = 0.7
    """Link confidence for a unique partial-name match (e.g. a surname only)."""
    fuzzy_link_threshold: float = 92.0
    """rapidfuzz ratio (0-100) above which a query n-gram fuzzily matches an alias."""


class ConfidenceSettings(BaseModel):
    threshold: float = 0.45
    """Results are returned only if the aggregate confidence is >= threshold."""
    # Signal weights (renormalized over the signals available for a mode).
    w_relevance: float = 0.45
    w_margin: float = 0.10
    w_agreement: float = 0.20
    w_support: float = 0.10
    w_entity: float = 0.15
    # Mapping of raw cosine similarity (MiniLM) onto a [0, 1] relevance signal.
    vector_floor: float = 0.20
    vector_ceiling: float = 0.60
    # Hard gates: abstain regardless of the aggregate when evidence is absent.
    min_vector_similarity: float = 0.25
    min_lexical_coverage: float = 0.34
    abstain_on_unknown_entities: bool = True
    support_window: int = 5
    """Number of top results inspected for multi-source support / agreement."""


class ResolutionSettings(BaseModel):
    auto_merge_threshold: float = 0.85
    candidate_threshold: float = 0.55
    """Pairs in [candidate_threshold, auto_merge_threshold) are recorded as ambiguous and kept separate."""
    # Log-odds weights for the pairwise match model (Fellegi-Sunter style, hand-set).
    bias: float = -4.0
    w_compact_equal: float = 6.0
    w_known_alias: float = 6.0
    w_string: float = 3.5
    w_surname_match: float = 3.0
    w_token_subset: float = 1.5
    w_context: float = 2.0
    w_doc_cooccur: float = 1.5
    w_unique_completion: float = 1.5
    w_ambiguous_completion: float = -2.5
    w_conflict: float = -6.0
    string_floor: float = 0.70
    context_floor: float = 0.20
    context_ceiling: float = 0.80


class GenerationSettings(BaseModel):
    backend: Literal["extractive", "ollama"] = "extractive"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    timeout_seconds: float = 60.0
    max_evidence: int = 5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CF_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    storage_backend: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://contextforge:contextforge@localhost:5432/contextforge"
    memory_snapshot_path: Path | None = None
    """If set, the in-memory backend persists its state to this file after writes."""

    embedding_backend: Literal["onnx", "sentence-transformers", "hashing"] = "onnx"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    model_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "contextforge" / "all-MiniLM-L6-v2-onnx"
    )
    model_auto_download: bool = True

    entity_extractor: Literal["auto", "spacy", "rules"] = "auto"

    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    entity_retrieval: EntityRetrievalSettings = EntityRetrievalSettings()
    confidence: ConfidenceSettings = ConfidenceSettings()
    resolution: ResolutionSettings = ResolutionSettings()
    generation: GenerationSettings = GenerationSettings()

    bootstrap_dir: Path | None = None
    """If set and the index is empty at startup, index every supported file in this directory."""

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_mb: int = 20


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
