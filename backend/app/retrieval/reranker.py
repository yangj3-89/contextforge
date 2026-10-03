"""Reranking stage.

The default reranker is deterministic weighted scoring over normalized
component signals (or weighted reciprocal-rank fusion). It is behind a small
protocol so a cross-encoder can be dropped in later: such a reranker would
take the top-N weighted candidates, score (query, chunk.text) pairs with the
model, and either replace or blend ``final_score``. Component scores remain on
each candidate either way, so results stay explainable.
"""

from __future__ import annotations

from typing import Protocol

from app.core.config import SignalWeights
from app.models.domain import Chunk, ScoredCandidate
from app.retrieval.normalization import ranks, reciprocal_rank


class Reranker(Protocol):
    name: str

    def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        chunks: dict[str, Chunk],
        weights: SignalWeights,
    ) -> list[ScoredCandidate]: ...


class WeightedReranker:
    """final = alpha * vector + beta * lexical + gamma * entity (weights normalized to sum to 1)."""

    name = "weighted"

    def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        chunks: dict[str, Chunk],
        weights: SignalWeights,
    ) -> list[ScoredCandidate]:
        w = weights.normalized()
        for c in candidates:
            c.final_score = (
                w.vector * (c.vector_score or 0.0)
                + w.lexical * (c.lexical_score or 0.0)
                + w.entity * (c.entity_score or 0.0)
            )
            c.fused_score = c.final_score
        return sorted(candidates, key=lambda c: (-c.final_score, c.chunk_id))


class RRFReranker:
    """Weighted reciprocal rank fusion: sum_i w_i / (k + rank_i). Scale-free, ignores score gaps."""

    name = "rrf"

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        chunks: dict[str, Chunk],
        weights: SignalWeights,
    ) -> list[ScoredCandidate]:
        w = weights.normalized()
        signal_ranks = {}
        for name, weight in (("vector", w.vector), ("lexical", w.lexical), ("entity", w.entity)):
            if weight <= 0:
                continue
            raw = {c.chunk_id: getattr(c, f"{name}_raw") for c in candidates if getattr(c, f"{name}_raw")}
            signal_ranks[name] = (weight, ranks(raw))
        for c in candidates:
            c.final_score = sum(
                weight * reciprocal_rank(r[c.chunk_id], self.k)
                for weight, r in signal_ranks.values()
                if c.chunk_id in r
            )
            c.fused_score = c.final_score
        return sorted(candidates, key=lambda c: (-c.final_score, c.chunk_id))
