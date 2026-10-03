"""Confidence estimation and abstention.

Confidence is an *engineered* score, not a calibrated probability. It combines
interpretable signals, each in [0, 1]:

relevance   absolute evidence strength of the best returned chunks: raw cosine
            similarity mapped through [vector_floor, vector_ceiling] and/or
            IDF-weighted query-term coverage (fraction of the query's
            informative terms present in the chunk).
margin      how clearly the top result separates from the runner-up.
agreement   overlap between the lexical and vector candidate lists (hybrid
            modes) - independent retrievers agreeing is evidence.
support     number of distinct source documents among the strong results.
entity      fraction of linked query entities that appear in the top results
            (hybrid_entity only).

Signals unavailable in a mode are dropped and the weights renormalized.
Hard gates abstain regardless of the aggregate: no semantically similar chunk
*and* no lexical coverage, or (hybrid_entity) a named entity in the query
that never occurs anywhere in the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import ConfidenceSettings

ABSTAIN_MESSAGE = "ContextForge could not find sufficiently strong supporting evidence."


@dataclass
class ConfidenceInputs:
    mode: str
    final_scores: list[float]
    """Final scores of the ranked results (descending)."""
    top_vector_raw: list[float | None]
    """Raw cosine of the top results (None when the vector signal is unused)."""
    top_coverage: list[float | None]
    """IDF-weighted query-term coverage of the top results (None when lexical is unused)."""
    top_documents: list[str]
    lexical_top: list[str] = field(default_factory=list)
    vector_top: list[str] = field(default_factory=list)
    top_chunk_ids: list[str] = field(default_factory=list)
    linked_entities: dict[str, float] = field(default_factory=dict)
    entities_in_top: set[str] = field(default_factory=set)
    unknown_entities: list[str] = field(default_factory=list)


@dataclass
class ConfidenceReport:
    confidence: float
    status: str  # "success" | "insufficient_evidence"
    signals: dict[str, float]
    reasons: list[str]
    gates_triggered: list[str]


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


class ConfidenceEstimator:
    def __init__(self, settings: ConfidenceSettings) -> None:
        self.s = settings

    def _vector_relevance(self, cosine: float) -> float:
        return _clip((cosine - self.s.vector_floor) / (self.s.vector_ceiling - self.s.vector_floor))

    def evaluate(self, x: ConfidenceInputs, threshold: float | None = None) -> ConfidenceReport:
        s = self.s
        threshold = s.threshold if threshold is None else threshold
        if not x.final_scores:
            return ConfidenceReport(0.0, "insufficient_evidence", {}, ["no candidate chunks were retrieved"], ["no_results"])

        window = 3
        cosines = [c for c in x.top_vector_raw[:window] if c is not None]
        coverages = [c for c in x.top_coverage[:window] if c is not None]
        best_cos = max(cosines) if cosines else None
        best_cov = max(coverages) if coverages else None

        signals: dict[str, float] = {}
        weights: dict[str, float] = {}
        rel_parts = []
        if best_cos is not None:
            signals["vector_similarity"] = round(best_cos, 4)
            rel_parts.append(self._vector_relevance(best_cos))
        if best_cov is not None:
            signals["lexical_coverage"] = round(best_cov, 4)
            rel_parts.append(best_cov)
        signals["relevance"] = sum(rel_parts) / len(rel_parts) if rel_parts else 0.0
        weights["relevance"] = s.w_relevance

        top = x.final_scores[0]
        second = x.final_scores[1] if len(x.final_scores) > 1 else 0.0
        signals["margin"] = _clip((top - second) / top) if top > 0 else 0.0
        weights["margin"] = s.w_margin

        if x.lexical_top and x.vector_top:
            k = s.support_window
            overlap = len(set(x.lexical_top[:k]) & set(x.vector_top[:k])) / k
            top1 = x.top_chunk_ids[0] if x.top_chunk_ids else None
            both = 1.0 if top1 in set(x.lexical_top[: 2 * k]) and top1 in set(x.vector_top[: 2 * k]) else 0.0
            signals["agreement"] = 0.5 * overlap + 0.5 * both
            weights["agreement"] = s.w_agreement

        strong_docs = {
            doc for doc, score in zip(x.top_documents[: s.support_window], x.final_scores, strict=False)
            if score >= 0.8 * top
        }
        signals["support"] = _clip(len(strong_docs) / 2)
        weights["support"] = s.w_support

        if x.mode == "hybrid_entity" and x.linked_entities:
            total = sum(x.linked_entities.values())
            hit = sum(c for e, c in x.linked_entities.items() if e in x.entities_in_top)
            signals["entity_consistency"] = hit / total if total > 0 else 0.0
            weights["entity_consistency"] = s.w_entity

        weight_sum = sum(weights.values())
        confidence = sum(signals[k] * w for k, w in weights.items()) / weight_sum
        signals = {k: round(v, 4) for k, v in signals.items()}

        gates: list[str] = []
        reasons: list[str] = []
        vector_weak = best_cos is None or best_cos < s.min_vector_similarity
        lexical_weak = best_cov is None or best_cov < s.min_lexical_coverage
        if vector_weak and lexical_weak:
            gates.append("weak_evidence")
            parts = []
            if best_cos is not None:
                parts.append(f"best cosine similarity {best_cos:.2f} < {s.min_vector_similarity}")
            if best_cov is not None:
                parts.append(f"best query-term coverage {best_cov:.2f} < {s.min_lexical_coverage}")
            reasons.append("no chunk is close to the query (" + "; ".join(parts) + ")")
        if x.mode == "hybrid_entity" and s.abstain_on_unknown_entities and x.unknown_entities:
            gates.append("unknown_entity")
            reasons.append(
                "query names entities that do not occur anywhere in the corpus: "
                + ", ".join(sorted(set(x.unknown_entities)))
            )
        if confidence < threshold:
            weakest = sorted(((signals[k], k) for k in weights), key=lambda t: t[0])[:2]
            reasons.append(
                f"aggregate confidence {confidence:.2f} < threshold {threshold:.2f} "
                f"(weakest signals: {', '.join(f'{k}={v:.2f}' for v, k in weakest)})"
            )

        status = "insufficient_evidence" if (gates or confidence < threshold) else "success"
        return ConfidenceReport(round(confidence, 4), status, signals, reasons, gates)
