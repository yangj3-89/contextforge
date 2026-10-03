"""Retrieval, latency and abstention metrics.

Definitions (computed over answerable queries unless stated):

* Recall@k  = |relevant ∩ top-k| / |relevant|, averaged over queries.
* Hit@k     = 1 if any relevant chunk is in the top-k.
* MRR       = mean of 1 / rank of the first relevant chunk (0 if none in the
              retrieved list, which has depth max(k)).
* nDCG@10   = binary-relevance nDCG.
* Source accuracy@1 = top-1 chunk comes from a relevant document.
* Abstention (all queries): positive class = "should abstain" (unanswerable).
  precision = correct abstentions / all abstentions,
  recall    = correct abstentions / unanswerable queries,
  false-abstention rate = abstentions on answerable queries / answerable queries.
"""

from __future__ import annotations

import math

import numpy as np


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def first_relevant_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return i
    return None


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    rank = first_relevant_rank(ranked, relevant)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked[:k]) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def latency_stats(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    arr = np.asarray(values_ms)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }


def abstention_metrics(abstained: list[bool], answerable: list[bool]) -> dict[str, float | int | None]:
    tp = sum(1 for a, ans in zip(abstained, answerable, strict=True) if a and not ans)
    fp = sum(1 for a, ans in zip(abstained, answerable, strict=True) if a and ans)
    fn = sum(1 for a, ans in zip(abstained, answerable, strict=True) if not a and not ans)
    n_answerable = sum(answerable)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "abstentions": tp + fp,
        "correct_abstentions": tp,
        "false_abstentions": fp,
        "missed_abstentions": fn,
        "abstention_precision": None if precision is None else round(precision, 4),
        "abstention_recall": None if recall is None else round(recall, 4),
        "abstention_f1": None if f1 is None else round(f1, 4),
        "false_abstention_rate": round(fp / n_answerable, 4) if n_answerable else None,
    }


def mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0
