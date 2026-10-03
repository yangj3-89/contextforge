"""Score normalization and rank fusion helpers.

Raw signals live on incompatible scales (BM25 is unbounded, cosine is in
[-1, 1]), so they are normalized per query over the candidate pool before
weighted fusion. Raw values are kept alongside for confidence estimation,
which needs absolute evidence strength rather than relative rank.
"""

from __future__ import annotations

_EPS = 1e-12


def min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < _EPS:
        return {k: (1.0 if hi > 0 else 0.0) for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def max_scale(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    hi = max(scores.values())
    if hi <= _EPS:
        return {k: 0.0 for k in scores}
    return {k: max(0.0, v) / hi for k, v in scores.items()}


def clip_unit(scores: dict[str, float]) -> dict[str, float]:
    return {k: min(1.0, max(0.0, v)) for k, v in scores.items()}


def normalize(scores: dict[str, float], method: str) -> dict[str, float]:
    if method == "minmax":
        return min_max(scores)
    if method == "max":
        return max_scale(scores)
    if method == "none":
        return clip_unit(scores)
    raise ValueError(f"unknown normalization method: {method}")


def ranks(scores: dict[str, float]) -> dict[str, int]:
    """1-based ranks, ties broken by key for determinism."""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {k: i + 1 for i, (k, _) in enumerate(ordered)}


def reciprocal_rank(rank: int, k: int) -> float:
    return 1.0 / (k + rank)
