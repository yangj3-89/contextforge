"""Evaluation runner: executes every query in every mode against the live index."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app import __version__
from app.core.container import Container
from app.evaluation.dataset import EvalDataset, resolve_relevance
from app.evaluation.metrics import (
    abstention_metrics,
    first_relevant_rank,
    hit_at_k,
    latency_stats,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

DEFAULT_MODES = ("lexical_only", "vector_only", "hybrid", "hybrid_entity")


def _git_state() -> dict[str, Any]:
    """Commit hash of the code that produced the results, or a dirty-tree marker.

    Results generated from uncommitted changes record ``git_commit: None`` and
    ``git_dirty: True`` rather than the (misleading) parent commit hash.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "backend/app", "sample_data", "eval/datasets"],
            capture_output=True, text=True, check=True, timeout=5, cwd=Path(__file__).resolve().parents[3],
        )
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    dirty = bool(status.stdout.strip())
    return {"git_commit": None if dirty else head.stdout.strip(), "git_dirty": dirty}


def _aggregate(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    answerable = [r for r in rows if r["answerable"]]
    out: dict[str, Any] = {"queries": len(rows), "answerable": len(answerable)}
    for k in k_values:
        out[f"recall@{k}"] = mean([r["recall"][str(k)] for r in answerable])
    for k in k_values:
        out[f"hit@{k}"] = mean([r["hit"][str(k)] for r in answerable])
    out["mrr"] = mean([r["rr"] for r in answerable])
    out["ndcg@10"] = mean([r["ndcg@10"] for r in answerable])
    out["source_accuracy@1"] = mean([r["source_hit@1"] for r in answerable])
    out["latency"] = latency_stats([r["latency_ms"] for r in rows])
    out["abstention"] = abstention_metrics([r["abstained"] for r in rows], [r["answerable"] for r in rows])
    answered = [r for r in answerable if not r["abstained"]]
    out["answered_hit@5"] = mean([r["hit"]["5"] for r in answered]) if answered else None
    out["mean_confidence_answerable"] = mean([r["confidence"] for r in answerable])
    unanswerable = [r for r in rows if not r["answerable"]]
    out["mean_confidence_unanswerable"] = mean([r["confidence"] for r in unanswerable]) if unanswerable else None
    return out


def run_evaluation(
    container: Container,
    dataset: EvalDataset,
    modes: list[str] | tuple[str, ...] = DEFAULT_MODES,
    k_values: list[int] | None = None,
    warmup: int = 3,
    weights_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    k_values = sorted(k_values or [1, 5, 10])
    depth = max(k_values)
    relevance = resolve_relevance(dataset, container.repo)
    engine = container.engine
    doc_of: dict[str, str] = {}

    per_query: dict[str, dict[str, Any]] = {
        q.id: {
            "id": q.id,
            "query": q.query,
            "category": q.category,
            "split": q.split,
            "answerable": q.answerable,
            "relevant_document_ids": sorted(relevance[q.id].document_ids),
            "relevant_chunk_ids": sorted(relevance[q.id].chunk_ids),
            "modes": {},
        }
        for q in dataset.queries
    }
    rows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for mode in modes:
        weights = (weights_override or {}).get(mode)
        for q in dataset.queries[:warmup]:
            engine.search(q.query, mode, top_k=depth, weights=weights)
        for q in dataset.queries:
            t0 = time.perf_counter()
            outcome = engine.search(q.query, mode, top_k=depth, weights=weights)
            latency = (time.perf_counter() - t0) * 1000
            ranked = [r.chunk_id for r in outcome.results]
            for r in outcome.results:
                doc_of[r.chunk_id] = r.document_id
            rel = relevance[q.id]
            row = {
                "answerable": q.answerable,
                "category": q.category,
                "split": q.split,
                "recall": {str(k): recall_at_k(ranked, rel.chunk_ids, k) for k in k_values},
                "hit": {str(k): hit_at_k(ranked, rel.chunk_ids, k) for k in k_values},
                "rr": reciprocal_rank(ranked, rel.chunk_ids),
                "ndcg@10": ndcg_at_k(ranked, rel.chunk_ids, 10),
                "source_hit@1": 1.0 if ranked and doc_of.get(ranked[0]) in rel.document_ids else 0.0,
                "latency_ms": latency,
                "abstained": outcome.abstained,
                "confidence": outcome.confidence.confidence,
            }
            rows_by_mode[mode].append(row)
            per_query[q.id]["modes"][mode] = {
                "first_relevant_rank": first_relevant_rank(ranked, rel.chunk_ids),
                "top5": ranked[:5],
                "confidence": outcome.confidence.confidence,
                "abstained": outcome.abstained,
                "gates": outcome.confidence.gates_triggered,
                "signals": outcome.confidence.signals,
                "linked_entities": [e.canonical_name for e in outcome.query_entities],
                "latency_ms": round(latency, 2),
            }

    summary = {mode: _aggregate(rows, k_values) for mode, rows in rows_by_mode.items()}
    by_category: dict[str, dict[str, Any]] = {}
    by_split: dict[str, dict[str, Any]] = {}
    for mode, rows in rows_by_mode.items():
        cats: dict[str, list[dict[str, Any]]] = defaultdict(list)
        splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            cats[r["category"]].append(r)
            splits[r["split"]].append(r)
        by_category[mode] = {c: _aggregate(rs, k_values) for c, rs in sorted(cats.items())}
        by_split[mode] = {s: _aggregate(rs, k_values) for s, rs in sorted(splits.items())}

    settings = container.settings
    categories: dict[str, int] = defaultdict(int)
    for q in dataset.queries:
        categories[q.category] += 1
    meta = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "contextforge_version": __version__,
        **_git_state(),
        "storage_backend": container.repo.backend_name,
        "embedding_model": container.embedder.name,
        "entity_extractor": container.extractor.name,
        "corpus": container.repo.counts(),
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            "queries": len(dataset.queries),
            "answerable": sum(q.answerable for q in dataset.queries),
            "unanswerable": sum(not q.answerable for q in dataset.queries),
            "categories": dict(sorted(categories.items())),
        },
        "config": {
            "candidate_pool": settings.retrieval.candidate_pool,
            "fusion": settings.retrieval.fusion,
            "normalization": settings.retrieval.normalization,
            "hybrid_weights": engine.default_weights("hybrid").normalized().model_dump(),
            "hybrid_entity_weights": engine.default_weights("hybrid_entity").normalized().model_dump(),
            "weights_override": weights_override,
            "confidence_threshold": settings.confidence.threshold,
            "k_values": k_values,
            "warmup_queries_per_mode": warmup,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    return {
        "meta": meta,
        "summary": summary,
        "by_category": by_category,
        "by_split": by_split,
        "per_query": list(per_query.values()),
    }
