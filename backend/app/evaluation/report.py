"""Render evaluation results as Markdown tables."""

from __future__ import annotations

from typing import Any

MODE_LABELS = {
    "lexical_only": "Lexical (BM25)",
    "vector_only": "Vector (MiniLM)",
    "hybrid": "Hybrid",
    "hybrid_entity": "Hybrid + Entity",
}


def _f(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def retrieval_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Method | Recall@1 | Recall@5 | Recall@10 | Hit@5 | MRR | nDCG@10 | Source acc@1 | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, s in summary.items():
        lat = s["latency"]
        lines.append(
            f"| {MODE_LABELS.get(mode, mode)} | {_f(s['recall@1'])} | {_f(s['recall@5'])} | {_f(s['recall@10'])} "
            f"| {_f(s['hit@5'])} | {_f(s['mrr'])} | {_f(s['ndcg@10'])} | {_f(s['source_accuracy@1'])} "
            f"| {lat['p50_ms']:.1f} ms | {lat['p95_ms']:.1f} ms |"
        )
    return "\n".join(lines)


def abstention_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Method | Abstain precision | Abstain recall | False-abstention rate | Hit@5 on answered | Mean conf. (answerable) | Mean conf. (unanswerable) |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, s in summary.items():
        a = s["abstention"]
        lines.append(
            f"| {MODE_LABELS.get(mode, mode)} | {_f(a['abstention_precision'])} | {_f(a['abstention_recall'])} "
            f"| {_f(a['false_abstention_rate'])} | {_f(s['answered_hit@5'])} "
            f"| {_f(s['mean_confidence_answerable'])} | {_f(s['mean_confidence_unanswerable'])} |"
        )
    return "\n".join(lines)


def category_table(by_category: dict[str, Any], metric: str = "mrr") -> str:
    modes = list(by_category)
    categories = sorted({c for m in modes for c in by_category[m]})
    header = "| Category | n | " + " | ".join(MODE_LABELS.get(m, m) for m in modes) + " |"
    lines = [header, "|---|---|" + "---|" * len(modes)]
    for c in categories:
        first = by_category[modes[0]].get(c, {})
        n = first.get("answerable", 0)
        if n == 0:
            continue
        cells = [_f(by_category[m].get(c, {}).get(metric)) for m in modes]
        lines.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def split_table(by_split: dict[str, Any]) -> str:
    modes = list(by_split)
    splits = sorted({s for m in modes for s in by_split[m]})
    lines = ["| Split | Method | Recall@5 | MRR | Abstain P | Abstain R |", "|---|---|---|---|---|---|"]
    for sp in splits:
        for m in modes:
            s = by_split[m].get(sp)
            if not s:
                continue
            a = s["abstention"]
            lines.append(
                f"| {sp} | {MODE_LABELS.get(m, m)} | {_f(s['recall@5'])} | {_f(s['mrr'])} "
                f"| {_f(a['abstention_precision'])} | {_f(a['abstention_recall'])} |"
            )
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    ds = meta["dataset"]
    corpus = meta["corpus"]
    cfg = meta["config"]
    parts = [
        "# ContextForge evaluation results",
        "",
        f"Generated {meta['timestamp']} by `scripts/run_eval.py` "
        + (
            f"from commit `{meta['git_commit']}`. "
            if meta.get("git_commit")
            else "from the working tree committed together with this file. "
        )
        + "All numbers below are produced by the evaluation harness; nothing is hand-entered.",
        "",
        f"* Storage backend: **{meta['storage_backend']}**; embeddings: **{meta['embedding_model']}**; "
        f"entity extraction: **{meta['entity_extractor']}**",
        f"* Corpus: {corpus['documents']} documents, {corpus['chunks']} chunks, {corpus['entities']} entities, "
        f"{corpus['relationships']} relationship rows",
        f"* Dataset `{ds['name']}` v{ds['version']}: {ds['queries']} queries "
        f"({ds['answerable']} answerable, {ds['unanswerable']} unanswerable); categories: "
        + ", ".join(f"{k}={v}" for k, v in ds["categories"].items()),
        f"* Candidate pool {cfg['candidate_pool']} per retriever, fusion `{cfg['fusion']}`, normalization "
        f"`{cfg['normalization']}`, confidence threshold {cfg['confidence_threshold']}",
        f"* Hybrid weights {cfg['hybrid_weights']}; hybrid+entity weights {cfg['hybrid_entity_weights']}",
        f"* Environment: Python {meta['environment']['python']}, {meta['environment']['cpu_count']} CPUs, "
        f"{meta['environment']['platform']}",
        "",
        "## Retrieval quality (answerable queries)",
        "",
        retrieval_table(report["summary"]),
        "",
        "Latency is end-to-end `engine.search` time per query (query embedding, candidate generation, "
        "scoring, reranking and confidence estimation), measured after "
        f"{cfg['warmup_queries_per_mode']} warm-up queries per mode.",
        "",
        "## MRR by query category",
        "",
        category_table(report["by_category"], "mrr"),
        "",
        "## Hit@5 by query category",
        "",
        category_table(report["by_category"], "hit@5"),
        "",
        "## Abstention (all queries; positive class = unanswerable)",
        "",
        abstention_table(report["summary"]),
        "",
        "## Dev / test split",
        "",
        split_table(report["by_split"]),
        "",
    ]
    if "entity_resolution" in report:
        parts += ["## Entity resolution", "", entity_resolution_table(report["entity_resolution"]), ""]
    return "\n".join(parts)


def entity_resolution_table(er: dict[str, Any]) -> str:
    lines = [
        "| Method | Pairwise precision | Pairwise recall | Pairwise F1 | Predicted clusters | Gold clusters |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in er["methods"].items():
        lines.append(
            f"| {name} | {_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} | {m['predicted_clusters']} | {m['gold_clusters']} |"
        )
    lines.append("")
    lines.append(
        f"Gold mentions: {er['gold_mentions']}; found among extracted mentions: {er['matched_mentions']} "
        f"(coverage {er['coverage']:.3f}). Pairwise scores are computed over matched mentions only."
    )
    return "\n".join(lines)
