"""Command-line entry point: rebuild the index from a corpus and run the benchmark.

    python -m app.evaluation.cli --data ../sample_data --dataset ../eval/datasets/queries.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from app.core.config import RETRIEVAL_MODES, Settings
from app.core.container import Container
from app.evaluation.dataset import load_dataset
from app.evaluation.entity_eval import evaluate_entity_resolution
from app.evaluation.report import render_markdown
from app.evaluation.runner import run_evaluation


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Run the ContextForge retrieval benchmark.")
    parser.add_argument("--backend", choices=["memory", "postgres"], default=None, help="override CF_STORAGE_BACKEND")
    parser.add_argument("--data", type=Path, default=root / "sample_data")
    parser.add_argument("--dataset", type=Path, default=root / "eval" / "datasets" / "queries.json")
    parser.add_argument("--er-gold", type=Path, default=root / "eval" / "datasets" / "entity_resolution_gold.json")
    parser.add_argument("--out", type=Path, default=root / "eval" / "results" / "latest.json")
    parser.add_argument("--modes", nargs="+", choices=RETRIEVAL_MODES, default=list(RETRIEVAL_MODES))
    parser.add_argument("--no-reindex", action="store_true", help="evaluate the existing index as-is")
    parser.add_argument("--summary-only", action="store_true", help="omit per-query details from the JSON output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    settings = Settings(storage_backend=args.backend) if args.backend else Settings()
    container = Container(settings)

    if not args.no_reindex:
        t0 = time.perf_counter()
        container.reset()
        results = container.ingest_directory(args.data)
        failed = [r for r in results if r.status == "error"]
        if failed:
            raise SystemExit(f"ingestion failed: {[(r.filename, r.error) for r in failed]}")
        print(f"Indexed {len(results)} files in {time.perf_counter() - t0:.1f}s -> {container.repo.counts()}")

    dataset = load_dataset(args.dataset)
    report = run_evaluation(container, dataset, args.modes)
    if args.er_gold and args.er_gold.exists():
        report["entity_resolution"] = evaluate_entity_resolution(args.er_gold, container.repo)

    if args.summary_only:
        report.pop("per_query", None)
        if "entity_resolution" in report:
            report["entity_resolution"].pop("unmatched_gold_mentions", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_markdown(report)
    args.out.with_suffix(".md").write_text(markdown)
    print(markdown)
    print(f"\nWrote {args.out} and {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
