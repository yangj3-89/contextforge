"""Compare several evaluation result files (e.g. ablations) in one Markdown table.

    python scripts/compare_results.py eval/results/latest.json eval/results/ablation_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODE_LABELS = {"lexical_only": "Lexical", "vector_only": "Vector", "hybrid": "Hybrid", "hybrid_entity": "Hybrid + Entity"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--mode", action="append", help="restrict to these modes (repeatable)")
    args = parser.parse_args()
    lines = [
        "| Run | Method | Recall@1 | Recall@5 | MRR | Relational MRR | Relational Hit@5 | Abstain P | Abstain R | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for path in args.results:
        report = json.loads(path.read_text())
        for mode, s in report["summary"].items():
            if args.mode and mode not in args.mode:
                continue
            rel = report["by_category"][mode].get("relational", {})
            a = s["abstention"]

            def f(x):
                return "n/a" if x is None else f"{x:.3f}"

            lines.append(
                f"| {path.stem} | {MODE_LABELS.get(mode, mode)} | {f(s['recall@1'])} | {f(s['recall@5'])} | {f(s['mrr'])} "
                f"| {f(rel.get('mrr'))} | {f(rel.get('hit@5'))} | {f(a['abstention_precision'])} | {f(a['abstention_recall'])} "
                f"| {s['latency']['p95_ms']:.1f} |"
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
