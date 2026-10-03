"""Sweep the abstention threshold using confidences recorded by an evaluation run.

No retrieval is re-run: for each query/mode the evaluation stored the aggregate
confidence and which hard gates fired, so abstaining at threshold t is simply
``gates_fired or confidence < t``. The threshold is selected on the *dev*
split (max F1 of the abstain decision, ties broken by fewer false abstentions)
and the same threshold is then reported on the *test* split.

    python scripts/sweep_confidence.py eval/results/latest.json --out eval/results/confidence_sweep.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

THRESHOLDS = [round(0.30 + 0.025 * i, 3) for i in range(17)]  # 0.30 .. 0.70


def metrics(rows: list[tuple[bool, float, bool]], t: float) -> dict[str, float | None]:
    tp = fp = fn = answerable = 0
    for answerable_q, conf, gated in rows:
        abstain = gated or conf < t
        answerable += answerable_q
        if abstain and not answerable_q:
            tp += 1
        elif abstain and answerable_q:
            fp += 1
        elif not abstain and not answerable_q:
            fn += 1
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "false_abstention_rate": fp / answerable if answerable else None}


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    modes = list(report["summary"])
    lines = [
        "# Confidence threshold sweep",
        "",
        f"Source: `{args.results}` (generated {report['meta']['timestamp']}, backend {report['meta']['storage_backend']}).",
        "Abstain if a hard gate fired or confidence < threshold. Threshold chosen on the dev split only.",
        "",
    ]
    for mode in modes:
        split_rows: dict[str, list[tuple[bool, float, bool]]] = {"dev": [], "test": []}
        for q in report["per_query"]:
            m = q["modes"].get(mode)
            if m:
                split_rows[q["split"]].append((q["answerable"], m["confidence"], bool(m["gates"])))
        best = max(
            THRESHOLDS,
            key=lambda t: (
                metrics(split_rows["dev"], t)["f1"],
                -(metrics(split_rows["dev"], t)["false_abstention_rate"] or 0),
                -t,
            ),
        )
        lines += [
            f"## {mode}",
            "",
            "| Threshold | Dev P | Dev R | Dev F1 | Dev false-abst. | Test P | Test R | Test F1 | Test false-abst. |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for t in THRESHOLDS:
            d, te = metrics(split_rows["dev"], t), metrics(split_rows["test"], t)
            mark = " **(dev-selected)**" if t == best else ""
            lines.append(
                f"| {t:.3f}{mark} | {fmt(d['precision'])} | {fmt(d['recall'])} | {fmt(d['f1'])} | {fmt(d['false_abstention_rate'])} "
                f"| {fmt(te['precision'])} | {fmt(te['recall'])} | {fmt(te['f1'])} | {fmt(te['false_abstention_rate'])} |"
            )
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
