"""Entity-resolution evaluation against hand-labelled gold clusters.

Gold file format::

    {"name": "...", "entities": [
        {"gold_id": "alice_chen", "type": "PERSON",
         "mentions": [{"source": "halcyon_design.md", "surface": "Alice Chen"}, ...]}]}

Each gold mention is matched to extracted mentions with the same surface in the
same source file; its predicted entity is the majority entity of those
mentions. Pairwise precision/recall/F1 are computed over matched mentions and
compared against two string-matching baselines.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from app.db.repository import Repository
from app.entities.normalization import basic_normalize, normalize_name


def _pairwise(items: list[tuple[str, str]]) -> dict[str, Any]:
    tp = fp = fn = 0
    for (g1, p1), (g2, p2) in combinations(items, 2):
        same_gold, same_pred = g1 == g2, p1 == p2
        if same_gold and same_pred:
            tp += 1
        elif same_pred:
            fp += 1
        elif same_gold:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "predicted_clusters": len({p for _, p in items}),
        "gold_clusters": len({g for g, _ in items}),
    }


def evaluate_entity_resolution(gold_path: Path, repo: Repository) -> dict[str, Any]:
    gold = json.loads(gold_path.read_text())
    filename_of = {d.id: d.filename for d in repo.list_documents()}
    index: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for m in repo.all_mentions():
        if m.entity_id and m.extractor != "header":
            index[(filename_of.get(m.document_id, ""), m.surface)][m.entity_id] += 1

    system: list[tuple[str, str]] = []
    exact: list[tuple[str, str]] = []
    normalized: list[tuple[str, str]] = []
    unmatched: list[dict[str, str]] = []
    total = 0
    for entity in gold["entities"]:
        for mention in entity["mentions"]:
            total += 1
            counts = index.get((mention["source"], mention["surface"]))
            if not counts:
                unmatched.append({"gold_id": entity["gold_id"], **mention})
                continue
            gid = entity["gold_id"]
            system.append((gid, counts.most_common(1)[0][0]))
            exact.append((gid, f"{entity['type']}:{basic_normalize(mention['surface'])}"))
            normalized.append((gid, f"{entity['type']}:{normalize_name(mention['surface'], entity['type'])}"))

    return {
        "gold_file": gold_path.name,
        "gold_mentions": total,
        "matched_mentions": len(system),
        "coverage": len(system) / total if total else 0.0,
        "methods": {
            "Exact string match (baseline)": _pairwise(exact),
            "Normalized name match (baseline)": _pairwise(normalized),
            "ContextForge resolver": _pairwise(system),
        },
        "unmatched_gold_mentions": unmatched,
    }
