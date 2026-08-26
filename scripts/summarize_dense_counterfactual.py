#!/usr/bin/env python3
"""Summarize dense counterfactual trajectory support and GT ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.scores.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ranks = []
    topk = {1: [], 5: [], 10: [], 20: []}
    candidate_counts = []
    prediction_sizes = []
    margins = []
    for row in rows:
        gt = row.get("gt_candidate_id")
        scores = sorted(row["candidate_scores"], key=lambda item: float(item["energy"]))
        ids = [str(item["candidate_id"]) for item in scores]
        if gt is None or str(gt) not in ids:
            continue
        rank = ids.index(str(gt)) + 1
        ranks.append(rank)
        candidate_counts.append(len(ids))
        prediction_sizes.append(int(row["prediction_set_size"]))
        gt_energy = float(scores[rank - 1]["energy"])
        next_energy = float(scores[rank]["energy"]) if rank < len(scores) else None
        if next_energy is not None:
            margins.append(next_energy - gt_energy)
        for k in topk:
            topk[k].append(rank <= k)
    result = {
        "rows": len(rows),
        "eligible_rows": len(ranks),
        "mean_candidate_count": float(np.mean(candidate_counts)) if candidate_counts else None,
        "median_gt_energy_rank": float(np.median(ranks)) if ranks else None,
        "max_gt_energy_rank": int(max(ranks)) if ranks else None,
        "topk_gt_coverage": {
            str(k): float(np.mean(values)) if values else None for k, values in topk.items()
        },
        "mean_prediction_set_size": float(np.mean(prediction_sizes)) if prediction_sizes else None,
        "median_gt_margin_to_next_candidate": float(np.median(margins)) if margins else None,
        "invalid_rows": sum(not bool(row.get("valid", False)) for row in rows),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
