#!/usr/bin/env python3
"""Build a paired RAFT/flow-backend comparison from decoder score files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "weighted_mean_joint_error": "lower",
    "median_joint_error": "lower",
    "soft_compatibility": "higher",
    "joint_coverage": "higher",
    "mean_heading_cosine": "higher",
    "mean_lateral_abs_m": "lower",
    "mean_yaw_abs_rad": "lower",
    "mean_speed_relative_error": "lower",
    "mean_curvature_abs_1pm": "lower",
}


def read_scores(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {
        str(row["sample_id"]): row["comparison_to_logged_trajectory"]
        for row in rows
        if row.get("valid") and row.get("comparison_to_logged_trajectory")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--baseline-name", default="raft_large")
    parser.add_argument("--challenger-name", default="sea_raft")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_scores(args.baseline)
    challenger = read_scores(args.challenger)
    common = sorted(set(baseline) & set(challenger))
    report: dict[str, Any] = {
        "protocol": "paired-flow-backend-comparison-v1",
        "baseline": args.baseline_name,
        "challenger": args.challenger_name,
        "num_baseline": len(baseline),
        "num_challenger": len(challenger),
        "num_paired": len(common),
        "baseline_only_sample_ids": sorted(set(baseline) - set(challenger)),
        "challenger_only_sample_ids": sorted(set(challenger) - set(baseline)),
        "metrics": {},
    }
    for metric, direction in METRICS.items():
        base = np.asarray([float(baseline[key][metric]) for key in common])
        challenge = np.asarray([float(challenger[key][metric]) for key in common])
        if direction == "lower":
            wins = challenge < base
        else:
            wins = challenge > base
        report["metrics"][metric] = {
            "direction": direction,
            "baseline_mean": float(np.mean(base)),
            "challenger_mean": float(np.mean(challenge)),
            "challenger_minus_baseline": float(np.mean(challenge - base)),
            "challenger_win_rate": float(np.mean(wins)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
