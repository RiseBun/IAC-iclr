#!/usr/bin/env python3
"""Paired comparison of continuous motion reports on common samples/intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = {row["sample_id"]: row for row in load(args.first)["records"] if row.get("comparison")}
    second = {row["sample_id"]: row for row in load(args.second)["records"] if row.get("comparison")}
    common = sorted(set(first) & set(second))
    fields = ("speed_mps", "acceleration_mps2", "lateral_speed_mps", "yaw_rate_radps", "curvature_1pm")
    metrics = {}
    rng = np.random.default_rng(0)
    for field in fields:
        paired = []
        per_sample_difference = []
        for sample_id in common:
            left_rows = first[sample_id]["comparison"]["per_interval"]
            right_rows = second[sample_id]["comparison"]["per_interval"]
            sample_pairs = []
            for left, right in zip(left_rows, right_rows):
                if not left["evaluable"] or not right["evaluable"]:
                    continue
                left_error = left["absolute_errors"].get(field)
                right_error = right["absolute_errors"].get(field)
                if left_error is not None and right_error is not None:
                    pair = (float(left_error), float(right_error))
                    paired.append(pair)
                    sample_pairs.append(pair)
            if sample_pairs:
                per_sample_difference.append(float(np.mean([left - right for left, right in sample_pairs])))
        difference = np.asarray([left - right for left, right in paired], dtype=np.float64)
        sample_difference = np.asarray(per_sample_difference, dtype=np.float64)
        if len(sample_difference):
            bootstrap = np.mean(
                rng.choice(sample_difference, size=(10000, len(sample_difference)), replace=True), axis=1
            )
            confidence_interval = [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]
        else:
            confidence_interval = None
        metrics[field] = {
            "common_evaluable_intervals": len(paired),
            "common_evaluable_samples": len(per_sample_difference),
            f"{args.first_name}_mae": None if not paired else float(np.mean([item[0] for item in paired])),
            f"{args.second_name}_mae": None if not paired else float(np.mean([item[1] for item in paired])),
            f"{args.first_name}_minus_{args.second_name}_mae": None if not paired else float(np.mean(difference)),
            f"fraction_{args.first_name}_lower_error": None if not paired else float(np.mean(difference < 0.0)),
            "paired_sample_mean_difference_95ci": confidence_interval,
        }
    output = {
        "protocol": "paired-continuous-motion-ab-v1",
        "common_samples_with_decoder_outputs": len(common),
        "comparison_basis": "only intervals marked evaluable by both methods",
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
