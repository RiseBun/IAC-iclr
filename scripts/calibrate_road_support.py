#!/usr/bin/env python3
"""Calibrate road-relative support inflation on a calibration split.

This script only widens posterior intervals; it never changes the point
trajectory or decoder. Use a scene-disjoint split for a benchmark calibration
and keep the selected value fixed for holdout evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.road_relative import compare_action_to_support


def _inflate(posterior: dict[str, Any], lateral: float, heading: float, curvature: float) -> dict[str, Any]:
    result = copy.deepcopy(posterior)
    for row in result.get("support", []):
        for key, amount in (
            ("lateral_offset_range_m", lateral),
            ("heading_change_range_rad", heading),
            ("curvature_range_1pm", curvature),
        ):
            bounds = row[key]
            bounds["q05"] = float(bounds["q05"]) - amount
            bounds["q95"] = float(bounds["q95"]) + amount
    return result


def run(path: Path, lateral_values: list[float], heading: float, curvature: float, target: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for group in payload.get("groups", []):
        times = np.asarray([row["time_s"] for row in group["branches"][0]["road_relative_posterior"]["support"]], dtype=np.float64)
        for branch in group.get("branches", []):
            posterior = branch.get("road_relative_posterior")
            action = branch.get("executed_action")
            if posterior is None or action is None:
                continue
            rows.append((np.asarray(action, dtype=np.float64), posterior, times))
    sweep = []
    for lateral in lateral_values:
        scores = [compare_action_to_support(action, _inflate(posterior, lateral, heading, curvature), times) for action, posterior, times in rows]
        sweep.append({
            "lateral_inflation_m": float(lateral),
            "num_branches": len(scores),
            "mean_joint_support_coverage": float(np.mean([score["joint_support_coverage"] for score in scores])) if scores else None,
            "mean_heading_support_coverage": float(np.mean([score["heading_support_coverage"] for score in scores])) if scores else None,
            "mean_curvature_support_coverage": float(np.mean([score["curvature_support_coverage"] for score in scores])) if scores else None,
        })
    selected = next((row for row in sweep if row["mean_joint_support_coverage"] is not None and row["mean_joint_support_coverage"] >= target), None)
    return {
        "protocol": "road-relative-support-calibration-v1",
        "source": str(path),
        "target_joint_coverage": float(target),
        "heading_inflation_rad": float(heading),
        "curvature_inflation_1pm": float(curvature),
        "sweep": sweep,
        "selected": selected,
        "warning": "selection is valid for holdout only when this input is a scene-disjoint calibration split",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lateral-values", default="0,0.05,0.1,0.2,0.3,0.5,0.75")
    parser.add_argument("--heading-inflation", type=float, default=0.0)
    parser.add_argument("--curvature-inflation", type=float, default=0.0)
    parser.add_argument("--target", type=float, default=0.90)
    args = parser.parse_args()
    values = [float(value) for value in args.lateral_values.split(",") if value.strip()]
    result = run(args.input, values, args.heading_inflation, args.curvature_inflation, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
