"""Calibration-only quantile inflation for temporal road support."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from iac_new.road_relative import compare_action_to_support


def _inflate(posterior: dict, lateral: tuple[float, float, float], heading: float, curvature: float) -> dict:
    result = copy.deepcopy(posterior)
    rows = result.get("support") or []
    for index, row in enumerate(rows):
        band = min(2, int(index * 3 / max(len(rows), 1)))
        for key, amount in (
            ("lateral_offset_range_m", lateral[band]),
            ("heading_change_range_rad", heading),
            ("curvature_range_1pm", curvature),
        ):
            bounds = row[key]
            bounds["q05"] = float(bounds["q05"]) - amount
            bounds["q95"] = float(bounds["q95"]) + amount
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lateral-values", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--curvature", type=float, default=0.0)
    parser.add_argument("--target", type=float, default=0.90)
    args = parser.parse_args()
    manifest = {str(row["sample_id"]): row for row in (json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip())}
    scores = {str(row["sample_id"]): row for row in (json.loads(line) for line in args.scores.read_text().splitlines() if line.strip())}
    values = [float(value) for value in args.lateral_values.split(",") if value.strip()]
    sweep = []
    for value in values:
        results = []
        widths = []
        for sample_id, scored in scores.items():
            record = manifest.get(sample_id)
            posterior = scored.get("road_relative_posterior")
            if not record or posterior is None or record.get("gt_candidate_id") is None:
                continue
            candidate = next((item for item in record.get("candidates", []) if str(item.get("candidate_id")) == str(record["gt_candidate_id"])), None)
            if candidate is None:
                continue
            future_times = np.asarray(record.get("future_times_s"), dtype=np.float64)
            action = np.asarray(candidate["trajectory"], dtype=np.float64)
            inflated = _inflate(posterior, (value, value, value), args.heading, args.curvature)
            results.append(compare_action_to_support(action, inflated, future_times))
            widths.extend(
                float(item["lateral_offset_range_m"]["q95"] - item["lateral_offset_range_m"]["q05"])
                for item in inflated.get("support", [])
            )
        sweep.append({
            "lateral_inflation_m": value,
            "num_samples": len(results),
            "joint_support_coverage": float(np.mean([item["joint_support_coverage"] for item in results])) if results else None,
            "heading_support_coverage": float(np.mean([item["heading_support_coverage"] for item in results])) if results else None,
            "lateral_support_coverage": float(np.mean([item["lateral_support_coverage"] for item in results])) if results else None,
            "curvature_support_coverage": float(np.mean([item["curvature_support_coverage"] for item in results])) if results else None,
            "mean_lateral_support_width_m": float(np.mean(widths)) if widths else None,
        })
    eligible = [item for item in sweep if item["joint_support_coverage"] is not None and item["joint_support_coverage"] >= args.target]
    selected = min(eligible, key=lambda item: item["mean_lateral_support_width_m"]) if eligible else None
    result = {
        "protocol": "temporal-road-support-calibration-v1",
        "manifest": str(args.manifest.resolve()),
        "scores": str(args.scores.resolve()),
        "target_joint_coverage": float(args.target),
        "sweep": sweep,
        "selected": selected,
        "selection_rule": "smallest mean support width among calibration values meeting target coverage",
        "holdout_must_not_be_used_for_selection": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
