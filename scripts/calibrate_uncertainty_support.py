"""Calibrate uncertainty-conditioned road support inflation on a split."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.road_relative import compare_action_to_support


def _restore_and_inflate(
    posterior: dict[str, Any],
    uncertainty: dict[str, Any] | None,
    *,
    multiplier: float,
    base_factors: tuple[float, float, float],
    scale_px: float,
) -> dict[str, Any]:
    result = copy.deepcopy(posterior)
    rows = result.get("support") or []
    existing = posterior.get("support_inflation", {}).get("per_interval", {})
    old_lateral = np.asarray(existing.get("lateral_m") or [0.0] * len(rows), dtype=np.float64)
    old_heading = np.asarray(existing.get("heading_rad") or [0.0] * len(rows), dtype=np.float64)
    old_curvature = np.asarray(existing.get("curvature_1pm") or [0.0] * len(rows), dtype=np.float64)
    by_interval = (uncertainty or {}).get("by_interval") or []
    norms = np.asarray([
        float(item.get("median_px")) / max(scale_px, 1e-6) if item.get("median_px") is not None else 1.0
        for item in by_interval[: len(rows)]
    ], dtype=np.float64)
    if len(norms) < len(rows):
        norms = np.pad(norms, (0, len(rows) - len(norms)), constant_values=1.0)
    norms = np.clip(norms, 0.0, 4.0)
    for index, row in enumerate(rows):
        new_amounts = [float(multiplier * norms[index] * factor) for factor in base_factors]
        for key, old, amount in zip(
            ("lateral_offset_range_m", "heading_change_range_rad", "curvature_range_1pm"),
            (old_lateral[index], old_heading[index], old_curvature[index]),
            new_amounts,
        ):
            bounds = row[key]
            # The scored posterior may already include the default diagnostic
            # inflation. Restore the decoder support before applying a sweep.
            bounds["q05"] = float(bounds["q05"]) + float(old)
            bounds["q95"] = float(bounds["q95"]) - float(old)
            bounds["q05"] -= amount
            bounds["q95"] += amount
    return result


def _evaluate(scores: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], multipliers: list[float], factors: tuple[float, float, float], scale_px: float, target: float) -> dict[str, Any]:
    sweep = []
    for multiplier in multipliers:
        metrics = []
        widths = []
        for sample_id, scored in scores.items():
            record = manifest.get(sample_id)
            posterior = scored.get("road_relative_posterior")
            if not record or posterior is None:
                continue
            gt_id = record.get("gt_candidate_id")
            candidate = next((item for item in record.get("candidates", []) if str(item.get("candidate_id")) == str(gt_id)), None)
            uncertainty = scored.get("raft_refinement_uncertainty")
            if candidate is None or uncertainty is None:
                continue
            inflated = _restore_and_inflate(posterior, uncertainty, multiplier=multiplier, base_factors=factors, scale_px=scale_px)
            metrics.append(compare_action_to_support(np.asarray(candidate["trajectory"], dtype=np.float64), inflated, np.asarray(record["future_times_s"], dtype=np.float64)))
            widths.extend(float(item["lateral_offset_range_m"]["q95"] - item["lateral_offset_range_m"]["q05"]) for item in inflated.get("support", []))
        sweep.append({
            "multiplier": multiplier,
            "num_samples": len(metrics),
            "joint_support_coverage": float(np.mean([item["joint_support_coverage"] for item in metrics])) if metrics else None,
            "heading_support_coverage": float(np.mean([item["heading_support_coverage"] for item in metrics])) if metrics else None,
            "lateral_support_coverage": float(np.mean([item["lateral_support_coverage"] for item in metrics])) if metrics else None,
            "curvature_support_coverage": float(np.mean([item["curvature_support_coverage"] for item in metrics])) if metrics else None,
            "mean_lateral_support_width_m": float(np.mean(widths)) if widths else None,
        })
    eligible = [item for item in sweep if item["joint_support_coverage"] is not None and item["joint_support_coverage"] >= target]
    selected = min(eligible, key=lambda item: item["mean_lateral_support_width_m"]) if eligible else None
    return {"sweep": sweep, "selected": selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multipliers", default="0,0.25,0.5,0.75,1,1.5,2,3,4")
    parser.add_argument("--lateral-factor-m", type=float, default=0.05)
    parser.add_argument("--heading-factor-rad", type=float, default=0.01)
    parser.add_argument("--curvature-factor-1pm", type=float, default=0.005)
    parser.add_argument("--scale-px", type=float, default=0.25)
    parser.add_argument("--target", type=float, default=0.90)
    args = parser.parse_args()
    read = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = {str(row["sample_id"]): row for row in read(args.manifest)}
    scores = {str(row["sample_id"]): row for row in read(args.scores)}
    result = {
        "protocol": "uncertainty-conditioned-support-calibration-v1",
        "manifest": str(args.manifest.resolve()),
        "scores": str(args.scores.resolve()),
        "target_joint_coverage": float(args.target),
        "base_factors": [args.lateral_factor_m, args.heading_factor_rad, args.curvature_factor_1pm],
        "scale_px": float(args.scale_px),
        **_evaluate(scores, manifest, [float(item) for item in args.multipliers.split(",")], (args.lateral_factor_m, args.heading_factor_rad, args.curvature_factor_1pm), args.scale_px, args.target),
        "holdout_must_not_be_used_for_selection": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
