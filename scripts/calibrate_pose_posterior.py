#!/usr/bin/env python3
"""Fit scene-disjoint conformal inflation for image-side SE(2) intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.continuous_motion import (
    image_motion_profile,
    trajectory_to_motion_profile,
)
from scripts.evaluate_continuous_motion_alignment import (
    _history_state,
    _initial_speed,
    _reference,
    read_jsonl,
)
from scripts.calibrate_longitudinal_residual import split_scene_groups


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_record(row: dict[str, Any], score: dict[str, Any], reference_source: str) -> dict[str, Any]:
    if score.get("candidate_bank_used_by_decoder") is not False:
        raise ValueError(f"{row.get('sample_id')}: candidate-blind audit failed")
    times = [float(value) for value in row["future_times_s"]]
    _history_state(row)  # fail closed on missing historical contract
    reference = trajectory_to_motion_profile(
        _reference(row, reference_source), times, initial_speed_mps=_initial_speed(row)
    )
    return {
        "sample_id": str(row["sample_id"]),
        "scene_id": str(row.get("scene_id") or row["sample_id"]),
        "times": times,
        "decoder": score["decoder"],
        "image_profile": image_motion_profile(score["decoder"], times, initial_speed_mps=_initial_speed(row)),
        "reference_profile": reference,
    }


def _conformal_radius(values: list[float], nominal_coverage: float) -> tuple[float, int]:
    if not values:
        raise ValueError("no usable pose intervals for calibration")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = min(int(math.ceil((len(ordered) + 1) * nominal_coverage)), len(ordered))
    return float(max(ordered[rank - 1], 0.0)), rank


def fit_pose_interval_calibration(
    records: list[dict[str, Any]],
    *,
    nominal_coverage: float = 0.90,
) -> dict[str, Any]:
    """Fit per-component additive radii; joint target uses Bonferroni coverage."""
    if not 0.0 < nominal_coverage < 1.0 or not np.isfinite(nominal_coverage):
        raise ValueError("nominal_coverage must be between zero and one")
    components = {
        "x_m": ("progress_m", False),
        "y_m": ("lateral_offset_m", False),
        "heading_rad": ("heading_rad", True),
    }
    component_nominal = 1.0 - (1.0 - float(nominal_coverage)) / len(components)
    nonconformity = {component: [] for component in components}
    for record in records:
        for predicted, reference, gate in zip(
            record["image_profile"]["rows"],
            record["reference_profile"]["rows"],
            record["decoder"].get("speed_support") or [],
        ):
            if gate.get("status") != "usable":
                continue
            intervals = predicted.get("pose_intervals") or {}
            for component, (target_key, is_angle) in components.items():
                interval = intervals.get(component)
                target = reference.get(target_key)
                if interval is None or target is None:
                    continue
                target_value = float(target)
                if is_angle:
                    median = float(interval["q50"])
                    target_value = float(median + np.arctan2(np.sin(target_value - median), np.cos(target_value - median)))
                lower = float(interval["q05"])
                upper = float(interval["q95"])
                nonconformity[component].append(max(lower - target_value, target_value - upper, 0.0))
    radii = {}
    diagnostics = {}
    for component, values in nonconformity.items():
        radius, rank = _conformal_radius(values, component_nominal)
        radii[component] = radius
        diagnostics[component] = {
            "nominal_component_coverage": component_nominal,
            "finite_sample_rank": rank,
            "usable_intervals": len(values),
            "in_split_empirical_coverage": float(np.mean(np.asarray(values) <= radius)),
            "uncalibrated_nonconformity_p90": float(np.quantile(np.asarray(values), nominal_coverage)),
        }
    return {
        "protocol": "continuous-se2-pose-calibration-v1",
        "nominal_joint_coverage": float(nominal_coverage),
        "component_nominal_coverage": component_nominal,
        "parameters": {"conformal_radius": radii},
        "diagnostics": diagnostics,
        "action_waypoint_used_for_interval_fit": False,
        "independent_reference_used_for_interval_fit": True,
        "future_images_used_for_interval_fit": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", choices=("logged_gt", "realized"), default="logged_gt")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fit-fraction", type=float, default=0.40)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    scores = {str(row["sample_id"]): row for row in read_jsonl(args.scores)}
    usable_manifest = [row for row in manifest if str(row["sample_id"]) in scores]
    groups = split_scene_groups(
        usable_manifest,
        seed=args.seed,
        fit_fraction=args.fit_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    prepared = [_prepared_record(row, scores[str(row["sample_id"])], args.reference_source) for row in usable_manifest]
    by_split = {
        name: [record for record in prepared if record["scene_id"] in set(scene_ids)]
        for name, scene_ids in groups.items()
    }
    calibration = fit_pose_interval_calibration(by_split["calibration"])
    split = {
        "method": "sha256_ordered_scene_split_v1",
        "seed": args.seed,
        "fit_fraction": args.fit_fraction,
        "calibration_fraction": args.calibration_fraction,
        "fit_scene_ids": groups["fit"],
        "calibration_scene_ids": groups["calibration"],
        "evaluation_scene_ids": groups["evaluation"],
    }
    for name in ("fit", "calibration", "evaluation"):
        split[f"{name}_sample_ids"] = [record["sample_id"] for record in by_split[name]]
    artifact = {
        **calibration,
        "reference_source": args.reference_source,
        "split": split,
        "inputs": {
            "manifest_sha256": _file_sha256(args.manifest),
            "scores_sha256": _file_sha256(args.scores),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
