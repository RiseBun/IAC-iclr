#!/usr/bin/env python3
"""Fit the longitudinal image residual on scene-disjoint proxy data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.continuous_motion import (
    history_anchored_residual_motion_profile,
    history_only_motion_profile,
    longitudinal_residual_features,
    trajectory_to_motion_profile,
)
from scripts.evaluate_continuous_motion_alignment import (
    _history_state,
    _history_times,
    _initial_speed,
    _reference,
    read_jsonl,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_scene_groups(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    fit_fraction: float,
    calibration_fraction: float,
) -> dict[str, list[str]]:
    """Return deterministic scene-disjoint fit/calibration/evaluation IDs."""
    if fit_fraction <= 0.0 or calibration_fraction <= 0.0:
        raise ValueError("fit and calibration fractions must be positive")
    if fit_fraction + calibration_fraction >= 1.0:
        raise ValueError("fit and calibration fractions must leave an evaluation split")
    scenes = sorted({str(row.get("scene_id") or row["sample_id"]) for row in rows})
    if len(scenes) < 3:
        raise ValueError("at least three scenes are required for isolated splits")
    ordered = sorted(
        scenes,
        key=lambda scene: hashlib.sha256(f"{seed}|{scene}".encode("utf-8")).hexdigest(),
    )
    fit_count = max(1, int(round(len(ordered) * fit_fraction)))
    calibration_count = max(1, int(round(len(ordered) * calibration_fraction)))
    if fit_count + calibration_count >= len(ordered):
        calibration_count = len(ordered) - fit_count - 1
    return {
        "fit": ordered[:fit_count],
        "calibration": ordered[fit_count:fit_count + calibration_count],
        "evaluation": ordered[fit_count + calibration_count:],
    }


def _prepared_record(
    row: dict[str, Any],
    score: dict[str, Any],
    reference_source: str,
) -> dict[str, Any]:
    if score.get("candidate_bank_used_by_decoder") is not False:
        raise ValueError(f"{row.get('sample_id')}: candidate-blind audit failed")
    times = [float(value) for value in row["future_times_s"]]
    history_state = _history_state(row)
    history_profile = history_only_motion_profile(
        history_state,
        times,
        history_times_s=_history_times(row, len(history_state)),
        model="constant_acceleration_yaw_rate",
    )
    reference = trajectory_to_motion_profile(
        _reference(row, reference_source),
        times,
        initial_speed_mps=_initial_speed(row),
    )
    decoder = score["decoder"]
    features = longitudinal_residual_features(decoder, times, history_profile)
    return {
        "sample_id": str(row["sample_id"]),
        "scene_id": str(row.get("scene_id") or row["sample_id"]),
        "times": times,
        "decoder": decoder,
        "history_profile": history_profile,
        "reference_profile": reference,
        "features": features,
    }


def _eligible_rows(record: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    support = record["decoder"]["speed_support"]
    return [
        (feature, history, reference)
        for feature, history, reference, gate in zip(
            record["features"]["rows"],
            record["history_profile"]["rows"],
            record["reference_profile"]["rows"],
            support,
        )
        if gate.get("status") == "usable"
    ]


def fit_longitudinal_gain(
    records: list[dict[str, Any]],
    *,
    minimum_gain: float = -2.0,
    maximum_gain: float = 2.0,
) -> dict[str, Any]:
    feature_values = []
    target_values = []
    weights = []
    for record in records:
        support = record["decoder"]["speed_support"]
        for feature, history, reference, gate in zip(
            record["features"]["rows"],
            record["history_profile"]["rows"],
            record["reference_profile"]["rows"],
            support,
        ):
            if gate.get("status") != "usable":
                continue
            feature_values.append(float(feature["innovation_mps"]))
            target_values.append(float(reference["speed_mps"]) - float(history["speed_mps"]))
            weights.append(max(float(gate.get("observability", 0.0)), 1e-3))
    x = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(target_values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(weight * x * x))
    if len(x) < 2 or denominator <= 1e-9:
        raise ValueError("insufficient longitudinal residual support for gain fitting")
    unconstrained = float(np.sum(weight * x * y) / denominator)
    gain = float(np.clip(unconstrained, minimum_gain, maximum_gain))
    prediction = gain * x
    return {
        "longitudinal_gain": gain,
        "unconstrained_gain": unconstrained,
        "gain_bounds": [float(minimum_gain), float(maximum_gain)],
        "usable_intervals": int(len(x)),
        "weighted_residual_mae_mps": float(np.average(np.abs(y - prediction), weights=weight)),
        "weighted_history_mae_mps": float(np.average(np.abs(y), weights=weight)),
    }


def conformal_speed_radius(
    records: list[dict[str, Any]],
    *,
    longitudinal_gain: float,
    nominal_coverage: float = 0.90,
) -> dict[str, Any]:
    errors = []
    for record in records:
        profile = history_anchored_residual_motion_profile(
            record["decoder"],
            record["times"],
            record["history_profile"],
            longitudinal_gain=longitudinal_gain,
            speed_interval_radius_mps=1.0,
        )
        for predicted, reference, gate in zip(
            profile["rows"],
            record["reference_profile"]["rows"],
            record["decoder"]["speed_support"],
        ):
            if gate.get("status") == "usable":
                errors.append(abs(float(predicted["speed_mps"]) - float(reference["speed_mps"])))
    if not errors:
        raise ValueError("no usable calibration intervals for conformal radius")
    ordered = np.sort(np.asarray(errors, dtype=np.float64))
    rank = min(int(math.ceil((len(ordered) + 1) * nominal_coverage)), len(ordered))
    radius = float(ordered[rank - 1])
    return {
        "speed_interval_radius_mps": max(radius, 1e-6),
        "nominal_coverage": float(nominal_coverage),
        "finite_sample_rank": int(rank),
        "usable_intervals": int(len(ordered)),
        "in_split_empirical_coverage": float(np.mean(ordered <= radius)),
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
    prepared = [
        _prepared_record(row, scores[str(row["sample_id"])], args.reference_source)
        for row in usable_manifest
    ]
    by_split = {
        name: [record for record in prepared if record["scene_id"] in set(scene_ids)]
        for name, scene_ids in groups.items()
    }
    fit = fit_longitudinal_gain(by_split["fit"])
    conformal = conformal_speed_radius(
        by_split["calibration"], longitudinal_gain=float(fit["longitudinal_gain"])
    )
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
        "protocol": "longitudinal-residual-calibration-v1",
        "reference_source": args.reference_source,
        "prediction_formula": "v_history(t) + beta * (delta_v_image(t) - delta_v_history(t))",
        "absolute_image_speed_used": False,
        "action_waypoint_used_for_calibration": False,
        "parameters": {
            "longitudinal_gain": fit["longitudinal_gain"],
            "speed_interval_radius_mps": conformal["speed_interval_radius_mps"],
        },
        "fit_diagnostics": fit,
        "conformal_diagnostics": conformal,
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
