#!/usr/bin/env python3
"""Evaluate image-derived ego motion against independently held-out waypoints."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.continuous_motion import (
    MOTION_FIELDS,
    compare_future_control,
    compare_history_baseline,
    compare_longitudinal_behavior,
    compare_motion_profiles,
    compare_distance_profiles,
    compare_pose_profiles,
    foresight_gain,
    history_anchored_residual_motion_profile,
    history_only_motion_profile,
    image_motion_profile,
    reanchor_longitudinal_control_profile,
    trajectory_to_motion_profile,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _initial_speed(row: dict[str, Any]) -> float | None:
    metadata = row.get("metadata") or {}
    history = row.get("history_ego_state") or metadata.get("history_ego_state") or []
    if history and len(history[-1]) >= 4:
        return float(history[-1][3])
    return None


def _reference(row: dict[str, Any], source: str) -> list[list[float]]:
    metadata = row.get("metadata") or {}
    if source == "realized":
        value = row.get("realized_future_ego_state") or metadata.get("realized_future_ego_state")
        if value is None:
            raise ValueError(f"{row.get('sample_id')}: missing realized_future_ego_state")
        return [list(state[:3]) for state in value]
    if source == "action":
        value = row.get("action_trajectory") or metadata.get("action_trajectory")
        if value is None:
            raise ValueError(f"{row.get('sample_id')}: missing action_trajectory")
        return [list(state[:3]) for state in value]
    gt_id = str(row.get("gt_candidate_id"))
    for candidate in row.get("candidates") or []:
        if str(candidate.get("candidate_id")) == gt_id:
            return candidate["trajectory"]
    raise ValueError(f"{row.get('sample_id')}: logged GT candidate is missing")


def _level1_input_audit(row: dict[str, Any], reference_source: str) -> dict[str, Any]:
    if reference_source != "action":
        return {
            "ready": False,
            "issues": ["reference_is_not_native_wam_action_head"],
        }
    metadata = row.get("metadata") or {}
    future_source = row.get("future_images_source") or metadata.get("future_images_source")
    action_source = str(row.get("action_trajectory_source") or metadata.get("action_trajectory_source") or "").lower()
    wam_model_id = row.get("wam_model_id") or metadata.get("wam_model_id")
    issues = []
    if future_source != "wam_generated":
        issues.append("future_images_source_is_not_wam_generated")
    if not action_source:
        issues.append("missing_action_trajectory_source")
    elif any(token in action_source for token in ("logged", "oracle", "proxy", "candidate")):
        issues.append("action_trajectory_is_not_native_action_head")
    if wam_model_id is None:
        issues.append("missing_wam_model_id")
    return {
        "ready": not issues,
        "issues": issues,
        "future_images_source": future_source,
        "action_trajectory_source": action_source or None,
        "wam_model_id": wam_model_id,
    }


def _history_state(row: dict[str, Any]) -> list[list[float]]:
    metadata = row.get("metadata") or {}
    history = row.get("history_ego_state") or metadata.get("history_ego_state")
    if not history:
        raise ValueError(f"{row.get('sample_id')}: history_ego_state is required for the Level-1 null")
    return history


def _history_times(row: dict[str, Any], history_count: int) -> list[float]:
    times = row.get("history_times_s")
    if not times or len(times) != history_count:
        raise ValueError(f"{row.get('sample_id')}: history_times_s must match history_ego_state")
    return [float(value) for value in times]


def _retime_profile(
    profile: dict[str, Any],
    target_times_s: list[float],
    *,
    reverse: bool = False,
) -> dict[str, Any]:
    result = copy.deepcopy(profile)
    rows = list(result["rows"])
    if reverse:
        rows.reverse()
    if len(rows) != len(target_times_s):
        raise ValueError("control profile and target must have matching intervals")
    previous = 0.0
    for row, time_s in zip(rows, target_times_s):
        row["time_s"] = float(time_s)
        row["dt_s"] = float(time_s - previous)
        previous = float(time_s)
    result["rows"] = rows
    return result


def _gain(control: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for field in MOTION_FIELDS:
        control_mae = control["metrics"][field]["mae"]
        actual_mae = actual["metrics"][field]["mae"]
        lift = None if control_mae is None or actual_mae is None else float(control_mae - actual_mae)
        metrics[field] = {
            "actual_future_mae": actual_mae,
            "control_future_mae": control_mae,
            "absolute_lift": lift,
            "actual_beats_control": None if lift is None else lift > 0.0,
        }
    return {
        "definition": "control_future_mae_minus_actual_future_mae",
        "positive_means_future_is_specific": True,
        "metrics": metrics,
    }


def add_specificity_controls(records: list[dict[str, Any]], *, include_uncertain: bool) -> None:
    eligible = [record for record in records if record.get("comparison", {}).get("status") == "ok"]
    for record in eligible:
        candidates = [
            other for other in eligible
            if other["sample_id"] != record["sample_id"]
            and len(other["future_times_s"]) == len(record["future_times_s"])
        ]
        target_speed = float(record["history_motion_profile"]["history_anchor"]["speed_mps"])
        if record["image_motion_profile"].get("source") in {
            "history_anchored_image_residual_decoder",
            "history_anchored_optimizer_residual_decoder",
        }:
            reversed_profile = reanchor_longitudinal_control_profile(
                record["image_motion_profile"],
                record["history_motion_profile"],
                record["future_times_s"],
                reverse=True,
            )
        else:
            reversed_profile = _retime_profile(
                record["image_motion_profile"], record["future_times_s"], reverse=True
            )
        reversed_result = compare_future_control(
            reversed_profile,
            record["reference_motion_profile"],
            record["image_motion_profile"],
            include_uncertain=include_uncertain,
        )
        reversed_behavior = compare_longitudinal_behavior(
            reversed_profile,
            record["reference_motion_profile"],
            record["image_motion_profile"],
            include_uncertain=include_uncertain,
        )
        matched_shuffle: dict[str, Any] = {
            "status": "unavailable",
            "reason": "no donor within 0.5 m/s history-speed caliper",
        }
        if candidates:
            donor = min(
                candidates,
                key=lambda other: (
                    abs(float(other["history_motion_profile"]["history_anchor"]["speed_mps"]) - target_speed),
                    str(other["sample_id"]),
                ),
            )
            speed_gap = abs(
                float(donor["history_motion_profile"]["history_anchor"]["speed_mps"]) - target_speed
            )
            if speed_gap <= 0.5:
                if donor["image_motion_profile"].get("source") in {
                    "history_anchored_image_residual_decoder",
                    "history_anchored_optimizer_residual_decoder",
                }:
                    shuffled_profile = reanchor_longitudinal_control_profile(
                        donor["image_motion_profile"],
                        record["history_motion_profile"],
                        record["future_times_s"],
                    )
                else:
                    shuffled_profile = _retime_profile(
                        donor["image_motion_profile"], record["future_times_s"]
                    )
                shuffled = compare_future_control(
                    shuffled_profile,
                    record["reference_motion_profile"],
                    record["image_motion_profile"],
                    include_uncertain=include_uncertain,
                )
                matched_shuffle = {
                    "status": "ok",
                    "donor_sample_id": donor["sample_id"],
                    "history_speed_gap_mps": speed_gap,
                    "comparison": shuffled,
                    "lift": _gain(shuffled, record["comparison"]),
                    "longitudinal_behavior": compare_longitudinal_behavior(
                        shuffled_profile,
                        record["reference_motion_profile"],
                        record["image_motion_profile"],
                        include_uncertain=include_uncertain,
                    ),
                }
        record["specificity_controls"] = {
            "status": "ok",
            "matched_shuffle": matched_shuffle,
            "time_reversed": {
                "status": "ok",
                "comparison": reversed_result,
                "lift": _gain(reversed_result, record["comparison"]),
                "longitudinal_behavior": reversed_behavior,
            },
        }


def _mean_ci(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "confidence_interval_95": None, "samples": 0}
    array = np.asarray(values, dtype=np.float64)
    bootstrap = np.mean(rng.choice(array, size=(10000, len(array)), replace=True), axis=1)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "confidence_interval_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "samples": int(len(array)),
    }


def _incremental_evidence(valid: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    output: dict[str, Any] = {"foresight_gain_over_history": {}, "future_specificity": {}}
    for field in MOTION_FIELDS:
        history_values = [
            float(record["foresight_gain"]["metrics"][field]["absolute_gain"])
            for record in valid
            if record["foresight_gain"]["metrics"][field]["absolute_gain"] is not None
        ]
        output["foresight_gain_over_history"][field] = _mean_ci(history_values, rng) | {
            "positive_fraction": None if not history_values else float(np.mean(np.asarray(history_values) > 0.0)),
            "definition": "history_only_mae_minus_actual_future_mae",
        }
        controls = {}
        for control_name in ("matched_shuffle", "time_reversed"):
            values = [
                float(record["specificity_controls"][control_name]["lift"]["metrics"][field]["absolute_lift"])
                for record in valid
                if record.get("specificity_controls", {}).get("status") == "ok"
                and record["specificity_controls"].get(control_name, {}).get("status") == "ok"
                and record["specificity_controls"][control_name]["lift"]["metrics"][field]["absolute_lift"] is not None
            ]
            controls[control_name] = _mean_ci(values, rng) | {
                "positive_fraction": None if not values else float(np.mean(np.asarray(values) > 0.0)),
                "definition": "control_future_mae_minus_actual_future_mae",
            }
        output["future_specificity"][field] = controls
    donor_gaps = [
        float(record["specificity_controls"]["matched_shuffle"]["history_speed_gap_mps"])
        for record in valid
        if record.get("specificity_controls", {}).get("matched_shuffle", {}).get("status") == "ok"
    ]
    output["matched_shuffle_audit"] = {
        "matching_variables": ["history_speed_mps", "future_interval_count"],
        "history_speed_caliper_mps": 0.5,
        "action_or_future_reference_used_for_matching": False,
        "mean_history_speed_gap_mps": None if not donor_gaps else float(np.mean(donor_gaps)),
        "max_history_speed_gap_mps": None if not donor_gaps else float(np.max(donor_gaps)),
    }
    component_status = {}
    for field in MOTION_FIELDS:
        evidence = {
            "beats_strong_history_null": output["foresight_gain_over_history"][field]["confidence_interval_95"],
            "beats_matched_shuffle": output["future_specificity"][field]["matched_shuffle"]["confidence_interval_95"],
            "uses_temporal_order": output["future_specificity"][field]["time_reversed"]["confidence_interval_95"],
        }
        checks = {
            name: interval is not None and float(interval[0]) > 0.0
            for name, interval in evidence.items()
        }
        component_status[field] = {
            "incremental_signal_resolved": all(checks.values()),
            "checks": checks,
            "criterion": "lower bound of paired sample bootstrap 95% CI must exceed zero",
        }
    output["component_status"] = component_status
    behavior_pairs = {
        "history": [
            (
                float(record["history_longitudinal_behavior"]["delta_speed_mae_mps"])
                - float(record["longitudinal_behavior"]["delta_speed_mae_mps"]),
                float(record["longitudinal_behavior"]["change_direction_accuracy"])
                - float(record["history_longitudinal_behavior"]["change_direction_accuracy"]),
            )
            for record in valid
            if record.get("history_longitudinal_behavior", {}).get("status") == "ok"
            and record.get("longitudinal_behavior", {}).get("status") == "ok"
        ],
        "matched_shuffle": [
            (
                float(record["specificity_controls"]["matched_shuffle"]["longitudinal_behavior"]["delta_speed_mae_mps"])
                - float(record["longitudinal_behavior"]["delta_speed_mae_mps"]),
                float(record["longitudinal_behavior"]["change_direction_accuracy"])
                - float(record["specificity_controls"]["matched_shuffle"]["longitudinal_behavior"]["change_direction_accuracy"]),
            )
            for record in valid
            if record.get("specificity_controls", {}).get("matched_shuffle", {}).get("status") == "ok"
            and record["specificity_controls"]["matched_shuffle"].get("longitudinal_behavior", {}).get("status") == "ok"
        ],
        "time_reversed": [
            (
                float(record["specificity_controls"]["time_reversed"]["longitudinal_behavior"]["delta_speed_mae_mps"])
                - float(record["longitudinal_behavior"]["delta_speed_mae_mps"]),
                float(record["longitudinal_behavior"]["change_direction_accuracy"])
                - float(record["specificity_controls"]["time_reversed"]["longitudinal_behavior"]["change_direction_accuracy"]),
            )
            for record in valid
            if record.get("specificity_controls", {}).get("time_reversed", {}).get("status") == "ok"
            and record["specificity_controls"]["time_reversed"].get("longitudinal_behavior", {}).get("status") == "ok"
        ],
    }
    behavior_evidence = {}
    for name, pairs in behavior_pairs.items():
        behavior_evidence[name] = {
            "delta_speed_error_reduction": _mean_ci([pair[0] for pair in pairs], rng),
            "change_direction_accuracy_lift": _mean_ci([pair[1] for pair in pairs], rng),
        }
    output["longitudinal_behavior_incremental_evidence"] = behavior_evidence
    return output


def aggregate(records: list[dict[str, Any]], reference_source: str) -> dict[str, Any]:
    valid = [record for record in records if record.get("comparison", {}).get("status") == "ok"]
    fields = MOTION_FIELDS
    metrics: dict[str, Any] = {}
    for field in fields:
        values = [record["comparison"]["metrics"][field]["mae"] for record in valid]
        values = [float(value) for value in values if value is not None]
        within = [record["comparison"]["metrics"][field]["within_tolerance"] for record in valid]
        within = [float(value) for value in within if value is not None]
        metrics[field] = {
            "sample_mean_mae": None if not values else float(np.mean(values)),
            "sample_median_mae": None if not values else float(np.median(values)),
            "mean_within_tolerance": None if not within else float(np.mean(within)),
            "samples": len(values),
        }
    protocol_records = [record for record in records if record.get("future_times_s")]
    frame_counts = sorted({len(record["future_times_s"]) for record in records})
    horizons = [float(record["future_times_s"][-1]) for record in protocol_records]
    intervals = [
        float(value)
        for record in protocol_records
        for value in np.diff(np.concatenate([[0.0], np.asarray(record["future_times_s"], dtype=np.float64)]))
    ]
    coverage_risk: dict[str, list[dict[str, Any]]] = {}
    for field in ("speed_mps", "acceleration_mps2"):
        observations = []
        for record in records:
            for interval in record.get("comparison", {}).get("per_interval", []):
                error = interval.get("absolute_errors", {}).get(field)
                if error is not None and np.isfinite(error):
                    observations.append((float(interval["observability"]), float(error)))
        observations.sort(key=lambda item: item[0], reverse=True)
        curve = []
        for target in (0.25, 0.50, 0.75, 1.00):
            count = int(np.ceil(target * len(observations)))
            selected = observations[:count]
            curve.append({
                "target_coverage": target,
                "actual_coverage": None if not observations else float(count / len(observations)),
                "minimum_observability": None if not selected else float(selected[-1][0]),
                "mae": None if not selected else float(np.mean([item[1] for item in selected])),
                "intervals": len(selected),
            })
        coverage_risk[field] = curve
    posterior_rows = [
        (
            interval["speed_interval_contains_action"],
            interval["speed_interval_width_mps"],
            interval["speed_interval_score_90"],
            interval["speed_wis_90"],
            interval["observability"],
        )
        for record in valid
        for interval in record["comparison"]["per_interval"]
        if interval["evaluable"] and interval["speed_interval_contains_action"] is not None
    ]
    if posterior_rows:
        posterior = np.asarray(posterior_rows, dtype=np.float64)
        weights = np.maximum(posterior[:, 4], 1e-3)
        empirical = float(np.average(posterior[:, 0], weights=weights))
        speed_posterior = {
            "nominal_coverage": 0.90,
            "empirical_coverage": empirical,
            "absolute_calibration_error": abs(empirical - 0.90),
            "mean_interval_width_mps": float(np.average(posterior[:, 1], weights=weights)),
            "mean_interval_score_90": float(np.average(posterior[:, 2], weights=weights)),
            "mean_wis_90": float(np.average(posterior[:, 3], weights=weights)),
            "intervals": int(len(posterior)),
        }
    else:
        speed_posterior = {
            "nominal_coverage": 0.90,
            "empirical_coverage": None,
            "absolute_calibration_error": None,
            "mean_interval_width_mps": None,
            "mean_interval_score_90": None,
            "mean_wis_90": None,
            "intervals": 0,
        }
    target_protocol_ready = bool(
        frame_counts == [8] and horizons and all(abs(value - 4.0) <= 0.05 for value in horizons)
    )
    input_audit_ready = bool(records) and all(
        record.get("level1_input_audit", {}).get("ready") is True for record in records
    )
    raw_metrics: dict[str, Any] = {}
    for field in fields:
        values = [
            record.get("raw_image_comparison", {}).get("metrics", {}).get(field, {}).get("mae")
            for record in valid
        ]
        values = [float(value) for value in values if value is not None]
        raw_metrics[field] = {
            "sample_mean_mae": None if not values else float(np.mean(values)),
            "samples": len(values),
        }
    distance_summary: dict[str, Any] = {}
    for mode, record_key in (
        ("metric", "distance_alignment_metric"),
        ("scale_free", "distance_alignment_scale_free"),
    ):
        distance_records = [
            record.get(record_key, {}) for record in records
            if record.get(record_key, {}).get("status") == "ok"
        ]
        distance_metrics = [
            record.get("metrics", {}).get("forward_displacement_profile", {})
            for record in distance_records
        ]
        values = [float(item["mae"]) for item in distance_metrics if item.get("mae") is not None]
        increments = [float(item["increment_mae"]) for item in distance_metrics if item.get("increment_mae") is not None]
        endpoint = [float(item["endpoint_abs_error"]) for item in distance_metrics if item.get("endpoint_abs_error") is not None]
        cosines = [float(item["curve_cosine"]) for item in distance_metrics if item.get("curve_cosine") is not None]
        distance_summary[mode] = {
            "samples": len(distance_records),
            "sample_mean_profile_mae": None if not values else float(np.mean(values)),
            "sample_mean_increment_mae": None if not increments else float(np.mean(increments)),
            "sample_mean_endpoint_abs_error": None if not endpoint else float(np.mean(endpoint)),
            "sample_mean_curve_cosine": None if not cosines else float(np.mean(cosines)),
            "unit": "m" if mode == "metric" else "normalized_max_abs_forward_displacement",
        }
    pose_summary: dict[str, Any] = {}
    for mode, record_key in (
        ("metric", "pose_alignment_metric"),
        ("scale_free", "pose_alignment_scale_free"),
    ):
        pose_records = [
            record.get(record_key, {}) for record in records
            if record.get(record_key, {}).get("status") == "ok"
        ]
        pose_metrics = [record.get("metrics", {}).get("se2_pose", {}) for record in pose_records]
        def mean_value(key: str) -> float | None:
            values = [float(item[key]) for item in pose_metrics if item.get(key) is not None]
            return None if not values else float(np.mean(values))
        pose_summary[mode] = {
            "samples": len(pose_records),
            "sample_mean_translation_mae": mean_value("translation_mae"),
            "sample_mean_forward_mae": mean_value("forward_mae"),
            "sample_mean_lateral_mae": mean_value("lateral_mae"),
            "sample_mean_heading_mae_rad": mean_value("heading_mae_rad"),
            "sample_mean_endpoint_translation_error": mean_value("endpoint_translation_error"),
            "sample_mean_endpoint_heading_error_rad": mean_value("endpoint_heading_error_rad"),
            "sample_mean_path_cosine": mean_value("path_cosine"),
            "unit": "m/rad" if mode == "metric" else "normalized_translation/rad",
        }
    behavior_rows = [record.get("longitudinal_behavior") for record in valid if record.get("longitudinal_behavior")]
    behavior = {
        "delta_speed_mae_mps": None if not behavior_rows else float(np.mean([row["delta_speed_mae_mps"] for row in behavior_rows])),
        "change_direction_accuracy": None if not behavior_rows else float(np.mean([row["change_direction_accuracy"] for row in behavior_rows])),
        "significant_change_direction_accuracy": None if not behavior_rows else float(np.mean([
            row["significant_change_direction_accuracy"] for row in behavior_rows
            if row["significant_change_direction_accuracy"] is not None
        ])),
        "samples": len(behavior_rows),
        "change_deadband_mps": None if not behavior_rows else float(behavior_rows[0]["change_deadband_mps"]),
    }
    behavior["capability_gate"] = {
        "delta_speed_mae_at_most_0p5_mps": (
            behavior["delta_speed_mae_mps"] is not None and behavior["delta_speed_mae_mps"] <= 0.5
        ),
        "change_direction_accuracy_at_least_0p70": (
            behavior["change_direction_accuracy"] is not None and behavior["change_direction_accuracy"] >= 0.70
        ),
        "mean_interval_coverage_at_least_0p50": (
            bool(valid) and float(np.mean([record["comparison"]["coverage"] for record in valid])) >= 0.50
        ),
    }
    behavior["capability_sufficient_for_level1_probe"] = all(behavior["capability_gate"].values())
    behavior["capability_gate_status"] = "provisional_proxy_thresholds_not_causal_evidence"
    longitudinal_models = {
        str(record.get("image_motion_profile", {}).get("longitudinal_model", {}).get("protocol"))
        for record in valid
        if record.get("image_motion_profile", {}).get("longitudinal_model")
    }
    return {
        "protocol": (
            "continuous-foresight-action-level1-optimizer-residual-v4"
            if "optimizer-internal-longitudinal-residual-v1" in longitudinal_models
            else "continuous-foresight-action-level1-longitudinal-v3"
            if longitudinal_models
            else "continuous-foresight-action-level1-v2"
        ),
        "evidence_scope": (
            "image_measurement_validation_only"
            if reference_source != "action"
            else "single_branch_image_action_alignment"
        ),
        "future_action_alignment_eligible": reference_source == "action",
        "formal_level1_evidence_eligible": reference_source == "action" and target_protocol_ready and input_audit_ready,
        "level1_input_audit": {
            "ready": input_audit_ready,
            "samples_ready": sum(record.get("level1_input_audit", {}).get("ready") is True for record in records),
            "samples_total": len(records),
        },
        "causal_claim_eligible": False,
        "samples_total": len(records),
        "samples_evaluable": len(valid),
        "samples_missing_decoder_score": sum(record.get("status") == "missing_decoder_score" for record in records),
        "mean_interval_coverage": None if not valid else float(np.mean([record["comparison"]["coverage"] for record in valid])),
        "metrics": metrics,
        "raw_absolute_image_metrics": raw_metrics,
        "forward_distance_alignment": distance_summary,
        "se2_pose_alignment": pose_summary,
        "longitudinal_behavior": behavior,
        "longitudinal_model_protocols": sorted(longitudinal_models),
        "metric_families": [
            "continuous_alignment",
            "foresight_gain_over_history",
            "matched_shuffle_specificity",
            "time_order_specificity",
            "proper_speed_posterior",
            "coverage_risk",
        ],
        "incremental_evidence": _incremental_evidence(valid),
        "coverage_risk_curve": coverage_risk,
        "speed_posterior": speed_posterior,
        "speed_posterior_coverage": speed_posterior["empirical_coverage"],
        "observed_protocol": {
            "future_frame_counts": frame_counts,
            "future_horizon_s_min": None if not horizons else float(min(horizons)),
            "future_horizon_s_max": None if not horizons else float(max(horizons)),
            "median_interval_s": None if not intervals else float(np.median(intervals)),
            "meets_target_8_frames_4_seconds": target_protocol_ready,
        },
        "failure_boundary": (
            "A single branch cannot establish counterfactual causality; paired controlled interventions are required."
            if reference_source == "action"
            else "No WAM action-head trajectory was used; this report cannot establish future-to-action causality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", choices=("logged_gt", "realized", "action"), default="logged_gt")
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--require-eight-frame-four-second", action="store_true")
    parser.add_argument("--longitudinal-calibration", type=Path)
    parser.add_argument(
        "--calibration-application-split",
        choices=("fit", "calibration", "evaluation"),
        default="evaluation",
    )
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    scores = {str(row["sample_id"]): row for row in read_jsonl(args.scores)}
    calibration = None
    evaluation_ids = None
    if args.longitudinal_calibration is not None:
        calibration = json.loads(args.longitudinal_calibration.read_text(encoding="utf-8"))
        if calibration.get("protocol") != "longitudinal-residual-calibration-v1":
            raise ValueError("unsupported longitudinal calibration protocol")
        if calibration.get("reference_source") != args.reference_source:
            raise ValueError("calibration reference source does not match evaluation")
        split_key = f"{args.calibration_application_split}_sample_ids"
        evaluation_ids = set(calibration.get("split", {}).get(split_key) or [])
        if not evaluation_ids:
            raise ValueError(f"calibration artifact must provide {split_key}")
        manifest = [row for row in manifest if str(row["sample_id"]) in evaluation_ids]
        found_ids = {str(row["sample_id"]) for row in manifest}
        if found_ids != evaluation_ids:
            missing = sorted(evaluation_ids - found_ids)
            raise ValueError(f"evaluation samples missing from manifest: {missing[:5]}")
    records = []
    for row in manifest:
        sample_id = str(row["sample_id"])
        if sample_id not in scores:
            records.append({
                "sample_id": sample_id,
                "scene_id": row.get("scene_id"),
                "future_times_s": list(row["future_times_s"]),
                "reference_source": args.reference_source,
                "level1_input_audit": _level1_input_audit(row, args.reference_source),
                "status": "missing_decoder_score",
            })
            continue
        times = list(row["future_times_s"])
        if args.require_eight_frame_four_second and (len(times) != 8 or abs(float(times[-1]) - 4.0) > 0.05):
            raise ValueError(f"{sample_id}: expected 8 future frames ending at 4.0 seconds")
        score = scores[sample_id]
        if score.get("candidate_bank_used_by_decoder") is not False:
            raise ValueError(f"{sample_id}: candidate-blind audit failed")
        history_state = _history_state(row)
        initial_speed = _initial_speed(row)
        reference = trajectory_to_motion_profile(
            _reference(row, args.reference_source), times, initial_speed_mps=initial_speed
        )
        history_times = _history_times(row, len(history_state))
        history_cv_profile = history_only_motion_profile(history_state, times)
        history_profile = history_only_motion_profile(
            history_state,
            times,
            history_times_s=history_times,
            model="constant_acceleration_yaw_rate",
        )
        raw_imagined = image_motion_profile(score["decoder"], times, initial_speed_mps=initial_speed)
        if calibration is None:
            imagined = raw_imagined
        else:
            parameters = calibration.get("parameters") or {}
            imagined = history_anchored_residual_motion_profile(
                score["decoder"],
                times,
                history_profile,
                longitudinal_gain=float(parameters["longitudinal_gain"]),
                speed_interval_radius_mps=float(parameters["speed_interval_radius_mps"]),
            )
        comparison = compare_motion_profiles(imagined, reference, include_uncertain=args.include_uncertain)
        distance_alignment_metric = compare_distance_profiles(
            imagined, reference, scale_mode="metric", include_uncertain=args.include_uncertain
        )
        distance_alignment_scale_free = compare_distance_profiles(
            imagined, reference, scale_mode="scale_free", include_uncertain=args.include_uncertain
        )
        pose_alignment_metric = compare_pose_profiles(
            imagined, reference, scale_mode="metric", include_uncertain=args.include_uncertain
        )
        pose_alignment_scale_free = compare_pose_profiles(
            imagined, reference, scale_mode="scale_free", include_uncertain=args.include_uncertain
        )
        raw_comparison = compare_motion_profiles(
            raw_imagined, reference, include_uncertain=args.include_uncertain
        )
        longitudinal_behavior = compare_longitudinal_behavior(
            imagined,
            reference,
            imagined,
            include_uncertain=args.include_uncertain,
        )
        history_longitudinal_behavior = compare_longitudinal_behavior(
            history_profile,
            reference,
            imagined,
            include_uncertain=args.include_uncertain,
        )
        history_comparison = compare_history_baseline(
            history_profile,
            reference,
            imagined,
            include_uncertain=args.include_uncertain,
        )
        history_cv_comparison = compare_history_baseline(
            history_cv_profile,
            reference,
            imagined,
            include_uncertain=args.include_uncertain,
        )
        records.append({
            "sample_id": sample_id,
            "scene_id": row.get("scene_id"),
            "future_times_s": times,
            "reference_source": args.reference_source,
            "level1_input_audit": _level1_input_audit(row, args.reference_source),
            "image_motion_profile": imagined,
            "raw_image_motion_profile": raw_imagined,
            "history_motion_profile": history_profile,
            "history_nulls": {
                "primary": "constant_acceleration_yaw_rate",
                "constant_speed_yaw_rate": {
                    "profile": history_cv_profile,
                    "comparison": history_cv_comparison,
                },
                "constant_acceleration_yaw_rate": {
                    "profile": history_profile,
                    "comparison": history_comparison,
                },
            },
            "reference_motion_profile": reference,
            "comparison": comparison,
            "distance_alignment_metric": distance_alignment_metric,
            "distance_alignment_scale_free": distance_alignment_scale_free,
            "pose_alignment_metric": pose_alignment_metric,
            "pose_alignment_scale_free": pose_alignment_scale_free,
            "raw_image_comparison": raw_comparison,
            "longitudinal_behavior": longitudinal_behavior,
            "history_longitudinal_behavior": history_longitudinal_behavior,
            "history_baseline_comparison": history_comparison,
            "foresight_gain": foresight_gain(comparison, history_comparison),
        })
    add_specificity_controls(records, include_uncertain=args.include_uncertain)
    report = {
        "summary": aggregate(records, args.reference_source),
        "calibration_application_split": args.calibration_application_split,
        "longitudinal_calibration": calibration,
        "records": records,
    }
    if calibration is not None and args.calibration_application_split != "evaluation":
        report["summary"]["formal_level1_evidence_eligible"] = False
        report["summary"]["calibration_split_audit_only"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
