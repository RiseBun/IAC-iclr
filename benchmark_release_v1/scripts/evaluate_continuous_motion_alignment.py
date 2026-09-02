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
    compare_pose_posteriors,
    apply_pose_interval_calibration,
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


def _shape_eligibility(
    interval_observability: list[dict[str, Any]],
    road_relative_support: list[dict[str, Any]],
    *,
    enable_fallback: bool = True,
    fallback_min_observability: float = 0.05,
) -> tuple[list[str], list[float], list[str]]:
    """Build shape-only eligibility with a conservative geometric fallback.

    The fallback is deliberately uncertain: it can rescue a direction/shape
    measurement when pixelwise FB is dominated by glare or dynamic foreground,
    but it must never make the interval speed-usable.
    """
    shape_status: list[str] = []
    shape_observability: list[float] = []
    reasons: list[str] = []
    for index, item in enumerate(interval_observability):
        direction_ok = bool(item.get("direction_observable"))
        curvature_status = str(item.get("curvature_status", "abstain"))
        support = road_relative_support[index] if index < len(road_relative_support) else {}
        support_obs = float(np.clip(support.get("observability", 0.0), 0.0, 1.0))
        fallback = (
            enable_fallback
            and
            not direction_ok
            and "forward_backward_inconsistent" in str(item.get("status", ""))
            and curvature_status == "usable"
            and support_obs >= float(fallback_min_observability)
        )
        if direction_ok and curvature_status == "usable":
            shape_status.append("usable")
            reasons.append("direct_flow_geometry")
        elif direction_ok and curvature_status == "uncertain":
            shape_status.append("uncertain")
            reasons.append("direct_flow_geometry_uncertain")
        elif fallback:
            shape_status.append("uncertain")
            reasons.append("robust_geometry_fallback_after_fb_gate")
        else:
            shape_status.append("abstain")
            reasons.append("no_shape_support")
        shape_observability.append(float(np.clip(
            max(item.get("effective_static_pixel_fraction", 0.0), support_obs if fallback else 0.0),
            0.0,
            1.0,
        )))
    return shape_status, shape_observability, reasons


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


def _distance_gain(control: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    control_metric = control.get("metrics", {}).get("forward_displacement_profile", {})
    actual_metric = actual.get("metrics", {}).get("forward_displacement_profile", {})
    control_mae = control_metric.get("mae")
    actual_mae = actual_metric.get("mae")
    lift = None if control_mae is None or actual_mae is None else float(control_mae - actual_mae)
    return {
        "actual_future_mae": actual_mae,
        "control_future_mae": control_mae,
        "absolute_lift": lift,
        "actual_beats_control": None if lift is None else lift > 0.0,
    }


def _relative_distance_stratum(record: dict[str, Any]) -> str:
    rows = list(record.get("reference_motion_profile", {}).get("rows") or [])
    if not rows:
        return "unknown"
    speeds = np.asarray([float(row.get("speed_mps", 0.0)) for row in rows], dtype=np.float64)
    lateral = np.asarray([abs(float(row.get("lateral_speed_mps", 0.0))) for row in rows], dtype=np.float64)
    yaw = np.asarray([abs(float(row.get("yaw_rate_radps", 0.0))) for row in rows], dtype=np.float64)
    speed_delta = float(speeds[-1] - speeds[0])
    if speed_delta <= -1.0:
        return "braking"
    if speed_delta >= 1.0:
        return "acceleration"
    if float(max(lateral.max(initial=0.0), yaw.max(initial=0.0))) >= 0.08:
        return "lateral_turn"
    return "straight_cruise"


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
            include_shape_uncertain=True,
        )
        reversed_behavior = compare_longitudinal_behavior(
            reversed_profile,
            record["reference_motion_profile"],
            record["image_motion_profile"],
            include_uncertain=include_uncertain,
        )
        actual_relative = record["distance_alignment_relative"]
        history_relative = compare_distance_profiles(
            record["history_motion_profile"],
            record["reference_motion_profile"],
            scale_mode="relative",
            include_uncertain=include_uncertain,
            include_shape_uncertain=True,
            allow_non_image_source=True,
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
                    include_shape_uncertain=True,
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
                    "relative_distance": compare_distance_profiles(
                        shuffled_profile,
                        record["reference_motion_profile"],
                        scale_mode="relative",
                        include_uncertain=include_uncertain,
                        include_shape_uncertain=True,
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
                "relative_distance": compare_distance_profiles(
                    reversed_profile,
                    record["reference_motion_profile"],
                    scale_mode="relative",
                    include_uncertain=include_uncertain,
                ),
            },
        }
        record["relative_distance_controls"] = {
            "status": "ok",
            "actual": actual_relative,
            "history": {
                "comparison": history_relative,
                "lift": _distance_gain(history_relative, actual_relative),
            },
            "matched_shuffle": {
                "comparison": matched_shuffle.get("relative_distance"),
                "lift": (
                    None
                    if matched_shuffle.get("relative_distance") is None
                    else _distance_gain(matched_shuffle["relative_distance"], actual_relative)
                ),
            },
            "time_reversed": {
                "comparison": record["specificity_controls"]["time_reversed"]["relative_distance"],
                "lift": _distance_gain(
                    record["specificity_controls"]["time_reversed"]["relative_distance"],
                    actual_relative,
                ),
            },
            "stratum": _relative_distance_stratum(record),
        }


def _pose_translation_mae(result: dict[str, Any]) -> float | None:
    value = result.get("metrics", {}).get("se2_pose", {}).get("translation_mae")
    return None if value is None else float(value)


def _pose_lift(control: dict[str, Any], actual: dict[str, Any]) -> float | None:
    control_value = _pose_translation_mae(control)
    actual_value = _pose_translation_mae(actual)
    return None if control_value is None or actual_value is None else float(control_value - actual_value)


def add_arc_pose_controls(records: list[dict[str, Any]], *, include_uncertain: bool) -> None:
    """Add history/order controls for the 8-frame arc-normalized pose lane."""
    eligible = [
        record for record in records
        if record.get("pose_alignment_arc_relative", {}).get("status") == "ok"
    ]
    for record in eligible:
        actual = record["pose_alignment_arc_relative"]
        history = compare_pose_profiles(
            record["history_motion_profile"],
            record["reference_motion_profile"],
            scale_mode="arc_relative",
            include_uncertain=include_uncertain,
            allow_non_image_source=True,
        )
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
        reversed_result = compare_pose_profiles(
            reversed_profile,
            record["reference_motion_profile"],
            scale_mode="arc_relative",
            include_uncertain=include_uncertain,
            include_shape_uncertain=True,
        )
        candidates = [
            other for other in eligible
            if other["sample_id"] != record["sample_id"]
            and len(other["future_times_s"]) == len(record["future_times_s"])
        ]
        matched: dict[str, Any] = {
            "status": "unavailable",
            "reason": "no donor within 0.5 m/s history-speed caliper",
        }
        if candidates:
            target_speed = float(record["history_motion_profile"]["history_anchor"]["speed_mps"])
            donor = min(
                candidates,
                key=lambda other: (
                    abs(float(other["history_motion_profile"]["history_anchor"]["speed_mps"]) - target_speed),
                    str(other["sample_id"]),
                ),
            )
            speed_gap = abs(float(donor["history_motion_profile"]["history_anchor"]["speed_mps"]) - target_speed)
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
                    shuffled_profile = _retime_profile(donor["image_motion_profile"], record["future_times_s"])
                shuffled_result = compare_pose_profiles(
                    shuffled_profile,
                    record["reference_motion_profile"],
                    scale_mode="arc_relative",
                    include_uncertain=include_uncertain,
                    include_shape_uncertain=True,
                )
                matched = {
                    "status": "ok",
                    "donor_sample_id": donor["sample_id"],
                    "history_speed_gap_mps": speed_gap,
                    "comparison": shuffled_result,
                    "lift": _pose_lift(shuffled_result, actual),
                }
        record["arc_pose_controls"] = {
            "status": "ok",
            "actual": actual,
            "history": {"comparison": history, "lift": _pose_lift(history, actual)},
            "matched_shuffle": matched,
            "time_reversed": {"comparison": reversed_result, "lift": _pose_lift(reversed_result, actual)},
        }


def add_relative_distance_controls(records: list[dict[str, Any]]) -> None:
    """Evaluate relative displacement independently of speed observability.

    The image decoder can have uncertain speed intervals while still producing
    usable pose samples. Relative displacement therefore gets its own control
    lane with explicit ``include_uncertain=True`` semantics; it never changes
    the strict speed/acceleration eligibility set.
    """
    eligible = [
        record for record in records
        if record.get("distance_alignment_relative_observable", {}).get("status") == "ok"
    ]
    for record in eligible:
        actual = record["distance_alignment_relative_observable"]
        history = compare_distance_profiles(
            record["history_motion_profile"],
            record["reference_motion_profile"],
            scale_mode="relative",
            include_uncertain=True,
            allow_non_image_source=True,
        )
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
        reversed_relative = compare_distance_profiles(
            reversed_profile,
            record["reference_motion_profile"],
            scale_mode="relative",
            include_uncertain=True,
        )
        candidates = [
            other for other in eligible
            if other["sample_id"] != record["sample_id"]
            and len(other["future_times_s"]) == len(record["future_times_s"])
        ]
        matched = {"status": "unavailable", "reason": "no usable relative-distance donor"}
        if candidates:
            target_speed = float(record["history_motion_profile"]["history_anchor"]["speed_mps"])
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
                    shuffled = reanchor_longitudinal_control_profile(
                        donor["image_motion_profile"],
                        record["history_motion_profile"],
                        record["future_times_s"],
                    )
                else:
                    shuffled = _retime_profile(donor["image_motion_profile"], record["future_times_s"])
                matched_relative = compare_distance_profiles(
                    shuffled,
                    record["reference_motion_profile"],
                    scale_mode="relative",
                    include_uncertain=True,
                )
                matched = {
                    "status": "ok",
                    "donor_sample_id": donor["sample_id"],
                    "history_speed_gap_mps": speed_gap,
                    "comparison": matched_relative,
                    "lift": _distance_gain(matched_relative, actual),
                }
        record["relative_distance_controls_observable"] = {
            "status": "ok",
            "actual": actual,
            "history": {"comparison": history, "lift": _distance_gain(history, actual)},
            "matched_shuffle": matched,
            "time_reversed": {
                "comparison": reversed_relative,
                "lift": _distance_gain(reversed_relative, actual),
            },
            "stratum": _relative_distance_stratum(record),
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


def _incremental_evidence(
    valid: list[dict[str, Any]],
    *,
    relative_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
    relative_controls = {
        "history": [],
        "matched_shuffle": [],
        "time_reversed": [],
    }
    for record in valid:
        controls = record.get("relative_distance_controls", {})
        for name in relative_controls:
            lift = controls.get(name, {}).get("lift") or {}
            value = lift.get("absolute_lift")
            if value is not None and np.isfinite(value):
                relative_controls[name].append(float(value))
    relative_evidence = {}
    for name, values in relative_controls.items():
        relative_evidence[name] = _mean_ci(values, rng) | {
            "positive_fraction": None if not values else float(np.mean(np.asarray(values) > 0.0)),
            "definition": "control_relative_progress_mae_minus_actual_relative_progress_mae",
        }
    relative_evidence["gate"] = {
        "history": bool(
            relative_evidence["history"]["confidence_interval_95"]
            and relative_evidence["history"]["confidence_interval_95"][0] > 0.0
        ),
        "matched_shuffle": bool(
            relative_evidence["matched_shuffle"]["confidence_interval_95"]
            and relative_evidence["matched_shuffle"]["confidence_interval_95"][0] > 0.0
        ),
        "time_reversed": bool(
            relative_evidence["time_reversed"]["confidence_interval_95"]
            and relative_evidence["time_reversed"]["confidence_interval_95"][0] > 0.0
        ),
    }
    relative_evidence["relative_progress_signal_resolved"] = all(relative_evidence["gate"].values())
    relative_evidence["criterion"] = "lower bound of paired sample bootstrap 95% CI must exceed zero"
    output["relative_distance_specificity"] = relative_evidence
    arc_pose_controls = {"history": [], "matched_shuffle": [], "time_reversed": []}
    for record in valid:
        controls = record.get("arc_pose_controls", {})
        for name in arc_pose_controls:
            value = controls.get(name, {}).get("lift")
            if value is not None and np.isfinite(value):
                arc_pose_controls[name].append(float(value))
    arc_pose_evidence = {
        name: _mean_ci(values, rng) | {
            "positive_fraction": None if not values else float(np.mean(np.asarray(values) > 0.0)),
            "definition": "control_arc_pose_translation_mae_minus_actual_arc_pose_translation_mae",
        }
        for name, values in arc_pose_controls.items()
    }
    arc_pose_evidence["gate"] = {
        name: bool(
            arc_pose_evidence[name]["confidence_interval_95"]
            and arc_pose_evidence[name]["confidence_interval_95"][0] > 0.0
        )
        for name in arc_pose_controls
    }
    arc_pose_evidence["arc_pose_signal_resolved"] = all(arc_pose_evidence["gate"].values())
    arc_pose_evidence["criterion"] = "lower bound of paired sample bootstrap 95% CI must exceed zero"
    output["arc_pose_alignment_specificity"] = arc_pose_evidence
    strata: dict[str, dict[str, Any]] = {}
    for record in valid:
        controls = record.get("relative_distance_controls", {})
        actual = controls.get("actual", {}).get("metrics", {}).get("forward_displacement_profile", {})
        value = actual.get("mae")
        if value is None or not np.isfinite(value):
            continue
        stratum = str(controls.get("stratum", "unknown"))
        bucket = strata.setdefault(stratum, {"actual_mae": [], "sample_ids": []})
        bucket["actual_mae"].append(float(value))
        bucket["sample_ids"].append(record["sample_id"])
    output["relative_distance_strata"] = {
        name: {
            "samples": len(bucket["actual_mae"]),
            "mean_actual_relative_mae": float(np.mean(bucket["actual_mae"])),
            "sample_ids": bucket["sample_ids"],
        }
        for name, bucket in sorted(strata.items())
    }
    if relative_records is not None:
        relaxed_controls = {
            "history": [],
            "matched_shuffle": [],
            "time_reversed": [],
        }
        for record in relative_records:
            controls = record.get("relative_distance_controls_observable", {})
            for name in relaxed_controls:
                value = (
                    controls.get(name, {})
                    .get("lift", {})
                    .get("absolute_lift")
                )
                if value is not None and np.isfinite(value):
                    relaxed_controls[name].append(float(value))
        relaxed_evidence = {}
        for name, values in relaxed_controls.items():
            relaxed_evidence[name] = _mean_ci(values, rng) | {
                "positive_fraction": None if not values else float(np.mean(np.asarray(values) > 0.0)),
                "definition": "control_relative_progress_mae_minus_actual_relative_progress_mae",
            }
        relaxed_evidence["gate"] = {
            name: bool(
                relaxed_evidence[name]["confidence_interval_95"]
                and relaxed_evidence[name]["confidence_interval_95"][0] > 0.0
            )
            for name in relaxed_controls
        }
        relaxed_evidence["relative_progress_signal_resolved"] = all(relaxed_evidence["gate"].values())
        relaxed_evidence["criterion"] = "lower bound of paired sample bootstrap 95% CI must exceed zero"
        output["relative_distance_specificity_observable"] = relaxed_evidence
        relaxed_strata: dict[str, dict[str, Any]] = {}
        for record in relative_records:
            controls = record.get("relative_distance_controls_observable", {})
            value = (
                controls.get("actual", {})
                .get("metrics", {})
                .get("forward_displacement_profile", {})
                .get("mae")
            )
            if value is None or not np.isfinite(value):
                continue
            name = str(controls.get("stratum", "unknown"))
            bucket = relaxed_strata.setdefault(name, {"actual_mae": [], "sample_ids": []})
            bucket["actual_mae"].append(float(value))
            bucket["sample_ids"].append(record["sample_id"])
        output["relative_distance_strata_observable"] = {
            name: {
                "samples": len(bucket["actual_mae"]),
                "mean_actual_relative_mae": float(np.mean(bucket["actual_mae"])),
                "sample_ids": bucket["sample_ids"],
            }
            for name, bucket in sorted(relaxed_strata.items())
        }
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


def aggregate(
    records: list[dict[str, Any]],
    reference_source: str,
    *,
    shape_fallback_enabled: bool = True,
) -> dict[str, Any]:
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
        ("relative", "distance_alignment_relative"),
        ("relative_observable", "distance_alignment_relative_observable"),
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
            "unit": (
                "m" if mode == "metric"
                else "normalized_terminal_forward_displacement"
                if mode in {"relative", "relative_observable"}
                else "normalized_max_abs_forward_displacement"
            ),
        }
    pose_summary: dict[str, Any] = {}
    for mode, record_key in (
        ("metric", "pose_alignment_metric"),
        ("scale_free", "pose_alignment_scale_free"),
        ("relative", "pose_alignment_relative"),
        ("arc_relative", "pose_alignment_arc_relative"),
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
            "unit": (
                "m/rad" if mode == "metric"
                else "normalized_terminal_forward/rad"
                if mode == "relative"
                else "normalized_arc_length/rad"
                if mode == "arc_relative"
                else "normalized_translation/rad"
            ),
        }
    pose_posterior_records = [
        record.get("pose_posterior", {})
        for record in records
        if record.get("pose_posterior", {}).get("status") == "ok"
    ]
    pose_posterior_summary: dict[str, Any] = {}
    for component in ("x_m", "y_m", "heading_rad"):
        component_rows = [record["metrics"].get(component, {}) for record in pose_posterior_records]
        def posterior_mean(key: str) -> float | None:
            values = [float(row[key]) for row in component_rows if row.get(key) is not None]
            return None if not values else float(np.mean(values))
        pose_posterior_summary[component] = {
            "empirical_coverage": posterior_mean("empirical_coverage"),
            "absolute_calibration_error": posterior_mean("absolute_calibration_error"),
            "mean_interval_width": posterior_mean("mean_interval_width"),
            "mean_interval_score": posterior_mean("mean_interval_score"),
            "mean_wis": posterior_mean("mean_wis"),
            "samples": len(component_rows),
        }
    joint_rows = [record["joint_pose"] for record in pose_posterior_records]
    joint_coverages = [float(row["empirical_coverage"]) for row in joint_rows if row.get("empirical_coverage") is not None]
    pose_posterior_summary["joint_pose"] = {
        "empirical_coverage": None if not joint_coverages else float(np.mean(joint_coverages)),
        "absolute_calibration_error": None if not joint_coverages else float(np.mean([
            abs(value - 0.90) for value in joint_coverages
        ])),
        "samples": len(joint_coverages),
    }
    behavior_rows = [
        record.get("longitudinal_behavior")
        for record in valid
        if record.get("longitudinal_behavior", {}).get("status") == "ok"
        and record.get("longitudinal_behavior", {}).get("delta_speed_mae_mps") is not None
    ]
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
        "shape_fallback_enabled": bool(shape_fallback_enabled),
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
        "primary_distance_alignment": "relative_observable",
        "primary_pose_alignment": "arc_relative",
        "se2_pose_posterior": {
            "samples": len(pose_posterior_records),
            "nominal_coverage": 0.90,
            "metrics": pose_posterior_summary,
        },
        "longitudinal_behavior": behavior,
        "longitudinal_model_protocols": sorted(longitudinal_models),
        "metric_families": [
            "continuous_alignment",
            "foresight_gain_over_history",
            "matched_shuffle_specificity",
            "time_order_specificity",
            "proper_speed_posterior",
            "proper_pose_posterior",
            "relative_distance_specificity",
            "coverage_risk",
        ],
        "incremental_evidence": _incremental_evidence(
            valid,
            relative_records=[
                record
                for record in records
                if record.get("relative_distance_controls_observable", {}).get("status") == "ok"
            ],
        ),
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
    parser.add_argument(
        "--disable-shape-fallback",
        action="store_true",
        help="disable the experimental shape-only FB fallback and use strict gate A",
    )
    parser.add_argument("--require-eight-frame-four-second", action="store_true")
    parser.add_argument("--longitudinal-calibration", type=Path)
    parser.add_argument("--pose-calibration", type=Path)
    parser.add_argument(
        "--pose-calibration-application-split",
        choices=("fit", "calibration", "evaluation"),
        default="evaluation",
    )
    parser.add_argument(
        "--calibration-application-split",
        choices=("fit", "calibration", "evaluation"),
        default="evaluation",
    )
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    scores = {str(row["sample_id"]): row for row in read_jsonl(args.scores)}
    calibration = None
    pose_calibration = None
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
    if args.pose_calibration is not None:
        pose_calibration = json.loads(args.pose_calibration.read_text(encoding="utf-8"))
        if pose_calibration.get("protocol") != "continuous-se2-pose-calibration-v1":
            raise ValueError("unsupported pose calibration protocol")
        if pose_calibration.get("reference_source") != args.reference_source:
            raise ValueError("pose calibration reference source does not match evaluation")
        pose_split_ids = set(
            pose_calibration.get("split", {}).get(
                f"{args.pose_calibration_application_split}_sample_ids"
            )
            or []
        )
        if pose_split_ids:
            manifest = [row for row in manifest if str(row["sample_id"]) in pose_split_ids]
            found_ids = {str(row["sample_id"]) for row in manifest}
            if found_ids != pose_split_ids:
                missing = sorted(pose_split_ids - found_ids)
                raise ValueError(f"pose calibration samples missing from manifest: {missing[:5]}")
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
        # Split eligibility before constructing the profile: speed posterior
        # quality (0.25/0.55) must not gate lateral/yaw/curvature.  The shape
        # gate is derived only from the image-side observability record.
        decoder_for_profile = copy.deepcopy(score["decoder"])
        interval_observability = list(score.get("observability_by_future_interval") or [])
        if len(interval_observability) == len(times):
            shape_status, shape_observability, shape_fallback_reasons = _shape_eligibility(
                interval_observability,
                list((score.get("road_relative_posterior") or {}).get("support") or []),
                enable_fallback=not args.disable_shape_fallback,
            )
            flow_status = [str(item.get("status", "abstain")) for item in interval_observability]
            decoder_for_profile["shape_status_by_interval"] = shape_status
            decoder_for_profile["shape_observability_by_interval"] = shape_observability
            decoder_for_profile["flow_status_by_interval"] = flow_status
            decoder_for_profile["shape_fallback_reason_by_interval"] = shape_fallback_reasons
        raw_imagined = image_motion_profile(
            decoder_for_profile, times, initial_speed_mps=initial_speed
        )
        if calibration is None:
            imagined = raw_imagined
        else:
            parameters = calibration.get("parameters") or {}
            imagined = history_anchored_residual_motion_profile(
                decoder_for_profile,
                times,
                history_profile,
                longitudinal_gain=float(parameters["longitudinal_gain"]),
                speed_interval_radius_mps=float(parameters["speed_interval_radius_mps"]),
            )
        comparison = compare_motion_profiles(
            imagined, reference, include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
        )
        distance_alignment_metric = compare_distance_profiles(
            imagined, reference, scale_mode="metric", include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
        )
        distance_alignment_scale_free = compare_distance_profiles(
            imagined, reference, scale_mode="scale_free", include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
        )
        pose_alignment_metric = compare_pose_profiles(
            imagined, reference, scale_mode="metric", include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
        )
        pose_alignment_scale_free = compare_pose_profiles(
            imagined, reference, scale_mode="scale_free", include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
        )
        distance_alignment_relative = compare_distance_profiles(
            imagined, reference, scale_mode="relative", include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
        )
        distance_alignment_relative_observable = compare_distance_profiles(
            imagined, reference, scale_mode="relative", include_uncertain=True,
            include_shape_uncertain=True,
        )
        pose_alignment_relative = compare_pose_profiles(
            imagined, reference, scale_mode="relative", include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
        )
        pose_alignment_arc_relative = compare_pose_profiles(
            imagined, reference, scale_mode="arc_relative", include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
        )
        pose_image_profile = (
            apply_pose_interval_calibration(raw_imagined, pose_calibration)
            if pose_calibration is not None
            else raw_imagined
        )
        pose_posterior = compare_pose_posteriors(
            pose_image_profile, reference, include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
        )
        raw_comparison = compare_motion_profiles(
            raw_imagined, reference, include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
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
            include_shape_uncertain=True,
        )
        history_cv_comparison = compare_history_baseline(
            history_cv_profile,
            reference,
            imagined,
            include_uncertain=args.include_uncertain,
            include_shape_uncertain=True,
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
            "distance_alignment_relative": distance_alignment_relative,
            "distance_alignment_relative_observable": distance_alignment_relative_observable,
            "pose_alignment_metric": pose_alignment_metric,
            "pose_alignment_scale_free": pose_alignment_scale_free,
            "pose_alignment_relative": pose_alignment_relative,
            "pose_alignment_arc_relative": pose_alignment_arc_relative,
            "pose_posterior": pose_posterior,
            "raw_image_comparison": raw_comparison,
            "longitudinal_behavior": longitudinal_behavior,
            "history_longitudinal_behavior": history_longitudinal_behavior,
            "history_baseline_comparison": history_comparison,
            "foresight_gain": foresight_gain(comparison, history_comparison),
        })
    add_specificity_controls(records, include_uncertain=args.include_uncertain)
    add_arc_pose_controls(records, include_uncertain=args.include_uncertain)
    add_relative_distance_controls(records)
    report = {
        "summary": aggregate(
            records,
            args.reference_source,
            shape_fallback_enabled=not args.disable_shape_fallback,
        ),
        "calibration_application_split": args.calibration_application_split,
        "longitudinal_calibration": calibration,
        "pose_interval_calibration": pose_calibration,
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
