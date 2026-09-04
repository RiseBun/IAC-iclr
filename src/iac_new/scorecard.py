"""Capability-stratified IAC scorecard.

Optional capabilities (CCFC, FAU and FCS) are reported as ``unavailable`` when a
model does not expose the required interface.  ``missing`` is reserved for a
claimed capability whose evidence is incomplete; ``ineligible`` is reserved for
hard protocol violations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CAPABILITIES = (
    "native_action_conditioned",
    "externally_controlled_video",
    "video_only",
    "action_only",
)

CELLS = ("l1", "a2f", "f2a", "cfac", "ccfc", "fau_f", "fau_a", "fau", "fcs", "coverage")

CLAIMED = {
    "native_action_conditioned": ("l1", "a2f", "f2a", "cfac"),
    "externally_controlled_video": ("a2f",),
    "video_only": (),
    "action_only": (),
}

STATUSES = ("pass", "fail", "pilot", "unavailable", "ineligible", "missing")
OPTIONAL_CELLS = frozenset({"ccfc", "fau_f", "fau_a", "fau", "fcs", "coverage"})


def claimed_cells(capability: str) -> tuple[str, ...]:
    if capability not in CLAIMED:
        raise ValueError(f"unknown capability: {capability}")
    return CLAIMED[capability]


def empty_cell(status: str, *, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status}")
    row = {"status": status}
    if reason:
        row["reason"] = reason
    row.update(extra)
    return row


def validate_submission_row(
    row: dict[str, Any],
    *,
    public_ids: set[str],
    expected_future_count: int | None = None,
) -> list[str]:
    issues: list[str] = []
    sample_id = str(row.get("sample_id") or row.get("source_key") or "")
    if not sample_id:
        issues.append("missing_sample_id")
    elif sample_id not in public_ids:
        issues.append("sample_id_not_in_public_split")
    capability = str(row.get("capability") or "")
    if capability not in CLAIMED:
        issues.append("missing_or_unknown_capability")
        return issues
    if not str(row.get("wam_model_id") or ""):
        issues.append("missing_wam_model_id")
    claimed = claimed_cells(capability)
    images = row.get("future_images") or row.get("generated_future_images") or []
    times = np.asarray(row.get("future_times_s"), dtype=np.float64) if row.get("future_times_s") is not None else np.asarray([])
    needs_video = any(cell in claimed for cell in ("l1", "a2f", "f2a", "ccfc"))
    if needs_video:
        count = len(images) if isinstance(images, list) else 0
        if expected_future_count is not None:
            valid_count = count == expected_future_count
        else:
            valid_count = count >= 4
        if row.get("future_images_source") != "wam_generated":
            issues.append("future_images_source_is_not_wam_generated")
        if not valid_count:
            expected = str(expected_future_count) if expected_future_count is not None else "at_least_4"
            issues.append(f"future_images_must_have_{expected}_paths")
        if times.shape != (count,) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            issues.append("future_times_s_invalid")
        elif float(times[0]) <= 0.0 or not (3.95 <= float(times[-1]) <= 4.05):
            issues.append("future_times_s_does_not_cover_0p5_to_4p0_seconds")
    action = row.get("action_trajectory")
    if action is None:
        action = (row.get("action_condition") or {}).get("trajectory")
    action_array = np.asarray(action, dtype=np.float64) if action is not None else np.zeros((0, 3))
    if "l1" in claimed or "ccfc" in claimed or "f2a" in claimed:
        action_count = len(images) if isinstance(images, list) else expected_future_count or 0
        if action_array.shape != (action_count, 3) or not np.all(np.isfinite(action_array)):
            issues.append("native_action_trajectory_invalid")
        source = str(row.get("action_source") or "")
        if capability == "native_action_conditioned" and (
            not source or source in {"logged", "oracle", "proxy", "candidate"}
        ):
            issues.append("action_source_is_not_native")
        if capability == "externally_controlled_video" and source not in {"external_control", "injected_pose"}:
            issues.append("external_control_source_required")
    if row.get("realized_future_ego_state") is not None:
        issues.append("realized_future_state_leakage")
    return issues


def validate_submission(
    rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    public_ids = {
        str(row.get("sample_id") or row.get("source_key") or "")
        for row in public_rows
    }
    public_ids.discard("")
    issues = []
    for index, row in enumerate(rows):
        row_issues = validate_submission_row(row, public_ids=public_ids)
        if row_issues:
            issues.append({
                "row": index,
                "sample_id": row.get("sample_id") or row.get("source_key"),
                "issues": row_issues,
            })
    pair_ids = {}
    for row in rows:
        group = str(row.get("counterfactual_group_id") or "")
        if group:
            pair_ids.setdefault(group, set()).add(str(row.get("branch_mode") or row.get("branch_id") or ""))
    return {
        "protocol": "iac-wam-submission-audit-v1",
        "rows": len(rows),
        "invalid_rows": len(issues),
        "ready": bool(rows) and not issues,
        "issues": issues,
        "counterfactual_groups": {
            group: sorted(modes) for group, modes in sorted(pair_ids.items())
        },
    }


def _cell_from_measurement(
    measurement: dict[str, Any] | None,
    claimed: bool,
    cell_name: str,
) -> dict[str, Any]:
    if measurement:
        status = str(measurement.get("status") or "missing")
        if status not in STATUSES:
            raise ValueError(f"unknown status: {status}")
        return {"status": status, **{k: v for k, v in measurement.items() if k != "status"}}
    if not claimed:
        status = "unavailable" if cell_name in OPTIONAL_CELLS else "ineligible"
        return empty_cell(status, reason="capability_not_declared")
    return empty_cell("missing", reason="no_measurement")


def build_model_scorecard(
    *,
    model_id: str,
    capability: str,
    measurements: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    claimed = claimed_cells(capability)
    measurements = measurements or {}
    cells = {
        cell: _cell_from_measurement(measurements.get(cell), cell in claimed, cell)
        for cell in CELLS
    }
    return {
        "model_id": model_id,
        "capability": capability,
        "claimed_cells": list(claimed),
        "cells": cells,
    }


def frozen_pilot_scorecard() -> dict[str, Any]:
    """Official v1 capability-stratified scoreboard with measured pilot cells."""
    models = [
        build_model_scorecard(
            model_id="worlddrive_tadwm",
            capability="native_action_conditioned",
            measurements={
                "l1": {
                    "status": "pilot",
                    "n": 5,
                    "lateral_mae_m": 0.3155,
                    "yaw_mae_rad": 0.0391,
                    "curvature_mae_1pm": 0.00816,
                    "reason": "native action-conditioned futures scored by frozen L1; not benchmark_v1 580",
                },
                "a2f": {
                    "status": "pass",
                    "n": 5,
                    "mean_left_right_image_l1": 0.0871,
                    "bootstrap_l1_lower_95": 0.0736,
                    "threshold": 0.005,
                },
                "f2a": {
                    "status": "pilot",
                    "n": 25,
                    "material_pairs": "1/25",
                    "reason": "internal future-latent swap; sparse under frozen magnitude gate",
                },
                "cfac": {"status": "unavailable", "reason": "benchmark_v1 CFAC measurement not frozen"},
                "ccfc": {
                    "status": "unavailable",
                    "reason": "no reproducible paired intervention submitted",
                },
                "fcs": {
                    "status": "unavailable",
                    "reason": "independent rollout not joined on generated futures",
                },
            },
        ),
        build_model_scorecard(
            model_id="drivewam_navsim",
            capability="native_action_conditioned",
            measurements={
                "l1": {
                    "status": "pilot",
                    "score": 0.7610,
                    "n": 564,
                    "total": 580,
                    "coverage": 0.9724,
                    "interval_coverage": 0.6981,
                    "motion_fields": ["lateral_speed_mps", "yaw_rate_radps", "curvature_1pm"],
                    "reference_source": "native_wam_action",
                    "reason": "shape-only native4 alignment; 16 samples abstained under the frozen observability gate",
                },
                "cfac": {
                    "status": "pilot",
                    "score": 0.7610,
                    "n": 564,
                    "total": 580,
                    "coverage": 0.9724,
                    "interval_coverage": 0.6981,
                    "score_definition": "primary_shape_composite",
                    "motion_fields": ["lateral_speed_mps", "yaw_rate_radps", "curvature_1pm"],
                    "legacy_diagnostic_score": 0.4825,
                    "dynamic_score": 0.4321,
                    "legacy_coverage": 0.9966,
                    "reason": "formal pilot score now uses shape-only native-action alignment; legacy metric-longitudinal score retained for audit",
                },
                "a2f": {
                    "status": "fail",
                    "n": 60,
                    "mean_left_right_image_l1": 0.000831,
                    "threshold": 0.005,
                    "reason": "external action injection does not move pixels",
                },
                "f2a": {
                    "status": "pass",
                    "n": 25,
                    "transplant_closer_to_source": 0.88,
                    "mean_trajectory_distance_reduction": 0.252,
                    "reason": "legacy future-cache content transplant; retained as F2A diagnostic",
                },
                "fau_f": {"status": "missing", "legacy_diagnostic_score": 0.61, "n": 562, "coverage": 0.9689, "reason": "legacy FAU_F includes unvalidated longitudinal metric scale; shape/relative recomputation required"},
                "fau_a": {"status": "missing", "legacy_diagnostic_score": 0.73, "n": 562, "coverage": 0.9689, "reason": "legacy FAU_A includes unvalidated longitudinal metric scale; shape/relative recomputation required"},
                "fau": {"status": "missing", "legacy_diagnostic_score": 0.6509, "n": 562, "coverage": 0.9689, "aggregation": "sqrt(fau_f*fau_a)", "reason": "legacy FAU combines unvalidated longitudinal metric scale; shape/relative recomputation required"},
                "ccfc": {
                    "status": "missing",
                    "legacy_diagnostic_score": 0.1235,
                    "n": 357,
                    "coverage": 0.6155,
                    "intervention_type": "paired_command",
                    "reason": "legacy metric magnitude includes unvalidated longitudinal scale; shape-only paired recomputation required",
                },
                "fcs": {
                    "status": "pilot",
                    "score": 0.8086,
                    "n": 491,
                    "coverage": 0.982,
                    "environment": "NAVSIM-PDM static_cached_objects_compat",
                },
                "coverage": {
                    "status": "pilot",
                    "per_metric": {
                        "cfac": {"n": 564, "total": 580, "rate": 0.9724},
                        "ccfc": {"n": 357, "total": 580, "rate": 0.6155},
                        "fau": {"n": 562, "total": 580, "rate": 0.9689},
                        "fcs": {"n": 491, "total": 500, "rate": 0.982},
                    },
                },
            },
        ),
        build_model_scorecard(
            model_id="epona_nuplan",
            capability="externally_controlled_video",
            measurements={
                "a2f": {
                    "status": "pilot",
                    "n": 5,
                    "mean_left_right_image_l1": 0.229,
                    "bootstrap_l1_lower_95": 0.131,
                    "threshold": 0.005,
                    "reason": "pose/yaw injection; not scene-disjoint; no native action head",
                },
            },
        ),
        build_model_scorecard(
            model_id="driveva_navsim",
            capability="native_action_conditioned",
            measurements={
                "a2f": {
                    "status": "fail",
                    "n": 5,
                    "mean_left_right_image_l1": 0.00340,
                    "bootstrap_l1_lower_95": 0.00205,
                    "threshold": 0.005,
                },
                "l1": {"status": "missing", "reason": "joint generation not scored on benchmark_v1"},
                "f2a": {"status": "missing", "reason": "future-swap gate not run on benchmark_v1"},
                "ccfc": {
                    "status": "unavailable",
                    "reason": "no reproducible paired intervention submitted",
                },
                "fau": {"status": "unavailable", "reason": "no private-GT joined measurement"},
                "fcs": {
                    "status": "unavailable",
                    "reason": "no compatible independent rollout submitted",
                },
            },
        ),
    ]
    return {
        "protocol": "iac-scorecard-v1",
        "benchmark": "benchmark_v1",
        "probe": "raft_large_ground_plane",
        "claim": "capability-stratified WAM measurement; primary scores use shape/relative motion only",
        "primary_score_policy": {
            "included_motion_fields": ["lateral_speed_mps", "yaw_rate_radps", "curvature_1pm"],
            "excluded_motion_fields": ["speed_mps", "acceleration_mps2"],
            "excluded_distance_mode": "metric_forward_displacement",
            "included_distance_modes": ["relative_observable", "arc_relative"],
            "reason": "longitudinal metric scale is not within the validated error budget",
            "excluded_fields_are_diagnostic_only": True,
        },
        "main_columns": ["cfac", "ccfc", "fau_f", "fau_a", "fau", "fcs", "coverage"],
        "status_policy": {
            "available": "measured and reproducible",
            "pilot": "measured on a pilot or limited subset",
            "unavailable": "optional interface or compatible evaluation environment is absent",
            "missing": "the model claims the capability but the evidence is incomplete",
            "ineligible": "a hard admission or leakage rule was violated",
        },
        "models": models,
    }
