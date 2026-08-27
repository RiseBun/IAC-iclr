"""Continuous ego-motion alignment for image futures and action waypoints.

The image branch and action branch are intentionally converted separately.
No action waypoint is accepted by :func:`image_motion_profile`; this keeps the
candidate-blind image measurement upstream of the comparison boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np


MOTION_FIELDS = (
    "speed_mps",
    "acceleration_mps2",
    "lateral_speed_mps",
    "yaw_rate_radps",
    "curvature_1pm",
)


def _trajectory(trajectory: Any, future_times_s: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError("trajectory must have shape [T,3]")
    if times.shape != (len(points),) or not np.all(np.isfinite(times)):
        raise ValueError("future_times_s must be finite and match trajectory")
    if times[0] <= 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("future_times_s must be positive and strictly increasing")
    if not np.all(np.isfinite(points)):
        raise ValueError("trajectory must be finite")
    return points, times


def _wrapped_delta(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def trajectory_to_motion_profile(
    trajectory: Any,
    future_times_s: Any,
    *,
    initial_speed_mps: float | None = None,
) -> dict[str, Any]:
    """Convert cumulative ego-frame SE(2) waypoints to interval motion.

    Translation is resolved in the previous pose frame. This preserves the
    distinction between forward and lateral motion while avoiding any attempt
    to reconstruct a globally exact path from monocular images.
    """
    points, times = _trajectory(trajectory, future_times_s)
    previous = np.vstack([np.zeros((1, 3), dtype=np.float64), points[:-1]])
    dt = np.diff(np.concatenate([[0.0], times]))
    delta_xy = points[:, :2] - previous[:, :2]
    previous_yaw = previous[:, 2]
    cos_yaw = np.cos(previous_yaw)
    sin_yaw = np.sin(previous_yaw)
    forward_delta = cos_yaw * delta_xy[:, 0] + sin_yaw * delta_xy[:, 1]
    lateral_delta = -sin_yaw * delta_xy[:, 0] + cos_yaw * delta_xy[:, 1]
    distance = np.linalg.norm(delta_xy, axis=1)
    yaw_delta = _wrapped_delta(points[:, 2] - previous_yaw)
    speed = distance / dt
    longitudinal_speed = forward_delta / dt
    acceleration = np.full(len(points), np.nan, dtype=np.float64)
    if initial_speed_mps is not None:
        if not np.isfinite(initial_speed_mps) or initial_speed_mps < 0.0:
            raise ValueError("initial_speed_mps must be finite and non-negative")
        acceleration[0] = (longitudinal_speed[0] - float(initial_speed_mps)) / dt[0]
    if len(points) > 1:
        acceleration[1:] = np.diff(longitudinal_speed) / dt[1:]
    rows = []
    for index in range(len(points)):
        rows.append({
            "time_s": float(times[index]),
            "dt_s": float(dt[index]),
            "progress_m": float(points[index, 0]),
            "lateral_offset_m": float(points[index, 1]),
            "heading_rad": float(points[index, 2]),
            "speed_mps": float(speed[index]),
            "longitudinal_speed_mps": float(longitudinal_speed[index]),
            "acceleration_mps2": None if not np.isfinite(acceleration[index]) else float(acceleration[index]),
            "lateral_speed_mps": float(lateral_delta[index] / dt[index]),
            "yaw_rate_radps": float(yaw_delta[index] / dt[index]),
            "curvature_1pm": float(yaw_delta[index] / max(distance[index], 1e-3)),
            "observability": 1.0,
            "status": "usable",
        })
    return {
        "representation": "continuous-ego-motion-v1",
        "source": "waypoint_kinematics",
        "initial_speed_mps": initial_speed_mps,
        "rows": rows,
    }


def image_motion_profile(
    decoder: dict[str, Any],
    future_times_s: Any,
    *,
    initial_speed_mps: float | None = None,
) -> dict[str, Any]:
    """Build an image-only motion posterior from a completed decoder output."""
    if decoder.get("protocol") != "candidate-blind-continuous-trajectory-v1":
        raise ValueError("decoder must use the candidate-blind continuous protocol")
    profile = trajectory_to_motion_profile(
        decoder.get("trajectory"), future_times_s, initial_speed_mps=initial_speed_mps
    )
    speed_support = list(decoder.get("speed_support") or [])
    if len(speed_support) != len(profile["rows"]):
        raise ValueError("decoder speed_support must match trajectory intervals")
    speeds = np.asarray([float(item["q50"]) for item in speed_support], dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    dt = np.diff(np.concatenate([[0.0], times]))
    acceleration = np.full(len(speeds), np.nan, dtype=np.float64)
    if initial_speed_mps is not None:
        acceleration[0] = (speeds[0] - float(initial_speed_mps)) / dt[0]
    if len(speeds) > 1:
        acceleration[1:] = np.diff(speeds) / dt[1:]
    for index, (row, support) in enumerate(zip(profile["rows"], speed_support)):
        row["speed_mps"] = float(speeds[index])
        row["longitudinal_speed_mps"] = float(speeds[index])
        row["acceleration_mps2"] = None if not np.isfinite(acceleration[index]) else float(acceleration[index])
        row["speed_interval_mps"] = {
            "q05": float(support["q05"]),
            "q50": float(support["q50"]),
            "q95": float(support["q95"]),
        }
        row["observability"] = float(np.clip(support.get("observability", 0.0), 0.0, 1.0))
        row["status"] = str(support.get("status", "abstain"))
    profile["source"] = "image_only_candidate_blind_decoder"
    profile["candidate_bank_used"] = False
    return profile


def _finite_pair(first: Any, second: Any) -> bool:
    return first is not None and second is not None and np.isfinite(first) and np.isfinite(second)


def compare_motion_profiles(
    image_profile: dict[str, Any],
    action_profile: dict[str, Any],
    *,
    include_uncertain: bool = False,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare image-derived and action-derived motion without text labels."""
    image_rows = list(image_profile.get("rows") or [])
    action_rows = list(action_profile.get("rows") or [])
    if len(image_rows) != len(action_rows) or not image_rows:
        raise ValueError("image and action profiles must have matching non-empty rows")
    if image_profile.get("source") != "image_only_candidate_blind_decoder":
        raise ValueError("image_profile must be produced independently from action waypoints")
    allowed = {"usable", "uncertain"} if include_uncertain else {"usable"}
    limits = {
        "speed_mps": 1.5,
        "acceleration_mps2": 1.5,
        "lateral_speed_mps": 0.5,
        "yaw_rate_radps": 0.15,
        "curvature_1pm": 0.06,
    }
    if tolerances:
        unknown = set(tolerances) - set(limits)
        if unknown:
            raise ValueError(f"unknown tolerances: {sorted(unknown)}")
        limits.update({key: float(value) for key, value in tolerances.items()})
    if any(value <= 0.0 for value in limits.values()):
        raise ValueError("all tolerances must be positive")

    per_interval = []
    metric_errors: dict[str, list[float]] = {field: [] for field in MOTION_FIELDS}
    metric_weights: dict[str, list[float]] = {field: [] for field in MOTION_FIELDS}
    speed_interval_hits: list[bool] = []
    evaluable = 0
    for image_row, action_row in zip(image_rows, action_rows):
        if abs(float(image_row["time_s"]) - float(action_row["time_s"])) > 1e-6:
            raise ValueError("image and action timestamps must match")
        use = image_row.get("status") in allowed
        observability = float(np.clip(image_row.get("observability", 0.0), 0.0, 1.0))
        errors: dict[str, float | None] = {}
        if use:
            evaluable += 1
        for field in MOTION_FIELDS:
            left, right = image_row.get(field), action_row.get(field)
            error = abs(float(left) - float(right)) if _finite_pair(left, right) else None
            errors[field] = error
            if use and error is not None:
                metric_errors[field].append(error)
                metric_weights[field].append(max(observability, 1e-3))
        interval = image_row.get("speed_interval_mps")
        speed_hit = None
        if use and interval and _finite_pair(action_row.get("speed_mps"), interval.get("q05")):
            speed_hit = float(interval["q05"]) <= float(action_row["speed_mps"]) <= float(interval["q95"])
            speed_interval_hits.append(speed_hit)
        per_interval.append({
            "time_s": float(image_row["time_s"]),
            "status": image_row.get("status"),
            "observability": observability,
            "evaluable": use,
            "absolute_errors": errors,
            "speed_interval_contains_action": speed_hit,
        })

    metrics: dict[str, Any] = {}
    normalized = []
    for field in MOTION_FIELDS:
        errors = np.asarray(metric_errors[field], dtype=np.float64)
        weights = np.asarray(metric_weights[field], dtype=np.float64)
        if len(errors):
            mae = float(np.average(errors, weights=weights))
            metrics[field] = {
                "mae": mae,
                "rmse": float(np.sqrt(np.average(errors ** 2, weights=weights))),
                "within_tolerance": float(np.average(errors <= limits[field], weights=weights)),
                "count": int(len(errors)),
            }
            normalized.append(mae / limits[field])
        else:
            metrics[field] = {"mae": None, "rmse": None, "within_tolerance": None, "count": 0}
    status = "ok" if evaluable else "abstain"
    return {
        "protocol": "continuous-foresight-action-alignment-v1",
        "status": status,
        "coverage": float(evaluable / len(image_rows)),
        "evaluable_intervals": int(evaluable),
        "total_intervals": int(len(image_rows)),
        "metrics": metrics,
        "speed_posterior_coverage": None if not speed_interval_hits else float(np.mean(speed_interval_hits)),
        "experimental_composite": None if not normalized else float(np.exp(-np.mean(normalized))),
        "experimental_composite_status": "calibration_only_not_formal_score",
        "tolerances": limits,
        "per_interval": per_interval,
        "leakage_audit": {
            "image_source": image_profile.get("source"),
            "action_waypoint_visible_to_image_decoder": False,
            "candidate_bank_used": bool(image_profile.get("candidate_bank_used", True)),
        },
    }


def compare_counterfactual_motion_deltas(
    clear_image: dict[str, Any],
    risk_image: dict[str, Any],
    clear_action: dict[str, Any],
    risk_action: dict[str, Any],
    *,
    minimum_action_delta: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare risk-minus-clear image and action effects in continuous space."""
    profiles = (clear_image, risk_image, clear_action, risk_action)
    lengths = {len(profile.get("rows") or []) for profile in profiles}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise ValueError("all counterfactual profiles must have matching rows")
    for profile in (clear_image, risk_image):
        if profile.get("source") != "image_only_candidate_blind_decoder":
            raise ValueError("counterfactual image profiles must be action-blind")
    thresholds = {
        "speed_mps": 0.25,
        "acceleration_mps2": 0.25,
        "lateral_speed_mps": 0.10,
        "yaw_rate_radps": 0.03,
        "curvature_1pm": 0.01,
    }
    if minimum_action_delta:
        thresholds.update({key: float(value) for key, value in minimum_action_delta.items()})
    output: dict[str, Any] = {}
    for field in MOTION_FIELDS:
        image_delta = []
        action_delta = []
        weights = []
        for ci, ri, ca, ra in zip(
            clear_image["rows"], risk_image["rows"], clear_action["rows"], risk_action["rows"]
        ):
            values = (ci.get(field), ri.get(field), ca.get(field), ra.get(field))
            usable = ci.get("status") == "usable" and ri.get("status") == "usable"
            if usable and all(value is not None and np.isfinite(value) for value in values):
                image_delta.append(float(values[1]) - float(values[0]))
                action_delta.append(float(values[3]) - float(values[2]))
                weights.append(max(min(float(ci.get("observability", 0.0)), float(ri.get("observability", 0.0))), 1e-3))
        image_array = np.asarray(image_delta, dtype=np.float64)
        action_array = np.asarray(action_delta, dtype=np.float64)
        weight_array = np.asarray(weights, dtype=np.float64)
        active = np.abs(action_array) >= thresholds[field]
        if len(image_array) == 0 or not np.any(active):
            output[field] = {"status": "no_material_action_intervention", "count": 0}
            continue
        image_active = image_array[active]
        action_active = action_array[active]
        active_weights = weight_array[active]
        denominator = float(np.linalg.norm(image_active) * np.linalg.norm(action_active))
        output[field] = {
            "status": "ok",
            "count": int(len(image_active)),
            "delta_mae": float(np.average(np.abs(image_active - action_active), weights=active_weights)),
            "sign_agreement": float(np.average(np.sign(image_active) == np.sign(action_active), weights=active_weights)),
            "cosine_alignment": None if denominator <= 1e-12 else float(np.dot(image_active, action_active) / denominator),
            "image_delta": image_active.tolist(),
            "action_delta": action_active.tolist(),
        }
    return {
        "protocol": "counterfactual-continuous-delta-alignment-v1",
        "definition": "risk_minus_clear_image_effect_vs_risk_minus_clear_action_effect",
        "metrics": output,
        "leakage_audit": {"action_waypoint_visible_to_image_decoder": False},
    }
