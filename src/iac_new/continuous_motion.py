"""Continuous ego-motion alignment for image futures and action waypoints.

The image branch and action branch are intentionally converted separately.
No action waypoint is accepted by :func:`image_motion_profile`; this keeps the
candidate-blind image measurement upstream of the comparison boundary.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


MOTION_FIELDS = (
    "speed_mps",
    "acceleration_mps2",
    "lateral_speed_mps",
    "yaw_rate_radps",
    "curvature_1pm",
)

IMAGE_PROFILE_SOURCES = {
    "image_only_candidate_blind_decoder",
    "history_anchored_image_residual_decoder",
    "history_anchored_optimizer_residual_decoder",
}


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


def _cumulative_arc_length(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [T,2]")
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def discrete_frechet_distance(first_xy: np.ndarray, second_xy: np.ndarray) -> float:
    """Return the discrete Fréchet distance between two ordered 2-D curves."""
    first = np.asarray(first_xy, dtype=np.float64)
    second = np.asarray(second_xy, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != 2 or second.shape[1] != 2:
        raise ValueError("curves must have shape [T,2]")
    if not len(first) or not len(second):
        raise ValueError("curves must be non-empty")
    table = np.full((len(first), len(second)), np.inf, dtype=np.float64)
    for i in range(len(first)):
        for j in range(len(second)):
            distance = float(np.linalg.norm(first[i] - second[j]))
            if i == 0 and j == 0:
                table[i, j] = distance
            elif i == 0:
                table[i, j] = max(table[i, j - 1], distance)
            elif j == 0:
                table[i, j] = max(table[i - 1, j], distance)
            else:
                table[i, j] = max(min(table[i - 1, j], table[i - 1, j - 1], table[i, j - 1]), distance)
    return float(table[-1, -1])


def constrained_dtw_distance(first_xy: np.ndarray, second_xy: np.ndarray, *, window: int = 1) -> tuple[float, float]:
    """Return mean constrained-DTW distance and normalized warp cost.

    A narrow window preserves temporal meaning; unconstrained DTW is unsuitable
    for causal evaluation because it can align delayed actions away.
    """
    first = np.asarray(first_xy, dtype=np.float64)
    second = np.asarray(second_xy, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != 2 or second.shape[1] != 2:
        raise ValueError("curves must have shape [T,2]")
    if not len(first) or not len(second) or int(window) < 0:
        raise ValueError("curves must be non-empty and window must be non-negative")
    cost = np.full((len(first) + 1, len(second) + 1), np.inf, dtype=np.float64)
    steps = np.zeros_like(cost, dtype=np.int32)
    cost[0, 0] = 0.0
    for i in range(1, len(first) + 1):
        low, high = max(1, i - int(window)), min(len(second), i + int(window))
        for j in range(low, high + 1):
            distance = float(np.linalg.norm(first[i - 1] - second[j - 1]))
            predecessor = min((cost[i - 1, j], i - 1, j), (cost[i, j - 1], i, j - 1), (cost[i - 1, j - 1], i - 1, j - 1), key=lambda item: item[0])
            cost[i, j] = distance + predecessor[0]
            steps[i, j] = steps[predecessor[1], predecessor[2]] + 1
    if not np.isfinite(cost[-1, -1]):
        return float("nan"), float("nan")
    path_steps = max(int(steps[-1, -1]), 1)
    return float(cost[-1, -1] / path_steps), float((path_steps - max(len(first), len(second))) / path_steps)


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


def history_only_motion_profile(
    history_ego_state: Any,
    future_times_s: Any,
    *,
    history_times_s: Any | None = None,
    model: str = "constant_speed_yaw_rate",
) -> dict[str, Any]:
    """Extrapolate history without reading future images or action waypoints."""
    history = np.asarray(history_ego_state, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    if history.ndim != 2 or history.shape[1] < 4 or len(history) < 1:
        raise ValueError("history_ego_state must have shape [H,>=4]")
    if not np.all(np.isfinite(history[:, :4])):
        raise ValueError("history_ego_state must be finite through speed")
    speed = max(float(history[-1, 3]), 0.0)
    yaw_rate = (
        float(history[-1, 4])
        if history.shape[1] >= 5 and np.isfinite(history[-1, 4])
        else 0.0
    )
    if model not in {"constant_speed_yaw_rate", "constant_acceleration_yaw_rate"}:
        raise ValueError("unknown history-only baseline model")
    acceleration = 0.0
    if model == "constant_acceleration_yaw_rate":
        history_times = np.asarray(history_times_s, dtype=np.float64)
        if history_times.shape != (len(history),) or np.any(np.diff(history_times) <= 0.0):
            raise ValueError("history_times_s must be increasing and match history state")
        if len(history) >= 2:
            design = np.column_stack([history_times - history_times[-1], np.ones(len(history_times))])
            acceleration = float(np.linalg.lstsq(design, history[:, 3], rcond=None)[0][0])
            acceleration = float(np.clip(acceleration, -5.0, 3.0))
    trajectory = np.zeros((len(times), 3), dtype=np.float64)
    previous_time = 0.0
    for index, time_s in enumerate(times):
        dt = float(time_s - previous_time)
        previous = trajectory[index - 1] if index else np.zeros(3, dtype=np.float64)
        interval_speed = max(speed + acceleration * 0.5 * (previous_time + float(time_s)), 0.0)
        yaw_mid = float(previous[2] + 0.5 * yaw_rate * dt)
        trajectory[index] = [
            previous[0] + interval_speed * np.cos(yaw_mid) * dt,
            previous[1] + interval_speed * np.sin(yaw_mid) * dt,
            previous[2] + yaw_rate * dt,
        ]
        previous_time = float(time_s)
    profile = trajectory_to_motion_profile(
        trajectory, times, initial_speed_mps=speed
    )
    profile["source"] = f"history_only_{model}"
    profile["history_anchor"] = {
        "speed_mps": speed,
        "acceleration_mps2": acceleration,
        "yaw_rate_radps": yaw_rate,
    }
    return profile


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
    pose_support = list(decoder.get("profile_support") or [])
    if pose_support and len(pose_support) != len(profile["rows"]):
        raise ValueError("decoder profile_support must match trajectory intervals")
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
        if pose_support:
            support_row = pose_support[index]
            intervals = {}
            for source_key, output_key in (("x_m", "x_m"), ("y_m", "y_m"), ("yaw_rad", "heading_rad")):
                interval = support_row.get(source_key)
                if interval is None:
                    continue
                if not all(key in interval and np.isfinite(interval[key]) for key in ("q05", "q50", "q95")):
                    raise ValueError(f"decoder profile_support[{index}] has invalid {source_key}")
                if not float(interval["q05"]) <= float(interval["q50"]) <= float(interval["q95"]):
                    raise ValueError(f"decoder profile_support[{index}] has unordered {source_key}")
                intervals[output_key] = {
                    "q05": float(interval["q05"]),
                    "q50": float(interval["q50"]),
                    "q95": float(interval["q95"]),
                }
            if intervals:
                row["pose_intervals"] = intervals
    residual_mode = bool(
        (decoder.get("decoder_parameters") or {}).get("history_anchored_speed_residual")
    )
    if residual_mode:
        history_speeds = list(decoder.get("history_speed_profile_mps") or [])
        residuals = [
            float(row["speed_mps"]) - float(history_speed)
            for row, history_speed in zip(profile["rows"], history_speeds)
        ]
        if len(history_speeds) != len(profile["rows"]):
            raise ValueError("history-anchored decoder must export matching speed residuals")
        profile["source"] = "history_anchored_optimizer_residual_decoder"
        history_initial_speed = decoder.get("history_initial_speed_mps")
        if history_initial_speed is None or not np.isfinite(history_initial_speed):
            raise ValueError("history-anchored decoder must export initial history speed")
        profile["initial_speed_mps"] = float(history_initial_speed)
        profile["longitudinal_model"] = {
            "protocol": "optimizer-internal-longitudinal-residual-v1",
            "history_speed_profile_mps": [float(value) for value in history_speeds],
            "speed_residual_mps": [float(value) for value in residuals],
            "absolute_image_speed_used": False,
            "action_waypoint_visible_to_predictor": False,
        }
    else:
        profile["source"] = "image_only_candidate_blind_decoder"
    profile["candidate_bank_used"] = False
    return profile


def longitudinal_residual_features(
    decoder: dict[str, Any],
    future_times_s: Any,
    history_profile: dict[str, Any],
) -> dict[str, Any]:
    """Extract action-blind future speed changes relative to the history null.

    Decoder speeds describe intervals, so the trend is fitted at interval
    midpoints. The absolute image scale is discarded; only the fitted change
    from t=0 is retained.
    """
    if decoder.get("protocol") != "candidate-blind-continuous-trajectory-v1":
        raise ValueError("decoder must use the candidate-blind continuous protocol")
    times = np.asarray(future_times_s, dtype=np.float64)
    rows = list(history_profile.get("rows") or [])
    support = list(decoder.get("speed_support") or [])
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("at least two increasing future times are required")
    if len(rows) != len(times) or len(support) != len(times):
        raise ValueError("history profile and speed support must match future times")
    anchor = history_profile.get("history_anchor") or {}
    initial_speed = anchor.get("speed_mps")
    if initial_speed is None or not np.isfinite(initial_speed):
        raise ValueError("history profile must provide a finite speed anchor")
    raw_speed = np.asarray([float(item["q50"]) for item in support], dtype=np.float64)
    observability = np.asarray(
        [float(np.clip(item.get("observability", 0.0), 0.0, 1.0)) for item in support],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(raw_speed)):
        raise ValueError("decoder speed medians must be finite")
    previous_times = np.concatenate([[0.0], times[:-1]])
    midpoints = 0.5 * (previous_times + times)
    design = np.column_stack([np.ones(len(times)), midpoints])
    root_weight = np.sqrt(np.maximum(observability, 0.05))
    coefficients = np.linalg.lstsq(design * root_weight[:, None], raw_speed * root_weight, rcond=None)[0]
    image_slope = float(coefficients[1])
    raw_scale = float(np.average(np.abs(raw_speed), weights=np.maximum(observability, 0.05)))
    raw_scale = max(raw_scale, 0.5)
    relative_slope = image_slope / raw_scale
    image_delta = float(initial_speed) * relative_slope * midpoints
    history_speed = np.asarray([float(row["speed_mps"]) for row in rows], dtype=np.float64)
    history_delta = history_speed - float(initial_speed)
    innovation = image_delta - history_delta
    return {
        "protocol": "longitudinal-residual-feature-v1",
        "absolute_image_speed_used": False,
        "action_waypoint_used": False,
        "initial_speed_mps": float(initial_speed),
        "image_speed_slope_mps2": image_slope,
        "raw_image_scale_mps": raw_scale,
        "relative_image_slope_per_s": relative_slope,
        "raw_image_intercept_mps": float(coefficients[0]),
        "rows": [
            {
                "time_s": float(time_s),
                "feature_time_s": float(midpoint),
                "image_speed_delta_mps": float(image_value),
                "history_speed_delta_mps": float(history_value),
                "innovation_mps": float(residual),
            }
            for time_s, midpoint, image_value, history_value, residual in zip(
                times, midpoints, image_delta, history_delta, innovation
            )
        ],
    }


def history_anchored_residual_motion_profile(
    decoder: dict[str, Any],
    future_times_s: Any,
    history_profile: dict[str, Any],
    *,
    longitudinal_gain: float,
    speed_interval_radius_mps: float,
) -> dict[str, Any]:
    """Add calibrated image speed-change residuals to a frozen history null."""
    if not np.isfinite(longitudinal_gain):
        raise ValueError("longitudinal_gain must be finite")
    if not np.isfinite(speed_interval_radius_mps) or speed_interval_radius_mps <= 0.0:
        raise ValueError("speed_interval_radius_mps must be finite and positive")
    features = longitudinal_residual_features(decoder, future_times_s, history_profile)
    profile = image_motion_profile(
        decoder,
        future_times_s,
        initial_speed_mps=float(features["initial_speed_mps"]),
    )
    history_rows = list(history_profile["rows"])
    feature_rows = list(features["rows"])
    times = np.asarray(future_times_s, dtype=np.float64)
    dt = np.diff(np.concatenate([[0.0], times]))
    predicted_speed = np.asarray([
        max(
            float(history_row["speed_mps"])
            + float(longitudinal_gain) * float(feature_row["innovation_mps"]),
            0.0,
        )
        for history_row, feature_row in zip(history_rows, feature_rows)
    ])
    acceleration = np.diff(
        np.concatenate([[float(features["initial_speed_mps"])], predicted_speed])
    ) / dt
    for row, feature_row, speed, accel in zip(
        profile["rows"], feature_rows, predicted_speed, acceleration
    ):
        row["speed_mps"] = float(speed)
        row["longitudinal_speed_mps"] = float(speed)
        row["acceleration_mps2"] = float(accel)
        row["speed_interval_mps"] = {
            "q05": float(max(0.0, speed - speed_interval_radius_mps)),
            "q50": float(speed),
            "q95": float(speed + speed_interval_radius_mps),
        }
        row["longitudinal_residual_feature"] = feature_row
    profile["source"] = "history_anchored_image_residual_decoder"
    profile["longitudinal_model"] = {
        "protocol": "history-anchored-longitudinal-residual-v1",
        "history_profile_source": history_profile.get("source"),
        "longitudinal_gain": float(longitudinal_gain),
        "speed_interval_radius_mps": float(speed_interval_radius_mps),
        "feature": features,
        "absolute_image_speed_used": False,
        "action_waypoint_visible_to_predictor": False,
    }
    return profile


def reanchor_longitudinal_control_profile(
    source_profile: dict[str, Any],
    target_history_profile: dict[str, Any],
    target_times_s: Any,
    *,
    reverse: bool = False,
) -> dict[str, Any]:
    """Transfer only a future image residual onto the recipient history null."""
    if source_profile.get("source") not in {
        "history_anchored_image_residual_decoder",
        "history_anchored_optimizer_residual_decoder",
    }:
        raise ValueError("source_profile must use the longitudinal residual protocol")
    times = np.asarray(target_times_s, dtype=np.float64)
    source_rows = list(source_profile.get("rows") or [])
    history_rows = list(target_history_profile.get("rows") or [])
    if len(source_rows) != len(times) or len(history_rows) != len(times):
        raise ValueError("source, history, and target times must have matching rows")
    result = {
        key: value for key, value in source_profile.items()
        if key not in {"rows", "longitudinal_model"}
    }
    selected_rows = list(reversed(source_rows)) if reverse else source_rows
    result["rows"] = [{key: value for key, value in row.items()} for row in selected_rows]
    model = dict(source_profile["longitudinal_model"])
    if source_profile.get("source") == "history_anchored_optimizer_residual_decoder":
        source_residuals = list(model.get("speed_residual_mps") or [])
        if len(source_residuals) != len(times):
            raise ValueError("optimizer residual profile must provide matching residuals")
        if reverse:
            source_residuals.reverse()
        target_anchor = float(target_history_profile["history_anchor"]["speed_mps"])
        previous_time = 0.0
        previous_speed = target_anchor
        target_history_speeds = []
        for row, history_row, residual, time_s in zip(
            result["rows"], history_rows, source_residuals, times
        ):
            speed = max(float(history_row["speed_mps"]) + float(residual), 0.0)
            dt = float(time_s - previous_time)
            row["time_s"] = float(time_s)
            row["dt_s"] = dt
            row["speed_mps"] = speed
            row["longitudinal_speed_mps"] = speed
            row["acceleration_mps2"] = (speed - previous_speed) / dt
            target_history_speeds.append(float(history_row["speed_mps"]))
            previous_time = float(time_s)
            previous_speed = speed
        model["history_speed_profile_mps"] = target_history_speeds
        model["speed_residual_mps"] = [float(value) for value in source_residuals]
        model["control_reanchored"] = True
        model["control_signal_reversed"] = bool(reverse)
        result["longitudinal_model"] = model
        return result
    gain = float(model["longitudinal_gain"])
    radius = float(model["speed_interval_radius_mps"])
    target_anchor = float(target_history_profile["history_anchor"]["speed_mps"])
    previous_time = 0.0
    previous_speed = target_anchor
    transferred_features = []
    for row, history_row, time_s in zip(result["rows"], history_rows, times):
        source_feature = row["longitudinal_residual_feature"]
        image_delta = float(source_feature["image_speed_delta_mps"])
        target_history_delta = float(history_row["speed_mps"]) - target_anchor
        innovation = image_delta - target_history_delta
        speed = max(float(history_row["speed_mps"]) + gain * innovation, 0.0)
        dt = float(time_s - previous_time)
        feature = {
            "time_s": float(time_s),
            "feature_time_s": float(0.5 * (previous_time + time_s)),
            "image_speed_delta_mps": image_delta,
            "history_speed_delta_mps": target_history_delta,
            "innovation_mps": innovation,
        }
        row["time_s"] = float(time_s)
        row["dt_s"] = dt
        row["speed_mps"] = speed
        row["longitudinal_speed_mps"] = speed
        row["acceleration_mps2"] = (speed - previous_speed) / dt
        row["speed_interval_mps"] = {
            "q05": max(0.0, speed - radius),
            "q50": speed,
            "q95": speed + radius,
        }
        row["longitudinal_residual_feature"] = feature
        transferred_features.append(feature)
        previous_time = float(time_s)
        previous_speed = speed
    model["control_reanchored"] = True
    model["control_signal_reversed"] = bool(reverse)
    model["target_history_profile_source"] = target_history_profile.get("source")
    model["feature"] = dict(model["feature"], rows=transferred_features)
    result["longitudinal_model"] = model
    return result


def _finite_pair(first: Any, second: Any) -> bool:
    return first is not None and second is not None and np.isfinite(first) and np.isfinite(second)


def _comparison_limits(tolerances: dict[str, float] | None) -> dict[str, float]:
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
    return limits


def _compare_profile_rows(
    predicted_profile: dict[str, Any],
    action_profile: dict[str, Any],
    eligibility_profile: dict[str, Any],
    *,
    include_uncertain: bool,
    limits: dict[str, float],
    score_speed_posterior: bool,
) -> dict[str, Any]:
    predicted_rows = list(predicted_profile.get("rows") or [])
    action_rows = list(action_profile.get("rows") or [])
    eligibility_rows = list(eligibility_profile.get("rows") or [])
    if not predicted_rows or len(predicted_rows) != len(action_rows) or len(predicted_rows) != len(eligibility_rows):
        raise ValueError("predicted, action, and eligibility profiles must have matching non-empty rows")
    allowed = {"usable", "uncertain"} if include_uncertain else {"usable"}
    per_interval = []
    metric_errors: dict[str, list[float]] = {field: [] for field in MOTION_FIELDS}
    metric_weights: dict[str, list[float]] = {field: [] for field in MOTION_FIELDS}
    posterior_rows: list[tuple[float, float, float, float, float]] = []
    evaluable = 0
    for predicted_row, action_row, gate_row in zip(predicted_rows, action_rows, eligibility_rows):
        timestamps = (predicted_row["time_s"], action_row["time_s"], gate_row["time_s"])
        if max(float(value) for value in timestamps) - min(float(value) for value in timestamps) > 1e-6:
            raise ValueError("predicted, action, and eligibility timestamps must match")
        use = gate_row.get("status") in allowed
        observability = float(np.clip(gate_row.get("observability", 0.0), 0.0, 1.0))
        errors: dict[str, float | None] = {}
        if use:
            evaluable += 1
        for field in MOTION_FIELDS:
            left, right = predicted_row.get(field), action_row.get(field)
            error = abs(float(left) - float(right)) if _finite_pair(left, right) else None
            errors[field] = error
            if use and error is not None:
                metric_errors[field].append(error)
                metric_weights[field].append(max(observability, 1e-3))
        interval = predicted_row.get("speed_interval_mps") if score_speed_posterior else None
        speed_hit = None
        interval_width = None
        interval_score = None
        wis = None
        if use and interval and _finite_pair(action_row.get("speed_mps"), interval.get("q05")):
            lower = float(interval["q05"])
            median = float(interval["q50"])
            upper = float(interval["q95"])
            target = float(action_row["speed_mps"])
            if not lower <= median <= upper:
                raise ValueError("speed posterior quantiles must be ordered")
            alpha = 0.10
            interval_width = upper - lower
            interval_score = interval_width
            if target < lower:
                interval_score += 2.0 / alpha * (lower - target)
            elif target > upper:
                interval_score += 2.0 / alpha * (target - upper)
            wis = (0.5 * abs(target - median) + alpha / 2.0 * interval_score) / 1.5
            speed_hit = lower <= target <= upper
            posterior_rows.append((float(speed_hit), upper - lower, interval_score, wis, max(observability, 1e-3)))
        per_interval.append({
            "time_s": float(gate_row["time_s"]),
            "status": gate_row.get("status"),
            "observability": observability,
            "evaluable": use,
            "absolute_errors": errors,
            "speed_interval_contains_action": speed_hit,
            "speed_interval_width_mps": interval_width,
            "speed_interval_score_90": interval_score,
            "speed_wis_90": wis,
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
    if posterior_rows:
        posterior = np.asarray(posterior_rows, dtype=np.float64)
        weights = posterior[:, 4]
        empirical_coverage = float(np.average(posterior[:, 0], weights=weights))
        speed_posterior = {
            "nominal_coverage": 0.90,
            "empirical_coverage": empirical_coverage,
            "absolute_calibration_error": abs(empirical_coverage - 0.90),
            "mean_interval_width_mps": float(np.average(posterior[:, 1], weights=weights)),
            "mean_interval_score_90": float(np.average(posterior[:, 2], weights=weights)),
            "mean_wis_90": float(np.average(posterior[:, 3], weights=weights)),
            "count": int(len(posterior)),
        }
    else:
        speed_posterior = {
            "nominal_coverage": 0.90,
            "empirical_coverage": None,
            "absolute_calibration_error": None,
            "mean_interval_width_mps": None,
            "mean_interval_score_90": None,
            "mean_wis_90": None,
            "count": 0,
        }
    return {
        "status": "ok" if evaluable else "abstain",
        "coverage": float(evaluable / len(predicted_rows)),
        "evaluable_intervals": int(evaluable),
        "total_intervals": int(len(predicted_rows)),
        "metrics": metrics,
        "speed_posterior": speed_posterior,
        "speed_posterior_coverage": speed_posterior["empirical_coverage"],
        "experimental_composite": None if not normalized else float(np.exp(-np.mean(normalized))),
        "experimental_composite_status": "calibration_only_not_formal_score",
        "tolerances": limits,
        "per_interval": per_interval,
    }


def compare_motion_profiles(
    image_profile: dict[str, Any],
    action_profile: dict[str, Any],
    *,
    include_uncertain: bool = False,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare image-derived and action-derived motion without text labels."""
    if image_profile.get("source") not in IMAGE_PROFILE_SOURCES:
        raise ValueError("image_profile must be produced independently from action waypoints")
    result = _compare_profile_rows(
        image_profile,
        action_profile,
        image_profile,
        include_uncertain=include_uncertain,
        limits=_comparison_limits(tolerances),
        score_speed_posterior=True,
    )
    return result | {
        "protocol": "continuous-foresight-action-alignment-v1",
        "leakage_audit": {
            "image_source": image_profile.get("source"),
            "action_waypoint_visible_to_image_decoder": False,
            "candidate_bank_used": bool(image_profile.get("candidate_bank_used", True)),
        },
    }


def compare_distance_profiles(
    image_profile: dict[str, Any],
    action_profile: dict[str, Any],
    *,
    scale_mode: str = "metric",
    include_uncertain: bool = False,
    allow_non_image_source: bool = False,
) -> dict[str, Any]:
    """Compare forward displacement in metric or relative-shape coordinates.

    ``metric`` compares the independently recovered forward displacement in
    metres. ``scale_free`` normalizes each profile by its largest absolute
    displacement. ``relative`` normalizes by each profile's terminal forward
    displacement and is the Level-1 primary distance shape measure; neither
    mode borrows scale from the action branch.
    """
    if not allow_non_image_source and image_profile.get("source") not in IMAGE_PROFILE_SOURCES:
        raise ValueError("image_profile must be produced independently from action waypoints")
    if scale_mode not in {"metric", "scale_free", "relative"}:
        raise ValueError("scale_mode must be metric, scale_free, or relative")
    predicted_rows = list(image_profile.get("rows") or [])
    action_rows = list(action_profile.get("rows") or [])
    if not predicted_rows or len(predicted_rows) != len(action_rows):
        raise ValueError("distance profiles must have matching non-empty rows")
    predicted = np.asarray([float(row["progress_m"]) for row in predicted_rows], dtype=np.float64)
    action = np.asarray([float(row["progress_m"]) for row in action_rows], dtype=np.float64)
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(action)):
        raise ValueError("forward displacement must be finite")
    scale_ratio = None
    if scale_mode in {"scale_free", "relative"}:
        if scale_mode == "relative":
            predicted_scale = float(abs(predicted[-1]))
            action_scale = float(abs(action[-1]))
        else:
            predicted_scale = float(np.max(np.abs(predicted)))
            action_scale = float(np.max(np.abs(action)))
        if predicted_scale < 0.5 or action_scale < 0.5:
            return {
                "protocol": "continuous-forward-distance-alignment-v1",
                "scale_mode": scale_mode,
                "status": "abstain",
                "coverage": 0.0,
                "evaluable_intervals": 0,
                "total_intervals": len(predicted_rows),
                "reason": "forward_displacement_amplitude_too_small_for_relative_profile",
                "metrics": {"forward_displacement_profile": {"mae": None, "rmse": None, "endpoint_abs_error": None, "increment_mae": None, "curve_cosine": None, "count": 0}},
            }
        scale_ratio = predicted_scale / action_scale
        predicted = predicted / predicted_scale
        action = action / action_scale
    allowed = {"usable", "uncertain"} if include_uncertain else {"usable"}
    errors = []
    increments_predicted = np.diff(np.concatenate([[0.0], predicted]))
    increments_action = np.diff(np.concatenate([[0.0], action]))
    per_interval = []
    for index, (predicted_row, action_row) in enumerate(zip(predicted_rows, action_rows)):
        status = str(predicted_row.get("status", "abstain"))
        use = status in allowed
        error = abs(float(predicted[index] - action[index])) if use else None
        if error is not None:
            errors.append(error)
        per_interval.append({
            "time_s": float(predicted_row["time_s"]),
            "status": status,
            "evaluable": use,
            "forward_displacement_error": error,
            "forward_displacement_increment_error": (
                abs(float(increments_predicted[index] - increments_action[index])) if use else None
            ),
        })
    valid = [index for index, row in enumerate(per_interval) if row["evaluable"]]
    increment_errors = [abs(float(increments_predicted[index] - increments_action[index])) for index in valid]
    if errors:
        pred_valid = predicted[valid]
        action_valid = action[valid]
        pred_norm = float(np.linalg.norm(pred_valid))
        action_norm = float(np.linalg.norm(action_valid))
        cosine = float(np.dot(pred_valid, action_valid) / (pred_norm * action_norm)) if pred_norm > 1e-9 and action_norm > 1e-9 else None
        metric = {
            "mae": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "endpoint_abs_error": float(abs(predicted[-1] - action[-1])) if valid[-1] == len(predicted) - 1 else None,
            "increment_mae": float(np.mean(increment_errors)) if increment_errors else None,
            "curve_cosine": cosine,
            "count": len(errors),
        }
    else:
        metric = {"mae": None, "rmse": None, "endpoint_abs_error": None, "increment_mae": None, "curve_cosine": None, "count": 0}
    return {
        "protocol": "continuous-forward-distance-alignment-v1",
        "scale_mode": scale_mode,
        "status": "ok" if errors else "abstain",
        "coverage": float(len(valid) / len(predicted_rows)),
        "evaluable_intervals": len(valid),
        "total_intervals": len(predicted_rows),
        "independent_terminal_scale_ratio": scale_ratio,
        "metrics": {"forward_displacement_profile": metric},
        "per_interval": per_interval,
        "leakage_audit": {
            "image_source": image_profile.get("source"),
            "action_waypoint_visible_to_image_decoder": False,
            "action_used_for_image_scale": False,
        },
    }


def compare_pose_profiles(
    image_profile: dict[str, Any],
    action_profile: dict[str, Any],
    *,
    scale_mode: str = "metric",
    include_uncertain: bool = False,
    allow_non_image_source: bool = False,
) -> dict[str, Any]:
    """Compare the time-indexed planar pose ``[x, y, heading]``."""
    if not allow_non_image_source and image_profile.get("source") not in IMAGE_PROFILE_SOURCES:
        raise ValueError("image_profile must be produced independently from action waypoints")
    if scale_mode not in {"metric", "scale_free", "relative", "arc_relative"}:
        raise ValueError("scale_mode must be metric, scale_free, relative, or arc_relative")
    predicted_rows = list(image_profile.get("rows") or [])
    action_rows = list(action_profile.get("rows") or [])
    if not predicted_rows or len(predicted_rows) != len(action_rows):
        raise ValueError("pose profiles must have matching non-empty rows")
    predicted = np.asarray([[float(row["progress_m"]), float(row["lateral_offset_m"]), float(row["heading_rad"])] for row in predicted_rows], dtype=np.float64)
    action = np.asarray([[float(row["progress_m"]), float(row["lateral_offset_m"]), float(row["heading_rad"])] for row in action_rows], dtype=np.float64)
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(action)):
        raise ValueError("pose values must be finite")
    scale_ratio = None
    if scale_mode in {"scale_free", "relative", "arc_relative"}:
        if scale_mode == "relative":
            predicted_scale = float(abs(predicted[-1, 0]))
            action_scale = float(abs(action[-1, 0]))
        elif scale_mode == "arc_relative":
            predicted_scale = float(_cumulative_arc_length(predicted[:, :2])[-1])
            action_scale = float(_cumulative_arc_length(action[:, :2])[-1])
        else:
            predicted_scale = float(np.max(np.linalg.norm(predicted[:, :2], axis=1)))
            action_scale = float(np.max(np.linalg.norm(action[:, :2], axis=1)))
        if predicted_scale < 0.5 or action_scale < 0.5:
            return {
                "protocol": "continuous-se2-pose-alignment-v1",
                "scale_mode": scale_mode,
                "status": "abstain",
                "coverage": 0.0,
                "evaluable_intervals": 0,
                "total_intervals": len(predicted_rows),
                "reason": "translation_amplitude_too_small_for_relative_pose",
                "metrics": {"se2_pose": {"translation_mae": None, "forward_mae": None, "lateral_mae": None, "heading_mae_rad": None, "endpoint_translation_error": None, "endpoint_heading_error_rad": None, "path_cosine": None, "frechet_distance": None, "constrained_dtw_distance": None, "dtw_warp_ratio": None, "count": 0}},
            }
        scale_ratio = predicted_scale / action_scale
        predicted[:, :2] /= predicted_scale
        action[:, :2] /= action_scale
    allowed = {"usable", "uncertain"} if include_uncertain else {"usable"}
    valid = [index for index, row in enumerate(predicted_rows) if str(row.get("status", "abstain")) in allowed]
    if not valid:
        return {
            "protocol": "continuous-se2-pose-alignment-v1",
            "scale_mode": scale_mode,
            "status": "abstain",
            "coverage": 0.0,
            "evaluable_intervals": 0,
            "total_intervals": len(predicted_rows),
            "independent_translation_scale_ratio": scale_ratio,
            "metrics": {"se2_pose": {"translation_mae": None, "forward_mae": None, "lateral_mae": None, "heading_mae_rad": None, "endpoint_translation_error": None, "endpoint_heading_error_rad": None, "path_cosine": None, "frechet_distance": None, "constrained_dtw_distance": None, "dtw_warp_ratio": None, "count": 0}},
        }
    translation_error = np.linalg.norm(predicted[valid, :2] - action[valid, :2], axis=1)
    forward_error = np.abs(predicted[valid, 0] - action[valid, 0])
    lateral_error = np.abs(predicted[valid, 1] - action[valid, 1])
    heading_error = np.abs(_wrapped_delta(predicted[valid, 2] - action[valid, 2]))
    predicted_xy = predicted[valid, :2].reshape(-1)
    action_xy = action[valid, :2].reshape(-1)
    xy_norms = np.linalg.norm(predicted_xy) * np.linalg.norm(action_xy)
    cosine = float(np.dot(predicted_xy, action_xy) / xy_norms) if xy_norms > 1e-9 else None
    frechet = discrete_frechet_distance(predicted[valid, :2], action[valid, :2])
    dtw_distance, dtw_warp_ratio = constrained_dtw_distance(predicted[valid, :2], action[valid, :2], window=1)
    last = valid[-1]
    metric = {
        "translation_mae": float(np.mean(translation_error)),
        "forward_mae": float(np.mean(forward_error)),
        "lateral_mae": float(np.mean(lateral_error)),
        "heading_mae_rad": float(np.mean(heading_error)),
        "endpoint_translation_error": float(np.linalg.norm(predicted[last, :2] - action[last, :2])) if last == len(predicted) - 1 else None,
        "endpoint_heading_error_rad": float(abs(_wrapped_delta(predicted[last, 2] - action[last, 2]))) if last == len(predicted) - 1 else None,
        "path_cosine": cosine,
        "frechet_distance": frechet,
        "constrained_dtw_distance": dtw_distance,
        "dtw_warp_ratio": dtw_warp_ratio,
        "count": len(valid),
    }
    return {
        "protocol": "continuous-se2-pose-alignment-v1",
        "scale_mode": scale_mode,
        "status": "ok",
        "coverage": float(len(valid) / len(predicted_rows)),
        "evaluable_intervals": len(valid),
        "total_intervals": len(predicted_rows),
        "independent_translation_scale_ratio": scale_ratio,
        "metrics": {"se2_pose": metric},
        "leakage_audit": {
            "image_source": image_profile.get("source"),
            "action_waypoint_visible_to_image_decoder": False,
            "action_used_for_image_scale": False,
        },
    }


def compare_pose_posteriors(
    image_profile: dict[str, Any],
    action_profile: dict[str, Any],
    *,
    include_uncertain: bool = False,
    nominal_coverage: float = 0.90,
) -> dict[str, Any]:
    """Evaluate calibrated ``x/y/heading`` intervals against action waypoints."""
    if image_profile.get("source") not in IMAGE_PROFILE_SOURCES:
        raise ValueError("image_profile must be produced independently from action waypoints")
    if not np.isfinite(nominal_coverage) or not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be between zero and one")
    predicted_rows = list(image_profile.get("rows") or [])
    action_rows = list(action_profile.get("rows") or [])
    if not predicted_rows or len(predicted_rows) != len(action_rows):
        raise ValueError("pose posterior profiles must have matching non-empty rows")
    allowed = {"usable", "uncertain"} if include_uncertain else {"usable"}
    components = {
        "x_m": ("progress_m", False),
        "y_m": ("lateral_offset_m", False),
        "heading_rad": ("heading_rad", True),
    }
    alpha = 1.0 - float(nominal_coverage)
    rows_by_component: dict[str, list[tuple[bool, float, float, float, float, float]]] = {
        key: [] for key in components
    }
    joint_rows: list[tuple[bool, float]] = []
    for predicted, action in zip(predicted_rows, action_rows):
        if str(predicted.get("status", "abstain")) not in allowed:
            continue
        intervals = predicted.get("pose_intervals") or {}
        component_hits = []
        for component, (target_key, is_angle) in components.items():
            interval = intervals.get(component)
            target = action.get(target_key)
            if interval is None or target is None or not all(
                np.isfinite(interval.get(key, np.nan)) for key in ("q05", "q50", "q95")
            ) or not np.isfinite(target):
                continue
            lower = float(interval["q05"])
            median = float(interval["q50"])
            upper = float(interval["q95"])
            target_value = float(target)
            if is_angle:
                target_value = float(median + _wrapped_delta(np.asarray(target_value - median)))
            hit = lower <= target_value <= upper
            width = upper - lower
            interval_score = width
            if target_value < lower:
                interval_score += 2.0 / alpha * (lower - target_value)
            elif target_value > upper:
                interval_score += 2.0 / alpha * (target_value - upper)
            wis = (0.5 * abs(target_value - median) + alpha / 2.0 * interval_score) / 1.5
            observability = max(float(predicted.get("observability", 0.0)), 1e-3)
            rows_by_component[component].append(
                (bool(hit), width, interval_score, wis, observability, abs(target_value - median))
            )
            component_hits.append(bool(hit))
        if len(component_hits) == len(components):
            joint_rows.append((all(component_hits), max(float(predicted.get("observability", 0.0)), 1e-3)))

    def summarize(rows: list[tuple[bool, float, float, float, float, float]]) -> dict[str, Any]:
        if not rows:
            return {
                "status": "abstain",
                "empirical_coverage": None,
                "absolute_calibration_error": None,
                "mean_interval_width": None,
                "mean_interval_score": None,
                "mean_wis": None,
                "mean_median_abs_error": None,
                "intervals": 0,
            }
        values = np.asarray(rows, dtype=np.float64)
        weights = np.maximum(values[:, 4], 1e-3)
        empirical = float(np.average(values[:, 0], weights=weights))
        return {
            "status": "ok",
            "empirical_coverage": empirical,
            "absolute_calibration_error": abs(empirical - nominal_coverage),
            "mean_interval_width": float(np.average(values[:, 1], weights=weights)),
            "mean_interval_score": float(np.average(values[:, 2], weights=weights)),
            "mean_wis": float(np.average(values[:, 3], weights=weights)),
            "mean_median_abs_error": float(np.average(values[:, 5], weights=weights)),
            "intervals": int(len(rows)),
        }

    joint = None
    if joint_rows:
        joint_values = np.asarray(joint_rows, dtype=np.float64)
        joint_weights = np.maximum(joint_values[:, 1], 1e-3)
        joint = {
            "status": "ok",
            "empirical_coverage": float(np.average(joint_values[:, 0], weights=joint_weights)),
            "absolute_calibration_error": abs(float(np.average(joint_values[:, 0], weights=joint_weights)) - nominal_coverage),
            "intervals": int(len(joint_rows)),
        }
    total_eligible = sum(1 for row in predicted_rows if str(row.get("status", "abstain")) in allowed)
    evaluated = sum(1 for row in rows_by_component["x_m"])
    return {
        "protocol": "continuous-se2-pose-posterior-v1",
        "status": "ok" if any(rows_by_component.values()) else "abstain",
        "nominal_coverage": float(nominal_coverage),
        "coverage": float(evaluated / len(predicted_rows)),
        "evaluable_intervals": int(evaluated),
        "total_intervals": int(len(predicted_rows)),
        "eligible_point_intervals": int(total_eligible),
        "metrics": {component: summarize(rows) for component, rows in rows_by_component.items()},
        "joint_pose": joint or {
            "status": "abstain",
            "empirical_coverage": None,
            "absolute_calibration_error": None,
            "intervals": 0,
        },
        "leakage_audit": {
            "action_waypoint_visible_to_image_decoder": False,
            "action_used_for_pose_interval_calibration": False,
        },
    }


def apply_pose_interval_calibration(
    image_profile: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Inflate pose intervals with a calibration-only conformal radius."""
    if image_profile.get("source") not in IMAGE_PROFILE_SOURCES:
        raise ValueError("image_profile must be produced independently from action waypoints")
    if calibration.get("protocol") != "continuous-se2-pose-calibration-v1":
        raise ValueError("unsupported pose calibration protocol")
    calibrated = copy.deepcopy(image_profile)
    radii = calibration.get("parameters", {}).get("conformal_radius", {})
    for row in calibrated.get("rows") or []:
        intervals = row.get("pose_intervals") or {}
        for component, interval in intervals.items():
            radius = float(radii.get(component, 0.0))
            if not np.isfinite(radius) or radius < 0.0:
                raise ValueError(f"invalid conformal radius for {component}")
            interval["q05"] = float(interval["q05"]) - radius
            interval["q95"] = float(interval["q95"]) + radius
    calibrated["pose_interval_calibration"] = {
        "protocol": calibration["protocol"],
        "parameters": {"conformal_radius": {key: float(value) for key, value in radii.items()}},
        "action_waypoint_used": False,
    }
    return calibrated


def compare_history_baseline(
    history_profile: dict[str, Any],
    action_profile: dict[str, Any],
    image_profile: dict[str, Any],
    *,
    include_uncertain: bool = False,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a history-only null on exactly the image probe's eligible rows."""
    if history_profile.get("source") not in {
        "history_only_constant_speed_yaw_rate",
        "history_only_constant_acceleration_yaw_rate",
    }:
        raise ValueError("history_profile must be the frozen history-only null")
    if image_profile.get("source") not in IMAGE_PROFILE_SOURCES:
        raise ValueError("image_profile must provide the eligibility mask")
    result = _compare_profile_rows(
        history_profile,
        action_profile,
        image_profile,
        include_uncertain=include_uncertain,
        limits=_comparison_limits(tolerances),
        score_speed_posterior=False,
    )
    return result | {
        "protocol": "history-only-action-null-v1",
        "eligibility_mask_source": "image_probe_observability",
    }


def compare_future_control(
    control_profile: dict[str, Any],
    action_profile: dict[str, Any],
    target_image_profile: dict[str, Any],
    *,
    include_uncertain: bool = False,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a corrupted/shuffled future on the target probe's eligible rows."""
    if control_profile.get("source") not in IMAGE_PROFILE_SOURCES:
        raise ValueError("control_profile must originate from the image-only decoder")
    if target_image_profile.get("source") not in IMAGE_PROFILE_SOURCES:
        raise ValueError("target_image_profile must provide the eligibility mask")
    result = _compare_profile_rows(
        control_profile,
        action_profile,
        target_image_profile,
        include_uncertain=include_uncertain,
        limits=_comparison_limits(tolerances),
        score_speed_posterior=False,
    )
    return result | {
        "protocol": "future-specificity-control-v1",
        "eligibility_mask_source": "target_image_probe_observability",
    }


def foresight_gain(
    image_comparison: dict[str, Any],
    history_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Return paired error reduction over the history-only null."""
    if image_comparison.get("coverage") != history_comparison.get("coverage"):
        raise ValueError("image and history comparisons must use the same eligibility mask")
    metrics = {}
    for field in MOTION_FIELDS:
        image_mae = image_comparison["metrics"][field]["mae"]
        history_mae = history_comparison["metrics"][field]["mae"]
        if image_mae is None or history_mae is None:
            metrics[field] = {
                "image_mae": image_mae,
                "history_mae": history_mae,
                "absolute_gain": None,
                "relative_gain": None,
                "future_beats_history": None,
            }
            continue
        gain = float(history_mae - image_mae)
        metrics[field] = {
            "image_mae": float(image_mae),
            "history_mae": float(history_mae),
            "absolute_gain": gain,
            "relative_gain": float(gain / max(float(history_mae), 1e-6)),
            "future_beats_history": gain > 0.0,
        }
    return {
        "protocol": "foresight-gain-over-history-null-v1",
        "definition": "history_only_mae_minus_image_future_mae",
        "positive_means_future_adds_information": True,
        "coverage": image_comparison.get("coverage"),
        "metrics": metrics,
    }


def compare_longitudinal_behavior(
    predicted_profile: dict[str, Any],
    action_profile: dict[str, Any],
    eligibility_profile: dict[str, Any],
    *,
    change_deadband_mps: float = 0.15,
    include_uncertain: bool = False,
) -> dict[str, Any]:
    """Compare longitudinal change, rather than absolute speed.

    This is the Level 1 behavior signal: a predictor is useful when it gets the
    direction and magnitude of future speed change right, even if monocular
    scale prevents accurate absolute speed recovery.
    """
    if change_deadband_mps < 0.0 or not np.isfinite(change_deadband_mps):
        raise ValueError("change_deadband_mps must be finite and non-negative")
    predicted_rows = list(predicted_profile.get("rows") or [])
    action_rows = list(action_profile.get("rows") or [])
    gate_rows = list(eligibility_profile.get("rows") or [])
    if not predicted_rows or len(predicted_rows) != len(action_rows) or len(predicted_rows) != len(gate_rows):
        raise ValueError("longitudinal profiles must have matching non-empty rows")
    allowed = {"usable", "uncertain"} if include_uncertain else {"usable"}
    predicted_anchor = predicted_profile.get("initial_speed_mps")
    action_anchor = action_profile.get("initial_speed_mps")
    if predicted_anchor is None:
        predicted_anchor = predicted_rows[0].get("speed_mps")
    if action_anchor is None:
        action_anchor = action_rows[0].get("speed_mps")
    if not _finite_pair(predicted_anchor, action_anchor):
        raise ValueError("profiles must provide finite initial speeds")
    delta_errors = []
    direction_matches = []
    significant_matches = []
    predicted_signs = []
    action_signs = []
    for predicted, action, gate in zip(predicted_rows, action_rows, gate_rows):
        if gate.get("status") not in allowed:
            continue
        predicted_delta = float(predicted["speed_mps"]) - float(predicted_anchor)
        action_delta = float(action["speed_mps"]) - float(action_anchor)
        delta_errors.append(abs(predicted_delta - action_delta))
        def sign(value: float) -> int:
            return 1 if value > change_deadband_mps else -1 if value < -change_deadband_mps else 0
        predicted_sign = sign(predicted_delta)
        action_sign = sign(action_delta)
        predicted_signs.append(predicted_sign)
        action_signs.append(action_sign)
        direction_matches.append(float(predicted_sign == action_sign))
        if action_sign != 0:
            significant_matches.append(float(predicted_sign == action_sign))
    if not delta_errors:
        return {
            "status": "abstain",
            "evaluable_intervals": 0,
            "delta_speed_mae_mps": None,
            "change_direction_accuracy": None,
            "significant_change_direction_accuracy": None,
            "predicted_accel_fraction": None,
            "action_accel_fraction": None,
        }
    return {
        "status": "ok",
        "evaluable_intervals": len(delta_errors),
        "change_deadband_mps": float(change_deadband_mps),
        "delta_speed_mae_mps": float(np.mean(delta_errors)),
        "change_direction_accuracy": float(np.mean(direction_matches)),
        "significant_change_direction_accuracy": (
            None if not significant_matches else float(np.mean(significant_matches))
        ),
        "predicted_accel_fraction": float(np.mean(np.asarray(predicted_signs) > 0)),
        "action_accel_fraction": float(np.mean(np.asarray(action_signs) > 0)),
        "predicted_decel_fraction": float(np.mean(np.asarray(predicted_signs) < 0)),
        "action_decel_fraction": float(np.mean(np.asarray(action_signs) < 0)),
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
        if profile.get("source") not in IMAGE_PROFILE_SOURCES:
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


def compare_counterfactual_se2_consistency(
    clear_image: dict[str, Any],
    risk_image: dict[str, Any],
    clear_action: dict[str, Any],
    risk_action: dict[str, Any],
    *,
    scale_mode: str = "metric",
    minimum_translation_delta_m: float = 0.05,
    minimum_heading_delta_rad: float = 0.01,
    translation_tolerance_m: float = 0.50,
    heading_tolerance_rad: float = 0.05,
) -> dict[str, Any]:
    """Score causal consistency between imagined and executed SE(2) responses.

    The response is defined as risk-minus-clear.  The image branch is required
    to be candidate-blind, so this function measures whether the action branch
    changes in the same direction, with a similar magnitude and at the same
    times.  ``scale_free`` is a shape-only diagnostic; ``metric`` is the
    primary report because it retains metre/radian response magnitude.
    """
    if scale_mode not in {"metric", "scale_free", "arc_relative"}:
        raise ValueError("scale_mode must be metric, scale_free, or arc_relative")
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (
            minimum_translation_delta_m,
            minimum_heading_delta_rad,
            translation_tolerance_m,
            heading_tolerance_rad,
        )
    ):
        raise ValueError("thresholds and tolerances must be finite and non-negative")
    profiles = (clear_image, risk_image, clear_action, risk_action)
    lengths = {len(profile.get("rows") or []) for profile in profiles}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise ValueError("all counterfactual profiles must have matching rows")
    for profile in (clear_image, risk_image):
        if profile.get("source") not in IMAGE_PROFILE_SOURCES:
            raise ValueError("counterfactual image profiles must be action-blind")
    total = next(iter(lengths))
    image_times = np.asarray([float(row["time_s"]) for row in clear_image["rows"]], dtype=np.float64)
    for profile in profiles[1:]:
        times = np.asarray([float(row["time_s"]) for row in profile["rows"]], dtype=np.float64)
        if not np.allclose(image_times, times, atol=1e-6, rtol=0.0):
            raise ValueError("counterfactual profiles must share an identical time axis")

    def pose(profile: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [[float(row["progress_m"]), float(row["lateral_offset_m"]), float(row["heading_rad"])]
             for row in profile["rows"]],
            dtype=np.float64,
        )

    image_clear = pose(clear_image)
    image_risk = pose(risk_image)
    action_clear = pose(clear_action)
    action_risk = pose(risk_action)
    raw_action_clear = action_clear.copy()
    raw_action_risk = action_risk.copy()
    if not all(np.all(np.isfinite(value)) for value in (image_clear, image_risk, action_clear, action_risk)):
        raise ValueError("counterfactual pose values must be finite")
    arc_scales = None
    if scale_mode == "arc_relative":
        arc_scales = []
        for trajectory in (image_clear, image_risk, action_clear, action_risk):
            arc = float(_cumulative_arc_length(trajectory[:, :2])[-1])
            if arc < 1e-9:
                return {
                    "protocol": "continuous-counterfactual-foresight-consistency-v1",
                    "scale_mode": scale_mode,
                    "status": "abstain",
                    "reason": "trajectory_arc_length_too_small",
                    "coverage": 0.0,
                    "evaluable_intervals": 0,
                    "total_intervals": total,
                    "score": None,
                }
            arc_scales.append(arc)
            trajectory[:, :2] /= arc
    image_response = image_risk - image_clear
    image_response[:, 2] = _wrapped_delta(image_response[:, 2])
    action_response = action_risk - action_clear
    action_response[:, 2] = _wrapped_delta(action_response[:, 2])
    translation_scale = 1.0
    image_heading_scale = 1.0
    action_heading_scale = 1.0
    if scale_mode == "scale_free":
        image_translation_scale = float(np.max(np.linalg.norm(image_response[:, :2], axis=1)))
        action_translation_scale = float(np.max(np.linalg.norm(action_response[:, :2], axis=1)))
        image_heading_scale = float(np.max(np.abs(image_response[:, 2])))
        action_heading_scale = float(np.max(np.abs(action_response[:, 2])))
        if max(image_translation_scale, action_translation_scale) < 1e-9 and max(image_heading_scale, action_heading_scale) < 1e-9:
            return {
                "protocol": "continuous-counterfactual-foresight-consistency-v1",
                "scale_mode": scale_mode,
                "status": "abstain",
                "reason": "no_counterfactual_response",
                "coverage": 0.0,
                "evaluable_intervals": 0,
                "total_intervals": total,
                "score": None,
            }
        translation_scale = max(image_translation_scale, action_translation_scale, 1e-9)
        image_heading_scale = max(image_heading_scale, 1e-9)
        action_heading_scale = max(action_heading_scale, 1e-9)
        image_response[:, :2] /= translation_scale
        action_response[:, :2] /= translation_scale
        image_response[:, 2] /= image_heading_scale
        action_response[:, 2] /= action_heading_scale

    raw_action_response = raw_action_risk - raw_action_clear
    raw_action_response[:, 2] = _wrapped_delta(raw_action_response[:, 2])
    raw_action_translation_norm = np.linalg.norm(raw_action_response[:, :2], axis=1)
    raw_action_heading_abs = np.abs(raw_action_response[:, 2])
    active = (raw_action_translation_norm >= minimum_translation_delta_m) | (
        raw_action_heading_abs >= minimum_heading_delta_rad
    )
    valid = []
    weights = []
    for index, (ci, ri) in enumerate(zip(clear_image["rows"], risk_image["rows"])):
        image_usable = ci.get("status") == "usable" and ri.get("status") == "usable"
        if image_usable and active[index]:
            valid.append(index)
            weights.append(max(min(float(ci.get("observability", 0.0)), float(ri.get("observability", 0.0))), 1e-3))
    if not valid:
        return {
            "protocol": "continuous-counterfactual-foresight-consistency-v1",
            "scale_mode": scale_mode,
            "status": "abstain",
            "reason": "no_observable_material_action_response",
            "coverage": 0.0,
            "evaluable_intervals": 0,
            "total_intervals": total,
            "score": None,
            "metrics": {},
        }
    valid_array = np.asarray(valid, dtype=np.int64)
    weight_array = np.asarray(weights, dtype=np.float64)
    image_valid = image_response[valid_array]
    action_valid = action_response[valid_array]

    translation_errors = np.linalg.norm(image_valid[:, :2] - action_valid[:, :2], axis=1)
    forward_errors = np.abs(image_valid[:, 0] - action_valid[:, 0])
    lateral_errors = np.abs(image_valid[:, 1] - action_valid[:, 1])
    heading_errors = np.abs(_wrapped_delta(image_valid[:, 2] - action_valid[:, 2]))
    if scale_mode == "metric":
        translation_scores = np.clip(1.0 - translation_errors / max(translation_tolerance_m, 1e-9), 0.0, 1.0)
        heading_scores = np.clip(1.0 - heading_errors / max(heading_tolerance_rad, 1e-9), 0.0, 1.0)
    else:
        translation_scores = np.clip(1.0 - translation_errors, 0.0, 1.0)
        heading_scores = np.clip(1.0 - heading_errors, 0.0, 1.0)
    translation_active = raw_action_translation_norm[valid_array] >= minimum_translation_delta_m
    heading_active = raw_action_heading_abs[valid_array] >= minimum_heading_delta_rad

    def weighted_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
        if mask is not None:
            values = values[mask]
            local_weights = weight_array[mask]
        else:
            local_weights = weight_array
        if len(values) == 0:
            return None
        return float(np.average(values, weights=local_weights))

    translation_direction = []
    translation_direction_weights = []
    for image_value, action_value, weight, is_active in zip(
        image_valid[:, :2], action_valid[:, :2], weight_array, translation_active
    ):
        if not is_active:
            continue
        denominator = float(np.linalg.norm(image_value) * np.linalg.norm(action_value))
        cosine = float(np.dot(image_value, action_value) / denominator) if denominator > 1e-9 else -1.0
        translation_direction.append((cosine + 1.0) / 2.0)
        translation_direction_weights.append(weight)
    heading_direction = [
        float(np.sign(image_value) == np.sign(action_value))
        for image_value, action_value, is_active in zip(image_valid[:, 2], action_valid[:, 2], heading_active)
        if is_active
    ]
    direction_parts = []
    if translation_direction:
        direction_parts.append(float(np.average(translation_direction, weights=translation_direction_weights)))
    if heading_direction:
        direction_parts.append(float(np.mean(heading_direction)))
    direction_score = float(np.mean(direction_parts)) if direction_parts else None

    response_matrix_image = image_valid.copy()
    response_matrix_action = action_valid.copy()
    matrix_weights = np.sqrt(weight_array)[:, None]
    image_flat = (response_matrix_image * matrix_weights).reshape(-1)
    action_flat = (response_matrix_action * matrix_weights).reshape(-1)
    denominator = float(np.linalg.norm(image_flat) * np.linalg.norm(action_flat))
    temporal_cosine = float(np.dot(image_flat, action_flat) / denominator) if denominator > 1e-9 else -1.0
    temporal_score = (temporal_cosine + 1.0) / 2.0
    # If an interval has both translation and heading responses, score both.
    component_scores = []
    component_weights = []
    for score, weight, is_active in zip(translation_scores, weight_array, translation_active):
        if is_active:
            component_scores.append(float(score))
            component_weights.append(float(weight))
    for score, weight, is_active in zip(heading_scores, weight_array, heading_active):
        if is_active:
            component_scores.append(float(score))
            component_weights.append(float(weight))
    magnitude_score = float(np.average(component_scores, weights=component_weights)) if component_scores else None
    sub_scores = [value for value in (direction_score, magnitude_score, temporal_score) if value is not None]
    score = float(np.prod(sub_scores) ** (1.0 / len(sub_scores))) if sub_scores else None
    return {
        "protocol": "continuous-counterfactual-foresight-consistency-v1",
        "definition": "risk_minus_clear_SE2_image_response_vs_action_response",
        "scale_mode": scale_mode,
        "status": "ok" if score is not None else "abstain",
        "coverage": float(len(valid) / total),
        "evaluable_intervals": len(valid),
        "total_intervals": total,
        "score": score,
        "subscores": {
            "response_direction": direction_score,
            "response_magnitude": magnitude_score,
            "response_temporal_alignment": temporal_score,
        },
        "metrics": {
            "translation_response": {
                "mae": float(np.average(translation_errors, weights=weight_array)),
                "forward_mae": float(np.average(forward_errors, weights=weight_array)),
                "lateral_mae": float(np.average(lateral_errors, weights=weight_array)),
                "direction_score": float(np.average(translation_direction, weights=translation_direction_weights)) if translation_direction else None,
                "count": int(np.sum(translation_active)),
            },
            "heading_response": {
                "mae_rad": float(np.average(heading_errors, weights=weight_array)),
                "direction_accuracy": float(np.mean(heading_direction)) if heading_direction else None,
                "count": int(np.sum(heading_active)),
            },
            "response_temporal": {"cosine": temporal_cosine},
        },
        "normalization": {
            "translation_scale": float(translation_scale),
            "arc_scales": None if arc_scales is None else {
                "image_clear": float(arc_scales[0]),
                "image_risk": float(arc_scales[1]),
                "action_clear": float(arc_scales[2]),
                "action_risk": float(arc_scales[3]),
            },
            "image_heading_scale": float(image_heading_scale),
            "action_heading_scale": float(action_heading_scale),
            "translation_tolerance_m": float(translation_tolerance_m),
            "heading_tolerance_rad": float(heading_tolerance_rad),
        },
        "leakage_audit": {
            "action_waypoint_visible_to_image_decoder": False,
            "action_used_for_image_scale": False,
        },
    }
