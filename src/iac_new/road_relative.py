"""Road-relative support regions and tolerant action comparison."""

from __future__ import annotations

from typing import Any

import numpy as np

from .region import trajectory_states


def _bounds(values: np.ndarray) -> dict[str, float]:
    return {"q05": float(np.quantile(values, 0.05)), "q50": float(np.quantile(values, 0.50)), "q95": float(np.quantile(values, 0.95))}


def _inflate_bounds(bounds: dict[str, Any], amount: float) -> dict[str, float]:
    """Expand a calibrated interval without moving its median."""
    value = float(amount)
    return {
        "q05": float(bounds["q05"]) - value,
        "q50": float(bounds["q50"]),
        "q95": float(bounds["q95"]) + value,
    }


def road_relative_posterior(
    trajectory: np.ndarray,
    future_times_s: np.ndarray,
    *,
    profile_support: list[dict[str, Any]] | None = None,
    observability: np.ndarray | None = None,
    lateral_inflation_m: float = 0.0,
    heading_inflation_rad: float = 0.0,
    curvature_inflation_1pm: float = 0.0,
    lateral_inflation_by_interval: np.ndarray | None = None,
    heading_inflation_by_interval: np.ndarray | None = None,
    curvature_inflation_by_interval: np.ndarray | None = None,
) -> dict[str, Any]:
    """Convert an ego-frame support tube into road-relative intervals.

    The x/y/yaw frame is only used as a local road frame; no camera calibration
    or absolute map is required.  Progress is reported separately so speed
    uncertainty cannot contaminate direction/curvature scores.
    """
    if min(lateral_inflation_m, heading_inflation_rad, curvature_inflation_1pm) < 0.0:
        raise ValueError("support inflation must be non-negative")
    states = trajectory_states(np.asarray(trajectory, dtype=np.float64), np.asarray(future_times_s, dtype=np.float64))
    support = profile_support or []
    quality = np.ones(len(states), dtype=np.float64) if observability is None else np.asarray(observability, dtype=np.float64)
    if quality.shape != (len(states),):
        raise ValueError("observability must match trajectory knots")
    lateral_by = None if lateral_inflation_by_interval is None else np.asarray(lateral_inflation_by_interval, dtype=np.float64)
    heading_by = None if heading_inflation_by_interval is None else np.asarray(heading_inflation_by_interval, dtype=np.float64)
    curvature_by = None if curvature_inflation_by_interval is None else np.asarray(curvature_inflation_by_interval, dtype=np.float64)
    for name, array in (("lateral", lateral_by), ("heading", heading_by), ("curvature", curvature_by)):
        if array is not None and array.shape != (len(states),):
            raise ValueError(f"{name}_inflation_by_interval must match trajectory knots")
    points: list[dict[str, Any]] = []
    for index, (state, q) in enumerate(zip(states, quality)):
        if index < len(support):
            item = support[index]
            lateral = item.get("y_m", {"q05": state["y_m"], "q50": state["y_m"], "q95": state["y_m"]})
            heading = item.get("yaw_rad", {"q05": state["yaw_rad"], "q50": state["yaw_rad"], "q95": state["yaw_rad"]})
            progress = item.get("x_m", {"q05": state["x_m"], "q50": state["x_m"], "q95": state["x_m"]})
        else:
            lateral = {"q05": state["y_m"], "q50": state["y_m"], "q95": state["y_m"]}
            heading = {"q05": state["yaw_rad"], "q50": state["yaw_rad"], "q95": state["yaw_rad"]}
            progress = {"q05": state["x_m"], "q50": state["x_m"], "q95": state["x_m"]}
        # Curvature is derived from the joint trajectory; the support interval
        # is conservative when only knotwise yaw quantiles are available.
        curvature = item.get("curvature_1pm", {"q05": state["curvature_1pm"], "q50": state["curvature_1pm"], "q95": state["curvature_1pm"]}) if index < len(support) else {"q05": state["curvature_1pm"], "q50": state["curvature_1pm"], "q95": state["curvature_1pm"]}
        lateral_amount = float(lateral_by[index]) if lateral_by is not None else float(lateral_inflation_m)
        heading_amount = float(heading_by[index]) if heading_by is not None else float(heading_inflation_rad)
        curvature_amount = float(curvature_by[index]) if curvature_by is not None else float(curvature_inflation_1pm)
        points.append({
            "time_s": float(state["time_s"]),
            "heading_change_range_rad": _inflate_bounds(heading, heading_amount),
            "lateral_offset_range_m": _inflate_bounds(lateral, lateral_amount),
            "curvature_range_1pm": _inflate_bounds(curvature, curvature_amount),
            "progress_range_m": dict(progress),
            "observability": float(np.clip(q, 0.0, 1.0)),
            "speed_status": "abstain" if q < 0.25 else ("uncertain" if q < 0.55 else "diagnostic"),
        })
    return {
        "representation": "road_relative_trajectory_support",
        "variables": ["heading_change", "lateral_offset", "curvature", "progress"],
        "support": points,
        "summary": {
            "heading_change_range_rad": _bounds(np.asarray([state["yaw_rad"] for state in states], dtype=np.float64)),
            "lateral_offset_range_m": _bounds(np.asarray([state["y_m"] for state in states], dtype=np.float64)),
            "curvature_range_1pm": _bounds(np.asarray([state["curvature_1pm"] for state in states], dtype=np.float64)),
            "progress_range_m": _bounds(np.asarray([state["x_m"] for state in states], dtype=np.float64)),
        },
        "mean_observability": float(np.mean(np.clip(quality, 0.0, 1.0))),
        "speed_primary_score": False,
        "support_inflation": {
            "lateral_m": float(lateral_inflation_m),
            "heading_rad": float(heading_inflation_rad),
            "curvature_1pm": float(curvature_inflation_1pm),
            "per_interval": {
                "lateral_m": None if lateral_by is None else lateral_by.tolist(),
                "heading_rad": None if heading_by is None else heading_by.tolist(),
                "curvature_1pm": None if curvature_by is None else curvature_by.tolist(),
            },
        },
    }


def compare_action_to_support(
    action_trajectory: np.ndarray,
    posterior: dict[str, Any],
    future_times_s: np.ndarray,
    *,
    heading_tolerance_rad: float = 0.10,
    lateral_tolerance_m: float = 0.50,
    curvature_tolerance_1pm: float = 0.06,
) -> dict[str, Any]:
    """Score whether an action falls inside the recovered support region."""
    states = trajectory_states(np.asarray(action_trajectory, dtype=np.float64), np.asarray(future_times_s, dtype=np.float64))
    rows = posterior.get("support") or []
    if len(states) != len(rows):
        raise ValueError("action and posterior must have matching knots")
    checks = []
    heading_sign = []
    curvature_sign = []
    lateral_errors = []
    curvature_errors = []
    for state, row in zip(states, rows):
        def inside(value: float, bounds: dict[str, Any], tolerance: float) -> bool:
            return value >= float(bounds["q05"]) - tolerance and value <= float(bounds["q95"]) + tolerance
        h = inside(state["yaw_rad"], row["heading_change_range_rad"], heading_tolerance_rad)
        y = inside(state["y_m"], row["lateral_offset_range_m"], lateral_tolerance_m)
        c = inside(state["curvature_1pm"], row["curvature_range_1pm"], curvature_tolerance_1pm)
        checks.append({"heading": h, "lateral": y, "curvature": c, "joint": bool(h and y and c), "observability": row["observability"]})
        pred_heading = float(row["heading_change_range_rad"]["q50"])
        pred_curvature = float(row["curvature_range_1pm"]["q50"])
        heading_sign.append(bool(np.sign(state["yaw_rad"]) == np.sign(pred_heading) or abs(state["yaw_rad"]) < heading_tolerance_rad))
        curvature_sign.append(bool(np.sign(state["curvature_1pm"]) == np.sign(pred_curvature) or abs(state["curvature_1pm"]) < curvature_tolerance_1pm))
        lateral_errors.append(abs(float(state["y_m"]) - float(row["lateral_offset_range_m"]["q50"])))
        curvature_errors.append(abs(float(state["curvature_1pm"]) - pred_curvature))
    weights = np.asarray([item["observability"] for item in checks], dtype=np.float64)
    joint = np.asarray([item["joint"] for item in checks], dtype=np.float64)
    return {
        "joint_support_coverage": float(np.average(joint, weights=np.maximum(weights, 1e-3))),
        "heading_support_coverage": float(np.average([item["heading"] for item in checks], weights=np.maximum(weights, 1e-3))),
        "lateral_support_coverage": float(np.average([item["lateral"] for item in checks], weights=np.maximum(weights, 1e-3))),
        "curvature_support_coverage": float(np.average([item["curvature"] for item in checks], weights=np.maximum(weights, 1e-3))),
        "heading_sign_accuracy": float(np.average(heading_sign, weights=np.maximum(weights, 1e-3))),
        "left_right_direction_accuracy": float(np.average(heading_sign, weights=np.maximum(weights, 1e-3))),
        "curvature_sign_accuracy": float(np.average(curvature_sign, weights=np.maximum(weights, 1e-3))),
        "lateral_error_m": float(np.average(lateral_errors, weights=np.maximum(weights, 1e-3))),
        "curvature_error_1pm": float(np.average(curvature_errors, weights=np.maximum(weights, 1e-3))),
        "by_interval": checks,
    }
