"""Joint trajectory-mode summaries and finite discrete regions."""

from __future__ import annotations

from typing import Any

import numpy as np


def _angle_delta(current: float, previous: float) -> float:
    return float(np.arctan2(np.sin(current - previous), np.cos(current - previous)))


def trajectory_states(trajectory: np.ndarray, future_times_s: np.ndarray) -> list[dict[str, float]]:
    """Convert [x, y, yaw] knots into coupled motion quantities."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or times.shape != (trajectory.shape[0],):
        raise ValueError("trajectory and future_times_s have incompatible shapes")
    states: list[dict[str, float]] = []
    previous_position = np.zeros(2, dtype=np.float64)
    previous_yaw = 0.0
    previous_time = 0.0
    for knot, time_s in zip(trajectory, times):
        x_m, y_m, yaw_rad = (float(value) for value in knot)
        dt = float(time_s - previous_time)
        if dt <= 0.0:
            raise ValueError("future times must be strictly increasing from the anchor")
        displacement = np.asarray([x_m, y_m]) - previous_position
        distance = float(np.linalg.norm(displacement))
        speed_mps = distance / dt
        motion_direction = float(np.arctan2(displacement[1], displacement[0])) if distance > 1e-8 else previous_yaw
        yaw_change = _angle_delta(yaw_rad, previous_yaw)
        curvature = yaw_change / max(distance, 1e-3)
        states.append(
            {
                "time_s": float(time_s),
                "x_m": x_m,
                "y_m": y_m,
                "yaw_rad": yaw_rad,
                "motion_direction_rad": motion_direction,
                "speed_mps": speed_mps,
                "curvature_1pm": curvature,
            }
        )
        previous_position = np.asarray([x_m, y_m])
        previous_yaw = yaw_rad
        previous_time = float(time_s)
    return states


def _range(values: list[float]) -> list[float]:
    return [float(min(values)), float(max(values))] if values else [None, None]


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float | None:
    if not values:
        return None
    order = np.argsort(np.asarray(values, dtype=np.float64))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    total = float(sorted_weights.sum())
    if total <= 0.0:
        return None
    index = int(np.searchsorted(np.cumsum(sorted_weights), float(quantile) * total, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def build_trajectory_region(
    *,
    candidates: list[dict[str, Any]],
    probabilities: np.ndarray,
    selected_indices: list[int],
    future_times_s: np.ndarray,
    target_coverage: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    mode_summaries: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        states = trajectory_states(candidate["trajectory"], future_times_s)
        speeds = [state["speed_mps"] for state in states]
        mode_summaries.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "probability": float(probabilities[index]),
                "selected": index in selected_indices,
                "speed_range_mps": _range(speeds),
                "trajectory_states": states,
            }
        )

    selected_modes = [mode_summaries[index] for index in selected_indices]
    support: list[dict[str, Any]] = []
    for state_index, time_s in enumerate(np.asarray(future_times_s, dtype=np.float64)):
        points = []
        for mode in selected_modes:
            state = mode["trajectory_states"][state_index]
            points.append(
                {
                    "candidate_id": mode["candidate_id"],
                    "probability": mode["probability"],
                    "lateral_y_m": state["y_m"],
                    "yaw_rad": state["yaw_rad"],
                    "curvature_1pm": state["curvature_1pm"],
                    "speed_mps": state["speed_mps"],
                }
            )
        support.append(
            {
                "time_s": float(time_s),
                "joint_support": points,
                "marginal_bounds": {
                    "lateral_y_m": _range([point["lateral_y_m"] for point in points]),
                    "yaw_rad": _range([point["yaw_rad"] for point in points]),
                    "curvature_1pm": _range([point["curvature_1pm"] for point in points]),
                    "speed_mps": _range([point["speed_mps"] for point in points]),
                },
            }
        )
    region = {
        "representation": "weighted_discrete_joint_support",
        "target_coverage": float(target_coverage),
        "selected_probability_mass": float(sum(mode["probability"] for mode in selected_modes)),
        "selected_mode_ids": [mode["candidate_id"] for mode in selected_modes],
        "joint_lateral_yaw_curvature": support,
        "continuous_support": {
            "representation": "weighted_empirical_trajectory_cloud",
            "num_selected_modes": len(selected_modes),
            "knotwise_quantiles": [
                {
                    "time_s": float(time_s),
                    **{
                        key: {
                            "q05": _weighted_quantile(
                                [float(mode["trajectory_states"][state_index][key]) for mode in selected_modes],
                                [float(mode["probability"]) for mode in selected_modes],
                                0.05,
                            ),
                            "q50": _weighted_quantile(
                                [float(mode["trajectory_states"][state_index][key]) for mode in selected_modes],
                                [float(mode["probability"]) for mode in selected_modes],
                                0.50,
                            ),
                            "q95": _weighted_quantile(
                                [float(mode["trajectory_states"][state_index][key]) for mode in selected_modes],
                                [float(mode["probability"]) for mode in selected_modes],
                                0.95,
                            ),
                        }
                        for key in ("x_m", "y_m", "yaw_rad", "speed_mps", "curvature_1pm")
                    },
                }
                for state_index, time_s in enumerate(np.asarray(future_times_s, dtype=np.float64))
            ],
        },
        "warning": "marginal_bounds are summaries; feasible points are joint_support tuples",
    }
    return mode_summaries, region
