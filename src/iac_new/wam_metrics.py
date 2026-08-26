"""Joint future-action metrics for paired WAM counterfactual evaluation.

The functions in this module intentionally do not consume image-flow energy or
the IAC posterior. A WAM pair contains the same history/task under two future
conditions, the model-imagined future, the executed action trajectory, and an
optional task-success label. This keeps the causal comparison outside the
image-side probe being evaluated.
"""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_SCALES = {
    "lateral_y_m": 0.50,
    "yaw_rad": 0.10,
    "speed_mps": 2.0,
    "curvature_1pm": 0.06,
}

DEFAULT_EGO_STATE_SCALES = {
    "x_m": 2.0,
    "y_m": 0.50,
    "yaw_rad": 0.10,
    "speed_mps": 2.0,
    "yaw_rate_rps": 0.10,
}


def _states(trajectory: np.ndarray, future_times_s: np.ndarray) -> np.ndarray:
    """Return [T,4] of lateral, yaw, speed, curvature."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape [T,3]")
    if times.shape != (trajectory.shape[0],) or np.any(np.diff(times) <= 0.0):
        raise ValueError("future_times_s must be increasing and match trajectory")
    result = np.zeros((trajectory.shape[0], 4), dtype=np.float64)
    previous_xy = np.zeros(2, dtype=np.float64)
    previous_yaw = 0.0
    previous_time = 0.0
    for index, (point, time_s) in enumerate(zip(trajectory, times)):
        dt = float(time_s - previous_time)
        displacement = point[:2] - previous_xy
        distance = float(np.linalg.norm(displacement))
        yaw_delta = float(np.arctan2(np.sin(point[2] - previous_yaw), np.cos(point[2] - previous_yaw)))
        result[index] = [point[1], point[2], distance / dt, yaw_delta / max(distance, 1e-3)]
        previous_xy = point[:2].copy()
        previous_yaw = float(point[2])
        previous_time = float(time_s)
    return result


def trajectory_descriptor(
    trajectory: np.ndarray,
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
) -> np.ndarray:
    """Flatten a trajectory into normalized motion dimensions."""
    scales = {**DEFAULT_SCALES, **(scales or {})}
    values = _states(trajectory, future_times_s)
    divisors = np.asarray([
        scales["lateral_y_m"],
        scales["yaw_rad"],
        scales["speed_mps"],
        scales["curvature_1pm"],
    ], dtype=np.float64)
    if np.any(divisors <= 0.0) or not np.all(np.isfinite(divisors)):
        raise ValueError("descriptor scales must be positive and finite")
    return (values / divisors).reshape(-1)


def ego_state_trajectory(trajectory: np.ndarray, future_times_s: np.ndarray) -> np.ndarray:
    """Return [x, y, yaw, speed, yaw_rate] without camera calibration."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape [T,3]")
    if times.shape != (trajectory.shape[0],) or np.any(np.diff(times) <= 0.0):
        raise ValueError("future_times_s must be increasing and match trajectory")
    states = np.zeros((trajectory.shape[0], 5), dtype=np.float64)
    previous = np.zeros(3, dtype=np.float64)
    previous_time = 0.0
    for index, (point, time_s) in enumerate(zip(trajectory, times)):
        dt = float(time_s - previous_time)
        displacement = point[:2] - previous[:2]
        speed = float(np.linalg.norm(displacement) / dt)
        yaw_delta = float(np.arctan2(np.sin(point[2] - previous[2]), np.cos(point[2] - previous[2])))
        states[index] = [point[0], point[1], point[2], speed, yaw_delta / dt]
        previous = point.copy()
        previous_time = float(time_s)
    return states


def ego_state_descriptor(
    trajectory: np.ndarray,
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
) -> np.ndarray:
    """Flatten a trajectory into normalized ego-state dimensions."""
    scales = {**DEFAULT_EGO_STATE_SCALES, **(scales or {})}
    values = ego_state_trajectory(trajectory, future_times_s)
    divisors = np.asarray(
        [scales["x_m"], scales["y_m"], scales["yaw_rad"], scales["speed_mps"], scales["yaw_rate_rps"]],
        dtype=np.float64,
    )
    if np.any(divisors <= 0.0) or not np.all(np.isfinite(divisors)):
        raise ValueError("ego-state descriptor scales must be positive and finite")
    return (values / divisors).reshape(-1)


def normalized_ego_state_distance(
    first: np.ndarray,
    second: np.ndarray,
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
) -> float:
    """RMS normalized distance in physical ego-state space."""
    left = ego_state_descriptor(first, future_times_s, scales=scales)
    right = ego_state_descriptor(second, future_times_s, scales=scales)
    return float(np.sqrt(np.mean(np.square(left - right))))


def normalized_ego_state_support_distance(
    trajectory: np.ndarray,
    support: list[dict[str, Any]],
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
) -> float:
    """Distance from a trajectory to an image-supported ego-state tube.

    ``support`` contains q05/q95 boxes for x, y and yaw at each knot. A point
    inside a box has zero distance for that dimension; only the normalized
    amount outside the box contributes. This is intentionally set-valued: a
    trajectory can be consistent without matching the decoder median exactly.
    Speed and yaw-rate are omitted because the monocular image-side estimate
    does not reliably observe metric speed.
    """
    values = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(support) != len(values):
        raise ValueError("trajectory and support knots must have matching [T,3] shapes")
    scales = {**DEFAULT_EGO_STATE_SCALES, **(scales or {})}
    divisors = np.asarray([scales["x_m"], scales["y_m"], scales["yaw_rad"]], dtype=np.float64)
    if np.any(divisors <= 0.0) or not np.all(np.isfinite(divisors)):
        raise ValueError("support scales must be positive and finite")
    errors = []
    for point, item in zip(values, support):
        bounds = np.asarray([
            [float(item["x_m"]["q05"]), float(item["x_m"]["q95"])],
            [float(item["y_m"]["q05"]), float(item["y_m"]["q95"])],
            [float(item["yaw_rad"]["q05"]), float(item["yaw_rad"]["q95"])],
        ], dtype=np.float64)
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        # Yaw support intervals are local (< pi); unwrap the point to the
        # nearest equivalent angle before measuring distance to the interval.
        yaw_mid = 0.5 * (lower[2] + upper[2])
        yaw = point[2] + 2.0 * np.pi * np.round((yaw_mid - point[2]) / (2.0 * np.pi))
        point_values = np.asarray([point[0], point[1], yaw], dtype=np.float64)
        outside = np.maximum(lower - point_values, 0.0) + np.minimum(upper - point_values, 0.0)
        errors.append(outside / divisors)
    return float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=np.float64)))))


def ego_state_action_compatibility(
    imagined_future: np.ndarray,
    executed_action: np.ndarray,
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
) -> float:
    """Foresight/action compatibility using only ego-state trajectories."""
    return float(np.exp(-normalized_ego_state_distance(
        imagined_future, executed_action, future_times_s, scales=scales
    )))


def ego_state_response_alignment(
    imagined_future_a: np.ndarray,
    imagined_future_b: np.ndarray,
    action_condition_a: np.ndarray,
    action_condition_b: np.ndarray,
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
    minimum_intervention_norm: float = 1e-3,
) -> dict[str, float | bool | None]:
    """Align paired image-inferred state changes with action interventions."""
    future_delta = (
        ego_state_descriptor(imagined_future_b, future_times_s, scales=scales)
        - ego_state_descriptor(imagined_future_a, future_times_s, scales=scales)
    )
    action_delta = (
        ego_state_descriptor(action_condition_b, future_times_s, scales=scales)
        - ego_state_descriptor(action_condition_a, future_times_s, scales=scales)
    )
    future_norm = float(np.linalg.norm(future_delta))
    action_norm = float(np.linalg.norm(action_delta))
    if action_norm < float(minimum_intervention_norm):
        return {
            "evaluable": False,
            "degenerate_intervention": True,
            "future_delta_norm": future_norm,
            "action_delta_norm": action_norm,
            "cosine_alignment": None,
            "response_sensitivity": None,
            "alignment_score": None,
        }
    cosine = 0.0 if future_norm < float(minimum_intervention_norm) else float(
        np.dot(future_delta, action_delta) / (future_norm * action_norm)
    )
    sensitivity = float(min(1.0, future_norm / action_norm))
    return {
        "evaluable": True,
        "degenerate_intervention": False,
        "future_delta_norm": future_norm,
        "action_delta_norm": action_norm,
        "cosine_alignment": cosine,
        "response_sensitivity": sensitivity,
        "alignment_score": float(max(0.0, cosine) * sensitivity),
    }


def realized_state_counterfactual_consistency(
    imagined_future_a: np.ndarray,
    imagined_future_b: np.ndarray,
    realized_future_a: np.ndarray,
    realized_future_b: np.ndarray,
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare imagined futures with independently logged future ego states."""
    compatibility = [
        ego_state_action_compatibility(imagined_future_a, realized_future_a, future_times_s, scales=scales),
        ego_state_action_compatibility(imagined_future_b, realized_future_b, future_times_s, scales=scales),
    ]
    response = ego_state_response_alignment(
        imagined_future_a,
        imagined_future_b,
        realized_future_a,
        realized_future_b,
        future_times_s,
        scales=scales,
    )
    mean_compatibility = float(np.mean(compatibility))
    alignment = response["alignment_score"]
    joint = None if alignment is None else float(np.sqrt(max(0.0, mean_compatibility * float(alignment))))
    return {
        "branch_compatibility": compatibility,
        "mean_foresight_realized_state_compatibility": mean_compatibility,
        "realized_state_response_alignment": response,
        "realized_state_counterfactual_consistency": joint,
    }


def normalized_trajectory_distance(
    first: np.ndarray,
    second: np.ndarray,
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
) -> float:
    """RMS normalized distance between a future and an executed trajectory."""
    left = trajectory_descriptor(first, future_times_s, scales=scales)
    right = trajectory_descriptor(second, future_times_s, scales=scales)
    if left.shape != right.shape:
        raise ValueError("trajectories must have matching shapes")
    return float(np.sqrt(np.mean(np.square(left - right))))


def foresight_action_compatibility(
    imagined_future: np.ndarray,
    executed_action: np.ndarray,
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
) -> float:
    """Continuous compatibility in (0,1], independent of image scoring."""
    distance = normalized_trajectory_distance(
        imagined_future, executed_action, future_times_s, scales=scales
    )
    return float(np.exp(-distance))


def counterfactual_response_alignment(
    imagined_future_a: np.ndarray,
    imagined_future_b: np.ndarray,
    executed_action_a: np.ndarray,
    executed_action_b: np.ndarray,
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
    minimum_intervention_norm: float = 1e-3,
) -> dict[str, float | bool | None]:
    """Measure whether action changes follow paired imagined-future changes.

    The score is directional cosine alignment multiplied by a bounded response
    sensitivity. A model that emits the same action for both futures therefore
    receives zero, even if the cosine is undefined. Pairs with no meaningful
    future intervention are marked non-evaluable.
    """
    future_a = trajectory_descriptor(imagined_future_a, future_times_s, scales=scales)
    future_b = trajectory_descriptor(imagined_future_b, future_times_s, scales=scales)
    action_a = trajectory_descriptor(executed_action_a, future_times_s, scales=scales)
    action_b = trajectory_descriptor(executed_action_b, future_times_s, scales=scales)
    future_delta = future_b - future_a
    action_delta = action_b - action_a
    future_norm = float(np.linalg.norm(future_delta))
    action_norm = float(np.linalg.norm(action_delta))
    if future_norm < float(minimum_intervention_norm):
        return {
            "evaluable": False,
            "degenerate_intervention": True,
            "future_delta_norm": future_norm,
            "action_delta_norm": action_norm,
            "cosine_alignment": None,
            "response_sensitivity": None,
            "alignment_score": None,
        }
    if action_norm < float(minimum_intervention_norm):
        cosine = 0.0
    else:
        cosine = float(np.dot(future_delta, action_delta) / (future_norm * action_norm))
    sensitivity = float(min(1.0, action_norm / future_norm))
    alignment = float(max(0.0, cosine) * sensitivity)
    return {
        "evaluable": True,
        "degenerate_intervention": False,
        "future_delta_norm": future_norm,
        "action_delta_norm": action_norm,
        "cosine_alignment": cosine,
        "response_sensitivity": sensitivity,
        "alignment_score": alignment,
    }


def paired_counterfactual_consistency(
    branch_a: dict[str, Any],
    branch_b: dict[str, Any],
    future_times_s: np.ndarray,
    *,
    scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return the joint paired CC score and all auditable components."""
    required = ("imagined_future", "executed_action")
    for branch in (branch_a, branch_b):
        missing = [key for key in required if branch.get(key) is None]
        if missing:
            raise ValueError(f"paired branch is missing {missing}")
    compatibility = [
        foresight_action_compatibility(
            np.asarray(branch["imagined_future"]),
            np.asarray(branch["executed_action"]),
            future_times_s,
            scales=scales,
        )
        for branch in (branch_a, branch_b)
    ]
    state_compatibility = [
        ego_state_action_compatibility(
            np.asarray(branch["imagined_future"]),
            np.asarray(branch["executed_action"]),
            future_times_s,
        )
        for branch in (branch_a, branch_b)
    ]
    state_response = ego_state_response_alignment(
        np.asarray(branch_a["imagined_future"]),
        np.asarray(branch_b["imagined_future"]),
        np.asarray(branch_a["executed_action"]),
        np.asarray(branch_b["executed_action"]),
        future_times_s,
    )
    response = counterfactual_response_alignment(
        np.asarray(branch_a["imagined_future"]),
        np.asarray(branch_b["imagined_future"]),
        np.asarray(branch_a["executed_action"]),
        np.asarray(branch_b["executed_action"]),
        future_times_s,
        scales=scales,
    )
    compatibility_mean = float(np.mean(compatibility))
    state_compatibility_mean = float(np.mean(state_compatibility))
    alignment = response["alignment_score"]
    joint = None if alignment is None else float(np.sqrt(max(0.0, compatibility_mean * float(alignment))))
    state_alignment = state_response["alignment_score"]
    state_joint = None if state_alignment is None else float(
        np.sqrt(max(0.0, state_compatibility_mean * float(state_alignment)))
    )
    return {
        "branch_compatibility": compatibility,
        "mean_foresight_action_compatibility": compatibility_mean,
        "branch_ego_state_compatibility": state_compatibility,
        "mean_ego_state_action_compatibility": state_compatibility_mean,
        "ego_state_response_alignment": state_response,
        "ego_state_counterfactual_consistency": state_joint,
        "response_alignment": response,
        "counterfactual_consistency": joint,
    }


def foresight_conditioned_success(
    branches: list[dict[str, Any]],
    *,
    compatibility_threshold: float = 0.70,
    scales: dict[str, float] | None = None,
    future_times_s: np.ndarray,
    allow_action_proxy: bool = False,
) -> dict[str, Any]:
    """Measure task success conditional on realizing the imagined future.

    Planned/executed actions are not realized state. The optional proxy exists
    only for backward-compatible diagnostics and is labelled in the result.
    """
    if not branches:
        raise ValueError("at least one branch is required")
    scores = []
    successes = []
    reference_kinds = []
    for branch in branches:
        reference = branch.get("realized_future")
        reference_kind = "realized_state"
        if reference is None and allow_action_proxy:
            reference = branch.get("executed_action")
            reference_kind = "action_proxy"
        if reference is None or branch.get("imagined_future") is None:
            raise ValueError("every branch needs imagined_future and realized_future for FCS")
        if branch.get("task_success") is None:
            raise ValueError("every branch needs task_success for FCS")
        compatibility = ego_state_action_compatibility(
            np.asarray(branch["imagined_future"]),
            np.asarray(reference),
            future_times_s,
            scales=scales,
        )
        scores.append(compatibility)
        successes.append(bool(branch["task_success"]))
        reference_kinds.append(reference_kind)
    compatible = np.asarray(scores) >= float(compatibility_threshold)
    success_array = np.asarray(successes, dtype=np.float64)
    return {
        "compatibility_threshold": float(compatibility_threshold),
        "branches": len(branches),
        "compatible_branches": int(compatible.sum()),
        "reference_kind": "realized_state" if set(reference_kinds) == {"realized_state"} else "action_proxy",
        "foresight_conditioned_success": float(success_array[compatible].mean()) if compatible.any() else None,
        "unconditional_success": float(success_array.mean()),
        "success_lift": float(success_array[compatible].mean() - success_array.mean()) if compatible.any() else None,
        "success_compatibility_product": float(np.mean(success_array * np.asarray(scores))),
    }
