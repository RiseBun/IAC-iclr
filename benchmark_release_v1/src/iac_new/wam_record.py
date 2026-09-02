"""Canonical records for trajectory-image consistency evaluation.

The WAM is only responsible for future images and its action/trajectory
condition. Dataset provenance and logged ego states are evaluation metadata;
keeping them in one strict record prevents accidental state/image misjoins.
"""

from __future__ import annotations

from typing import Any

import numpy as np


REQUIRED_METADATA_FIELDS = (
    "dataset",
    "source_key",
    "scene_name",
    "timestamp_us",
    "history_ego_state",
    "realized_future_ego_state",
)
REQUIRED_SIGNAL_FIELDS = (
    "history_images",
    "future_images",
    "future_times_s",
)


def _state_array(value: Any, field: str) -> np.ndarray:
    states = np.asarray(value, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 5 or states.shape[0] < 1:
        raise ValueError(f"{field} must have shape [T,5] with T >= 1")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"{field} contains non-finite values")
    return states


def _trajectory_array(value: Any, field: str, length: int) -> np.ndarray:
    trajectory = np.asarray(value, dtype=np.float64)
    if trajectory.shape != (length, 3) or not np.all(np.isfinite(trajectory)):
        raise ValueError(f"{field} must have finite shape [{length},3]")
    return trajectory


def validate_trajectory_image_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one strict trajectory-image consistency record.

    ``trajectory`` is the canonical action/trajectory condition. For WAM
    manifests that call it ``action_condition``, the latter is accepted and
    copied into the normalized result as ``condition_trajectory``. The WAM
    is scored from ``future_images``; the condition trajectory is never fed
    back into the image probe. ``realized_future_ego_state`` and
    ``task_success`` are evaluation-only references for realized-state CC/FCS.
    """
    missing = [field for field in REQUIRED_METADATA_FIELDS + REQUIRED_SIGNAL_FIELDS if record.get(field) is None]
    if missing:
        raise ValueError(f"record is missing required fields: {missing}")
    for field in ("dataset", "source_key", "scene_name"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    try:
        timestamp_us = int(record["timestamp_us"])
    except (TypeError, ValueError) as error:
        raise ValueError("timestamp_us must be an integer") from error
    if timestamp_us < 0:
        raise ValueError("timestamp_us must be non-negative")
    history = _state_array(record["history_ego_state"], "history_ego_state")
    realized = _state_array(record["realized_future_ego_state"], "realized_future_ego_state")
    times = np.asarray(record["future_times_s"], dtype=np.float64)
    if times.shape != (realized.shape[0],) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("future_times_s must be finite, increasing, and match realized_future_ego_state")
    history_images = list(record["history_images"])
    future_images = list(record["future_images"])
    if len(history_images) != history.shape[0]:
        raise ValueError("history_images length must match history_ego_state")
    if len(future_images) != realized.shape[0]:
        raise ValueError("future_images length must match realized_future_ego_state")
    if not all(isinstance(path, str) and path for path in history_images + future_images):
        raise ValueError("history_images and future_images must contain non-empty paths")
    trajectory_value = record.get("trajectory", record.get("action_condition"))
    if trajectory_value is None:
        raise ValueError("record needs trajectory or action_condition")
    trajectory = _trajectory_array(trajectory_value, "trajectory/action_condition", realized.shape[0])
    normalized = dict(record)
    normalized.update({
        "timestamp_us": timestamp_us,
        "history_ego_state": history.tolist(),
        "realized_future_ego_state": realized.tolist(),
        "future_times_s": times.tolist(),
        "trajectory": trajectory.tolist(),
        "condition_trajectory": trajectory.tolist(),
        "evaluation_roles": {
            "history_images": "wam_input_context",
            "history_ego_state": "wam_input_context",
            "condition_trajectory": "wam_action_condition",
            "future_images": "wam_output_observation",
            "realized_future_ego_state": "independent_evaluation_reference",
            "task_success": "independent_evaluation_label",
        },
    })
    return normalized
