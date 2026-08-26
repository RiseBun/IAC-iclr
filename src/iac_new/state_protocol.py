"""Dataset-neutral ego-state adapters for the WAM evaluation protocol.

The adapters intentionally accept plain dictionaries so the evaluator does
not depend on a particular NuPlan/NavSim/Waymo runtime. Dataset loaders may
export their native records once, then this module provides the same
calibration-free state contract to all three datasets.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


STATE_FIELDS = ("x_m", "y_m", "yaw_rad", "speed_mps", "yaw_rate_rps")


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def _relative_pose(global_poses: np.ndarray, anchor_index: int) -> np.ndarray:
    """Convert [x,y,yaw] global poses into an anchor ego frame."""
    poses = np.asarray(global_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("global_poses must have shape [T,3]")
    anchor = poses[anchor_index]
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    rotation_t = np.asarray([[c, s], [-s, c]], dtype=np.float64)
    xy = (poses[:, :2] - anchor[:2]) @ rotation_t.T
    yaw = np.asarray([_wrap(value - anchor[2]) for value in poses[:, 2]], dtype=np.float64)
    return np.column_stack([xy, yaw])


def _states_from_poses(
    poses: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    speeds: np.ndarray | None = None,
    yaw_rates: np.ndarray | None = None,
    anchor_index: int,
) -> np.ndarray:
    relative = _relative_pose(poses, anchor_index)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if times.shape != (len(relative),) or np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps_s must be increasing and match poses")
    if speeds is None:
        speeds = np.zeros(len(relative), dtype=np.float64)
        for index in range(1, len(relative)):
            dt = max(float(times[index] - times[index - 1]), 1e-6)
            speeds[index] = np.linalg.norm(poses[index, :2] - poses[index - 1, :2]) / dt
        speeds[0] = speeds[1] if len(speeds) > 1 else 0.0
    if yaw_rates is None:
        yaw_rates = np.zeros(len(relative), dtype=np.float64)
        for index in range(1, len(relative)):
            dt = max(float(times[index] - times[index - 1]), 1e-6)
            yaw_rates[index] = _wrap(poses[index, 2] - poses[index - 1, 2]) / dt
        yaw_rates[0] = yaw_rates[1] if len(yaw_rates) > 1 else 0.0
    return np.column_stack([relative, np.asarray(speeds), np.asarray(yaw_rates)])


def canonical_states_from_pose_arrays(
    poses: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    anchor_index: int,
    speeds: np.ndarray | None = None,
    yaw_rates: np.ndarray | None = None,
) -> np.ndarray:
    """Public adapter for simulator/runtime pose arrays."""
    return _states_from_poses(
        poses, timestamps_s, speeds=speeds, yaw_rates=yaw_rates, anchor_index=anchor_index
    )


def nuplan_states(rows: Iterable[dict[str, Any]], *, anchor_index: int) -> np.ndarray:
    """Adapt NuPlan ``ego_pose`` exports to the canonical state sequence."""
    records = list(rows)
    poses = np.asarray(
        [[float(row["x"]), float(row["y"]), _yaw_from_quaternion(
            float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])
        )] for row in records], dtype=np.float64)
    timestamps = np.asarray([float(row["timestamp"]) * 1e-6 for row in records], dtype=np.float64)
    speeds = np.asarray([
        np.linalg.norm([float(row.get("vx", 0.0)), float(row.get("vy", 0.0))])
        for row in records
    ], dtype=np.float64)
    yaw_rates = np.asarray([float(row.get("angular_rate_z", 0.0)) for row in records], dtype=np.float64)
    return _states_from_poses(poses, timestamps, speeds=speeds, yaw_rates=yaw_rates, anchor_index=anchor_index)


def _matrix_from_value(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.size == 16:
        return matrix.reshape(4, 4)
    raise ValueError("ego2global must be a 4x4 matrix")


def navsim_states(records: Iterable[dict[str, Any]], *, anchor_index: int) -> np.ndarray:
    """Adapt NAVSIM scene records (``ego2global`` + dynamic state)."""
    records = list(records)
    poses_global = []
    speeds = []
    for record in records:
        pose = _matrix_from_value(record["ego2global"])
        poses_global.append([pose[0, 3], pose[1, 3], math.atan2(pose[1, 0], pose[0, 0])])
        dynamic = np.asarray(record.get("ego_dynamic_state") or [], dtype=np.float64)
        speeds.append(float(np.linalg.norm(dynamic[:2])) if dynamic.size >= 2 else 0.0)
    timestamps = np.asarray([
        float(record.get("timestamp", index)) * (1e-6 if float(record.get("timestamp", index)) > 1e9 else 1.0)
        for index, record in enumerate(records)
    ], dtype=np.float64)
    return _states_from_poses(
        np.asarray(poses_global), timestamps, speeds=np.asarray(speeds), anchor_index=anchor_index
    )


def waymo_states(rows: Iterable[dict[str, Any]], *, anchor_index: int) -> np.ndarray:
    """Adapt a lightweight Waymo export without importing TensorFlow."""
    records = list(rows)
    poses = []
    speeds = []
    yaw_rates = []
    timestamps = []
    for index, row in enumerate(records):
        if row.get("pose") is not None:
            pose = _matrix_from_value(row["pose"])
            poses.append([pose[0, 3], pose[1, 3], math.atan2(pose[1, 0], pose[0, 0])])
        else:
            poses.append([float(row["x"]), float(row["y"]), float(row.get("heading", row.get("yaw", 0.0)))])
        velocity = row.get("velocity") or row.get("linear_velocity") or [row.get("vx", 0.0), row.get("vy", 0.0)]
        speeds.append(float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)[:2])))
        yaw_rates.append(float(row.get("yaw_rate", row.get("angular_velocity", 0.0))))
        timestamps.append(float(row.get("timestamp", index)))
    return _states_from_poses(
        np.asarray(poses), np.asarray(timestamps), speeds=np.asarray(speeds),
        yaw_rates=np.asarray(yaw_rates), anchor_index=anchor_index
    )


def states_to_trajectory(states: np.ndarray) -> np.ndarray:
    """Drop speed/yaw-rate columns for the trajectory metric API."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("states must have shape [T,5]")
    return values[:, :3].copy()


def task_success_from_label(row: dict[str, Any], *, field: str = "task_success") -> bool | None:
    """Read an explicit task-success label; never infer it from image quality."""
    value = row.get(field)
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"success", "successful", "true", "1", "pass", "passed"}:
            return True
        if normalized in {"failure", "failed", "false", "0", "fail"}:
            return False
        raise ValueError(f"unknown task success label: {value}")
    return bool(value)


def navsim_pdm_success(score: float | None, *, threshold: float = 0.50) -> bool | None:
    """Convert an explicitly reported NAVSIM PDM score to a task label."""
    if score is None or not np.isfinite(float(score)):
        return None
    return bool(float(score) >= float(threshold))
