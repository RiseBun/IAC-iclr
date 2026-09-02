"""Speed-tolerant maneuver skeletons for image-action consistency."""

from __future__ import annotations

from typing import Any

import numpy as np

from .event_posterior import build_event_posterior


def _angle_delta(current: float, previous: float) -> float:
    return float(np.arctan2(np.sin(current - previous), np.cos(current - previous)))


def _interval_curvature(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(trajectory, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError("trajectory must have shape [T,3]")
    previous_xy = np.zeros(2, dtype=np.float64)
    previous_yaw = 0.0
    curvature = []
    distance = []
    for point in points:
        delta_xy = point[:2] - previous_xy
        dist = float(np.linalg.norm(delta_xy))
        curvature.append(_angle_delta(float(point[2]), previous_yaw) / max(dist, 1e-3))
        distance.append(dist)
        previous_xy = point[:2].copy()
        previous_yaw = float(point[2])
    return np.asarray(curvature), np.asarray(distance)


def extract_maneuver(
    trajectory: np.ndarray,
    future_times_s: np.ndarray,
    *,
    curvature_threshold_1pm: float = 0.012,
    heading_threshold_rad: float = 0.028,
    stop_speed_mps: float = 0.75,
    speed_change_mps: float = 0.75,
    lane_change_offset_m: float = 0.75,
    lane_change_heading_rad: float = 0.08,
) -> dict[str, Any]:
    """Extract a speed-tolerant maneuver/event skeleton.

    The output uses event intervals and turn direction, rather than absolute
    x/y points. Direction events are deliberately defined by heading change,
    not curvature per metre: monocular scale error must not turn a nearly
    stationary or straight interval into a turn. Positive yaw change is left
    under the ego convention used by this project; negative is right.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    if times.shape != (len(trajectory),) or len(times) < 1 or np.any(np.diff(times) <= 0.0):
        raise ValueError("future_times_s must be increasing and match trajectory")
    curvature, distances = _interval_curvature(trajectory)
    dt = np.diff(np.concatenate(([0.0], times)))
    speeds = distances / np.maximum(dt, 1e-3)
    yaw = trajectory[:, 2]
    labels: list[str] = []
    heading_changes = np.asarray([
        _angle_delta(float(yaw[index]), float(yaw[index - 1]) if index else 0.0)
        for index in range(len(yaw))
    ])
    for yaw_change in heading_changes:
        if abs(float(yaw_change)) < float(heading_threshold_rad):
            labels.append("straight")
        elif yaw_change > 0.0:
            labels.append("left")
        else:
            labels.append("right")

    # A lane change is a sustained lateral displacement with little yaw.  It
    # is kept separate from turn direction because the two are visually and
    # causally different action classes.
    lateral_change = np.asarray(trajectory[:, 1], dtype=np.float64)
    lane_change = []
    for index, value in enumerate(lateral_change):
        yaw_now = abs(float(trajectory[index, 2]))
        if abs(float(value)) >= float(lane_change_offset_m) and yaw_now <= float(lane_change_heading_rad):
            lane_change.append("lane_change_left" if value > 0.0 else "lane_change_right")
        else:
            lane_change.append("keep_lane")

    longitudinal = []
    for index, speed in enumerate(speeds):
        if speed <= float(stop_speed_mps):
            longitudinal.append("stop")
        elif index == 0:
            longitudinal.append("cruise")
        elif speed - speeds[index - 1] >= float(speed_change_mps):
            longitudinal.append("accelerate")
        elif speed - speeds[index - 1] <= -float(speed_change_mps):
            longitudinal.append("brake")
        else:
            longitudinal.append("cruise")

    # Collapse adjacent equal labels, while retaining the interval support.
    segments: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            segment_labels = labels[start:index]
            segment = {
                "type": segment_labels[0],
                "interval_start": int(start),
                "interval_end": int(index - 1),
                "time_start": float(0.0 if start == 0 else times[start - 1]),
                "time_end": float(times[index - 1]),
                "curvature_min": float(np.min(curvature[start:index])),
                "curvature_max": float(np.max(curvature[start:index])),
            }
            segments.append(segment)
            start = index

    events = []
    for segment in segments:
        if segment["type"] == "straight":
            continue
        start_i = int(segment["interval_start"])
        end_i = int(segment["interval_end"])
        local = heading_changes[start_i : end_i + 1]
        peak_offset = int(np.argmax(np.abs(local)))
        peak_i = start_i + peak_offset
        events.append({
            "type": segment["type"],
            "onset_time": segment["time_start"],
            "peak_time": float(times[peak_i]),
            "exit_time": segment["time_end"],
            "peak_curvature_1pm": float(curvature[peak_i]),
            "cumulative_yaw_rad": float(yaw[end_i]),
        })

    result = {
        "representation": "maneuver-skeleton-v2",
        "segments": segments,
        "segment_types": [segment["type"] for segment in segments],
        "events": events,
        "lateral_action": labels,
        "lane_change_action": lane_change,
        "longitudinal_action": longitudinal,
        # These interval evidence arrays are retained for the event posterior
        # layer. They are not a claim of metric trajectory accuracy.
        "heading_change_rad": heading_changes.tolist(),
        "curvature_1pm": curvature.tolist(),
        "cumulative_yaw_rad": yaw.tolist(),
        "speed_mps": speeds.tolist(),
        "maneuver_class": (
            next((item for item in lane_change if item != "keep_lane"), None)
            or next((item for item in labels if item != "straight"), "keep_lane")
        ),
        "speed_invariant": False,
        "curvature_threshold_1pm": float(curvature_threshold_1pm),
        "heading_threshold_rad": float(heading_threshold_rad),
        "direction_rule": "interval_heading_change",
        "stop_speed_mps": float(stop_speed_mps),
        "speed_change_mps": float(speed_change_mps),
        "lane_change_offset_m": float(lane_change_offset_m),
        "total_progress_m": float(np.sum(distances)),
    }
    result["event_posterior"] = build_event_posterior(result, times)
    result["event_protocol"] = "iac-event-posterior-v1"
    return result


def compare_maneuvers(
    reference: dict[str, Any],
    observed: dict[str, Any],
    *,
    onset_tolerance_s: float = 0.5,
    peak_tolerance_s: float = 0.6,
) -> dict[str, Any]:
    """Compare two maneuver skeletons while allowing speed/distance slack."""
    ref_types = list(reference.get("segment_types", []))
    obs_types = list(observed.get("segment_types", []))
    sequence_match = ref_types == obs_types
    ref_events = list(reference.get("events", []))
    obs_events = list(observed.get("events", []))
    event_rows = []
    for index, ref_event in enumerate(ref_events):
        obs_event = obs_events[index] if index < len(obs_events) else None
        if obs_event is None:
            event_rows.append({"type_match": False, "onset_match": False, "peak_match": False})
            continue
        event_rows.append({
            "type_match": ref_event["type"] == obs_event.get("type"),
            "onset_match": abs(float(ref_event["onset_time"]) - float(obs_event.get("onset_time", 1e9))) <= float(onset_tolerance_s),
            "peak_match": abs(float(ref_event["peak_time"]) - float(obs_event.get("peak_time", 1e9))) <= float(peak_tolerance_s),
            "reference": ref_event,
            "observed": obs_event,
        })
    if not ref_events and not obs_events:
        event_score = 1.0
    elif len(ref_events) != len(obs_events):
        event_score = 0.0
    else:
        event_score = float(np.mean([
            float(row["type_match"]) * (0.5 * float(row["onset_match"]) + 0.5 * float(row["peak_match"]))
            for row in event_rows
        ]))
    score = float(0.5 * float(sequence_match) + 0.5 * event_score)
    return {
        "score": score,
        "sequence_match": bool(sequence_match),
        "event_score": event_score,
        "event_matches": event_rows,
        "speed_distance_relaxed": True,
    }
