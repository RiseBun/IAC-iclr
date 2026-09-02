"""Event-level image/action support derived from the existing maneuver evidence.

This module deliberately keeps the output set-valued.  A trajectory is used
only as an ego-frame evidence source; it is not treated as the unique answer.
Road-topology events require independent road-structure evidence and otherwise
remain ``unknown``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


LATERAL_EVENTS = (
    "keep_lane",
    "turn_left",
    "turn_right",
    "lane_change_left",
    "lane_change_right",
    "u_turn",
)
LONGITUDINAL_EVENTS = (
    "cruise",
    "accelerate",
    "brake",
    "stop",
    "yield",
    "start",
)
ROAD_EVENTS = (
    "intersection",
    "roundabout",
    "merge",
    "split",
    "obstacle_avoidance",
    "unknown",
)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(float(value), low, high))


def _normalise(scores: Mapping[str, float], labels: Sequence[str]) -> dict[str, float]:
    values = np.asarray([max(0.0, float(scores.get(label, 0.0))) for label in labels], dtype=np.float64)
    total = float(values.sum())
    if total <= 1e-12:
        values[:] = 1.0 / max(len(labels), 1)
    else:
        values /= total
    return {label: float(value) for label, value in zip(labels, values)}


def _lateral_distribution(
    lateral: str,
    heading_change_rad: float,
    heading_threshold_rad: float,
    *,
    lane_change: str,
    cumulative_yaw_rad: float,
) -> tuple[dict[str, float], float, list[str]]:
    """Return a conservative lateral posterior and its confidence."""
    magnitude = abs(float(heading_change_rad))
    threshold = max(float(heading_threshold_rad), 1e-6)
    # A full U-turn is a separate event, not a very large ordinary turn.
    if abs(float(cumulative_yaw_rad)) >= 0.75 * np.pi:
        return _normalise({"u_turn": 1.0, "keep_lane": 0.02}, LATERAL_EVENTS), 0.92, ["u_turn"]
    if lane_change in {"lane_change_left", "lane_change_right"}:
        # Lane changes can have a small heading transient. Keep a turn as a
        # possible explanation when the heading evidence is not decisive.
        target = lane_change
        confidence = _clip(0.55 + magnitude / max(4.0 * threshold, 1e-6), 0.55, 0.95)
        scores = {target: confidence, "keep_lane": 1.0 - confidence}
        if heading_change_rad > threshold:
            scores["turn_left"] = 0.12 * confidence
        elif heading_change_rad < -threshold:
            scores["turn_right"] = 0.12 * confidence
        allowed = [target, "keep_lane"]
        return _normalise(scores, LATERAL_EVENTS), confidence, allowed
    if lateral == "left":
        confidence = _clip(0.60 + (magnitude - threshold) / max(4.0 * threshold, 1e-6), 0.60, 0.98)
        scores = {"turn_left": confidence, "keep_lane": 1.0 - confidence}
        allowed = ["turn_left"]
        if confidence < 0.82:
            allowed.append("keep_lane")
        return _normalise(scores, LATERAL_EVENTS), confidence, allowed
    if lateral == "right":
        confidence = _clip(0.60 + (magnitude - threshold) / max(4.0 * threshold, 1e-6), 0.60, 0.98)
        scores = {"turn_right": confidence, "keep_lane": 1.0 - confidence}
        allowed = ["turn_right"]
        if confidence < 0.82:
            allowed.append("keep_lane")
        return _normalise(scores, LATERAL_EVENTS), confidence, allowed
    # Straight/near-threshold intervals retain both hypotheses.
    confidence = _clip(1.0 - magnitude / threshold, 0.45, 0.90)
    scores = {"keep_lane": confidence}
    if heading_change_rad >= 0.0:
        scores["turn_left"] = 1.0 - confidence
    else:
        scores["turn_right"] = 1.0 - confidence
    allowed = ["keep_lane", "turn_left" if heading_change_rad >= 0.0 else "turn_right"]
    return _normalise(scores, LATERAL_EVENTS), confidence, allowed


def _longitudinal_distribution(longitudinal: str, speed_observability: float) -> tuple[dict[str, float], float, list[str]]:
    """Keep longitudinal estimates explicitly lower confidence than turns."""
    confidence = _clip(float(speed_observability) * (0.90 if longitudinal in {"stop", "start"} else 0.65), 0.20, 0.90)
    if longitudinal == "cruise":
        scores = {"cruise": confidence, "accelerate": (1.0 - confidence) * 0.5, "brake": (1.0 - confidence) * 0.5}
        allowed = ["cruise", "accelerate", "brake"]
    elif longitudinal == "stop":
        scores = {"stop": confidence, "brake": 1.0 - confidence}
        allowed = ["stop", "brake"]
    elif longitudinal == "brake":
        scores = {"brake": confidence, "stop": (1.0 - confidence) * 0.7, "cruise": (1.0 - confidence) * 0.3}
        allowed = ["brake", "stop", "cruise"]
    elif longitudinal == "accelerate":
        scores = {"accelerate": confidence, "start": (1.0 - confidence) * 0.5, "cruise": (1.0 - confidence) * 0.5}
        allowed = ["accelerate", "start", "cruise"]
    else:
        scores = {"cruise": 0.5, "brake": 0.5}
        allowed = ["cruise", "brake"]
    return _normalise(scores, LONGITUDINAL_EVENTS), confidence, allowed


def build_event_posterior(
    maneuver: Mapping[str, Any],
    times_s: Sequence[float],
    *,
    observability: Sequence[float] | None = None,
    speed_observability: Sequence[float] | None = None,
    topology_evidence: Sequence[Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Convert an existing maneuver skeleton into interval event posteriors.

    ``topology_evidence`` is optional by design.  Each item may contain scores
    for names in :data:`ROAD_EVENTS`; without it, topology is ``unknown``.
    """
    lateral = list(maneuver.get("lateral_action", []))
    lane = list(maneuver.get("lane_change_action", []))
    longitudinal = list(maneuver.get("longitudinal_action", []))
    heading_changes = list(maneuver.get("heading_change_rad", []))
    cumulative = list(maneuver.get("cumulative_yaw_rad", []))
    curvature = list(maneuver.get("curvature_1pm", []))
    times = np.asarray(times_s, dtype=np.float64)
    count = len(lateral)
    if len(times) != count:
        raise ValueError("times_s must match maneuver interval count")
    obs = list(observability or [1.0] * count)
    speed_obs = list(speed_observability or obs)
    topology = list(topology_evidence or [{} for _ in range(count)])
    if not (len(lane) == len(longitudinal) == len(obs) == len(speed_obs) == len(topology) == count):
        raise ValueError("all interval evidence arrays must have matching length")
    threshold = float(maneuver.get("heading_threshold_rad", 0.028))
    output: list[dict[str, Any]] = []
    for index in range(count):
        dyaw = float(heading_changes[index]) if index < len(heading_changes) else 0.0
        cum_yaw = float(cumulative[index]) if index < len(cumulative) else dyaw
        lateral_prob, direction_conf, allowed_lateral = _lateral_distribution(
            lateral[index], dyaw, threshold, lane_change=lane[index], cumulative_yaw_rad=cum_yaw
        )
        longitudinal_prob, longitudinal_conf, allowed_longitudinal = _longitudinal_distribution(
            longitudinal[index], float(speed_obs[index])
        )
        evidence = {str(key): _clip(value) for key, value in topology[index].items() if key in ROAD_EVENTS and key != "unknown"}
        if evidence:
            road_prob = _normalise({**evidence, "unknown": max(0.0, 1.0 - max(evidence.values()))}, ROAD_EVENTS)
            road_event = max(road_prob, key=road_prob.get)
        else:
            road_prob = _normalise({"unknown": 1.0}, ROAD_EVENTS)
            road_event = "unknown"
        curvature_value = float(curvature[index]) if index < len(curvature) else 0.0
        output.append({
            "interval_index": index,
            "event_interval": [float(0.0 if index == 0 else times[index - 1]), float(times[index])],
            "lateral_event": max(lateral_prob, key=lateral_prob.get),
            "longitudinal_event": max(longitudinal_prob, key=longitudinal_prob.get),
            "road_event": road_event,
            "lateral_posterior": lateral_prob,
            "longitudinal_posterior": longitudinal_prob,
            "road_posterior": road_prob,
            "allowed_events": sorted(set(allowed_lateral + allowed_longitudinal + [road_event])),
            "direction_confidence": direction_conf,
            "curvature_confidence": _clip(0.5 * float(obs[index]) + 0.5 * direction_conf),
            "observability": _clip(obs[index]),
            "speed_observability": _clip(speed_obs[index]),
            "abstain": bool(float(obs[index]) < 0.25),
        })
    return output
