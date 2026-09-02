"""Independent trajectory support and counterfactual consistency utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


_ACCEPTABLE_LABELS = frozenset({"known_valid", "plausible"})


def _trajectory_states(trajectory: np.ndarray, future_times_s: np.ndarray) -> np.ndarray:
    """Return [T,5]: x, y, yaw, speed, curvature."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape [T,3]")
    if times.shape != (trajectory.shape[0],) or np.any(np.diff(times) <= 0.0):
        raise ValueError("future_times_s must be increasing and match trajectory")
    states = np.zeros((trajectory.shape[0], 5), dtype=np.float64)
    previous = np.zeros(2, dtype=np.float64)
    previous_yaw = 0.0
    previous_time = 0.0
    for index, (point, time_s) in enumerate(zip(trajectory, times)):
        dt = float(time_s - previous_time)
        displacement = point[:2] - previous
        distance = float(np.linalg.norm(displacement))
        motion_yaw = float(np.arctan2(displacement[1], displacement[0])) if distance > 1e-8 else previous_yaw
        yaw_delta = float(np.arctan2(np.sin(point[2] - previous_yaw), np.cos(point[2] - previous_yaw)))
        states[index] = [point[0], point[1], point[2], distance / dt, yaw_delta / max(distance, 1e-3)]
        previous = point[:2].copy()
        previous_yaw = float(point[2])
        previous_time = float(time_s)
    return states


def independent_support_mask(
    candidates: list[dict[str, Any]],
    gt_candidate_id: str,
    future_times_s: np.ndarray,
    *,
    lateral_tolerance_m: float = 0.50,
    yaw_tolerance_rad: float = 0.10,
    speed_relative_tolerance: float = 0.20,
    curvature_tolerance_1pm: float = 0.06,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a support tube around the logged trajectory, without model scores.

    The support is a kinematic neighborhood of the logged action. It is used
    only after scoring, so it cannot be optimized by the image-flow energy.
    """
    gt = next((c for c in candidates if str(c["candidate_id"]) == str(gt_candidate_id)), None)
    if gt is None:
        raise ValueError(f"ground-truth candidate {gt_candidate_id} is absent")
    gt_states = _trajectory_states(np.asarray(gt["trajectory"]), future_times_s)
    support = []
    distances = []
    for candidate in candidates:
        states = _trajectory_states(np.asarray(candidate["trajectory"]), future_times_s)
        lateral = np.max(np.abs(states[:, 1] - gt_states[:, 1]))
        yaw = np.max(np.abs(np.arctan2(np.sin(states[:, 2] - gt_states[:, 2]), np.cos(states[:, 2] - gt_states[:, 2]))))
        speed_rel = np.max(np.abs(states[:, 3] - gt_states[:, 3]) / np.maximum(gt_states[:, 3], 1.0))
        curvature = np.max(np.abs(states[:, 4] - gt_states[:, 4]))
        normalized = max(
            float(lateral / max(lateral_tolerance_m, 1e-6)),
            float(yaw / max(yaw_tolerance_rad, 1e-6)),
            float(speed_rel / max(speed_relative_tolerance, 1e-6)),
            float(curvature / max(curvature_tolerance_1pm, 1e-6)),
        )
        support.append(normalized <= 1.0)
        distances.append(normalized)
    return np.asarray(support, dtype=bool), {
        "definition": "logged_trajectory_kinematic_tube",
        "tolerances": {
            "lateral_tolerance_m": float(lateral_tolerance_m),
            "yaw_tolerance_rad": float(yaw_tolerance_rad),
            "speed_relative_tolerance": float(speed_relative_tolerance),
            "curvature_tolerance_1pm": float(curvature_tolerance_1pm),
        },
        "normalized_distance_by_candidate": {
            str(candidate["candidate_id"]): float(distance)
            for candidate, distance in zip(candidates, distances)
        },
    }


def classify_trajectory_candidates(
    candidates: list[dict[str, Any]],
    gt_candidate_id: str,
    future_times_s: np.ndarray,
    *,
    lateral_tolerance_m: float = 0.75,
    yaw_tolerance_rad: float = 0.14,
    speed_relative_tolerance: float = 0.25,
    curvature_tolerance_1pm: float = 0.08,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign auditable multi-label feasibility classes to a candidate bank.

    This is deliberately conservative. The logged trajectory is the only
    ``known_valid`` sample. Nearby candidates that satisfy an independent
    kinematic tube are ``plausible``; candidates outside that tube remain
    ``unknown`` unless an upstream dataset oracle explicitly marks them as
    invalid. Unknown is not a failure label: front-view images alone cannot
    certify collision freedom or task success.
    """
    support, metadata = independent_support_mask(
        candidates,
        gt_candidate_id,
        future_times_s,
        lateral_tolerance_m=lateral_tolerance_m,
        yaw_tolerance_rad=yaw_tolerance_rad,
        speed_relative_tolerance=speed_relative_tolerance,
        curvature_tolerance_1pm=curvature_tolerance_1pm,
    )
    output: list[dict[str, Any]] = []
    counts = {"known_valid": 0, "plausible": 0, "known_invalid": 0, "unknown": 0}
    for index, candidate in enumerate(candidates):
        explicit = candidate.get("feasibility_label", candidate.get("support_label"))
        if explicit is not None:
            label = str(explicit)
        elif str(candidate.get("candidate_id")) == str(gt_candidate_id):
            label = "known_valid"
        elif bool(candidate.get("offroad")) or bool(candidate.get("collision")):
            label = "known_invalid"
        elif bool(support[index]):
            label = "plausible"
        else:
            label = "unknown"
        if label not in counts:
            raise ValueError(f"unsupported feasibility label: {label}")
        counts[label] += 1
        row = {
            "candidate_id": str(candidate["candidate_id"]),
            "label": label,
            "acceptable": bool(label in _ACCEPTABLE_LABELS),
            "label_confidence": (
                1.0 if label == "known_valid" else 0.75 if label == "plausible" else 1.0 if label == "known_invalid" else 0.0
            ),
            "support_distance": float(metadata["normalized_distance_by_candidate"][str(candidate["candidate_id"])]),
        }
        if candidate.get("feasibility_reason") is not None:
            row["reason"] = str(candidate["feasibility_reason"])
        output.append(row)
    metadata = dict(metadata)
    metadata.update({"label_definition": "gt_known_valid_kinematic_tube_else_unknown", "label_counts": counts})
    return output, metadata


def counterfactual_consistency(
    probabilities: np.ndarray,
    support_mask: np.ndarray,
) -> dict[str, float | int | None]:
    """Measure posterior mass agreement with independent trajectory support."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    support_mask = np.asarray(support_mask, dtype=bool)
    if probabilities.ndim != 1 or support_mask.shape != probabilities.shape:
        raise ValueError("probabilities and support_mask must be aligned vectors")
    total = float(probabilities.sum())
    if total <= 0.0 or not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite and have positive mass")
    inside = float(probabilities[support_mask].sum() / total)
    outside = max(0.0, 1.0 - inside)
    return {
        "support_mass": inside,
        "outside_support_mass": outside,
        "support_size": int(support_mask.sum()),
        "candidate_count": int(probabilities.size),
        "support_nonempty": bool(support_mask.any()),
    }


def acceptable_set_metrics(
    probabilities: np.ndarray,
    candidate_labels: list[dict[str, Any]],
    *,
    prediction_set: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score posterior mass against a multi-modal acceptable candidate set."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or len(candidate_labels) != probabilities.size:
        raise ValueError("probabilities and candidate_labels must be aligned")
    if not np.all(np.isfinite(probabilities)) or float(probabilities.sum()) <= 0.0:
        raise ValueError("probabilities must be finite and have positive mass")
    acceptable = np.asarray([bool(item.get("acceptable", False)) for item in candidate_labels], dtype=bool)
    known_invalid = np.asarray([item.get("label") == "known_invalid" for item in candidate_labels], dtype=bool)
    unknown = np.asarray([item.get("label") == "unknown" for item in candidate_labels], dtype=bool)
    posterior = probabilities / float(probabilities.sum())
    top_index = int(np.argmax(posterior))
    distances = np.asarray([float(item.get("support_distance", np.inf)) for item in candidate_labels])
    acceptable_distance = float(np.min(distances[acceptable])) if np.any(acceptable) else None
    result: dict[str, Any] = {
        "acceptable_mass": float(posterior[acceptable].sum()),
        "known_invalid_mass": float(posterior[known_invalid].sum()),
        "unknown_mass": float(posterior[unknown].sum()),
        "top_candidate_id": str(candidate_labels[top_index]["candidate_id"]),
        "top_is_acceptable": bool(acceptable[top_index]),
        "distance_to_acceptable_set": acceptable_distance,
        "acceptable_set_size": int(acceptable.sum()),
        "known_invalid_set_size": int(known_invalid.sum()),
        "unknown_set_size": int(unknown.sum()),
    }
    if prediction_set is not None:
        prediction_set = np.asarray(prediction_set, dtype=bool)
        if prediction_set.shape != acceptable.shape:
            raise ValueError("prediction_set must align with candidate_labels")
        result["prediction_set_covers_acceptable"] = bool(np.any(prediction_set & acceptable))
        result["prediction_set_acceptable_fraction"] = float(
            posterior[prediction_set & acceptable].sum() / max(float(posterior[acceptable].sum()), 1e-12)
        ) if np.any(acceptable) else None
        result["prediction_set_unknown_fraction"] = float(
            np.count_nonzero(prediction_set & unknown) / max(np.count_nonzero(prediction_set), 1)
        )
    return result
