"""Event-level counterfactual consistency and foresight-success metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .action_image_matrix import action_image_cross_matrix
from .maneuver import extract_maneuver


_DIMENSION_KEYS = {
    "lateral": ("lateral_posterior", "lateral_event"),
    "longitudinal": ("longitudinal_posterior", "longitudinal_event"),
    "road": ("road_posterior", "road_event"),
}
_LABEL_ALIASES = {
    "straight": "keep_lane",
    "left": "turn_left",
    "right": "turn_right",
}
_INDEPENDENT_REALIZED_SOURCES = {
    "ego_state",
    "logged_state",
    "simulator_state",
    "telemetry",
}


def action_trajectory_event_target(
    trajectory: np.ndarray,
    future_times_s: np.ndarray,
) -> list[dict[str, Any]]:
    """Create an image-independent event target from an action trajectory."""
    maneuver = extract_maneuver(
        np.asarray(trajectory, dtype=np.float64),
        np.asarray(future_times_s, dtype=np.float64),
    )
    return list(maneuver["event_posterior"])


def _distribution(row: Mapping[str, Any], dimension: str) -> dict[str, float] | None:
    posterior_key, label_key = _DIMENSION_KEYS[dimension]
    posterior = row.get(posterior_key)
    if posterior is not None:
        if not isinstance(posterior, Mapping) or not posterior:
            raise ValueError(f"{posterior_key} must be a non-empty mapping")
        values = {
            _LABEL_ALIASES.get(str(label), str(label)): float(value)
            for label, value in posterior.items()
        }
        array = np.asarray(list(values.values()), dtype=np.float64)
        if not np.all(np.isfinite(array)) or np.any(array < 0.0) or float(array.sum()) <= 0.0:
            raise ValueError(f"{posterior_key} must contain finite non-negative mass")
        total = float(array.sum())
        return {label: value / total for label, value in values.items()}
    label = row.get(label_key)
    if label is None:
        return None
    canonical = _LABEL_ALIASES.get(str(label), str(label))
    return {canonical: 1.0}


def _js_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Normalized Jensen-Shannon divergence in [0, 1]."""
    labels = sorted(set(first) | set(second))
    left = np.asarray([float(first.get(label, 0.0)) for label in labels], dtype=np.float64)
    right = np.asarray([float(second.get(label, 0.0)) for label in labels], dtype=np.float64)
    left /= float(left.sum())
    right /= float(right.sum())
    middle = 0.5 * (left + right)

    def kl(values: np.ndarray) -> float:
        valid = values > 0.0
        return float(np.sum(values[valid] * np.log(values[valid] / middle[valid])))

    return float(np.clip(0.5 * (kl(left) + kl(right)) / np.log(2.0), 0.0, 1.0))


def event_sequence_distance(
    observed: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
    *,
    dimensions: Sequence[str] = ("lateral",),
    minimum_observability: float = 0.25,
) -> dict[str, Any]:
    """Compare aligned interval event posteriors without metric trajectories."""
    dimensions = tuple(str(item) for item in dimensions)
    unknown = [item for item in dimensions if item not in _DIMENSION_KEYS]
    if unknown:
        raise ValueError(f"unknown event dimensions: {unknown}")
    if not dimensions:
        raise ValueError("at least one event dimension is required")
    if len(observed) != len(target) or not observed:
        raise ValueError("observed and target event sequences must have equal non-zero length")
    if not 0.0 <= float(minimum_observability) <= 1.0:
        raise ValueError("minimum_observability must be in [0,1]")

    interval_rows = []
    weighted_distances = []
    weights = []
    for index, (observed_row, target_row) in enumerate(zip(observed, target)):
        observability = float(observed_row.get("observability", 1.0))
        if not np.isfinite(observability):
            raise ValueError("observability must be finite")
        observability = float(np.clip(observability, 0.0, 1.0))
        reason = None
        distances = {}
        if bool(observed_row.get("abstain", False)):
            reason = "image_probe_abstain"
        elif observability < float(minimum_observability):
            reason = "low_observability"
        else:
            for dimension in dimensions:
                observed_distribution = _distribution(observed_row, dimension)
                target_distribution = _distribution(target_row, dimension)
                if observed_distribution is None or target_distribution is None:
                    continue
                if dimension == "road" and set(target_distribution) == {"unknown"}:
                    continue
                distances[dimension] = _js_distance(observed_distribution, target_distribution)
            if not distances:
                reason = "missing_target_dimension"
        distance = None if reason is not None else float(np.mean(list(distances.values())))
        if distance is not None:
            weighted_distances.append(distance * observability)
            weights.append(observability)
        interval_rows.append({
            "interval_index": index,
            "evaluable": distance is not None,
            "reason": reason,
            "observability": observability,
            "distance_by_dimension": distances,
            "distance": distance,
        })

    evaluable = len(weighted_distances)
    distance = None if not weights else float(sum(weighted_distances) / sum(weights))
    return {
        "protocol": "aligned-event-js-distance-v1",
        "dimensions": list(dimensions),
        "distance": distance,
        "compatibility": None if distance is None else float(1.0 - distance),
        "num_intervals": len(observed),
        "num_evaluable_intervals": evaluable,
        "coverage": evaluable / len(observed),
        "minimum_observability": float(minimum_observability),
        "intervals": interval_rows,
    }


def event_counterfactual_matrix(
    branches: list[dict[str, Any]],
    *,
    dimensions: Sequence[str] = ("lateral",),
    minimum_observability: float = 0.25,
    minimum_interval_coverage: float = 0.5,
    temperature: float = 0.10,
    decision_margin: float | None = None,
) -> dict[str, Any]:
    """Cross-score image event posteriors against every branch action event."""
    if len(branches) < 2:
        raise ValueError("at least two action-conditioned branches are required")
    if not 0.0 <= float(minimum_interval_coverage) <= 1.0:
        raise ValueError("minimum_interval_coverage must be in [0,1]")
    action_ids = [str(branch["condition_action_id"]) for branch in branches]
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("condition_action_id must be unique within a group")
    targets = []
    for branch in branches:
        target = branch.get("action_event_target")
        if target is None:
            raise ValueError("every branch requires an image-independent action_event_target")
        targets.append(target)

    scored = []
    diagnostics = []
    for row_index, branch in enumerate(branches):
        observed = branch.get("imagined_event_posterior")
        if observed is None:
            raise ValueError("every branch requires imagined_event_posterior")
        cross_scores = [
            event_sequence_distance(
                observed,
                target,
                dimensions=dimensions,
                minimum_observability=minimum_observability,
            )
            for target in targets
        ]
        cross_evaluable = all(
            item["distance"] is not None
            and item["coverage"] >= float(minimum_interval_coverage)
            for item in cross_scores
        )
        valid = bool(branch.get("valid", True)) and cross_evaluable
        reasons = list(branch.get("abstain_reasons", []))
        if not cross_evaluable:
            reasons.append("insufficient_event_coverage")
        scored.append({
            "branch_id": str(branch.get("branch_id", row_index)),
            "condition_action_id": action_ids[row_index],
            "valid": valid,
            "abstain_reasons": reasons,
            "candidate_scores": [
                {
                    "candidate_id": action_id,
                    "energy": 1.0 if score["distance"] is None else score["distance"],
                }
                for action_id, score in zip(action_ids, cross_scores)
            ],
        })
        diagnostics.append(cross_scores)

    result = action_image_cross_matrix(
        scored,
        temperature=temperature,
        decision_margin=decision_margin,
    )
    result["protocol"] = "event-counterfactual-consistency-v1"
    result["evidence_source"] = "image_event_posterior"
    result["target_source"] = "action_event_target"
    result["distance_definition"] = "observability-weighted-normalized-js"
    result["dimensions"] = list(dimensions)
    result["minimum_observability"] = float(minimum_observability)
    result["minimum_interval_coverage"] = float(minimum_interval_coverage)
    result["event_cross_score_diagnostics"] = diagnostics
    result["chance_top1_accuracy"] = 1.0 / len(branches)
    result["top1_lift_over_chance"] = (
        None if result["diagonal_top1_accuracy"] is None
        else float(result["diagonal_top1_accuracy"] - result["chance_top1_accuracy"])
    )

    valid_rows = [
        (index, np.asarray(row, dtype=np.float64))
        for index, row in enumerate(result["probability_matrix"])
        if row[0] is not None
    ]
    cyclic_margins = []
    for row_index, probabilities in valid_rows:
        swapped_index = (row_index + 1) % len(branches)
        cyclic_margins.append(float(
            probabilities[swapped_index]
            - np.max(np.delete(probabilities, swapped_index))
        ))
    cyclic_margin = None if not cyclic_margins else float(np.mean(cyclic_margins))
    result["cyclic_action_swap_mean_cc_margin"] = cyclic_margin
    result["cc_margin_lift_over_cyclic_swap"] = (
        None if cyclic_margin is None or result["mean_cc_margin"] is None
        else float(result["mean_cc_margin"] - cyclic_margin)
    )
    return result


def event_foresight_conditioned_success(
    episodes: list[dict[str, Any]],
    *,
    compatibility_threshold: float = 0.70,
    dimensions: Sequence[str] = ("lateral",),
    minimum_observability: float = 0.25,
    minimum_interval_coverage: float = 0.5,
) -> dict[str, Any]:
    """Joint foresight/event alignment and task success with fail-closed eligibility."""
    if not episodes:
        raise ValueError("at least one episode is required")
    if not 0.0 <= float(compatibility_threshold) <= 1.0:
        raise ValueError("compatibility_threshold must be in [0,1]")

    rows = []
    for index, episode in enumerate(episodes):
        reasons = []
        source = str(episode.get("realized_event_source", ""))
        if source not in _INDEPENDENT_REALIZED_SOURCES:
            reasons.append("realized_event_source_not_independent")
        success = episode.get("task_success")
        if not isinstance(success, (bool, np.bool_)):
            reasons.append("missing_task_success")
        imagined = episode.get("imagined_event_posterior")
        realized = episode.get("realized_event_target")
        if imagined is None or realized is None:
            reasons.append("missing_event_sequence")
            comparison = None
        else:
            comparison = event_sequence_distance(
                imagined,
                realized,
                dimensions=dimensions,
                minimum_observability=minimum_observability,
            )
            if comparison["coverage"] < float(minimum_interval_coverage):
                reasons.append("insufficient_event_coverage")
        evaluable = not reasons
        compatibility = None if comparison is None else comparison["compatibility"]
        compatible = bool(
            evaluable
            and compatibility is not None
            and compatibility >= float(compatibility_threshold)
        )
        rows.append({
            "episode_id": str(episode.get("episode_id", index)),
            "evaluable": evaluable,
            "ineligible_reasons": reasons,
            "realized_event_source": source or None,
            "event_compatibility": compatibility,
            "event_coverage": None if comparison is None else comparison["coverage"],
            "compatible": compatible,
            "task_success": bool(success) if isinstance(success, (bool, np.bool_)) else None,
        })

    evaluable_rows = [row for row in rows if row["evaluable"]]
    compatible_rows = [row for row in evaluable_rows if row["compatible"]]
    successful_compatible = sum(bool(row["task_success"]) for row in compatible_rows)
    unconditional_success = (
        None if not evaluable_rows
        else float(np.mean([row["task_success"] for row in evaluable_rows]))
    )
    fcs = None if not compatible_rows else successful_compatible / len(compatible_rows)
    return {
        "protocol": "event-foresight-conditioned-success-v1",
        "dimensions": list(dimensions),
        "compatibility_threshold": float(compatibility_threshold),
        "num_episodes": len(episodes),
        "num_evaluable": len(evaluable_rows),
        "evaluation_coverage": len(evaluable_rows) / len(episodes),
        "num_compatible": len(compatible_rows),
        "foresight_coverage": len(compatible_rows) / len(episodes),
        "foresight_coverage_among_evaluable": (
            None if not evaluable_rows else len(compatible_rows) / len(evaluable_rows)
        ),
        "unconditional_success": unconditional_success,
        "foresight_conditioned_success": fcs,
        "joint_fcs": successful_compatible / len(episodes),
        "success_lift": (
            None if fcs is None or unconditional_success is None
            else float(fcs - unconditional_success)
        ),
        "rows": rows,
        "eligibility": (
            "realized events must come from independent ego/simulator telemetry; "
            "task_success is required and missing values are never imputed"
        ),
    }
