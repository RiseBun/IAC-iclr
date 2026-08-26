"""Readiness, split, controls, and foresight-use evaluation for Event-Causal V1."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .event_metrics import event_sequence_distance


FORMAL_IMAGE_EVENT_SOURCE = "frozen_iac_image_probe_v1"
_INDEPENDENT_REALIZED_SOURCES = {
    "ego_state",
    "logged_state",
    "simulator_state",
    "telemetry",
}


def _action_ids(group: Mapping[str, Any]) -> list[str]:
    return [str(branch.get("condition_action_id", "")) for branch in group.get("branches", [])]


def _action_diversity(
    branches: Sequence[Mapping[str, Any]],
    dimensions: Sequence[str],
) -> tuple[float | None, list[str]]:
    distances = []
    reasons = []
    for first_index in range(len(branches)):
        first = branches[first_index].get("action_event_target")
        if first is None:
            reasons.append("missing_action_event_target")
            continue
        for second_index in range(first_index + 1, len(branches)):
            second = branches[second_index].get("action_event_target")
            if second is None:
                reasons.append("missing_action_event_target")
                continue
            comparison = event_sequence_distance(
                first, second, dimensions=dimensions, minimum_observability=0.0
            )
            if comparison["distance"] is not None:
                distances.append(float(comparison["distance"]))
    return (None if not distances else min(distances), sorted(set(reasons)))


def audit_event_benchmark_groups(
    groups: list[dict[str, Any]],
    *,
    dimensions: Sequence[str] = ("lateral",),
    minimum_branches: int = 3,
    minimum_pairwise_action_distance: float = 0.05,
) -> dict[str, Any]:
    """Fail-closed readiness audit with three explicit benchmark tiers."""
    if minimum_branches < 2:
        raise ValueError("minimum_branches must be at least two")
    if not groups:
        raise ValueError("at least one counterfactual group is required")
    rows = []
    seen_group_ids = set()
    for index, group in enumerate(groups):
        group_id = str(group.get("counterfactual_group_id", ""))
        branches = list(group.get("branches", []))
        level1_reasons = []
        level2_reasons = []
        level3_reasons = []
        if not group_id:
            level1_reasons.append("missing_counterfactual_group_id")
        elif group_id in seen_group_ids:
            level1_reasons.append("duplicate_counterfactual_group_id")
        seen_group_ids.add(group_id)
        for field in ("scene_id", "history_id", "generation_seed"):
            if group.get(field) is None:
                level1_reasons.append(f"missing_{field}")
        if len(branches) < minimum_branches:
            level1_reasons.append("insufficient_branches")
        action_ids = _action_ids(group)
        if any(not item for item in action_ids) or len(set(action_ids)) != len(action_ids):
            level1_reasons.append("invalid_or_duplicate_action_ids")
        for branch in branches:
            if branch.get("imagined_event_source") != FORMAL_IMAGE_EVENT_SOURCE:
                level1_reasons.append("unfrozen_or_missing_image_event_source")
            if branch.get("generated_future_id") is None:
                level1_reasons.append("missing_generated_future_id")
            if branch.get("imagined_event_posterior") is None:
                level1_reasons.append("missing_imagined_event_posterior")
            if branch.get("action_event_target") is None:
                level1_reasons.append("missing_action_event_target")
            if branch.get("realized_event_target") is None:
                level2_reasons.append("missing_realized_event_target")
            if branch.get("realized_event_source") not in _INDEPENDENT_REALIZED_SOURCES:
                level2_reasons.append("realized_event_source_not_independent")
            if not isinstance(branch.get("task_success"), (bool, np.bool_)):
                level2_reasons.append("missing_task_success")
        minimum_distance, diversity_reasons = _action_diversity(branches, dimensions)
        level1_reasons.extend(diversity_reasons)
        if minimum_distance is None or minimum_distance < minimum_pairwise_action_distance:
            level1_reasons.append("action_events_not_materially_distinct")

        if group.get("task_event_target") is None:
            level3_reasons.append("missing_task_event_target")
        if str(group.get("planner_baseline_selected_action_id", "")) not in set(action_ids):
            level3_reasons.append("missing_or_invalid_baseline_selection")
        for field in ("planner_id", "planner_baseline_run_id", "planner_nuisance_seed"):
            if group.get(field) is None:
                level3_reasons.append(f"missing_{field}")
        trials = list(group.get("fui_trials", []))
        types = {str(trial.get("intervention_type", "")) for trial in trials}
        if "future_permutation" not in types:
            level3_reasons.append("missing_future_permutation_trial")
        if "null_resample" not in types:
            level3_reasons.append("missing_null_resample_trial")
        identity = {action_id: action_id for action_id in action_ids}
        baseline_future_ids = {
            str(branch.get("condition_action_id", "")): str(branch.get("generated_future_id", ""))
            for branch in branches
        }
        trial_ids = []
        planner_run_ids = []
        for trial in trials:
            trial_id = str(trial.get("trial_id", ""))
            planner_run_id = str(trial.get("planner_run_id", ""))
            trial_ids.append(trial_id)
            planner_run_ids.append(planner_run_id)
            if not trial_id:
                level3_reasons.append("missing_fui_trial_id")
            if not planner_run_id:
                level3_reasons.append("missing_planner_run_id")
            if trial.get("planner_nuisance_seed") != group.get("planner_nuisance_seed"):
                level3_reasons.append("planner_nuisance_seed_not_held_fixed")
            selected = str(trial.get("selected_action_id", ""))
            if selected not in set(action_ids):
                level3_reasons.append("invalid_fui_selected_action")
            assignment = {
                str(key): str(value)
                for key, value in trial.get("future_assignment", {}).items()
            }
            if set(assignment) != set(action_ids) or set(assignment.values()) != set(action_ids):
                level3_reasons.append("invalid_future_assignment")
                continue
            trial_type = str(trial.get("intervention_type", ""))
            if trial_type == "future_permutation" and assignment == identity:
                level3_reasons.append("future_permutation_is_identity")
            if trial_type == "null_resample":
                if assignment != identity:
                    level3_reasons.append("null_resample_changes_action_assignment")
                future_ids = {
                    str(key): str(value)
                    for key, value in trial.get("generated_future_id_by_action", {}).items()
                }
                if set(future_ids) != set(action_ids) or any(
                    future_ids[action_id] == baseline_future_ids[action_id]
                    for action_id in action_ids
                ):
                    level3_reasons.append("null_resample_future_ids_not_independent")
                posteriors = trial.get("imagined_event_posterior_by_source_action")
                if not isinstance(posteriors, Mapping) or set(posteriors) != set(action_ids):
                    level3_reasons.append("missing_null_resample_event_posteriors")
        if len(set(trial_ids)) != len(trial_ids):
            level3_reasons.append("duplicate_fui_trial_ids")
        if len(set(planner_run_ids)) != len(planner_run_ids):
            level3_reasons.append("duplicate_planner_run_ids")

        level1_reasons = sorted(set(level1_reasons))
        level2_reasons = sorted(set(level2_reasons))
        level3_reasons = sorted(set(level3_reasons))
        level1_ready = not level1_reasons
        level2_ready = level1_ready and not level2_reasons
        level3_ready = level2_ready and not level3_reasons
        rows.append({
            "counterfactual_group_id": group_id or str(index),
            "scene_id": group.get("scene_id"),
            "num_branches": len(branches),
            "minimum_pairwise_action_event_distance": minimum_distance,
            "level1_action_response_ready": level1_ready,
            "level2_event_cc_fcs_ready": level2_ready,
            "level3_fui_ready": level3_ready,
            "level1_reasons": level1_reasons,
            "level2_reasons": level2_reasons,
            "level3_reasons": level3_reasons,
        })

    def count(key: str) -> int:
        return sum(bool(row[key]) for row in rows)

    return {
        "protocol": "event-causal-benchmark-readiness-v1",
        "dimensions": list(dimensions),
        "num_groups": len(groups),
        "level1_ready_groups": count("level1_action_response_ready"),
        "level2_ready_groups": count("level2_event_cc_fcs_ready"),
        "level3_ready_groups": count("level3_fui_ready"),
        "formal_event_cc_fcs_ready": count("level2_event_cc_fcs_ready") == len(groups),
        "causal_closure_fui_ready": count("level3_fui_ready") == len(groups),
        "rows": rows,
    }


def scene_disjoint_split(
    groups: list[dict[str, Any]],
    *,
    calibration_fraction: float = 0.6,
    seed: str = "event-causal-v1",
) -> dict[str, Any]:
    """Deterministically assign complete scenes to calibration or holdout."""
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be in (0,1)")
    scenes = sorted({str(group.get("scene_id", "")) for group in groups})
    if "" in scenes:
        raise ValueError("every group requires scene_id")
    if len(scenes) < 2:
        raise ValueError("at least two scenes are required")
    ordered = sorted(
        scenes,
        key=lambda scene: hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).hexdigest(),
    )
    calibration_count = int(np.clip(round(len(ordered) * calibration_fraction), 1, len(ordered) - 1))
    calibration_scenes = set(ordered[:calibration_count])
    holdout_scenes = set(ordered[calibration_count:])
    assignments = {
        str(group["counterfactual_group_id"]): (
            "calibration" if str(group["scene_id"]) in calibration_scenes else "holdout"
        )
        for group in groups
    }
    return {
        "protocol": "scene-disjoint-event-split-v1",
        "seed": seed,
        "calibration_fraction": float(calibration_fraction),
        "calibration_scenes": sorted(calibration_scenes),
        "holdout_scenes": sorted(holdout_scenes),
        "num_calibration_groups": sum(value == "calibration" for value in assignments.values()),
        "num_holdout_groups": sum(value == "holdout" for value in assignments.values()),
        "assignments": assignments,
        "scene_overlap": sorted(calibration_scenes & holdout_scenes),
    }


def build_event_control_groups(
    groups: list[dict[str, Any]],
    control: str,
) -> list[dict[str, Any]]:
    """Create oracle, identical-future, or cyclic action-swap controls."""
    if control not in {"oracle", "identical_future", "action_swap"}:
        raise ValueError("unknown event control")
    output = copy.deepcopy(groups)
    for group in output:
        branches = group.get("branches", [])
        if len(branches) < 2:
            raise ValueError("controls require at least two branches")
        original = [copy.deepcopy(branch["imagined_event_posterior"]) for branch in branches]
        if control == "oracle":
            for branch in branches:
                branch["imagined_event_posterior"] = copy.deepcopy(branch["action_event_target"])
        elif control == "identical_future":
            for branch in branches:
                branch["imagined_event_posterior"] = copy.deepcopy(original[0])
        else:
            for index, branch in enumerate(branches):
                branch["imagined_event_posterior"] = copy.deepcopy(original[(index + 1) % len(branches)])
        group["event_control"] = control
    return output


def _event_optimal_action(
    branches: Sequence[Mapping[str, Any]],
    task_target: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, str],
    dimensions: Sequence[str],
    posterior_by_source_action: Mapping[str, Any] | None = None,
) -> tuple[str | None, dict[str, float]]:
    posterior_by_action = (
        {
            str(branch["condition_action_id"]): branch["imagined_event_posterior"]
            for branch in branches
        }
        if posterior_by_source_action is None
        else {str(key): value for key, value in posterior_by_source_action.items()}
    )
    compatibilities = {}
    for action_id, source_action_id in assignment.items():
        comparison = event_sequence_distance(
            posterior_by_action[source_action_id],
            task_target,
            dimensions=dimensions,
        )
        if comparison["compatibility"] is None:
            return None, {}
        compatibilities[action_id] = float(comparison["compatibility"])
    best = max(compatibilities.values())
    winners = [key for key, value in compatibilities.items() if np.isclose(value, best)]
    return (winners[0] if len(winners) == 1 else None), compatibilities


def evaluate_fui_group(
    group: Mapping[str, Any],
    *,
    dimensions: Sequence[str] = ("lateral",),
) -> dict[str, Any]:
    """Evaluate planner reruns under explicit imagined-future assignments."""
    branches = list(group.get("branches", []))
    action_ids = _action_ids(group)
    action_set = set(action_ids)
    if len(action_set) != len(action_ids) or len(action_ids) < 2:
        raise ValueError("FUI requires unique action ids")
    task_target = group.get("task_event_target")
    if task_target is None:
        raise ValueError("FUI requires task_event_target")
    baseline_selected = str(group.get("planner_baseline_selected_action_id", ""))
    if baseline_selected not in action_set:
        raise ValueError("invalid planner_baseline_selected_action_id")
    identity = {action_id: action_id for action_id in action_ids}
    baseline_expected, baseline_scores = _event_optimal_action(
        branches, task_target, identity, dimensions
    )
    rows = []
    for index, trial in enumerate(group.get("fui_trials", [])):
        assignment = {str(key): str(value) for key, value in trial.get("future_assignment", {}).items()}
        if set(assignment) != action_set or set(assignment.values()) != action_set:
            raise ValueError("future_assignment must be a complete action permutation")
        selected = str(trial.get("selected_action_id", ""))
        if selected not in action_set:
            raise ValueError("FUI trial selected_action_id is invalid")
        posterior_override = trial.get("imagined_event_posterior_by_source_action")
        if posterior_override is not None and set(posterior_override) != action_set:
            raise ValueError(
                "imagined_event_posterior_by_source_action must cover every source action"
            )
        expected, scores = _event_optimal_action(
            branches,
            task_target,
            assignment,
            dimensions,
            posterior_by_source_action=posterior_override,
        )
        trial_type = str(trial.get("intervention_type", ""))
        evaluable = expected is not None
        rows.append({
            "trial_id": str(trial.get("trial_id", index)),
            "intervention_type": trial_type,
            "future_assignment": assignment,
            "selected_action_id": selected,
            "event_optimal_action_id": expected,
            "event_compatibility_by_action": scores,
            "evaluable": evaluable,
            "selection_follows_future": bool(evaluable and selected == expected),
            "event_optimum_changed": bool(evaluable and expected != baseline_expected),
            "selection_changed": selected != baseline_selected,
        })
    swap_rows = [
        row for row in rows
        if row["intervention_type"] == "future_permutation"
        and row["evaluable"] and row["event_optimum_changed"]
    ]
    null_rows = [row for row in rows if row["intervention_type"] == "null_resample" and row["evaluable"]]
    follow_rate = None if not swap_rows else float(np.mean([row["selection_follows_future"] for row in swap_rows]))
    null_change_rate = None if not null_rows else float(np.mean([row["selection_changed"] for row in null_rows]))
    return {
        "protocol": "foresight-use-intervention-v1",
        "counterfactual_group_id": group.get("counterfactual_group_id"),
        "baseline_selected_action_id": baseline_selected,
        "baseline_event_optimal_action_id": baseline_expected,
        "baseline_event_compatibility_by_action": baseline_scores,
        "baseline_selection_agreement": baseline_expected is not None and baseline_selected == baseline_expected,
        "num_future_swap_trials": len(swap_rows),
        "num_null_trials": len(null_rows),
        "future_follow_rate": follow_rate,
        "null_selection_change_rate": null_change_rate,
        "fui_lift": (
            None if follow_rate is None or null_change_rate is None
            else float(follow_rate - null_change_rate)
        ),
        "rows": rows,
    }
