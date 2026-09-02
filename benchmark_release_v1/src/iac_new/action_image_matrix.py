"""Cross-score action-conditioned future images against a shared action bank."""

from __future__ import annotations

from typing import Any

import numpy as np

from .wam_metrics import (
    ego_state_descriptor,
    normalized_ego_state_distance,
    normalized_ego_state_support_distance,
)


def _row_probabilities(energies: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = -(np.asarray(energies, dtype=np.float64) - float(np.min(energies))) / temperature
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    return weights / float(weights.sum())


def action_image_cross_matrix(
    branches: list[dict[str, Any]],
    *,
    temperature: float = 1.0,
    decision_margin: float | None = None,
) -> dict[str, Any]:
    """Build a KxK future-image/action matrix from image-probe energies.

    Each branch is one future generated from the same history under a different
    action condition. ``candidate_scores`` must score that future against every
    condition action in the group. Invalid branches are retained as abstentions
    but excluded from aggregate consistency statistics.
    """
    if len(branches) < 2:
        raise ValueError("at least two action-conditioned branches are required")
    action_ids = [str(branch["condition_action_id"]) for branch in branches]
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("condition_action_id must be unique within a group")

    energy_rows: list[list[float | None]] = []
    probability_rows: list[list[float | None]] = []
    branch_results: list[dict[str, Any]] = []
    valid_probability_rows: list[np.ndarray] = []
    for row_index, branch in enumerate(branches):
        score_by_id = {
            str(item["candidate_id"]): float(item["energy"])
            for item in branch.get("candidate_scores", [])
        }
        missing = [action_id for action_id in action_ids if action_id not in score_by_id]
        if missing:
            raise ValueError(
                f"{branch.get('branch_id', row_index)} is missing action scores {missing}"
            )
        energies = np.asarray([score_by_id[action_id] for action_id in action_ids])
        if not np.all(np.isfinite(energies)):
            raise ValueError("candidate energies must be finite")
        valid = bool(branch.get("valid", True))
        if not valid:
            energy_rows.append(energies.tolist())
            probability_rows.append([None] * len(action_ids))
            branch_results.append({
                "branch_id": str(branch.get("branch_id", row_index)),
                "condition_action_id": action_ids[row_index],
                "decision": "abstain",
                "abstain_reasons": list(branch.get("abstain_reasons", [])),
                "matched_rank": None,
                "matched_probability": None,
                "cc_margin": None,
            })
            continue

        probabilities = _row_probabilities(energies, temperature)
        diagonal = float(probabilities[row_index])
        counterfactual = np.delete(probabilities, row_index)
        best_counterfactual = float(np.max(counterfactual))
        other_energies = np.delete(energies, row_index)
        lower_count = int(np.count_nonzero(other_energies < energies[row_index]))
        tie_count = int(np.count_nonzero(np.isclose(
            other_energies, energies[row_index], rtol=1e-9, atol=1e-12
        )))
        rank = 1.0 + lower_count + 0.5 * tie_count
        unique_top1 = lower_count == 0 and tie_count == 0
        cc_margin = diagonal - best_counterfactual
        energy_margin = float(np.min(other_energies) - energies[row_index])
        if decision_margin is None:
            decision = "diagnostic_only"
        elif cc_margin >= decision_margin and unique_top1:
            decision = "supported"
        elif cc_margin <= -decision_margin and rank > 1:
            decision = "mismatched"
        else:
            decision = "uncertain"
        energy_rows.append(energies.tolist())
        probability_rows.append(probabilities.tolist())
        valid_probability_rows.append(probabilities)
        branch_results.append({
            "branch_id": str(branch.get("branch_id", row_index)),
            "condition_action_id": action_ids[row_index],
            "decision": decision,
            "abstain_reasons": [],
            "matched_rank": rank,
            "matched_unique_top1": unique_top1,
            "reciprocal_rank": 1.0 / rank,
            "matched_probability": diagonal,
            "best_counterfactual_probability": best_counterfactual,
            "cc_margin": cc_margin,
            "energy_margin": energy_margin,
            "pairwise_matched_accuracy": float(
                np.mean(energies[row_index] < np.delete(energies, row_index))
            ),
        })

    valid_results = [row for row in branch_results if row["matched_rank"] is not None]
    pairwise_tv = []
    for first_index in range(len(valid_probability_rows)):
        for second_index in range(first_index + 1, len(valid_probability_rows)):
            pairwise_tv.append(float(
                0.5 * np.abs(
                    valid_probability_rows[first_index] - valid_probability_rows[second_index]
                ).sum()
            ))
    return {
        "protocol": "counterfactual-action-image-matrix-v1",
        "action_ids": action_ids,
        "energy_matrix": energy_rows,
        "probability_matrix": probability_rows,
        "branches": branch_results,
        "num_branches": len(branches),
        "num_evaluable": len(valid_results),
        "num_abstain": len(branches) - len(valid_results),
        "coverage": len(valid_results) / len(branches),
        "diagonal_top1_accuracy": (
            float(np.mean([row["matched_unique_top1"] for row in valid_results]))
            if valid_results else None
        ),
        "mean_reciprocal_rank": (
            float(np.mean([row["reciprocal_rank"] for row in valid_results]))
            if valid_results else None
        ),
        "mean_matched_probability": (
            float(np.mean([row["matched_probability"] for row in valid_results]))
            if valid_results else None
        ),
        "mean_cc_margin": (
            float(np.mean([row["cc_margin"] for row in valid_results]))
            if valid_results else None
        ),
        "mean_energy_margin": (
            float(np.mean([row["energy_margin"] for row in valid_results]))
            if valid_results else None
        ),
        "mean_pairwise_matched_accuracy": (
            float(np.mean([row["pairwise_matched_accuracy"] for row in valid_results]))
            if valid_results else None
        ),
        "mean_pairwise_response_tv": float(np.mean(pairwise_tv)) if pairwise_tv else None,
        "decision_margin": decision_margin,
        "interpretation": (
            "Positive CC margin means each generated future supports its own action "
            "more than the strongest counterfactual action. Decisions require a "
            "margin calibrated on held-out native data."
        ),
    }


def decoded_trajectory_cross_matrix(
    branches: list[dict[str, Any]],
    future_times_s: np.ndarray,
    *,
    temperature: float = 1.0,
    decision_margin: float | None = None,
    scales: dict[str, float] | None = None,
    support_aware: bool = True,
) -> dict[str, Any]:
    """Cross-score image-decoded trajectories against every branch action."""
    actions = [
        np.asarray(branch.get("action_condition", branch.get("executed_action")), dtype=np.float64)
        for branch in branches
    ]
    scored = []
    for row_index, branch in enumerate(branches):
        imagined = np.asarray(branch["imagined_future"], dtype=np.float64)
        support = branch.get("imagined_support")
        scored.append({
            "branch_id": str(branch.get("branch_id", row_index)),
            "condition_action_id": str(branch.get("branch_id", row_index)),
            "valid": bool(branch.get("valid", True)),
            "abstain_reasons": list(branch.get("abstain_reasons", [])),
            "candidate_scores": [
                {
                    "candidate_id": str(candidate.get("branch_id", action_index)),
                    "energy": (
                        normalized_ego_state_support_distance(
                            action, support, np.asarray(future_times_s, dtype=np.float64), scales=scales
                        ) if support_aware and support is not None else normalized_ego_state_distance(
                            imagined, action, np.asarray(future_times_s, dtype=np.float64), scales=scales
                        )
                    ),
                    "point_energy": normalized_ego_state_distance(
                        imagined, action, np.asarray(future_times_s, dtype=np.float64), scales=scales
                    ),
                }
                for action_index, (candidate, action) in enumerate(zip(branches, actions))
            ],
        })
    result = action_image_cross_matrix(
        scored,
        temperature=temperature,
        decision_margin=decision_margin,
    )
    result["evidence_source"] = "image_decoded_ego_trajectory"
    result["energy_definition"] = "normalized_ego_state_rms_distance"
    result["scales"] = scales
    result["support_aware"] = bool(support_aware and any(branch.get("imagined_support") is not None for branch in branches))
    if result["support_aware"]:
        result["energy_definition"] = "normalized_ego_state_support_distance"
    return result


def decoded_intervention_delta_matrix(
    branches: list[dict[str, Any]],
    future_times_s: np.ndarray,
    *,
    baseline_index: int = 0,
    scales: dict[str, float] | None = None,
    temperature: float = 1.0,
    decision_margin: float | None = None,
) -> dict[str, Any]:
    """Score branch responses after removing the shared-history baseline.

    Absolute image-to-action distance is confounded when a generated WAM
    stream has a different forward scale from the dataset action units.  For a
    same-history intervention group, subtract the baseline branch descriptor
    from every imagined trajectory and every action condition, then compare
    only the intervention delta.  This is an action-response metric, not a
    claim that the absolute generated trajectory is metric-accurate.
    """
    if len(branches) < 2:
        raise ValueError("at least two action-conditioned branches are required")
    if not 0 <= int(baseline_index) < len(branches):
        raise ValueError("baseline_index is outside the branch list")
    times = np.asarray(future_times_s, dtype=np.float64)
    baseline = branches[int(baseline_index)]
    baseline_imagined = ego_state_descriptor(
        np.asarray(baseline["imagined_future"], dtype=np.float64), times, scales=scales
    )
    actions = [
        ego_state_descriptor(
            np.asarray(branch.get("action_condition", branch.get("executed_action")), dtype=np.float64),
            times,
            scales=scales,
        )
        for branch in branches
    ]
    baseline_action = actions[int(baseline_index)]
    imagined_deltas = [
        ego_state_descriptor(np.asarray(branch["imagined_future"], dtype=np.float64), times, scales=scales)
        - baseline_imagined
        for branch in branches
    ]
    action_deltas = [action - baseline_action for action in actions]
    scored = []
    for row_index, branch in enumerate(branches):
        scored.append({
            "branch_id": str(branch.get("branch_id", row_index)),
            "condition_action_id": str(branch.get("branch_id", row_index)),
            "valid": bool(branch.get("valid", True)),
            "abstain_reasons": list(branch.get("abstain_reasons", [])),
            "candidate_scores": [
                {
                    "candidate_id": str(candidate.get("branch_id", action_index)),
                    "energy": float(np.sqrt(np.mean(np.square(imagined_deltas[row_index] - action_deltas[action_index])))),
                }
                for action_index, candidate in enumerate(branches)
            ],
        })
    result = action_image_cross_matrix(
        scored, temperature=temperature, decision_margin=decision_margin
    )
    result["evidence_source"] = "image_decoded_intervention_delta"
    result["energy_definition"] = "baseline_relative_normalized_ego_state_rms_distance"
    result["baseline_branch_id"] = str(baseline.get("branch_id", baseline_index))
    result["baseline_index"] = int(baseline_index)
    result["absolute_scale_invariant"] = True
    return result
