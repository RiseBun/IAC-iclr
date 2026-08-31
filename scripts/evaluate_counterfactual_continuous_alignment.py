#!/usr/bin/env python3
"""Evaluate paired risk/clear WAM branches in continuous motion space."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.continuous_motion import (
    compare_counterfactual_motion_deltas,
    compare_counterfactual_se2_consistency,
    image_motion_profile,
    trajectory_to_motion_profile,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def classify_counterfactual_claim(
    *, intervention_types: list[str], specificity_controls: list[str], structurally_ready: bool
) -> dict[str, Any]:
    """Separate structural, internal-foresight, and semantic-hazard claims."""
    command_only = intervention_types == ["navigation_command_onehot"]
    internal_future_only = intervention_types == ["internal_future_latent_permutation"]
    formal_foresight_ready = structurally_ready and not command_only and not specificity_controls
    semantic_hazard_ready = formal_foresight_ready and not internal_future_only
    scope = (
        "specificity_control" if specificity_controls else
        "command_conditioned_action_image_consistency" if command_only else
        "internal_foresight_mediation" if internal_future_only else
        "semantic_foresight_counterfactual_consistency"
    )
    return {
        "formal_foresight_ready": bool(formal_foresight_ready),
        "semantic_hazard_ready": bool(semantic_hazard_ready),
        "claim_scope": scope,
    }


def audit_pair(
    group_id: str,
    roles: dict[str, dict[str, Any]],
    *,
    role_a: str = "clear",
    role_b: str = "risk",
) -> list[str]:
    issues = []
    for field in ("history_fingerprint", "wam_model_id", "nuisance_seed"):
        values = [roles[role].get(field) for role in (role_a, role_b)]
        if any(value is None for value in values):
            issues.append(f"missing_{field}")
        elif values[0] != values[1]:
            issues.append(f"mismatched_{field}")
    for role, row in roles.items():
        if row.get("future_images_source") != "wam_generated":
            issues.append(f"{role}_future_not_wam_generated")
        source = str(row.get("action_trajectory_source") or "").lower()
        if not source:
            issues.append(f"{role}_missing_action_trajectory_source")
        elif any(token in source for token in ("logged", "oracle", "proxy", "candidate")):
            issues.append(f"{role}_non_native_action_source")
        if row.get("candidate_bank_used_by_decoder") is not False:
            issues.append(f"{role}_candidate_blind_audit_failed")
    clear_times = np.asarray(roles[role_a].get("future_times_s") or [], dtype=np.float64)
    risk_times = np.asarray(roles[role_b].get("future_times_s") or [], dtype=np.float64)
    if clear_times.shape != risk_times.shape or not np.allclose(clear_times, risk_times, atol=1e-6, rtol=0.0):
        issues.append("mismatched_future_timestamps")
    clear_action = np.asarray(roles[role_a].get("action_trajectory") or [], dtype=np.float64)
    risk_action = np.asarray(roles[role_b].get("action_trajectory") or [], dtype=np.float64)
    if clear_action.shape != risk_action.shape or clear_action.ndim != 2 or clear_action.shape[1:] != (3,):
        issues.append("invalid_or_mismatched_action_trajectories")
    elif float(np.max(np.abs(clear_action - risk_action))) <= 1e-4:
        issues.append("action_intervention_has_no_effect")
    if not group_id or group_id == "None":
        issues.append("missing_counterfactual_group_id")
    return sorted(set(issues))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-eight-frame-four-second", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--role-a", default="clear")
    parser.add_argument("--role-b", default="risk")
    args = parser.parse_args()

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.records):
        groups[str(row.get("counterfactual_group_id"))].append(row)
    reports = []
    for group_id, branches in sorted(groups.items()):
        roles = {str(row.get("branch_role")): row for row in branches}
        if set(roles) != {args.role_a, args.role_b}:
            raise ValueError(f"{group_id}: exactly one {args.role_a} and one {args.role_b} branch are required")
        issues = audit_pair(group_id, roles, role_a=args.role_a, role_b=args.role_b)
        if args.require_ready and issues:
            raise ValueError(f"{group_id}: counterfactual readiness failed: {issues}")
        profiles = {}
        for role, row in roles.items():
            times = list(row.get("future_times_s") or [])
            if args.require_eight_frame_four_second and (len(times) != 8 or abs(float(times[-1]) - 4.0) > 0.05):
                raise ValueError(f"{group_id}/{role}: expected 8 frames ending at 4.0 seconds")
            if row.get("candidate_bank_used_by_decoder") is not False:
                raise ValueError(f"{group_id}/{role}: candidate-blind audit failed")
            history = row.get("history_ego_state") or []
            initial_speed = float(history[-1][3]) if history and len(history[-1]) >= 4 else None
            profiles[role] = (
                image_motion_profile(row.get("decoder") or {}, times, initial_speed_mps=initial_speed),
                trajectory_to_motion_profile(row.get("action_trajectory"), times, initial_speed_mps=initial_speed),
            )
        image_clear, action_clear = profiles[args.role_a]
        image_risk, action_risk = profiles[args.role_b]
        reports.append({
            "counterfactual_group_id": group_id,
            "causal_claim_eligible": not issues,
            "readiness_issues": issues,
            "comparison": compare_counterfactual_motion_deltas(
                image_clear, image_risk, action_clear, action_risk
            ),
            "continuous_cfc": {
                "metric": compare_counterfactual_se2_consistency(
                    image_clear, image_risk, action_clear, action_risk, scale_mode="metric"
                ),
                "scale_free": compare_counterfactual_se2_consistency(
                    image_clear, image_risk, action_clear, action_risk, scale_mode="scale_free"
                ),
                "arc_relative": compare_counterfactual_se2_consistency(
                    image_clear, image_risk, action_clear, action_risk, scale_mode="arc_relative"
                ),
            },
        })
    eligible = [row for row in reports if row["causal_claim_eligible"]]
    metric_scores = [
        row["continuous_cfc"]["metric"]["score"]
        for row in eligible
        if row["continuous_cfc"]["metric"].get("score") is not None
    ]
    scale_free_scores = [
        row["continuous_cfc"]["scale_free"]["score"]
        for row in eligible
        if row["continuous_cfc"]["scale_free"].get("score") is not None
    ]
    arc_relative_scores = [
        row["continuous_cfc"]["arc_relative"]["score"]
        for row in eligible
        if row["continuous_cfc"]["arc_relative"].get("score") is not None
    ]
    intervention_types = sorted({
        str(row.get("intervention_type"))
        for branches in groups.values()
        for row in branches
        if row.get("intervention_type") is not None
    })
    specificity_controls = sorted({
        str(row.get("specificity_control"))
        for branches in groups.values()
        for row in branches
        if row.get("specificity_control") is not None
    })
    structurally_eligible = bool(reports) and all(row["causal_claim_eligible"] for row in reports)
    claim = classify_counterfactual_claim(
        intervention_types=intervention_types,
        specificity_controls=specificity_controls,
        structurally_ready=structurally_eligible,
    )
    formal_foresight_eligible = claim["formal_foresight_ready"]
    semantic_hazard_eligible = claim["semantic_hazard_ready"]
    for report in reports:
        report["structural_pair_eligible"] = report["causal_claim_eligible"]
        report["formal_foresight_claim_eligible"] = bool(
            report["causal_claim_eligible"] and formal_foresight_eligible
        )
        report["semantic_hazard_claim_eligible"] = bool(
            report["causal_claim_eligible"] and semantic_hazard_eligible
        )
        report["causal_claim_eligible"] = report["formal_foresight_claim_eligible"]
    output = {
        "protocol": "counterfactual-continuous-alignment-report-v2",
        "primary_metric": "continuous-counterfactual-foresight-consistency-v1",
        "pair_roles": [args.role_a, args.role_b],
        "groups": len(reports),
        "causal_claim_eligible": formal_foresight_eligible,
        "structural_pair_eligible": structurally_eligible,
        "formal_foresight_claim_eligible": formal_foresight_eligible,
        "semantic_hazard_claim_eligible": semantic_hazard_eligible,
        "claim_scope": claim["claim_scope"],
        "intervention_types": intervention_types,
        "specificity_controls": specificity_controls,
        "summary": {
            "eligible_groups": len(eligible),
            "metric_score_mean": None if not metric_scores else float(np.mean(metric_scores)),
            "scale_free_score_mean": None if not scale_free_scores else float(np.mean(scale_free_scores)),
            "metric_score_count": len(metric_scores),
            "scale_free_score_count": len(scale_free_scores),
            "arc_relative_score_mean": None if not arc_relative_scores else float(np.mean(arc_relative_scores)),
            "arc_relative_score_count": len(arc_relative_scores),
        },
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
