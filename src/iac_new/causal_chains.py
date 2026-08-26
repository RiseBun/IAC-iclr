"""Fail-closed contracts and metrics for interaction-level causal chains."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PROTOCOL = "iac-interaction-causal-chain-v1"

FORMAL_IMAGINED_SOURCES = {
    "frozen_iac_interaction_probe_v1",
    "frozen_iac_latent_event_probe_v1",
    "blinded_human_consensus",
}
INDEPENDENT_TRIGGER_SOURCES = {
    "dataset_annotation",
    "blinded_human_consensus",
    "logged_state",
    "simulator_state",
    "telemetry",
}
ACTION_SOURCES = {
    "wam_action_trajectory",
    "planner_output",
    "controller_telemetry",
}
INDEPENDENT_OUTCOME_SOURCES = {
    "logged_state",
    "simulator_state",
    "telemetry",
}


CAUSAL_CHAIN_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "cut_in_or_lead_brake": {
        "risk_triggers": ("vehicle_cut_in", "lead_vehicle_brake"),
        "clear_triggers": (
            "vehicle_keeps_lane",
            "lead_vehicle_maintains_speed",
            "no_vehicle_conflict",
        ),
        "risk_consequences": ("collision_risk", "unsafe_ttc", "headway_conflict"),
        "safe_consequences": ("no_conflict", "safe_ttc", "safe_headway"),
        "protective_responses": ("decelerate", "emergency_brake", "yield"),
        "permissive_responses": ("maintain_speed", "accelerate", "follow"),
        "safe_outcomes": ("no_collision", "safe_ttc", "safe_headway"),
        "failure_outcomes": ("collision", "unsafe_ttc", "unsafe_headway"),
    },
    "pedestrian_crossing": {
        "risk_triggers": ("pedestrian_crossing", "pedestrian_popout"),
        "clear_triggers": ("pedestrian_clear", "crosswalk_clear"),
        "risk_consequences": ("occupancy_conflict", "collision_risk"),
        "safe_consequences": ("path_clear", "no_conflict", "safe_clearance"),
        "protective_responses": ("yield", "stop", "decelerate"),
        "permissive_responses": ("start", "restart", "creep", "maintain_speed"),
        "safe_outcomes": (
            "no_collision",
            "safe_clearance",
            "safe_stop",
            "progress_after_clearance",
        ),
        "failure_outcomes": ("collision", "unsafe_clearance", "no_restart_after_clearance"),
    },
    "blocked_lane": {
        "risk_triggers": (
            "blocked_lane",
            "stopped_vehicle",
            "road_debris",
            "construction_barrier",
        ),
        "clear_triggers": ("lane_clear", "obstacle_cleared"),
        "risk_consequences": ("obstruction", "lane_blocked", "collision_risk"),
        "safe_consequences": ("lane_clear", "path_clear", "no_conflict"),
        "protective_responses": (
            "lane_change_left",
            "lane_change_right",
            "avoid_left",
            "avoid_right",
            "stop",
        ),
        "permissive_responses": ("keep_lane", "maintain_speed"),
        "safe_outcomes": ("no_collision", "drivable_area_compliant", "safe_progress"),
        "failure_outcomes": ("collision", "off_road", "no_progress"),
    },
    "unprotected_turn_or_merge": {
        "risk_triggers": ("unprotected_turn", "merge_gap", "gap_closing"),
        "clear_triggers": ("unprotected_turn", "merge_gap", "gap_open"),
        "risk_consequences": ("gap_unsafe", "gap_closing", "collision_risk"),
        "safe_consequences": ("gap_safe", "gap_open", "no_conflict"),
        "protective_responses": ("yield", "stop", "creep"),
        "permissive_responses": (
            "gap_accept",
            "turn_left",
            "turn_right",
            "merge_left",
            "merge_right",
        ),
        "safe_outcomes": ("no_collision", "safe_ttc", "safe_progress"),
        "failure_outcomes": ("collision", "unsafe_ttc", "blocked_or_no_progress"),
    },
}


RESPONSE_REQUIREMENT_GROUPS: dict[
    str, dict[str, tuple[tuple[str, ...], ...]]
] = {
    "cut_in_or_lead_brake": {
        "risk": (("decelerate", "emergency_brake", "yield"),),
        "clear": (("maintain_speed", "accelerate", "follow"),),
    },
    "pedestrian_crossing": {
        "risk": (
            ("yield", "stop", "decelerate"),
            ("start", "restart"),
        ),
        "clear": (("start", "restart", "creep", "maintain_speed"),),
    },
    "blocked_lane": {
        "risk": ((
            "lane_change_left",
            "lane_change_right",
            "avoid_left",
            "avoid_right",
            "stop",
        ),),
        "clear": (("keep_lane", "maintain_speed"),),
    },
    "unprotected_turn_or_merge": {
        "risk": (("yield", "stop", "creep"),),
        "clear": ((
            "gap_accept",
            "turn_left",
            "turn_right",
            "merge_left",
            "merge_right",
        ),),
    },
}


OUTCOME_REQUIREMENT_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "cut_in_or_lead_brake": (
        ("no_collision",),
        ("safe_ttc", "safe_headway"),
    ),
    "pedestrian_crossing": (
        ("no_collision",),
        ("safe_clearance", "safe_stop"),
        ("progress_after_clearance",),
    ),
    "blocked_lane": (
        ("no_collision",),
        ("drivable_area_compliant",),
        ("safe_progress",),
    ),
    "unprotected_turn_or_merge": (
        ("no_collision",),
        ("safe_ttc",),
        ("safe_progress",),
    ),
}


def _stage_scores(stage: Any, field: str, reasons: list[str]) -> dict[str, float]:
    if not isinstance(stage, Mapping):
        reasons.append(f"missing_{field}")
        return {}
    if not stage.get("evidence_id"):
        reasons.append(f"missing_{field}_evidence_id")
    scores = stage.get("scores")
    if not isinstance(scores, Mapping) or not scores:
        reasons.append(f"missing_{field}_scores")
        return {}
    output: dict[str, float] = {}
    for label, value in scores.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            reasons.append(f"invalid_{field}_scores")
            continue
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            reasons.append(f"invalid_{field}_scores")
            continue
        output[str(label)] = number
    return output


def _support(scores: Mapping[str, float], labels: Sequence[str]) -> float:
    return max((float(scores.get(label, 0.0)) for label in labels), default=0.0)


def _group_support(
    scores: Mapping[str, float],
    groups: Sequence[Sequence[str]],
) -> float:
    """Require every semantic group while allowing aliases within a group."""
    return min((_support(scores, labels) for labels in groups), default=0.0)


def _record_audit(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    reasons: list[str] = []
    for field in (
        "chain_episode_id",
        "counterfactual_pair_id",
        "scene_id",
        "history_id",
        "world_intervention_id",
        "generated_future_id",
        "planner_id",
        "planner_run_id",
        "planner_nuisance_seed",
    ):
        if record.get(field) is None or record.get(field) == "":
            reasons.append(f"missing_{field}")

    chain_type = str(record.get("chain_type", ""))
    template = CAUSAL_CHAIN_TEMPLATES.get(chain_type)
    if template is None:
        reasons.append("unknown_chain_type")
        template = {}
    world_state = str(record.get("world_state", ""))
    if world_state not in {"risk", "clear"}:
        reasons.append("invalid_world_state")

    trigger = record.get("trigger")
    trigger_label = ""
    if not isinstance(trigger, Mapping):
        reasons.append("missing_trigger")
    else:
        if not trigger.get("evidence_id"):
            reasons.append("missing_trigger_evidence_id")
        trigger_label = str(trigger.get("label", ""))
        if not trigger_label:
            reasons.append("missing_trigger_label")
        if trigger.get("source") not in INDEPENDENT_TRIGGER_SOURCES:
            reasons.append("trigger_source_not_independent")
        allowed_triggers = template.get(f"{world_state}_triggers", ())
        if allowed_triggers and trigger_label not in allowed_triggers:
            reasons.append("trigger_label_inconsistent_with_chain_state")

    imagined = record.get("imagined_consequence")
    imagined_scores = _stage_scores(imagined, "imagined_consequence", reasons)
    if isinstance(imagined, Mapping):
        if imagined.get("source") not in FORMAL_IMAGINED_SOURCES:
            reasons.append("unfrozen_or_missing_imagined_source")
        observability = imagined.get("observability")
        try:
            observability_value = float(observability)
        except (TypeError, ValueError):
            observability_value = -1.0
        if not np.isfinite(observability_value) or not 0.0 <= observability_value <= 1.0:
            reasons.append("invalid_imagined_observability")
        if bool(imagined.get("abstain", False)):
            reasons.append("imagined_probe_abstained")

    response = record.get("selected_response")
    response_scores = _stage_scores(response, "selected_response", reasons)
    if isinstance(response, Mapping) and response.get("source") not in ACTION_SOURCES:
        reasons.append("invalid_selected_response_source")

    outcome_reasons: list[str] = []
    outcome = record.get("realized_outcome")
    outcome_scores = _stage_scores(outcome, "realized_outcome", outcome_reasons)
    if isinstance(outcome, Mapping):
        if outcome.get("source") not in INDEPENDENT_OUTCOME_SOURCES:
            outcome_reasons.append("realized_outcome_source_not_independent")
        if not isinstance(outcome.get("task_success"), (bool, np.bool_)):
            outcome_reasons.append("missing_task_success")

    if template:
        allowed_imagined = set(template["risk_consequences"]) | set(template["safe_consequences"])
        allowed_responses = set(template["protective_responses"]) | set(template["permissive_responses"])
        allowed_outcomes = set(template["safe_outcomes"]) | set(template["failure_outcomes"])
        if set(imagined_scores) - allowed_imagined:
            reasons.append("unknown_imagined_consequence_label")
        if set(response_scores) - allowed_responses:
            reasons.append("unknown_selected_response_label")
        if set(outcome_scores) - allowed_outcomes:
            outcome_reasons.append("unknown_realized_outcome_label")

    evidence_ready = not reasons
    outcome_ready = evidence_ready and not outcome_reasons
    return {
        "index": index,
        "chain_episode_id": str(record.get("chain_episode_id", index)),
        "counterfactual_pair_id": record.get("counterfactual_pair_id"),
        "chain_type": chain_type or None,
        "world_state": world_state or None,
        "evidence_ready": evidence_ready,
        "outcome_ready": outcome_ready,
        "evidence_reasons": sorted(set(reasons)),
        "outcome_reasons": sorted(set(outcome_reasons)),
        "imagined_scores": imagined_scores,
        "response_scores": response_scores,
        "outcome_scores": outcome_scores,
    }


def _pair_reasons(records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if len(records) != 2:
        reasons.append("pair_must_contain_exactly_risk_and_clear")
        return reasons
    states = [str(record.get("world_state", "")) for record in records]
    if sorted(states) != ["clear", "risk"]:
        reasons.append("pair_must_contain_exactly_risk_and_clear")
    for field in ("chain_type", "scene_id", "history_id", "planner_id", "planner_nuisance_seed"):
        values = {str(record.get(field, "")) for record in records}
        if len(values) != 1:
            reasons.append(f"pair_{field}_not_held_fixed")
    for field in (
        "chain_episode_id",
        "world_intervention_id",
        "generated_future_id",
        "planner_run_id",
    ):
        values = [str(record.get(field, "")) for record in records]
        if len(set(values)) != len(values):
            reasons.append(f"pair_{field}_not_distinct")
    if not all(bool(row.get("evidence_ready")) for row in rows):
        reasons.append("pair_contains_invalid_evidence")
    if not all(bool(row.get("outcome_ready")) for row in rows):
        reasons.append("pair_contains_invalid_outcome")
    return sorted(set(reasons))


def audit_causal_chain_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit record evidence and paired counterfactual identifiability."""
    if not records:
        raise ValueError("at least one causal-chain record is required")
    rows = [_record_audit(record, index) for index, record in enumerate(records)]
    indices_by_episode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_episode[str(row["chain_episode_id"])].append(index)
    for indices in indices_by_episode.values():
        if len(indices) <= 1:
            continue
        for index in indices:
            rows[index]["evidence_reasons"].append("duplicate_chain_episode_id")
            rows[index]["evidence_reasons"] = sorted(set(rows[index]["evidence_reasons"]))
            rows[index]["evidence_ready"] = False
            rows[index]["outcome_ready"] = False

    by_pair: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_pair[str(record.get("counterfactual_pair_id", ""))].append(index)

    pair_rows = []
    for pair_id, indices in sorted(by_pair.items()):
        pair_records = [records[index] for index in indices]
        audited = [rows[index] for index in indices]
        reasons = _pair_reasons(pair_records, audited)
        pair_rows.append({
            "counterfactual_pair_id": pair_id or None,
            "chain_type": pair_records[0].get("chain_type") if pair_records else None,
            "counterfactual_ready": not reasons,
            "reasons": reasons,
            "chain_episode_ids": [row["chain_episode_id"] for row in audited],
        })

    ready_types = {
        str(row["chain_type"])
        for row in pair_rows
        if row["counterfactual_ready"]
    }
    missing_chain_types = sorted(set(CAUSAL_CHAIN_TEMPLATES) - ready_types)
    all_pairs_ready = all(row["counterfactual_ready"] for row in pair_rows)
    return {
        "protocol": f"{PROTOCOL}-readiness",
        "required_chain_types": sorted(CAUSAL_CHAIN_TEMPLATES),
        "num_records": len(records),
        "num_pairs": len(pair_rows),
        "evidence_ready_records": sum(bool(row["evidence_ready"]) for row in rows),
        "outcome_ready_records": sum(bool(row["outcome_ready"]) for row in rows),
        "counterfactual_ready_pairs": sum(bool(row["counterfactual_ready"]) for row in pair_rows),
        "formal_counterfactual_ready": all_pairs_ready,
        "four_chain_suite_ready": all_pairs_ready and not missing_chain_types,
        "missing_chain_types": missing_chain_types,
        "rows": rows,
        "pairs": pair_rows,
    }


def _score_record(record: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    template = CAUSAL_CHAIN_TEMPLATES[str(record["chain_type"])]
    state = str(record["world_state"])
    imagined_scores = row["imagined_scores"]
    response_scores = row["response_scores"]
    outcome_scores = row["outcome_scores"]
    risk_support = _support(imagined_scores, template["risk_consequences"])
    safe_support = _support(imagined_scores, template["safe_consequences"])
    protective_support = _support(response_scores, template["protective_responses"])
    permissive_support = _support(response_scores, template["permissive_responses"])
    desired_imagined = risk_support if state == "risk" else safe_support
    desired_response = _group_support(
        response_scores,
        RESPONSE_REQUIREMENT_GROUPS[str(record["chain_type"])][state],
    )
    safe_outcome = _group_support(
        outcome_scores,
        OUTCOME_REQUIREMENT_GROUPS[str(record["chain_type"])],
    )
    outcome = record["realized_outcome"]
    task_success = bool(outcome["task_success"])
    return {
        "chain_episode_id": record["chain_episode_id"],
        "world_state": state,
        "risk_consequence_support": risk_support,
        "safe_consequence_support": safe_support,
        "protective_response_support": protective_support,
        "permissive_response_support": permissive_support,
        "desired_imagined_support": desired_imagined,
        "desired_response_support": desired_response,
        "safe_outcome_support": safe_outcome,
        "task_success": task_success,
        "stage_bottleneck": min(desired_imagined, desired_response, safe_outcome),
    }


def _mean_or_none(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def evaluate_causal_chain_records(
    records: list[dict[str, Any]],
    *,
    minimum_contrast: float = 0.20,
) -> dict[str, Any]:
    """Score risk/clear contrasts and independent outcomes for each chain."""
    if not 0.0 <= float(minimum_contrast) <= 1.0:
        raise ValueError("minimum_contrast must be in [0,1]")
    audit = audit_causal_chain_records(records)
    rows_by_id = {str(row["chain_episode_id"]): row for row in audit["rows"]}
    records_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_pair[str(record.get("counterfactual_pair_id", ""))].append(record)
    audit_pair_by_id = {
        str(row["counterfactual_pair_id"]): row for row in audit["pairs"]
    }

    scored_pairs = []
    for pair_id, pair_records in sorted(records_by_pair.items()):
        pair_audit = audit_pair_by_id[pair_id]
        if not pair_audit["counterfactual_ready"]:
            scored_pairs.append({
                "counterfactual_pair_id": pair_id or None,
                "chain_type": pair_audit["chain_type"],
                "evaluable": False,
                "reasons": pair_audit["reasons"],
            })
            continue
        by_state = {str(record["world_state"]): record for record in pair_records}
        risk_record = by_state["risk"]
        clear_record = by_state["clear"]
        risk = _score_record(risk_record, rows_by_id[str(risk_record["chain_episode_id"])])
        clear = _score_record(clear_record, rows_by_id[str(clear_record["chain_episode_id"])])
        imagined_contrast = (
            risk["risk_consequence_support"] - clear["risk_consequence_support"]
        )
        action_contrast = (
            risk["protective_response_support"] - clear["protective_response_support"]
        )
        aligned = bool(
            imagined_contrast >= float(minimum_contrast)
            and action_contrast >= float(minimum_contrast)
        )
        outcome_support = min(
            risk["safe_outcome_support"], clear["safe_outcome_support"]
        )
        causal_score = min(
            max(0.0, imagined_contrast),
            max(0.0, action_contrast),
            risk["desired_imagined_support"],
            clear["desired_imagined_support"],
            risk["desired_response_support"],
            clear["desired_response_support"],
            outcome_support,
        )
        scored_pairs.append({
            "counterfactual_pair_id": pair_id,
            "chain_type": risk_record["chain_type"],
            "evaluable": True,
            "reasons": [],
            "imagined_risk_contrast": float(imagined_contrast),
            "protective_action_contrast": float(action_contrast),
            "directionally_aligned": aligned,
            "independent_safe_outcome_support": float(outcome_support),
            "joint_chain_success": bool(
                aligned and risk["task_success"] and clear["task_success"]
            ),
            "causal_chain_score": float(causal_score),
            "risk": risk,
            "clear": clear,
        })

    evaluable = [row for row in scored_pairs if row["evaluable"]]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluable:
        by_type[str(row["chain_type"])].append(row)

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "num_pairs": len(rows),
            "mean_imagined_risk_contrast": _mean_or_none(rows, "imagined_risk_contrast"),
            "mean_protective_action_contrast": _mean_or_none(rows, "protective_action_contrast"),
            "directional_alignment_rate": _mean_or_none(rows, "directionally_aligned"),
            "joint_chain_success_rate": _mean_or_none(rows, "joint_chain_success"),
            "mean_causal_chain_score": _mean_or_none(rows, "causal_chain_score"),
        }

    summaries_by_type = {
        chain_type: summary(by_type.get(chain_type, []))
        for chain_type in sorted(CAUSAL_CHAIN_TEMPLATES)
    }
    complete_types = [
        row for row in summaries_by_type.values() if row["num_pairs"] > 0
    ]
    macro_score = (
        None
        if len(complete_types) != len(CAUSAL_CHAIN_TEMPLATES)
        else float(np.mean([row["mean_causal_chain_score"] for row in complete_types]))
    )
    return {
        "protocol": PROTOCOL,
        "minimum_contrast": float(minimum_contrast),
        "num_records": len(records),
        "num_pairs": len(scored_pairs),
        "num_evaluable_pairs": len(evaluable),
        "counterfactual_coverage": (
            0.0 if not scored_pairs else len(evaluable) / len(scored_pairs)
        ),
        "four_chain_suite_ready": audit["four_chain_suite_ready"],
        "macro_mean_causal_chain_score": macro_score,
        "overall": summary(evaluable),
        "by_chain_type": summaries_by_type,
        "pairs": scored_pairs,
        "readiness": audit,
    }
