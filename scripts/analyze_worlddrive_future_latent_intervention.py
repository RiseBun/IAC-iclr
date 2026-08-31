#!/usr/bin/env python3
"""Summarize WorldDrive's internal future-latent intervention response."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch-a", default="future_native")
    parser.add_argument("--branch-b", default="future_reverse")
    parser.add_argument("--min-max-lateral-delta-m", type=float, default=1.0)
    parser.add_argument("--min-max-yaw-delta-rad", type=float, default=0.1)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["counterfactual_group_id"])][str(row["branch_id"])] = row

    details = []
    for group_id, branches in sorted(grouped.items()):
        if set((args.branch_a, args.branch_b)).difference(branches):
            raise ValueError(f"{group_id}: incomplete branch pair")
        first = np.asarray(branches[args.branch_a]["stage2_trajectory"], dtype=np.float64)
        second = np.asarray(branches[args.branch_b]["stage2_trajectory"], dtype=np.float64)
        delta = second - first
        delta[:, 2] = (delta[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
        max_delta = np.max(np.abs(delta), axis=0)
        rank_a = int(branches[args.branch_a]["stage2_selected_topk_rank"])
        rank_b = int(branches[args.branch_b]["stage2_selected_topk_rank"])
        material = bool(
            max_delta[1] >= args.min_max_lateral_delta_m
            or max_delta[2] >= args.min_max_yaw_delta_rad
        )
        details.append({
            "counterfactual_group_id": group_id,
            "selected_rank_a": rank_a,
            "selected_rank_b": rank_b,
            "selected_rank_changed": rank_a != rank_b,
            "trajectory_changed": bool(float(np.max(np.abs(delta))) > 1e-9),
            "max_abs_longitudinal_delta_m": float(max_delta[0]),
            "max_abs_lateral_delta_m": float(max_delta[1]),
            "max_abs_yaw_delta_rad": float(max_delta[2]),
            "primary_lateral_yaw_material_gate": material,
        })

    total = len(details)
    rank_changed = sum(item["selected_rank_changed"] for item in details)
    trajectory_changed = sum(item["trajectory_changed"] for item in details)
    material = sum(item["primary_lateral_yaw_material_gate"] for item in details)
    report = {
        "protocol": "worlddrive-internal-future-latent-intervention-audit-v1",
        "intervention": "fixed_reverse_candidate_future_assignment_[4,3,2,1,0]",
        "held_fixed": [
            "history", "ego_status", "topk_trajectories", "topk_embeddings",
            "visual_tokens", "model_weights",
        ],
        "groups": total,
        "selected_rank_changed": rank_changed,
        "selected_rank_changed_fraction": None if not total else rank_changed / total,
        "trajectory_changed": trajectory_changed,
        "trajectory_changed_fraction": None if not total else trajectory_changed / total,
        "primary_lateral_yaw_material_groups": material,
        "primary_lateral_yaw_material_fraction": None if not total else material / total,
        "primary_lateral_yaw_material_wilson_95": _wilson(material, total),
        "thresholds_pre_registered_before_pixel_generation": {
            "min_max_lateral_delta_m": args.min_max_lateral_delta_m,
            "min_max_yaw_delta_rad": args.min_max_yaw_delta_rad,
        },
        "semantic_hazard_intervention": False,
        "claim_scope": "internal_foresight_mediation",
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
