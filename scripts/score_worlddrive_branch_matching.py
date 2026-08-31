#!/usr/bin/env python3
"""Score whether decoded WorldDrive futures match their injected branch."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from iac_new.trajectory_decode import compare_continuous_trajectory


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {str(row["sample_id"]): row for row in _read(args.manifest)}
    scores = {str(row["sample_id"]): row for row in _read(args.scores)}
    details = []
    grouped = defaultdict(list)
    for sample_id, row in manifest.items():
        score = scores.get(sample_id)
        if score is None or not score.get("valid"):
            details.append({"sample_id": sample_id, "status": "missing_or_invalid"})
            continue
        predicted = np.asarray(score["decoder"]["trajectory"], dtype=np.float64)
        times = np.asarray(row["future_times_s"], dtype=np.float64)
        comparisons = {}
        for candidate in row["candidates"]:
            comparisons[candidate["candidate_id"]] = compare_continuous_trajectory(
                predicted,
                np.asarray(candidate["trajectory"], dtype=np.float64),
                times,
                score_speed=False,
            )
        target = str(row["gt_candidate_id"])
        ordered = sorted(comparisons, key=lambda key: comparisons[key]["weighted_mean_joint_error"])
        wrong = [key for key in ordered if key != target]
        item = {
            "sample_id": sample_id,
            "counterfactual_group_id": row["counterfactual_group_id"],
            "branch_mode": row["branch_mode"],
            "status": "ok",
            "target_candidate_id": target,
            "predicted_candidate_id": ordered[0],
            "target_is_top1": ordered[0] == target,
            "target_error": comparisons[target]["weighted_mean_joint_error"],
            "best_wrong_error": comparisons[wrong[0]]["weighted_mean_joint_error"],
            "diagonal_margin": comparisons[wrong[0]]["weighted_mean_joint_error"] - comparisons[target]["weighted_mean_joint_error"],
            "decoded_endpoint": predicted[-1].tolist(),
            "action_endpoint": next(c["trajectory"][-1] for c in row["candidates"] if c["candidate_id"] == target),
            "candidate_errors": {
                key: value["weighted_mean_joint_error"] for key, value in comparisons.items()
            },
        }
        details.append(item)
        grouped[row["counterfactual_group_id"]].append(item)

    valid = [row for row in details if row.get("status") == "ok"]
    lateral_pairs = []
    yaw_pairs = []
    for branches in grouped.values():
        for first_index, first in enumerate(branches):
            for second in branches[first_index + 1:]:
                action_delta_y = first["action_endpoint"][1] - second["action_endpoint"][1]
                decoded_delta_y = first["decoded_endpoint"][1] - second["decoded_endpoint"][1]
                action_delta_yaw = first["action_endpoint"][2] - second["action_endpoint"][2]
                decoded_delta_yaw = first["decoded_endpoint"][2] - second["decoded_endpoint"][2]
                if action_delta_y != 0:
                    lateral_pairs.append(float(action_delta_y * decoded_delta_y > 0))
                if action_delta_yaw != 0:
                    yaw_pairs.append(float(action_delta_yaw * decoded_delta_yaw > 0))

    report = {
        "protocol": "worlddrive-external-action-branch-matching-v1",
        "evidence_tier": "action_response_probe",
        "native_action_head_recorded": False,
        "causal_claim_eligible": False,
        "rows": len(details),
        "rows_valid": len(valid),
        "diagonal_top1": None if not valid else float(np.mean([row["target_is_top1"] for row in valid])),
        "mean_diagonal_margin": None if not valid else float(np.mean([row["diagonal_margin"] for row in valid])),
        "endpoint_lateral_pairwise_accuracy": None if not lateral_pairs else float(np.mean(lateral_pairs)),
        "endpoint_yaw_pairwise_accuracy": None if not yaw_pairs else float(np.mean(yaw_pairs)),
        "details": details,
        "interpretation": "Tests image recovery under external trajectory interventions; native planner output is still required for formal CCFC.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
