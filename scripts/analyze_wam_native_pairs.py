#!/usr/bin/env python3
"""Evaluate the running DrivingWorld WAM on paired future/action branches.

The WAM manifest provides the action-conditioned candidate motion. The paired
motion-probe output provides an independent estimate of the motion represented
by each generated future. This is a causal response audit, not a visual-quality
score. Task success is reported only when an explicit success field exists.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.action_image_matrix import decoded_trajectory_cross_matrix
from iac_new.calibration import calibration_status
from iac_new.wam_metrics import (
    counterfactual_response_alignment,
    foresight_action_compatibility,
    paired_counterfactual_consistency,
    realized_state_counterfactual_consistency,
)
from iac_new.state_protocol import states_to_trajectory


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def motion_to_trajectory(motion: dict[str, Any], times: np.ndarray) -> np.ndarray:
    """Integrate constant body-frame motion into [x,y,yaw] knots."""
    forward = float(motion["forward_rate_mps"])
    lateral = float(motion["lateral_rate_mps"])
    yaw_rate = float(motion["yaw_rate_rps"])
    return np.asarray(
        [[forward * time_s, lateral * time_s, yaw_rate * time_s] for time_s in times],
        dtype=np.float64,
    )


def _branch(manifest: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    times = np.asarray(manifest["frame_times_s"], dtype=np.float64)
    supported = int(manifest["supported_candidate_index"])
    candidates = list(manifest["candidates"])
    action_motion = candidates[supported]["motion"]
    predicted_motion = dict(score["predicted_motion"])
    realized_states = manifest.get("realized_future_ego_state")
    realized_future = None
    if realized_states is not None:
        realized_future = states_to_trajectory(np.asarray(realized_states, dtype=np.float64)).tolist()
        if len(realized_future) != len(times):
            raise ValueError(f"{manifest['video_id']}: realized_future_ego_state length does not match frame_times_s")
    return {
        "branch_id": str(manifest["video_id"]),
        "condition_family": str(manifest.get("condition_family") or "unknown"),
        "imagined_future": motion_to_trajectory(predicted_motion, times).tolist(),
        "executed_action": motion_to_trajectory(action_motion, times).tolist(),
        "action_condition": motion_to_trajectory(action_motion, times).tolist(),
        "realized_future": realized_future,
        "task_success": score.get("task_success") if score.get("task_success") is not None else manifest.get("task_success"),
        "probe_correct": bool(score.get("correct", False)),
        "supported_candidate_index": supported,
        "predicted_motion": predicted_motion,
        "action_motion": action_motion,
        "calibration": calibration_status(manifest),
        "source_images": list(manifest.get("history_images") or manifest.get("future_images") or []),
        "state_reference_source": manifest.get("state_reference_source"),
        "task_success_source": manifest.get("task_success_source"),
    }


def _metric_for_pair(first: dict[str, Any], second: dict[str, Any], times: np.ndarray) -> dict[str, Any]:
    result = paired_counterfactual_consistency(first, second, times)
    swapped = paired_counterfactual_consistency(
        {**first, "executed_action": second["executed_action"]},
        {**second, "executed_action": first["executed_action"]},
        times,
    )
    result["swapped_action_control"] = {
        "counterfactual_consistency": swapped["counterfactual_consistency"],
        "ego_state_counterfactual_consistency": swapped["ego_state_counterfactual_consistency"],
        "response_alignment": swapped["response_alignment"],
    }
    result["probe_condition_accuracy"] = float(np.mean([first["probe_correct"], second["probe_correct"]]))
    result["condition_switch_detected"] = bool(first["probe_correct"] and second["probe_correct"])
    result["action_image_matrix"] = decoded_trajectory_cross_matrix(
        [first, second], times
    )
    if first.get("realized_future") is not None and second.get("realized_future") is not None:
        realized = realized_state_counterfactual_consistency(
            np.asarray(first["imagined_future"]),
            np.asarray(second["imagined_future"]),
            np.asarray(first["realized_future"]),
            np.asarray(second["realized_future"]),
            times,
        )
        result.update(realized)
    else:
        result.update({
            "realized_state_counterfactual_consistency": None,
            "mean_foresight_realized_state_compatibility": None,
            "realized_state_response_alignment": None,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    manifest_rows = read_jsonl(args.manifest)
    score_rows = read_jsonl(args.scores)
    score_by_video = {str(row["video_id"]): row for row in score_rows}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_scores = []
    for row in manifest_rows:
        video_id = str(row["video_id"])
        if video_id not in score_by_video:
            missing_scores.append(video_id)
            continue
        groups[str(row["twin_id"])].append(_branch(row, score_by_video[video_id]))
    pair_results = []
    for twin_id, branches in sorted(groups.items()):
        if len(branches) != 2:
            raise ValueError(f"{twin_id}: expected exactly two reciprocal branches, got {len(branches)}")
        times = np.asarray(next(row["frame_times_s"] for row in manifest_rows if str(row["twin_id"]) == twin_id), dtype=np.float64)
        result = _metric_for_pair(branches[0], branches[1], times)
        pair_results.append({
            "twin_id": twin_id,
            "condition_family": branches[0]["condition_family"],
            "branches": branches,
            **result,
        })
    cc = [float(row["counterfactual_consistency"]) for row in pair_results if row["counterfactual_consistency"] is not None]
    swapped = [float(row["swapped_action_control"]["counterfactual_consistency"]) for row in pair_results if row["swapped_action_control"]["counterfactual_consistency"] is not None]
    state_cc = [float(row["ego_state_counterfactual_consistency"]) for row in pair_results if row["ego_state_counterfactual_consistency"] is not None]
    state_swapped = [float(row["swapped_action_control"]["ego_state_counterfactual_consistency"]) for row in pair_results if row["swapped_action_control"]["ego_state_counterfactual_consistency"] is not None]
    realized_cc = [float(row["realized_state_counterfactual_consistency"]) for row in pair_results if row["realized_state_counterfactual_consistency"] is not None]
    probe = [float(row["probe_condition_accuracy"]) for row in pair_results]
    matrix_cc = [float(row["action_image_matrix"]["mean_cc_margin"]) for row in pair_results]
    matrix_top1 = [float(row["action_image_matrix"]["diagonal_top1_accuracy"]) for row in pair_results]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_results:
        by_family[row["condition_family"]].append(row)
    calibration_counts: dict[str, int] = {key: 0 for key in ("complete", "partial", "missing", "invalid")}
    for pair in pair_results:
        for branch in pair["branches"]:
            status = str(branch["calibration"]["status"])
            calibration_counts[status] = calibration_counts.get(status, 0) + 1
    summary = {
        "protocol": "wam-ego-state-paired-v1",
        "manifest": str(args.manifest),
        "scores": str(args.scores),
        "pairs": len(pair_results),
        "missing_scores": missing_scores,
        "mean_counterfactual_consistency": float(np.mean(cc)) if cc else None,
        "median_counterfactual_consistency": float(np.median(cc)) if cc else None,
        "mean_swapped_action_control": float(np.mean(swapped)) if swapped else None,
        "mean_probe_condition_accuracy": float(np.mean(probe)) if probe else None,
        "mean_action_image_matrix_cc_margin": float(np.mean(matrix_cc)) if matrix_cc else None,
        "action_image_matrix_diagonal_top1": float(np.mean(matrix_top1)) if matrix_top1 else None,
        "mean_ego_state_action_compatibility": float(np.mean([
            row["mean_ego_state_action_compatibility"] for row in pair_results
        ])) if pair_results else None,
        "mean_ego_state_counterfactual_consistency": float(np.mean(state_cc)) if state_cc else None,
        "median_ego_state_counterfactual_consistency": float(np.median(state_cc)) if state_cc else None,
        "mean_ego_state_swapped_action_control": float(np.mean(state_swapped)) if state_swapped else None,
        "ego_state_cc_lift_over_swapped_action": float(np.mean(state_cc) - np.mean(state_swapped)) if state_cc and state_swapped else None,
        "mean_realized_state_counterfactual_consistency": float(np.mean(realized_cc)) if realized_cc else None,
        "median_realized_state_counterfactual_consistency": float(np.median(realized_cc)) if realized_cc else None,
        "counterfactual_lift_over_swapped_action": float(np.mean(cc) - np.mean(swapped)) if cc and swapped else None,
        "state_reference": {
            "available": bool(realized_cc),
            "source": "dataset_logged_ego_state" if realized_cc else "action_condition",
            "fields": ["x_m", "y_m", "yaw_rad", "speed_mps", "yaw_rate_rps"],
            "independent_realized_future_ego_state": bool(realized_cc),
            "interpretation": (
                "compares generated-image motion with independent realized future ego state"
                if realized_cc else
                "tests whether generated images respond to the intervention; not closed-loop execution fidelity"
            ),
        },
        "calibration": {
            "schema": "camera-calibration-v1",
            "branch_counts": calibration_counts,
            "projectable_fraction": calibration_counts["complete"] / max(sum(calibration_counts.values()), 1),
            "projection_policy": "metric_ego only when complete; image_plane_only otherwise",
        },
        "by_condition_family": {
            family: {
                "pairs": len(values),
                "mean_counterfactual_consistency": float(np.mean([v["counterfactual_consistency"] for v in values])),
                "mean_swapped_action_control": float(np.mean([v["swapped_action_control"]["counterfactual_consistency"] for v in values])),
                "mean_probe_condition_accuracy": float(np.mean([v["probe_condition_accuracy"] for v in values])),
                "mean_ego_state_action_compatibility": float(np.mean([v["mean_ego_state_action_compatibility"] for v in values])),
                "mean_ego_state_counterfactual_consistency": float(np.mean([v["ego_state_counterfactual_consistency"] for v in values])),
            }
            for family, values in sorted(by_family.items())
        },
        "foresight_conditioned_success": {
            "status": "unavailable",
            "reason": "requires realized_future_ego_state and explicit task_success labels.",
        },
        "results": pair_results,
    }
    fcs_branches = [branch for pair in pair_results for branch in pair["branches"] if branch.get("realized_future") is not None]
    if fcs_branches and all(branch.get("task_success") is not None for branch in fcs_branches):
        from iac_new.wam_metrics import foresight_conditioned_success
        summary["foresight_conditioned_success"] = foresight_conditioned_success(
            fcs_branches, future_times_s=np.asarray(manifest_rows[0]["frame_times_s"], dtype=np.float64)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
