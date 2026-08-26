#!/usr/bin/env python3
"""Create a compact, split-aware report from an image motion-probe audit.

The probe prediction is loaded from a separate score file. Candidate actions
are used only for the post-hoc paired CC and action-swap control; they are not
used to produce ``predicted_motion``. The script keeps calibration and holdout
numbers together so a positive holdout result cannot be confused with a
calibration result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.analyze_wam_native_pairs import _branch, _metric_for_pair, read_jsonl
except ModuleNotFoundError:
    from analyze_wam_native_pairs import _branch, _metric_for_pair, read_jsonl  # type: ignore[no-redef]


def split_report(manifest_path: Path, scores_path: Path) -> dict[str, Any]:
    manifests = read_jsonl(manifest_path)
    scores = read_jsonl(scores_path)
    score_by_video = {str(row["video_id"]): row for row in scores}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = []
    for row in manifests:
        video_id = str(row["video_id"])
        score = score_by_video.get(video_id)
        if score is None:
            missing.append(video_id)
            continue
        if not isinstance(score.get("predicted_motion"), dict):
            raise ValueError(f"{video_id}: score has no candidate-blind predicted_motion")
        groups[str(row["twin_id"])].append(_branch(row, score))
    results = []
    invalid = []
    for twin_id, branches in sorted(groups.items()):
        if len(branches) != 2:
            invalid.append({"twin_id": twin_id, "branches": len(branches)})
            continue
        times = np.asarray(
            next(row["frame_times_s"] for row in manifests if str(row["twin_id"]) == twin_id),
            dtype=np.float64,
        )
        results.append(_metric_for_pair(branches[0], branches[1], times))
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in results if row.get(key) is not None]
    cc = values("counterfactual_consistency")
    swap = [float(row["swapped_action_control"]["counterfactual_consistency"]) for row in results]
    matrix_top1 = [float(row["action_image_matrix"]["diagonal_top1_accuracy"]) for row in results]
    matrix_margin = [float(row["action_image_matrix"]["mean_cc_margin"]) for row in results]
    probe = [float(row["probe_condition_accuracy"]) for row in results]
    return {
        "manifest": str(manifest_path),
        "scores": str(scores_path),
        "pairs": len(results),
        "missing_scores": missing,
        "invalid_groups": invalid,
        "mean_counterfactual_consistency": float(np.mean(cc)) if cc else None,
        "median_counterfactual_consistency": float(np.median(cc)) if cc else None,
        "mean_swapped_action_control": float(np.mean(swap)) if swap else None,
        "counterfactual_lift_over_swap": float(np.mean(cc) - np.mean(swap)) if cc and swap else None,
        "probe_condition_accuracy": float(np.mean(probe)) if probe else None,
        "action_image_matrix_diagonal_top1": float(np.mean(matrix_top1)) if matrix_top1 else None,
        "action_image_matrix_mean_cc_margin": float(np.mean(matrix_margin)) if matrix_margin else None,
        "calibration_status": "missing_generated_camera_calibration",
        "metric_ego_projection_valid": False,
        "candidate_blind_prediction_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--holdout-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "protocol": "wam-candidate-blind-probe-split-report-v1",
        "calibration": split_report(args.calibration_manifest, args.calibration_scores),
        "holdout": split_report(args.holdout_manifest, args.holdout_scores),
        "interpretation": {
            "positive_evidence": "holdout action-image diagonal Top-1 and CC lift over the same-history action-swap control",
            "not_claimed": "metric ego trajectory or Foresight-Conditioned Success, because generated frames have no independent calibration, realized future ego state, or task-success label",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
