#!/usr/bin/env python3
"""Audit native WorldDrive action/future pairs against identity and order controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.trajectory_decode import compare_continuous_trajectory


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _error(predicted: np.ndarray, action: np.ndarray, times: np.ndarray) -> float:
    return float(compare_continuous_trajectory(predicted, action, times, score_speed=False)["weighted_mean_joint_error"])


def _bootstrap(values: list[float], *, seed: int = 20260901, draws: int = 20000) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": None, "confidence_interval_95": None}
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "confidence_interval_95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reversed-manifest", type=Path, required=True)
    parser.add_argument("--reversed-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = _read(args.manifest)
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    scores = {str(row["sample_id"]): row for row in _read(args.scores)}
    reversed_manifest = _read(args.reversed_manifest)
    reversed_to_original = {
        str(row["sample_id"]): str(row["control_of_sample_id"]) for row in reversed_manifest
    }
    reversed_scores = {str(row["sample_id"]): row for row in _read(args.reversed_scores)}
    actions = {
        sample_id: np.asarray(row["action_trajectory"], dtype=np.float64)
        for sample_id, row in manifest.items()
    }

    details = []
    identity_deltas = []
    reversal_deltas = []
    identity_top1 = []
    normal_beats_reversal = []
    for sample_id, row in manifest.items():
        score = scores.get(sample_id)
        if score is None or not score.get("valid"):
            details.append({"sample_id": sample_id, "status": "missing_or_invalid_normal"})
            continue
        predicted = np.asarray(score["decoder"]["trajectory"], dtype=np.float64)
        times = np.asarray(row["future_times_s"], dtype=np.float64)
        correct = _error(predicted, actions[sample_id], times)
        wrong = {
            other_id: _error(predicted, other_action, times)
            for other_id, other_action in actions.items()
            if other_id != sample_id
        }
        mean_wrong = float(np.mean(list(wrong.values())))
        best_wrong = float(min(wrong.values()))
        identity_deltas.append(mean_wrong - correct)
        identity_top1.append(float(correct < best_wrong))

        reverse_ids = [key for key, original in reversed_to_original.items() if original == sample_id]
        if len(reverse_ids) != 1:
            raise ValueError(f"{sample_id}: expected one time-reversal control")
        reverse_score = reversed_scores.get(reverse_ids[0])
        if reverse_score is None or not reverse_score.get("valid"):
            details.append({"sample_id": sample_id, "status": "missing_or_invalid_reversal"})
            continue
        reverse_predicted = np.asarray(reverse_score["decoder"]["trajectory"], dtype=np.float64)
        reverse_error = _error(reverse_predicted, actions[sample_id], times)
        reversal_deltas.append(reverse_error - correct)
        normal_beats_reversal.append(float(correct < reverse_error))
        details.append({
            "sample_id": sample_id,
            "status": "ok",
            "correct_action_error": correct,
            "mean_wrong_identity_error": mean_wrong,
            "best_wrong_identity_error": best_wrong,
            "identity_margin_mean_wrong_minus_correct": mean_wrong - correct,
            "identity_top1": correct < best_wrong,
            "time_reversed_error": reverse_error,
            "time_reversal_delta": reverse_error - correct,
            "normal_beats_time_reversal": correct < reverse_error,
        })

    valid = [row for row in details if row.get("status") == "ok"]
    report = {
        "protocol": "worlddrive-native-action-future-specificity-v1",
        "evidence_tier": "native_action_future_pair_pilot",
        "native_action_head_recorded": True,
        "candidate_bank_used_by_decoder": False,
        "speed_scored": False,
        "rows": len(manifest),
        "rows_valid": len(valid),
        "identity_top1": float(np.mean(identity_top1)) if identity_top1 else None,
        "identity_margin": _bootstrap(identity_deltas),
        "normal_beats_time_reversal": float(np.mean(normal_beats_reversal)) if normal_beats_reversal else None,
        "time_reversal_error_increase": _bootstrap(reversal_deltas, seed=20260902),
        "causal_claim_eligible": False,
        "causal_claim_blocker": "five samples and no paired upstream intervention producing two native selected actions for the same scene",
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
