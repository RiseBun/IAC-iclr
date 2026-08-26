#!/usr/bin/env python3
"""Validate whether image-side curvature evidence predicts decoder error.

The threshold is selected only from scene-disjoint calibration scenes.  It is
then evaluated on held-out scenes, so it cannot be a local threshold fitted to
one road sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _balanced_scene_folds(scene_ids: list[str], folds: int) -> dict[str, int]:
    """Assign complete scenes to folds with nearly equal scene counts."""
    ordered = sorted(
        scene_ids,
        key=lambda scene_id: hashlib.sha256(
            f"iac-curvature-observability-v1:{scene_id}".encode()
        ).digest(),
    )
    return {scene_id: index % folds for index, scene_id in enumerate(ordered)}


def _balanced_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    positives = target.sum()
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    true_positive_rate = (predicted & target).sum() / positives
    true_negative_rate = ((~predicted) & (~target)).sum() / negatives
    return float((true_positive_rate + true_negative_rate) / 2.0)


def _auc(scores: np.ndarray, target: np.ndarray) -> float:
    """Mann-Whitney AUC; larger score means the target is more likely."""
    positive = scores[target]
    negative = scores[~target]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def _best_threshold(evidence: np.ndarray, high_error: np.ndarray) -> tuple[float, float]:
    # Low lateral contrast should mean an unreliable curvature estimate.
    candidates = np.unique(evidence)
    candidates = (candidates[:-1] + candidates[1:]) / 2.0 if len(candidates) > 1 else candidates
    scores = [_balanced_accuracy(evidence <= threshold, high_error) for threshold in candidates]
    index = int(np.nanargmax(scores))
    return float(candidates[index]), float(scores[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--high-error-quantile", type=float, default=0.75)
    parser.add_argument(
        "--high-error-threshold-1pm",
        type=float,
        default=None,
        help="Use one fixed curvature-error target instead of a per-fold quantile.",
    )
    args = parser.parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")
    if not 0.0 < args.high_error_quantile < 1.0:
        raise SystemExit("--high-error-quantile must be between 0 and 1")
    if args.high_error_threshold_1pm is not None and args.high_error_threshold_1pm <= 0.0:
        raise SystemExit("--high-error-threshold-1pm must be positive")

    results = json.loads(args.summary.read_text(encoding="utf-8"))["results"]
    rows = []
    for result in results:
        intervals = result.get("observability_by_future_interval", [])
        contrast = [item.get("curvature_lateral_contrast_rad") for item in intervals]
        contrast = [float(value) for value in contrast if value is not None]
        error = result.get("comparison_to_logged_trajectory", {}).get("mean_curvature_abs_1pm")
        scene_id = result.get("scene_id")
        if contrast and error is not None and scene_id is not None:
            rows.append((str(scene_id), float(np.mean(contrast)), float(error)))
    if len(rows) < args.folds * 2:
        raise SystemExit("not enough valid rows for group-disjoint validation")

    scene_ids = sorted({scene_id for scene_id, _, _ in rows})
    evidence = np.asarray([row[1] for row in rows])
    errors = np.asarray([row[2] for row in rows])
    fold_by_scene = _balanced_scene_folds(scene_ids, args.folds)
    scene_folds = np.asarray([fold_by_scene[row[0]] for row in rows])
    report_folds = []
    for fold in range(args.folds):
        test = scene_folds == fold
        train = ~test
        if not test.any() or not train.any():
            continue
        error_threshold = (
            float(args.high_error_threshold_1pm)
            if args.high_error_threshold_1pm is not None
            else float(np.quantile(errors[train], args.high_error_quantile))
        )
        train_target = errors[train] >= error_threshold
        threshold, train_bacc = _best_threshold(evidence[train], train_target)
        test_target = errors[test] >= error_threshold
        report_folds.append({
            "fold": fold,
            "calibration_scenes": sorted({row[0] for index, row in enumerate(rows) if train[index]}),
            "holdout_scenes": sorted({row[0] for index, row in enumerate(rows) if test[index]}),
            "calibration_rows": int(train.sum()),
            "holdout_rows": int(test.sum()),
            "high_curvature_error_threshold_1pm": error_threshold,
            "low_contrast_threshold_rad": threshold,
            "calibration_balanced_accuracy": train_bacc,
            "holdout_balanced_accuracy": _balanced_accuracy(evidence[test] <= threshold, test_target),
            "holdout_low_contrast_auc": _auc(-evidence[test], test_target),
            "holdout_high_error_fraction": float(test_target.mean()),
        })
    output = {
        "protocol": "scene-disjoint-curvature-observability-v1",
        "input_summary": str(args.summary.resolve()),
        "rows": len(rows),
        "scenes": len(scene_ids),
        "high_error_target": (
            {"kind": "absolute", "threshold_1pm": args.high_error_threshold_1pm}
            if args.high_error_threshold_1pm is not None
            else {"kind": "calibration_quantile", "quantile": args.high_error_quantile}
        ),
        "folds": report_folds,
        "mean_holdout_balanced_accuracy": float(np.nanmean([item["holdout_balanced_accuracy"] for item in report_folds])),
        "mean_holdout_low_contrast_auc": float(np.nanmean([item["holdout_low_contrast_auc"] for item in report_folds])),
        "interpretation": "A useful abstention signal needs holdout AUC and balanced accuracy materially above 0.5; otherwise retain the evidence as a diagnostic only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
