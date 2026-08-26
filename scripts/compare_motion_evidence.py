#!/usr/bin/env python3
"""Compare RAFT and CoTracker curvature observability on matched samples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _auc(scores: np.ndarray, target: np.ndarray) -> float | None:
    positive, negative = scores[target], scores[~target]
    if len(positive) == 0 or len(negative) == 0:
        return None
    return float((positive[:, None] > negative[None, :]).mean() + 0.5 * (positive[:, None] == negative[None, :]).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raft-summary", type=Path, required=True)
    parser.add_argument("--cotracker-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-threshold-1pm", type=float, default=0.01)
    args = parser.parse_args()
    raft = json.loads(args.raft_summary.read_text(encoding="utf-8"))["results"]
    cotracker = json.loads(args.cotracker_report.read_text(encoding="utf-8"))["rows"]
    cotracker_by_id = {str(row["sample_id"]): row for row in cotracker}
    rows = []
    for row in raft:
        sample_id = str(row["sample_id"])
        if sample_id not in cotracker_by_id:
            continue
        intervals = row.get("observability_by_future_interval", [])
        contrast = [float(item["curvature_lateral_contrast_rad"]) for item in intervals if item.get("curvature_lateral_contrast_rad") is not None]
        error = row.get("comparison_to_logged_trajectory", {}).get("mean_curvature_abs_1pm")
        if not contrast or error is None:
            continue
        point = cotracker_by_id[sample_id]
        rows.append({
            "sample_id": sample_id,
            "scene_id": str(row.get("scene_id")),
            "curvature_error_1pm": float(error),
            "high_error": float(error) >= args.error_threshold_1pm,
            "raft_lateral_contrast_rad": float(np.mean(contrast)),
            "cotracker_evidence": float(point["curvature_evidence"]),
            "cotracker_valid_fraction": float(point["valid_fraction"]),
        })
    target = np.asarray([bool(row["high_error"]) for row in rows])
    raft_scores = -np.asarray([row["raft_lateral_contrast_rad"] for row in rows])
    cotracker_scores = -np.asarray([row["cotracker_evidence"] for row in rows])
    by_scene: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_scene[row["scene_id"]].append(index)
    scenes = []
    for scene, indices in sorted(by_scene.items()):
        selected = np.asarray(indices)
        scene_target = target[selected]
        scenes.append({
            "scene_id": scene,
            "rows": len(selected),
            "high_error_fraction": float(scene_target.mean()),
            "raft_auc": _auc(raft_scores[selected], scene_target),
            "cotracker_auc": _auc(cotracker_scores[selected], scene_target),
        })
    output = {
        "protocol": "matched-motion-evidence-ab-v1",
        "rows": len(rows),
        "scenes": len(by_scene),
        "high_error_threshold_1pm": args.error_threshold_1pm,
        "high_error_fraction": float(target.mean()) if len(target) else None,
        "raft_low_contrast_auc": _auc(raft_scores, target),
        "cotracker_low_evidence_auc": _auc(cotracker_scores, target),
        "mean_cotracker_valid_fraction": float(np.mean([row["cotracker_valid_fraction"] for row in rows])) if rows else None,
        "scene_results": scenes,
        "recommendation": "Keep CoTracker diagnostic-only unless it improves scene-disjoint holdout over RAFT; do not fuse from this report alone.",
        "rows_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key not in {"rows_detail", "scene_results"}}, indent=2))


if __name__ == "__main__":
    main()
