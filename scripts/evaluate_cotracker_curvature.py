#!/usr/bin/env python3
"""Run a CoTracker3 curvature-evidence A/B against the existing RAFT report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.cotracker import CoTrackerExtractor, point_track_curvature_features


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _auc(scores: np.ndarray, target: np.ndarray) -> float | None:
    positive = scores[target]
    negative = scores[~target]
    if len(positive) == 0 or len(negative) == 0:
        return None
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * ties)


def _summary(rows: list[dict[str, Any]], error_threshold: float) -> dict[str, Any]:
    valid = [row for row in rows if row.get("curvature_evidence") is not None and row.get("curvature_error_1pm") is not None]
    if not valid:
        return {"rows": 0, "auc_low_contrast_for_high_error": None}
    evidence = np.asarray([float(row["curvature_evidence"]) for row in valid])
    errors = np.asarray([float(row["curvature_error_1pm"]) for row in valid])
    target = errors >= float(error_threshold)
    return {
        "rows": len(valid),
        "high_error_threshold_1pm": float(error_threshold),
        "high_error_fraction": float(target.mean()),
        "auc_low_evidence_for_high_error": _auc(-evidence, target),
        "mean_curvature_evidence": float(np.mean(evidence)),
        "median_curvature_evidence": float(np.median(evidence)),
        "mean_valid_fraction": float(np.mean([float(row["valid_fraction"]) for row in valid])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raft-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--target-height", type=int, default=288)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--error-threshold-1pm", type=float, default=0.01)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--model-name", default="cotracker3_offline")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    manifest = _read_jsonl(args.manifest)
    raft_summary = json.loads(args.raft_summary.read_text(encoding="utf-8"))
    raft_by_sample = {
        str(row["sample_id"]): row
        for row in raft_summary.get("results", [])
        if row.get("sample_id") is not None
    }
    if args.max_samples > 0:
        manifest = manifest[: args.max_samples]
    polygon = [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]]
    extractor = CoTrackerExtractor(
        device=args.device,
        grid_size=args.grid_size,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, record in enumerate(manifest, start=1):
        sample_id = str(record.get("sample_id") or record.get("source_key") or record.get("video_id") or index)
        paths = list(record.get("history_frame_paths") or record.get("history_images") or [])
        paths += list(record.get("future_frame_paths") or record.get("future_images") or [])
        if len(paths) < 2:
            errors.append({"sample_id": sample_id, "error": "fewer than two frames"})
            continue
        try:
            observation = extractor.observe(
                paths,
                target_size=(args.target_width, args.target_height),
                polygon_normalized=polygon,
            )
            features = point_track_curvature_features(
                observation,
                future_start_interval=max(len(record.get("history_frame_paths") or record.get("history_images") or []) - 1, 0),
                confidence_threshold=args.confidence_threshold,
            )
            future = [item for item in features if item.get("future_interval")]
            contrast = [item["curvature_lateral_contrast_rad"] for item in future if item.get("curvature_lateral_contrast_rad") is not None]
            support = [min(float(item["left_support_fraction"]), float(item["right_support_fraction"])) for item in future]
            mean_contrast = float(np.mean(contrast)) if contrast else None
            mean_bilateral_support = float(np.mean(support)) if support else 0.0
            evidence = float((mean_contrast or 0.0) * min(mean_bilateral_support / 0.10, 1.0))
            mean_valid_fraction = float(np.mean([float(item["valid_fraction"]) for item in future])) if future else 0.0
            row = {
                "sample_id": sample_id,
                "scene_id": str(record.get("scene_id") or record.get("scene_name")),
                "curvature_lateral_contrast_rad": mean_contrast,
                "mean_bilateral_support": mean_bilateral_support,
                "valid_fraction": mean_valid_fraction,
                "curvature_evidence": evidence,
                "curvature_evidence_reason": "bilateral_tracks" if contrast else "insufficient_bilateral_tracks",
                "intervals": future,
                "curvature_error_1pm": raft_by_sample.get(sample_id, {}).get("comparison_to_logged_trajectory", {}).get("mean_curvature_abs_1pm"),
                "backend": "cotracker3",
            }
            rows.append(row)
            print(json.dumps({"completed": index, "total": len(manifest)}), flush=True)
        except Exception as error:  # preserve per-sample failure provenance
            errors.append({"sample_id": sample_id, "error": str(error)})
    output = {
        "protocol": "cotracker3-curvature-evidence-ab-v1",
        "manifest": str(args.manifest.resolve()),
        "raft_summary": str(args.raft_summary.resolve()),
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "grid_size": args.grid_size,
        "num_input": len(manifest),
        "num_scored": len(rows),
        "num_error": len(errors),
        "error_threshold_1pm": args.error_threshold_1pm,
        "summary": _summary(rows, args.error_threshold_1pm),
        "errors": errors,
        "rows": rows,
        "interpretation": "CoTracker is an auxiliary point-track evidence probe; it does not replace the candidate-blind IAC decoder in this experiment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key not in {"rows", "errors"}}, indent=2))


if __name__ == "__main__":
    main()
