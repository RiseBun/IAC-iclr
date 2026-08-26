#!/usr/bin/env python3
"""Measure RAFT against the known historical ego-motion on state-aware NAVSIM clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from iac_new.flow import RaftFlowExtractor
from iac_new.geometry import ground_plane_homography, homography_flow, se2_to_transform
from iac_new.perception import build_perception, temporal_road_consensus
from iac_new.protocol import read_jsonl, validate_record
from iac_new.scoring import polygon_mask


def _metrics(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(observed).all(-1) & np.isfinite(predicted).all(-1)
    if int(valid.sum()) < 20:
        return {"num_points": int(valid.sum()), "median_epe_px": None}
    error = np.linalg.norm(observed[valid] - predicted[valid], axis=1)
    obs = observed[valid]
    pred = predicted[valid]
    on = np.linalg.norm(obs, axis=1)
    pn = np.linalg.norm(pred, axis=1)
    good = (on > 0.25) & (pn > 0.25)
    cosine = np.sum(obs[good] * pred[good], axis=1) / (on[good] * pn[good]) if good.any() else np.asarray([])
    return {
        "num_points": int(valid.sum()),
        "median_epe_px": float(np.median(error)),
        "p75_epe_px": float(np.quantile(error, 0.75)),
        "median_direction_cosine": float(np.median(cosine)) if cosine.size else None,
        "median_observed_flow_px": float(np.median(on)),
        "median_predicted_flow_px": float(np.median(pn)),
        "median_signed_error_x_px": float(np.median((observed[valid] - predicted[valid])[:, 0])),
        "median_signed_error_y_px": float(np.median((observed[valid] - predicted[valid])[:, 1])),
    }


def _band_metrics(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    height = observed.shape[0]
    # The evaluation ROI begins at roughly 0.53H, so a band above 0.50H
    # contains no valid pixels.  Keep the split inside the visible road ROI.
    bands = {
        "far": (np.arange(height) >= int(0.53 * height)) & (np.arange(height) < int(0.62 * height)),
        "mid": (np.arange(height) >= int(0.62 * height)) & (np.arange(height) < int(0.72 * height)),
        "near": np.arange(height) >= int(0.72 * height),
    }
    return {name: _metrics(observed, predicted, mask & rows[:, None]) for name, rows in bands.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--fb-abs-threshold-px", type=float, default=1.5)
    parser.add_argument("--fb-relative-threshold", type=float, default=0.05)
    parser.add_argument("--semantic-filter", action="store_true", help="restrict audit to SegFormer road pixels and suppress actors")
    args = parser.parse_args()
    raw_rows = read_jsonl(args.manifest)
    rows = [validate_record(row, manifest_root=args.manifest.parent) for row in raw_rows]
    if args.max_samples:
        rows = rows[: args.max_samples]
    extractor = RaftFlowExtractor(
        model_size="large", device=args.device, updates=32, batch_size=4,
        forward_backward=True, fb_abs_threshold_px=args.fb_abs_threshold_px,
        fb_relative_threshold=args.fb_relative_threshold,
    )
    roi = polygon_mask(args.height, args.width, [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]])
    perception = None
    if args.semantic_filter:
        perception = build_perception({
            "perception": {
                "enabled": True, "backend": "segformer", "model_id": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
                "local_files_only": True, "traversable_labels": ["road"],
                "actor_labels": ["car", "truck", "bus", "person", "rider", "bicycle", "motorcycle"],
                "confidence_threshold": 0.55, "constraint_mode": "soft", "temporal_consensus": True,
            }
        }, device=args.device)
    output = []
    for index, record in enumerate(rows, start=1):
        observation = extractor.observe(record["frame_paths"], record["intrinsics"], record["distortion"], (args.width, args.height))
        semantic_masks = None
        if perception is not None:
            semantic = perception.observe(record["frame_paths"], target_size=(args.width, args.height), intrinsics=record["intrinsics"], distortion=record["distortion"])
            semantic = temporal_road_consensus(semantic, observation.forward, road_dilation_px=4, actor_dilation_px=3)
            semantic_masks = np.asarray(semantic.traversable_masks[:len(observation.forward)], dtype=bool) & ~np.asarray(semantic.actor_masks[:len(observation.forward)], dtype=bool)
        state = np.asarray((record.get("metadata") or {}).get("history_ego_state"), dtype=np.float64)
        if state.ndim != 2 or state.shape[0] < 2 or state.shape[1] < 3:
            continue
        # State rows are ego poses expressed in the anchor (last-history) frame.
        poses = [se2_to_transform(*row[:3]) @ np.asarray(record["camera_to_ego"], dtype=np.float64) for row in state]
        per_interval = []
        for interval in range(min(len(state) - 1, len(observation.forward))):
            next_from_current = np.linalg.inv(poses[interval + 1]) @ poses[interval]
            H = ground_plane_homography(observation.intrinsics, next_from_current, poses[interval])
            predicted, valid = homography_flow(H, args.height, args.width)
            mask = roi & valid
            if observation.consistency_masks is not None:
                mask &= observation.consistency_masks[interval]
            if semantic_masks is not None:
                mask &= semantic_masks[interval]
            metrics = _metrics(observation.forward[interval], predicted, mask)
            metrics["bands"] = _band_metrics(observation.forward[interval], predicted, mask)
            per_interval.append(metrics)
        output.append({"sample_id": record["sample_id"], "scene_id": record["scene_id"], "history_intervals": per_interval})
        print(json.dumps({"completed": index, "total": len(rows)}), flush=True)
    finite = [item for row in output for item in row["history_intervals"] if item.get("median_epe_px") is not None]
    band_summary = {}
    for band in ("far", "mid", "near"):
        values = [item["bands"][band] for item in finite if item.get("bands", {}).get(band, {}).get("median_epe_px") is not None]
        band_summary[band] = {
            "num_intervals": len(values),
            "median_epe_px": float(np.median([x["median_epe_px"] for x in values])) if values else None,
            "p75_epe_px": float(np.quantile([x["median_epe_px"] for x in values], 0.75)) if values else None,
            "median_direction_cosine": float(np.median([x["median_direction_cosine"] for x in values if x.get("median_direction_cosine") is not None])) if values else None,
            "median_observed_flow_px": float(np.median([x["median_observed_flow_px"] for x in values])) if values else None,
        }
    summary = {
        "protocol": "historical-pose-flow-audit-v1",
        "num_samples": len(output),
        "num_intervals": len(finite),
        "median_epe_px": float(np.median([x["median_epe_px"] for x in finite])) if finite else None,
        "p75_epe_px": float(np.quantile([x["median_epe_px"] for x in finite], 0.75)) if finite else None,
        "median_direction_cosine": float(np.median([x["median_direction_cosine"] for x in finite if x["median_direction_cosine"] is not None])) if finite else None,
        "bands": band_summary,
        "rows": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
