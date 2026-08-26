#!/usr/bin/env python3
"""Fit a small non-planar road-plane correction from causal history flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from iac_new.flow import RaftFlowExtractor
from iac_new.geometry import ground_plane_homography, homography_flow, se2_to_transform
from iac_new.perception import build_perception, temporal_road_consensus
from iac_new.protocol import read_jsonl, validate_record
from iac_new.scoring import polygon_mask


def _plane_homography(K: np.ndarray, transform: np.ndarray, camera_to_anchor: np.ndarray, params: np.ndarray) -> np.ndarray:
    normal_anchor = np.asarray([-float(params[0]), -float(params[1]), 1.0], dtype=np.float64)
    offset_anchor = float(params[2])
    pose = np.asarray(camera_to_anchor, dtype=np.float64)
    normal_camera = pose[:3, :3].T @ normal_anchor
    distance = float(normal_anchor @ pose[:3, 3] + offset_anchor)
    if abs(distance) < 1e-5:
        raise ValueError("fitted plane is too close to the camera")
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    translation = np.asarray(transform, dtype=np.float64)[:3, 3:4]
    normalized = rotation - translation @ normal_camera.reshape(1, 3) / distance
    H = np.asarray(K, dtype=np.float64) @ normalized @ np.linalg.inv(np.asarray(K, dtype=np.float64))
    return H / H[2, 2]


def _project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((len(points), 1))], axis=1).T
    projected = np.asarray(H, dtype=np.float64) @ hom
    return (projected[:2] / np.maximum(projected[2:3], 1e-8)).T


def _sample_points(flow: np.ndarray, mask: np.ndarray, max_points: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(flow).all(axis=-1)
    valid &= np.linalg.norm(flow, axis=-1) > 0.5
    yy, xx = np.nonzero(valid)
    if len(xx) > max_points:
        choice = np.linspace(0, len(xx) - 1, max_points).astype(int)
        xx, yy = xx[choice], yy[choice]
    points = np.stack([xx, yy], axis=1).astype(np.float64)
    return points, flow[yy, xx].astype(np.float64)


def _epe(flow: np.ndarray, H: np.ndarray, points: np.ndarray, vectors: np.ndarray) -> float | None:
    if len(points) == 0:
        return None
    residual = np.linalg.norm((_project(H, points) - points) - vectors, axis=1)
    return float(np.median(residual))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--semantic-filter", action="store_true")
    args = parser.parse_args()
    rows = [validate_record(row, manifest_root=args.manifest.parent) for row in read_jsonl(args.manifest)]
    rows = rows[: args.max_samples] if args.max_samples else rows
    extractor = RaftFlowExtractor(model_size="large", device=args.device, updates=32, batch_size=4,
                                  forward_backward=True, fb_abs_threshold_px=1.5, fb_relative_threshold=0.05)
    roi = polygon_mask(args.height, args.width, [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]])
    perception = None
    if args.semantic_filter:
        perception = build_perception({"perception": {"enabled": True, "backend": "segformer",
            "model_id": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", "local_files_only": True,
            "traversable_labels": ["road"], "actor_labels": ["car", "truck", "bus", "person", "rider", "bicycle", "motorcycle"],
            "confidence_threshold": 0.55, "constraint_mode": "soft", "temporal_consensus": True}}, device=args.device)
    outputs = []
    for index, record in enumerate(rows, start=1):
        flow = extractor.observe(record["frame_paths"], record["intrinsics"], record["distortion"], (args.width, args.height))
        state = np.asarray((record.get("metadata") or {}).get("history_ego_state"), dtype=np.float64)
        if state.ndim != 2 or state.shape[0] < 4 or state.shape[1] < 3:
            continue
        semantic_masks = None
        if perception is not None:
            sem = perception.observe(record["frame_paths"], target_size=(args.width, args.height), intrinsics=record["intrinsics"], distortion=record["distortion"])
            sem = temporal_road_consensus(sem, flow.forward, road_dilation_px=4, actor_dilation_px=3)
            semantic_masks = np.asarray(sem.traversable_masks[:len(flow.forward)], dtype=bool) & ~np.asarray(sem.actor_masks[:len(flow.forward)], dtype=bool)
        poses = [se2_to_transform(*row[:3]) @ np.asarray(record["camera_to_ego"], dtype=np.float64) for row in state]
        K = flow.intrinsics
        samples = []
        for interval in range(2):
            mask = roi & flow.consistency_masks[interval]
            if semantic_masks is not None:
                mask &= semantic_masks[interval]
            points, vectors = _sample_points(flow.forward[interval], mask)
            if len(points) >= 20:
                transform = np.linalg.inv(poses[interval + 1]) @ poses[interval]
                samples.append((interval, points, vectors, transform, poses[interval]))
        if len(samples) < 1:
            continue
        def residual(params: np.ndarray) -> np.ndarray:
            chunks = []
            for _, points, vectors, transform, pose in samples:
                try:
                    H = _plane_homography(K, transform, pose, params)
                    chunks.append((_project(H, points) - points - vectors).reshape(-1))
                except ValueError:
                    return np.full(2 * sum(len(item[1]) for item in samples), 1e3)
            return np.concatenate(chunks)
        fit = least_squares(residual, np.zeros(3), bounds=([-0.25, -0.25, -1.0], [0.25, 0.25, 1.0]),
                            loss="soft_l1", f_scale=2.0, max_nfev=60)
        params = fit.x.astype(np.float64)
        # Evaluate causal held-out interval 2, plus baseline/fitted training error.
        metrics = {"plane_params": params.tolist(), "fit_cost": float(fit.cost), "fit_success": bool(fit.success)}
        for interval in range(min(3, len(flow.forward))):
            mask = roi & flow.consistency_masks[interval]
            if semantic_masks is not None:
                mask &= semantic_masks[interval]
            points, vectors = _sample_points(flow.forward[interval], mask, max_points=3000)
            transform = np.linalg.inv(poses[interval + 1]) @ poses[interval]
            baseline_H = ground_plane_homography(K, transform, poses[interval])
            fitted_H = _plane_homography(K, transform, poses[interval], params)
            metrics[f"interval_{interval}"] = {"num_points": int(len(points)),
                "baseline_median_epe_px": _epe(flow.forward[interval], baseline_H, points, vectors),
                "fitted_median_epe_px": _epe(flow.forward[interval], fitted_H, points, vectors)}
        outputs.append({"sample_id": record["sample_id"], "scene_id": record["scene_id"], **metrics})
        print(json.dumps({"completed": index, "total": len(rows)}), flush=True)
    summary = {"protocol": "causal-road-plane-calibration-v1", "num_samples": len(outputs), "rows": outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
