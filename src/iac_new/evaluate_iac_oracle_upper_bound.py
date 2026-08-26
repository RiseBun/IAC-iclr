#!/usr/bin/env python3
"""Measure candidate-blind IAC error with known synthetic camera motion.

The experiment supplies the decoder with flow generated from the same camera
geometry used by the decoder.  It therefore measures the implementation and
parameterization ceiling, separately from RAFT, monocular depth, and WAM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.scoring import polygon_mask
from iac_new.trajectory_decode import (
    _sparse_predicted_flows,
    compare_continuous_trajectory,
    decode_continuous_trajectory,
    integrate_piecewise_controls,
)


def _make_flow(trajectory: np.ndarray, intrinsics: np.ndarray, camera_to_ego: np.ndarray, size: tuple[int, int], noise_std: float, rng: np.random.Generator) -> np.ndarray:
    width, height = size
    yy, xx = np.indices((height, width), dtype=np.float64)
    pixels = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    sparse, valid = _sparse_predicted_flows(
        trajectory,
        camera_to_ego,
        intrinsics,
        pixels,
        image_size=size,
        depths_m=None,
    )
    flow = np.full((len(trajectory), height, width, 2), np.nan, dtype=np.float32)
    for index in range(len(trajectory)):
        flat = flow[index].reshape(-1, 2)
        flat[valid[index]] = sparse[index, valid[index]].astype(np.float32)
        if noise_std > 0.0:
            noise = rng.normal(0.0, noise_std, size=flat.shape).astype(np.float32)
            flat[valid[index]] += noise[valid[index]]
    return flow


def run_case(noise_std: float, *, seed: int = 7) -> dict[str, object]:
    width, height = 160, 90
    times = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    intrinsics = np.asarray([[140.0, 0.0, 80.0], [0.0, 140.0, 45.0], [0.0, 0.0, 1.0]])
    camera_to_ego = np.eye(4, dtype=np.float64)
    # Camera: x-right, y-down, z-forward. Ego: x-forward, y-left, z-up.
    camera_to_ego[:3, :3] = np.asarray(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    camera_to_ego[2, 3] = 1.5
    reference = integrate_piecewise_controls(
        times,
        speeds_mps=np.asarray([4.0, 4.0, 4.0, 4.0]),
        curvatures_1pm=np.asarray([0.0, 0.015, 0.025, 0.03]),
    )
    observed = _make_flow(reference, intrinsics, camera_to_ego, (width, height), noise_std, np.random.default_rng(seed))
    roi = polygon_mask(height, width, [[0.05, 0.98], [0.95, 0.98], [0.80, 0.45], [0.20, 0.45]])
    decoded = decode_continuous_trajectory(
        observed_flows=observed,
        camera_to_ego=camera_to_ego,
        intrinsics=intrinsics,
        future_times_s=times,
        roi_mask=roi,
        dynamic_weights=np.ones(observed.shape[:-1], dtype=np.float32),
        image_size=(width, height),
        minimum_flow_scale_px=0.25,
        max_points=900,
        max_iterations=16,
        initial_speeds_mps=(2.0, 4.0, 8.0),
        profile_radius=0.12,
        interval_observability=np.ones(len(times), dtype=np.float64),
        curvature_multistart=True,
    )
    comparison = compare_continuous_trajectory(
        np.asarray(decoded["trajectory"], dtype=np.float64),
        reference,
        times,
        observability=np.ones(len(times), dtype=np.float64),
        score_speed=False,
    )
    return {
        "noise_std_px": float(noise_std),
        "comparison": comparison,
        "speed_support": decoded["speed_support"],
        "decoded_trajectory": decoded["trajectory"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-levels", default="0.0,0.25,0.5,1.0")
    args = parser.parse_args()
    rows = [run_case(float(value)) for value in args.noise_levels.split(",") if value.strip()]
    summary = {
        "protocol": "iac-synthetic-known-motion-upper-bound-v1",
        "assumptions": {
            "camera_model": "known pinhole",
            "road_geometry": "known planar road",
            "motion": "known SE(2) trajectory used to synthesize flow",
            "primary_score": "lateral+yaw+curvature; speed diagnostic only",
        },
        "rows": rows,
        "best_case": rows[0] if rows else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
