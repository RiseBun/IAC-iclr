#!/usr/bin/env python3
"""Create a one-branch IAC manifest from an Epona generated frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--future", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = [str(path) for path in sorted(args.history_dir.glob("*.png"))[:10]]
    if len(history) != 10:
        raise ValueError(f"expected 10 history frames, found {len(history)}")
    # Synthetic calibration is only for the smoke path; native dataset
    # experiments must replace it with source camera calibration.
    row = {
        "sample_id": "epona_iac_smoke",
        "scene_id": "iac_smoke",
        "history_frame_paths": history,
        "future_frame_paths": [str(path) for path in args.future],
        "history_times_s": [-1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0],
        "future_times_s": [0.2 * (index + 1) for index in range(len(args.future))],
        "intrinsics": [[500.0, 0.0, 224.0], [0.0, 500.0, 128.0], [0.0, 0.0, 1.0]],
        "distortion": [],
        "camera_to_ego": [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]],
        "candidates": [
            {"candidate_id": "logged", "trajectory": [[0.6 * (index + 1), 0.0, 0.0] for index in range(len(args.future))], "prior": 1.0},
            {"candidate_id": "counterfactual_left", "trajectory": [[0.6 * (index + 1), 0.15 * (index + 1), 0.04 * (index + 1)] for index in range(len(args.future))], "prior": 1.0},
        ],
        "gt_candidate_id": "logged",
        "metadata": {"protocol": "epona-iac-smoke", "future_images_source": "epona"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
