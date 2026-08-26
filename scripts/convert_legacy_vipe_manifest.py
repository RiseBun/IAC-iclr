#!/usr/bin/env python3
"""Convert the existing ViPE holdout manifest and depth caches to iac_new."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def structured_candidates(raw: dict) -> list[dict]:
    supported = int(raw["supported_candidate_index"])
    exact = np.asarray(
        next(
            candidate["candidate_traj"]
            for candidate in raw["candidates"]
            if int(candidate["candidate_index"]) == supported
        ),
        dtype=np.float64,
    )
    candidates = [
        {"candidate_id": "logged", "prior": 1.0, "trajectory": exact.tolist()}
    ]
    for factor in (0.8, 0.9, 1.1, 1.2):
        trajectory = exact.copy()
        trajectory[:, 0] *= factor
        candidates.append(
            {
                "candidate_id": f"scale_{factor:.1f}".replace(".", "p"),
                "prior": 1.0,
                "trajectory": trajectory.tolist(),
            }
        )
    curvature_observable = bool(
        np.max(np.abs(exact[:, 1])) >= 0.3
        or np.max(np.abs(exact[:, 2])) >= 0.03
    )
    if curvature_observable:
        for factor in (-1.0, 0.0, 0.5, 1.5):
            trajectory = exact.copy()
            trajectory[:, 1] *= factor
            trajectory[:, 2] *= factor
            candidates.append(
                {
                    "candidate_id": f"curvature_{factor:.1f}".replace("-", "m").replace(".", "p"),
                    "prior": 1.0,
                    "trajectory": trajectory.tolist(),
                }
            )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--depth-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-bank",
        choices=("structured", "legacy"),
        default="structured",
    )
    args = parser.parse_args()
    rows = []
    for raw_line in args.legacy_manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        raw = json.loads(raw_line)
        index = int(raw["video_index"])
        cache_path = args.depth_cache_dir / f"video_{index:03d}.npz"
        cache = np.load(cache_path)
        is_history4_future4 = (
            int(raw.get("history_frame_count", 0)) == 4
            and int(raw.get("future_frame_count", 0)) == 4
            and len(raw.get("frame_paths") or []) == 8
        )
        if is_history4_future4:
            frame_paths = list(raw["frame_paths"])
            timestamps = np.asarray(raw["frame_timestamps_us"], dtype=np.float64)
            timestamps = (timestamps - timestamps[3]) / 1_000_000.0
        else:
            frame_indices = cache["frame_indices"].astype(np.int64).tolist()
            sequence_indices = [frame_indices[0], *[value + 1 for value in frame_indices]]
            frame_paths = [raw["frame_paths"][value] for value in sequence_indices]
            timestamps = np.asarray(raw["frame_timestamps_us"], dtype=np.float64)[sequence_indices]
            timestamps = (timestamps - timestamps[0]) / 1_000_000.0
        if args.candidate_bank == "structured":
            candidates = structured_candidates(raw)
            gt_candidate_id = "logged"
        else:
            candidates = [
                {
                    "candidate_id": str(candidate["candidate_index"]),
                    "prior": 1.0,
                    "trajectory": candidate["candidate_traj"],
                }
                for candidate in raw["candidates"]
            ]
            gt_candidate_id = str(raw["supported_candidate_index"])
        output_row = {
                "sample_id": str(raw["video_pair_id"]),
                "scene_id": str(raw["scene_name"]),
                "intrinsics": cache["original_intrinsics"].tolist(),
                "distortion": cache["distortion"].tolist(),
                "camera_to_ego": cache["camera_to_ego"].tolist(),
                "metric_depth_path": str(cache_path.resolve()),
                "gt_candidate_id": gt_candidate_id,
                "candidates": candidates,
                "metadata": {
                    "legacy_video_index": index,
                    "control_type": raw.get("control_type"),
                    "source_scene": raw.get("scene_name"),
                },
            }
        if is_history4_future4:
            output_row.update(
                {
                    "history_frame_paths": frame_paths[:4],
                    "history_times_s": timestamps[:4].tolist(),
                    "future_frame_paths": frame_paths[4:],
                    "future_times_s": timestamps[4:].tolist(),
                }
            )
        else:
            output_row.update(
                {
                    "frame_paths": frame_paths,
                    "frame_times_s": timestamps.tolist(),
                }
            )
        rows.append(output_row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
