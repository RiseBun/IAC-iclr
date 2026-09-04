#!/usr/bin/env python3
"""Build compact DriveWAM NAVSIM inputs from the private v3 manifest.

DriveWAM resizes frames to 256x448 before encoding.  Resizing once here keeps
the v3 input reproducible while avoiding a 70+ GB copy of 1920x1080 images.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image(path: str, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = _rows(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(rows):
        history = list(row.get("history_images") or [])
        future = list(row.get("future_images") or [])
        if len(history) != 4 or len(future) != 8:
            raise ValueError(f"{row.get('sample_id')}: expected 4 history + 8 future frames")
        image_paths = history + future
        images = np.stack([_image(path, (448, 256)) for path in image_paths], axis=0)
        states = list(row.get("history_ego_state") or [])
        realized = list(row.get("realized_future_ego_state") or [])
        action = list(row.get("trajectory") or [])
        if len(states) != 4 or len(realized) != 8 or len(action) != 8:
            raise ValueError(f"{row.get('sample_id')}: invalid state/action length")
        anchor = states[-1]
        command = np.asarray(row.get("driving_command", [0, 1, 0, 0]), dtype=np.float32)
        if command.shape != (4,):
            command = np.asarray([0, 1, 0, 0], dtype=np.float32)
        sample = {
            "images": images,
            "history_poses": np.asarray([list(map(float, state[:3])) for state in states], dtype=np.float32),
            "ego_status": {
                "velocity": np.asarray([float(anchor[3]) if len(anchor) > 3 else 0.0, 0.0], dtype=np.float32),
                "acceleration": np.zeros(2, dtype=np.float32),
                "driving_command": command,
            },
            "future_trajectory": [
                {"pose": list(map(float, point[:3])), "velocity": float(realized[i][3]), "acceleration": 0.0}
                for i, point in enumerate(action)
            ],
            "metadata": {
                "benchmark_id": row.get("benchmark_id"),
                "benchmark": "benchmark_v3",
                "dataset": "navsim",
                "source_key": row["source_key"],
                "sample_id": row.get("sample_id", row["source_key"]),
                "source_pkl": row.get("source_pkl"),
                "scene_name": row.get("scene_name"),
                "scene_token": row.get("scene_token", ""),
                "scene_group": row.get("scene_group"),
                "stratum": row.get("stratum"),
                "frame_idx": row.get("frame_idx"),
                "timestamp_us": row.get("timestamp_us"),
                "future_times_s": row.get("future_times_s"),
                "history_ego_state": states,
                "realized_future_ego_state": realized,
                "action_trajectory": action,
                "action_trajectory_source": "navsim_native_realized_oracle_for_input_condition",
                "future_images_source": "navsim_native_realized_for_input_condition",
                "image_paths": image_paths,
                "camera_intrinsic": row.get("camera_intrinsic"),
                "camera_distortion": row.get("camera_distortion"),
                "camera_to_ego": row.get("camera_to_ego"),
            },
        }
        sample_path = args.output / f"sample_{index:06d}.pkl"
        with sample_path.open("wb") as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        manifest.append({"sample_index": index, "sample": str(sample_path), "source_key": row["source_key"], "sample_id": row.get("sample_id", row["source_key"])})
        if (index + 1) % 50 == 0:
            print(json.dumps({"completed": index + 1, "total": len(rows)}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "num_samples": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
