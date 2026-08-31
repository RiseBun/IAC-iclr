#!/usr/bin/env python3
"""Convert a frozen Level-1 manifest into WorldDrive planner input pickles."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image(path: str) -> np.ndarray:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(value)
    return np.asarray(Image.open(value).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_root / "drivewam_samples_logged"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for index, row in enumerate(_read(args.manifest)):
        history_paths = list(row.get("history_frame_paths") or [])
        future_paths = list(row.get("future_frame_paths") or [])
        if len(history_paths) != 4 or len(future_paths) != 8:
            raise ValueError(f"{row.get('sample_id')}: expected 4 history + 8 future frames")
        images = [_image(path) for path in history_paths + future_paths]
        if len({image.shape for image in images}) != 1:
            raise ValueError(f"{row.get('sample_id')}: image shapes differ")

        metadata = dict(row.get("metadata") or {})
        history_state = np.asarray(metadata.get("history_ego_state"), dtype=np.float32)
        if history_state.shape[0] != 4 or history_state.shape[1] < 4:
            raise ValueError(f"{row.get('sample_id')}: expected four history ego states")
        candidate = next(
            item for item in row["candidates"] if item["candidate_id"] == row["gt_candidate_id"]
        )
        trajectory = np.asarray(candidate["trajectory"], dtype=np.float32)
        if trajectory.shape != (8, 3):
            raise ValueError(f"{row.get('sample_id')}: expected logged 8x3 trajectory")
        times = np.asarray(row["future_times_s"], dtype=np.float32)
        interval = np.diff(np.concatenate([[0.0], times]))
        displacement = np.linalg.norm(
            np.diff(np.vstack([np.zeros((1, 2), dtype=np.float32), trajectory[:, :2]]), axis=0),
            axis=1,
        )
        speeds = displacement / np.maximum(interval, 1e-6)
        command_value = row.get("driving_command", metadata.get("driving_command"))
        if command_value is None:
            command = np.asarray([0, 1, 0, 0], dtype=np.float32)
            command_source = "default_straight_missing_native_field"
        else:
            command = np.asarray(command_value, dtype=np.float32)
            command_source = "level1_manifest_native_field"
        if command.shape != (4,):
            raise ValueError(f"{row.get('sample_id')}: driving command must be 4-D")
        acceleration = float(history_state[-1, 4]) if history_state.shape[1] > 4 else 0.0
        sample = {
            "images": np.stack(images, axis=0),
            "history_poses": history_state[:, :3],
            "ego_status": {
                "velocity": np.asarray([history_state[-1, 3], 0.0], dtype=np.float32),
                "acceleration": np.asarray([acceleration, 0.0], dtype=np.float32),
                "driving_command": command,
            },
            "future_trajectory": [
                {"pose": trajectory[position], "velocity": float(speeds[position]), "acceleration": 0.0}
                for position in range(8)
            ],
            "metadata": {
                "dataset": "navsim",
                "source_key": metadata.get("source_key", row.get("source_key", row["sample_id"])),
                "scene_token": row.get("scene_id", ""),
                "camera_intrinsic": row["intrinsics"],
                "camera_to_ego": row["camera_to_ego"],
                "image_paths": history_paths + future_paths,
                "manifest_role": metadata.get("manifest_role"),
                "command_source": command_source,
            },
        }
        output_path = output_dir / f"sample_{index:06d}.pkl"
        with output_path.open("wb") as stream:
            pickle.dump(sample, stream, protocol=pickle.HIGHEST_PROTOCOL)
        index_rows.append({
            "sample_index": index,
            "sample_id": row["sample_id"],
            "source_key": sample["metadata"]["source_key"],
            "sample": str(output_path.resolve()),
            "command_source": command_source,
        })

    index_path = args.output_root / "manifest.json"
    index_path.write_text(json.dumps(index_rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "samples": len(index_rows),
        "command_sources": sorted({row["command_source"] for row in index_rows}),
    }, indent=2))


if __name__ == "__main__":
    main()
