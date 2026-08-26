#!/usr/bin/env python3
"""Convert exact NAVSIM native records into DriveWAM's pickle contract.

The selected records are joined by the immutable ``source_key``.  No temporal
nearest-neighbour lookup is used: the four history and eight future images are
the native images already present in the record.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _image(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def _finite_pose(state):
    values = list(state or [])
    if len(values) < 3:
        raise ValueError("ego state must contain x, y, yaw")
    return [float(values[0]), float(values[1]), float(values[2])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="native records JSONL")
    ap.add_argument("--branches", required=True, help="cache-aligned branches JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--branch-mode", choices=("logged", "left", "right", "stop"), default="logged")
    args = ap.parse_args()

    native = {row["source_key"]: row for row in _load_jsonl(args.records)}
    branches = [row for row in _load_jsonl(args.branches) if row.get("branch_mode") == args.branch_mode]
    if args.limit > 0:
        branches = branches[: args.limit]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, branch in enumerate(branches):
        key = branch["source_key"]
        if key not in native:
            raise KeyError(f"missing native record for {key}")
        row = native[key]
        history_images = list(row.get("history_images") or [])
        future_images = list(row.get("future_images") or [])
        if len(history_images) != 4 or len(future_images) != 8:
            raise ValueError(f"{key}: expected 4 history + 8 future images")
        paths = history_images + future_images
        images = [_image(p) for p in paths]
        if len({tuple(image.shape) for image in images}) != 1:
            raise ValueError(f"{key}: image shapes differ")

        history_states = list(row.get("history_ego_state") or [])
        future_states = list(row.get("realized_future_ego_state") or [])
        if len(history_states) != 4 or len(future_states) != 8:
            raise ValueError(f"{key}: expected 4 history + 8 future ego states")
        action = list(branch.get("action_trajectory") or row.get("trajectory") or [])
        if len(action) != 8:
            raise ValueError(f"{key}: expected 8 action points")
        speeds = [float(state[3]) if len(state) > 3 else 0.0 for state in future_states]
        anchor = history_states[-1]
        command = np.asarray(row.get("driving_command", [0, 1, 0, 0]), dtype=np.float32)
        if command.shape != (4,):
            command = np.asarray([0, 1, 0, 0], dtype=np.float32)
        sample = {
            "images": np.stack(images, axis=0),
            "history_poses": np.asarray([_finite_pose(state) for state in history_states], dtype=np.float32),
            "ego_status": {
                "velocity": np.asarray([float(anchor[3]) if len(anchor) > 3 else 0.0, 0.0], dtype=np.float32),
                "acceleration": np.zeros(2, dtype=np.float32),
                "driving_command": command,
            },
            "future_trajectory": [
                {"pose": _finite_pose(point), "velocity": speeds[i], "acceleration": 0.0}
                for i, point in enumerate(action)
            ],
            "metadata": {
                "dataset": row.get("dataset", "navsim"),
                "source_key": key,
                "source_pkl": row.get("source_pkl"),
                "scene_name": row.get("scene_name"),
                "scene_token": row.get("scene_token", ""),
                "frame_idx": row.get("frame_idx"),
                "timestamp_us": row.get("timestamp_us"),
                "branch_id": branch.get("branch_id"),
                "counterfactual_group_id": branch.get("counterfactual_group_id"),
                "branch_mode": branch.get("branch_mode", "logged"),
                "action_trajectory": action,
                "camera_intrinsic": row.get("camera_intrinsic"),
                "camera_to_ego": row.get("camera_to_ego"),
                "image_paths": paths,
            },
        }
        sample_path = out / f"sample_{idx:06d}.pkl"
        with open(sample_path, "wb") as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
        manifest.append({
            "sample_index": idx,
            "sample": str(sample_path),
            "source_key": key,
            "branch_id": branch.get("branch_id"),
            "counterfactual_group_id": branch.get("counterfactual_group_id"),
            "branch_mode": branch.get("branch_mode", "logged"),
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "num_samples": len(manifest), "branches": len(branches)}, indent=2))


if __name__ == "__main__":
    main()
