#!/usr/bin/env python3
"""Build DriveWAM-compatible samples directly from native NAVSIM scene pickles."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def rel_pose(frame, anchor):
    """Approximate ego-frame [forward, lateral, yaw] from NAVSIM global poses."""
    pa = np.asarray(anchor["ego2global_translation"], dtype=float)[:2]
    pp = np.asarray(frame["ego2global_translation"], dtype=float)[:2]
    # NAVSIM ego2global rotation is xyzw in these exports; yaw is encoded by the
    # planar rotation matrix, which is more robust than quaternion conventions.
    Ra = np.asarray(anchor["ego2global"], dtype=float)[:2, :2]
    d = Ra.T @ (pp - pa)
    Rf = np.asarray(frame["ego2global"], dtype=float)[:2, :2]
    yaw = float(np.arctan2((Ra.T @ Rf)[1, 0], (Ra.T @ Rf)[0, 0]))
    return [float(d[0]), float(d[1]), yaw]


def image_path(sensor_root, frame):
    rel = frame["cams"]["CAM_F0"]["data_path"]
    p = Path(sensor_root) / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--sensor-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--future", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    frames = pickle.load(open(args.pkl, "rb"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    needed = args.history + args.future
    for sample_id, start in enumerate(range(0, max(0, len(frames) - needed), args.stride)):
        if len(rows) >= args.num_samples:
            break
        window = frames[start : start + needed]
        anchor = window[args.history - 1]
        paths = [image_path(args.sensor_root, f) for f in window]
        # Validate the native frames and retain their original resolution.
        images = [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]
        if len({im.shape for im in images}) != 1:
            continue
        history = [rel_pose(f, anchor) for f in window[: args.history]]
        future = [rel_pose(f, anchor) for f in window[args.history :]]
        speeds = [float(np.asarray(f.get("ego_dynamic_state", [0]))[0]) for f in window[args.history :]]
        # Dataset-native driving command is one-hot (left, straight, right, u-turn).
        command = np.asarray(anchor.get("driving_command", [0, 1, 0, 0]), dtype=np.float32)
        if command.shape != (4,):
            command = np.asarray([0, 1, 0, 0], dtype=np.float32)
        sample = {
            "images": np.stack(images, axis=0),
            "history_poses": np.asarray(history, dtype=np.float32),
            "ego_status": {
                # DriveWAM's trained NavSim input is [vx, vy, ax, ay].
                "velocity": np.asarray([float(anchor["ego_dynamic_state"][0]), 0.0], dtype=np.float32),
                "acceleration": np.zeros(2, dtype=np.float32),
                "driving_command": command,
            },
            "future_trajectory": [
                {"pose": pose, "velocity": speeds[i], "acceleration": 0.0}
                for i, pose in enumerate(future)
            ],
            "metadata": {
                "scene_token": str(anchor.get("scene_token", "")),
                "source_pkl": str(args.pkl),
                "frame_idx": int(anchor.get("frame_idx", start + args.history - 1)),
                "image_paths": [str(p) for p in paths],
            },
        }
        path = out / f"sample_{sample_id:06d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
        rows.append({"sample": str(path), "frame_idx": sample["metadata"]["frame_idx"], "scene_token": sample["metadata"]["scene_token"]})
    (out / "manifest.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"output": str(out), "num_samples": len(rows), "needed_frames": needed}, indent=2))


if __name__ == "__main__":
    main()
