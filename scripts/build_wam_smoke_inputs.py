#!/usr/bin/env python3
"""Build small, lineage-preserving inputs for WAM runtime smoke tests.

This is deliberately not a benchmark exporter: it only makes a deterministic
sample from an existing front-camera sequence so model loading and generation
can be tested before native NavSim/NuPlan sensor blobs are available.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def load_frames(source: Path, count: int, size: tuple[int, int]) -> np.ndarray:
    paths = sorted(source.glob("*.png"))[:count]
    if len(paths) < count:
        raise FileNotFoundError(f"need {count} png frames under {source}, found {len(paths)}")
    width, height = size
    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames, axis=0)


def build_drivewam_sample(source: Path, output: Path) -> None:
    images = load_frames(source, count=15, size=(448, 256))
    history_poses = np.zeros((4, 3), dtype=np.float32)
    history_poses[:, 0] = np.arange(4, dtype=np.float32) * 0.3
    future = []
    for index in range(8):
        future.append({
            "pose": np.asarray([(index + 1) * 0.3, 0.0, 0.0], dtype=np.float32),
            "velocity": np.asarray([3.0, 0.0, 0.0], dtype=np.float32),
            "acceleration": np.zeros(3, dtype=np.float32),
        })
    sample = {
        "images": images,
        "history_poses": history_poses,
        "ego_status": {
            "pose": np.zeros(3, dtype=np.float32),
            "velocity": np.asarray([3.0, 0.0, 0.0], dtype=np.float32),
            "acceleration": np.zeros(3, dtype=np.float32),
            "driving_command": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        },
        "future_trajectory": future,
        "metadata": {"scene_token": "iac_smoke", "source": str(source)},
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sample_000000.pkl").open("wb") as handle:
        pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_epona_demo(source: Path, output: Path) -> None:
    sequence = output / "iac_smoke"
    sequence.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.glob("*.png"))[:15]
    if len(paths) < 15:
        raise FileNotFoundError(f"need 15 png frames under {source}, found {len(paths)}")
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB").resize((1024, 512), Image.Resampling.BILINEAR)
        image.save(sequence / f"{index:03d}.png")
    pose = np.zeros((1, 15, 2), dtype=np.float32)
    pose[0, :, 0] = np.arange(15, dtype=np.float32) * 0.3
    yaw = np.zeros((1, 15, 1), dtype=np.float32)
    np.save(sequence / "pose.npy", pose)
    np.save(sequence / "yaw.npy", yaw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--drivewam-output", type=Path)
    parser.add_argument("--epona-output", type=Path)
    args = parser.parse_args()
    if args.drivewam_output:
        build_drivewam_sample(args.source, args.drivewam_output)
    if args.epona_output:
        build_epona_demo(args.source, args.epona_output)
    if not args.drivewam_output and not args.epona_output:
        parser.error("at least one output is required")


if __name__ == "__main__":
    main()
