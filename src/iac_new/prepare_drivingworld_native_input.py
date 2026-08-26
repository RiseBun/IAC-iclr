#!/usr/bin/env python3
"""Prepare native branch folders for the 15-frame DrivingWorld demo loader.

DrivingWorld's demo loader is directory based rather than JSONL based. It
expects numbered PNGs plus ``pose.npy`` and ``yaw.npy``. This adapter keeps the
native branch id in a sidecar manifest, writes 15 native history images, and
appends one placeholder image per requested future knot so the existing
autoregressive smoke script can read the future action targets from the pose
arrays. The placeholder images are never used as WAM context: the model takes
only the first ``condition_frames`` images.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _branch_id(row: dict[str, Any]) -> str:
    value = row.get("branch_id")
    if value is None or not str(value):
        raise ValueError("every branch needs branch_id")
    return str(value)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)


def prepare_rows(rows: list[dict[str, Any]], output_dir: Path, *, placeholder: str = "copy_last") -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    for row in rows:
        branch_id = _branch_id(row)
        history = [str(path) for path in row.get("history_images") or []]
        action = np.asarray(row.get("action_trajectory"), dtype=np.float32)
        history_state = np.asarray(row.get("history_ego_state"), dtype=np.float32)
        if len(history) != 15:
            raise ValueError(f"{branch_id}: DrivingWorld adapter requires exactly 15 history images")
        if action.ndim != 2 or action.shape[1] != 3 or len(action) < 1:
            raise ValueError(f"{branch_id}: action_trajectory must have shape [T,3]")
        if history_state.shape != (15, 5):
            raise ValueError(f"{branch_id}: history_ego_state must have shape [15,5]")
        future_count = len(action)
        branch_dir = output_dir / _safe_name(branch_id)
        branch_dir.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(history):
            destination = branch_dir / f"{index:03d}.png"
            shutil.copyfile(source, destination)
        if placeholder == "copy_last":
            for index in range(15, 15 + future_count):
                shutil.copyfile(history[-1], branch_dir / f"{index:03d}.png")
        else:
            raise ValueError("only --placeholder copy_last is supported")
        pose = np.concatenate([history_state[:, :2], action[:, :2]], axis=0)
        yaw = np.concatenate([history_state[:, 2:3], action[:, 2:3]], axis=0)
        np.save(branch_dir / "pose.npy", pose[None, ...].astype(np.float32))
        np.save(branch_dir / "yaw.npy", yaw[None, ...].astype(np.float32))
        result.append({
            "branch_id": branch_id,
            "counterfactual_group_id": row.get("counterfactual_group_id"),
            "source_key": row.get("source_key"),
            "wam_model_id": row.get("wam_model_id", "drivingworld"),
            "input_dir": str(branch_dir.resolve()),
            "history_frames": 15,
            "future_action_targets": future_count,
            "placeholder_future_images": True,
            "placeholder_policy": placeholder,
            "generated_future_images_expected": [str((branch_dir / f"{index:03d}.png").resolve()) for index in range(15, 15 + future_count)],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--placeholder", default="copy_last")
    args = parser.parse_args()
    rows = prepare_rows(read_jsonl(args.branches), args.output_dir, placeholder=args.placeholder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"protocol": "drivingworld-native-15h-v1", "branches": len(rows), "output": str(args.output.resolve()), "input_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
