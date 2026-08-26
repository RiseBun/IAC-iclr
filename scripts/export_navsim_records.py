#!/usr/bin/env python3
"""Export NAVSIM native image/state windows for trajectory-image evaluation.

This exporter deliberately emits *native dataset future images*.  They are an
oracle/reference stream for validating timestamp and state joins, not WAM
generated futures.  WAM evaluation must replace ``future_images`` while
preserving the history, source key, and realized ego-state fields.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.state_protocol import navsim_states


def _as_pose(record: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(record["ego2global_translation"], dtype=np.float64)
    rotation = np.asarray(record["ego2global_rotation"], dtype=np.float64)
    if translation.shape != (3,) or rotation.shape != (4,):
        raise ValueError("NAVSIM ego2global translation/rotation have unexpected shapes")
    qw, qx, qy, qz = rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    matrix[:3, 3] = translation
    return matrix


def _image_path(record: dict[str, Any], sensor_root: Path, camera: str) -> Path:
    camera_record = record.get("cams", {}).get(camera)
    if not isinstance(camera_record, dict) or not camera_record.get("data_path"):
        raise ValueError(f"missing {camera} data_path")
    path = sensor_root / str(camera_record["data_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _lidar_path(record: dict[str, Any], sensor_root: Path) -> Path:
    relative = record.get("lidar_path")
    if not relative:
        raise ValueError("missing lidar_path")
    path = sensor_root / str(relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _camera_to_ego(record: dict[str, Any], camera_record: dict[str, Any]) -> np.ndarray:
    sensor_to_lidar = np.eye(4, dtype=np.float64)
    sensor_to_lidar[:3, :3] = np.asarray(
        camera_record["sensor2lidar_rotation"], dtype=np.float64
    )
    sensor_to_lidar[:3, 3] = np.asarray(
        camera_record["sensor2lidar_translation"], dtype=np.float64
    )
    lidar_to_ego = np.asarray(record.get("lidar2ego", np.eye(4)), dtype=np.float64)
    if lidar_to_ego.shape != (4, 4):
        raise ValueError("NAVSIM lidar2ego has unexpected shape")
    return lidar_to_ego @ sensor_to_lidar


def _scene_windows(
    frames: list[dict[str, Any]],
    *,
    sensor_root: Path,
    camera: str,
    history_count: int,
    future_count: int,
    stride: int,
    source_pkl: Path,
    require_lidar: bool,
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        grouped[str(frame.get("scene_name") or frame.get("scene_token"))].append(frame)
    records: list[dict[str, Any]] = []
    skipped = 0
    for scene_name, scene_frames in sorted(grouped.items()):
        scene_frames.sort(key=lambda row: (int(row.get("frame_idx", 0)), int(row["timestamp"])))
        required = history_count + future_count
        if len(scene_frames) < required:
            skipped += max(1, len(scene_frames))
            continue
        for anchor_pos in range(history_count - 1, len(scene_frames) - future_count, stride):
            window = scene_frames[anchor_pos - history_count + 1 : anchor_pos + future_count + 1]
            anchor = window[history_count - 1]
            try:
                image_paths = [_image_path(row, sensor_root, camera) for row in window]
                if require_lidar:
                    [_lidar_path(row, sensor_root) for row in window]
                states = navsim_states(
                    [{**row, "ego2global": _as_pose(row)} for row in window],
                    anchor_index=history_count - 1,
                )
            except (FileNotFoundError, KeyError, ValueError, TypeError):
                skipped += 1
                continue
            future_states = states[history_count:]
            anchor_timestamp = int(anchor["timestamp"])
            future_times = np.asarray(
                [(int(row["timestamp"]) - anchor_timestamp) * 1e-6 for row in window[history_count:]],
                dtype=np.float64,
            )
            if np.any(future_times <= 0.0) or np.any(np.diff(future_times) <= 0.0):
                skipped += 1
                continue
            camera_record = anchor["cams"][camera]
            camera_to_ego = _camera_to_ego(anchor, camera_record)
            records.append({
                "protocol": "wam-trajectory-image-record-v1",
                "record_type": "native_dataset_pair",
                "dataset": "navsim",
                "source_key": f"navsim:{anchor['log_name']}:{scene_name}:{anchor['token']}",
                "source_pkl": str(source_pkl),
                "scene_name": scene_name,
                "scene_token": str(anchor.get("scene_token", "")),
                "log_name": str(anchor["log_name"]),
                "frame_idx": int(anchor.get("frame_idx", 0)),
                "timestamp_us": anchor_timestamp,
                "history_images": [str(path) for path in image_paths[:history_count]],
                "future_images": [str(path) for path in image_paths[history_count:]],
                "future_images_source": "navsim_native_realized",
                "history_ego_state": states[:history_count].tolist(),
                "realized_future_ego_state": future_states.tolist(),
                "future_times_s": future_times.tolist(),
                "trajectory": future_states[:, :3].tolist(),
                "trajectory_source": "navsim_native_realized_oracle",
                "camera": camera,
                "camera_intrinsic": np.asarray(camera_record["cam_intrinsic"], dtype=np.float64).tolist(),
                "camera_distortion": np.asarray(
                    camera_record.get("distortion", []), dtype=np.float64
                ).tolist(),
                "camera_to_ego": camera_to_ego.tolist(),
                "sensor2lidar_rotation": np.asarray(camera_record["sensor2lidar_rotation"], dtype=np.float64).tolist(),
                "sensor2lidar_translation": np.asarray(camera_record["sensor2lidar_translation"], dtype=np.float64).tolist(),
                "lidar_required": bool(require_lidar),
            })
    return records, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="CAM_F0")
    parser.add_argument("--history-count", type=int, default=8)
    parser.add_argument("--future-count", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--require-lidar",
        action="store_true",
        help="Keep only windows whose every frame has a local lidar blob.",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=0,
        help="Optional diversity cap applied before the global max-records limit.",
    )
    args = parser.parse_args()
    if args.history_count < 1 or args.future_count < 1 or args.stride < 1:
        raise SystemExit("history-count, future-count, and stride must be positive")
    if args.max_records < 0 or args.max_records_per_file < 0:
        raise SystemExit("record limits must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_files = total_frames = total_records = skipped = 0
    with args.output.open("w", encoding="utf-8") as output:
        for source_pkl in sorted(args.pkl_root.glob("*.pkl")):
            with source_pkl.open("rb") as handle:
                payload = pickle.load(handle)
            frames = payload if isinstance(payload, list) else payload.get("frames", [])
            if not isinstance(frames, list):
                continue
            total_files += 1
            total_frames += len(frames)
            records, file_skipped = _scene_windows(
                frames,
                sensor_root=args.sensor_root,
                camera=args.camera,
                history_count=args.history_count,
                future_count=args.future_count,
                stride=args.stride,
                source_pkl=source_pkl,
                require_lidar=args.require_lidar,
            )
            skipped += file_skipped
            selected_records = (
                records[: args.max_records_per_file]
                if args.max_records_per_file
                else records
            )
            for record in selected_records:
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                total_records += 1
                if args.max_records and total_records >= args.max_records:
                    break
            if args.max_records and total_records >= args.max_records:
                break
    summary = {
        "dataset": "navsim",
        "camera": args.camera,
        "history_count": args.history_count,
        "future_count": args.future_count,
        "stride": args.stride,
        "pkl_files": total_files,
        "source_frames": total_frames,
        "records": total_records,
        "skipped_windows": skipped,
        "future_images_source": "navsim_native_realized",
        "require_lidar": bool(args.require_lidar),
        "status": "ok" if total_records else "empty",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
