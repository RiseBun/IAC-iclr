#!/usr/bin/env python3
"""Materialize Waymo Perception v2 camera/state windows for IAC Level-1.

The exporter samples native 10 Hz FRONT-camera sequences every five frames:
4 history points at -1.5,-1.0,-0.5,0 and 8 future points at 0.5,...,4.0 s.
Windows are non-overlapping within a segment by default, so a later selector
can safely split them by segment without leakage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


FRONT = 1
STEP = 5
HISTORY = 4
FUTURE = 8
WINDOW = (HISTORY + FUTURE - 1) * STEP + 1


def _col(table: Any, name: str) -> list[Any]:
    return table[name].to_pylist()


def _yaw(matrix: np.ndarray) -> float:
    return float(math.atan2(matrix[1, 0], matrix[0, 0]))


def _wrap(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def _calibration(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    names = _col(table, "key.camera_name")
    idx = next((i for i, n in enumerate(names) if int(n) == FRONT), 0)
    p = "[CameraCalibrationComponent]"
    transform = np.asarray(_col(table, f"{p}.extrinsic.transform")[idx], dtype=np.float64).reshape(4, 4)
    intrinsics = [[float(_col(table, f"{p}.intrinsic.f_u")[idx]), 0.0, float(_col(table, f"{p}.intrinsic.c_u")[idx])],
                  [0.0, float(_col(table, f"{p}.intrinsic.f_v")[idx]), float(_col(table, f"{p}.intrinsic.c_v")[idx])],
                  [0.0, 0.0, 1.0]]
    distortion = [float(_col(table, f"{p}.intrinsic.{k}")[idx]) for k in ("k1", "k2", "p1", "p2", "k3")]
    return {"intrinsics": intrinsics, "distortion": distortion, "camera_to_ego": transform.tolist()}


def _read_segment(image_path: Path, pose_path: Path) -> list[dict[str, Any]]:
    image_table = pq.read_table(image_path)
    pose_table = pq.read_table(pose_path)
    image_ts = _col(image_table, "key.frame_timestamp_micros")
    image_camera = _col(image_table, "key.camera_name")
    image_bytes = _col(image_table, "[CameraImageComponent].image")
    image_vel = [
        _col(image_table, "[CameraImageComponent].velocity.linear_velocity.x"),
        _col(image_table, "[CameraImageComponent].velocity.linear_velocity.y"),
        _col(image_table, "[CameraImageComponent].velocity.linear_velocity.z"),
    ]
    pose_ts = _col(pose_table, "key.frame_timestamp_micros")
    pose_values = _col(pose_table, "[VehiclePoseComponent].world_from_vehicle.transform")
    pose_by_ts = {int(t): np.asarray(v, dtype=np.float64).reshape(4, 4) for t, v in zip(pose_ts, pose_values)}
    rows = []
    for i, (ts, camera, blob) in enumerate(zip(image_ts, image_camera, image_bytes)):
        if int(camera) != FRONT or int(ts) not in pose_by_ts:
            continue
        rows.append({"timestamp": int(ts), "image": bytes(blob), "pose": pose_by_ts[int(ts)], "velocity": [float(v[i]) for v in image_vel]})
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _state(anchor: np.ndarray, pose: np.ndarray, velocity: list[float], prev: dict[str, Any] | None, dt: float) -> list[float]:
    relative = np.linalg.inv(anchor) @ pose
    speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)[:2]))
    if prev is None:
        yaw_rate = 0.0
    else:
        yaw_rate = _wrap(_yaw(pose) - _yaw(prev["pose"])) / max(dt, 1e-3)
    return [float(relative[0, 3]), float(relative[1, 3]), _wrap(_yaw(relative)), speed, float(yaw_rate)]


def export_segment(image_path: Path, calibration_path: Path, pose_path: Path, output_root: Path, stride: int) -> list[dict[str, Any]]:
    segment = image_path.stem
    rows = _read_segment(image_path, pose_path)
    if len(rows) < WINDOW:
        return []
    calibration = _calibration(calibration_path)
    segment_root = output_root / "frames" / segment / "front"
    segment_root.mkdir(parents=True, exist_ok=True)
    records = []
    for anchor_idx in range((HISTORY - 1) * STEP, len(rows) - (FUTURE * STEP), stride):
        indices = [anchor_idx + (j - (HISTORY - 1)) * STEP for j in range(HISTORY + FUTURE)]
        if indices[-1] >= len(rows):
            break
        selected = [rows[i] for i in indices]
        anchor = selected[HISTORY - 1]["pose"]
        paths = []
        for item in selected:
            path = segment_root / f"{item['timestamp']}.jpg"
            if not path.exists():
                path.write_bytes(item["image"])
            paths.append(str(path.resolve()))
        states = [_state(anchor, item["pose"], item["velocity"], selected[max(0, j - 1)] if j else None, 0.5) for j, item in enumerate(selected)]
        future_times = [0.5 * (i + 1) for i in range(FUTURE)]
        records.append({
            "protocol": "iac-level1-continuous-v2",
            "record_type": "native_dataset_pair",
            "dataset": "waymo_perception_v2",
            "sample_id": f"{segment}:{selected[HISTORY - 1]['timestamp']}",
            "source_key": f"waymo_perception_v2:{segment}:{selected[HISTORY - 1]['timestamp']}",
            "scene_id": segment,
            "scene_name": segment,
            "segment_id": segment,
            "anchor_timestamp_micros": selected[HISTORY - 1]["timestamp"],
            "camera": "FRONT",
            **calibration,
            "history_times_s": [-1.5, -1.0, -0.5, 0.0],
            "future_times_s": future_times,
            "history_images": paths[:HISTORY],
            "future_images": paths[HISTORY:],
            "history_ego_state": states[:HISTORY],
            "realized_future_ego_state": states[HISTORY:],
            "trajectory": [s[:3] for s in states[HISTORY:]],
            "future_images_source": "native_dataset_realized",
            "state_reference_source": "waymo_vehicle_pose_plus_camera_velocity",
            "continuous_source_fps": 10.0,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw/perception_v2/validation")
    parser.add_argument("--output", type=Path, required=True, help="frames/level1_v2 output root")
    parser.add_argument("--stride-frames", type=int, default=WINDOW)
    args = parser.parse_args()
    image_root = args.input / "camera_image"
    cal_root = args.input / "camera_calibration"
    pose_root = args.input / "vehicle_pose"
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "manifest.jsonl"
    total = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for image_path in sorted(image_root.glob("*.parquet")):
            calibration_path = cal_root / image_path.name
            pose_path = pose_root / image_path.name
            if not calibration_path.is_file() or not pose_path.is_file():
                continue
            try:
                records = export_segment(image_path, calibration_path, pose_path, args.output, args.stride_frames)
            except Exception as exc:
                print(json.dumps({"segment": image_path.stem, "status": "failed", "error": repr(exc)}))
                continue
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            total += len(records)
            print(json.dumps({"segment": image_path.stem, "records": len(records)}))
    print(json.dumps({"records": total, "output": str(out_path.resolve()), "stride_frames": args.stride_frames}, indent=2))


if __name__ == "__main__":
    main()
