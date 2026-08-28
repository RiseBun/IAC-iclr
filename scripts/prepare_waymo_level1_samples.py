#!/usr/bin/env python3
"""Materialize continuous Waymo camera/pose data into the IAC Level-1 protocol.

The Waymo modular Perception v2 export is sampled at 10 Hz.  This converter
keeps one front-camera JPEG per source timestamp and creates non-overlapping
or sliding 4 s samples with four 0.5 s history frames and eight 0.5 s future
frames.  No TensorFlow/Waymo runtime is required; only pyarrow is needed.

Input layout (one or more segment parquet files per component)::

  <input>/camera_image/<segment>.parquet
  <input>/camera_calibration/<segment>.parquet
  <input>/vehicle_pose/<segment>.parquet

Output layout::

  <output>/frames/<segment>/front/<timestamp>.jpg
  <output>/camera/<segment>.json
  <output>/manifest.jsonl

The manifest stores canonical relative ego states
``[x_m, y_m, yaw_rad, speed_mps, yaw_rate_rps]`` in the anchor vehicle frame.
It is deliberately independent of E2E labels; E2E scenario tags can be joined
later by an explicit segment/timestamp mapping when such a mapping exists.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


IMG_TS = "key.frame_timestamp_micros"
CAM = "key.camera_name"
IMG = "[CameraImageComponent].image"
VX = "[CameraImageComponent].velocity.linear_velocity.x"
VY = "[CameraImageComponent].velocity.linear_velocity.y"
WZ = "[CameraImageComponent].velocity.angular_velocity.z"
POSE = "[VehiclePoseComponent].world_from_vehicle.transform"


def _yaw(transform: Any) -> float:
    m = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return float(math.atan2(m[1, 0], m[0, 0]))


def _relative_states(rows: list[dict[str, Any]], anchor: int) -> list[list[float]]:
    a = rows[anchor]
    ax, ay, ayaw = float(a["x"]), float(a["y"]), float(a["yaw"])
    c, s = math.cos(ayaw), math.sin(ayaw)
    out: list[list[float]] = []
    for row in rows:
        dx, dy = float(row["x"]) - ax, float(row["y"]) - ay
        # world -> anchor vehicle frame: +x forward, +y left
        x = c * dx + s * dy
        y = -s * dx + c * dy
        yaw = math.atan2(math.sin(float(row["yaw"]) - ayaw), math.cos(float(row["yaw"]) - ayaw))
        out.append([x, y, yaw, float(row["speed"]), float(row["yaw_rate"])])
    return out


def _load_segment(image_path: Path, pose_path: Path, camera: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    image_cols = [IMG_TS, CAM, IMG, VX, VY, WZ]
    image = pq.read_table(image_path, columns=image_cols).to_pydict()
    timestamps = image[IMG_TS]
    front: dict[int, dict[str, Any]] = {}
    for i, cam in enumerate(image[CAM]):
        if int(cam) != camera:
            continue
        ts = int(timestamps[i])
        vx = float(image[VX][i] or 0.0)
        vy = float(image[VY][i] or 0.0)
        front[ts] = {
            "timestamp_micros": ts,
            "image": bytes(image[IMG][i]),
            "speed": math.hypot(vx, vy),
            "yaw_rate": float(image[WZ][i] or 0.0),
        }
    if not front:
        raise ValueError(f"no camera {camera} rows in {image_path}")

    pose = pq.read_table(pose_path, columns=[IMG_TS, POSE]).to_pydict()
    pose_by_ts = {
        int(ts): (float(np.asarray(tf).reshape(4, 4)[0, 3]), float(np.asarray(tf).reshape(4, 4)[1, 3]), _yaw(tf))
        for ts, tf in zip(pose[IMG_TS], pose[POSE])
    }
    rows = []
    for ts in sorted(front):
        if ts not in pose_by_ts:
            continue
        x, y, yaw = pose_by_ts[ts]
        rows.append({**front[ts], "x": x, "y": y, "yaw": yaw})
    if len(rows) < 50:
        raise ValueError(f"{image_path.name}: only {len(rows)} usable front frames")

    cal = {}
    cal_path = image_path.parent.parent / "camera_calibration" / image_path.name
    if cal_path.exists():
        ctab = pq.read_table(cal_path).to_pydict()
        for i, cam in enumerate(ctab[CAM]):
            if int(cam) != camera:
                continue
            cal = {
                "camera_name": int(cam),
                "intrinsics": {
                    "f_u": float(ctab["[CameraCalibrationComponent].intrinsic.f_u"][i]),
                    "f_v": float(ctab["[CameraCalibrationComponent].intrinsic.f_v"][i]),
                    "c_u": float(ctab["[CameraCalibrationComponent].intrinsic.c_u"][i]),
                    "c_v": float(ctab["[CameraCalibrationComponent].intrinsic.c_v"][i]),
                    "k1": float(ctab["[CameraCalibrationComponent].intrinsic.k1"][i]),
                    "k2": float(ctab["[CameraCalibrationComponent].intrinsic.k2"][i]),
                    "p1": float(ctab["[CameraCalibrationComponent].intrinsic.p1"][i]),
                    "p2": float(ctab["[CameraCalibrationComponent].intrinsic.p2"][i]),
                    "k3": float(ctab["[CameraCalibrationComponent].intrinsic.k3"][i]),
                },
                "extrinsic_transform": [float(v) for v in ctab["[CameraCalibrationComponent].extrinsic.transform"][i]],
                "width": int(ctab["[CameraCalibrationComponent].width"][i]),
                "height": int(ctab["[CameraCalibrationComponent].height"][i]),
            }
            break
    return image_path.stem, rows, cal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Perception root containing component directories")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--camera", type=int, default=1, help="Waymo FRONT camera (default: 1)")
    ap.add_argument("--stride-10hz", type=int, default=5, help="Anchor stride in source 10 Hz frames")
    ap.add_argument("--sample-stride", type=int, default=1, help="Keep every N generated anchors")
    args = ap.parse_args()
    image_dir = args.input / "camera_image"
    pose_dir = args.input / "vehicle_pose"
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"
    total = 0
    with manifest_path.open("w", encoding="utf-8") as out:
        for image_path in sorted(image_dir.glob("*.parquet")):
            pose_path = pose_dir / image_path.name
            if not pose_path.exists():
                print(f"skip {image_path.name}: missing vehicle_pose parquet")
                continue
            segment, rows, calibration = _load_segment(image_path, pose_path, args.camera)
            frame_root = args.output / "frames" / segment / "front"
            frame_root.mkdir(parents=True, exist_ok=True)
            image_paths = {}
            for row in rows:
                ts = row["timestamp_micros"]
                path = frame_root / f"{ts}.jpg"
                if not path.exists():
                    path.write_bytes(row["image"])
                image_paths[ts] = str(path.relative_to(args.output).as_posix())
            if calibration:
                cam_path = args.output / "camera" / f"{segment}.json"
                cam_path.parent.mkdir(parents=True, exist_ok=True)
                cam_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

            # 10 Hz source -> 0.5 s protocol points. Need 1.5 s history and 4 s future.
            hist_offsets = [-3 * args.stride_10hz, -2 * args.stride_10hz, -args.stride_10hz, 0]
            fut_offsets = [i * args.stride_10hz for i in range(1, 9)]
            lo, hi = 3 * args.stride_10hz, len(rows) - 40
            states = _relative_states(rows, 0)  # recomputed per anchor below
            anchors = list(range(lo, hi + 1, args.stride_10hz * args.sample_stride))
            for anchor in anchors:
                indices = [anchor + o for o in hist_offsets + fut_offsets]
                if min(indices) < 0 or max(indices) >= len(rows):
                    continue
                states = _relative_states(rows, anchor)
                hidx, fidx = indices[:4], indices[4:]
                row = {
                    "dataset": "waymo_perception_v2",
                    "segment_id": segment,
                    "anchor_timestamp_micros": rows[anchor]["timestamp_micros"],
                    "camera": "FRONT",
                    "camera_calibration": str((Path("camera") / f"{segment}.json").as_posix()) if calibration else None,
                    "history_times_s": [-1.5, -1.0, -0.5, 0.0],
                    "future_times_s": [0.5 * i for i in range(1, 9)],
                    "history_images": [image_paths[rows[i]["timestamp_micros"]] for i in hidx],
                    "future_images": [image_paths[rows[i]["timestamp_micros"]] for i in fidx],
                    "history_ego_state": [states[i] for i in hidx],
                    "realized_future_ego_state": [states[i] for i in fidx],
                    "state_reference_source": "waymo_vehicle_pose_plus_camera_velocity",
                    "continuous_source_fps": 10.0,
                    "protocol": "iac-level1-continuous-v1",
                }
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                total += 1
    print(json.dumps({"manifest": str(manifest_path), "samples": total}, indent=2))


if __name__ == "__main__":
    main()
