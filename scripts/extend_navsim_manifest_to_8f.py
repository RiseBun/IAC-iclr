#!/usr/bin/env python3
"""Extend selected NAVSIM anchors to native 4-history/8-future windows."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.counterfactual import densify_record
from iac_new.state_protocol import navsim_states


def _as_pose(record: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(record["ego2global_translation"], dtype=np.float64)
    rotation = np.asarray(record["ego2global_rotation"], dtype=np.float64)
    qw, qx, qy, qz = rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    matrix[:3, 3] = translation
    return matrix


def _camera_to_ego(record: dict[str, Any], camera_record: dict[str, Any]) -> np.ndarray:
    sensor_to_lidar = np.eye(4, dtype=np.float64)
    sensor_to_lidar[:3, :3] = np.asarray(camera_record["sensor2lidar_rotation"], dtype=np.float64)
    sensor_to_lidar[:3, 3] = np.asarray(camera_record["sensor2lidar_translation"], dtype=np.float64)
    return np.asarray(record.get("lidar2ego", np.eye(4)), dtype=np.float64) @ sensor_to_lidar


def _source_log(sample_id: str) -> str:
    parts = str(sample_id).split(":")
    if len(parts) < 2:
        raise ValueError(f"invalid NAVSIM sample id: {sample_id}")
    return parts[1]


def _build_index(pkl_root: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for source in sorted(pkl_root.glob("*.pkl")):
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        frames = payload if isinstance(payload, list) else payload.get("frames", [])
        if isinstance(frames, list):
            index[source.stem] = sorted(frames, key=lambda row: int(row.get("timestamp", 0)))
    return index


def _convert(
    row: dict[str, Any],
    *,
    frames: list[dict[str, Any]],
    sensor_root: Path,
    camera: str,
) -> dict[str, Any]:
    anchor_timestamp = int(row["metadata"]["timestamp_us"])
    anchor_indices = [i for i, frame in enumerate(frames) if int(frame.get("timestamp", -1)) == anchor_timestamp]
    if len(anchor_indices) != 1:
        raise ValueError(f"anchor timestamp has {len(anchor_indices)} matches: {anchor_timestamp}")
    anchor_index = anchor_indices[0]
    start = anchor_index - 3
    stop = anchor_index + 9
    if start < 0 or stop > len(frames):
        raise ValueError("native sequence does not contain 4 history + 8 future frames")
    window = frames[start:stop]
    camera_records = [frame.get("cams", {}).get(camera) for frame in window]
    if any(not isinstance(value, dict) or not value.get("data_path") for value in camera_records):
        raise ValueError("camera path missing in native window")
    image_paths = [sensor_root / str(value["data_path"]) for value in camera_records]
    if not all(path.is_file() for path in image_paths):
        raise ValueError("native camera image missing in 4-second window")
    states = navsim_states(
        [{**frame, "ego2global": _as_pose(frame)} for frame in window],
        anchor_index=3,
    )
    timestamps = np.asarray([int(frame["timestamp"]) for frame in window], dtype=np.int64)
    future_times = (timestamps[4:] - timestamps[3]) * 1e-6
    if len(future_times) != 8 or np.any(future_times <= 0.0) or np.any(np.diff(future_times) <= 0.0):
        raise ValueError("native future timestamps are not strictly increasing")
    anchor = window[3]
    camera_record = camera_records[3]
    selected_state = states[4:, :3].tolist()
    result = {
        "sample_id": str(row["sample_id"]),
        "scene_id": str(row.get("scene_id") or anchor.get("scene_name")),
        "history_frame_paths": [str(path) for path in image_paths[:4]],
        "history_times_s": [float(value) for value in ((timestamps[:4] - timestamps[3]) * 1e-6)],
        "future_frame_paths": [str(path) for path in image_paths[4:]],
        "future_times_s": [float(value) for value in future_times],
        "intrinsics": np.asarray(camera_record["cam_intrinsic"], dtype=np.float64).tolist(),
        "distortion": np.asarray(camera_record.get("distortion", []), dtype=np.float64).tolist(),
        "camera_to_ego": _camera_to_ego(anchor, camera_record).tolist(),
        "gt_candidate_id": "logged",
        "candidates": [{"candidate_id": "logged", "prior": 1.0, "trajectory": selected_state}],
        "metadata": {
            **dict(row.get("metadata") or {}),
            "protocol": "navsim-state-aware-history4-future8-v1",
            "history_ego_state": states[:4].tolist(),
            "realized_future_ego_state": selected_state,
            "future_times_s": [float(value) for value in future_times],
            "state_source": "navsim_native_realized",
            "native_window_frame_indices": [int(frame.get("frame_idx", -1)) for frame in window],
        },
    }
    return densify_record(
        result,
        reference_candidate_id="logged",
        speed_factors=[0.85, 1.0, 1.15],
        lateral_offsets_m=[-0.5, 0.0, 0.5],
        curvature_offsets_1pm=[-0.015, 0.0, 0.015],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pkl-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--camera", default="CAM_F0")
    args = parser.parse_args()
    index = _build_index(args.pkl_root)
    converted = []
    failures = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        source = _source_log(row["sample_id"])
        try:
            if source not in index:
                raise ValueError(f"source pickle not found: {source}")
            converted.append(_convert(row, frames=index[source], sensor_root=args.sensor_root, camera=args.camera))
        except (KeyError, TypeError, ValueError, FileNotFoundError, IndexError) as exc:
            failures.append({"line": line_number, "sample_id": row.get("sample_id"), "reason": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in converted), encoding="utf-8")
    args.audit_output.write_text(json.dumps({
        "input_records": len(converted) + len(failures),
        "converted_records": len(converted),
        "failed_records": len(failures),
        "history_count": 4,
        "future_count": 8,
        "future_horizon_s": 4.0,
        "failures": failures,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"converted_records": len(converted), "failed_records": len(failures), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
