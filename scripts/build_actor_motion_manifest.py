#!/usr/bin/env python3
"""Build an independent 8-frame actor-motion manifest from NuPlan SQLite logs.

The output contains camera frames/calibration and actor centers from the
dataset lidar boxes. It never includes WAM action conditions or generated
future images, so it can serve as an independent metric-speed reference.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import math
import pickle
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TAG_SCOPE: dict[str, tuple[str, str]] = {
    "following_lane_with_slow_lead": ("cut_in_or_lead_brake", "lead_vehicle"),
    "following_lane_with_lead": ("cut_in_or_lead_brake", "lead_vehicle"),
    "stopping_with_lead": ("cut_in_or_lead_brake", "lead_vehicle_brake"),
    "waiting_for_pedestrian_to_cross": ("pedestrian_crossing", "pedestrian"),
    "near_pedestrian_on_crosswalk_with_ego": ("pedestrian_crossing", "pedestrian"),
    "stopping_at_crosswalk": ("pedestrian_crossing", "pedestrian"),
    "stationary_at_crosswalk": ("pedestrian_crossing", "pedestrian"),
    "accelerating_at_crosswalk": ("pedestrian_crossing", "pedestrian"),
    "near_trafficcone_on_driveable": ("blocked_lane", "traffic_cone"),
    "near_barrier_on_driveable": ("blocked_lane", "barrier"),
    "near_construction_zone_sign": ("blocked_lane", "construction"),
    "traversing_narrow_lane": ("blocked_lane", "narrow_lane"),
    "starting_unprotected_cross_turn": ("unprotected_turn_or_merge", "unprotected_turn"),
    "starting_unprotected_noncross_turn": ("unprotected_turn_or_merge", "unprotected_turn"),
}
CHAIN_ORDER = tuple(sorted({value[0] for value in TAG_SCOPE.values()}))


class _PickleVector:
    """Minimal unpickle target for NuPlan Translation/Rotation blobs."""

    def __new__(cls):
        value = object.__new__(cls)
        value.values = []
        return value

    def append(self, value: Any) -> None:
        self.values.append(value)

    def __setstate__(self, state: Any) -> None:
        self.values = state


class _NuPlanUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "nuplan.database.common.data_types" and name in {
            "Translation", "Rotation", "CameraIntrinsic"
        }:
            return _PickleVector
        return super().find_class(module, name)


def _decode(blob: bytes | None) -> Any:
    if blob is None:
        return None
    return _NuPlanUnpickler(io.BytesIO(blob)).load()


def _token(value: Any) -> str:
    return value.hex() if isinstance(value, bytes) else str(value)


def _quat_matrix(quaternion: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(quaternion), dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("rotation blob must contain [qw,qx,qy,qz]")
    qw, qx, qy, qz = values
    return np.asarray([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _ego_pose(row: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, qw, qx, qy, qz = map(float, row)
    rotation = _quat_matrix([qw, qx, qy, qz])
    return np.asarray([x, y, z], dtype=np.float64), rotation


def _global_to_ego(point: np.ndarray, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return rotation.T @ (np.asarray(point, dtype=np.float64) - translation)


def _stable_key(*values: object) -> str:
    return hashlib.sha256("\0".join(str(value) for value in values).encode()).hexdigest()


def _nearest(rows: list[tuple[Any, ...]], target: int, timestamp_index: int = 1) -> tuple[Any, ...]:
    timestamps = [int(row[timestamp_index]) for row in rows]
    insertion = bisect.bisect_left(timestamps, target)
    indices = [index for index in (insertion - 1, insertion) if 0 <= index < len(rows)]
    return min((rows[index] for index in indices), key=lambda row: abs(int(row[timestamp_index]) - target))


def _camera_calibration(connection: sqlite3.Connection, channel: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT translation, rotation, intrinsic, distortion, width, height "
        "FROM camera WHERE channel=? LIMIT 1", (channel,)
    ).fetchone()
    if row is None:
        raise ValueError(f"camera channel not found: {channel}")
    translation = np.asarray(_decode(row[0]).values, dtype=np.float64)
    rotation = _quat_matrix(_decode(row[1]).values)
    intrinsic = np.asarray(_decode(row[2]).values, dtype=np.float64)
    distortion = np.asarray(_decode(row[3]), dtype=np.float64)
    camera_to_ego = np.eye(4, dtype=np.float64)
    camera_to_ego[:3, :3] = rotation
    camera_to_ego[:3, 3] = translation
    return {
        "intrinsics": intrinsic.tolist(),
        "distortion": distortion.tolist(),
        "camera_to_ego": camera_to_ego.tolist(),
        "image_size": [int(row[4]), int(row[5])],
    }


def _scene_candidates(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT st.lidar_pc_token, st.agent_track_token, st.type, lp.timestamp, "
        "lp.scene_token, lp.ego_pose_token, lg.logfile "
        "FROM scenario_tag st JOIN lidar_pc lp ON lp.token=st.lidar_pc_token "
        "JOIN lidar ld ON ld.token=lp.lidar_token JOIN log lg ON lg.token=ld.log_token "
        "WHERE st.type IN (%s) ORDER BY lp.timestamp" % ",".join("?" * len(TAG_SCOPE)),
        tuple(TAG_SCOPE),
    ).fetchall()
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for lidar_token, agent_token, tag, timestamp, scene_token, ego_token, logfile in rows:
        key = (_token(lidar_token), int(timestamp))
        item = grouped.setdefault(key, {
            "lidar_pc_token": lidar_token,
            "agent_track_token": agent_token,
            "timestamp": int(timestamp),
            "scene_token": _token(scene_token),
            "ego_pose_token": ego_token,
            "logfile": str(logfile),
            "tags": [],
        })
        item["tags"].append(str(tag))
        if item.get("agent_track_token") is None and agent_token is not None:
            item["agent_track_token"] = agent_token
    output = []
    for item in grouped.values():
        chains = defaultdict(list)
        scopes = defaultdict(set)
        for tag in item["tags"]:
            chain, scope = TAG_SCOPE[tag]
            chains[chain].append(tag)
            scopes[chain].add(scope)
        for chain in sorted(chains):
            output.append({**item, "chain_type": chain, "candidate_scope": sorted(scopes[chain])})
    return output


def _build_one(
    connection: sqlite3.Connection,
    candidate: dict[str, Any],
    sensor_root: Path,
    *,
    camera_channel: str,
    history_offsets_s: tuple[float, ...],
    future_offsets_s: tuple[float, ...],
    tolerance_us: int,
) -> dict[str, Any] | None:
    anchor_timestamp = int(candidate["timestamp"])
    offsets = history_offsets_s + future_offsets_s
    image_rows = connection.execute(
        "SELECT im.timestamp, im.filename_jpg FROM image im JOIN camera cam ON cam.token=im.camera_token "
        "WHERE cam.channel=? AND im.timestamp BETWEEN ? AND ? ORDER BY im.timestamp",
        (camera_channel, anchor_timestamp + round(min(offsets) * 1e6) - tolerance_us,
         anchor_timestamp + round(max(offsets) * 1e6) + tolerance_us),
    ).fetchall()
    lidar_rows = connection.execute(
        "SELECT token, timestamp, ego_pose_token, scene_token FROM lidar_pc "
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (anchor_timestamp + round(min(offsets) * 1e6) - tolerance_us,
         anchor_timestamp + round(max(offsets) * 1e6) + tolerance_us),
    ).fetchall()
    if not image_rows or not lidar_rows:
        return None
    selected_images = []
    selected_lidar = []
    for offset in offsets:
        target = anchor_timestamp + round(offset * 1e6)
        image = _nearest(image_rows, target, timestamp_index=0)
        lidar = _nearest(lidar_rows, target)
        if abs(int(image[0]) - target) > tolerance_us or abs(int(lidar[1]) - target) > tolerance_us:
            return None
        relative = Path(str(image[1]))
        options = [sensor_root / relative, sensor_root / "mini" / relative,
                   sensor_root / "trainval" / relative, sensor_root / "test" / relative]
        path = next((item for item in options if item.is_file()), options[0])
        if not path.is_file():
            return None
        selected_images.append({"offset_s": float(offset), "timestamp_us": int(image[0]), "path": str(path.resolve())})
        selected_lidar.append(lidar)

    calibration = _camera_calibration(connection, camera_channel)
    track_token = candidate.get("agent_track_token")
    if track_token is None:
        # Fall back to the nearest front actor with a stable track token.
        anchor_lidar = selected_lidar[len(history_offsets_s)]
        boxes = connection.execute(
            "SELECT track_token, x, y, z, confidence FROM lidar_box WHERE lidar_pc_token=?",
            (anchor_lidar[0],),
        ).fetchall()
        anchor_pose_row = connection.execute(
            "SELECT x,y,z,qw,qx,qy,qz FROM ego_pose WHERE token=?", (anchor_lidar[2],)
        ).fetchone()
        if anchor_pose_row is None:
            return None
        anchor_translation, anchor_rotation = _ego_pose(anchor_pose_row)
        front = []
        for box in boxes:
            point = _global_to_ego(np.asarray(box[1:4]), anchor_translation, anchor_rotation)
            if point[0] > 2.0 and np.linalg.norm(point[:2]) < 60.0 and float(box[4]) >= 0.3:
                front.append((float(np.linalg.norm(point[:2])), box[0]))
        if not front:
            return None
        track_token = min(front)[1]

    actor_positions = []
    visible = []
    actor_confidence = []
    actor_class = None
    for lidar in selected_lidar[len(history_offsets_s):]:
        box = connection.execute(
            "SELECT lb.x,lb.y,lb.z,lb.confidence,c.name FROM lidar_box lb "
            "JOIN track tr ON tr.token=lb.track_token JOIN category c ON c.token=tr.category_token "
            "WHERE lb.lidar_pc_token=? AND lb.track_token=? LIMIT 1",
            (lidar[0], track_token),
        ).fetchone()
        pose = connection.execute(
            "SELECT x,y,z,qw,qx,qy,qz FROM ego_pose WHERE token=?", (lidar[2],)
        ).fetchone()
        if pose is None:
            return None
        translation, rotation = _ego_pose(pose)
        if box is None:
            actor_positions.append([None, None])
            visible.append(False)
            actor_confidence.append(0.0)
            continue
        point = _global_to_ego(np.asarray(box[:3]), translation, rotation)
        actor_positions.append([float(point[0]), float(point[1])])
        confidence = float(box[3]) if np.isfinite(float(box[3])) else 0.0
        is_visible = bool(np.isfinite(point).all() and confidence >= 0.2)
        visible.append(is_visible)
        actor_confidence.append(confidence if is_visible else 0.0)
        actor_class = actor_class or str(box[4])
    if sum(visible) < 3:
        return None
    source_key = f"nuplan:{candidate['logfile']}:{_token(candidate['lidar_pc_token'])}"
    return {
        "protocol": "actor-motion-reference-v1",
        "sample_id": f"{source_key}:{candidate['chain_type']}",
        "scene_id": candidate["scene_token"],
        "dataset": "navsim_sensor_backed_nuplan",
        "chain_type": candidate["chain_type"],
        "candidate_scope": candidate["candidate_scope"],
        "source_key": source_key,
        "history_frame_paths": [item["path"] for item in selected_images[:len(history_offsets_s)]],
        "future_frame_paths": [item["path"] for item in selected_images[len(history_offsets_s):]],
        "history_times_s": list(history_offsets_s),
        "future_times_s": list(future_offsets_s),
        "history_timestamps_us": [item["timestamp_us"] for item in selected_images[:len(history_offsets_s)]],
        "future_timestamps_us": [item["timestamp_us"] for item in selected_images[len(history_offsets_s):]],
        **calibration,
        "actor_tracks": [{
            "actor_id": _token(track_token),
            "class_label": actor_class or "unknown",
            "times_s": list(future_offsets_s),
            "positions_ego_m": actor_positions,
            "visibility": visible,
            "confidence": actor_confidence,
            "source": "nuplan_lidar_box_center",
        }],
        "independent_reference": {
            "source": "nuplan_lidar_box_center",
            "future_action_used": False,
            "candidate_bank_used": False,
        },
    }


def build_manifest(
    db_paths: Iterable[Path],
    sensor_root: Path,
    *,
    max_per_chain: int,
    seed: str,
    camera_channel: str,
    tolerance_ms: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_by_chain: Counter[str] = Counter()
    for db_path in sorted(Path(path) for path in db_paths):
        connection = sqlite3.connect(db_path)
        try:
            candidates = _scene_candidates(connection)
            by_chain = defaultdict(list)
            for candidate in candidates:
                by_chain[candidate["chain_type"]].append(candidate)
            for chain in CHAIN_ORDER:
                if selected_by_chain[chain] >= max_per_chain:
                    continue
                ordered = sorted(
                    by_chain[chain],
                    key=lambda row: _stable_key(seed, row["logfile"], row["timestamp"], chain),
                )
                for candidate in ordered:
                    if selected_by_chain[chain] >= max_per_chain:
                        break
                    record = _build_one(
                        connection, candidate, sensor_root,
                        camera_channel=camera_channel,
                        history_offsets_s=(-1.5, -1.0, -0.5, 0.0),
                        future_offsets_s=tuple(0.5 * (index + 1) for index in range(8)),
                        tolerance_us=round(float(tolerance_ms) * 1000),
                    )
                    if record is None:
                        continue
                    selected.append(record)
                    selected_by_chain[chain] += 1
        finally:
            connection.close()
    counts = {chain: sum(row["chain_type"] == chain for row in selected) for chain in CHAIN_ORDER}
    return selected, {
        "protocol": "actor-motion-reference-v1",
        "num_records": len(selected),
        "counts_by_chain": counts,
        "history_frames": 4,
        "future_frames": 8,
        "future_times_s": [0.5 * (index + 1) for index in range(8)],
        "reference_source": "nuplan_lidar_box_center",
        "contains_wam_action": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-chain", type=int, default=10)
    parser.add_argument("--seed", default="iac-actor-motion-v1")
    parser.add_argument("--camera-channel", default="CAM_F0")
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    args = parser.parse_args()
    db_paths = sorted(args.db_root.glob("*.db")) if args.db_root.is_dir() else [args.db_root]
    records, summary = build_manifest(
        db_paths, args.sensor_root, max_per_chain=args.max_per_chain, seed=args.seed,
        camera_channel=args.camera_channel, tolerance_ms=args.tolerance_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary["output"] = str(args.output.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
