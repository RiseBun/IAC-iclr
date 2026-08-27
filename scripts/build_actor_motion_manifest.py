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
import cv2


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


def _project_ego_point(
    point_ego: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[list[float] | None, bool]:
    camera_to_ego = np.asarray(calibration["camera_to_ego"], dtype=np.float64)
    point_camera = camera_to_ego[:3, :3].T @ (
        np.asarray(point_ego, dtype=np.float64) - camera_to_ego[:3, 3]
    )
    if not np.isfinite(point_camera).all() or point_camera[2] <= 1e-3:
        return None, False
    pixels, _ = cv2.projectPoints(
        point_camera.reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        np.asarray(calibration["intrinsics"], dtype=np.float64),
        np.asarray(calibration["distortion"], dtype=np.float64),
    )
    u, v = map(float, pixels.reshape(2))
    width, height = calibration["image_size"]
    visible = bool(np.isfinite([u, v]).all() and 0.0 <= u < width and 0.0 <= v < height)
    return ([u, v] if visible else None), visible


def _project_box_xyxy(
    center_global: np.ndarray,
    *,
    width_m: float,
    length_m: float,
    height_m: float,
    yaw_rad: float,
    ego_translation: np.ndarray,
    ego_rotation: np.ndarray,
    calibration: dict[str, Any],
) -> list[float] | None:
    forward = np.asarray([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    lateral = np.asarray([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0])
    camera_to_ego = np.asarray(calibration["camera_to_ego"], dtype=np.float64)
    corners = []
    for longitudinal_sign in (-1.0, 1.0):
        for lateral_sign in (-1.0, 1.0):
            for vertical_sign in (-1.0, 1.0):
                point = np.asarray(center_global, dtype=np.float64).copy()
                point += longitudinal_sign * 0.5 * float(length_m) * forward
                point += lateral_sign * 0.5 * float(width_m) * lateral
                point[2] += vertical_sign * 0.5 * float(height_m)
                point_ego = _global_to_ego(point, ego_translation, ego_rotation)
                point_camera = camera_to_ego[:3, :3].T @ (
                    point_ego - camera_to_ego[:3, 3]
                )
                if not np.isfinite(point_camera).all() or point_camera[2] <= 1e-3:
                    continue
                pixel, _ = cv2.projectPoints(
                    point_camera.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                    np.asarray(calibration["intrinsics"], dtype=np.float64),
                    np.asarray(calibration["distortion"], dtype=np.float64),
                )
                corners.append(pixel.reshape(2))
    if len(corners) < 4:
        return None
    corners_array = np.asarray(corners, dtype=np.float64)
    image_width, image_height = calibration["image_size"]
    x0 = float(np.clip(np.min(corners_array[:, 0]), 0.0, image_width - 1.0))
    y0 = float(np.clip(np.min(corners_array[:, 1]), 0.0, image_height - 1.0))
    x1 = float(np.clip(np.max(corners_array[:, 0]), 0.0, image_width - 1.0))
    y1 = float(np.clip(np.max(corners_array[:, 1]), 0.0, image_height - 1.0))
    if x1 - x0 < 2.0 or y1 - y0 < 2.0:
        return None
    return [x0, y0, x1, y1]


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
    minimum_image_visible_frames: int,
) -> dict[str, Any] | None:
    anchor_timestamp = int(candidate["timestamp"])
    offsets = history_offsets_s + future_offsets_s
    image_rows = connection.execute(
        "SELECT im.timestamp, im.filename_jpg, im.ego_pose_token FROM image im JOIN camera cam ON cam.token=im.camera_token "
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
        selected_images.append({
            "offset_s": float(offset),
            "timestamp_us": int(image[0]),
            "path": str(path.resolve()),
            "ego_pose_token": image[2],
        })
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
    lidar_visibility = []
    image_visibility = []
    ground_contact_pixels = []
    actor_boxes_xyxy = []
    actor_confidence = []
    actor_class = None
    future_images = selected_images[len(history_offsets_s):]
    future_lidar = selected_lidar[len(history_offsets_s):]
    for image, lidar in zip(future_images, future_lidar):
        box = connection.execute(
            "SELECT lb.x,lb.y,lb.z,lb.width,lb.length,lb.height,lb.yaw,lb.confidence,c.name FROM lidar_box lb "
            "JOIN track tr ON tr.token=lb.track_token JOIN category c ON c.token=tr.category_token "
            "WHERE lb.lidar_pc_token=? AND lb.track_token=? LIMIT 1",
            (lidar[0], track_token),
        ).fetchone()
        pose = connection.execute(
            "SELECT x,y,z,qw,qx,qy,qz FROM ego_pose WHERE token=?", (image["ego_pose_token"],)
        ).fetchone()
        if pose is None:
            return None
        translation, rotation = _ego_pose(pose)
        if box is None:
            actor_positions.append([None, None])
            lidar_visibility.append(False)
            image_visibility.append(False)
            ground_contact_pixels.append([None, None])
            actor_boxes_xyxy.append([None, None, None, None])
            actor_confidence.append(0.0)
            continue
        center_global = np.asarray(box[:3], dtype=np.float64)
        bottom_global = center_global.copy()
        bottom_global[2] -= 0.5 * float(box[5])
        center_ego = _global_to_ego(center_global, translation, rotation)
        bottom_ego = _global_to_ego(bottom_global, translation, rotation)
        pixel, in_image = _project_ego_point(bottom_ego, calibration)
        actor_box = _project_box_xyxy(
            center_global,
            width_m=float(box[3]), length_m=float(box[4]), height_m=float(box[5]),
            yaw_rad=float(box[6]), ego_translation=translation, ego_rotation=rotation,
            calibration=calibration,
        )
        actor_positions.append([float(center_ego[0]), float(center_ego[1])])
        confidence = float(box[7]) if np.isfinite(float(box[7])) else 0.0
        lidar_visible = bool(np.isfinite(center_ego).all() and confidence >= 0.2)
        lidar_visibility.append(lidar_visible)
        image_visibility.append(bool(lidar_visible and in_image))
        ground_contact_pixels.append(pixel if pixel is not None else [None, None])
        actor_boxes_xyxy.append(actor_box if actor_box is not None else [None, None, None, None])
        actor_confidence.append(confidence if lidar_visible else 0.0)
        actor_class = actor_class or str(box[8])
    if sum(lidar_visibility) < 3:
        return None
    if sum(image_visibility) < int(minimum_image_visible_frames):
        return None
    source_key = f"nuplan:{candidate['logfile']}:{_token(candidate['lidar_pc_token'])}"
    return {
        "protocol": "actor-motion-reference-v3",
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
            "visibility": lidar_visibility,
            "lidar_visibility": lidar_visibility,
            "image_visibility": image_visibility,
            "ground_contact_pixels_uv": ground_contact_pixels,
            "actor_boxes_xyxy": actor_boxes_xyxy,
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
    minimum_image_visible_frames: int,
    max_per_log_per_chain: int,
    max_per_scene_per_chain: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_by_chain: Counter[str] = Counter()
    selected_by_log: Counter[tuple[str, str]] = Counter()
    selected_by_scene: Counter[tuple[str, str]] = Counter()
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
                    log_key = (chain, str(candidate["logfile"]))
                    scene_key = (chain, str(candidate["scene_token"]))
                    if selected_by_log[log_key] >= int(max_per_log_per_chain):
                        continue
                    if selected_by_scene[scene_key] >= int(max_per_scene_per_chain):
                        continue
                    record = _build_one(
                        connection, candidate, sensor_root,
                        camera_channel=camera_channel,
                        history_offsets_s=(-1.5, -1.0, -0.5, 0.0),
                        future_offsets_s=tuple(0.5 * (index + 1) for index in range(8)),
                        tolerance_us=round(float(tolerance_ms) * 1000),
                        minimum_image_visible_frames=minimum_image_visible_frames,
                    )
                    if record is None:
                        continue
                    selected.append(record)
                    selected_by_chain[chain] += 1
                    selected_by_log[log_key] += 1
                    selected_by_scene[scene_key] += 1
        finally:
            connection.close()
    counts = {chain: sum(row["chain_type"] == chain for row in selected) for chain in CHAIN_ORDER}
    return selected, {
        "protocol": "actor-motion-reference-v3",
        "num_records": len(selected),
        "counts_by_chain": counts,
        "history_frames": 4,
        "future_frames": 8,
        "future_times_s": [0.5 * (index + 1) for index in range(8)],
        "reference_source": "nuplan_lidar_box_center",
        "minimum_image_visible_frames": int(minimum_image_visible_frames),
        "max_per_log_per_chain": int(max_per_log_per_chain),
        "max_per_scene_per_chain": int(max_per_scene_per_chain),
        "unique_logs_by_chain": {
            chain: len({row["source_key"].split(":", 2)[1] for row in selected if row["chain_type"] == chain})
            for chain in CHAIN_ORDER
        },
        "unique_scenes_by_chain": {
            chain: len({row["scene_id"] for row in selected if row["chain_type"] == chain})
            for chain in CHAIN_ORDER
        },
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
    parser.add_argument("--minimum-image-visible-frames", type=int, default=3)
    parser.add_argument("--max-per-log-per-chain", type=int, default=2)
    parser.add_argument("--max-per-scene-per-chain", type=int, default=1)
    args = parser.parse_args()
    db_paths = sorted(args.db_root.glob("*.db")) if args.db_root.is_dir() else [args.db_root]
    records, summary = build_manifest(
        db_paths, args.sensor_root, max_per_chain=args.max_per_chain, seed=args.seed,
        camera_channel=args.camera_channel, tolerance_ms=args.tolerance_ms,
        minimum_image_visible_frames=args.minimum_image_visible_frames,
        max_per_log_per_chain=args.max_per_log_per_chain,
        max_per_scene_per_chain=args.max_per_scene_per_chain,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary["output"] = str(args.output.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
