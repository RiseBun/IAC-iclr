#!/usr/bin/env python3
"""Mine high-recall four-chain annotation candidates from nuPlan SQLite logs."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


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
    "starting_unprotected_cross_turn": (
        "unprotected_turn_or_merge",
        "unprotected_turn",
    ),
    "starting_unprotected_noncross_turn": (
        "unprotected_turn_or_merge",
        "unprotected_turn",
    ),
}


def _stable_key(seed: str, *values: object) -> str:
    payload = "\0".join([seed, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_hex(value: Any) -> str:
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _parse_offsets(value: str, *, require_nonempty: bool = True) -> tuple[float, ...]:
    offsets = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if require_nonempty and not offsets:
        raise ValueError("at least one frame offset is required")
    if len(set(offsets)) != len(offsets):
        raise ValueError("frame offsets must be distinct")
    return offsets


def _metadata_candidates(db_path: Path) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in TAG_SCOPE)
    query = f"""
        SELECT st.lidar_pc_token, st.type, lp.timestamp, lp.scene_token,
               lg.logfile, lg.location
        FROM scenario_tag AS st
        JOIN lidar_pc AS lp ON lp.token = st.lidar_pc_token
        JOIN lidar AS ld ON ld.token = lp.lidar_token
        JOIN log AS lg ON lg.token = ld.log_token
        WHERE st.type IN ({placeholders})
        ORDER BY lp.timestamp, st.type
    """
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    connection = sqlite3.connect(db_path)
    try:
        for token, tag, timestamp, scene_token, logfile, location in connection.execute(
            query, tuple(TAG_SCOPE)
        ):
            key = (_token_hex(token), int(timestamp))
            row = grouped.setdefault(
                key,
                {
                    "db_path": str(db_path.resolve()),
                    "log_name": str(logfile or db_path.stem),
                    "location": location,
                    "lidar_pc_token": key[0],
                    "scene_token": _token_hex(scene_token),
                    "anchor_timestamp_us": key[1],
                    "scenario_tags": [],
                },
            )
            row["scenario_tags"].append(str(tag))
    finally:
        connection.close()

    output: list[dict[str, Any]] = []
    for row in grouped.values():
        tags_by_chain: dict[str, list[str]] = defaultdict(list)
        scopes_by_chain: dict[str, set[str]] = defaultdict(set)
        for tag in row["scenario_tags"]:
            chain_type, scope = TAG_SCOPE[tag]
            tags_by_chain[chain_type].append(tag)
            scopes_by_chain[chain_type].add(scope)
        for chain_type, tags in tags_by_chain.items():
            output.append(
                {
                    **row,
                    "chain_type": chain_type,
                    "candidate_trigger_tags": sorted(tags),
                    "candidate_scope": sorted(scopes_by_chain[chain_type]),
                }
            )
    return output


def _diverse_sample(
    rows: Sequence[dict[str, Any]],
    *,
    max_per_chain: int,
    seed: str,
) -> list[dict[str, Any]]:
    by_chain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chain[str(row["chain_type"])].append(row)
    selected: list[dict[str, Any]] = []
    for chain_type, candidates in sorted(by_chain.items()):
        ordered = sorted(
            candidates,
            key=lambda row: _stable_key(
                seed,
                chain_type,
                row["log_name"],
                row["lidar_pc_token"],
            ),
        )
        unique_logs: list[dict[str, Any]] = []
        remainder: list[dict[str, Any]] = []
        used_logs: set[str] = set()
        for row in ordered:
            log_name = str(row["log_name"])
            if log_name not in used_logs:
                unique_logs.append(row)
                used_logs.add(log_name)
            else:
                remainder.append(row)
        selected.extend((unique_logs + remainder)[:max_per_chain])
    return selected


def _nearest_frames(
    db_path: Path,
    *,
    camera_channel: str,
    anchor_timestamp_us: int,
    offsets_s: Sequence[float],
    sensor_root: Path,
    tolerance_us: int,
) -> list[dict[str, Any]] | None:
    targets = [anchor_timestamp_us + round(offset * 1_000_000) for offset in offsets_s]
    lower = min(targets) - tolerance_us
    upper = max(targets) + tolerance_us
    query = """
        SELECT im.timestamp, im.filename_jpg
        FROM image AS im
        JOIN camera AS cam ON cam.token = im.camera_token
        WHERE cam.channel = ? AND im.timestamp BETWEEN ? AND ?
        ORDER BY im.timestamp
    """
    connection = sqlite3.connect(db_path)
    try:
        images = list(connection.execute(query, (camera_channel, lower, upper)))
    finally:
        connection.close()
    if not images:
        return None
    timestamps = [int(row[0]) for row in images]
    output: list[dict[str, Any]] = []
    for offset, target in zip(offsets_s, targets):
        insertion = bisect.bisect_left(timestamps, target)
        indices = [index for index in (insertion - 1, insertion) if 0 <= index < len(images)]
        nearest = min(indices, key=lambda index: abs(timestamps[index] - target))
        delta = timestamps[nearest] - target
        if abs(delta) > tolerance_us:
            return None
        relative_path = Path(str(images[nearest][1]))
        path_candidates = [
            sensor_root / relative_path,
            *(sensor_root / split / relative_path for split in ("mini", "trainval", "test")),
        ]
        resolved_path = next(
            (path for path in path_candidates if path.is_file()),
            path_candidates[0],
        )
        output.append(
            {
                "offset_s": float(offset),
                "timestamp_us": timestamps[nearest],
                "timestamp_error_us": int(delta),
                "path": str(resolved_path.resolve()),
            }
        )
    return output


def mine_candidates(
    db_paths: Iterable[Path],
    *,
    sensor_root: Path,
    max_per_chain: int = 40,
    history_offsets_s: Sequence[float] = (-2.0, -1.0, 0.0),
    future_offsets_s: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    camera_channel: str = "CAM_F0",
    tolerance_ms: float = 150.0,
    seed: str = "iac-causal-candidates-v1",
    require_images: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_per_chain < 1:
        raise ValueError("max_per_chain must be positive")
    offsets = tuple(history_offsets_s) + tuple(future_offsets_s)
    if any(offset > 0 for offset in history_offsets_s):
        raise ValueError("history offsets must be non-positive")
    if any(offset <= 0 for offset in future_offsets_s):
        raise ValueError("future offsets must be positive")
    tolerance_us = round(float(tolerance_ms) * 1000)
    if tolerance_us < 0:
        raise ValueError("tolerance_ms must be non-negative")

    metadata: list[dict[str, Any]] = []
    paths = sorted(Path(path) for path in db_paths)
    for db_path in paths:
        metadata.extend(_metadata_candidates(db_path))
    sampled = _diverse_sample(metadata, max_per_chain=max_per_chain, seed=seed)

    records: list[dict[str, Any]] = []
    missing_windows = 0
    missing_files = 0
    for row in sampled:
        frames = _nearest_frames(
            Path(row["db_path"]),
            camera_channel=camera_channel,
            anchor_timestamp_us=int(row["anchor_timestamp_us"]),
            offsets_s=offsets,
            sensor_root=sensor_root,
            tolerance_us=tolerance_us,
        )
        if frames is None:
            missing_windows += 1
            continue
        if require_images and not all(Path(frame["path"]).is_file() for frame in frames):
            missing_files += 1
            continue
        chain_type = str(row["chain_type"])
        source_key = f"nuplan:{row['log_name']}:{row['lidar_pc_token']}"
        records.append(
            {
                "protocol": "iac-causal-candidate-v1",
                "candidate_id": _stable_key(seed, source_key, chain_type)[:20],
                "source_key": source_key,
                "dataset": "nuplan",
                "chain_type": chain_type,
                "candidate_scope": row["candidate_scope"],
                "candidate_trigger_tags": row["candidate_trigger_tags"],
                "trigger_label_status": "candidate_only_requires_blind_confirmation",
                "counterfactual_pair_status": "not_constructed",
                "log_name": row["log_name"],
                "location": row["location"],
                "scene_token": row["scene_token"],
                "lidar_pc_token": row["lidar_pc_token"],
                "anchor_timestamp_us": row["anchor_timestamp_us"],
                "camera_channel": camera_channel,
                "history_images": [frame["path"] for frame in frames[: len(history_offsets_s)]],
                "history_offsets_s": list(history_offsets_s),
                "future_images": [frame["path"] for frame in frames[len(history_offsets_s) :]],
                "future_offsets_s": list(future_offsets_s),
                "frame_timestamp_errors_us": [frame["timestamp_error_us"] for frame in frames],
                "source_db": row["db_path"],
            }
        )

    counts = Counter(row["chain_type"] for row in records)
    summary = {
        "protocol": "iac-causal-candidate-mining-v1",
        "num_databases": len(paths),
        "num_tagged_candidates_before_sampling": len(metadata),
        "num_sampled_before_frame_checks": len(sampled),
        "num_records": len(records),
        "records_by_chain_type": dict(sorted(counts.items())),
        "missing_frame_windows": missing_windows,
        "missing_image_files": missing_files,
        "seed": seed,
        "candidate_only": True,
        "known_tag_coverage_gaps": ["vehicle_cut_in", "merge_gap"],
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-chain", type=int, default=40)
    parser.add_argument("--history-offsets-s", default="-2,-1,0")
    parser.add_argument("--future-offsets-s", default="1,2,3,4,5,6")
    parser.add_argument("--camera-channel", default="CAM_F0")
    parser.add_argument("--tolerance-ms", type=float, default=150.0)
    parser.add_argument("--seed", default="iac-causal-candidates-v1")
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()

    db_paths = sorted(args.db_root.glob("*.db"))
    if not db_paths:
        raise ValueError(f"no .db files found under {args.db_root}")
    records, summary = mine_candidates(
        db_paths,
        sensor_root=args.sensor_root,
        max_per_chain=args.max_per_chain,
        history_offsets_s=_parse_offsets(args.history_offsets_s),
        future_offsets_s=_parse_offsets(args.future_offsets_s),
        camera_channel=args.camera_channel,
        tolerance_ms=args.tolerance_ms,
        seed=args.seed,
        require_images=not args.allow_missing_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in records),
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
