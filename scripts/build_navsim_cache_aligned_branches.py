#!/usr/bin/env python3
"""Build counterfactual branches whose native tokens exist in NAVSIM cache.

The previous smoke manifest selected consecutive source frames, while the
available metric cache contains a sparse set of native tokens.  This utility
does not remap tokens or use nearest-neighbour state: it filters native records
by exact `source_key` token and creates deterministic logged/left/right action
conditions for those exact records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _token(source_key: str) -> str:
    return source_key.rsplit(":", 1)[-1]


def _cached_tokens(cache_root: Path, log_name: str) -> set[str]:
    return {
        path.parent.name
        for path in (cache_root / log_name).glob("*/*/metric_cache.pkl")
    }


def _action(points: np.ndarray, mode: str) -> np.ndarray:
    if mode == "logged":
        return points.copy()
    ramp = np.linspace(0.0, 1.0, len(points), dtype=np.float64)
    result = points.copy()
    # Positive lateral is the left-side convention used by the native ego
    # state protocol.  Keep the first knot fixed so the intervention starts at
    # the anchor rather than teleporting the vehicle.
    sign = 1.0 if mode == "left" else -1.0
    result[:, 1] += sign * 0.75 * ramp
    result[:, 2] += sign * 0.06 * ramp
    return result


def _motion(points: np.ndarray, times: np.ndarray) -> dict[str, float]:
    duration = float(times[-1])
    return {
        "duration_s": duration,
        "forward_rate_mps": float(points[-1, 0] / duration),
        "lateral_rate_mps": float(points[-1, 1] / duration),
        "yaw_rate_rps": float(points[-1, 2] / duration),
    }


def build(records: list[dict[str, Any]], cache_root: Path, *, max_groups: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    for record in records:
        source_key = str(record.get("source_key") or "")
        log_name = str(record.get("log_name") or "")
        token = _token(source_key)
        if not source_key or not log_name or token in seen:
            skipped += 1
            continue
        if token not in _cached_tokens(cache_root, log_name):
            skipped += 1
            continue
        points = np.asarray(record.get("trajectory"), dtype=np.float64)
        times = np.asarray(record.get("future_times_s"), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or times.shape != (len(points),):
            skipped += 1
            continue
        if np.any(~np.isfinite(points)) or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            skipped += 1
            continue
        selected.append(record)
        seen.add(token)
        if max_groups and len(selected) >= max_groups:
            break

    output: list[dict[str, Any]] = []
    for record in selected:
        source_key = str(record["source_key"])
        points = np.asarray(record["trajectory"], dtype=np.float64)
        times = np.asarray(record["future_times_s"], dtype=np.float64)
        for mode in ("logged", "left", "right"):
            action = _action(points, mode)
            branch_id = f"{source_key}::branch={mode}"
            output.append({
                "protocol": "wam-native-counterfactual-branch-v1",
                "record_type": "wam_generation_branch",
                "dataset": "navsim",
                "source_key": source_key,
                "source_pkl": record.get("source_pkl"),
                "scene_name": record.get("scene_name"),
                "scene_token": record.get("scene_token"),
                "log_name": record.get("log_name"),
                "frame_idx": record.get("frame_idx"),
                "timestamp_us": record.get("timestamp_us"),
                "history_images": record.get("history_images", []),
                "future_images": [],
                "future_images_source": "wam_pending",
                "history_ego_state": record.get("history_ego_state"),
                "realized_future_ego_state": record.get("realized_future_ego_state") if mode == "logged" else None,
                "future_times_s": times.tolist(),
                "trajectory": action.tolist(),
                "trajectory_source": "navsim_native_realized_oracle" if mode == "logged" else "counterfactual_action_condition_only",
                "camera": record.get("camera"),
                "camera_intrinsic": record.get("camera_intrinsic"),
                "camera_distortion": record.get("camera_distortion", []),
                "camera_to_ego": record.get("camera_to_ego"),
                "sensor2lidar_rotation": record.get("sensor2lidar_rotation"),
                "sensor2lidar_translation": record.get("sensor2lidar_translation"),
                "counterfactual_group_id": source_key,
                "branch_id": branch_id,
                "branch_mode": mode,
                "action_condition": {"trajectory": action.tolist(), "motion": _motion(action, times)},
                "action_trajectory": action.tolist(),
                "wam_generation_status": "pending",
                "realized_state_available": mode == "logged" and record.get("realized_future_ego_state") is not None,
                "realized_state_source": "navsim_native_realized" if mode == "logged" else None,
                "task_success": None,
                "task_success_source": None,
            })

    summary = {
        "protocol": "navsim-cache-aligned-branch-build-v1",
        "native_records": len(records),
        "selected_groups": len(selected),
        "branch_rows": len(output),
        "skipped_records": skipped,
        "exact_token_join": True,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=20)
    args = parser.parse_args()
    rows, summary = build(_read(args.records), args.metric_cache, max_groups=args.max_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
