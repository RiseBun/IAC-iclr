#!/usr/bin/env python3
"""Build lineage-preserving WAM action branches from native NavSim records.

The output is a WAM-generation manifest, not generated imagery. Every branch
keeps the native ``source_key`` and history state. Only the logged branch has
an independently realized future; hypothetical branches are explicitly
marked unevaluated until a closed-loop execution produces their state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trajectory(row: dict[str, Any], mode: str, lateral_offset_m: float, yaw_offset_rad: float) -> np.ndarray:
    base = np.asarray(row["trajectory"], dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("native trajectory must have shape [T,3]")
    if mode == "logged":
        return base.copy()
    sign = 1.0 if mode == "left" else -1.0
    progress = np.linspace(0.0, 1.0, len(base), dtype=np.float64)
    result = base.copy()
    result[:, 1] += sign * float(lateral_offset_m) * progress
    result[:, 2] += sign * float(yaw_offset_rad) * progress
    return result


def _motion(trajectory: np.ndarray, times: np.ndarray) -> dict[str, float]:
    duration = max(float(times[-1]), 1e-6)
    return {
        "duration_s": duration,
        "forward_rate_mps": float(trajectory[-1, 0] / duration),
        "lateral_rate_mps": float(trajectory[-1, 1] / duration),
        "yaw_rate_rps": float(trajectory[-1, 2] / duration),
    }


def build_rows(
    native_rows: list[dict[str, Any]],
    *,
    lateral_offset_m: float,
    yaw_offset_rad: float,
    modes: tuple[str, ...],
    max_records: int = 0,
    model_id: str = "unspecified_wam",
    expected_history_count: int | None = None,
    expected_future_count: int | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for native in native_rows:
        required = ("source_key", "history_images", "history_ego_state", "realized_future_ego_state", "trajectory", "future_times_s")
        missing = [key for key in required if native.get(key) is None]
        if missing:
            raise ValueError(f"{native.get('source_key', '<unknown>')}: missing {missing}")
        times = np.asarray(native["future_times_s"], dtype=np.float64)
        if times.ndim != 1 or len(times) == 0 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"{native['source_key']}: invalid future_times_s")
        history_count = len(native["history_images"])
        future_count = len(native.get("future_images") or native["realized_future_ego_state"])
        if expected_history_count is not None and history_count != expected_history_count:
            raise ValueError(
                f"{native['source_key']}: expected {expected_history_count} history frames, got {history_count}"
            )
        if expected_future_count is not None and future_count != expected_future_count:
            raise ValueError(
                f"{native['source_key']}: expected {expected_future_count} future frames, got {future_count}"
            )
        for mode in modes:
            if mode not in {"logged", "left", "right"}:
                raise ValueError(f"unsupported branch mode: {mode}")
            trajectory = _trajectory(native, mode, lateral_offset_m, yaw_offset_rad)
            logged = mode == "logged"
            branch = dict(native)
            branch.update({
                "protocol": "wam-native-counterfactual-branch-v1",
                "record_type": "wam_generation_branch",
                "counterfactual_group_id": str(native["source_key"]),
                "branch_id": f"{native['source_key']}::branch={mode}",
                "branch_mode": mode,
                "wam_model_id": model_id,
                "window_spec": {
                    "history_frames": history_count,
                    "future_frames": future_count,
                    "history_source": "native_dataset",
                    "future_source": "wam_generated_pending",
                },
                "generation_contract": {
                    "condition": "history_images + history_ego_state + action_condition",
                    "output": "future_images",
                    "lineage_key": "branch_id",
                },
                "source_key": str(native["source_key"]),
                "action_condition": {
                    "trajectory": trajectory.tolist(),
                    "motion": _motion(trajectory, times),
                },
                "action_trajectory": trajectory.tolist(),
                "future_images": [],
                "future_images_source": "wam_pending",
                "wam_generation_status": "pending",
                "realized_state_available": logged,
                "realized_state_source": native.get("state_reference_source") if logged else None,
                "task_success": native.get("task_success") if logged else None,
                "task_success_source": native.get("task_success_source") if logged else None,
            })
            if not logged:
                branch["realized_future_ego_state"] = None
                branch["trajectory_source"] = "counterfactual_action_condition_only"
            output.append(branch)
            if max_records and len(output) >= max_records:
                return output
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", default="logged,left,right")
    parser.add_argument("--lateral-offset-m", type=float, default=0.75)
    parser.add_argument("--yaw-offset-rad", type=float, default=0.12)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--model-id", default="unspecified_wam")
    parser.add_argument("--expected-history-count", type=int)
    parser.add_argument("--expected-future-count", type=int)
    args = parser.parse_args()
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    rows = build_rows(
        read_jsonl(args.native),
        lateral_offset_m=args.lateral_offset_m,
        yaw_offset_rad=args.yaw_offset_rad,
        modes=modes,
        max_records=args.max_records,
        model_id=args.model_id,
        expected_history_count=args.expected_history_count,
        expected_future_count=args.expected_future_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "protocol": "wam-native-counterfactual-branch-v1",
        "native_rows": len(read_jsonl(args.native)),
        "branch_rows": len(rows),
        "branches_per_source": len(modes),
        "logged_branches": sum(row["branch_mode"] == "logged" for row in rows),
        "counterfactual_branches": sum(row["branch_mode"] != "logged" for row in rows),
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
