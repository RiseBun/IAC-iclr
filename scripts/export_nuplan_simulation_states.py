#!/usr/bin/env python3
"""Export NuPlan closed-loop states and native metric success as WAM annotations."""

from __future__ import annotations

import argparse
import glob
import json
import lzma
import pickle
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd

from iac_new.state_protocol import canonical_states_from_pose_arrays


def _nearest_indices(times: np.ndarray, targets: np.ndarray) -> list[int]:
    indices = [int(np.argmin(np.abs(times - target))) for target in targets]
    if len(set(indices)) != len(indices):
        raise ValueError("sampling produced duplicate simulation frames")
    return indices


def _metric_scores(root: Path) -> dict[str, dict]:
    paths = list((root / "aggregator_metric").glob("*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"expected one NuPlan aggregator parquet in {root}")
    table = pd.read_parquet(paths[0])
    rows = {}
    for _, row in table.iterrows():
        scenario = str(row.get("scenario") or "")
        log_name = row.get("log_name")
        if not scenario or scenario in {"unknown", "final_score"} or log_name is None:
            continue
        rows[scenario] = {
            "score": float(row["score"]),
            "scenario_type": str(row.get("scenario_type") or "unknown"),
            "log_name": str(log_name),
        }
    return rows


def _load_log(path: Path):
    packed = msgpack.unpackb(lzma.open(path, "rb").read(), raw=False, strict_map_key=False)
    if not isinstance(packed, bytes):
        raise ValueError(f"unexpected NuPlan simulation log envelope: {path}")
    return pickle.loads(packed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=4)
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    args = parser.parse_args()
    scores = _metric_scores(args.simulation_root)
    rows = []
    skipped = []
    pattern = str(args.simulation_root / "simulation_log" / "*" / "*" / "*" / "*" / "*.msgpack.xz")
    for name in sorted(glob.glob(pattern)):
        path = Path(name)
        scenario_name = path.stem.split(".")[0]
        if scenario_name not in scores:
            skipped.append({"path": str(path), "reason": "missing_aggregated_metric"})
            continue
        simulation = _load_log(path)
        samples = list(simulation.simulation_history.data)
        times = np.asarray([sample.ego_state.time_us * 1e-6 for sample in samples], dtype=np.float64)
        anchor_target = times[0] + (args.history_frames - 1) * args.interval_s
        offsets = np.arange(-(args.history_frames - 1), args.future_frames + 1, dtype=np.float64) * args.interval_s
        indices = _nearest_indices(times, anchor_target + offsets)
        selected = [samples[index].ego_state for index in indices]
        poses = np.asarray([[state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading] for state in selected])
        speeds = np.asarray([state.dynamic_car_state.speed for state in selected])
        yaw_rates = np.asarray([state.dynamic_car_state.angular_velocity for state in selected])
        selected_times = times[indices]
        states = canonical_states_from_pose_arrays(
            poses, selected_times, speeds=speeds, yaw_rates=yaw_rates,
            anchor_index=args.history_frames - 1,
        )
        metric = scores[scenario_name]
        rows.append({
            "source_key": f"nuplan:{metric['log_name']}:{scenario_name}",
            "scene_name": metric["log_name"],
            "scenario_name": scenario_name,
            "scenario_type": metric["scenario_type"],
            "history_ego_state": states[: args.history_frames].tolist(),
            "realized_future_ego_state": states[args.history_frames :].tolist(),
            "state_times_s": (selected_times[args.history_frames :] - selected_times[args.history_frames - 1]).tolist(),
            "task_score": metric["score"],
            "task_success": bool(metric["score"] >= args.success_threshold),
            "task_success_source": f"nuplan_weighted_aggregate_score>={args.success_threshold}",
            "state_reference_source": "nuplan_closed_loop_simulation_log",
            "simulation_log": str(path),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "protocol": "wam-ego-state-annotation-v1",
        "dataset": "nuplan",
        "simulation_root": str(args.simulation_root),
        "annotations": len(rows),
        "successes": sum(row["task_success"] for row in rows),
        "failures": sum(not row["task_success"] for row in rows),
        "success_threshold": args.success_threshold,
        "skipped": skipped,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
