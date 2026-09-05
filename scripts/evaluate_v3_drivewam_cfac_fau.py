#!/usr/bin/env python3
"""Recompute v3 CFAC/FAU from frozen Level-1 outputs and private GT.

The WAM only submits generated frames and native actions.  This server-side
join reconstructs the image motion profile with the frozen probe, then joins
private NAVSIM realized states.  Rows without an exact common timestamp are
kept as unavailable rather than silently interpolated.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.continuous_motion import (
    SHAPE_FIELDS,
    compare_motion_profiles,
    history_only_motion_profile,
    image_motion_profile,
    trajectory_to_motion_profile,
)
from iac_new.foresight_metrics import evaluate_fau


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def history_speed(row: dict[str, Any]) -> float | None:
    history = (row.get("metadata") or {}).get("history_ego_state") or row.get("history_ego_state") or []
    return float(history[-1][3]) if history and len(history[-1]) >= 4 else None


def shape_gate(score: dict[str, Any], count: int) -> tuple[list[str], list[float], list[str]]:
    statuses, observability, reasons = [], [], []
    intervals = list(score.get("observability_by_future_interval") or [])
    for index in range(count):
        item = intervals[index] if index < len(intervals) else {}
        direction = bool(item.get("direction_observable"))
        curvature = str(item.get("curvature_status", "abstain"))
        if direction and curvature == "usable":
            statuses.append("usable"); reasons.append("direct_flow_geometry")
        elif direction and curvature == "uncertain":
            statuses.append("uncertain"); reasons.append("direct_flow_geometry_uncertain")
        else:
            statuses.append("abstain"); reasons.append("no_shape_support")
        observability.append(float(np.clip(item.get("effective_static_pixel_fraction", 0.0), 0.0, 1.0)))
    return statuses, observability, reasons


def exact_gt(row: dict[str, Any], target_times: list[float], tolerance: float = 0.02) -> list[list[float]] | None:
    times = list(row.get("future_times_s") or [])
    trajectory = list(row.get("trajectory") or [])
    if len(times) != len(trajectory):
        return None
    indices: list[int] = []
    for target in target_times:
        candidates = [i for i, value in enumerate(times) if abs(float(value) - float(target)) <= tolerance]
        if not candidates:
            return None
        indices.append(candidates[0])
    return [list(trajectory[i][:3]) for i in indices]


def bootstrap(values: list[float], seed: int = 0) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level1-input", type=Path, required=True)
    ap.add_argument("--level1-scores", type=Path, required=True)
    ap.add_argument("--private-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    inputs = {str(r["sample_id"]): r for r in read_jsonl(args.level1_input)}
    scores = {str(r["sample_id"]): r for r in read_jsonl(args.level1_scores)}
    private = {str(r["sample_id"]): r for r in read_jsonl(args.private_manifest)}
    records: list[dict[str, Any]] = []
    cfac_values: list[float] = []
    fau_f_values: list[float] = []
    fau_a_values: list[float] = []
    excluded_stop = 0
    for sample_id, row in inputs.items():
        score = scores.get(sample_id)
        private_row = private.get(sample_id)
        base = {
            "sample_id": sample_id,
            "stratum": (row.get("metadata") or {}).get("stratum"),
            "wam_model_id": row.get("wam_model_id"),
            "future_images_source": row.get("future_images_source"),
            "action_trajectory_source": row.get("action_trajectory_source"),
            "image_timestamps_s": list(row.get("future_times_s") or []),
            "action_timestamps_s": list(row.get("future_times_s") or []),
        }
        if score is None or private_row is None:
            base.update({"status": "unavailable", "reason": "missing_level1_or_private_join"})
            records.append(base); continue
        target_times = [float(x) for x in row["future_times_s"]]
        gt = exact_gt(private_row, target_times)
        if gt is None:
            base.update({
                "status": "unavailable",
                "reason": "private_gt_missing_exact_common_time_axis",
                "private_gt_timestamps_s": list(private_row.get("future_times_s") or []),
            })
            records.append(base); continue
        decoder = copy.deepcopy(score["decoder"])
        statuses, obs, reasons = shape_gate(score, len(target_times))
        decoder["shape_status_by_interval"] = statuses
        decoder["shape_observability_by_interval"] = obs
        decoder["flow_status_by_interval"] = [str(x.get("status", "abstain")) for x in score.get("observability_by_future_interval", [])]
        decoder["shape_fallback_reason_by_interval"] = reasons
        initial = history_speed(row)
        image = image_motion_profile(decoder, target_times, initial_speed_mps=initial)
        action = trajectory_to_motion_profile(row["action_trajectory"], target_times, initial_speed_mps=initial)
        truth = trajectory_to_motion_profile(gt, target_times, initial_speed_mps=initial)
        history = history_only_motion_profile(
            (row.get("metadata") or {}).get("history_ego_state") or [], target_times,
            history_times_s=row.get("history_times_s"), model="constant_acceleration_yaw_rate",
        )
        cfac = compare_motion_profiles(image, action, include_uncertain=True, include_shape_uncertain=True, primary_fields=set(SHAPE_FIELDS))
        fau = evaluate_fau(image, action, truth, history_profile=history)
        stop = base["stratum"] == "stop"
        base.update({
            "status": "ok",
            "excluded_from_motion_average": stop,
            "private_gt_timestamps_s": list(private_row.get("future_times_s") or []),
            "image_motion_profile": image,
            "native_action_motion_profile": action,
            "private_gt_motion_profile": truth,
            "history_motion_profile": history,
            "cfac": cfac,
            "fau": fau,
        })
        records.append(base)
        if stop:
            excluded_stop += 1
        elif cfac.get("primary_shape_composite") is not None:
            cfac_values.append(float(cfac["primary_shape_composite"]))
        if not stop and fau.get("fau_f") is not None: fau_f_values.append(float(fau["fau_f"]))
        if not stop and fau.get("fau_a") is not None: fau_a_values.append(float(fau["fau_a"]))
    fau = None if not fau_f_values or not fau_a_values else float(np.sqrt(np.mean(fau_f_values) * np.mean(fau_a_values)))
    report = {
        "protocol": "iac-benchmark-v3-cfac-fau-server-join-v1",
        "model": "drivewam_navsim_checkpoint_20260824",
        "rows": len(records),
        "summary": {
            "cfac": {"status": "ok" if cfac_values else "unavailable", "mean_primary_shape_composite": None if not cfac_values else float(np.mean(cfac_values)), "n": len(cfac_values), "ci95": bootstrap(cfac_values)},
            "fau_f": {"status": "ok" if fau_f_values else "unavailable", "mean": None if not fau_f_values else float(np.mean(fau_f_values)), "n": len(fau_f_values), "ci95": bootstrap(fau_f_values)},
            "fau_a": {"status": "ok" if fau_a_values else "unavailable", "mean": None if not fau_a_values else float(np.mean(fau_a_values)), "n": len(fau_a_values), "ci95": bootstrap(fau_a_values)},
            "fau": {"status": "ok" if fau is not None else "unavailable", "geometric_mean_of_aggregate_components": fau},
            "stop_samples_excluded_from_motion_average": excluded_stop,
            "unavailable_rows": sum(r.get("status") == "unavailable" for r in records),
            "primary_fields": list(SHAPE_FIELDS),
            "longitudinal_metric_fields": ["speed_mps", "acceleration_mps2"],
            "longitudinal_policy": "diagnostic_only",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
