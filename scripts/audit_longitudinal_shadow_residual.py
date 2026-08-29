#!/usr/bin/env python3
"""Audit longitudinal shadow residuals against strong history and overlap controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=4).digest()
    return int(seed) + int.from_bytes(digest, "little") % 100000


def bootstrap(values: list[float], *, seed: int, draws: int = 20000) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "trimmed_mean_10pct": float(np.mean(np.sort(array)[max(0, len(array) // 10): max(1, len(array) - len(array) // 10)])),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "n": int(len(array)),
    }


def history_constant_acceleration(row: dict[str, Any]) -> np.ndarray:
    history = np.asarray(row.get("history_ego_state", row.get("metadata", {}).get("history_ego_state", [])), dtype=np.float64)
    times = np.asarray(row.get("history_times_s", [-1.5, -1.0, -0.5, 0.0]), dtype=np.float64)
    future_times = np.asarray(row["future_times_s"], dtype=np.float64)
    if history.ndim != 2 or history.shape[0] < 2 or history.shape[1] < 5:
        return np.full(len(future_times) - 1, np.nan)
    rel = times - times[-1]
    acceleration = float(np.clip(np.linalg.lstsq(np.column_stack([rel, np.ones(len(rel))]), history[:, 3], rcond=None)[0][0], -5.0, 3.0))
    speed = max(float(history[-1, 3]), 0.0)
    yaw_rate = float(history[-1, 4])
    out = []
    previous_t = 0.0
    previous_yaw = 0.0
    previous_x = 0.0
    for t in future_times:
        dt = float(t - previous_t)
        mid = previous_t + 0.5 * dt
        interval_speed = max(speed + acceleration * mid, 0.0)
        mid_yaw = previous_yaw + 0.5 * yaw_rate * dt
        previous_x += interval_speed * np.cos(mid_yaw) * dt
        out.append(previous_x)
        previous_yaw += yaw_rate * dt
        previous_t = float(t)
    return np.diff(np.asarray(out, dtype=np.float64))


def bucket(row: dict[str, Any]) -> str:
    candidates = row.get("candidates") or []
    gt = next((c for c in candidates if str(c.get("candidate_id")) == str(row.get("gt_candidate_id"))), None)
    if gt is None:
        return "unknown"
    traj = np.asarray(gt.get("trajectory", []), dtype=np.float64)
    if traj.ndim != 2 or len(traj) < 2:
        return "unknown"
    dt = np.diff(np.concatenate([[0.0], np.asarray(row["future_times_s"], dtype=np.float64)]))
    speed = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), traj[:, :2]]), axis=0), axis=1) / dt
    delta = float(speed[-1] - speed[0])
    if delta <= -1.0:
        return "braking"
    if delta >= 1.0:
        return "acceleration"
    return "cruise"


def audit(manifest_path: Path, probe_path: Path, *, seed: int = 0) -> dict[str, Any]:
    manifests = {str(row["sample_id"]): row for row in read_jsonl(manifest_path)}
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    rows = []
    pair_counter: Counter[str] = Counter()
    artifact_values = []
    for item in probe.get("results", []):
        sample_id = str(item.get("sample_id"))
        row = manifests.get(sample_id)
        if row is None:
            continue
        predicted = np.asarray(item.get("predicted_progress_m", []), dtype=np.float64)
        gt = np.asarray(item.get("gt_progress_m", []), dtype=np.float64)
        strong = history_constant_acceleration(row)
        n = min(len(predicted), len(gt), len(strong), len(row.get("future_frame_paths", [])) - 1)
        for index in range(n):
            if not np.isfinite(predicted[index]) or not np.isfinite(gt[index]) or not np.isfinite(strong[index]):
                continue
            pair_key = "|".join(row["future_frame_paths"][index:index + 2])
            pair_counter[pair_key] += 1
            if abs(float(predicted[index]) - 0.3393201223) < 1e-5:
                artifact_values.append({"sample_id": sample_id, "interval": index, "pair_key": pair_key, "value": float(predicted[index])})
            rows.append({
                "sample_id": sample_id,
                "scene_id": str(row.get("scene_id") or row.get("metadata", {}).get("scene_name") or sample_id),
                "bucket": bucket(row),
                "interval": index,
                "pair_key": pair_key,
                "predicted_error_m": float(abs(predicted[index] - gt[index])),
                "strong_null_error_m": float(abs(strong[index] - gt[index])),
                "gain_m": float(abs(strong[index] - gt[index]) - abs(predicted[index] - gt[index])),
            })
    def summarize(subset: list[dict[str, Any]], label: str) -> dict[str, Any]:
        method = [r["predicted_error_m"] for r in subset]
        strong = [r["strong_null_error_m"] for r in subset]
        gain = [r["gain_m"] for r in subset]
        return {
            "label": label,
            "n_intervals": len(subset),
            "method_mae": bootstrap(method, seed=stable_seed(seed, label + ":method")),
            "strong_null_mae": bootstrap(strong, seed=stable_seed(seed, label + ":null")),
            "paired_gain_null_minus_method": bootstrap(gain, seed=stable_seed(seed, label + ":gain")),
            "method_win_rate": float(np.mean(np.asarray(gain) > 0.0)) if gain else None,
        }
    by_bucket = {name: summarize([r for r in rows if r["bucket"] == name], name) for name in ("acceleration", "braking", "cruise", "unknown")}
    scene_groups = defaultdict(list)
    for row in rows:
        scene_groups[row["scene_id"]].append(row)
    scene_gain = [float(np.mean([r["gain_m"] for r in group])) for group in scene_groups.values()]
    unique_pairs = [key for key, count in pair_counter.items() if count == 1]
    duplicate_pairs = {key: count for key, count in pair_counter.items() if count > 1}
    unique_rows = [r for r in rows if r["pair_key"] in set(unique_pairs)]
    return {
        "protocol": "longitudinal-shadow-residual-audit-v1",
        "manifest": str(manifest_path.resolve()),
        "probe": str(probe_path.resolve()),
        "manifest_samples": len(manifests),
        "probe_samples": int(probe.get("num_evaluable", 0)),
        "interval_rows": len(rows),
        "scene_count": len(scene_groups),
        "all_window_summary": summarize(rows, "all_windows"),
        "unique_frame_pair_summary": summarize(unique_rows, "unique_frame_pairs"),
        "scene_cluster_gain": bootstrap(scene_gain, seed=stable_seed(seed, "scene_gain")),
        "by_bucket": by_bucket,
        "artifact_0p3393201223": {
            "occurrences": len(artifact_values),
            "unique_frame_pairs": len({item["pair_key"] for item in artifact_values}),
            "examples": artifact_values[:20],
        },
        "overlap": {
            "unique_frame_pairs": len(pair_counter),
            "duplicated_frame_pairs": len(duplicate_pairs),
            "max_reuse": max(pair_counter.values(), default=0),
        },
        "formal_metric_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = audit(args.manifest, args.probe, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"by_bucket", "artifact_0p3393201223"}}, indent=2))


if __name__ == "__main__":
    main()
