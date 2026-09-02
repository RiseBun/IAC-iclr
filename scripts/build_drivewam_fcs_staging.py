#!/usr/bin/env python3
"""Build a NAVSIM PDM rollout manifest from the frozen 580-row DriveWAM run.

The NAVSIM metric cache uses its scenario token as lookup key.  This script
joins the private benchmark rows to cached scenarios by the ego timestamp and
then joins the DriveWAM shard manifests by their global shard order.  Missing
cache rows are retained in a diagnostics JSON rather than silently dropped.
"""
from __future__ import annotations

import argparse
import json
import lzma
import pickle
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--wam-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--diagnostics", type=Path, required=True)
    args = ap.parse_args()

    all_rows = read_jsonl(args.benchmark)
    nav_rows = [r for r in all_rows if r.get("dataset") == "navsim"]

    time_to_cache: dict[int, str] = {}
    for pkl in args.cache_root.glob("**/metric_cache.pkl"):
        try:
            with lzma.open(pkl, "rb") as f:
                cache = pickle.load(f)
            time_to_cache[int(cache.ego_state.time_point.time_us)] = pkl.parent.name
        except Exception:
            continue

    raw: list[dict] = []
    for manifest in sorted(args.wam_root.glob("shard_*/manifest.json")):
        raw.extend(json.loads(manifest.read_text()))

    branches: list[dict] = []
    missing: list[dict] = []
    for global_idx, row in enumerate(nav_rows):
        cache_token = time_to_cache.get(int(row["timestamp_us"]))
        if cache_token is None:
            missing.append(
                {
                    "benchmark_id": row.get("benchmark_id"),
                    "source_key": row.get("source_key"),
                    "reason": "metric_cache_missing",
                }
            )
            continue
        if global_idx >= len(raw):
            missing.append(
                {
                    "benchmark_id": row.get("benchmark_id"),
                    "source_key": row.get("source_key"),
                    "reason": "wam_manifest_missing",
                }
            )
            continue
        points = np.asarray(raw[global_idx].get("predicted_action_trajectory"), dtype=float)
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]
        if points.shape == (3, 8):
            points = points.T
        if points.shape != (8, 3):
            missing.append(
                {
                    "benchmark_id": row.get("benchmark_id"),
                    "source_key": row.get("source_key"),
                    "reason": f"invalid_action_shape_{tuple(points.shape)}",
                }
            )
            continue
        branches.append(
            {
                "branch_id": f"{row['source_key']}::drivewam_native",
                "branch_mode": "predict",
                "source_key": row["source_key"],
                "sample_id": row.get("sample_id"),
                "dataset": "navsim",
                "counterfactual_group_id": row["source_key"],
                "future_times_s": row.get("future_times_s")
                or [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                "action_trajectory": points.tolist(),
                "action_trajectory_source": "drivewam_native_action",
                "cache_token": cache_token,
                "cache_token_source": "metric_cache_dir_from_timestamp_join",
                "future_images_source": "drivewam_generated",
                "action_injection_verified": False,
                "model_revision": "DriveWAM benchmark_v1 native4",
                "seed": raw[global_idx].get("seed"),
                "lineage": {"global_sample_index": global_idx},
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x) + "\n" for x in branches))
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.write_text(
        json.dumps(
            {
                "benchmark_rows": len(all_rows),
                "navsim_rows": len(nav_rows),
                "wam_rows": len(raw),
                "cache_rows": len(time_to_cache),
                "staged_rows": len(branches),
                "missing_rows": len(missing),
                "missing": missing,
            },
            indent=2,
        )
    )
    print(json.dumps({"staged_rows": len(branches), "missing_rows": len(missing)}))


if __name__ == "__main__":
    main()
