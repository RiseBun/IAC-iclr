#!/usr/bin/env python3
"""Strip private future references and absolute paths from a benchmark manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def public_row(row: dict) -> dict:
    history = row.get("history_images") or row.get("history_frame_paths") or []
    intrinsics = row.get("intrinsics", row.get("camera_intrinsic"))
    dataset = str(row.get("dataset", "unknown"))
    source_key = row.get("source_key", row.get("sample_id", ""))
    return {
        "protocol": "iac-level1-benchmark-public-v1",
        "benchmark": "benchmark_v3" if str(row.get("split", "")).endswith("v3") else "benchmark_v1",
        "benchmark_id": row.get("benchmark_id"),
        "sample_id": row.get("sample_id", source_key),
        "source_key": source_key,
        "dataset": dataset,
        "scene_group": row.get("scene_group"),
        "split": row.get("split"),
        "stratum": row.get("stratum"),
        "camera": row.get("camera"),
        "history_times_s": row.get("history_times_s", [-1.5, -1.0, -0.5, 0.0]),
        "future_times_s": row.get("future_times_s"),
        "history_ego_state": row.get("history_ego_state"),
        "intrinsics": intrinsics,
        # Public calibration is expressed in the source-camera coordinate
        # system. Submission manifests must provide K scaled to their output
        # image size; keeping this explicit prevents the v3 resize/K bug.
        "intrinsics_coordinate_size": row.get("intrinsics_coordinate_size", [1920, 1080]),
        "distortion": row.get("distortion", row.get("camera_distortion")),
        "camera_to_ego": row.get("camera_to_ego"),
        "history_frame_ids": [Path(str(path)).name for path in history],
        "data_uri": f"{dataset.split('_')[0]}://{source_key}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(public_row(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"input_records": len(rows), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
