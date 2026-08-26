#!/usr/bin/env python3
"""Convert strict NAVSIM records into the image-probe manifest format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.counterfactual import densify_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--speed-factors", default="0.85,1.0,1.15")
    parser.add_argument("--lateral-offsets-m", default="-0.75,-0.375,0,0.375,0.75")
    parser.add_argument("--curvature-offsets-1pm", default="-0.02,0,0.02")
    args = parser.parse_args()
    rows = []
    for line in args.records.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        native = json.loads(line)
        history = list(native["history_images"])
        future = list(native["future_images"])
        if len(history) != 4 or len(future) != 4:
            raise ValueError("the image probe currently requires 4 history and 4 future frames")
        row = {
            "sample_id": native["source_key"],
            "scene_id": native["scene_name"],
            "history_frame_paths": history,
            "history_times_s": [-1.5, -1.0, -0.5, 0.0],
            "future_frame_paths": future,
            "future_times_s": native["future_times_s"],
            "intrinsics": native["camera_intrinsic"],
            "distortion": native.get("camera_distortion", []),
            "camera_to_ego": native.get("camera_to_ego") or [
                [*native["sensor2lidar_rotation"][0], native["sensor2lidar_translation"][0]],
                [*native["sensor2lidar_rotation"][1], native["sensor2lidar_translation"][1]],
                [*native["sensor2lidar_rotation"][2], native["sensor2lidar_translation"][2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "gt_candidate_id": "logged",
            "candidates": [{
                "candidate_id": "logged",
                "prior": 1.0,
                "trajectory": native["trajectory"],
            }],
            "metadata": {
                "dataset": "navsim",
                "source_key": native["source_key"],
                "timestamp_us": native["timestamp_us"],
                "future_images_source": native.get("future_images_source"),
                "trajectory_source": native.get("trajectory_source"),
                "camera_model": "undistorted_pinhole" if native.get("camera_distortion") else "pinhole",
            },
        }
        rows.append(densify_record(
            row,
            reference_candidate_id="logged",
            speed_factors=[float(value) for value in args.speed_factors.split(",") if value.strip()],
            lateral_offsets_m=[float(value) for value in args.lateral_offsets_m.split(",") if value.strip()],
            curvature_offsets_1pm=[float(value) for value in args.curvature_offsets_1pm.split(",") if value.strip()],
        ))
        if args.max_records and len(rows) >= args.max_records:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    print(json.dumps({"records": len(rows), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
