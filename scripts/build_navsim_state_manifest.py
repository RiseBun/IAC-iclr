#!/usr/bin/env python3
"""Build a state-aware 4-history/4-future IAC manifest from NAVSIM records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.counterfactual import densify_record


def convert(native: dict, *, future_count: int) -> dict:
    history = list(native["history_images"])
    future = list(native["future_images"])
    if len(history) != 4 or len(future) < future_count:
        raise ValueError("NAVSIM record must contain 4 history frames and enough future frames")
    if len(native.get("history_ego_state", [])) != len(history):
        raise ValueError("history_ego_state must align with history images")
    if len(native.get("realized_future_ego_state", [])) < future_count:
        raise ValueError("realized_future_ego_state must align with selected future frames")
    selected_future = future[:future_count]
    selected_state = [list(state[:3]) for state in native["realized_future_ego_state"][:future_count]]
    selected_times = list(native["future_times_s"][:future_count])
    row = {
        "sample_id": str(native["source_key"]),
        "scene_id": str(native.get("scene_name") or native["source_key"]),
        "history_frame_paths": history,
        "history_times_s": [-1.5, -1.0, -0.5, 0.0],
        "future_frame_paths": selected_future,
        "future_times_s": selected_times,
        "intrinsics": native.get("camera_intrinsic"),
        "distortion": native.get("camera_distortion", []),
        "camera_to_ego": native.get("camera_to_ego"),
        "gt_candidate_id": "logged",
        "candidates": [{
            "candidate_id": "logged",
            "prior": 1.0,
            "trajectory": selected_state,
        }],
        "metadata": {
            "protocol": "navsim-state-aware-history4-future4-v1",
            "dataset": "navsim",
            "source_key": native["source_key"],
            "scene_name": native.get("scene_name"),
            "timestamp_us": native.get("timestamp_us"),
            "history_ego_state": native["history_ego_state"],
            "realized_future_ego_state": selected_state,
            "future_images_source": native.get("future_images_source"),
            "trajectory_source": native.get("trajectory_source"),
            "state_source": "navsim_native_realized",
        },
    }
    return densify_record(
        row,
        reference_candidate_id="logged",
        speed_factors=[0.85, 1.0, 1.15],
        lateral_offsets_m=[-0.5, 0.0, 0.5],
        curvature_offsets_1pm=[-0.015, 0.0, 0.015],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for line in args.records.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            handle.write(json.dumps(convert(json.loads(line), future_count=4), ensure_ascii=True) + "\n")
            count += 1
            if args.max_records and count >= args.max_records:
                break
    print(json.dumps({"records": count, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
