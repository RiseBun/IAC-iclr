#!/usr/bin/env python3
"""Estimate actor-relative metric state from candidate-blind image observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.relative_motion import (
    ActorPixelTrack,
    ActorRelativeTrack,
    estimate_actor_relative_motion,
    project_actor_pixel_track,
    validate_actor_future_window,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corridor-half-width-m", type=float, default=1.25)
    parser.add_argument("--minimum-samples", type=int, default=3)
    parser.add_argument("--minimum-span-s", type=float, default=1.0)
    parser.add_argument(
        "--require-eight-frame-four-second",
        action="store_true",
        help="require the formal 0.5...4.0 s actor window",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in read_jsonl(args.tracks):
            common = {
                "actor_id": str(row["actor_id"]),
                "class_label": str(row["class_label"]),
                "times_s": np.asarray(row["times_s"], dtype=np.float64),
                "visibility": (
                    None if row.get("visibility") is None
                    else np.asarray(row["visibility"], dtype=bool)
                ),
                "confidence": (
                    None if row.get("confidence") is None
                    else np.asarray(row["confidence"], dtype=np.float64)
                ),
            }
            if args.require_eight_frame_four_second:
                common["times_s"] = validate_actor_future_window(common["times_s"])
            projection = None
            if row.get("positions_ego_m") is not None:
                track = ActorRelativeTrack(
                    **common,
                    positions_ego_m=np.asarray(row["positions_ego_m"], dtype=np.float64),
                )
            elif row.get("pixels_uv") is not None:
                track, projection = project_actor_pixel_track(
                    ActorPixelTrack(
                        **common,
                        pixels_uv=np.asarray(row["pixels_uv"], dtype=np.float64),
                    ),
                    np.asarray(row["intrinsics"], dtype=np.float64),
                    np.asarray(row["camera_to_ego"], dtype=np.float64),
                    depth_m=(
                        None if row.get("depth_m") is None
                        else np.asarray(row["depth_m"], dtype=np.float64)
                    ),
                )
            else:
                raise ValueError("each row needs positions_ego_m or pixels_uv")
            result = estimate_actor_relative_motion(
                track,
                corridor_half_width_m=args.corridor_half_width_m,
                minimum_samples=args.minimum_samples,
                minimum_span_s=args.minimum_span_s,
            )
            if projection is not None:
                result["projection"] = projection
            if "sample_id" in row:
                result["sample_id"] = row["sample_id"]
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
