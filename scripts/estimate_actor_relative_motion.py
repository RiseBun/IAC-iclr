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
            rows = row.get("actor_tracks") if row.get("actor_tracks") is not None else [row]
            for actor_row in rows:
                actor_row = dict(actor_row)
                actor_row.setdefault("sample_id", row.get("sample_id"))
                actor_row.setdefault("intrinsics", row.get("intrinsics"))
                actor_row.setdefault("camera_to_ego", row.get("camera_to_ego"))
                actor_row.setdefault("depth_m", row.get("depth_m"))
                if actor_row.get("positions_ego_m") is None and actor_row.get("pixels_uv") is None:
                    raise ValueError("each actor row needs positions_ego_m or pixels_uv")
                common = {
                    "actor_id": str(actor_row["actor_id"]),
                    "class_label": str(actor_row.get("class_label", "unknown")),
                    "times_s": np.asarray(actor_row["times_s"], dtype=np.float64),
                    "visibility": (
                        None if actor_row.get("visibility") is None
                        else np.asarray(actor_row["visibility"], dtype=bool)
                    ),
                    "confidence": (
                        None if actor_row.get("confidence") is None
                        else np.asarray(actor_row["confidence"], dtype=np.float64)
                    ),
                }
                if args.require_eight_frame_four_second:
                    common["times_s"] = validate_actor_future_window(common["times_s"])
                projection = None
                if actor_row.get("positions_ego_m") is not None:
                    track = ActorRelativeTrack(
                        **common,
                        positions_ego_m=np.asarray(actor_row["positions_ego_m"], dtype=np.float64),
                    )
                else:
                    if actor_row.get("intrinsics") is None or actor_row.get("camera_to_ego") is None:
                        raise ValueError("pixel tracks need intrinsics and camera_to_ego")
                    track, projection = project_actor_pixel_track(
                        ActorPixelTrack(
                            **common,
                            pixels_uv=np.asarray(actor_row["pixels_uv"], dtype=np.float64),
                        ),
                        np.asarray(actor_row["intrinsics"], dtype=np.float64),
                        np.asarray(actor_row["camera_to_ego"], dtype=np.float64),
                        depth_m=(
                            None if actor_row.get("depth_m") is None
                            else np.asarray(actor_row["depth_m"], dtype=np.float64)
                        ),
                    )
                result = estimate_actor_relative_motion(
                    track,
                    corridor_half_width_m=args.corridor_half_width_m,
                    minimum_samples=args.minimum_samples,
                    minimum_span_s=args.minimum_span_s,
                )
                if projection is not None:
                    result["projection"] = projection
                if actor_row.get("sample_id") is not None:
                    result["sample_id"] = actor_row["sample_id"]
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
