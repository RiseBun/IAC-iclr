#!/usr/bin/env python3
"""Select a deterministic, motion-diverse subset of Waymo segments.

This operates only on the small ``vehicle_pose`` parquet files.  Camera JPEG
parquets are downloaded after selection, preventing an expensive random crawl
of the full dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

POSE = "[VehiclePoseComponent].world_from_vehicle.transform"


def features(path: Path) -> dict[str, float | str]:
    d = pq.read_table(path, columns=[POSE]).to_pydict()[POSE]
    xy = np.asarray([np.asarray(t, dtype=float).reshape(4, 4)[:2, 3] for t in d])
    yaw = np.unwrap(np.asarray([math.atan2(float(np.asarray(t).reshape(4, 4)[1, 0]), float(np.asarray(t).reshape(4, 4)[0, 0])) for t in d]))
    dt = 0.1
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    accel = np.diff(speed) / dt
    yaw_rate = np.diff(yaw) / dt
    lateral = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    turn = float(np.max(np.abs(yaw_rate))) if len(yaw_rate) else 0.0
    amin = float(np.min(accel)) if len(accel) else 0.0
    amax = float(np.max(accel)) if len(accel) else 0.0
    if turn >= 0.12:
        bucket = "turn"
    elif amin <= -1.5:
        bucket = "braking"
    elif amax >= 1.5:
        bucket = "acceleration"
    else:
        bucket = "straight_cruise"
    return {"segment_id": path.stem, "bucket": bucket, "max_abs_yaw_rate_rps": turn, "min_accel_mps2": amin, "max_accel_mps2": amax, "lateral_span_m": lateral, "duration_s": len(d) * dt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--count", type=int, default=50)
    args = ap.parse_args()
    rows = [features(p) for p in sorted(args.pose_dir.glob("*.parquet"))]
    # Round-robin over buckets, strongest motion first within each bucket.
    buckets = {k: sorted([r for r in rows if r["bucket"] == k], key=lambda r: (float(r["max_abs_yaw_rate_rps"]), float(r["max_accel_mps2"])), reverse=True) for k in ("turn", "braking", "acceleration", "straight_cruise")}
    selected = []
    while len(selected) < min(args.count, len(rows)) and any(buckets.values()):
        for key in buckets:
            if buckets[key] and len(selected) < args.count:
                selected.append(buckets[key].pop(0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in selected), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": len(selected), "bucket_counts": {k: sum(r["bucket"] == k for r in selected) for k in buckets}}, indent=2))


if __name__ == "__main__":
    main()
