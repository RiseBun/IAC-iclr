#!/usr/bin/env python3
"""Audit Level-1 error by independent ground-truth turn magnitude."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    def stats(key: str) -> dict:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
        }
    return {
        "count": len(rows),
        "mean_lateral_abs_m": stats("mean_lateral_abs_m"),
        "mean_yaw_abs_rad": stats("mean_yaw_abs_rad"),
        "endpoint_lateral_abs_m": stats("endpoint_lateral_abs_m"),
        "endpoint_yaw_abs_rad": stats("endpoint_yaw_abs_rad"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {str(row["sample_id"]): row for row in _read(args.manifest)}
    scores = {str(row["sample_id"]): row for row in _read(args.scores)}
    rows = []
    for sample_id, row in manifest.items():
        score = scores.get(sample_id)
        if score is None or not score.get("valid"):
            continue
        candidate = next(
            item for item in row["candidates"] if item["candidate_id"] == row["gt_candidate_id"]
        )
        action = np.asarray(candidate["trajectory"], dtype=np.float64)
        decoded = np.asarray(score["decoder"]["trajectory"], dtype=np.float64)
        comparison = score["comparison_to_logged_trajectory"]
        heading_error = (decoded[-1, 2] - action[-1, 2] + np.pi) % (2 * np.pi) - np.pi
        rows.append({
            "sample_id": sample_id,
            "gt_endpoint_abs_lateral_m": float(abs(action[-1, 1])),
            "gt_endpoint_abs_yaw_rad": float(abs(action[-1, 2])),
            "mean_lateral_abs_m": float(comparison["mean_lateral_abs_m"]),
            "mean_yaw_abs_rad": float(comparison["mean_yaw_abs_rad"]),
            "endpoint_lateral_abs_m": float(abs(decoded[-1, 1] - action[-1, 1])),
            "endpoint_yaw_abs_rad": float(abs(heading_error)),
        })

    bins = {
        "yaw_lt_0p3": [row for row in rows if row["gt_endpoint_abs_yaw_rad"] < 0.3],
        "yaw_0p3_to_0p8": [row for row in rows if 0.3 <= row["gt_endpoint_abs_yaw_rad"] < 0.8],
        "yaw_ge_0p8": [row for row in rows if row["gt_endpoint_abs_yaw_rad"] >= 0.8],
        "lateral_ge_6m": [row for row in rows if row["gt_endpoint_abs_lateral_m"] >= 6.0],
        "lateral_ge_8m": [row for row in rows if row["gt_endpoint_abs_lateral_m"] >= 8.0],
    }
    report = {
        "protocol": "level1-independent-turn-regime-audit-v1",
        "rows": len(rows),
        "bins": {name: _summary(values) for name, values in bins.items()},
        "interpretation": "Uses only real-image logged trajectories; no WAM action is exposed to the decoder.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
