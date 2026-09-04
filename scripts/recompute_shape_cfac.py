#!/usr/bin/env python3
"""Recompute a shape-only CFAC candidate from validated alignment lanes.

This transform never uses speed, acceleration, or metric forward displacement
in the formal CFAC candidate.  Those values are retained as diagnostics in the
input alignment and are explicitly excluded here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SHAPE_FIELDS = ("lateral_speed_mps", "yaw_rate_radps", "curvature_1pm")


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _shape_score(record: dict[str, Any]) -> float | None:
    comparison = record.get("comparison") or {}
    metrics = comparison.get("metrics") or {}
    tolerances = comparison.get("tolerances") or {}
    normalized = []
    for field in SHAPE_FIELDS:
        mae = (metrics.get(field) or {}).get("mae")
        tolerance = tolerances.get(field)
        if mae is None or tolerance in (None, 0):
            continue
        normalized.append(float(mae) / float(tolerance))
    return None if not normalized else float(np.exp(-np.mean(normalized)))


def _metric(record: dict[str, Any], key: str, nested: str | None = None) -> float | None:
    lane = record.get(key) or {}
    values = lane.get("metrics") or {}
    if nested is not None:
        values = values.get(nested) or {}
    elif "forward_displacement_profile" in values:
        values = values["forward_displacement_profile"] or {}
    value = values.get("path_cosine") if nested == "se2_pose" else values.get("curve_cosine")
    return None if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.alignment.read_text(encoding="utf-8"))
    rows = []
    for record in payload.get("records") or []:
        shape_score = _shape_score(record)
        arc_shape = _metric(record, "pose_alignment_arc_relative", "se2_pose")
        relative_shape = _metric(record, "distance_alignment_relative_observable")
        rows.append({
            "sample_id": record.get("sample_id"),
            "status": (record.get("comparison") or {}).get("status"),
            "coverage": (record.get("comparison") or {}).get("coverage"),
            "shape_cfac": shape_score,
            "arc_relative_path_cosine": arc_shape,
            "relative_observable_curve_cosine": relative_shape,
            "future_times_s": record.get("future_times_s"),
        })
    evaluable = [row for row in rows if row["shape_cfac"] is not None]
    output = {
        "protocol": "shape-relative-cfac-v1",
        "source_alignment": str(args.alignment),
        "formal_score_policy": {
            "included_motion_fields": list(SHAPE_FIELDS),
            "excluded_motion_fields": ["speed_mps", "acceleration_mps2"],
            "excluded_distance_mode": "metric_forward_displacement",
            "supporting_shape_lanes": ["arc_relative_path_cosine", "relative_observable_curve_cosine"],
        },
        "summary": {
            "samples_total": len(rows),
            "samples_evaluable": len(evaluable),
            "coverage": None if not rows else float(len(evaluable) / len(rows)),
            "cfac_shape": _mean([row["shape_cfac"] for row in evaluable]),
            "arc_relative_path_cosine": _mean([row["arc_relative_path_cosine"] for row in rows if row["arc_relative_path_cosine"] is not None]),
            "relative_observable_curve_cosine": _mean([row["relative_observable_curve_cosine"] for row in rows if row["relative_observable_curve_cosine"] is not None]),
        },
        "rows": rows,
        "note": "The supplied alignment is logged_gt image-measurement validation, not native WAM action. This is a shape-only CFAC candidate/upper-bound and must not be reported as formal WAM CFAC until reference_source=action is available.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
