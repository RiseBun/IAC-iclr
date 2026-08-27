#!/usr/bin/env python3
"""Build a scene-aware Level-1 relative-progress stratified manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.continuous_motion import trajectory_to_motion_profile


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify_relative_stratum(
    row: dict[str, Any],
    *,
    speed_change_threshold_mps: float = 1.0,
    lateral_speed_threshold_mps: float = 0.08,
    yaw_rate_threshold_radps: float = 0.08,
) -> str:
    candidates = row.get("candidates") or []
    gt_id = str(row.get("gt_candidate_id"))
    candidate = next((item for item in candidates if str(item.get("candidate_id")) == gt_id), None)
    if candidate is None:
        raise ValueError(f"{row.get('sample_id')}: gt_candidate_id is missing from candidates")
    profile = trajectory_to_motion_profile(candidate["trajectory"], row["future_times_s"])
    speed = np.asarray([float(item["speed_mps"]) for item in profile["rows"]], dtype=np.float64)
    lateral = np.asarray([abs(float(item["lateral_speed_mps"])) for item in profile["rows"]], dtype=np.float64)
    yaw = np.asarray([abs(float(item["yaw_rate_radps"])) for item in profile["rows"]], dtype=np.float64)
    threshold = float(speed_change_threshold_mps)
    speed_delta = float(speed[-1] - speed[0])
    if speed_delta <= -threshold:
        return "braking"
    if speed_delta >= threshold:
        return "acceleration"
    if float(max(lateral.max(initial=0.0), yaw.max(initial=0.0))) >= max(
        float(lateral_speed_threshold_mps), float(yaw_rate_threshold_radps)
    ):
        return "lateral_turn"
    return "straight_cruise"


def select_scene_aware(
    rows: list[dict[str, Any]],
    strata: dict[str, str],
    *,
    max_per_stratum: int | None,
) -> list[dict[str, Any]]:
    if max_per_stratum is None:
        return list(rows)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[strata[row["sample_id"]]].append(row)
    selected: list[dict[str, Any]] = []
    for name in sorted(by_stratum):
        scene_seen: set[str] = set()
        for row in by_stratum[name]:
            scene_id = str(row.get("scene_id") or row["sample_id"])
            if scene_id in scene_seen:
                continue
            selected.append(row)
            scene_seen.add(scene_id)
            if len([item for item in selected if strata[item["sample_id"]] == name]) >= int(max_per_stratum):
                break
    selected_ids = {row["sample_id"] for row in selected}
    return [row for row in rows if row["sample_id"] in selected_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-per-stratum", type=int)
    parser.add_argument("--speed-change-threshold-mps", type=float, default=1.0)
    parser.add_argument("--lateral-speed-threshold-mps", type=float, default=0.08)
    parser.add_argument("--yaw-rate-threshold-radps", type=float, default=0.08)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    strata = {
        str(row["sample_id"]): classify_relative_stratum(
            row,
            speed_change_threshold_mps=args.speed_change_threshold_mps,
            lateral_speed_threshold_mps=args.lateral_speed_threshold_mps,
            yaw_rate_threshold_radps=args.yaw_rate_threshold_radps,
        )
        for row in rows
    }
    selected = select_scene_aware(rows, strata, max_per_stratum=args.max_per_stratum)
    annotated = []
    for row in selected:
        copy = dict(row)
        metadata = dict(copy.get("metadata") or {})
        metadata["relative_progress_stratum"] = strata[str(row["sample_id"])]
        metadata["relative_progress_stratum_protocol"] = "longitudinal-first-v1"
        copy["metadata"] = metadata
        annotated.append(copy)
    counts = Counter(strata.values())
    selected_counts = Counter(strata[str(row["sample_id"])] for row in selected)
    scene_counts = {
        name: len({str(row.get("scene_id") or row["sample_id"]) for row in rows if strata[str(row["sample_id"])] == name})
        for name in sorted(counts)
    }
    report = {
        "protocol": "level1-relative-progress-stratification-v1",
        "input_samples": len(rows),
        "selected_samples": len(selected),
        "thresholds": {
            "speed_change_threshold_mps": args.speed_change_threshold_mps,
            "lateral_speed_threshold_mps": args.lateral_speed_threshold_mps,
            "yaw_rate_threshold_radps": args.yaw_rate_threshold_radps,
        },
        "all_counts": dict(sorted(counts.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
        "scene_counts": scene_counts,
        "selected_scene_ids": sorted({str(row.get("scene_id") or row["sample_id"]) for row in selected}),
        "selection": {
            "max_per_stratum": args.max_per_stratum,
            "scene_unique_within_stratum": True,
        },
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in annotated), encoding="utf-8")
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
