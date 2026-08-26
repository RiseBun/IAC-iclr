#!/usr/bin/env python3
"""Evaluate image-derived maneuver events against an independent trajectory."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.maneuver import extract_maneuver


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reference_trajectory(record: dict[str, Any]) -> np.ndarray:
    if record.get("realized_future_ego_state") is not None:
        return np.asarray(record["realized_future_ego_state"], dtype=np.float64)[:, :3]
    if record.get("trajectory") is not None and not record.get("candidates"):
        return np.asarray(record["trajectory"], dtype=np.float64)
    gt_id = str(record["gt_candidate_id"])
    for candidate in record["candidates"]:
        if str(candidate["candidate_id"]) == gt_id:
            return np.asarray(candidate["trajectory"], dtype=np.float64)
    raise KeyError(f"missing gt candidate {gt_id} for {record['sample_id']}")


def macro_f1(confusion: dict[str, Counter[str]], labels: list[str]) -> float:
    values = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[truth][label] for truth in labels if truth != label)
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        values.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = {row["sample_id"]: row for row in read_jsonl(args.manifest)}
    scores = read_jsonl(args.scores)
    lateral_labels = ["straight", "left", "right"]
    lane_labels = ["keep_lane", "lane_change_left", "lane_change_right"]
    longitudinal_labels = ["stop", "cruise", "accelerate", "brake"]
    confusion = {
        "lateral": defaultdict(Counter),
        "lane": defaultdict(Counter),
        "longitudinal": defaultdict(Counter),
    }
    rows = []
    onset_errors = []
    for score in scores:
        record = records.get(score["sample_id"])
        trajectory = (score.get("decoder") or {}).get("trajectory")
        if trajectory is None:
            trajectory = score.get("predicted_trajectory")
        if record is None or trajectory is None:
            continue
        profile_support = (score.get("decoder") or {}).get("profile_support") or []
        if len(profile_support) == len(trajectory):
            times = np.asarray([item["time_s"] for item in profile_support], dtype=np.float64)
        else:
            times = np.asarray(record["future_times_s"], dtype=np.float64)[: len(trajectory)]
        kwargs = {
            "curvature_threshold_1pm": args.curvature_threshold,
            "heading_threshold_rad": args.heading_threshold,
            "stop_speed_mps": args.stop_speed,
            "speed_change_mps": args.speed_change,
            "lane_change_offset_m": args.lane_change_offset,
            "lane_change_heading_rad": args.lane_change_heading,
        }
        observed = extract_maneuver(np.asarray(trajectory), times, **kwargs)
        if args.temporal_heading_fallback:
            # The flow observability pass has a signed, scale-free temporal
            # heading transition. Use it only when the continuous decoder says
            # straight; this preserves its resolved turn direction while
            # recovering turns that the metric decoder collapses to straight.
            intervals = score.get("observability_by_future_interval") or []
            for index, item in enumerate(intervals[: len(observed["lateral_action"])]):
                signed = item.get("curvature_temporal_heading_delta_rad")
                if observed["lateral_action"][index] == "straight" and signed is not None and abs(float(signed)) >= args.temporal_heading_threshold:
                    observed["lateral_action"][index] = "left" if float(signed) > 0.0 else "right"
        reference = extract_maneuver(reference_trajectory(record)[: len(trajectory)], times, **kwargs)
        for truth, pred in zip(reference["lateral_action"], observed["lateral_action"]):
            confusion["lateral"][truth][pred] += 1
        for truth, pred in zip(reference["lane_change_action"], observed["lane_change_action"]):
            confusion["lane"][truth][pred] += 1
        for truth, pred in zip(reference["longitudinal_action"], observed["longitudinal_action"]):
            confusion["longitudinal"][truth][pred] += 1
        ref_events = reference["events"]
        obs_events = observed["events"]
        for event in ref_events:
            matching = [item for item in obs_events if item["type"] == event["type"]]
            if matching:
                closest = min(matching, key=lambda item: abs(item["onset_time"] - event["onset_time"]))
                onset_errors.append(abs(float(closest["onset_time"] - event["onset_time"])))
        rows.append({
            "sample_id": score["sample_id"],
            "reference": reference,
            "observed": observed,
            "mean_observability": float((score.get("road_relative_posterior") or {}).get("mean_observability", 0.0)),
        })

    def summarize(name: str, labels: list[str]) -> dict[str, Any]:
        table = confusion[name]
        total = sum(sum(table[truth].values()) for truth in labels)
        correct = sum(table[label][label] for label in labels)
        return {
            "accuracy": correct / max(total, 1),
            "macro_f1": macro_f1(table, labels),
            "num_intervals": total,
            "reference_counts": {label: sum(table[label].values()) for label in labels},
            "confusion": {truth: {pred: table[truth][pred] for pred in labels} for truth in labels},
        }

    summary = {
        "protocol": "iac-image-maneuver-event-v1",
        "candidate_blind": True,
        "scores": str(args.scores.resolve()),
        "manifest": str(args.manifest.resolve()),
        "num_samples": len(rows),
        "lateral_direction": summarize("lateral", lateral_labels),
        "lane_change": summarize("lane", lane_labels),
        "longitudinal_state_diagnostic": summarize("longitudinal", longitudinal_labels),
        "turn_onset_mae_s": float(np.mean(onset_errors)) if onset_errors else None,
        "turn_onset_matched_events": len(onset_errors),
        "thresholds": {
            "curvature_1pm": args.curvature_threshold,
            "heading_rad": args.heading_threshold,
            "stop_speed_mps": args.stop_speed,
            "speed_change_mps": args.speed_change,
            "lane_change_offset_m": args.lane_change_offset,
            "lane_change_heading_rad": args.lane_change_heading,
            "temporal_heading_threshold_rad": args.temporal_heading_threshold,
        },
        "limitations": [
            "lane-change labels are geometric proxies until lane topology labels are connected",
            "longitudinal state remains diagnostic because monocular speed is not in the primary score",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curvature-threshold", type=float, default=0.012)
    parser.add_argument("--heading-threshold", type=float, default=0.028)
    parser.add_argument("--stop-speed", type=float, default=0.75)
    parser.add_argument("--speed-change", type=float, default=0.75)
    parser.add_argument("--lane-change-offset", type=float, default=0.75)
    parser.add_argument("--lane-change-heading", type=float, default=0.08)
    parser.add_argument("--temporal-heading-threshold", type=float, default=0.02)
    parser.add_argument("--temporal-heading-fallback", dest="temporal_heading_fallback", action="store_true")
    parser.set_defaults(temporal_heading_fallback=False)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
