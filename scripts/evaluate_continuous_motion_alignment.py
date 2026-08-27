#!/usr/bin/env python3
"""Evaluate image-derived ego motion against independently held-out waypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.continuous_motion import (
    compare_motion_profiles,
    image_motion_profile,
    trajectory_to_motion_profile,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _initial_speed(row: dict[str, Any]) -> float | None:
    metadata = row.get("metadata") or {}
    history = row.get("history_ego_state") or metadata.get("history_ego_state") or []
    if history and len(history[-1]) >= 4:
        return float(history[-1][3])
    return None


def _reference(row: dict[str, Any], source: str) -> list[list[float]]:
    metadata = row.get("metadata") or {}
    if source == "realized":
        value = row.get("realized_future_ego_state") or metadata.get("realized_future_ego_state")
        if value is None:
            raise ValueError(f"{row.get('sample_id')}: missing realized_future_ego_state")
        return [list(state[:3]) for state in value]
    if source == "action":
        value = row.get("action_trajectory") or metadata.get("action_trajectory")
        if value is None:
            raise ValueError(f"{row.get('sample_id')}: missing action_trajectory")
        return [list(state[:3]) for state in value]
    gt_id = str(row.get("gt_candidate_id"))
    for candidate in row.get("candidates") or []:
        if str(candidate.get("candidate_id")) == gt_id:
            return candidate["trajectory"]
    raise ValueError(f"{row.get('sample_id')}: logged GT candidate is missing")


def _coarse_longitudinal(profile: dict[str, Any], threshold: float = 0.5) -> str:
    values = [row.get("acceleration_mps2") for row in profile["rows"]]
    finite = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)])
    mean = float(np.mean(finite)) if len(finite) else 0.0
    return "decelerate" if mean < -threshold else ("accelerate" if mean > threshold else "cruise")


def aggregate(records: list[dict[str, Any]], reference_source: str) -> dict[str, Any]:
    valid = [record for record in records if record.get("comparison", {}).get("status") == "ok"]
    fields = ("speed_mps", "acceleration_mps2", "lateral_speed_mps", "yaw_rate_radps", "curvature_1pm")
    metrics: dict[str, Any] = {}
    for field in fields:
        values = [record["comparison"]["metrics"][field]["mae"] for record in valid]
        values = [float(value) for value in values if value is not None]
        within = [record["comparison"]["metrics"][field]["within_tolerance"] for record in valid]
        within = [float(value) for value in within if value is not None]
        metrics[field] = {
            "sample_mean_mae": None if not values else float(np.mean(values)),
            "sample_median_mae": None if not values else float(np.median(values)),
            "mean_within_tolerance": None if not within else float(np.mean(within)),
            "samples": len(values),
        }
    coarse_agreement = [record["coarse_event_audit"]["agreement"] for record in valid]
    same_event = [record for record in valid if record["coarse_event_audit"]["agreement"]]
    same_event_large = [
        record for record in same_event
        if record["comparison"]["metrics"]["speed_mps"]["mae"]
        > record["comparison"]["tolerances"]["speed_mps"]
    ]
    protocol_records = [record for record in records if record.get("future_times_s")]
    frame_counts = sorted({len(record["future_times_s"]) for record in records})
    horizons = [float(record["future_times_s"][-1]) for record in protocol_records]
    intervals = [
        float(value)
        for record in protocol_records
        for value in np.diff(np.concatenate([[0.0], np.asarray(record["future_times_s"], dtype=np.float64)]))
    ]
    coverage_risk: dict[str, list[dict[str, Any]]] = {}
    for field in ("speed_mps", "acceleration_mps2"):
        observations = []
        for record in records:
            for interval in record.get("comparison", {}).get("per_interval", []):
                error = interval.get("absolute_errors", {}).get(field)
                if error is not None and np.isfinite(error):
                    observations.append((float(interval["observability"]), float(error)))
        observations.sort(key=lambda item: item[0], reverse=True)
        curve = []
        for target in (0.25, 0.50, 0.75, 1.00):
            count = int(np.ceil(target * len(observations)))
            selected = observations[:count]
            curve.append({
                "target_coverage": target,
                "actual_coverage": None if not observations else float(count / len(observations)),
                "minimum_observability": None if not selected else float(selected[-1][0]),
                "mae": None if not selected else float(np.mean([item[1] for item in selected])),
                "intervals": len(selected),
            })
        coverage_risk[field] = curve
    return {
        "protocol": "continuous-motion-measurement-validation-v1",
        "evidence_scope": (
            "image_measurement_validation_only"
            if reference_source != "action"
            else "single_branch_image_action_alignment"
        ),
        "future_action_alignment_eligible": reference_source == "action",
        "causal_claim_eligible": False,
        "samples_total": len(records),
        "samples_evaluable": len(valid),
        "samples_missing_decoder_score": sum(record.get("status") == "missing_decoder_score" for record in records),
        "mean_interval_coverage": None if not valid else float(np.mean([record["comparison"]["coverage"] for record in valid])),
        "metrics": metrics,
        "coverage_risk_curve": coverage_risk,
        "speed_posterior_coverage": None if not valid else float(np.mean([
            record["comparison"]["speed_posterior_coverage"]
            for record in valid if record["comparison"]["speed_posterior_coverage"] is not None
        ])),
        "coarse_event_information_loss_audit": {
            "event_agreement": None if not coarse_agreement else float(np.mean(coarse_agreement)),
            "same_event_samples": len(same_event),
            "same_event_but_speed_mae_above_tolerance": len(same_event_large),
            "fraction": None if not same_event else float(len(same_event_large) / len(same_event)),
            "interpretation": "coarse events are diagnostic labels and do not replace continuous errors",
        },
        "observed_protocol": {
            "future_frame_counts": frame_counts,
            "future_horizon_s_min": None if not horizons else float(min(horizons)),
            "future_horizon_s_max": None if not horizons else float(max(horizons)),
            "median_interval_s": None if not intervals else float(np.median(intervals)),
            "meets_target_8_frames_4_seconds": frame_counts == [8] and horizons and all(abs(value - 4.0) <= 0.05 for value in horizons),
        },
        "failure_boundary": (
            "A single branch cannot establish counterfactual causality; paired controlled interventions are required."
            if reference_source == "action"
            else "No WAM action-head trajectory was used; this report cannot establish future-to-action causality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", choices=("logged_gt", "realized", "action"), default="logged_gt")
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--require-eight-frame-four-second", action="store_true")
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    scores = {str(row["sample_id"]): row for row in read_jsonl(args.scores)}
    records = []
    for row in manifest:
        sample_id = str(row["sample_id"])
        if sample_id not in scores:
            records.append({
                "sample_id": sample_id,
                "scene_id": row.get("scene_id"),
                "future_times_s": list(row["future_times_s"]),
                "reference_source": args.reference_source,
                "status": "missing_decoder_score",
            })
            continue
        times = list(row["future_times_s"])
        if args.require_eight_frame_four_second and (len(times) != 8 or abs(float(times[-1]) - 4.0) > 0.05):
            raise ValueError(f"{sample_id}: expected 8 future frames ending at 4.0 seconds")
        score = scores[sample_id]
        if score.get("candidate_bank_used_by_decoder") is not False:
            raise ValueError(f"{sample_id}: candidate-blind audit failed")
        initial_speed = _initial_speed(row)
        imagined = image_motion_profile(score["decoder"], times, initial_speed_mps=initial_speed)
        reference = trajectory_to_motion_profile(
            _reference(row, args.reference_source), times, initial_speed_mps=initial_speed
        )
        comparison = compare_motion_profiles(imagined, reference, include_uncertain=args.include_uncertain)
        image_event = _coarse_longitudinal(imagined)
        reference_event = _coarse_longitudinal(reference)
        records.append({
            "sample_id": sample_id,
            "scene_id": row.get("scene_id"),
            "future_times_s": times,
            "reference_source": args.reference_source,
            "image_motion_profile": imagined,
            "reference_motion_profile": reference,
            "comparison": comparison,
            "coarse_event_audit": {
                "image_event": image_event,
                "reference_event": reference_event,
                "agreement": image_event == reference_event,
                "primary_score": False,
            },
        })
    report = {
        "summary": aggregate(records, args.reference_source),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
