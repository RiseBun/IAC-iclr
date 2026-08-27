#!/usr/bin/env python3
"""Evaluate CoTracker actor motion against independent NuPlan lidar truth.

This is an oracle-initialized capability upper bound, not the final blind WAM
metric. One ground-contact query is supplied at the actor's first visible
future frame. That query frame and every earlier frame are excluded from all
reported errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from iac_new.cotracker import (
    CoTrackerExtractor,
    actor_box_query_points,
    aggregate_actor_region_tracks,
)
from iac_new.geometry import scale_intrinsics
from iac_new.relative_motion import (
    ActorPixelTrack,
    ActorRelativeTrack,
    estimate_actor_relative_motion,
    evaluate_relative_motion_metrics,
    project_actor_pixel_track,
    validate_actor_future_window,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _undistort_pixels(
    pixels_uv: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    output = np.full_like(pixels, np.nan)
    finite = np.isfinite(pixels).all(axis=1)
    if finite.any():
        output[finite] = cv2.undistortPoints(
            pixels[finite].reshape(-1, 1, 2),
            np.asarray(intrinsics, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
            P=np.asarray(intrinsics, dtype=np.float64),
        ).reshape(-1, 2)
    return output


def _support_by_time(result: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {round(float(item["time_s"]), 6): item for item in result.get("support", [])}


def _q50(item: dict[str, Any] | None, field: str) -> float | None:
    if item is None or not isinstance(item.get(field), dict):
        return None
    value = item[field].get("q50")
    return float(value) if value is not None and np.isfinite(float(value)) else None


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _pixel_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [value for sample in samples for value in sample["pixel_errors_px"]]
    num_gold = sum(sample["num_post_query_gold_frames"] for sample in samples)
    num_scored = sum(sample["num_pixel_scored_frames"] for sample in samples)
    return {
        "num_gold": num_gold,
        "num_scored": num_scored,
        "coverage": float(num_scored / max(num_gold, 1)),
        "epe_mean_px": _safe_mean(errors),
        "epe_median_px": float(np.median(errors)) if errors else None,
        "epe_p90_px": float(np.percentile(errors, 90)) if errors else None,
        "epe_p95_px": float(np.percentile(errors, 95)) if errors else None,
    }


def _nullable_pixels(values: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in point]
        for point in np.asarray(values, dtype=np.float64)
    ]


def evaluate_record(
    row: dict[str, Any],
    extractor: CoTrackerExtractor,
    *,
    target_size: tuple[int, int],
    corridor_half_width_m: float,
    query_mode: str,
) -> dict[str, Any]:
    if row.get("protocol") not in {"actor-motion-reference-v2", "actor-motion-reference-v3"}:
        raise ValueError("CoTracker actor evaluation requires reference v2 or v3")
    times = validate_actor_future_window(np.asarray(row["future_times_s"], dtype=np.float64))
    actor = row["actor_tracks"][0]
    image_visibility = np.asarray(actor["image_visibility"], dtype=bool)
    reference_pixels = np.asarray(actor["ground_contact_pixels_uv"], dtype=np.float64)
    query_candidates = np.flatnonzero(image_visibility & np.isfinite(reference_pixels).all(axis=1))
    if not len(query_candidates):
        raise ValueError("record has no finite visible actor query")
    query_index = int(query_candidates[0])

    source_size = tuple(map(int, row["image_size"]))
    scale = np.asarray(
        [target_size[0] / source_size[0], target_size[1] / source_size[1]],
        dtype=np.float64,
    )
    anchor_query = reference_pixels[query_index] * scale
    if query_mode == "actor-box-grid":
        if row.get("protocol") != "actor-motion-reference-v3":
            raise ValueError("actor-box-grid requires actor-motion-reference-v3")
        box = np.asarray(actor["actor_boxes_xyxy"][query_index], dtype=np.float64)
        box *= np.asarray([scale[0], scale[1], scale[0], scale[1]])
        try:
            query_points = actor_box_query_points(
                box, height=target_size[1], width=target_size[0]
            )
        except ValueError as error:
            scaled_reference_pixels = reference_pixels * scale
            post_query_gold = image_visibility & (np.arange(len(times)) > query_index)
            metric_rows = [{
                "sample_id": row["sample_id"],
                "chain_type": row["chain_type"],
                "actor_id": actor["actor_id"],
                "time_s": float(times[index]),
                "observability": 0.0,
                "abstain": True,
                "abstain_reason": "actor_box_too_small",
                "future_action_used": False,
                "candidate_bank_used": False,
            } for index in np.flatnonzero(post_query_gold)]
            return {
                "sample_id": row["sample_id"],
                "chain_type": row["chain_type"],
                "actor_id": actor["actor_id"],
                "query_frame_index": query_index,
                "query_time_s": float(times[query_index]),
                "query_mode": query_mode,
                "num_query_points": 0,
                "sample_abstain_reason": str(error),
                "num_post_query_gold_frames": int(post_query_gold.sum()),
                "num_pixel_scored_frames": 0,
                "pixel_errors_px": [],
                "pixel_tracks": {
                    "predicted_distorted_uv": [[None, None]] * len(times),
                    "reference_distorted_uv": _nullable_pixels(scaled_reference_pixels),
                    "reference_visibility": image_visibility.tolist(),
                    "tracker_visibility": [False] * len(times),
                    "scored": [False] * len(times),
                },
                "predicted_result": {
                    "protocol": "actor-relative-motion-v1",
                    "available": False,
                    "status": "abstain",
                    "abstain_reason": "actor_box_too_small",
                    "observability": 0.0,
                    "support": [],
                },
                "reference_result": None,
                "projection": None,
                "metric_rows": metric_rows,
            }
    else:
        query_points = anchor_query.reshape(1, 2).astype(np.float32)
    observation = extractor.observe(
        list(map(str, row["future_frame_paths"])),
        target_size=target_size,
        query_points=query_points,
        query_frame_indices=np.full(len(query_points), query_index, dtype=np.int64),
    )
    if observation.source_size != source_size:
        raise ValueError(
            f"manifest image_size {source_size} does not match image {observation.source_size}"
        )

    if query_mode == "actor-box-grid":
        predicted_pixels_distorted, tracker_visibility, tracker_confidence = (
            aggregate_actor_region_tracks(observation, anchor_query)
        )
    else:
        predicted_pixels_distorted = np.asarray(observation.tracks[:, 0], dtype=np.float64)
        tracker_visibility = np.asarray(observation.visibility[:, 0], dtype=np.float64) > 0.5
        tracker_confidence = np.clip(
            np.asarray(observation.confidence[:, 0], dtype=np.float64), 0.0, 1.0
        )
    scaled_intrinsics = scale_intrinsics(
        np.asarray(row["intrinsics"], dtype=np.float64), source_size, target_size
    )
    predicted_pixels = _undistort_pixels(
        predicted_pixels_distorted,
        scaled_intrinsics,
        np.asarray(row["distortion"], dtype=np.float64),
    )
    # The oracle query may not be used to infer frames before it appears.
    causal_window = np.arange(len(times)) >= query_index
    tracker_visibility &= causal_window

    predicted_track, projection = project_actor_pixel_track(
        ActorPixelTrack(
            actor_id=str(actor["actor_id"]),
            class_label=str(actor.get("class_label", "unknown")),
            times_s=times,
            pixels_uv=predicted_pixels,
            visibility=tracker_visibility,
            confidence=tracker_confidence,
        ),
        scaled_intrinsics,
        np.asarray(row["camera_to_ego"], dtype=np.float64),
    )
    predicted_result = estimate_actor_relative_motion(
        predicted_track, corridor_half_width_m=corridor_half_width_m
    )

    reference_positions = np.asarray(actor["positions_ego_m"], dtype=np.float64)
    reference_confidence = np.asarray(actor["confidence"], dtype=np.float64)
    reference_visibility = image_visibility & causal_window
    reference_result = estimate_actor_relative_motion(
        ActorRelativeTrack(
            actor_id=str(actor["actor_id"]),
            class_label=str(actor.get("class_label", "unknown")),
            times_s=times,
            positions_ego_m=reference_positions,
            visibility=reference_visibility,
            confidence=reference_confidence,
        ),
        corridor_half_width_m=corridor_half_width_m,
    )

    predicted_support = _support_by_time(predicted_result)
    reference_support = _support_by_time(reference_result)
    metric_rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(times):
        # Exclude the supplied oracle anchor itself as well as pre-query frames.
        if index <= query_index or not image_visibility[index]:
            continue
        pred_item = predicted_support.get(round(float(time_s), 6))
        ref_item = reference_support.get(round(float(time_s), 6))
        pred_position = predicted_track.positions_ego_m[index]
        ref_position = reference_positions[index]
        pred_speed = _q50(pred_item, "closing_speed_mps")
        ref_speed = _q50(ref_item, "closing_speed_mps")
        pred_lateral_speed = _q50(pred_item, "lateral_speed_mps")
        ref_lateral_speed = _q50(ref_item, "lateral_speed_mps")
        finite_pair = np.isfinite(pred_position).all() and np.isfinite(ref_position).all()
        abstain = (
            predicted_result.get("status") != "usable"
            or not finite_pair
            or pred_speed is None
            or ref_speed is None
        )
        metric_rows.append({
            "sample_id": row["sample_id"],
            "chain_type": row["chain_type"],
            "actor_id": actor["actor_id"],
            "time_s": float(time_s),
            "predicted_distance_m": float(np.linalg.norm(pred_position)) if finite_pair else None,
            "reference_distance_m": float(np.linalg.norm(ref_position)) if finite_pair else None,
            "predicted_closing_speed_mps": pred_speed,
            "reference_closing_speed_mps": ref_speed,
            "predicted_lateral_speed_mps": pred_lateral_speed,
            "reference_lateral_speed_mps": ref_lateral_speed,
            "predicted_ttc_s": None if pred_item is None else pred_item.get("corridor_conflict_ttc_s"),
            "reference_ttc_s": None if ref_item is None else ref_item.get("corridor_conflict_ttc_s"),
            "observability": float(predicted_result.get("observability", 0.0)),
            "abstain": bool(abstain),
            "abstain_reason": (
                "observability_below_usable_threshold"
                if predicted_result.get("status") != "usable"
                else "missing_metric_support" if abstain else None
            ),
            "future_action_used": False,
            "candidate_bank_used": False,
        })

    scaled_reference_pixels = reference_pixels * scale
    post_query_gold = image_visibility & (np.arange(len(times)) > query_index)
    pixel_scored = post_query_gold & tracker_visibility
    pixel_errors = np.linalg.norm(
        predicted_pixels_distorted - scaled_reference_pixels, axis=1
    )
    pixel_errors = pixel_errors[pixel_scored & np.isfinite(pixel_errors)]
    return {
        "sample_id": row["sample_id"],
        "chain_type": row["chain_type"],
        "actor_id": actor["actor_id"],
        "query_frame_index": query_index,
        "query_time_s": float(times[query_index]),
        "query_mode": query_mode,
        "num_query_points": int(len(query_points)),
        "num_post_query_gold_frames": int(post_query_gold.sum()),
        "num_pixel_scored_frames": int(pixel_scored.sum()),
        "pixel_errors_px": pixel_errors.tolist(),
        "pixel_tracks": {
            "predicted_distorted_uv": _nullable_pixels(predicted_pixels_distorted),
            "reference_distorted_uv": _nullable_pixels(scaled_reference_pixels),
            "reference_visibility": image_visibility.tolist(),
            "tracker_visibility": tracker_visibility.tolist(),
            "scored": pixel_scored.tolist(),
        },
        "predicted_result": predicted_result,
        "reference_result": reference_result,
        "projection": projection,
        "metric_rows": metric_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--corridor-half-width-m", type=float, default=1.25)
    parser.add_argument("--dangerous-ttc-s", type=float, default=4.0)
    parser.add_argument(
        "--query-mode", choices=("ground-contact", "actor-box-grid"),
        default="ground-contact",
    )
    args = parser.parse_args()

    records = _read_jsonl(args.manifest)
    if args.limit is not None:
        records = records[:max(int(args.limit), 0)]
    extractor = CoTrackerExtractor(
        device=args.device,
        checkpoint=args.checkpoint,
    )
    samples = []
    for index, row in enumerate(records):
        sample = evaluate_record(
            row,
            extractor,
            target_size=(args.width, args.height),
            corridor_half_width_m=args.corridor_half_width_m,
            query_mode=args.query_mode,
        )
        samples.append(sample)
        print(f"[{index + 1}/{len(records)}] {row['sample_id']}", flush=True)

    metric_rows = [item for sample in samples for item in sample["metric_rows"]]
    metric_report = evaluate_relative_motion_metrics(
        metric_rows, dangerous_ttc_s=args.dangerous_ttc_s
    )
    by_chain = {}
    for chain in sorted({sample["chain_type"] for sample in samples}):
        chain_samples = [sample for sample in samples if sample["chain_type"] == chain]
        chain_rows = [item for sample in chain_samples for item in sample["metric_rows"]]
        by_chain[chain] = {
            "num_samples": len(chain_samples),
            "pixel_tracking": _pixel_summary(chain_samples),
            "relative_motion_metrics": evaluate_relative_motion_metrics(
                chain_rows, dangerous_ttc_s=args.dangerous_ttc_s
            ),
        }
    report = {
        "protocol": "cotracker-actor-motion-oracle-v1",
        "manifest": str(args.manifest.resolve()),
        "num_samples": len(samples),
        "model": "CoTracker3-offline",
        "query_mode": args.query_mode,
        "target_size": [args.width, args.height],
        "oracle_initialization": True,
        "oracle_query": "first_image_visible_lidar_ground_contact",
        "query_and_pre_query_frames_scored": False,
        "future_action_used": False,
        "candidate_bank_used": False,
        "formal_blind_benchmark_ready": False,
        "pixel_tracking": _pixel_summary(samples),
        "relative_motion_metrics": metric_report,
        "by_chain": by_chain,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
