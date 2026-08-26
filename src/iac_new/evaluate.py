"""Run the lightweight image-side trajectory probe."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .depth import load_cached_metric_depth, metric_depth_reliability_masks
from .flow import RaftFlowExtractor, cuda_peak_memory_mb
from .protocol import read_jsonl, validate_record, write_jsonl
from .perception import (
    PerceptionObservation,
    build_perception,
    semantic_motion_summary,
    temporal_road_consensus,
    trajectory_traversability,
)
from .region import build_trajectory_region
from .scoring import (
    candidate_score_dict,
    dynamic_suppression_weights,
    interval_observability,
    mass_prediction_set,
    posterior_from_energies,
    polygon_mask,
    predict_candidate_flows,
    score_candidate,
)
from .visualize import (
    render_image_diagnostics,
    render_metric_depth_diagnostics,
    render_trajectory_region,
    safe_visualization_name,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_record(
    record: dict[str, Any],
    extractor: RaftFlowExtractor,
    config: dict[str, Any],
    calibration: dict[str, Any] | None,
    visualization_path: Path | None = None,
    perception: Any | None = None,
) -> dict[str, Any]:
    image_cfg = config["image"]
    flow_observation = extractor.observe(
        record["frame_paths"],
        record["intrinsics"],
        record["distortion"],
        (int(image_cfg["width"]), int(image_cfg["height"])),
    )
    history_count = int(record["history_count"])
    future_start = history_count - 1
    future_observation = flow_observation.forward[future_start:]
    future_consistency = (
        None
        if flow_observation.consistency_masks is None
        else flow_observation.consistency_masks[future_start:]
    )
    perception_cfg = config.get("perception", {})
    perception_observation: PerceptionObservation | None = None
    if perception is not None:
        temporal_consensus = bool(perception_cfg.get("temporal_consensus", False))
        frame_paths = record["frame_paths"] if temporal_consensus else record["frame_paths"][:-1]
        perception_observation = perception.observe(
            frame_paths,
            target_size=(int(image_cfg["width"]), int(image_cfg["height"])),
            intrinsics=record["intrinsics"],
            distortion=record["distortion"],
        )
        if temporal_consensus:
            perception_observation = temporal_road_consensus(
                perception_observation,
                flow_observation.forward,
                road_dilation_px=int(perception_cfg.get("road_dilation_px", 4)),
                actor_dilation_px=int(perception_cfg.get("actor_dilation_px", 3)),
            )
        if len(perception_observation.traversable_masks) != len(flow_observation.forward):
            raise ValueError("perception intervals must match optical-flow intervals")
    mask_cfg = config["mask"]
    roi = polygon_mask(
        int(image_cfg["height"]),
        int(image_cfg["width"]),
        mask_cfg["polygon_normalized"],
    )
    score_cfg = config["score"]
    geometry_cfg = config.get("geometry", {"backend": "plane"})
    geometry_backend = str(geometry_cfg.get("backend", "plane"))
    metric_depth = None
    depth_reliability = None
    depth_diagnostics = None
    if geometry_backend == "cached_metric_depth":
        metric_depth = load_cached_metric_depth(
            record,
            geometry_cfg,
            expected_intervals=len(future_observation),
            expected_size=(int(image_cfg["width"]), int(image_cfg["height"])),
            expected_intrinsics=flow_observation.intrinsics,
        )
        depth_reliability, depth_diagnostics = metric_depth_reliability_masks(
            metric_depth,
            future_observation,
            min_depth_m=float(geometry_cfg.get("min_depth_m", 1.0)),
            max_depth_m=float(geometry_cfg.get("max_depth_m", 100.0)),
            confidence_quantile=float(geometry_cfg.get("confidence_quantile", 0.25)),
            observed_flow_quantile=float(
                geometry_cfg.get("observed_flow_quantile", 0.997)
            ),
        )
    elif geometry_backend != "plane":
        raise ValueError(f"unsupported geometry.backend: {geometry_backend}")
    predicted_flows: list[np.ndarray] = []
    geometry_validities: list[np.ndarray] = []
    for candidate in record["candidates"]:
        predicted, geometry_validity = predict_candidate_flows(
            trajectory=candidate["trajectory"],
            camera_to_ego=record["camera_to_ego"],
            intrinsics=flow_observation.intrinsics,
            height=int(image_cfg["height"]),
            width=int(image_cfg["width"]),
            depths_m=None if metric_depth is None else metric_depth.depths_m,
        )
        predicted_flows.append(predicted)
        geometry_validities.append(geometry_validity)
    common_geometry_masks = np.all(np.stack(geometry_validities), axis=0)
    if depth_reliability is not None:
        common_geometry_masks &= depth_reliability
    future_perception = (
        None if perception_observation is None else perception_observation.future(future_start)
    )
    semantic_weights = None
    semantic_gate_statistics = None
    if future_perception is not None and bool(perception_cfg.get("use_traversable_mask", False)):
        semantic_mode = str(perception_cfg.get("constraint_mode", "soft")).lower()
        road = np.asarray(future_perception.traversable_masks, dtype=bool)
        actors = np.asarray(future_perception.actor_masks, dtype=bool)
        if semantic_mode == "hard":
            common_geometry_masks &= road & ~actors
        elif semantic_mode == "soft":
            road_weight = float(perception_cfg.get("road_weight", 1.0))
            offroad_weight = float(perception_cfg.get("offroad_weight", 0.20))
            actor_weight = float(perception_cfg.get("actor_weight", 0.08))
            if not (0.0 < offroad_weight <= road_weight and 0.0 < actor_weight <= road_weight):
                raise ValueError("perception soft weights must satisfy 0 < actor/offroad <= road")
            semantic_weights = np.where(road, road_weight, offroad_weight).astype(np.float32)
            semantic_weights = np.where(actors, np.minimum(semantic_weights, actor_weight), semantic_weights)
            semantic_gate_statistics = [
                {
                    "interval_index": int(index),
                    "road_fraction": float(road[index].mean()),
                    "actor_fraction": float(actors[index].mean()),
                    "mean_weight": float(semantic_weights[index].mean()),
                }
                for index in range(len(road))
            ]
        else:
            raise ValueError("perception.constraint_mode must be 'soft' or 'hard'")
    dynamic_cfg = config.get("dynamic_suppression", {})
    if bool(dynamic_cfg.get("enabled", True)):
        dynamic_weights, best_rigid_residual = dynamic_suppression_weights(
            observed_flows=future_observation,
            predicted_flows=np.stack(predicted_flows),
            roi_mask=roi,
            consistency_masks=future_consistency,
            common_geometry_masks=common_geometry_masks,
            absolute_threshold_px=float(dynamic_cfg.get("absolute_threshold_px", 3.0)),
            relative_threshold=float(dynamic_cfg.get("relative_threshold", 0.15)),
        )
    else:
        dynamic_weights = np.ones(future_observation.shape[:-1], dtype=np.float32)
        best_rigid_residual = np.zeros(future_observation.shape[:-1], dtype=np.float32)
    if semantic_weights is not None:
        dynamic_weights = dynamic_weights * semantic_weights
    candidate_scores = []
    for candidate, predicted, geometry_validity in zip(
        record["candidates"], predicted_flows, geometry_validities
    ):
        score = score_candidate(
            candidate_id=candidate["candidate_id"],
            trajectory=candidate["trajectory"],
            camera_to_ego=record["camera_to_ego"],
            intrinsics=flow_observation.intrinsics,
            observed_flows=future_observation,
            roi_mask=roi,
            consistency_masks=future_consistency,
            min_valid_pixels=int(mask_cfg["min_valid_pixels"]),
            minimum_flow_scale_px=float(score_cfg["minimum_flow_scale_px"]),
            predicted_flows=predicted,
            predicted_validity=geometry_validity,
            dynamic_weights=dynamic_weights,
            dynamic_weight_floor=float(dynamic_cfg.get("minimum_weight", 0.05)),
            common_geometry_masks=common_geometry_masks,
            energy_metric=str(score_cfg.get("energy_metric", "normalized_median_epe")),
        )
        candidate_scores.append(score)

    obs_cfg = config.get("observability", {})
    history_observability = interval_observability(
        observed_flows=flow_observation.forward[:future_start],
        roi_mask=roi,
        consistency_masks=(
            None if flow_observation.consistency_masks is None else flow_observation.consistency_masks[:future_start]
        ),
        dynamic_weights=None,
        minimum_flow_scale_px=float(score_cfg["minimum_flow_scale_px"]),
        static_weight_threshold=float(obs_cfg.get("static_weight_threshold", 0.5)),
        min_effective_pixel_fraction=float(obs_cfg.get("min_effective_pixel_fraction", 0.02)),
        role="history",
        curvature_min_support_fraction=float(obs_cfg.get("curvature_min_support_fraction", 0.05)),
        curvature_min_lateral_contrast_rad=float(obs_cfg.get("curvature_min_lateral_contrast_rad", 0.02)),
        curvature_min_flow_gradient_px=float(obs_cfg.get("curvature_min_flow_gradient_px", 0.02)),
        curvature_reliable_lateral_contrast_rad=float(obs_cfg.get("curvature_reliable_lateral_contrast_rad", 1.43)),
    )
    future_observability = interval_observability(
        observed_flows=future_observation,
        roi_mask=roi,
        consistency_masks=future_consistency,
        dynamic_weights=(
            dynamic_weights * common_geometry_masks.astype(np.float32)
            if bool(dynamic_cfg.get("enabled", True))
            else None
        ),
        minimum_flow_scale_px=float(score_cfg["minimum_flow_scale_px"]),
        static_weight_threshold=float(obs_cfg.get("static_weight_threshold", 0.5)),
        min_effective_pixel_fraction=float(obs_cfg.get("min_effective_pixel_fraction", 0.02)),
        role="future",
        common_geometry_masks=common_geometry_masks,
        curvature_min_support_fraction=float(obs_cfg.get("curvature_min_support_fraction", 0.05)),
        curvature_min_lateral_contrast_rad=float(obs_cfg.get("curvature_min_lateral_contrast_rad", 0.02)),
        curvature_min_flow_gradient_px=float(obs_cfg.get("curvature_min_flow_gradient_px", 0.02)),
        curvature_reliable_lateral_contrast_rad=float(obs_cfg.get("curvature_reliable_lateral_contrast_rad", 1.43)),
    )
    observability = history_observability + future_observability
    relative_times = record["frame_times_s"] - float(record["anchor_time_s"])
    for index, item in enumerate(observability):
        item["interval_index"] = index
        item["is_boundary"] = index == future_start
        item["start_time_s"] = float(relative_times[index])
        item["end_time_s"] = float(relative_times[index + 1])
    # History is diagnostic until a history-conditioned prior is implemented.
    # Only evidence used by candidate scoring may invalidate the sample.
    abstain_reasons = [
        f"interval_{item['interval_index']}:{item['status']}"
        for item in future_observability
        if item["status"] != "good"
    ]
    if len(future_observability) > 1:
        boundary_support = float(future_observability[0]["effective_static_pixel_fraction"])
        internal_support = float(np.median([
            item["effective_static_pixel_fraction"] for item in future_observability[1:]
        ]))
        boundary_ratio = float(obs_cfg.get("boundary_relative_threshold", 0.35))
        if internal_support > 0.0 and boundary_support < boundary_ratio * internal_support:
            abstain_reasons.append("real_to_generated_domain_break")
    energies = np.asarray([score.energy for score in candidate_scores], dtype=np.float64)
    priors = np.asarray([candidate["prior"] for candidate in record["candidates"]])
    temperature = float(
        (calibration or {}).get("temperature", score_cfg.get("temperature", 1.0))
    )
    probabilities = posterior_from_energies(
        energies, temperature=temperature, priors=priors
    )
    target_coverage = float(
        (calibration or {}).get("target_coverage", score_cfg["target_coverage"])
    )
    if calibration and calibration.get("nll_threshold") is not None:
        from .scoring import conformal_prediction_set

        prediction_indices = conformal_prediction_set(
            probabilities, float(calibration["nll_threshold"])
        )
    else:
        prediction_indices = mass_prediction_set(probabilities, target_coverage)
    order = np.argsort(energies)
    candidate_ids = [score.candidate_id for score in candidate_scores]
    top_index = int(np.argmax(probabilities))
    gt_id = record["gt_candidate_id"]
    gt_index = candidate_ids.index(gt_id) if gt_id in candidate_ids else None
    mode_summaries, trajectory_region = build_trajectory_region(
        candidates=record["candidates"],
        probabilities=probabilities,
        selected_indices=prediction_indices,
        future_times_s=record["future_times_s"],
        target_coverage=target_coverage,
    )
    if perception_observation is not None:
        anchor_traversable = perception_observation.traversable_masks[future_start]
        feasibility_by_id = {}
        for candidate, mode in zip(record["candidates"], mode_summaries):
            feasibility = trajectory_traversability(
                candidate["trajectory"],
                anchor_traversable,
                record["camera_to_ego"],
                flow_observation.intrinsics,
                half_width_m=float(perception_cfg.get("trajectory_half_width_m", 1.1)),
            )
            mode["semantic_feasibility"] = feasibility
            feasibility_by_id[mode["candidate_id"]] = feasibility
        trajectory_region["semantic_feasibility_by_mode"] = feasibility_by_id
    dynamic_statistics = []
    for interval_index, (weights, residual, observed) in enumerate(zip(
        dynamic_weights, best_rigid_residual, future_observation
    )):
        valid_roi = roi & np.isfinite(observed).all(axis=-1)
        valid_roi &= common_geometry_masks[interval_index]
        if future_consistency is not None:
            valid_roi &= future_consistency[interval_index]
        dynamic_statistics.append(
            {
                "mean_weight": float(np.mean(weights[valid_roi])) if valid_roi.any() else 0.0,
                "suppressed_fraction": (
                    float(np.mean(weights[valid_roi] < float(dynamic_cfg.get("minimum_weight", 0.05))))
                    if valid_roi.any()
                    else 1.0
                ),
                "median_best_rigid_residual_px": (
                    float(np.median(residual[valid_roi])) if valid_roi.any() else None
                ),
            }
        )
    semantic_motion = (
        None
        if future_perception is None
        else semantic_motion_summary(
            future_perception.actor_masks,
            dynamic_weights,
            roi_mask=roi,
            dynamic_threshold=float(dynamic_cfg.get("semantic_dynamic_threshold", 0.5)),
        )
    )
    if semantic_motion is not None:
        for item, semantic in zip(dynamic_statistics, semantic_motion):
            item["semantic_actor_fraction"] = semantic["actor_fraction"]
            item["semantic_actor_dynamic_fraction"] = semantic["actor_dynamic_fraction"]
            item["semantic_actor_classification"] = semantic["classification"]
    result = {
        "sample_id": record["sample_id"],
        "scene_id": record["scene_id"],
        "top_candidate_id": candidate_ids[top_index],
        "top1_correct": gt_id is not None and candidate_ids[top_index] == gt_id,
        "gt_candidate_id": gt_id,
        "gt_in_prediction_set": gt_index in prediction_indices if gt_index is not None else None,
        "prediction_set_ids": [candidate_ids[index] for index in prediction_indices],
        "prediction_set_size": len(prediction_indices),
        "candidate_order": [candidate_ids[index] for index in order],
        "candidate_scores": [candidate_score_dict(score) for score in candidate_scores],
        "candidate_priors": {
            candidate_ids[index]: float(priors[index])
            for index in range(len(candidate_ids))
        },
        "probabilities": {
            candidate_ids[index]: float(probabilities[index])
            for index in range(len(candidate_ids))
        },
        "temperature": temperature,
        "target_coverage": target_coverage,
        "flow_model": extractor.model_size,
        "forward_backward": extractor.use_forward_backward,
        "geometry_backend": geometry_backend,
        "metric_depth": (
            None
            if metric_depth is None
            else {
                "source": metric_depth.source,
                "cache_path": metric_depth.cache_path,
                "scale_divisor": metric_depth.scale_divisor,
                "intervals": depth_diagnostics,
            }
        ),
        "perception": (
            None
            if perception_observation is None
            else {
                "backend": perception_observation.backend,
                "model_id": perception_observation.model_id,
                "future_traversable_fraction": [
                    float(mask.mean()) for mask in future_perception.traversable_masks
                ],
                "future_actor_fraction": [
                    float(mask.mean()) for mask in future_perception.actor_masks
                ],
                "future_actor_motion": semantic_motion,
                "used_as_geometry_constraint": bool(
                    perception_cfg.get("use_traversable_mask", False)
                ),
                "constraint_mode": str(perception_cfg.get("constraint_mode", "soft")),
                "temporal_consensus": bool(perception_cfg.get("temporal_consensus", False)),
                "semantic_gate_statistics": semantic_gate_statistics,
            }
        ),
        "metadata": record["metadata"],
        "protocol_variant": record["protocol_variant"],
        "observability": observability,
        "valid": not abstain_reasons,
        "abstain_reasons": abstain_reasons,
        "dynamic_suppression": {
            "enabled": bool(dynamic_cfg.get("enabled", True)),
            "method": (
                "shared_minimum_rigid_residual"
                if bool(dynamic_cfg.get("enabled", True))
                else "disabled"
            ),
            "future_intervals": dynamic_statistics,
        },
        "mode_summaries": mode_summaries,
        "trajectory_region": trajectory_region,
    }
    if visualization_path is not None:
        render_trajectory_region(
            output_path=visualization_path,
            sample_id=record["sample_id"],
            mode_summaries=mode_summaries,
            region=trajectory_region,
            observability=observability,
        )
        image_diagnostics_path = visualization_path.with_name(
            f"{visualization_path.stem}_image_detection.png"
        )
        render_image_diagnostics(
            output_path=image_diagnostics_path,
            sample_id=record["sample_id"],
            frame_paths=record["frame_paths"],
            intrinsics=record["intrinsics"],
            distortion=record["distortion"],
            target_size=(int(image_cfg["width"]), int(image_cfg["height"])),
            observed_flows=flow_observation.forward,
            consistency_masks=flow_observation.consistency_masks,
            roi_mask=roi,
            future_start=future_start,
            future_dynamic_weights=dynamic_weights,
            top_future_predicted_flows=predicted_flows[top_index],
            observability=observability,
        )
        result["visualization_path"] = str(visualization_path)
        result["visualizations"] = {
            "trajectory_region": str(visualization_path),
            "image_detection": str(image_diagnostics_path),
        }
        if metric_depth is not None and depth_reliability is not None:
            depth_diagnostics_path = visualization_path.with_name(
                f"{visualization_path.stem}_metric_depth.png"
            )
            render_metric_depth_diagnostics(
                output_path=depth_diagnostics_path,
                sample_id=record["sample_id"],
                frame_paths=record["frame_paths"][future_start : future_start + len(metric_depth.depths_m)],
                intrinsics=record["intrinsics"],
                distortion=record["distortion"],
                target_size=(int(image_cfg["width"]), int(image_cfg["height"])),
                depths_m=metric_depth.depths_m,
                confidence=metric_depth.confidence,
                reliability_masks=(
                    common_geometry_masks & np.broadcast_to(roi, common_geometry_masks.shape)
                ),
                source=metric_depth.source,
            )
            result["visualizations"]["metric_depth"] = str(depth_diagnostics_path)
        if perception_observation is not None:
            trajectory_overlay_path = visualization_path.with_name(
                f"{visualization_path.stem}_trajectory_overlay.png"
            )
            from .visualize import render_image_trajectory_overlay

            render_image_trajectory_overlay(
                output_path=trajectory_overlay_path,
                sample_id=record["sample_id"],
                frame_path=record["frame_paths"][future_start],
                intrinsics=flow_observation.intrinsics,
                source_intrinsics=record["intrinsics"],
                distortion=record["distortion"],
                target_size=(int(image_cfg["width"]), int(image_cfg["height"])),
                camera_to_ego=record["camera_to_ego"],
                mode_summaries=mode_summaries,
                traversable_mask=perception_observation.traversable_masks[future_start],
                actor_mask=perception_observation.actor_masks[future_start],
            )
            result["visualizations"]["trajectory_overlay"] = str(trajectory_overlay_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    calibration = load_json(args.calibration) if args.calibration else None
    raw_rows = read_jsonl(args.manifest)
    if args.max_samples is not None:
        raw_rows = raw_rows[: args.max_samples]
    records = [
        validate_record(row, manifest_root=args.manifest.parent) for row in raw_rows
    ]
    flow_cfg = config["flow"]
    extractor = RaftFlowExtractor(
        model_size=str(flow_cfg["model"]),
        device=args.device,
        updates=int(flow_cfg["updates"]),
        batch_size=int(flow_cfg["batch_size"]),
        forward_backward=bool(flow_cfg["forward_backward"]),
        fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
        fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
    )
    perception = build_perception(config, device=args.device)
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        try:
            visualization_path = None
            if args.visualization_dir is not None:
                visualization_path = args.visualization_dir / safe_visualization_name(record["sample_id"])
            rows.append(
                evaluate_record(
                    record, extractor, config, calibration, visualization_path, perception
                )
            )
        except Exception as error:  # benchmark records invalidity instead of hiding it
            if args.fail_fast:
                raise
            invalid.append({"sample_id": record["sample_id"], "error": str(error)})
        print(json.dumps({"completed": index, "total": len(records)}), flush=True)
    elapsed = time.perf_counter() - started
    write_jsonl(args.output, rows)
    gt_rows = [row for row in rows if row["gt_candidate_id"] is not None]
    usable_gt_rows = [row for row in gt_rows if row["valid"]]
    pairwise_controls: dict[str, list[bool]] = {}
    for row in gt_rows:
        energy_by_id = {
            item["candidate_id"]: float(item["energy"])
            for item in row["candidate_scores"]
        }
        gt_id = str(row["gt_candidate_id"])
        gt_energy = energy_by_id[gt_id]
        for candidate_id, energy in energy_by_id.items():
            if candidate_id == gt_id:
                continue
            pairwise_controls.setdefault(candidate_id, []).append(gt_energy < energy)
    summary = {
        "protocol": "iac-new-image-v1",
        "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "num_input": len(records),
        "num_scored": len(rows),
        "num_usable": sum(bool(row["valid"]) for row in rows),
        "num_valid": len(rows),
        "num_invalid": len(invalid),
        "invalid_fraction": len(invalid) / len(records) if records else None,
        "top1_accuracy": (
            float(np.mean([row["top1_correct"] for row in gt_rows])) if gt_rows else None
        ),
        "top1_accuracy_usable": (
            float(np.mean([row["top1_correct"] for row in usable_gt_rows]))
            if usable_gt_rows
            else None
        ),
        "coverage": (
            float(np.mean([row["gt_in_prediction_set"] for row in gt_rows]))
            if gt_rows
            else None
        ),
        "mean_prediction_set_size": (
            float(np.mean([row["prediction_set_size"] for row in gt_rows]))
            if gt_rows
            else None
        ),
        "median_prediction_set_size": (
            float(np.median([row["prediction_set_size"] for row in gt_rows]))
            if gt_rows
            else None
        ),
        "mean_nll": (
            float(
                np.mean(
                    [
                        -np.log(
                            max(
                                float(row["probabilities"][row["gt_candidate_id"]]),
                                1e-12,
                            )
                        )
                        for row in gt_rows
                    ]
                )
            )
            if gt_rows
            else None
        ),
        "pairwise_gt_preference": {
            candidate_id: {
                "eligible": len(outcomes),
                "fraction": float(np.mean(outcomes)),
            }
            for candidate_id, outcomes in sorted(pairwise_controls.items())
        },
        "elapsed_s": elapsed,
        "samples_per_s": len(records) / elapsed if elapsed > 0.0 else None,
        "peak_cuda_memory_mb": cuda_peak_memory_mb(extractor.torch),
        "geometry_backend": str(config.get("geometry", {}).get("backend", "plane")),
        "energy_metric": str(config["score"].get("energy_metric", "normalized_median_epe")),
        "invalid_records": invalid,
        "abstained_fraction": (
            float(np.mean([not row["valid"] for row in rows])) if rows else None
        ),
        "mean_observability_effective_static_fraction": (
            float(np.mean([
                item["effective_static_pixel_fraction"]
                for row in rows for item in row["observability"]
            ])) if rows else None
        ),
    }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
