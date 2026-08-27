#!/usr/bin/env python3
"""Evaluate candidate-blind continuous trajectory recovery on image clips."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.depth import load_cached_metric_depth, metric_depth_reliability_masks
from iac_new.dino_features import DINOv2TemporalConsistency
from iac_new.flow import RaftFlowExtractor
from iac_new.sea_raft_flow import SeaRaftFlowExtractor
from iac_new.perception import build_perception, temporal_road_consensus
from iac_new.protocol import read_jsonl, validate_record, write_jsonl
from iac_new.road_relative import road_relative_posterior
from iac_new.road_structure import build_road_structure, extract_road_boundaries, boundary_pixels_to_ego, fuse_ego_boundary_keypoints
from iac_new.scoring import interval_observability, polygon_mask
from iac_new.trajectory_decode import compare_continuous_trajectory, decode_continuous_trajectory
from iac_new.continuous_motion import history_only_motion_profile
from iac_new.temporal_geometry import (
    RoadStateFilter,
    TemporalScaleCalibrator,
    estimate_history_flow_scale,
    fit_causal_road_plane,
)
from iac_new.temporal_geometry import HomographyBoundaryPropagator
from iac_new.flow_reliability import FlowReliabilityFusion, calibrate_historical_flow_bias, calibrate_historical_row_bias


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_record(record: dict[str, Any], extractor: RaftFlowExtractor, config: dict[str, Any], perception: Any | None, dino: Any | None = None) -> dict[str, Any]:
    image_cfg = config["image"]
    width, height = int(image_cfg["width"]), int(image_cfg["height"])
    flow = extractor.observe(
        record["frame_paths"],
        record["intrinsics"],
        record["distortion"],
        (width, height),
        allow_mixed_source_sizes=bool(config.get("allow_mixed_source_sizes", False)),
        return_uncertainty=bool(config.get("flow", {}).get("refinement_uncertainty", False)),
        uncertainty_tail=int(config.get("flow", {}).get("uncertainty_tail", 8)),
        long_range_consistency=bool(config.get("flow", {}).get("long_range_consistency", {}).get("enabled", False)),
    )
    history_count = int(record["history_count"])
    future_start = history_count - 1
    observed = flow.forward[future_start:]
    flow_scale_cfg = config.get("flow_scale_correction", {})
    flow_scale = float(flow_scale_cfg.get("scale", 1.0)) if bool(flow_scale_cfg.get("enabled", False)) else 1.0
    if not np.isfinite(flow_scale) or flow_scale <= 0.0:
        raise ValueError("flow_scale_correction.scale must be positive and finite")
    if flow_scale != 1.0:
        observed = np.asarray(observed, dtype=np.float32) * flow_scale
    historical_bias = None
    bias_cfg = config.get("historical_flow_bias", {})
    if bool(bias_cfg.get("enabled", False)):
        history_state = record.get("metadata", {}).get("history_ego_state")
        if history_state is not None:
            historical_bias = calibrate_historical_flow_bias(
                full_flows=flow.forward,
                history_ego_state=np.asarray(history_state, dtype=np.float64),
                history_count=history_count,
                camera_to_ego=record["camera_to_ego"],
                intrinsics=flow.intrinsics,
                roi_mask=polygon_mask(height, width, config["mask"]["polygon_normalized"]),
                consistency_masks=flow.consistency_masks,
                tile_size=int(bias_cfg.get("tile_size", 32)),
                shrinkage=float(bias_cfg.get("shrinkage", 0.35)),
                max_correction_px=float(bias_cfg.get("max_correction_px", 1.5)),
            )
            if historical_bias.get("available"):
                observed = historical_bias["corrected_flows"][future_start:]
    historical_row_bias = None
    row_bias_cfg = config.get("historical_row_bias", {})
    if bool(row_bias_cfg.get("enabled", False)):
        history_state = record.get("metadata", {}).get("history_ego_state")
        if history_state is not None:
            historical_row_bias = calibrate_historical_row_bias(
                full_flows=flow.forward,
                history_ego_state=np.asarray(history_state, dtype=np.float64),
                history_count=history_count,
                camera_to_ego=record["camera_to_ego"],
                intrinsics=flow.intrinsics,
                roi_mask=polygon_mask(height, width, config["mask"]["polygon_normalized"]),
                consistency_masks=flow.consistency_masks,
                bands=int(row_bias_cfg.get("bands", 12)),
                shrinkage=float(row_bias_cfg.get("shrinkage", 0.15)),
                max_correction_px=float(row_bias_cfg.get("max_correction_px", 0.8)),
            )
            if historical_row_bias.get("available"):
                observed = historical_row_bias["corrected_flows"][future_start:]
    refinement_uncertainty = None if flow.refinement_uncertainty is None else flow.refinement_uncertainty[future_start:]
    consistency = None if flow.consistency_masks is None else flow.consistency_masks[future_start:]
    roi = polygon_mask(height, width, config["mask"]["polygon_normalized"])
    persistent_scale = None
    persistent_scale_cfg = config.get("persistent_scale_calibration", {})
    if bool(persistent_scale_cfg.get("enabled", False)):
        history_state = record.get("metadata", {}).get("history_ego_state")
        if history_state is not None and future_start >= 1:
            try:
                persistent_scale = estimate_history_flow_scale(
                    history_flows=flow.forward[:future_start],
                    history_ego_state=np.asarray(history_state, dtype=np.float64),
                    history_times_s=np.asarray(record.get("history_times_s", []), dtype=np.float64),
                    camera_to_ego=record["camera_to_ego"],
                    intrinsics=flow.intrinsics,
                    roi_mask=roi,
                    consistency_masks=(
                        None if flow.consistency_masks is None
                        else flow.consistency_masks[:future_start]
                    ),
                    min_flow_px=float(persistent_scale_cfg.get("min_flow_px", 0.5)),
                    max_points=int(persistent_scale_cfg.get("max_points", 1600)),
                    min_predicted_flow_px=float(persistent_scale_cfg.get("min_predicted_flow_px", 0.25)),
                    correction_shrinkage=float(persistent_scale_cfg.get("correction_shrinkage", 0.25)),
                )
                if persistent_scale.get("available") and bool(
                    persistent_scale_cfg.get("apply_correction", False)
                ):
                    observed = np.asarray(observed, dtype=np.float32) * float(
                        persistent_scale["future_flow_correction"]
                    )
                    persistent_scale["applied_to_decoder"] = True
                elif persistent_scale.get("available"):
                    persistent_scale["applied_to_decoder"] = False
            except (ValueError, np.linalg.LinAlgError) as error:
                persistent_scale = {"available": False, "error": str(error)}
    support_weights = np.ones(observed.shape[:-1], dtype=np.float32)
    uncertainty_stats = None
    if refinement_uncertainty is not None:
        finite_uncertainty = refinement_uncertainty[np.isfinite(refinement_uncertainty)]
        uncertainty_stats = {
            "median_px": float(np.median(finite_uncertainty)) if finite_uncertainty.size else None,
            "p95_px": float(np.quantile(finite_uncertainty, 0.95)) if finite_uncertainty.size else None,
            "high_uncertainty_fraction": float(np.mean(finite_uncertainty > 0.5)) if finite_uncertainty.size else None,
            "by_interval": [
                {
                    "median_px": float(np.median(value[np.isfinite(value)])) if np.isfinite(value).any() else None,
                    "p95_px": float(np.quantile(value[np.isfinite(value)], 0.95)) if np.isfinite(value).any() else None,
                    "valid_fraction": float(np.mean(np.isfinite(value))),
                }
                for value in refinement_uncertainty
            ],
        }
        uncertainty_cfg = config.get("flow", {}).get("refinement_uncertainty", {})
        if isinstance(uncertainty_cfg, dict) and bool(uncertainty_cfg.get("enabled", False)):
            scale = float(max(uncertainty_cfg.get("scale_px", 0.25), 1e-3))
            alpha = float(np.clip(uncertainty_cfg.get("blend_alpha", 0.35), 0.0, 1.0))
            uncertainty_weight = np.exp(-np.clip(refinement_uncertainty / scale, 0.0, 8.0)).astype(np.float32)
            uncertainty_weight = (1.0 - alpha) + alpha * uncertainty_weight
            support_weights *= uncertainty_weight
    if consistency is not None:
        support_weights *= consistency.astype(np.float32)
    long_range_residual = None if flow.long_range_residual is None else flow.long_range_residual[future_start:]
    long_range_cfg = config.get("flow", {}).get("long_range_consistency", {})
    long_range_stats = None
    if long_range_residual is not None:
        finite = np.isfinite(long_range_residual)
        values = long_range_residual[finite]
        long_range_stats = {
            "median_px": float(np.median(values)) if values.size else None,
            "p90_px": float(np.quantile(values, 0.90)) if values.size else None,
            "valid_fraction": float(np.mean(finite)),
        }
        # Keep this a soft reliability cue.  The final interval has no
        # two-step observation and is assigned neutral weight.
        alpha = float(np.clip(long_range_cfg.get("decoder_blend_alpha", 0.0), 0.0, 1.0))
        sigma = float(max(long_range_cfg.get("residual_sigma_px", 2.0), 1e-3))
        cue = np.ones_like(long_range_residual, dtype=np.float32)
        valid = np.isfinite(long_range_residual)
        cue[valid] = np.exp(-np.clip(long_range_residual[valid] / sigma, 0.0, 8.0)).astype(np.float32)
        if alpha > 0.0:
            support_weights *= (1.0 - alpha) + alpha * cue
    geometric_support_weights = support_weights.copy()
    dino_weights = None
    if dino is not None:
        dino_weights = dino.observe(
            record["frame_paths"][future_start:], observed,
            target_size=(width, height),
        )
        dino_alpha = float(np.clip(config.get("dino", {}).get("blend_alpha", 1.0), 0.0, 1.0))
        dino_blend = (1.0 - dino_alpha) + dino_alpha * dino_weights
        support_weights *= dino_blend
        geometric_support_weights *= dino_blend

    perception_observation = None
    raw_perception_observation = None
    far_mask_disagreement = None
    road_masks = None
    road_oracle_used = False
    perception_cfg = config.get("perception", {})
    if perception is not None:
        all_frames = record["frame_paths"] if bool(perception_cfg.get("temporal_consensus", False)) else record["frame_paths"][:-1]
        perception_observation = perception.observe(all_frames, target_size=(width, height), intrinsics=record["intrinsics"], distortion=record["distortion"])
        raw_perception_observation = perception_observation
        if bool(perception_cfg.get("temporal_consensus", False)):
            perception_observation = temporal_road_consensus(
                perception_observation,
                flow.forward,
                road_dilation_px=int(perception_cfg.get("road_dilation_px", 4)),
                actor_dilation_px=int(perception_cfg.get("actor_dilation_px", 3)),
            )
        future_perception = perception_observation.future(future_start)
        road = np.asarray(future_perception.traversable_masks, dtype=bool)
        road_masks = road.astype(np.float32)
        actors = np.asarray(future_perception.actor_masks, dtype=bool)
        support_weights *= np.where(road, float(perception_cfg.get("road_weight", 1.0)), float(perception_cfg.get("offroad_weight", 0.35))).astype(np.float32)
        support_weights = np.where(actors, support_weights * float(perception_cfg.get("actor_weight", 0.15)), support_weights)

    # Optional LiDAR road oracle used only for IAC upper-bound ablations. It is
    # never populated from WAM future images and therefore cannot leak the
    # realized state into the image probe.
    oracle_path = record.get("metadata", {}).get("road_oracle_mask_path")
    if oracle_path is not None and bool(config.get("road_oracle_override", True)):
        oracle_mask = np.asarray(np.load(str(oracle_path)), dtype=np.float32)
        if oracle_mask.shape != (height, width):
            raise ValueError("road oracle mask does not match configured image size")
        road_masks = np.repeat(oracle_mask[None, ...], observed.shape[0], axis=0)
        road_oracle_used = True

    depths = None
    depth_reliability = None
    geometry_cfg = config.get("geometry", {})
    if str(geometry_cfg.get("backend", "plane")) == "cached_metric_depth":
        metric_depth = load_cached_metric_depth(
            record,
            geometry_cfg,
            expected_intervals=len(observed),
            expected_size=(width, height),
            expected_intrinsics=flow.intrinsics,
        )
        depths = metric_depth.depths_m
        depth_reliability, _ = metric_depth_reliability_masks(
            metric_depth,
            observed,
            min_depth_m=float(geometry_cfg.get("min_depth_m", 1.0)),
            max_depth_m=float(geometry_cfg.get("max_depth_m", 100.0)),
            confidence_quantile=float(geometry_cfg.get("confidence_quantile", 0.25)),
            observed_flow_quantile=float(geometry_cfg.get("observed_flow_quantile", 0.997)),
        )
        support_weights *= depth_reliability.astype(np.float32)

    # Reliability fusion is deliberately separate from candidate scoring.  It
    # combines forward/backward consistency with image-warp residuals and
    # optional depth validity, then applies the same soft support to both the
    # decoder and road-structure diagnostics.
    flow_reliability = None
    reliability_cfg = config.get("flow_reliability", {})
    if bool(reliability_cfg.get("enabled", False)):
        flow_reliability = FlowReliabilityFusion(
            tile_size=int(reliability_cfg.get("tile_size", 16)),
            photometric_sigma=float(reliability_cfg.get("photometric_sigma", 0.08)),
            photometric_floor=float(reliability_cfg.get("photometric_floor", 0.25)),
            tile_floor=float(reliability_cfg.get("tile_floor", 0.50)),
            weight_floor=float(reliability_cfg.get("weight_floor", 0.05)),
            min_valid_fraction=float(reliability_cfg.get("min_valid_fraction", 0.05)),
            repair_enabled=bool(reliability_cfg.get("repair_enabled", False)),
            repair_threshold_px=float(reliability_cfg.get("repair_threshold_px", 2.0)),
        ).estimate(
            observed_flows=observed,
            frame_paths=record["frame_paths"][future_start : future_start + len(observed) + 1],
            intrinsics=record["intrinsics"],
            distortion=record["distortion"],
            consistency_masks=consistency,
            depths_m=depths,
        )
        reliability_weights = np.asarray(flow_reliability["weights"], dtype=np.float32)
        # Keep the diagnostic cue independent from the scorer by default.  A
        # conservative blend can be enabled for an A/B; full multiplication
        # is intentionally not the default because photometric shadows can
        # erase otherwise useful rigid evidence.
        blend_alpha = float(np.clip(reliability_cfg.get("decoder_blend_alpha", 0.0), 0.0, 1.0))
        if blend_alpha > 0.0:
            blended = (1.0 - blend_alpha) + blend_alpha * reliability_weights
            support_weights *= blended
            geometric_support_weights *= blended
        if bool(reliability_cfg.get("use_repaired_flow", False)):
            observed = np.asarray(flow_reliability["repaired_flows"], dtype=np.float32)

    decoder_cfg = config.get("continuous_decoder", {})
    adaptive_plane = None
    adaptive_cfg = config.get("adaptive_road_plane", {})
    if bool(adaptive_cfg.get("enabled", False)) and depths is None:
        history_state = record.get("metadata", {}).get("history_ego_state")
        if history_state is not None and history_count >= 3:
            history_flow = flow.forward[: history_count - 1]
            history_consistency = None if flow.consistency_masks is None else flow.consistency_masks[: history_count - 1]
            history_road = None
            if perception_observation is not None:
                history_road = np.asarray(perception_observation.traversable_masks[: history_count - 1], dtype=bool)
                history_road &= ~np.asarray(perception_observation.actor_masks[: history_count - 1], dtype=bool)
            try:
                adaptive_plane = fit_causal_road_plane(
                    observed_flows=history_flow,
                    history_ego_state=np.asarray(history_state, dtype=np.float64),
                    camera_to_ego=record["camera_to_ego"],
                    intrinsics=flow.intrinsics,
                    roi_mask=roi,
                    consistency_masks=history_consistency,
                    road_masks=history_road,
                    fit_intervals=int(adaptive_cfg.get("fit_intervals", 2)),
                    max_points=int(adaptive_cfg.get("max_points", 1200)),
                    residual_scale_px=float(adaptive_cfg.get("residual_scale_px", 2.0)),
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                adaptive_plane = {"available": False, "error": str(error)}
    adaptive_params = None
    if adaptive_plane is not None and bool(adaptive_plane.get("available", False)):
        shrinkage = float(np.clip(adaptive_cfg.get("shrinkage", 1.0), 0.0, 1.0))
        adaptive_params = np.asarray(adaptive_plane["plane_params"], dtype=np.float64) * shrinkage
        adaptive_plane["applied_shrinkage"] = shrinkage
    road_masks_for_structure = road_masks
    temporal_geometry_cfg = config.get("temporal_geometry", {})
    road_state_mask_mode = str(temporal_geometry_cfg.get("road_state_mask_mode", "consensus"))
    if (
        road_masks_for_structure is not None
        and raw_perception_observation is not None
        and road_state_mask_mode == "consensus_near_raw_far"
    ):
        raw_future = raw_perception_observation.future(future_start)
        raw_road = np.asarray(raw_future.traversable_masks, dtype=np.float32)
        # Raw frame masks have one more element than interval-wise temporal
        # consensus masks.  Road state k is anchored at the current frame of
        # interval k, so drop only the terminal frame for exact alignment.
        if len(raw_road) == len(road_masks_for_structure) + 1:
            raw_road = raw_road[:-1]
        if raw_road.shape == road_masks_for_structure.shape:
            far_cut = int(height * float(temporal_geometry_cfg.get("far_start_fraction", 0.62)))
            consensus_road = np.asarray(road_masks_for_structure, dtype=bool)
            far_mask_disagreement = np.asarray([
                float(np.mean(np.logical_xor(raw_road[index, :far_cut], consensus_road[index, :far_cut])))
                for index in range(len(consensus_road))
            ], dtype=np.float64)
            road_masks_for_structure = np.asarray(road_masks_for_structure, dtype=np.float32).copy()
            road_masks_for_structure[:, :far_cut] = raw_road[:, :far_cut]
    if not bool(decoder_cfg.get("road_prior_enabled", False)):
        # Road evidence is still exported even when it is not fed back as an
        # optimizer penalty.  This keeps ablations from hiding observability.
        road_masks = None
    interval_quality = interval_observability(
        observed_flows=observed,
        roi_mask=roi,
        consistency_masks=consistency,
        dynamic_weights=support_weights,
        minimum_flow_scale_px=float(config["score"].get("minimum_flow_scale_px", 1.0)),
        static_weight_threshold=float(decoder_cfg.get("static_weight_threshold", 0.5)),
        min_effective_pixel_fraction=float(decoder_cfg.get("min_effective_pixel_fraction", 0.02)),
        role="future",
        curvature_min_support_fraction=float(config.get("observability", {}).get("curvature_min_support_fraction", 0.05)),
        curvature_min_lateral_contrast_rad=float(config.get("observability", {}).get("curvature_min_lateral_contrast_rad", 0.02)),
        curvature_min_flow_gradient_px=float(config.get("observability", {}).get("curvature_min_flow_gradient_px", 0.02)),
        curvature_reliable_lateral_contrast_rad=float(config.get("observability", {}).get("curvature_reliable_lateral_contrast_rad", 1.43)),
    )
    quality_vector = np.asarray([item["effective_static_pixel_fraction"] for item in interval_quality], dtype=np.float64)
    history_speed_prior = None
    history_speed_curve = None
    history_state = record.get("metadata", {}).get("history_ego_state")
    if bool(decoder_cfg.get("use_history_speed_prior", True)) and history_state is not None:
        history_array = np.asarray(history_state, dtype=np.float64)
        if history_array.ndim == 2 and history_array.shape[0] > 0 and history_array.shape[1] >= 4 and np.isfinite(history_array[-1, 3]):
            history_speed_prior = float(max(history_array[-1, 3], 0.2))
            if bool(decoder_cfg.get("history_anchored_speed_residual", False)):
                history_profile = history_only_motion_profile(
                    history_array,
                    record["future_times_s"],
                    history_times_s=record["history_times_s"],
                    model="constant_acceleration_yaw_rate",
                )
                history_speed_curve = np.asarray(
                    [row["speed_mps"] for row in history_profile["rows"]], dtype=np.float64
                )
    initial_speeds = tuple(float(v) for v in decoder_cfg.get("initial_speeds_mps", [3.0, 6.0, 10.0]))
    if history_speed_prior is not None:
        spread = float(decoder_cfg.get("history_speed_prior_spread", 0.35))
        initial_speeds = tuple(sorted({max(0.2, history_speed_prior * (1.0 - spread)), history_speed_prior, history_speed_prior * (1.0 + spread)}))
    decoder_kwargs = dict(
        observed_flows=observed,
        camera_to_ego=record["camera_to_ego"],
        intrinsics=flow.intrinsics,
        future_times_s=np.asarray(record["future_times_s"], dtype=np.float64),
        roi_mask=roi,
        consistency_masks=consistency,
        dynamic_weights=support_weights,
        depths_m=depths,
        image_size=(width, height),
        minimum_flow_scale_px=float(config["score"].get("minimum_flow_scale_px", 1.0)),
        max_points=int(decoder_cfg.get("max_points", 900)),
        max_iterations=int(decoder_cfg.get("max_iterations", 12)),
        initial_speeds_mps=initial_speeds,
        history_speeds_mps=history_speed_curve,
        history_initial_speed_mps=(history_speed_prior if history_speed_curve is not None else None),
        maximum_speed_residual_mps=float(decoder_cfg.get("maximum_speed_residual_mps", 3.0)),
        speed_residual_weight=float(decoder_cfg.get("speed_residual_weight", 0.02)),
        speed_residual_smoothness_weight=float(
            decoder_cfg.get("speed_residual_smoothness_weight", 0.05)
        ),
        speed_residual_curvature_weight=float(
            decoder_cfg.get("speed_residual_curvature_weight", 0.0)
        ),
        profile_radius=float(decoder_cfg.get("profile_radius", 0.12)),
        interval_observability=quality_vector,
        speed_uncertainty_thresholds=(
            float(decoder_cfg.get("speed_abstain_observability", 0.25)),
            float(decoder_cfg.get("speed_uncertain_observability", 0.55)),
        ),
        curvature_multistart=bool(decoder_cfg.get("curvature_multistart", False)),
        road_masks=road_masks,
        road_prior_weight=float(decoder_cfg.get("road_prior_weight", 0.0)),
        road_half_width_m=float(decoder_cfg.get("road_half_width_m", 1.1)),
        road_lateral_samples=int(decoder_cfg.get("road_lateral_samples", 5)),
        road_longitudinal_step_m=float(decoder_cfg.get("road_longitudinal_step_m", 0.5)),
        speed_smoothness_weight=float(decoder_cfg.get("speed_smoothness_weight", 0.0)),
        curvature_smoothness_weight=float(decoder_cfg.get("curvature_smoothness_weight", 0.0)),
        lateral_acceleration_weight=float(decoder_cfg.get("lateral_acceleration_weight", 0.0)),
        adaptive_plane_params=adaptive_params,
    )
    geometric_decoded = None
    if perception is not None and bool(perception_cfg.get("shape_only", False)):
        geometric_decoded = decode_continuous_trajectory(
            **{**decoder_kwargs, "dynamic_weights": geometric_support_weights}
        )
        geometric_trajectory = np.asarray(geometric_decoded["trajectory"], dtype=np.float64)
        times = np.asarray(record["future_times_s"], dtype=np.float64)
        deltas = np.diff(np.vstack([np.zeros((1, 2)), geometric_trajectory[:, :2]]), axis=0)
        fixed_speeds = np.linalg.norm(deltas, axis=1) / np.diff(np.concatenate([[0.0], times]))
        distances = np.maximum(np.linalg.norm(deltas, axis=1), 1e-3)
        initial_curvatures = np.diff(np.concatenate([[0.0], geometric_trajectory[:, 2]])) / distances
        decoder_kwargs.update(
            fixed_speeds_mps=fixed_speeds,
            initial_curvatures_1pm=initial_curvatures,
        )
    decoded = decode_continuous_trajectory(**decoder_kwargs)
    road_structure = None
    temporal_road_state = None
    ego_frame_boundary_fusion = None
    boundary_propagation = None
    temporal_scale = None
    if road_masks_for_structure is not None:
        try:
            road_structure = build_road_structure(
                road_masks_for_structure,
                observed,
                # Keep road weighting out of static-flow evidence.  A road
                # segmentation miss should not erase otherwise useful rigid
                # tracks; dynamic/FB consistency is already reflected in the
                # geometric support weights.
                geometric_support_weights,
                roi,
                near_start=float(perception_cfg.get("near_start", 0.55)),
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            road_structure = {"valid": False, "error": str(error)}
        temporal_cfg = temporal_geometry_cfg
        if bool(temporal_cfg.get("enabled", False)):
            try:
                ego_states = record.get("metadata", {}).get("history_ego_state")
                boundary_cfg = temporal_cfg.get("boundary_propagation", {})
                if bool(boundary_cfg.get("enabled", False)):
                    boundary_propagation = HomographyBoundaryPropagator(
                        keypoints_per_side=int(boundary_cfg.get("keypoints_per_side", 10)),
                        process_noise=float(boundary_cfg.get("process_noise", 0.025)),
                        propagation_decay=float(boundary_cfg.get("propagation_decay", 0.92)),
                        max_reprojection_residual_px=float(boundary_cfg.get("max_reprojection_residual_px", 32.0)),
                        road_half_width_m=float(decoder_cfg.get("road_half_width_m", 1.1)),
                        min_current_confidence_for_propagation=float(boundary_cfg.get("min_current_confidence_for_propagation", 0.45)),
                        min_far_support_fraction=float(boundary_cfg.get("min_far_support_fraction", 0.01)),
                        propagation_method=str(boundary_cfg.get("propagation_method", "homography")),
                        propagate_far_missing=bool(boundary_cfg.get("propagate_far_missing", True)),
                        global_flow_max_points=int(boundary_cfg.get("global_flow_max_points", 5000)),
                        global_flow_ransac_threshold_px=float(boundary_cfg.get("global_flow_ransac_threshold_px", 2.5)),
                    ).update(
                        road_masks_for_structure,
                        intrinsics=flow.intrinsics,
                        camera_to_ego=record["camera_to_ego"],
                        observed_flows=observed,
                        static_weights=geometric_support_weights,
                        history_ego_state=None if ego_states is None else np.asarray(ego_states, dtype=np.float64),
                        history_times_s=np.asarray(record.get("history_times_s", []), dtype=np.float64),
                        future_times_s=np.asarray(record["future_times_s"], dtype=np.float64),
                    )
                # The established recursive filter remains the main road-state
                # estimate.  Homography propagation is exported separately and
                # can only be promoted after an independent holdout gate.
                temporal_road_state = RoadStateFilter(
                    measurement_gain=float(temporal_cfg.get("measurement_gain", 0.65)),
                    process_noise=float(temporal_cfg.get("process_noise", 0.02)),
                        far_uncertainty_growth=float(temporal_cfg.get("far_uncertainty_growth", 0.04)),
                        road_half_width_m=float(decoder_cfg.get("road_half_width_m", 1.1)),
                        far_support_reference_fraction=float(temporal_cfg.get("far_support_reference_fraction", 0.03)),
                        far_disagreement_weight=float(temporal_cfg.get("far_disagreement_weight", 0.0)),
                        far_start_fraction=float(temporal_cfg.get("far_start_fraction", 0.62)),
                        boundary_keypoint_filter_enabled=bool(temporal_cfg.get("boundary_keypoint_filter", {}).get("enabled", False)),
                        boundary_keypoint_max_jump_px=float(temporal_cfg.get("boundary_keypoint_filter", {}).get("max_jump_px", 28.0)),
                        boundary_keypoint_huber_scale_px=float(temporal_cfg.get("boundary_keypoint_filter", {}).get("huber_scale_px", 8.0)),
                ).update(
                    road_masks_for_structure,
                    observed_flows=observed,
                    ego_states=None if ego_states is None else np.asarray(ego_states, dtype=np.float64),
                    future_times_s=np.asarray(record["future_times_s"], dtype=np.float64),
                    far_disagreement=far_mask_disagreement,
                )
                ego_cfg = temporal_cfg.get("ego_frame_boundary_fusion", {})
                if bool(ego_cfg.get("enabled", False)):
                    history_state = record.get("metadata", {}).get("history_ego_state")
                    image_boundaries = [extract_road_boundaries(np.asarray(mask, dtype=bool), row_step=4, polynomial_degree=2) for mask in road_masks_for_structure]
                    ego_boundaries = [boundary_pixels_to_ego(item, flow.intrinsics, record["camera_to_ego"]) for item in image_boundaries]
                    predicted = HomographyBoundaryPropagator._future_poses(
                        None if history_state is None else np.asarray(history_state, dtype=np.float64),
                        np.asarray(record.get("history_times_s", []), dtype=np.float64),
                        np.asarray(record["future_times_s"], dtype=np.float64),
                    )
                    transforms = [np.linalg.inv(pose) for pose in predicted[1:1 + len(ego_boundaries)]]
                    fused, fusion_diag = fuse_ego_boundary_keypoints(
                        ego_boundaries,
                        transforms,
                        max_lateral_jump_m=float(ego_cfg.get("max_lateral_jump_m", 1.2)),
                        huber_scale_m=float(ego_cfg.get("huber_scale_m", 0.35)),
                        width_shrink=float(ego_cfg.get("width_shrink", 1.0)),
                    )
                    ego_frame_boundary_fusion = {"protocol": "ego-frame-boundary-fusion-v1", "image_boundaries": image_boundaries, "raw_ego_boundaries": ego_boundaries, "ego_boundaries": fused, "diagnostics": fusion_diag}
            except (ValueError, np.linalg.LinAlgError) as error:
                temporal_road_state = {"available": False, "error": str(error)}
    temporal_cfg = config.get("temporal_geometry", {})
    # Keep speed/progress diagnostics available when a state-aware manifest
    # has an ego-speed prior but no metric-depth cache.  The calibrator then
    # emits a wide, low-confidence interval instead of silently dropping the
    # field; speed remains excluded from the primary score.
    if bool(temporal_cfg.get("enabled", False)):
        try:
            history_state = record.get("metadata", {}).get("history_ego_state")
            temporal_scale = TemporalScaleCalibrator(
                min_valid_fraction=float(temporal_cfg.get("min_valid_fraction", 0.02)),
                min_depth_m=float(geometry_cfg.get("min_depth_m", 1.0)),
                max_depth_m=float(geometry_cfg.get("max_depth_m", 100.0)),
                min_flow_px=float(temporal_cfg.get("min_flow_px", 0.5)),
                uncertainty_floor=float(temporal_cfg.get("speed_uncertainty_floor", 0.15)),
            ).estimate(
                observed_flows=observed,
                depths_m=depths,
                static_weights=support_weights,
                intrinsics=flow.intrinsics,
                future_times_s=np.asarray(record["future_times_s"], dtype=np.float64),
                history_ego_state=None if history_state is None else np.asarray(history_state, dtype=np.float64),
                history_times_s=np.asarray(record.get("history_times_s", []), dtype=np.float64),
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            temporal_scale = {"available": False, "error": str(error)}
    uncertainty_support_cfg = config.get("uncertainty_support", {})
    lateral_uncertainty_inflation = heading_uncertainty_inflation = curvature_uncertainty_inflation = None
    if bool(uncertainty_support_cfg.get("enabled", False)) and refinement_uncertainty is not None:
        scale_px = float(max(uncertainty_support_cfg.get("scale_px", 0.25), 1e-3))
        interval_u = np.asarray([
            float(np.median(value[np.isfinite(value)])) if np.isfinite(value).any() else scale_px
            for value in refinement_uncertainty
        ], dtype=np.float64)
        normalized_u = np.clip(interval_u / scale_px, 0.0, float(uncertainty_support_cfg.get("max_normalized", 4.0)))
        lateral_uncertainty_inflation = normalized_u * float(uncertainty_support_cfg.get("lateral_factor_m", 0.05))
        heading_uncertainty_inflation = normalized_u * float(uncertainty_support_cfg.get("heading_factor_rad", 0.01))
        curvature_uncertainty_inflation = normalized_u * float(uncertainty_support_cfg.get("curvature_factor_1pm", 0.005))
    # Boundary uncertainty is a separate failure mode from flow refinement
    # uncertainty.  Use it only as a conditional support inflation: the point
    # estimate and all normal-observability samples remain unchanged, while a
    # missing far boundary cannot leave an artificially narrow lateral tube.
    road_state_support_cfg = config.get("road_state_support", {})
    road_state_lateral_inflation = None
    if bool(road_state_support_cfg.get("enabled", False)) and temporal_road_state is not None:
        states = temporal_road_state.get("states", [])
        if len(states) == len(decoded.get("trajectory", [])):
            uncertainty = np.asarray([
                float(item.get("lateral_uncertainty_m", 0.0) or 0.0) for item in states
            ], dtype=np.float64)
            far_observability = np.asarray([
                float(item.get("far_range_observability", 0.0) or 0.0) for item in states
            ], dtype=np.float64)
            threshold = float(np.clip(road_state_support_cfg.get("observability_threshold", 0.65), 0.0, 1.0))
            floor_m = float(max(road_state_support_cfg.get("uncertainty_floor_m", 0.06), 0.0))
            factor = float(max(road_state_support_cfg.get("lateral_factor", 1.0), 0.0))
            maximum = float(max(road_state_support_cfg.get("max_lateral_inflation_m", 0.50), 0.0))
            conditional = np.maximum(uncertainty - floor_m, 0.0) * factor
            conditional = np.where(far_observability < threshold, conditional, 0.0)
            road_state_lateral_inflation = np.clip(conditional, 0.0, maximum)
            if lateral_uncertainty_inflation is None:
                lateral_uncertainty_inflation = road_state_lateral_inflation
            else:
                lateral_uncertainty_inflation = lateral_uncertainty_inflation + road_state_lateral_inflation
    road_posterior = road_relative_posterior(
        np.asarray(decoded["trajectory"], dtype=np.float64),
        np.asarray(record["future_times_s"], dtype=np.float64),
        profile_support=decoded.get("profile_support"),
        observability=quality_vector,
        lateral_inflation_m=float(decoder_cfg.get("support_lateral_inflation_m", 0.0)),
        heading_inflation_rad=float(decoder_cfg.get("support_heading_inflation_rad", 0.0)),
        curvature_inflation_1pm=float(decoder_cfg.get("support_curvature_inflation_1pm", 0.0)),
        lateral_inflation_by_interval=lateral_uncertainty_inflation,
        heading_inflation_by_interval=heading_uncertainty_inflation,
        curvature_inflation_by_interval=curvature_uncertainty_inflation,
    )
    if temporal_road_state is not None:
        road_posterior["temporal_road_state"] = temporal_road_state
        road_posterior["road_boundary_uncertainty"] = [
            item.get("lateral_uncertainty_m") for item in temporal_road_state.get("states", [])
        ]
        road_posterior["far_range_observability"] = [
            item.get("far_range_observability") for item in temporal_road_state.get("states", [])
        ]
        road_posterior["temporal_jitter_diagnostics"] = temporal_road_state.get("temporal_jitter_diagnostics")
        road_posterior["road_state_support"] = {
            "enabled": bool(road_state_support_cfg.get("enabled", False)),
            "lateral_inflation_m": None if road_state_lateral_inflation is None else road_state_lateral_inflation.tolist(),
            "observability_threshold": float(road_state_support_cfg.get("observability_threshold", 0.65)),
        }
    if temporal_scale is not None:
        road_posterior["scale_calibration"] = temporal_scale
        road_posterior["speed_interval_mps"] = [row.get("speed_interval_mps") for row in temporal_scale.get("rows", [])]
        road_posterior["progress_interval_m"] = [row.get("progress_interval_m") for row in temporal_scale.get("rows", [])]
    gt_id = record.get("gt_candidate_id")
    reference = next((candidate["trajectory"] for candidate in record["candidates"] if str(candidate["candidate_id"]) == str(gt_id)), None)
    comparison = None if reference is None else compare_continuous_trajectory(
        np.asarray(decoded["trajectory"], dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        np.asarray(record["future_times_s"], dtype=np.float64),
        observability=np.asarray([
            float(np.mean(support_weights[index] > 0.0)) for index in range(len(observed))
        ]),
        lateral_tolerance_m=float(decoder_cfg.get("lateral_tolerance_m", 0.50)),
        yaw_tolerance_rad=float(decoder_cfg.get("yaw_tolerance_rad", 0.10)),
        speed_relative_tolerance=float(decoder_cfg.get("speed_relative_tolerance", 0.20)),
        curvature_tolerance_1pm=float(decoder_cfg.get("curvature_tolerance_1pm", 0.06)),
        score_speed=bool(decoder_cfg.get("score_speed", False)),
    )
    return {
        "sample_id": record["sample_id"],
        "scene_id": record["scene_id"],
        "decoder": decoded,
        "geometric_decoder": geometric_decoded,
        "comparison_to_logged_trajectory": comparison,
        "observability_by_future_interval": interval_quality,
        "road_structure": road_structure,
        "temporal_road_state": temporal_road_state,
        "ego_frame_boundary_fusion": ego_frame_boundary_fusion,
        "boundary_propagation": boundary_propagation,
        "temporal_scale_calibration": temporal_scale,
        "persistent_scale_calibration": persistent_scale,
        "flow_reliability": None if flow_reliability is None else {
            key: value for key, value in flow_reliability.items() if key not in {"weights", "photometric_weights", "tile_weights", "repaired_flows"}
        } | {"decoder_blend_alpha": float(np.clip(reliability_cfg.get("decoder_blend_alpha", 0.0), 0.0, 1.0))},
        "long_range_flow_consistency": None if long_range_stats is None else long_range_stats | {
            "decoder_blend_alpha": float(np.clip(long_range_cfg.get("decoder_blend_alpha", 0.0), 0.0, 1.0)),
            "residual_sigma_px": float(max(long_range_cfg.get("residual_sigma_px", 2.0), 1e-3)),
        },
        "raft_refinement_uncertainty": None if uncertainty_stats is None else uncertainty_stats | {
            "enabled_for_decoder": bool(config.get("flow", {}).get("refinement_uncertainty", {}).get("enabled", False)) if isinstance(config.get("flow", {}).get("refinement_uncertainty", {}), dict) else False,
        },
        "historical_flow_bias": None if historical_bias is None else {
            key: value for key, value in historical_bias.items() if key != "corrected_flows"
        },
        "historical_row_bias": None if historical_row_bias is None else {
            key: value for key, value in historical_row_bias.items() if key != "corrected_flows"
        },
        "adaptive_road_plane": adaptive_plane,
        "road_relative_posterior": road_posterior,
        "history_speed_prior_mps": history_speed_prior,
        "road_oracle_used": road_oracle_used,
        "dino_temporal_consistency": None if dino_weights is None else {
            "mean": float(np.mean(dino_weights)),
            "median": float(np.median(dino_weights)),
            "low_weight_fraction": float(np.mean(dino_weights < 0.5)),
        },
        "candidate_bank_used_by_decoder": False,
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = _json(args.config)
    raw_rows = read_jsonl(args.manifest)
    if args.max_samples is not None:
        raw_rows = raw_rows[: args.max_samples]
    records = [validate_record(row, manifest_root=args.manifest.parent) for row in raw_rows]
    flow_cfg = config["flow"]
    if str(flow_cfg.get("backend", "torchvision_raft")) == "sea_raft":
        extractor = SeaRaftFlowExtractor(
            checkpoint=str(flow_cfg["checkpoint"]),
            device=args.device,
            iters=int(flow_cfg.get("iters", 12)),
        )
    else:
        extractor = RaftFlowExtractor(
            model_size=str(flow_cfg["model"]), device=args.device,
            updates=int(flow_cfg["updates"]), batch_size=int(flow_cfg["batch_size"]),
            forward_backward=bool(flow_cfg["forward_backward"]),
            fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
            fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
        )
    perception = build_perception(config, device=args.device)
    dino = None
    dino_cfg = config.get("dino", {})
    if bool(dino_cfg.get("enabled", False)):
        dino = DINOv2TemporalConsistency(
            device=args.device,
            model_name=str(dino_cfg.get("model_name", "dinov2_vits14")),
            hub_dir=dino_cfg.get("hub_dir"),
            weight_floor=float(dino_cfg.get("weight_floor", 0.25)),
        )
    rows = []
    errors = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        try:
            rows.append(evaluate_record(record, extractor, config, perception, dino))
        except Exception as error:
            errors.append({"sample_id": record["sample_id"], "error": str(error)})
        print(json.dumps({"completed": index, "total": len(records)}), flush=True)
    write_jsonl(args.output, rows)
    comparisons = [row["comparison_to_logged_trajectory"] for row in rows if row.get("comparison_to_logged_trajectory")]
    summary = {
        "protocol": "candidate-blind-continuous-trajectory-v1",
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()),
        "num_input": len(records),
        "num_scored": len(rows),
        "num_error": len(errors),
        "mean_weighted_joint_error": float(np.mean([row["weighted_mean_joint_error"] for row in comparisons])) if comparisons else None,
        "median_joint_error": float(np.median([row["median_joint_error"] for row in comparisons])) if comparisons else None,
        "mean_soft_compatibility": float(np.mean([row["soft_compatibility"] for row in comparisons])) if comparisons else None,
        "mean_joint_coverage": float(np.mean([row["joint_coverage"] for row in comparisons])) if comparisons else None,
        "mean_heading_cosine": float(np.mean([row["mean_heading_cosine"] for row in comparisons])) if comparisons else None,
        "mean_lateral_abs_m": float(np.mean([row["mean_lateral_abs_m"] for row in comparisons])) if comparisons else None,
        "mean_yaw_abs_rad": float(np.mean([row["mean_yaw_abs_rad"] for row in comparisons])) if comparisons else None,
        "mean_speed_relative_error": float(np.mean([row["mean_speed_relative_error"] for row in comparisons])) if comparisons else None,
        "mean_curvature_abs_1pm": float(np.mean([row["mean_curvature_abs_1pm"] for row in comparisons])) if comparisons else None,
        "speed_scored_in_primary_metric": bool(config.get("continuous_decoder", {}).get("score_speed", False)),
        "speed_abstain_fraction": (
            float(np.mean([
                sum(item["speed_status"] == "abstain" for item in row["observability_by_future_interval"])
                / max(len(row["observability_by_future_interval"]), 1)
                for row in rows
            ])) if rows else None
        ),
        "flow_reliability": {
            "enabled": bool(config.get("flow_reliability", {}).get("enabled", False)),
            "mean_effective_fraction": float(np.mean([
                (row.get("flow_reliability") or {}).get("valid_fraction", 0.0) for row in rows
            ])) if rows else None,
            "mean_photometric_residual": float(np.mean([
                value for row in rows for value in (row.get("flow_reliability") or {}).get("median_photometric_residual", []) if value is not None
            ])) if any(value is not None for row in rows for value in (row.get("flow_reliability") or {}).get("median_photometric_residual", [])) else None,
            "mean_repaired_fraction": float(np.mean([
                value for row in rows for value in (row.get("flow_reliability") or {}).get("repaired_fraction", [])
            ])) if any((row.get("flow_reliability") or {}).get("repaired_fraction", []) for row in rows) else None,
        },
        "boundary_propagation": {
            "enabled": bool(config.get("temporal_geometry", {}).get("boundary_propagation", {}).get("enabled", False)),
            "homography_use_fraction": float(np.mean([
                bool(item.get("homography_used"))
                for row in rows for item in (row.get("boundary_propagation") or {}).get("propagation", [])
            ])) if any((row.get("boundary_propagation") or {}).get("propagation", []) for row in rows) else None,
            "applied_fraction": float(np.mean([
                bool(item.get("propagation_applied"))
                for row in rows for item in (row.get("boundary_propagation") or {}).get("propagation", [])
            ])) if any((row.get("boundary_propagation") or {}).get("propagation", []) for row in rows) else None,
        },
        "elapsed_s": time.perf_counter() - started,
        "errors": errors,
    }
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
