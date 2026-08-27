"""Candidate-blind continuous ego-trajectory recovery from image flow.

The decoder fits a small continuous motion family directly to observed flow.
It never receives the benchmark candidate bank, so its output cannot be a
finite-bank retrieval result. The output is a point estimate plus a local
profile support tube that reflects image evidence and observability.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import (
    adjacent_camera_transforms,
    candidate_camera_poses,
    ground_plane_homography,
    se2_to_transform,
)
from .region import trajectory_states


def integrate_piecewise_controls(
    future_times_s: np.ndarray,
    *,
    speeds_mps: np.ndarray,
    curvatures_1pm: np.ndarray,
) -> np.ndarray:
    """Integrate one speed/curvature control per future interval into [T,3]."""
    times = np.asarray(future_times_s, dtype=np.float64)
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    curvatures = np.asarray(curvatures_1pm, dtype=np.float64)
    if times.ndim != 1 or len(times) < 1 or np.any(np.diff(times) <= 0.0):
        raise ValueError("future_times_s must be a strictly increasing vector")
    if speeds.shape != times.shape or curvatures.shape != times.shape:
        raise ValueError("piecewise controls must match future_times_s")
    if not np.all(np.isfinite(np.concatenate([times, speeds, curvatures]))):
        raise ValueError("future times and controls must be finite")
    if np.any(speeds < 0.0):
        raise ValueError("speeds must be non-negative")
    trajectory = np.zeros((len(times), 3), dtype=np.float64)
    previous_time = 0.0
    for index, time_s in enumerate(times):
        dt = float(time_s - previous_time)
        previous = trajectory[index - 1] if index else np.zeros(3, dtype=np.float64)
        speed = float(speeds[index])
        curvature = float(curvatures[index])
        yaw_mid = float(previous[2] + 0.5 * curvature * speed * dt)
        trajectory[index] = [
            previous[0] + speed * np.cos(yaw_mid) * dt,
            previous[1] + speed * np.sin(yaw_mid) * dt,
            previous[2] + curvature * speed * dt,
        ]
        previous_time = float(time_s)
    return trajectory


def _sample_pixels(
    observed_flows: np.ndarray,
    support_weights: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observed_flows, dtype=np.float32)
    weights = np.asarray(support_weights, dtype=np.float32)
    if observed.ndim != 4 or observed.shape[-1] != 2:
        raise ValueError("observed_flows must have shape [T,H,W,2]")
    if weights.shape != observed.shape[:-1]:
        raise ValueError("support_weights must have shape [T,H,W]")
    valid = np.isfinite(observed).all(axis=-1) & np.isfinite(weights) & (weights > 0.0)
    valid &= np.linalg.norm(observed, axis=-1) > 0.05
    indices = np.flatnonzero(valid.any(axis=0))
    if indices.size == 0:
        raise ValueError("no finite motion pixels are available for trajectory decoding")
    # Keep the same pixel set across intervals so the objective does not move
    # its evidence source as the optimizer changes the trajectory.
    score = np.mean(np.where(valid, weights, 0.0), axis=0).reshape(-1)
    order = np.argsort(score[indices])[::-1]
    chosen = indices[order[: int(max_points)]]
    yy, xx = np.unravel_index(chosen, valid.shape[1:])
    coords = np.stack([xx, yy], axis=1).astype(np.float64)
    return coords, observed[:, yy, xx, :].astype(np.float64), weights[:, yy, xx].astype(np.float64)


def _sparse_predicted_flows(
    trajectory: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    pixel_xy: np.ndarray,
    *,
    image_size: tuple[int, int],
    depths_m: np.ndarray | None,
    adaptive_plane_params: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project only selected pixels for fast continuous optimization."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64)
    width, height = image_size
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape [T,3]")
    if pixel_xy.ndim != 2 or pixel_xy.shape[1] != 2:
        raise ValueError("pixel_xy must have shape [N,2]")
    poses = candidate_camera_poses(trajectory, np.asarray(camera_to_ego, dtype=np.float64))
    transforms = adjacent_camera_transforms(poses)
    homogeneous = np.concatenate([pixel_xy, np.ones((len(pixel_xy), 1))], axis=1).T
    normalized = np.linalg.inv(intrinsics) @ homogeneous
    depths = None if depths_m is None else np.asarray(depths_m, dtype=np.float64)
    if depths is not None and depths.shape != (len(transforms), height, width):
        raise ValueError("depths_m does not match the number or size of intervals")
    outputs = []
    validities = []
    for index, transform in enumerate(transforms):
        if depths is None:
            homography = ground_plane_homography(intrinsics, transform, poses[index])
            if adaptive_plane_params is not None:
                params = np.asarray(adaptive_plane_params, dtype=np.float64)
                if params.shape != (3,):
                    raise ValueError("adaptive_plane_params must have shape [3]")
                normal_anchor = np.asarray([-params[0], -params[1], 1.0], dtype=np.float64)
                distance = float(normal_anchor @ poses[index][:3, 3] + params[2])
                if abs(distance) < 1e-5:
                    raise ValueError("adaptive road plane is too close to camera")
                normal_camera = poses[index][:3, :3].T @ normal_anchor
                normalized = transform[:3, :3] - transform[:3, 3:4] @ normal_camera.reshape(1, 3) / distance
                homography = intrinsics @ normalized @ np.linalg.inv(intrinsics)
                homography = homography / homography[2, 2]
            projected = homography @ homogeneous
        else:
            depth = depths[index, pixel_xy[:, 1].astype(int), pixel_xy[:, 0].astype(int)]
            points = normalized * depth.reshape(1, -1)
            points_next = transform[:3, :3] @ points + transform[:3, 3:4]
            projected = intrinsics @ points_next
        denominator = projected[2]
        valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
        next_xy = np.zeros((2, len(pixel_xy)), dtype=np.float64)
        next_xy[:, valid] = projected[:2, valid] / denominator[valid]
        valid &= np.isfinite(next_xy).all(axis=0)
        valid &= (next_xy[0] >= 0.0) & (next_xy[0] <= width - 1)
        valid &= (next_xy[1] >= 0.0) & (next_xy[1] <= height - 1)
        flow = (next_xy - homogeneous[:2]).T
        flow[~valid] = np.nan
        outputs.append(flow)
        validities.append(valid)
    return np.stack(outputs), np.stack(validities)


def estimate_temporal_flow_scale_state(
    trajectory: np.ndarray,
    observed_flows: np.ndarray,
    support_weights: np.ndarray,
    *,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    depths_m: np.ndarray | None = None,
    minimum_flow_scale_px: float = 1.0,
    initial_scale: float = 1.0,
    initial_scale_std: float = 0.10,
    process_noise: float = 0.03,
    measurement_noise: float = 0.08,
    max_points: int = 900,
    max_scale_change: float = 0.25,
    max_log_innovation: float = 0.20,
) -> dict[str, Any]:
    """Estimate a smooth future flow-scale state from image/geometric agreement.

    ``observed/predicted`` ratios are measured on the same static support used
    by the candidate-blind decoder.  A scalar random-walk/Kalman update keeps
    the scale anchored to the history prior instead of independently
    renormalizing each future interval.  This is an image-side consistency
    diagnostic, never an action-derived calibration.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    observed = np.asarray(observed_flows, dtype=np.float64)
    weights = np.asarray(support_weights, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape [T,3]")
    if observed.ndim != 4 or observed.shape[-1] != 2 or observed.shape[0] != len(trajectory):
        raise ValueError("observed_flows must have shape [T,H,W,2] matching trajectory")
    if weights.shape != observed.shape[:-1]:
        raise ValueError("support_weights must match observed_flows")
    values = [
        initial_scale,
        initial_scale_std,
        process_noise,
        measurement_noise,
        max_scale_change,
        max_log_innovation,
    ]
    if not np.all(np.isfinite(values)) or initial_scale <= 0.0 or initial_scale_std < 0.0:
        raise ValueError("scale state parameters must be finite and positive")
    if process_noise < 0.0 or measurement_noise <= 0.0 or max_scale_change <= 0.0 or max_log_innovation <= 0.0:
        raise ValueError("invalid temporal scale state noise or change bound")
    if max_points < 20 or minimum_flow_scale_px <= 0.0:
        raise ValueError("max_points and minimum_flow_scale_px are invalid")
    pixel_xy, sampled_observed, sampled_weights = _sample_pixels(
        observed.astype(np.float32), weights.astype(np.float32), max_points=int(max_points)
    )
    predicted, predicted_valid = _sparse_predicted_flows(
        trajectory,
        camera_to_ego,
        intrinsics,
        pixel_xy,
        image_size=image_size,
        depths_m=depths_m,
    )
    mean = float(initial_scale)
    variance = float(initial_scale_std**2)
    rows: list[dict[str, Any]] = []
    corrections: list[float] = []
    for index in range(len(trajectory)):
        observed_norm = np.linalg.norm(sampled_observed[index], axis=1)
        predicted_norm = np.linalg.norm(predicted[index], axis=1)
        valid = predicted_valid[index]
        valid &= sampled_weights[index] > 0.0
        valid &= np.isfinite(observed_norm) & np.isfinite(predicted_norm)
        valid &= observed_norm >= float(minimum_flow_scale_px)
        valid &= predicted_norm >= float(minimum_flow_scale_px) * 0.25
        ratios = observed_norm[valid] / np.maximum(predicted_norm[valid], 1e-6)
        ratios = ratios[np.isfinite(ratios) & (ratios > 0.05) & (ratios < 20.0)]
        prior_mean = mean
        prior_variance = variance + float(process_noise**2)
        if len(ratios) >= 20:
            measurement = float(np.median(ratios))
            mad = float(1.4826 * np.median(np.abs(ratios - measurement)))
            innovation = abs(float(np.log(max(measurement, 1e-6) / max(prior_mean, 1e-6))))
            accepted = innovation <= float(max_log_innovation)
            if accepted:
                measurement_variance = max(float(measurement_noise**2), (mad / np.sqrt(len(ratios))) ** 2)
                gain = prior_variance / (prior_variance + measurement_variance)
                proposed = prior_mean + gain * (measurement - prior_mean)
                delta = np.clip(proposed - prior_mean, -float(max_scale_change), float(max_scale_change))
                mean = float(np.clip(prior_mean + delta, 0.5, 2.0))
                variance = float(max((1.0 - gain) * prior_variance, 1e-8))
                available = True
            else:
                gain = 0.0
                mean = float(np.clip(prior_mean, 0.5, 2.0))
                variance = float(prior_variance)
                available = False
        else:
            measurement = None
            mad = None
            gain = 0.0
            innovation = None
            accepted = False
            mean = float(np.clip(prior_mean, 0.5, 2.0))
            variance = float(prior_variance)
            available = False
        std = float(np.sqrt(max(variance, 1e-8)))
        corrections.append(float(np.clip(1.0 / mean, 0.5, 2.0)))
        rows.append({
            "interval_index": index,
            "valid_points": int(len(ratios)),
            "available": available,
            "measurement_scale": measurement,
            "measurement_mad": mad,
            "log_innovation": innovation,
            "innovation_accepted": bool(available),
            "kalman_gain": float(gain),
            "scale_posterior": {
                "q05": float(max(mean - 1.645 * std, 1e-3)),
                "q50": mean,
                "q95": float(mean + 1.645 * std),
            },
            "future_flow_correction": corrections[-1],
        })
    return {
        "protocol": "temporal-shared-flow-scale-v1",
        "available": bool(any(row["available"] for row in rows)),
        "initial_scale_posterior": {
            "q05": float(max(initial_scale - 1.645 * initial_scale_std, 1e-3)),
            "q50": float(initial_scale),
            "q95": float(initial_scale + 1.645 * initial_scale_std),
        },
        "rows": rows,
        "future_flow_corrections": corrections,
        "action_waypoint_used": False,
    }


def _objective(
    trajectory: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    pixel_xy: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    depths_m: np.ndarray | None,
    minimum_flow_scale_px: float,
    road_masks: np.ndarray | None = None,
    road_prior_weight: float = 0.0,
    road_half_width_m: float = 1.1,
    road_lateral_samples: int = 5,
    road_longitudinal_step_m: float = 0.5,
    speed_smoothness_weight: float = 0.0,
    curvature_smoothness_weight: float = 0.0,
    lateral_acceleration_weight: float = 0.0,
    future_times_s: np.ndarray | None = None,
    adaptive_plane_params: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    predicted, valid = _sparse_predicted_flows(
        trajectory,
        camera_to_ego,
        intrinsics,
        pixel_xy,
        image_size=image_size,
        depths_m=depths_m,
        adaptive_plane_params=adaptive_plane_params,
    )
    finite = valid & np.isfinite(observed).all(axis=-1) & np.isfinite(predicted).all(axis=-1)
    residual = np.linalg.norm(predicted - observed, axis=-1)
    scale = np.maximum(np.linalg.norm(observed, axis=-1), float(minimum_flow_scale_px))
    normalized = residual / scale
    finite &= np.isfinite(normalized)
    robust = np.minimum(normalized, 4.0)
    weighted = np.where(finite, robust * weights, 0.0)
    denominator = float(np.where(finite, weights, 0.0).sum())
    if denominator <= 1e-6:
        # Keep the optimizer numerically total even when this candidate has no
        # valid projected pixels. The finite worst-case cost lets callers
        # return an explicit low-coverage result instead of raising; coverage
        # remains exposed through ``valid`` and downstream observability.
        return 4.0, predicted, valid
    flow_energy = float(weighted.sum() / denominator)
    road_penalty = _road_prior_penalty(
        trajectory,
        road_masks,
        camera_to_ego=camera_to_ego,
        intrinsics=intrinsics,
        image_size=image_size,
        half_width_m=road_half_width_m,
        lateral_samples=road_lateral_samples,
        longitudinal_step_m=road_longitudinal_step_m,
    ) if road_masks is not None and road_prior_weight > 0.0 else 0.0
    smoothness_penalty = _kinematic_smoothness_penalty(
        trajectory,
        (
            np.arange(1, len(trajectory) + 1, dtype=np.float64)
            if future_times_s is None
            else np.asarray(future_times_s, dtype=np.float64)
        ),
        speed_weight=speed_smoothness_weight,
        curvature_weight=curvature_smoothness_weight,
        lateral_acceleration_weight=lateral_acceleration_weight,
    ) if any(
        value > 0.0 for value in (
            speed_smoothness_weight,
            curvature_smoothness_weight,
            lateral_acceleration_weight,
        )
    ) else 0.0
    return (
        flow_energy
        + float(road_prior_weight) * road_penalty
        + smoothness_penalty,
        predicted,
        valid,
    )


def _kinematic_smoothness_penalty(
    trajectory: np.ndarray,
    future_times_s: np.ndarray,
    *,
    speed_weight: float,
    curvature_weight: float,
    lateral_acceleration_weight: float,
) -> float:
    """Softly penalize discontinuous piecewise vehicle controls."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape [T,3]")
    if times.shape != (len(trajectory),) or np.any(np.diff(times) <= 0.0):
        raise ValueError("future_times_s must match trajectory and be increasing")
    if min(speed_weight, curvature_weight, lateral_acceleration_weight) < 0.0:
        raise ValueError("smoothness weights must be non-negative")
    if len(trajectory) < 2:
        return 0.0
    dt = np.diff(np.concatenate([[0.0], times]))
    deltas = np.diff(np.vstack([np.zeros((1, 2)), trajectory[:, :2]]), axis=0)
    speeds = np.linalg.norm(deltas, axis=1) / np.maximum(dt, 1e-6)
    distances = np.linalg.norm(deltas, axis=1)
    curvatures = np.diff(np.concatenate([[0.0], trajectory[:, 2]])) / np.maximum(distances, 1e-3)
    speed_scale = max(float(np.mean(speeds)), 1.0)
    speed_term = float(np.mean((np.diff(speeds) / speed_scale) ** 2))
    curvature_scale = 0.10
    curvature_term = float(np.mean((np.diff(curvatures) / curvature_scale) ** 2))
    lateral_acceleration = speeds * speeds * curvatures
    lateral_scale = max(float(np.mean(speeds * speeds)) * curvature_scale, 0.10)
    lateral_term = float(np.mean((np.diff(lateral_acceleration) / lateral_scale) ** 2))
    return float(
        speed_weight * speed_term
        + curvature_weight * curvature_term
        + lateral_acceleration_weight * lateral_term
    )


def _road_prior_penalty(
    trajectory: np.ndarray,
    road_masks: np.ndarray | None,
    *,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    half_width_m: float,
    lateral_samples: int,
    longitudinal_step_m: float,
) -> float:
    """Measure how far a trajectory corridor falls outside a road mask.

    This is a soft image-space prior. It is deliberately separate from the
    flow residual: a road segmentation error should lower confidence, not
    silently become a hard obstacle.
    """
    if road_masks is None:
        return 0.0
    trajectory = np.asarray(trajectory, dtype=np.float64)
    masks = np.asarray(road_masks, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape [T,3]")
    if masks.shape != (len(trajectory), int(image_size[1]), int(image_size[0])):
        raise ValueError("road_masks must have shape [T,H,W] matching trajectory and image_size")
    if half_width_m < 0.0 or lateral_samples < 1 or longitudinal_step_m <= 0.0:
        raise ValueError("road corridor parameters must be positive")
    camera_from_ego = np.linalg.inv(np.asarray(camera_to_ego, dtype=np.float64))
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    width, height = image_size
    offsets = np.linspace(-float(half_width_m), float(half_width_m), int(lateral_samples))
    total_penalty = 0.0
    total_visible = 0
    previous = np.zeros(3, dtype=np.float64)
    # The trajectory is expressed in the anchor ego frame. Temporal road
    # masks are per-flow-frame masks, so using a later mask directly would
    # mix coordinate systems. The first consensus mask is the anchor-side
    # evidence and is the only mask used for this anchor-frame prior.
    anchor_mask = masks[0]
    for interval_index, endpoint in enumerate(trajectory):
        distance = float(np.linalg.norm(endpoint[:2] - previous[:2]))
        count = max(1, int(np.ceil(distance / float(longitudinal_step_m))))
        alphas = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)[1:]
        centers = previous[None, :] + alphas[:, None] * (endpoint - previous)[None, :]
        points = []
        for center in centers:
            lateral = np.asarray([-np.sin(center[2]), np.cos(center[2])], dtype=np.float64)
            for offset in offsets:
                xy = center[:2] + float(offset) * lateral
                points.append([xy[0], xy[1], 0.0])
        points_ego = np.asarray(points, dtype=np.float64)
        homogeneous = np.c_[points_ego, np.ones(len(points_ego), dtype=np.float64)]
        points_camera = (camera_from_ego @ homogeneous.T).T[:, :3]
        projected = (intrinsics @ points_camera.T).T
        valid = np.isfinite(projected).all(axis=1) & (points_camera[:, 2] > 1e-6)
        valid &= np.abs(projected[:, 2]) > 1e-8
        pixels = np.zeros((len(points_camera), 2), dtype=np.float64)
        pixels[valid] = projected[valid, :2] / projected[valid, 2:3]
        valid &= (pixels[:, 0] >= 0.0) & (pixels[:, 0] < width)
        valid &= (pixels[:, 1] >= 0.0) & (pixels[:, 1] < height)
        if valid.any():
            xy = np.rint(pixels[valid]).astype(np.int64)
            xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
            road_probability = np.clip(anchor_mask[xy[:, 1], xy[:, 0]], 0.0, 1.0)
            total_penalty += float(np.sum(1.0 - road_probability))
            total_visible += int(len(road_probability))
        previous = endpoint
    return float(total_penalty / total_visible) if total_visible else 0.0


def _longitudinal_residual_penalty(
    speeds_mps: np.ndarray,
    history_speeds_mps: np.ndarray,
    *,
    maximum_residual_mps: float,
    residual_weight: float,
    residual_smoothness_weight: float,
    residual_curvature_weight: float = 0.0,
) -> float:
    """Regularize image-driven speed residuals around a frozen history curve."""
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    history = np.asarray(history_speeds_mps, dtype=np.float64)
    if speeds.shape != history.shape or speeds.ndim != 1:
        raise ValueError("speed and history curves must be matching vectors")
    if not np.all(np.isfinite(speeds)) or not np.all(np.isfinite(history)):
        raise ValueError("speed and history curves must be finite")
    if maximum_residual_mps <= 0.0 or not np.isfinite(maximum_residual_mps):
        raise ValueError("maximum_residual_mps must be finite and positive")
    if min(residual_weight, residual_smoothness_weight, residual_curvature_weight) < 0.0:
        raise ValueError("longitudinal residual weights must be non-negative")
    residual = speeds - history
    if np.any(np.abs(residual) > maximum_residual_mps + 1e-6):
        return float("inf")
    scale = max(float(maximum_residual_mps), 1.0)
    magnitude = float(np.mean((residual / scale) ** 2))
    smoothness = (
        0.0 if len(residual) < 2
        else float(np.mean((np.diff(residual) / scale) ** 2))
    )
    curvature = (
        0.0 if len(residual) < 3
        else float(np.mean((np.diff(residual, n=2) / scale) ** 2))
    )
    return float(
        residual_weight * magnitude
        + residual_smoothness_weight * smoothness
        + residual_curvature_weight * curvature
    )


def _fit_once(
    *,
    future_times_s: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    pixel_xy: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    depths_m: np.ndarray | None,
    minimum_flow_scale_px: float,
    initial_speed_mps: float,
    max_iterations: int,
    initial_curvatures_1pm: np.ndarray | None = None,
    fixed_speeds_mps: np.ndarray | None = None,
    history_speeds_mps: np.ndarray | None = None,
    history_initial_speed_mps: float | None = None,
    maximum_speed_residual_mps: float = 3.0,
    speed_residual_weight: float = 0.02,
    speed_residual_smoothness_weight: float = 0.05,
    speed_residual_curvature_weight: float = 0.0,
    road_masks: np.ndarray | None = None,
    road_prior_weight: float = 0.0,
    road_half_width_m: float = 1.1,
    road_lateral_samples: int = 5,
    road_longitudinal_step_m: float = 0.5,
    speed_smoothness_weight: float = 0.0,
    curvature_smoothness_weight: float = 0.0,
    lateral_acceleration_weight: float = 0.0,
    adaptive_plane_params: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    count = len(future_times_s)
    # Piecewise controls are continuous and low-dimensional: one speed and
    # curvature per future interval, with coordinate descent avoiding scipy.
    curvature_start = (
        np.zeros(count, dtype=np.float64)
        if initial_curvatures_1pm is None
        else np.asarray(initial_curvatures_1pm, dtype=np.float64)
    )
    if curvature_start.shape != (count,) or not np.all(np.isfinite(curvature_start)):
        raise ValueError("initial_curvatures_1pm must match future_times_s")
    fixed_speeds = None if fixed_speeds_mps is None else np.asarray(fixed_speeds_mps, dtype=np.float64)
    history_speeds = None if history_speeds_mps is None else np.asarray(history_speeds_mps, dtype=np.float64)
    if fixed_speeds is not None and history_speeds is not None:
        raise ValueError("fixed speeds and history-anchored residual mode are mutually exclusive")
    if history_speeds is not None and (
        history_initial_speed_mps is None
        or not np.isfinite(history_initial_speed_mps)
        or history_initial_speed_mps < 0.0
    ):
        raise ValueError("history residual mode requires a finite non-negative initial speed")
    if history_speeds is not None:
        if history_speeds.shape != (count,) or not np.all(np.isfinite(history_speeds)):
            raise ValueError("history_speeds_mps must be a finite vector matching future_times_s")
        history_speeds = np.clip(history_speeds, 0.05, 30.0)
    if fixed_speeds is not None:
        if fixed_speeds.shape != (count,) or not np.all(np.isfinite(fixed_speeds)):
            raise ValueError("fixed_speeds_mps must be a finite vector matching future_times_s")
        fixed_speeds = np.clip(fixed_speeds, 0.05, 30.0)
        params = np.clip(curvature_start, -0.35, 0.35)
        steps = np.full(count, 0.08, dtype=np.float64)
    elif history_speeds is not None:
        params = np.concatenate([
            np.zeros(count, dtype=np.float64),
            np.clip(curvature_start, -0.35, 0.35),
        ])
        steps = np.concatenate([
            np.full(count, min(0.5, maximum_speed_residual_mps), dtype=np.float64),
            np.full(count, 0.08, dtype=np.float64),
        ])
    else:
        params = np.concatenate([
            np.full(count, np.clip(initial_speed_mps, 0.2, 25.0), dtype=np.float64),
            np.clip(curvature_start, -0.35, 0.35),
        ])
        steps = np.concatenate([
            np.full(count, 2.0, dtype=np.float64),
            np.full(count, 0.08, dtype=np.float64),
        ])

    def evaluate(values: np.ndarray) -> float:
        if fixed_speeds is not None:
            speeds = fixed_speeds
        elif history_speeds is not None:
            residual = np.clip(values[:count], -maximum_speed_residual_mps, maximum_speed_residual_mps)
            speeds = np.clip(history_speeds + residual, 0.05, 30.0)
        else:
            speeds = np.clip(values[:count], 0.05, 30.0)
        curvature = np.clip(values if fixed_speeds is not None else values[count:], -0.35, 0.35)
        trajectory = integrate_piecewise_controls(
            future_times_s, speeds_mps=speeds, curvatures_1pm=curvature
        )
        energy = _objective(
            trajectory, observed, weights, pixel_xy, camera_to_ego, intrinsics,
            image_size, depths_m, minimum_flow_scale_px, road_masks,
            road_prior_weight, road_half_width_m, road_lateral_samples,
            road_longitudinal_step_m,
            speed_smoothness_weight, curvature_smoothness_weight,
            lateral_acceleration_weight,
            future_times_s,
            adaptive_plane_params,
        )[0]
        if history_speeds is not None:
            energy += _longitudinal_residual_penalty(
                speeds,
                history_speeds,
                maximum_residual_mps=maximum_speed_residual_mps,
                residual_weight=speed_residual_weight,
                residual_smoothness_weight=speed_residual_smoothness_weight,
                residual_curvature_weight=speed_residual_curvature_weight,
            )
        return float(energy)

    best = evaluate(params)
    for _ in range(int(max_iterations)):
        improved = False
        for index in range(len(params)):
            for direction in (-1.0, 1.0):
                trial = params.copy()
                trial[index] += direction * steps[index]
                value = evaluate(trial)
                if value + 1e-7 < best:
                    params, best = trial, value
                    improved = True
        steps *= 0.55
        if not improved and float(np.max(steps)) < 1e-3:
            break
    if fixed_speeds is not None:
        final_speeds = fixed_speeds
    elif history_speeds is not None:
        final_speeds = np.clip(
            history_speeds + np.clip(params[:count], -maximum_speed_residual_mps, maximum_speed_residual_mps),
            0.05,
            30.0,
        )
    else:
        final_speeds = np.clip(params[:count], 0.05, 30.0)
    trajectory = integrate_piecewise_controls(
        future_times_s,
        speeds_mps=final_speeds,
        curvatures_1pm=np.clip(params if fixed_speeds is not None else params[count:], -0.35, 0.35),
    )
    return trajectory, float(best)


def decode_continuous_trajectory(
    *,
    observed_flows: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    future_times_s: np.ndarray,
    roi_mask: np.ndarray,
    consistency_masks: np.ndarray | None = None,
    dynamic_weights: np.ndarray | None = None,
    depths_m: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    minimum_flow_scale_px: float = 1.0,
    max_points: int = 900,
    max_iterations: int = 12,
    initial_speeds_mps: tuple[float, ...] = (3.0, 6.0, 10.0),
    profile_radius: float = 0.12,
    interval_observability: np.ndarray | None = None,
    speed_uncertainty_thresholds: tuple[float, float] = (0.25, 0.55),
    curvature_multistart: bool = False,
    fixed_speeds_mps: np.ndarray | None = None,
    history_speeds_mps: np.ndarray | None = None,
    history_initial_speed_mps: float | None = None,
    maximum_speed_residual_mps: float = 3.0,
    speed_residual_weight: float = 0.02,
    speed_residual_smoothness_weight: float = 0.05,
    speed_residual_curvature_weight: float = 0.0,
    initial_curvatures_1pm: np.ndarray | None = None,
    road_masks: np.ndarray | None = None,
    road_prior_weight: float = 0.0,
    road_half_width_m: float = 1.1,
    road_lateral_samples: int = 5,
    road_longitudinal_step_m: float = 0.5,
    speed_smoothness_weight: float = 0.0,
    curvature_smoothness_weight: float = 0.0,
    lateral_acceleration_weight: float = 0.0,
    adaptive_plane_params: np.ndarray | None = None,
) -> dict[str, Any]:
    """Recover a continuous trajectory and local support tube from flow."""
    observed = np.asarray(observed_flows, dtype=np.float32)
    if image_size is None:
        image_size = (observed.shape[2], observed.shape[1])
    roi = np.asarray(roi_mask, dtype=bool)
    if roi.shape != observed.shape[1:3]:
        raise ValueError("roi_mask must match observed flow spatial dimensions")
    if consistency_masks is None:
        consistency = np.ones(observed.shape[:-1], dtype=bool)
    else:
        consistency = np.asarray(consistency_masks, dtype=bool)
    if dynamic_weights is None:
        weights = np.ones(observed.shape[:-1], dtype=np.float32)
    else:
        weights = np.asarray(dynamic_weights, dtype=np.float32)
    if consistency.shape != weights.shape or consistency.shape != observed.shape[:-1]:
        raise ValueError("consistency_masks and dynamic_weights must match observed flow")
    if road_masks is not None:
        road_masks = np.asarray(road_masks, dtype=np.float32)
        if road_masks.shape != observed.shape[:-1]:
            raise ValueError("road_masks must match observed flow intervals and spatial size")
        if not np.all(np.isfinite(road_masks)):
            raise ValueError("road_masks must be finite")
        road_masks = np.clip(road_masks, 0.0, 1.0)
    if road_prior_weight < 0.0:
        raise ValueError("road_prior_weight must be non-negative")
    if min(
        speed_smoothness_weight,
        curvature_smoothness_weight,
        lateral_acceleration_weight,
    ) < 0.0:
        raise ValueError("smoothness weights must be non-negative")
    if maximum_speed_residual_mps <= 0.0 or not np.isfinite(maximum_speed_residual_mps):
        raise ValueError("maximum_speed_residual_mps must be finite and positive")
    if min(speed_residual_weight, speed_residual_smoothness_weight, speed_residual_curvature_weight) < 0.0:
        raise ValueError("speed residual weights must be non-negative")
    if interval_observability is None:
        interval_quality = np.mean(weights > 0.0, axis=(1, 2)).astype(np.float64)
    else:
        interval_quality = np.asarray(interval_observability, dtype=np.float64)
        if interval_quality.shape != (observed.shape[0],) or not np.all(np.isfinite(interval_quality)):
            raise ValueError("interval_observability must be a finite vector matching intervals")
        interval_quality = np.clip(interval_quality, 0.0, 1.0)
    weights = np.where(roi[None, ...] & consistency, np.maximum(weights, 0.0), 0.0)
    pixel_xy, sampled_observed, sampled_weights = _sample_pixels(
        observed, weights, max_points=max_points
    )
    best_trajectory = None
    best_energy = float("inf")
    fixed_speeds = None if fixed_speeds_mps is None else np.asarray(fixed_speeds_mps, dtype=np.float64)
    if fixed_speeds is not None and (
        fixed_speeds.shape != np.asarray(future_times_s).shape or not np.all(np.isfinite(fixed_speeds))
    ):
        raise ValueError("fixed_speeds_mps must be a finite vector matching future_times_s")
    history_speeds = None if history_speeds_mps is None else np.asarray(history_speeds_mps, dtype=np.float64)
    if history_speeds is not None and (
        history_speeds.shape != np.asarray(future_times_s).shape or not np.all(np.isfinite(history_speeds))
    ):
        raise ValueError("history_speeds_mps must be a finite vector matching future_times_s")
    if fixed_speeds is not None and history_speeds is not None:
        raise ValueError("fixed speeds and history-anchored residual mode are mutually exclusive")
    starts = (
        (float(np.mean(fixed_speeds)),)
        if fixed_speeds is not None
        else (float(np.mean(history_speeds)),)
        if history_speeds is not None
        else tuple(float(value) for value in initial_speeds_mps)
    )
    supplied_curvatures = None if initial_curvatures_1pm is None else np.asarray(initial_curvatures_1pm, dtype=np.float64)
    if supplied_curvatures is not None and (
        supplied_curvatures.shape != np.asarray(future_times_s).shape or not np.all(np.isfinite(supplied_curvatures))
    ):
        raise ValueError("initial_curvatures_1pm must be a finite vector matching future_times_s")
    curvature_starts = (supplied_curvatures,) if supplied_curvatures is not None else (
        np.zeros(len(future_times_s), dtype=np.float64),
        np.full(len(future_times_s), -0.04, dtype=np.float64),
        np.full(len(future_times_s), 0.04, dtype=np.float64),
        np.linspace(-0.03, 0.03, len(future_times_s), dtype=np.float64),
        np.linspace(0.03, -0.03, len(future_times_s), dtype=np.float64),
    ) if curvature_multistart else (np.zeros(len(future_times_s), dtype=np.float64),)
    for initial_speed in starts:
        for initial_curvatures in curvature_starts:
            trajectory, energy = _fit_once(
                future_times_s=np.asarray(future_times_s, dtype=np.float64),
                observed=sampled_observed,
                weights=sampled_weights,
                pixel_xy=pixel_xy,
                camera_to_ego=camera_to_ego,
                intrinsics=intrinsics,
                image_size=image_size,
                depths_m=depths_m,
                minimum_flow_scale_px=minimum_flow_scale_px,
                initial_speed_mps=initial_speed,
                initial_curvatures_1pm=initial_curvatures,
                fixed_speeds_mps=fixed_speeds,
                history_speeds_mps=history_speeds,
                history_initial_speed_mps=history_initial_speed_mps,
                maximum_speed_residual_mps=maximum_speed_residual_mps,
                speed_residual_weight=speed_residual_weight,
                speed_residual_smoothness_weight=speed_residual_smoothness_weight,
                speed_residual_curvature_weight=speed_residual_curvature_weight,
                max_iterations=max_iterations,
                road_masks=road_masks,
                road_prior_weight=road_prior_weight,
                road_half_width_m=road_half_width_m,
                road_lateral_samples=road_lateral_samples,
                road_longitudinal_step_m=road_longitudinal_step_m,
                speed_smoothness_weight=speed_smoothness_weight,
                curvature_smoothness_weight=curvature_smoothness_weight,
                lateral_acceleration_weight=lateral_acceleration_weight,
                adaptive_plane_params=adaptive_plane_params,
            )
            if energy < best_energy:
                best_trajectory, best_energy = trajectory, energy
    if best_trajectory is None or not np.isfinite(best_energy):
        raise ValueError("continuous trajectory optimizer found no valid solution")

    # Profile a local continuous neighborhood. This is an uncertainty tube,
    # not a second finite candidate bank: all perturbations are in control
    # space and receive weights from the image residual.
    controls = np.stack([
        np.diff(np.concatenate([[0.0], best_trajectory[:, 0]])) / np.diff(np.concatenate([[0.0], np.asarray(future_times_s)])),
        np.zeros(len(best_trajectory)),
    ], axis=1)
    speeds = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), best_trajectory[:, :2]]), axis=0), axis=1) / np.diff(np.concatenate([[0.0], np.asarray(future_times_s)]))
    yaws = best_trajectory[:, 2]
    distances = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), best_trajectory[:, :2]]), axis=0), axis=1)
    curvatures = np.diff(np.concatenate([[0.0], yaws])) / np.maximum(distances, 1e-3)
    profile = []
    for speed_delta in (-profile_radius, 0.0, profile_radius):
        for curvature_delta in (-profile_radius, 0.0, profile_radius):
            if history_speeds is None:
                candidate_speeds = np.clip(speeds * (1.0 + speed_delta), 0.05, 30.0)
            else:
                residual = speeds - history_speeds
                candidate_speeds = np.clip(
                    history_speeds
                    + np.clip(
                        residual + speed_delta * max(float(maximum_speed_residual_mps), 1.0),
                        -maximum_speed_residual_mps,
                        maximum_speed_residual_mps,
                    ),
                    0.05,
                    30.0,
                )
            candidate = integrate_piecewise_controls(
                np.asarray(future_times_s),
                speeds_mps=candidate_speeds,
                curvatures_1pm=np.clip(curvatures + curvature_delta * 0.1, -0.35, 0.35),
            )
            energy, _, _ = _objective(
                candidate, sampled_observed, sampled_weights, pixel_xy,
                camera_to_ego, intrinsics, image_size, depths_m, minimum_flow_scale_px,
                road_masks, road_prior_weight, road_half_width_m, road_lateral_samples,
                road_longitudinal_step_m,
                speed_smoothness_weight, curvature_smoothness_weight,
                lateral_acceleration_weight,
                np.asarray(future_times_s, dtype=np.float64),
                adaptive_plane_params,
            )
            if history_speeds is not None:
                energy += _longitudinal_residual_penalty(
                    candidate_speeds,
                    history_speeds,
                    maximum_residual_mps=maximum_speed_residual_mps,
                    residual_weight=speed_residual_weight,
                    residual_smoothness_weight=speed_residual_smoothness_weight,
                    residual_curvature_weight=speed_residual_curvature_weight,
                )
            profile.append((candidate, energy))
    profile.sort(key=lambda item: item[1])
    selected = [item for item in profile if item[1] <= best_energy + max(0.05, 0.5 * best_energy)]
    # Extreme adaptive-plane fits can make the local energy window empty due to
    # floating-point ties or invalid projections.  Keep the best finite profile
    # sample instead of turning one optional diagnostic into a whole-record
    # evaluation error.
    if not selected and profile:
        selected = [profile[0]]
    cloud = np.stack([item[0] for item in selected], axis=0)
    cloud_speeds = []
    for cloud_trajectory in cloud:
        cloud_states = trajectory_states(cloud_trajectory, np.asarray(future_times_s, dtype=np.float64))
        cloud_speeds.append([state["speed_mps"] for state in cloud_states])
    speed_cloud = np.asarray(cloud_speeds, dtype=np.float64)
    curvature_cloud = []
    for cloud_trajectory in cloud:
        cloud_states = trajectory_states(cloud_trajectory, np.asarray(future_times_s, dtype=np.float64))
        curvature_cloud.append([state["curvature_1pm"] for state in cloud_states])
    curvature_cloud = np.asarray(curvature_cloud, dtype=np.float64)
    speed_support = []
    low_speed_quality, uncertain_speed_quality = speed_uncertainty_thresholds
    for index, time_s in enumerate(np.asarray(future_times_s, dtype=np.float64)):
        q05, q50, q95 = (float(np.quantile(speed_cloud[:, index], q)) for q in (0.05, 0.50, 0.95))
        # A one-point local profile is not evidence of exact speed. Keep a
        # conservative uncertainty floor tied to the explored control radius,
        # then inflate it when motion observability is weak.
        uncertainty_floor = float(profile_radius) * max(abs(q50), 1.0)
        quality_scale = 1.0 / max(float(interval_quality[index]), 0.10)
        half_width = max((q95 - q05) * 0.5, uncertainty_floor) * quality_scale
        q05 = q50 - half_width
        q95 = q50 + half_width
        relative_width = (q95 - q05) / max(abs(q50), 1.0)
        if interval_quality[index] < float(low_speed_quality):
            status = "abstain"
        elif interval_quality[index] < float(uncertain_speed_quality):
            status = "uncertain"
        else:
            status = "usable"
        speed_support.append({
            "time_s": float(time_s),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "relative_width": float(relative_width),
            "observability": float(interval_quality[index]),
            "status": status,
        })
    support = []
    for index, time_s in enumerate(np.asarray(future_times_s, dtype=np.float64)):
        support.append({
            "time_s": float(time_s),
            "x_m": {"q05": float(np.quantile(cloud[:, index, 0], 0.05)), "q50": float(np.quantile(cloud[:, index, 0], 0.50)), "q95": float(np.quantile(cloud[:, index, 0], 0.95))},
            "y_m": {"q05": float(np.quantile(cloud[:, index, 1], 0.05)), "q50": float(np.quantile(cloud[:, index, 1], 0.50)), "q95": float(np.quantile(cloud[:, index, 1], 0.95))},
            "yaw_rad": {"q05": float(np.quantile(cloud[:, index, 2], 0.05)), "q50": float(np.quantile(cloud[:, index, 2], 0.50)), "q95": float(np.quantile(cloud[:, index, 2], 0.95))},
            "curvature_1pm": {"q05": float(np.quantile(curvature_cloud[:, index], 0.05)), "q50": float(np.quantile(curvature_cloud[:, index], 0.50)), "q95": float(np.quantile(curvature_cloud[:, index], 0.95))},
        })
    return {
        "protocol": "candidate-blind-continuous-trajectory-v1",
        "trajectory": best_trajectory.tolist(),
        "energy": float(best_energy),
        "profile_support": support,
        "profile_count": len(selected),
        "speed_support": speed_support,
        "speed_status_by_interval": [item["status"] for item in speed_support],
        "history_speed_profile_mps": None if history_speeds is None else history_speeds.tolist(),
        "history_initial_speed_mps": (
            None if history_speeds is None else float(history_initial_speed_mps)
        ),
        "speed_residual_mps": None if history_speeds is None else (speeds - history_speeds).tolist(),
        "speed_scored": False,
        "sampled_points": int(len(pixel_xy)),
        "effective_weight": float(sampled_weights.sum()),
        "observability": float(np.mean(sampled_weights > 0.0)),
        "initial_speeds_mps": list(starts),
        "decoder_parameters": {
            "max_points": int(max_points),
            "max_iterations": int(max_iterations),
            "profile_radius": float(profile_radius),
            "speed_uncertainty_thresholds": [float(low_speed_quality), float(uncertain_speed_quality)],
            "curvature_multistart": bool(curvature_multistart),
            "fixed_speed_shape_refinement": bool(fixed_speeds is not None),
            "history_anchored_speed_residual": bool(history_speeds is not None),
            "maximum_speed_residual_mps": float(maximum_speed_residual_mps),
            "speed_residual_weight": float(speed_residual_weight),
            "speed_residual_smoothness_weight": float(speed_residual_smoothness_weight),
            "speed_residual_curvature_weight": float(speed_residual_curvature_weight),
            "road_prior_enabled": bool(road_masks is not None and road_prior_weight > 0.0),
            "road_prior_weight": float(road_prior_weight),
            "road_half_width_m": float(road_half_width_m),
            "road_lateral_samples": int(road_lateral_samples),
            "road_longitudinal_step_m": float(road_longitudinal_step_m),
            "speed_smoothness_weight": float(speed_smoothness_weight),
            "curvature_smoothness_weight": float(curvature_smoothness_weight),
            "lateral_acceleration_weight": float(lateral_acceleration_weight),
        },
    }


def compare_continuous_trajectory(
    predicted: np.ndarray,
    reference: np.ndarray,
    future_times_s: np.ndarray,
    *,
    observability: np.ndarray | None = None,
    lateral_tolerance_m: float = 0.50,
    yaw_tolerance_rad: float = 0.10,
    speed_relative_tolerance: float = 0.20,
    curvature_tolerance_1pm: float = 0.06,
    score_speed: bool = False,
) -> dict[str, Any]:
    """Compare two trajectories with tolerant, joint, observability-weighted metrics."""
    predicted_states = trajectory_states(np.asarray(predicted), np.asarray(future_times_s))
    reference_states = trajectory_states(np.asarray(reference), np.asarray(future_times_s))
    if len(predicted_states) != len(reference_states):
        raise ValueError("predicted and reference trajectories must have matching knots")
    weights = np.ones(len(predicted_states), dtype=np.float64) if observability is None else np.asarray(observability, dtype=np.float64)
    if weights.shape != (len(predicted_states),) or not np.all(np.isfinite(weights)):
        raise ValueError("observability must be a finite vector matching trajectory knots")
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights[:] = 1.0
    errors = []
    joint = []
    heading_cosines = []
    for predicted_state, reference_state in zip(predicted_states, reference_states):
        lateral = abs(float(predicted_state["y_m"] - reference_state["y_m"]))
        yaw_delta = float(np.arctan2(
            np.sin(predicted_state["yaw_rad"] - reference_state["yaw_rad"]),
            np.cos(predicted_state["yaw_rad"] - reference_state["yaw_rad"]),
        ))
        speed_relative = abs(float(predicted_state["speed_mps"] - reference_state["speed_mps"])) / max(float(reference_state["speed_mps"]), 1.0)
        curvature = abs(float(predicted_state["curvature_1pm"] - reference_state["curvature_1pm"]))
        normalized_components = [
            lateral / max(float(lateral_tolerance_m), 1e-6),
            abs(yaw_delta) / max(float(yaw_tolerance_rad), 1e-6),
            curvature / max(float(curvature_tolerance_1pm), 1e-6),
        ]
        if score_speed:
            normalized_components.append(speed_relative / max(float(speed_relative_tolerance), 1e-6))
        normalized = np.asarray(normalized_components)
        errors.append({
            "lateral_abs_m": lateral,
            "yaw_abs_rad": abs(yaw_delta),
            "speed_relative_error": speed_relative,
            "curvature_abs_1pm": curvature,
            "normalized_joint_error": float(np.max(normalized)),
        })
        joint.append(bool(np.all(normalized <= 1.0)))
        heading_cosines.append(float(np.cos(yaw_delta)))
    normalized_errors = np.asarray([item["normalized_joint_error"] for item in errors])
    lateral_errors = np.asarray([item["lateral_abs_m"] for item in errors], dtype=np.float64)
    yaw_errors = np.asarray([item["yaw_abs_rad"] for item in errors], dtype=np.float64)
    speed_errors = np.asarray([item["speed_relative_error"] for item in errors], dtype=np.float64)
    curvature_errors = np.asarray([item["curvature_abs_1pm"] for item in errors], dtype=np.float64)
    return {
        "protocol": "continuous-trajectory-tolerant-comparison-v1",
        "weighted_mean_joint_error": float(np.average(normalized_errors, weights=weights)),
        "median_joint_error": float(np.median(normalized_errors)),
        "soft_compatibility": float(np.exp(-np.average(normalized_errors, weights=weights))),
        "joint_coverage": float(np.average(np.asarray(joint, dtype=np.float64), weights=weights)),
        "mean_heading_cosine": float(np.average(np.asarray(heading_cosines), weights=weights)),
        "mean_lateral_abs_m": float(np.average(lateral_errors, weights=weights)),
        "mean_yaw_abs_rad": float(np.average(yaw_errors, weights=weights)),
        "mean_speed_relative_error": float(np.average(speed_errors, weights=weights)),
        "mean_curvature_abs_1pm": float(np.average(curvature_errors, weights=weights)),
        "errors_by_knot": errors,
        "tolerances": {
            "lateral_tolerance_m": float(lateral_tolerance_m),
            "yaw_tolerance_rad": float(yaw_tolerance_rad),
            "speed_relative_tolerance": float(speed_relative_tolerance),
            "curvature_tolerance_1pm": float(curvature_tolerance_1pm),
        },
        "score_components": ["lateral", "yaw", "curvature"] + (["speed"] if score_speed else []),
        "speed_scored": bool(score_speed),
    }
