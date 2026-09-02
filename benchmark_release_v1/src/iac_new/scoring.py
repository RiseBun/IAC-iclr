"""Robust candidate energies, posterior weights, and finite-sample sets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import (
    adjacent_camera_transforms,
    candidate_camera_poses,
    ground_plane_homography,
    homography_flow,
    rigid_flow_from_depth,
)


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    energy: float
    median_epe_px: float
    p75_epe_px: float
    min_valid_pixel_fraction: float
    intervals: list[dict[str, float]]


def predict_candidate_flows(
    *,
    trajectory: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
    depths_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize candidate flow from a plane or per-interval metric depth."""
    poses = candidate_camera_poses(trajectory, camera_to_ego)
    transforms = adjacent_camera_transforms(poses)
    if depths_m is not None:
        depths = np.asarray(depths_m, dtype=np.float32)
        if depths.shape != (len(transforms), height, width):
            raise ValueError(
                f"depths must have shape {(len(transforms), height, width)}, got {depths.shape}"
            )
    else:
        depths = None
    predicted: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for index, (current_pose, transform) in enumerate(zip(poses[:-1], transforms)):
        if depths is None:
            homography = ground_plane_homography(intrinsics, transform, current_pose)
            flow, geometry_valid = homography_flow(homography, height, width)
        else:
            flow, geometry_valid = rigid_flow_from_depth(
                depths[index], intrinsics, transform
            )
        predicted.append(flow)
        valid.append(geometry_valid)
    return np.stack(predicted), np.stack(valid)


def dynamic_suppression_weights(
    *,
    observed_flows: np.ndarray,
    predicted_flows: np.ndarray,
    roi_mask: np.ndarray,
    consistency_masks: np.ndarray | None,
    absolute_threshold_px: float,
    relative_threshold: float,
    common_geometry_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build candidate-independent static weights from the best rigid explanation.

    Pixels unexplained by every candidate are likely dynamic, non-planar, or
    otherwise unreliable. The returned weights are shared by all candidates,
    so this step cannot selectively hide a candidate's own residual.
    """
    observed = np.asarray(observed_flows, dtype=np.float32)
    predictions = np.asarray(predicted_flows, dtype=np.float32)
    if predictions.ndim != 5 or predictions.shape[-1] != 2:
        raise ValueError("predicted_flows must have shape [C,T,H,W,2]")
    if observed.shape != predictions.shape[1:]:
        raise ValueError("observed and predicted flow shapes do not match")
    residual = np.linalg.norm(predictions - observed[None, ...], axis=-1)
    residual[~np.isfinite(residual)] = np.inf
    best_residual = np.min(residual, axis=0)
    observed_magnitude = np.linalg.norm(observed, axis=-1)
    scale = float(absolute_threshold_px) + float(relative_threshold) * observed_magnitude
    normalized = best_residual / np.maximum(scale, 1e-6)
    weights = np.exp(-0.5 * np.square(np.minimum(normalized, 12.0))).astype(np.float32)
    common = np.broadcast_to(
        np.asarray(roi_mask, dtype=bool)[None, ...], observed.shape[:-1]
    ).copy()
    common &= np.isfinite(observed).all(axis=-1)
    if consistency_masks is not None:
        common &= np.asarray(consistency_masks, dtype=bool)
    if common_geometry_masks is not None:
        common &= np.asarray(common_geometry_masks, dtype=bool)
    weights[~common] = 0.0
    weights[~np.isfinite(weights)] = 0.0
    return weights, best_residual.astype(np.float32)


def _curvature_observability(
    flow: np.ndarray,
    support: np.ndarray,
    *,
    roi_count: int,
    min_support_fraction: float,
    min_lateral_contrast_rad: float,
    min_flow_gradient_px: float,
    reliable_lateral_contrast_rad: float,
) -> dict[str, Any]:
    """Measure whether an interval contains lateral evidence for shape change.

    This is deliberately a diagnostic, not a curvature estimator.  A forward
    camera can show large flow while still being unable to distinguish speed,
    depth, and curvature.  We therefore require both support on the two sides
    of the image and a reproducible left/right flow contrast.
    """
    height, width = flow.shape[:2]
    finite = np.isfinite(flow).all(axis=-1) & (support > 0.0)
    count = int(finite.sum())
    support_fraction = float(support.sum() / max(roi_count, 1))
    if count < 2:
        return {
            "curvature_spatial_evidence": False,
            "curvature_observable": False,
            "curvature_status": "abstain",
            "curvature_support_fraction": support_fraction,
            "curvature_lateral_contrast_rad": 0.0,
            "curvature_flow_gradient_px": 0.0,
            "curvature_spatial_confidence": 0.0,
            "curvature_reason": "insufficient_lateral_support",
        }
    yy, xx = np.indices((height, width), dtype=np.float64)
    x = (xx[finite] / max(width - 1, 1))
    u = flow[..., 0][finite].astype(np.float64)
    v = flow[..., 1][finite].astype(np.float64)
    # Use the flow direction relative to the image axes; this suppresses most
    # of the common scale ambiguity that dominates raw flow magnitude.
    angle = np.arctan2(u, np.maximum(np.abs(v), 1e-3))
    left = x <= 0.40
    right = x >= 0.60
    if int(left.sum()) < 2 or int(right.sum()) < 2:
        return {
            "curvature_spatial_evidence": False,
            "curvature_observable": False,
            "curvature_status": "abstain",
            "curvature_support_fraction": support_fraction,
            "curvature_lateral_contrast_rad": 0.0,
            "curvature_flow_gradient_px": 0.0,
            "curvature_spatial_confidence": 0.0,
            "curvature_reason": "insufficient_left_right_support",
        }
    left_angle = float(np.median(angle[left]))
    right_angle = float(np.median(angle[right]))
    contrast = float(abs(np.arctan2(np.sin(right_angle - left_angle), np.cos(right_angle - left_angle))))
    centered_x = x - float(np.mean(x))
    centered_u = u - float(np.mean(u))
    denominator = float(np.sum(centered_x * centered_x))
    gradient = float(abs(np.sum(centered_x * centered_u) / denominator)) if denominator > 1e-8 else 0.0
    observable = (
        support_fraction >= float(min_support_fraction)
        and contrast >= float(min_lateral_contrast_rad)
        and gradient >= float(min_flow_gradient_px)
    )
    spatial_confidence = float(np.clip(
        min(
            support_fraction / max(float(min_support_fraction), 1e-6),
            contrast / max(float(reliable_lateral_contrast_rad), 1e-6),
        ),
        0.0,
        1.0,
    ))
    return {
        # This is spatial evidence only.  Perspective translation itself has
        # a strong horizontal gradient, so it must not independently claim
        # curvature observability; temporal evidence below makes that call.
        "curvature_spatial_evidence": bool(observable),
        "curvature_support_fraction": support_fraction,
        "curvature_lateral_contrast_rad": contrast,
        "curvature_flow_gradient_px": gradient,
        "curvature_spatial_confidence": spatial_confidence,
        "curvature_reason": None if observable else "weak_lateral_flow_structure",
    }


def _attach_temporal_curvature_observability(
    result: list[dict[str, Any]],
    observed: np.ndarray,
    roi: np.ndarray,
    dynamic_weights: np.ndarray | None,
    *,
    minimum_flow_scale_px: float,
) -> None:
    """Attach temporal turn evidence without confusing perspective with yaw.

    Curvature is a *change in heading over time*.  A single frame's radial
    flow cannot prove it.  We therefore calculate a robust central-flow angle
    per interval and mark an interval usable when it has reliable motion plus
    either a resolved turn transition or a stable straight continuation.
    """
    height, width = roi.shape
    yy, xx = np.indices((height, width))
    central = roi & (xx >= int(0.25 * width)) & (xx < int(0.75 * width)) & (yy >= int(0.45 * height))
    angles: list[float | None] = []
    supports: list[float] = []
    for index, flow in enumerate(observed):
        weights = np.ones((height, width), dtype=np.float32) if dynamic_weights is None else np.asarray(dynamic_weights[index], dtype=np.float32)
        valid = central & np.isfinite(flow).all(axis=-1) & (weights > 0.0)
        if int(valid.sum()) < 8:
            angles.append(None)
            supports.append(0.0)
            continue
        u = flow[..., 0][valid]
        v = flow[..., 1][valid]
        angles.append(float(np.arctan2(np.median(u), max(abs(float(np.median(v))), 1e-3))))
        supports.append(float(weights[valid].mean()))
    for index, item in enumerate(result):
        local = [angles[j] for j in (index - 1, index, index + 1) if 0 <= j < len(angles) and angles[j] is not None]
        transitions = []
        for previous, current in zip(local[:-1], local[1:]):
            transitions.append(float(np.arctan2(np.sin(current - previous), np.cos(current - previous))))
        turn_change = float(np.median(np.abs(transitions))) if transitions else None
        signed_turn_change = float(np.median(transitions)) if transitions else None
        flow_ok = float(item["median_observed_flow_px"]) >= float(minimum_flow_scale_px)
        support_ok = supports[index] >= 0.10
        if not flow_ok or not support_ok:
            status, reason = "abstain", "insufficient_temporal_static_support"
        elif turn_change is None:
            status, reason = "uncertain", "single_interval_temporal_ambiguity"
        elif turn_change >= 0.01:
            status, reason = "usable", "resolved_turn_transition"
        else:
            status, reason = "usable", "stable_straight_transition"
        item.update({
            "curvature_observable": status == "usable",
            "curvature_status": status,
            "curvature_temporal_turn_change_rad": turn_change,
            "curvature_temporal_heading_delta_rad": signed_turn_change,
            "curvature_temporal_static_support": supports[index],
            "curvature_confidence": float(
                item.get("curvature_spatial_confidence", 0.0)
                * min(supports[index] / 0.10, 1.0)
                if status != "abstain" else 0.0
            ),
            "curvature_reason": reason,
        })


def interval_observability(
    *,
    observed_flows: np.ndarray,
    roi_mask: np.ndarray,
    consistency_masks: np.ndarray | None,
    dynamic_weights: np.ndarray | None,
    minimum_flow_scale_px: float,
    static_weight_threshold: float,
    min_effective_pixel_fraction: float,
    role: str,
    common_geometry_masks: np.ndarray | None = None,
    speed_min_effective_pixel_fraction: float | None = None,
    speed_min_flow_scale_px: float | None = None,
    curvature_min_support_fraction: float = 0.05,
    curvature_min_lateral_contrast_rad: float = 0.02,
    curvature_min_flow_gradient_px: float = 0.02,
    curvature_reliable_lateral_contrast_rad: float = 1.43,
) -> list[dict[str, Any]]:
    """Summarize whether each temporal interval carries usable motion signal."""
    observed = np.asarray(observed_flows, dtype=np.float32)
    roi = np.asarray(roi_mask, dtype=bool)
    base = roi[None, ...] & np.isfinite(observed).all(axis=-1)
    if common_geometry_masks is not None:
        geometry_masks = np.asarray(common_geometry_masks, dtype=bool)
        if geometry_masks.shape != base.shape:
            raise ValueError("common_geometry_masks shape does not match observed flows")
        base &= geometry_masks
    fb_available = consistency_masks is not None
    if consistency_masks is None:
        fb = np.ones(base.shape, dtype=bool)
    else:
        fb = np.asarray(consistency_masks, dtype=bool)
    result: list[dict[str, Any]] = []
    roi_count = max(int(roi.sum()), 1)
    speed_support_threshold = (
        float(min_effective_pixel_fraction) * 2.0
        if speed_min_effective_pixel_fraction is None
        else float(speed_min_effective_pixel_fraction)
    )
    speed_flow_threshold = (
        float(minimum_flow_scale_px) * 1.5
        if speed_min_flow_scale_px is None
        else float(speed_min_flow_scale_px)
    )
    for index in range(observed.shape[0]):
        flow_magnitude = np.linalg.norm(observed[index], axis=-1)
        finite_roi = base[index]
        weights = (
            np.ones_like(flow_magnitude, dtype=np.float32)
            if dynamic_weights is None
            else np.asarray(dynamic_weights[index], dtype=np.float32)
        )
        weights = np.where(finite_roi & fb[index], weights, 0.0)
        curvature = _curvature_observability(
            observed[index], weights, roi_count=roi_count,
            min_support_fraction=float(curvature_min_support_fraction),
            min_lateral_contrast_rad=float(curvature_min_lateral_contrast_rad),
            min_flow_gradient_px=float(curvature_min_flow_gradient_px),
            reliable_lateral_contrast_rad=float(curvature_reliable_lateral_contrast_rad),
        )
        effective_fraction = float(weights.sum() / roi_count)
        static_fraction = float(np.count_nonzero(weights >= static_weight_threshold) / roi_count)
        fb_fraction = float(np.count_nonzero(finite_roi & fb[index]) / max(int(finite_roi.sum()), 1))
        median_flow = float(np.median(flow_magnitude[finite_roi])) if finite_roi.any() else 0.0
        reasons: list[str] = []
        if not finite_roi.any():
            reasons.append("no_finite_roi_flow")
        if fb_available and fb_fraction < 0.25:
            reasons.append("forward_backward_inconsistent")
        if effective_fraction < float(min_effective_pixel_fraction):
            reasons.append("low_static_support")
        if median_flow < float(minimum_flow_scale_px):
            reasons.append("low_flow_magnitude")
        if effective_fraction >= speed_support_threshold and median_flow >= speed_flow_threshold:
            speed_status = "usable"
        elif effective_fraction >= float(min_effective_pixel_fraction) and median_flow >= float(minimum_flow_scale_px):
            speed_status = "uncertain"
        else:
            speed_status = "abstain"
        result.append(
            {
                "interval_index": index,
                "role": role,
                "median_observed_flow_px": median_flow,
                "roi_pixel_fraction": float(finite_roi.sum() / roi_count),
                "forward_backward_fraction": fb_fraction if fb_available else None,
                "forward_backward_available": fb_available,
                "dynamic_weight_mean": (
                    float(weights[finite_roi].mean())
                    if dynamic_weights is not None and finite_roi.any()
                    else None
                ),
                "dynamic_weight_available": dynamic_weights is not None,
                "static_weight_fraction": static_fraction,
                "effective_static_pixel_fraction": effective_fraction,
                "direction_observable": not reasons,
                "speed_status": speed_status,
                "speed_observable": speed_status == "usable",
                "speed_uncertainty": (
                    "low" if speed_status == "usable" else
                    "high" if speed_status == "uncertain" else "undefined"
                ),
                **curvature,
                "status": "good" if not reasons else ";".join(reasons),
            }
        )
    _attach_temporal_curvature_observability(
        result, observed, roi, dynamic_weights,
        minimum_flow_scale_px=float(minimum_flow_scale_px),
    )
    return result


def polygon_mask(
    height: int, width: int, vertices_normalized: list[list[float]]
) -> np.ndarray:
    vertices = np.asarray(vertices_normalized, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError("mask polygon needs at least three [x,y] vertices")
    yy, xx = np.indices((height, width), dtype=np.float64)
    px = xx / max(width - 1, 1)
    py = yy / max(height - 1, 1)
    inside = np.zeros((height, width), dtype=bool)
    previous = vertices[-1]
    for current in vertices:
        x0, y0 = previous
        x1, y1 = current
        crosses = (y0 > py) != (y1 > py)
        intersection_x = (x1 - x0) * (py - y0) / (y1 - y0 + 1e-12) + x0
        inside ^= crosses & (px < intersection_x)
        previous = current
    return inside


def score_candidate(
    *,
    candidate_id: str,
    trajectory: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    observed_flows: np.ndarray,
    roi_mask: np.ndarray,
    consistency_masks: np.ndarray | None,
    min_valid_pixels: int,
    minimum_flow_scale_px: float,
    predicted_flows: np.ndarray | None = None,
    predicted_validity: np.ndarray | None = None,
    dynamic_weights: np.ndarray | None = None,
    dynamic_weight_floor: float = 0.05,
    common_geometry_masks: np.ndarray | None = None,
    energy_metric: str = "normalized_median_epe",
) -> CandidateScore:
    height, width = observed_flows.shape[1:3]
    if predicted_flows is None:
        predicted_flows, geometry_validity = predict_candidate_flows(
            trajectory=trajectory,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            height=height,
            width=width,
        )
    else:
        predicted_flows = np.asarray(predicted_flows, dtype=np.float32)
        geometry_validity = (
            np.isfinite(predicted_flows).all(axis=-1)
            if predicted_validity is None
            else np.asarray(predicted_validity, dtype=bool)
        )
        if geometry_validity.shape != predicted_flows.shape[:-1]:
            raise ValueError("predicted_validity shape does not match predicted_flows")
    if len(predicted_flows) != len(observed_flows):
        raise ValueError("trajectory knots must match the number of optical-flow intervals")
    intervals: list[dict[str, float]] = []
    for index, (predicted, geometry_valid, observed) in enumerate(
        zip(predicted_flows, geometry_validity, observed_flows)
    ):
        endpoint_error = np.linalg.norm(observed - predicted, axis=2)
        mask = roi_mask & geometry_valid & np.isfinite(endpoint_error)
        if common_geometry_masks is not None:
            mask &= np.asarray(common_geometry_masks[index], dtype=bool)
        if consistency_masks is not None:
            mask &= consistency_masks[index]
        if dynamic_weights is None:
            weights = mask.astype(np.float32)
        else:
            weights = np.where(mask, np.asarray(dynamic_weights[index], dtype=np.float32), 0.0)
        effective_count = float(weights.sum())
        if effective_count < float(min_valid_pixels):
            raise ValueError(
                f"candidate {candidate_id} interval {index} has only {effective_count:.1f} effective pixels"
            )
        valid_weights = weights[weights >= float(dynamic_weight_floor)]
        if valid_weights.size < max(10, int(min_valid_pixels // 2)):
            raise ValueError(f"candidate {candidate_id} interval {index} has too few weighted pixels")
        errors = endpoint_error[weights > 0.0]
        error_weights = weights[weights > 0.0]
        observed_scale_all = np.linalg.norm(observed, axis=2)
        observed_scale = observed_scale_all[weights > 0.0]
        median_epe = _weighted_quantile(errors, error_weights, 0.5)
        p75_epe = _weighted_quantile(errors, error_weights, 0.75)
        median_flow = _weighted_quantile(observed_scale, error_weights, 0.5)
        intervals.append(
            {
                "median_epe_px": median_epe,
                "p75_epe_px": p75_epe,
                "median_observed_flow_px": median_flow,
                "normalized_median_epe": median_epe
                / max(median_flow, float(minimum_flow_scale_px)),
                "valid_pixel_fraction": float((weights > 0.0).sum() / mask.size),
                "effective_static_pixel_fraction": float(effective_count / mask.size),
                "dynamic_weight_mean": float(weights[mask].mean()) if mask.any() else 0.0,
            }
        )
    normalized = np.asarray([item["normalized_median_epe"] for item in intervals])
    median_epe = np.asarray([item["median_epe_px"] for item in intervals])
    p75_epe = np.asarray([item["p75_epe_px"] for item in intervals])
    if energy_metric == "normalized_median_epe":
        energy = float(np.median(normalized))
    elif energy_metric == "median_epe_px":
        energy = float(np.median(median_epe))
    else:
        raise ValueError(f"unsupported score.energy_metric: {energy_metric}")
    return CandidateScore(
        candidate_id=str(candidate_id),
        energy=energy,
        median_epe_px=float(np.median(median_epe)),
        p75_epe_px=float(np.median(p75_epe)),
        min_valid_pixel_fraction=float(
            min(item["valid_pixel_fraction"] for item in intervals)
        ),
        intervals=intervals,
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(values[np.searchsorted(cumulative, float(quantile) * cumulative[-1], side="left")])


def posterior_from_energies(
    energies: np.ndarray,
    *,
    temperature: float,
    priors: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("at least two finite candidate energies are required")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if priors is None:
        prior = np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    else:
        prior = np.asarray(priors, dtype=np.float64)
        if prior.shape != values.shape or np.any(prior <= 0.0):
            raise ValueError("candidate priors must be positive and match energies")
        prior = prior / prior.sum()
    logits = -(values - values.min()) / float(temperature) + np.log(prior)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def mass_prediction_set(probabilities: np.ndarray, target_coverage: float) -> list[int]:
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target coverage must be between zero and one")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-probabilities)
    cumulative = 0.0
    selected: list[int] = []
    for index in order:
        selected.append(int(index))
        cumulative += float(probabilities[index])
        if cumulative >= target_coverage:
            break
    return selected


def conformal_prediction_set(
    probabilities: np.ndarray, nll_threshold: float
) -> list[int]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    scores = -np.log(np.maximum(probabilities, 1e-12))
    selected = np.flatnonzero(scores <= float(nll_threshold)).astype(int).tolist()
    if not selected:
        selected = [int(np.argmax(probabilities))]
    return selected


def finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("finite calibration scores are required")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    rank = int(math.ceil((values.size + 1) * coverage))
    return float(values[min(rank, values.size) - 1])


def candidate_score_dict(score: CandidateScore) -> dict[str, Any]:
    return {
        "candidate_id": score.candidate_id,
        "energy": score.energy,
        "median_epe_px": score.median_epe_px,
        "p75_epe_px": score.p75_epe_px,
        "min_valid_pixel_fraction": score.min_valid_pixel_fraction,
        "intervals": score.intervals,
    }
