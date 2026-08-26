"""Lightweight temporal geometry helpers for image-side IAC evaluation.

The module deliberately keeps the temporal state non-trainable.  It uses
history ego motion to propagate a compact road state and uses robust quantiles
for speed/scale diagnostics.  None of the speed outputs are part of the main
trajectory compatibility score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import cv2

from .geometry import ground_plane_homography, se2_to_transform
from .road_structure import extract_road_boundaries, causal_boundary_keypoint_filter


def _adaptive_plane_homography(
    intrinsics: np.ndarray,
    next_camera_from_current: np.ndarray,
    current_camera_to_anchor: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """Homography for a road plane z = ax + by + c in the anchor ego frame."""
    normal_anchor = np.asarray([-float(params[0]), -float(params[1]), 1.0], dtype=np.float64)
    offset_anchor = float(params[2])
    pose = np.asarray(current_camera_to_anchor, dtype=np.float64)
    normal_camera = pose[:3, :3].T @ normal_anchor
    distance = float(normal_anchor @ pose[:3, 3] + offset_anchor)
    if abs(distance) < 1e-5:
        raise ValueError("adaptive road plane is too close to the camera")
    transform = np.asarray(next_camera_from_current, dtype=np.float64)
    normalized = transform[:3, :3] - transform[:3, 3:4] @ normal_camera.reshape(1, 3) / distance
    K = np.asarray(intrinsics, dtype=np.float64)
    H = K @ normalized @ np.linalg.inv(K)
    return H / H[2, 2]


def fit_causal_road_plane(
    *, observed_flows: np.ndarray, history_ego_state: np.ndarray,
    camera_to_ego: np.ndarray, intrinsics: np.ndarray, roi_mask: np.ndarray,
    consistency_masks: np.ndarray | None = None, road_masks: np.ndarray | None = None,
    fit_intervals: int = 2, max_points: int = 1200, residual_scale_px: float = 2.0,
) -> dict[str, Any]:
    """Fit a low-dimensional road-plane correction from past image motion."""
    from scipy.optimize import least_squares
    flows = np.asarray(observed_flows, dtype=np.float64)
    states = np.asarray(history_ego_state, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    camera = np.asarray(camera_to_ego, dtype=np.float64)
    roi = np.asarray(roi_mask, dtype=bool)
    if flows.ndim != 4 or flows.shape[-1] != 2 or states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 3:
        raise ValueError("invalid flow/history state shapes for adaptive plane")
    if flows.shape[0] < 1 or roi.shape != flows.shape[1:3]:
        raise ValueError("flow and roi shapes do not match")
    fb = np.ones(flows.shape[:-1], dtype=bool) if consistency_masks is None else np.asarray(consistency_masks, dtype=bool)
    if fb.shape != flows.shape[:-1]:
        raise ValueError("consistency_masks must match observed flows")
    road = None if road_masks is None else np.asarray(road_masks, dtype=bool)
    if road is not None and road.shape != flows.shape[:-1]:
        raise ValueError("road_masks must match observed flows")
    poses = [se2_to_transform(*row[:3]) @ camera for row in states]
    samples = []
    for index in range(min(int(fit_intervals), len(flows), len(poses) - 1)):
        valid = roi & fb[index] & np.isfinite(flows[index]).all(axis=-1)
        valid &= np.linalg.norm(flows[index], axis=-1) > 0.5
        if road is not None:
            valid &= road[index]
        yy, xx = np.nonzero(valid)
        if len(xx) > max_points:
            choice = np.linspace(0, len(xx) - 1, max_points).astype(int)
            xx, yy = xx[choice], yy[choice]
        if len(xx) < 20:
            continue
        points = np.stack([xx, yy], axis=1).astype(np.float64)
        vectors = flows[index, yy, xx]
        transform = np.linalg.inv(poses[index + 1]) @ poses[index]
        samples.append((points, vectors, transform, poses[index]))
    if not samples:
        return {"available": False, "reason": "insufficient_causal_static_points"}
    def project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
        homogeneous = np.c_[points, np.ones(len(points))].T
        projected = H @ homogeneous
        return (projected[:2] / np.maximum(projected[2:3], 1e-8)).T
    def residual(params: np.ndarray) -> np.ndarray:
        values = []
        for points, vectors, transform, pose in samples:
            try:
                H = _adaptive_plane_homography(K, transform, pose, params)
            except ValueError:
                return np.full(2 * sum(len(item[0]) for item in samples), 1e3)
            values.append((project(H, points) - points - vectors).reshape(-1))
        return np.concatenate(values)
    fit = least_squares(residual, np.zeros(3, dtype=np.float64),
        bounds=([-0.25, -0.25, -1.0], [0.25, 0.25, 1.0]), loss="soft_l1",
        f_scale=float(max(residual_scale_px, 1e-3)), max_nfev=60)
    return {"protocol": "causal-road-plane-calibration-v1",
        "available": bool(fit.success) and np.all(np.isfinite(fit.x)),
        "plane_params": fit.x.astype(float).tolist(), "fit_cost": float(fit.cost),
        "fit_optimality": float(fit.optimality), "num_fit_intervals": len(samples),
        "num_fit_points": int(sum(len(item[0]) for item in samples))}


def _interval(values: np.ndarray, *, spread: float | None = None) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"q05": float("nan"), "q50": float("nan"), "q95": float("nan")}
    q05, q50, q95 = np.quantile(finite, [0.05, 0.50, 0.95])
    if spread is not None and np.isfinite(spread):
        amount = float(max(spread, 0.0))
        q05, q95 = q50 - amount, q50 + amount
    return {"q05": float(q05), "q50": float(q50), "q95": float(q95)}


def _mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


@dataclass(frozen=True)
class TemporalScaleCalibrator:
    """Estimate interval-valued speed/progress from static flow and depth."""

    min_valid_fraction: float = 0.02
    min_depth_m: float = 1.0
    max_depth_m: float = 100.0
    min_flow_px: float = 0.5
    uncertainty_floor: float = 0.15

    def estimate(
        self,
        *,
        observed_flows: np.ndarray,
        depths_m: np.ndarray | None,
        static_weights: np.ndarray | None,
        intrinsics: np.ndarray,
        future_times_s: np.ndarray,
        history_ego_state: np.ndarray | None,
        history_times_s: np.ndarray | None,
    ) -> dict[str, Any]:
        flow = np.asarray(observed_flows, dtype=np.float64)
        times = np.asarray(future_times_s, dtype=np.float64)
        if flow.ndim != 4 or flow.shape[-1] != 2:
            raise ValueError("observed_flows must have shape [T,H,W,2]")
        if times.shape != (len(flow),):
            raise ValueError("future_times_s must match observed flow intervals")
        depth = None if depths_m is None else np.asarray(depths_m, dtype=np.float64)
        weights = np.ones(flow.shape[:-1], dtype=np.float64) if static_weights is None else np.asarray(static_weights, dtype=np.float64)
        if depth is not None and depth.shape != flow.shape[:-1]:
            raise ValueError("depths_m must match observed flow intervals")
        if weights.shape != flow.shape[:-1]:
            raise ValueError("static_weights must match observed flow intervals")
        K = np.asarray(intrinsics, dtype=np.float64)
        focal = float(np.mean([K[0, 0], K[1, 1]])) if K.shape == (3, 3) else float("nan")
        state = None if history_ego_state is None else np.asarray(history_ego_state, dtype=np.float64)
        hist_times = None if history_times_s is None else np.asarray(history_times_s, dtype=np.float64)
        prior_speed = float("nan")
        if state is not None and state.ndim == 2 and state.shape[0] and state.shape[1] >= 4:
            prior_speed = float(state[-1, 3])
        if not np.isfinite(prior_speed) or prior_speed < 0.0:
            prior_speed = float("nan")
        rows: list[dict[str, Any]] = []
        for index, (interval_flow, interval_weight) in enumerate(zip(flow, weights)):
            magnitude = np.linalg.norm(interval_flow, axis=-1)
            valid = np.isfinite(magnitude) & np.isfinite(interval_weight) & (interval_weight > 0.25) & (magnitude >= self.min_flow_px)
            if depth is not None:
                valid &= np.isfinite(depth[index]) & (depth[index] > self.min_depth_m) & (depth[index] < self.max_depth_m)
            count = int(valid.sum())
            fraction = float(count / max(valid.size, 1))
            raw_speed = float("nan")
            relative_mad = float("nan")
            if count and np.isfinite(focal) and focal > 1e-6 and depth is not None:
                dt = float(times[index] - (0.0 if index == 0 else times[index - 1]))
                dt = max(dt, 1e-3)
                samples = magnitude[valid] * depth[index][valid] / focal / dt
                raw_speed = float(np.median(samples))
                relative_mad = float(_mad(samples) / max(raw_speed, 1e-3))
            if np.isfinite(prior_speed) and np.isfinite(raw_speed) and raw_speed > 1e-3:
                scale = prior_speed / raw_speed
                corrected = raw_speed * scale
                scale_confidence = float(np.clip(min(fraction / 0.10, 1.0) * np.exp(-max(relative_mad, 0.0)), 0.0, 1.0))
            elif np.isfinite(prior_speed):
                scale = 1.0
                corrected = prior_speed
                scale_confidence = float(np.clip(fraction / 0.10, 0.0, 0.35))
            else:
                scale = float("nan")
                corrected = raw_speed
                scale_confidence = float(np.clip(fraction / 0.10, 0.0, 1.0))
            uncertainty = max(self.uncertainty_floor, 0.20 * max(relative_mad, 0.0) if np.isfinite(relative_mad) else 0.75)
            if fraction < self.min_valid_fraction:
                uncertainty = max(uncertainty, 1.0)
            speed_interval = _interval(np.asarray([corrected]), spread=max(abs(corrected), 0.1) * uncertainty if np.isfinite(corrected) else float("nan"))
            dt = float(times[index] - (0.0 if index == 0 else times[index - 1]))
            progress = corrected * max(dt, 0.0) if np.isfinite(corrected) else float("nan")
            progress_interval = _interval(np.asarray([progress]), spread=max(abs(progress), 0.1) * uncertainty if np.isfinite(progress) else float("nan"))
            rows.append({
                "interval_index": index,
                "valid_fraction": fraction,
                "raw_flow_depth_speed_mps": raw_speed,
                "scale_correction": scale,
                "scale_confidence": scale_confidence,
                "speed_interval_mps": speed_interval,
                "progress_interval_m": progress_interval,
                "speed_observability": float(np.clip(scale_confidence, 0.0, 1.0)),
                "speed_status": (
                    "abstain" if fraction < self.min_valid_fraction else
                    "uncalibrated" if not np.isfinite(prior_speed) else
                    "uncertain" if scale_confidence < 0.55 else "diagnostic"
                ),
            })
        return {
            "protocol": "temporal-scale-calibration-v1",
            "available": bool(rows) and any(np.isfinite(row["speed_interval_mps"]["q50"]) for row in rows),
            "history_speed_prior_mps": prior_speed,
            "history_state_available": state is not None and state.ndim == 2 and state.shape[0] > 0,
            "history_speed_prior_available": bool(np.isfinite(prior_speed)),
            # A prior alone supplies a diagnostic interval but is not a scale
            # calibration.  Calibration requires at least one valid metric
            # flow-depth measurement to compare against the prior.
            "scale_calibrated": bool(
                np.isfinite(prior_speed)
                and any(np.isfinite(row["raw_flow_depth_speed_mps"]) for row in rows)
            ),
            "rows": rows,
            "mean_scale_confidence": float(np.mean([row["scale_confidence"] for row in rows])) if rows else 0.0,
        }


@dataclass
class RoadStateFilter:
    """Recursive, non-trained ego-frame filter over sparse road geometry."""

    measurement_gain: float = 0.65
    process_noise: float = 0.02
    far_uncertainty_growth: float = 0.04
    road_half_width_m: float = 1.1
    far_support_reference_fraction: float = 0.03
    far_disagreement_weight: float = 0.0
    far_start_fraction: float = 0.62
    boundary_keypoint_filter_enabled: bool = False
    boundary_keypoint_max_jump_px: float = 28.0
    boundary_keypoint_huber_scale_px: float = 8.0

    def _measurement(self, mask: np.ndarray, boundary_override: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        value = np.asarray(mask, dtype=bool)
        # Segmentation can return disconnected road-like islands (sidewalks,
        # parking patches, or road fragments behind actors).  Keep the
        # component that actually reaches the bottom of the camera view and is
        # closest to the ego center before fitting boundaries.
        binary = cv2.morphologyEx(value.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        if count > 1:
            h, w = value.shape
            bottom = labels[int(max(0, h * 0.90)) :]
            candidates = []
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < 32:
                    continue
                bottom_hits = int(np.count_nonzero(bottom == label))
                cx = float(centroids[label][0])
                score = 4.0 * bottom_hits + 0.01 * area - abs(cx - 0.5 * w)
                candidates.append((score, label))
            if candidates:
                keep = max(candidates)[1]
                value = labels == keep
        boundaries = boundary_override or extract_road_boundaries(value, row_step=4, polynomial_degree=2)
        if not boundaries.get("valid"):
            return np.full(6, np.nan, dtype=np.float64), {"valid": False, "boundaries": boundaries}
        rows = np.asarray(boundaries["rows"], dtype=np.float64)
        height = max(float(mask.shape[0] - 1), 1.0)
        yn = rows / height
        left = np.polyval(np.asarray(boundaries["left_coeff"], dtype=np.float64), yn)
        right = np.polyval(np.asarray(boundaries["right_coeff"], dtype=np.float64), yn)
        center = 0.5 * (left + right)
        center_coeff = np.polyfit(yn, center, min(3, len(center) - 1))
        near_index = int(np.argmax(rows))
        far_index = int(np.argmin(rows))
        # Use a robust image-row slope for heading.  High-degree derivatives
        # in normalized coordinates are ill-conditioned on sparse masks.
        slope_px = float(np.polyfit(rows, center, 1)[0]) if len(rows) >= 2 else 0.0
        derivative = float(np.clip(slope_px, -3.0, 3.0))
        if len(rows) >= 3:
            quadratic = np.polyfit(rows, center, 2)
            # For p(y)=a*y^2+b*y+c, p''=2a.  Keep this in image-space
            # units and normalize conservatively for the diagnostic state.
            curvature_px = float(2.0 * quadratic[0])
        else:
            curvature_px = 0.0
        second = float(np.clip(curvature_px * 1e-3, -0.2, 0.2))
        center_offset = (float(center[near_index]) - 0.5 * mask.shape[1]) / max(float(mask.shape[1]), 1.0)
        state = np.asarray([
            center_offset,
            float(np.arctan(derivative)),
            second,
            0.0,
            float((center[near_index] - left[near_index]) / max(mask.shape[1], 1)),
            float((right[near_index] - center[near_index]) / max(mask.shape[1], 1)),
        ], dtype=np.float64)
        info = {"valid": True, "boundaries": boundaries, "centerline_coeff_norm": center_coeff.tolist(), "confidence": float(boundaries.get("confidence", 0.0))}
        return state, info

    def update(
        self,
        road_masks: np.ndarray,
        *,
        observed_flows: np.ndarray | None = None,
        ego_states: np.ndarray | None = None,
        future_times_s: np.ndarray | None = None,
        far_disagreement: np.ndarray | None = None,
    ) -> dict[str, Any]:
        masks = np.asarray(road_masks, dtype=bool)
        if masks.ndim != 3 or not len(masks):
            raise ValueError("road_masks must have shape [T,H,W]")
        states: list[dict[str, Any]] = []
        previous = np.full(6, np.nan, dtype=np.float64)
        raw_values: list[np.ndarray] = []
        filtered_values: list[np.ndarray] = []
        boundary_sequence = None
        boundary_filter_diag = None
        if self.boundary_keypoint_filter_enabled:
            boundary_sequence, boundary_filter_diag = causal_boundary_keypoint_filter(
                masks,
                observed_flows=observed_flows,
                max_jump_px=self.boundary_keypoint_max_jump_px,
                huber_scale_px=self.boundary_keypoint_huber_scale_px,
            )
        state_array = None if ego_states is None else np.asarray(ego_states, dtype=np.float64)
        disagreement_array = None if far_disagreement is None else np.asarray(far_disagreement, dtype=np.float64)
        if disagreement_array is not None and disagreement_array.shape != (len(masks),):
            raise ValueError("far_disagreement must match road_masks intervals")
        for index, mask in enumerate(masks):
            override = None if boundary_sequence is None else boundary_sequence[index]
            measurement, info = self._measurement(mask, override if override and override.get("valid") else None)
            raw_values.append(measurement.copy())
            predicted = previous.copy()
            if np.isfinite(previous).all() and state_array is not None and state_array.ndim == 2 and len(state_array) > index and state_array.shape[1] >= 2:
                predicted[0] -= float(state_array[index, 1]) / max(10.0, self.road_half_width_m * 4.0)
                if state_array.shape[1] >= 3:
                    predicted[1] -= float(state_array[index, 2])
            gain = float(np.clip(self.measurement_gain * float(info.get("confidence", 0.0)), 0.0, 1.0)) if np.isfinite(measurement).all() else 0.0
            if not np.isfinite(previous).all():
                filtered = measurement
            elif gain > 0.0:
                filtered = predicted + gain * (measurement - predicted)
            else:
                filtered = predicted
            if np.isfinite(filtered).all():
                previous = filtered
            filtered_values.append(filtered.copy())
            far_cut = int(mask.shape[0] * float(np.clip(self.far_start_fraction, 0.0, 1.0)))
            far_support_fraction = float(np.mean(mask[:far_cut])) if far_cut > 0 else 0.0
            far_support_observability = float(np.clip(
                far_support_fraction / max(self.far_support_reference_fraction, 1e-6), 0.0, 1.0
            ))
            uncertainty = float(
                self.process_noise
                + self.far_uncertainty_growth * index
                + (1.0 - far_support_observability) * 0.20
                + self.far_disagreement_weight * (
                    float(np.clip(disagreement_array[index], 0.0, 1.0))
                    if disagreement_array is not None else 0.0
                )
            )
            states.append({
                "interval_index": index,
                "valid": bool(np.isfinite(filtered).all()),
                "center_offset_norm": float(filtered[0]) if np.isfinite(filtered[0]) else None,
                "heading_error_rad": float(filtered[1]) if np.isfinite(filtered[1]) else None,
                "curvature_norm": float(filtered[2]) if np.isfinite(filtered[2]) else None,
                "curvature_rate_norm": float(filtered[3]) if np.isfinite(filtered[3]) else None,
                "left_boundary_offset_norm": float(filtered[4]) if np.isfinite(filtered[4]) else None,
                "right_boundary_offset_norm": float(filtered[5]) if np.isfinite(filtered[5]) else None,
                "lateral_uncertainty_m": float(uncertainty * self.road_half_width_m),
                "far_range_observability": float(np.clip(float(info.get("confidence", 0.0)) * far_support_observability * np.exp(-self.far_uncertainty_growth * index), 0.0, 1.0)),
                "far_support_fraction": far_support_fraction,
                "far_support_observability": far_support_observability,
                "far_mask_disagreement": (
                    float(np.clip(disagreement_array[index], 0.0, 1.0))
                    if disagreement_array is not None else None
                ),
                "measurement": info,
            })
        raw = np.asarray(raw_values, dtype=np.float64)
        filtered = np.asarray(filtered_values, dtype=np.float64)
        jitter = np.nanmedian(np.abs(np.diff(filtered[:, 0]))) if len(filtered) > 1 else 0.0
        raw_jitter = np.nanmedian(np.abs(np.diff(raw[:, 0]))) if len(raw) > 1 else 0.0
        return {
            "protocol": "road-state-filter-v1",
            "available": bool(states) and any(item["valid"] for item in states),
            "states": states,
            "temporal_jitter_diagnostics": {
                "raw_center_jitter_norm": float(raw_jitter) if np.isfinite(raw_jitter) else None,
                "filtered_center_jitter_norm": float(jitter) if np.isfinite(jitter) else None,
                "jitter_reduction_ratio": float(1.0 - jitter / max(raw_jitter, 1e-8)) if np.isfinite(jitter) and np.isfinite(raw_jitter) and raw_jitter > 1e-8 else None,
                "boundary_keypoint_filter": boundary_filter_diag,
            },
        }


@dataclass
class HomographyBoundaryPropagator:
    """Propagate sparse road boundaries with a planar ego-motion prior.

    Only history ego state is used to extrapolate a constant-velocity prior
    after the anchor.  Current-frame segmentation remains a measurement; the
    warped boundary is a confidence-weighted prior and is rejected when its
    reprojection becomes implausible.
    """

    keypoints_per_side: int = 10
    process_noise: float = 0.025
    propagation_decay: float = 0.92
    max_reprojection_residual_px: float = 32.0
    road_half_width_m: float = 1.1
    min_current_confidence_for_propagation: float = 0.45
    min_far_support_fraction: float = 0.01
    far_support_reference_fraction: float = 0.03
    propagation_method: str = "homography"
    propagate_far_missing: bool = True
    global_flow_max_points: int = 5000
    global_flow_ransac_threshold_px: float = 2.5

    @staticmethod
    def _sample_boundary(boundaries: dict[str, Any], side: str, count: int) -> np.ndarray:
        if not boundaries.get("valid"):
            return np.empty((0, 2), dtype=np.float64)
        rows = np.asarray(boundaries.get("rows", []), dtype=np.float64)
        values = np.asarray(boundaries.get(f"{side}_x", []), dtype=np.float64)
        if len(rows) < 2 or len(values) != len(rows):
            return np.empty((0, 2), dtype=np.float64)
        query = np.linspace(float(rows.min()), float(rows.max()), max(int(count), 2))
        return np.stack([np.interp(query, rows, values), query], axis=1)

    @staticmethod
    def _warp_points(points: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(points) == 0:
            return points.copy(), np.zeros(0, dtype=bool)
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1).T
        projected = np.asarray(homography, dtype=np.float64) @ homogeneous
        denominator = projected[2]
        valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
        xy = np.zeros((len(points), 2), dtype=np.float64)
        xy[:, 0] = projected[0] / np.where(valid, denominator, 1.0)
        xy[:, 1] = projected[1] / np.where(valid, denominator, 1.0)
        valid &= np.isfinite(xy).all(axis=1)
        return xy, valid

    @staticmethod
    def _future_poses(
        history_ego_state: np.ndarray | None,
        history_times_s: np.ndarray | None,
        future_times_s: np.ndarray,
    ) -> list[np.ndarray]:
        state = None if history_ego_state is None else np.asarray(history_ego_state, dtype=np.float64)
        times = np.asarray(future_times_s, dtype=np.float64)
        if state is None or state.ndim != 2 or len(state) == 0 or state.shape[1] < 4:
            return [np.eye(4, dtype=np.float64) for _ in range(len(times) + 1)]
        anchor = state[-1]
        speed = float(anchor[3]) if np.isfinite(anchor[3]) else 0.0
        yaw_rate = float(anchor[4]) if state.shape[1] >= 5 and np.isfinite(anchor[4]) else 0.0
        previous_time = 0.0
        x, y, yaw = 0.0, 0.0, float(anchor[2]) if np.isfinite(anchor[2]) else 0.0
        poses = [se2_to_transform(x, y, yaw)]
        for time_s in times:
            dt = max(float(time_s - previous_time), 1e-3)
            if abs(yaw_rate) < 1e-5:
                x += speed * np.cos(yaw) * dt
                y += speed * np.sin(yaw) * dt
            else:
                radius = speed / yaw_rate
                next_yaw = yaw + yaw_rate * dt
                x += radius * (np.sin(next_yaw) - np.sin(yaw))
                y += radius * (-np.cos(next_yaw) + np.cos(yaw))
                yaw = next_yaw
            poses.append(se2_to_transform(x, y, yaw))
            previous_time = float(time_s)
        return poses

    @staticmethod
    def _fuse(
        current: np.ndarray,
        propagated: np.ndarray,
        current_confidence: float,
        propagated_confidence: float,
    ) -> np.ndarray:
        if len(propagated) == 0:
            return current
        if len(current) == 0:
            return propagated
        rows = propagated[:, 1]
        current_x = np.interp(rows, current[:, 1], current[:, 0])
        denominator = max(current_confidence + propagated_confidence, 1e-6)
        # Near-field mask pixels are generally well observed.  Let propagation
        # contribute mainly to the far rows where segmentation is sparse.
        far = np.clip(1.0 - rows / max(float(rows.max()), 1.0), 0.0, 1.0)
        alpha = np.clip(current_confidence / denominator + 0.55 * (1.0 - far), 0.0, 1.0)
        observed_range = (rows >= float(np.min(current[:, 1]))) & (rows <= float(np.max(current[:, 1])))
        # Never replace an actually observed near-range boundary.  Propagation
        # is only allowed to fill rows outside the current support range.
        alpha = np.where(observed_range, 1.0, alpha)
        fused_x = alpha * current_x + (1.0 - alpha) * propagated[:, 0]
        return np.stack([fused_x, rows], axis=1)

    @staticmethod
    def _flow_homography(
        points: np.ndarray,
        flow: np.ndarray,
        static_weights: np.ndarray | None = None,
        reprojection_threshold_px: float = 8.0,
    ) -> tuple[np.ndarray | None, float, int]:
        """Estimate a local road-plane homography from boundary flow samples."""
        if len(points) < 4:
            return None, 0.0, 0
        value = np.asarray(flow, dtype=np.float64)
        height, width = value.shape[:2]
        src: list[list[float]] = []
        dst: list[list[float]] = []
        weights = None if static_weights is None else np.asarray(static_weights, dtype=np.float64)
        for x, y in points:
            ix, iy = int(round(x)), int(round(y))
            if not (1 <= ix < width - 1 and 1 <= iy < height - 1):
                continue
            patch = value[iy - 1 : iy + 2, ix - 1 : ix + 2]
            finite = np.isfinite(patch).all(axis=-1)
            if not finite.any():
                continue
            median_flow = np.median(patch[finite], axis=0)
            if weights is not None and float(weights[iy, ix]) <= 0.0:
                continue
            src.append([float(x), float(y)])
            dst.append([float(x + median_flow[0]), float(y + median_flow[1])])
        if len(src) < 4:
            return None, 0.0, len(src)
        matrix, inlier = cv2.findHomography(
            np.asarray(src, dtype=np.float32), np.asarray(dst, dtype=np.float32),
            cv2.RANSAC, float(reprojection_threshold_px), maxIters=100,
        )
        if matrix is None or inlier is None:
            return None, 0.0, len(src)
        ratio = float(np.mean(inlier.astype(bool)))
        return matrix.astype(np.float64), ratio, len(src)

    @staticmethod
    def _flow_warp_points(
        points: np.ndarray,
        flow: np.ndarray,
        static_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp boundary points using local RAFT flow, without a global plane.

        Far road boundaries violate a single homography in practice because of
        depth variation, mask quantization, and imperfect calibration.  A
        local median flow keeps the measured per-point motion and only rejects
        samples with invalid/low static support.
        """
        value = np.asarray(flow, dtype=np.float64)
        if value.ndim != 3 or value.shape[-1] != 2:
            raise ValueError("flow must have shape [H,W,2]")
        if len(points) == 0:
            return points.copy(), np.zeros(0, dtype=bool)
        height, width = value.shape[:2]
        weights = None if static_weights is None else np.asarray(static_weights, dtype=np.float64)
        warped = np.full_like(np.asarray(points, dtype=np.float64), np.nan)
        valid = np.zeros(len(points), dtype=bool)
        for index, (x, y) in enumerate(np.asarray(points, dtype=np.float64)):
            ix, iy = int(round(x)), int(round(y))
            if not (1 <= ix < width - 1 and 1 <= iy < height - 1):
                continue
            patch = value[iy - 1 : iy + 2, ix - 1 : ix + 2]
            finite = np.isfinite(patch).all(axis=-1)
            if not finite.any():
                continue
            if weights is not None:
                local_weight = weights[iy - 1 : iy + 2, ix - 1 : ix + 2]
                finite &= np.isfinite(local_weight) & (local_weight > 0.0)
                if not finite.any():
                    continue
            displacement = np.median(patch[finite], axis=0)
            endpoint = np.asarray([x, y], dtype=np.float64) + displacement
            if not np.isfinite(endpoint).all():
                continue
            if not (-width <= endpoint[0] <= 2 * width and -height <= endpoint[1] <= 2 * height):
                continue
            warped[index] = endpoint
            valid[index] = True
        return warped, valid

    @staticmethod
    def _global_flow_homography(
        flow: np.ndarray,
        static_weights: np.ndarray | None,
        *,
        support_mask: np.ndarray | None = None,
        max_points: int,
        reprojection_threshold_px: float,
    ) -> tuple[np.ndarray | None, float, int]:
        """Fit a robust image-to-image homography to the dominant flow field."""
        value = np.asarray(flow, dtype=np.float64)
        if value.ndim != 3 or value.shape[-1] != 2:
            raise ValueError("flow must have shape [H,W,2]")
        valid = np.isfinite(value).all(axis=-1)
        if static_weights is not None:
            weights = np.asarray(static_weights, dtype=np.float64)
            if weights.shape != value.shape[:2]:
                raise ValueError("static_weights must match flow spatial shape")
            valid &= np.isfinite(weights) & (weights > 0.25)
        if support_mask is not None:
            road = np.asarray(support_mask, dtype=bool)
            if road.shape != value.shape[:2]:
                raise ValueError("support_mask must match flow spatial shape")
            valid &= road
        yy, xx = np.nonzero(valid)
        if len(xx) < 4:
            return None, 0.0, int(len(xx))
        stride = max(1, int(np.ceil(len(xx) / max(int(max_points), 4))))
        xx, yy = xx[::stride], yy[::stride]
        source = np.stack([xx, yy], axis=1).astype(np.float32)
        destination = source + value[yy, xx].astype(np.float32)
        matrix, inlier = cv2.findHomography(
            source, destination, cv2.RANSAC,
            float(reprojection_threshold_px), maxIters=3000,
            confidence=0.995,
        )
        if matrix is None or inlier is None:
            return None, 0.0, int(len(source))
        return matrix.astype(np.float64), float(np.mean(inlier.astype(bool))), int(len(source))

    def update(
        self,
        road_masks: np.ndarray,
        *,
        intrinsics: np.ndarray,
        camera_to_ego: np.ndarray,
        observed_flows: np.ndarray | None = None,
        static_weights: np.ndarray | None = None,
        history_ego_state: np.ndarray | None = None,
        history_times_s: np.ndarray | None = None,
        future_times_s: np.ndarray | None = None,
    ) -> dict[str, Any]:
        masks = np.asarray(road_masks, dtype=bool)
        if masks.ndim != 3 or not len(masks):
            raise ValueError("road_masks must have shape [T,H,W]")
        times = np.arange(len(masks), dtype=np.float64) + 1.0 if future_times_s is None else np.asarray(future_times_s, dtype=np.float64)
        if times.shape != (len(masks),):
            raise ValueError("future_times_s must match road_masks")
        K = np.asarray(intrinsics, dtype=np.float64)
        camera = np.asarray(camera_to_ego, dtype=np.float64)
        poses = self._future_poses(history_ego_state, history_times_s, times)
        previous: dict[str, np.ndarray] = {"left": np.empty((0, 2)), "right": np.empty((0, 2))}
        previous_confidence = 0.0
        states: list[dict[str, Any]] = []
        propagated_rows: list[dict[str, Any]] = []
        for index, mask in enumerate(masks):
            current = extract_road_boundaries(mask, row_step=4, polynomial_degree=2)
            current_confidence = float(current.get("confidence", 0.0)) if current.get("valid") else 0.0
            far_cut = int(mask.shape[0] * 0.62)
            far_support_fraction = float(np.mean(mask[:far_cut])) if far_cut > 0 else 0.0
            current_points = {
                side: self._sample_boundary(current, side, self.keypoints_per_side)
                for side in ("left", "right")
            }
            homography = None
            flow_inlier_ratio = 0.0
            flow_point_count = 0
            propagated = {"left": np.empty((0, 2)), "right": np.empty((0, 2))}
            valid_fraction = 0.0
            direct_flow_warp = False
            if index > 0 and previous_confidence > 0.0:
                try:
                    if observed_flows is not None and index - 1 < len(observed_flows):
                        flow_weights = None if static_weights is None else np.asarray(static_weights[index - 1])
                        all_previous = np.concatenate([previous["left"], previous["right"]], axis=0)
                        if self.propagation_method == "global_flow_homography":
                            homography, flow_inlier_ratio, flow_point_count = self._global_flow_homography(
                                observed_flows[index - 1], flow_weights,
                                support_mask=masks[index - 1],
                                max_points=self.global_flow_max_points,
                                reprojection_threshold_px=self.global_flow_ransac_threshold_px,
                            )
                            if flow_inlier_ratio < 0.50 or flow_point_count < 20:
                                homography = None
                        elif self.propagation_method == "flow_warp":
                            direct_flow_warp = True
                            split = len(previous["left"])
                            warped, valid = self._flow_warp_points(all_previous, observed_flows[index - 1], flow_weights)
                            propagated["left"] = warped[:split][valid[:split]]
                            propagated["right"] = warped[split:][valid[split:]]
                            valid_fraction = float(np.mean(valid)) if len(valid) else 0.0
                            flow_point_count = int(valid.sum())
                            flow_inlier_ratio = valid_fraction
                        else:
                            homography, flow_inlier_ratio, flow_point_count = self._flow_homography(
                                all_previous, observed_flows[index - 1], flow_weights,
                                reprojection_threshold_px=self.max_reprojection_residual_px,
                            )
                            if flow_inlier_ratio < 0.70 or flow_point_count < 6:
                                homography = None
                    else:
                        current_camera_to_anchor = poses[index] @ camera
                        next_camera_from_current = np.linalg.inv(poses[index + 1] @ camera) @ current_camera_to_anchor
                        homography = ground_plane_homography(K, next_camera_from_current, current_camera_to_anchor)
                        flow_inlier_ratio, flow_point_count = 0.0, 0
                    if not direct_flow_warp:
                        all_valid = []
                        for side in ("left", "right"):
                            warped, valid = self._warp_points(previous[side], homography)
                            propagated[side] = warped[valid]
                            all_valid.append(valid)
                        valid_fraction = float(np.mean(np.concatenate(all_valid))) if all_valid and sum(len(item) for item in all_valid) else 0.0
                except (ValueError, np.linalg.LinAlgError, FloatingPointError, cv2.error):
                    homography = None
                    flow_inlier_ratio, flow_point_count = 0.0, 0
            missing_current = not current.get("valid", False)
            missing_far = far_support_fraction < self.min_far_support_fraction
            propagate_condition = missing_current or (missing_far and self.propagate_far_missing)
            use_propagated = (homography is not None or direct_flow_warp) and propagate_condition
            fused = {
                side: self._fuse(
                    current_points[side],
                    propagated[side] if use_propagated else np.empty((0, 2)),
                    current_confidence,
                    previous_confidence * self.propagation_decay,
                )
                for side in ("left", "right")
            }
            # Reconstruct a compact state from the fused sparse centerline.
            if len(fused["left"]) >= 2 and len(fused["right"]) >= 2:
                query = np.linspace(max(fused["left"][:, 1].min(), fused["right"][:, 1].min()), min(fused["left"][:, 1].max(), fused["right"][:, 1].max()), self.keypoints_per_side)
                left_x = np.interp(query, fused["left"][:, 1], fused["left"][:, 0])
                right_x = np.interp(query, fused["right"][:, 1], fused["right"][:, 0])
                center_x = 0.5 * (left_x + right_x)
                slope = float(np.polyfit(query, center_x, 1)[0]) if len(query) >= 2 else 0.0
                curvature = float(2.0 * np.polyfit(query, center_x, 2)[0] * 1e-3) if len(query) >= 3 else 0.0
                state = {
                    "center_offset_norm": float((center_x[-1] - 0.5 * mask.shape[1]) / max(mask.shape[1], 1)),
                    "heading_error_rad": float(np.arctan(np.clip(slope, -3.0, 3.0))),
                    "curvature_norm": float(np.clip(curvature, -0.2, 0.2)),
                    "left_boundary_offset_norm": float((center_x[-1] - left_x[-1]) / max(mask.shape[1], 1)),
                    "right_boundary_offset_norm": float((right_x[-1] - center_x[-1]) / max(mask.shape[1], 1)),
                    "valid": True,
                }
            else:
                state = {"valid": False, "center_offset_norm": None, "heading_error_rad": None, "curvature_norm": None, "left_boundary_offset_norm": None, "right_boundary_offset_norm": None}
            confidence = float(np.clip(max(current_confidence, previous_confidence * self.propagation_decay) * (0.75 + 0.25 * valid_fraction), 0.0, 1.0))
            far_support_observability = float(np.clip(
                far_support_fraction / max(self.far_support_reference_fraction, 1e-6), 0.0, 1.0
            ))
            uncertainty = float(
                self.process_noise
                + (1.0 - confidence) * 0.20
                + (1.0 - far_support_observability) * 0.20
                + index * self.process_noise
            )
            state.update({
                "interval_index": index,
                "confidence": confidence,
                "lateral_uncertainty_m": uncertainty * self.road_half_width_m,
                "far_range_observability": float(np.clip(
                    confidence * far_support_observability * np.exp(-self.process_noise * index), 0.0, 1.0
                )),
                "far_support_fraction": far_support_fraction,
                "far_support_observability": far_support_observability,
                "left_boundary": fused["left"].tolist(),
                "right_boundary": fused["right"].tolist(),
            })
            states.append(state)
            propagated_rows.append({
                "interval_index": index,
                "current_confidence": current_confidence,
                "propagated_confidence": previous_confidence * self.propagation_decay,
                "homography_used": homography is not None and not direct_flow_warp,
                "flow_warp_used": bool(direct_flow_warp),
                "propagation_applied": use_propagated,
                "propagation_reason": "invalid_current" if missing_current else ("far_support_missing" if missing_far else "current_observation_kept"),
                "far_support_fraction": far_support_fraction,
                "valid_fraction": valid_fraction,
                "flow_homography_inlier_ratio": float(flow_inlier_ratio),
                "flow_homography_point_count": int(flow_point_count),
                "fused_keypoints": int(len(fused["left"]) + len(fused["right"])),
            })
            previous = fused
            previous_confidence = confidence
        centers = np.asarray([item.get("center_offset_norm", np.nan) for item in states], dtype=np.float64)
        jitter = float(np.nanmedian(np.abs(np.diff(centers)))) if len(centers) > 1 else 0.0
        return {
            "protocol": f"{self.propagation_method}-boundary-propagation-v2",
            "available": bool(states) and any(item.get("valid", False) for item in states),
            "states": states,
            "propagation": propagated_rows,
            "temporal_jitter_diagnostics": {"fused_center_jitter_norm": jitter if np.isfinite(jitter) else None},
            "uses_realized_future_state": False,
        }
