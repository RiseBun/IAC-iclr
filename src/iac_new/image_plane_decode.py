"""Calibration-free, candidate-blind motion evidence from optical flow."""

from __future__ import annotations

from typing import Any

import numpy as np


def _fit_affine(flow: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float, int]:
    height, width = flow.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    x = 2.0 * (xx / max(width - 1, 1)) - 1.0
    y = 2.0 * (yy / max(height - 1, 1)) - 1.0
    design = np.stack([np.ones_like(x), x, y], axis=-1).reshape(-1, 3)
    values = np.asarray(flow, dtype=np.float64).reshape(-1, 2)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(weights).reshape(-1) & (weights.reshape(-1) > 0.0)
    valid &= np.linalg.norm(values, axis=1) > 0.05
    if int(valid.sum()) < 16:
        return np.full((2, 3), np.nan), 0.0, int(valid.sum())
    w = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64).reshape(-1)[valid], 0.0, None))
    a = design[valid] * w[:, None]
    b = values[valid] * w[:, None]
    coef, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    residual = np.linalg.norm(a @ coef - b, axis=1)
    scale = max(float(np.median(np.linalg.norm(b, axis=1))), 1.0)
    quality = float(np.exp(-np.median(residual) / scale))
    return coef.T, quality, int(valid.sum())


def decode_image_plane_motion(
    observed_flows: np.ndarray,
    roi_mask: np.ndarray,
    *,
    consistency_masks: np.ndarray | None = None,
    dynamic_weights: np.ndarray | None = None,
    future_times_s: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a continuous normalized trajectory proxy, without calibration.

    The proxy has no metric units. ``forward_proxy`` is flow divergence,
    ``lateral_proxy`` is horizontal translation after removing the fitted
    rotational component, and ``yaw_proxy`` is the affine rotational term.
    This is suitable for direction/shape and paired intervention tests, but
    not for absolute speed or meters.
    """
    flows = np.asarray(observed_flows, dtype=np.float32)
    if flows.ndim != 4 or flows.shape[-1] != 2:
        raise ValueError("observed_flows must have shape [T,H,W,2]")
    roi = np.asarray(roi_mask, dtype=bool)
    if roi.shape != flows.shape[1:3]:
        raise ValueError("roi_mask must match flow spatial dimensions")
    masks = np.ones(flows.shape[:-1], dtype=bool) if consistency_masks is None else np.asarray(consistency_masks, dtype=bool)
    weights = np.ones(flows.shape[:-1], dtype=np.float32) if dynamic_weights is None else np.asarray(dynamic_weights, dtype=np.float32)
    if masks.shape != weights.shape or masks.shape != flows.shape[:-1]:
        raise ValueError("consistency_masks and dynamic_weights must match flow")
    weights = np.where(masks & roi[None, ...], np.maximum(weights, 0.0), 0.0)
    times = np.arange(len(flows), dtype=np.float64) + 1.0 if future_times_s is None else np.asarray(future_times_s, dtype=np.float64)
    if times.shape != (len(flows),) or np.any(np.diff(times) <= 0.0):
        raise ValueError("future_times_s must be strictly increasing and match flow intervals")
    rows = []
    for index, (flow, weight) in enumerate(zip(flows, weights)):
        coeff, quality, count = _fit_affine(flow, weight)
        if not np.isfinite(coeff).all():
            rows.append({"interval_index": index, "support_fraction": 0.0, "quality": 0.0, "valid_pixels": count, "forward_proxy": 0.0, "lateral_proxy": 0.0, "yaw_proxy": 0.0, "divergence_proxy": 0.0})
            continue
        # u=a0+a1*x+a2*y, v=b0+b1*x+b2*y. Curl is (b1-a2)/2;
        # subtract its contribution from horizontal translation at the road band.
        a0, a1, a2 = coeff[0]
        b0, b1, b2 = coeff[1]
        yaw = 0.5 * (b1 - a2)
        divergence = a1 + b2
        road_y = 0.55
        lateral = a0 + a2 * road_y - yaw * road_y
        forward = max(float(divergence), 0.0)
        rows.append({
            "interval_index": index,
            "support_fraction": float(np.mean(weight > 0.0)),
            "quality": quality,
            "valid_pixels": count,
            "forward_proxy": forward,
            "lateral_proxy": float(lateral),
            "yaw_proxy": float(yaw),
            "divergence_proxy": float(divergence),
            "affine_coefficients": coeff.tolist(),
        })
    dt = np.diff(np.concatenate([[0.0], times]))
    forward = np.asarray([row["forward_proxy"] for row in rows], dtype=np.float64)
    lateral = np.asarray([row["lateral_proxy"] for row in rows], dtype=np.float64)
    yaw = np.asarray([row["yaw_proxy"] for row in rows], dtype=np.float64)
    trajectory = np.zeros((len(rows), 3), dtype=np.float64)
    for index in range(len(rows)):
        previous = trajectory[index - 1] if index else np.zeros(3, dtype=np.float64)
        scale = max(forward[index], 1e-4)
        trajectory[index] = [
            previous[0] + scale * dt[index],
            previous[1] + lateral[index] * dt[index],
            previous[2] + yaw[index] * dt[index],
        ]
    final_x = max(abs(float(trajectory[-1, 0])), 1e-6)
    normalized = trajectory.copy()
    normalized[:, 0] /= final_x
    normalized[:, 1] /= final_x
    normalized[:, 2] /= max(abs(float(trajectory[-1, 2])), 1.0)
    return {
        "protocol": "candidate-blind-image-plane-motion-v1",
        "units": "normalized_image_plane_proxy",
        "trajectory": normalized.tolist(),
        "raw_trajectory_proxy": trajectory.tolist(),
        "intervals": rows,
        "forward_proxy_total": float(np.sum(forward * dt)),
        "support_fraction": float(np.mean(weights > 0.0)),
        "quality": float(np.average([row["quality"] for row in rows], weights=np.maximum(dt, 1e-6))),
        "speed_absolute_observable": False,
        "calibration_status": "missing",
    }


def compare_image_plane_trajectory(predicted: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Compare only scale-free lateral shape and heading response."""
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("predicted and reference must both have shape [T,3]")
    pred_x = max(abs(float(pred[-1, 0])), 1e-6)
    ref_x = max(abs(float(ref[-1, 0])), 1e-6)
    pred_norm = pred.copy(); ref_norm = ref.copy()
    pred_norm[:, :2] /= pred_x; ref_norm[:, :2] /= ref_x
    pred_norm[:, 2] /= max(abs(float(pred[-1, 2])), 1.0)
    ref_norm[:, 2] /= max(abs(float(ref[-1, 2])), 1.0)
    lateral_error = float(np.mean(np.abs(pred_norm[:, 1] - ref_norm[:, 1])))
    yaw_error = float(np.mean(np.abs(pred_norm[:, 2] - ref_norm[:, 2])))
    return {
        "protocol": "image-plane-shape-comparison-v1",
        "scale_free_lateral_mae": lateral_error,
        "scale_free_yaw_mae": yaw_error,
        "shape_compatibility": float(np.exp(-(lateral_error + yaw_error))),
        "speed_absolute_observable": False,
        "predicted_normalized": pred_norm.tolist(),
        "reference_normalized": ref_norm.tolist(),
    }
