"""Training-free reliability fusion for real optical flow observations.

The module is deliberately conservative: it never invents motion and it never
turns a semantic road mask into a hard obstacle.  It combines the existing
forward/backward gate with a photometric warp residual, optional metric-depth
validity, and tile-level robust statistics.  The result is a soft support map
plus diagnostics that can be audited independently of the trajectory decoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .geometry import adjacent_camera_transforms, candidate_camera_poses, ground_plane_homography, se2_to_transform


def read_resized_rgb_frames(
    frame_paths: Sequence[str], target_size: tuple[int, int],
    intrinsics: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Read frames in the same target geometry used by the RAFT extractor."""
    width, height = int(target_size[0]), int(target_size[1])
    if min(width, height) <= 0:
        raise ValueError("target_size must be positive")
    frames: list[np.ndarray] = []
    for path in frame_paths:
        image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        if intrinsics is not None and distortion is not None and np.asarray(distortion).size:
            image = cv2.undistort(image, np.asarray(intrinsics, dtype=np.float64), np.asarray(distortion, dtype=np.float64), None, np.asarray(intrinsics, dtype=np.float64))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frames


def _robust_scale(values: np.ndarray, floor: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(floor)
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    return float(max(floor, mad, 0.5 * median))


class FlowReliabilityFusion:
    """Fuse independent reliability cues without retraining a flow network."""

    def __init__(
        self,
        *,
        tile_size: int = 16,
        photometric_sigma: float = 0.08,
        photometric_floor: float = 0.25,
        tile_floor: float = 0.50,
        weight_floor: float = 0.05,
        min_valid_fraction: float = 0.05,
        repair_enabled: bool = False,
        repair_threshold_px: float = 2.0,
    ) -> None:
        self.tile_size = int(tile_size)
        self.photometric_sigma = float(max(photometric_sigma, 1e-4))
        self.photometric_floor = float(np.clip(photometric_floor, 0.0, 1.0))
        self.tile_floor = float(np.clip(tile_floor, 0.0, 1.0))
        self.weight_floor = float(np.clip(weight_floor, 0.0, 1.0))
        self.min_valid_fraction = float(np.clip(min_valid_fraction, 0.0, 1.0))
        self.repair_enabled = bool(repair_enabled)
        self.repair_threshold_px = float(max(repair_threshold_px, 0.1))
        if self.tile_size < 2:
            raise ValueError("tile_size must be at least 2")

    @staticmethod
    def _photometric_weight(
        first: np.ndarray, second: np.ndarray, flow: np.ndarray, sigma: float, floor: float
    ) -> tuple[np.ndarray, np.ndarray]:
        first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        height, width = first_gray.shape
        yy, xx = np.indices((height, width), dtype=np.float32)
        map_x = xx + flow[..., 0].astype(np.float32)
        map_y = yy + flow[..., 1].astype(np.float32)
        warped = cv2.remap(
            second_gray, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
        )
        residual = np.abs(first_gray - warped)
        valid = np.isfinite(residual)
        # Per-frame robust normalization absorbs exposure changes while still
        # penalizing local mismatches caused by wrong flow or occlusion.
        scale = _robust_scale(residual[valid], max(sigma * 0.25, 1e-3)) if valid.any() else sigma
        normalized = residual / max(scale, sigma * 0.5)
        weight = np.exp(-np.clip(normalized / max(sigma / 0.08, 0.25), 0.0, 8.0))
        weight = np.where(valid, np.clip(floor + (1.0 - floor) * weight, floor, 1.0), 0.0)
        return weight.astype(np.float32), residual.astype(np.float32)

    def _tile_weight(self, pixel_weight: np.ndarray, valid: np.ndarray) -> np.ndarray:
        height, width = pixel_weight.shape
        tile = np.ones_like(pixel_weight, dtype=np.float32)
        values: list[float] = []
        locations: list[tuple[int, int, int, int]] = []
        for y0 in range(0, height, self.tile_size):
            for x0 in range(0, width, self.tile_size):
                y1, x1 = min(y0 + self.tile_size, height), min(x0 + self.tile_size, width)
                region = valid[y0:y1, x0:x1]
                if not region.any():
                    score = 0.0
                else:
                    score = float(np.median(pixel_weight[y0:y1, x0:x1][region]))
                values.append(score)
                locations.append((y0, y1, x0, x1))
        if not values:
            return tile
        finite = np.asarray(values, dtype=np.float64)
        scale = _robust_scale(finite, 1e-3)
        # A tile is downweighted only when it is a robust outlier.  This avoids
        # converting one shadow or a small occluder into a full-image abstain.
        median = float(np.median(finite))
        robust = np.clip((finite - 0.5 * (1.0 - median)) / max(scale, 1e-3), 0.0, 1.0)
        robust = self.tile_floor + (1.0 - self.tile_floor) * robust
        for value, (y0, y1, x0, x1) in zip(robust, locations):
            tile[y0:y1, x0:x1] = float(value)
        return tile.astype(np.float32)

    def estimate(
        self,
        *,
        observed_flows: np.ndarray,
        frame_paths: Sequence[str] | None = None,
        frames: Sequence[np.ndarray] | None = None,
        intrinsics: np.ndarray | None = None,
        distortion: np.ndarray | None = None,
        consistency_masks: np.ndarray | None = None,
        base_weights: np.ndarray | None = None,
        depths_m: np.ndarray | None = None,
    ) -> dict[str, Any]:
        flow = np.asarray(observed_flows, dtype=np.float32)
        if flow.ndim != 4 or flow.shape[-1] != 2:
            raise ValueError("observed_flows must have shape [T,H,W,2]")
        intervals, height, width = flow.shape[:3]
        if frames is None and frame_paths is not None:
            frames = read_resized_rgb_frames(frame_paths, (width, height), intrinsics, distortion)
        if frames is not None and len(frames) != intervals + 1:
            raise ValueError("frames must contain one more frame than observed_flows")
        if consistency_masks is not None and np.asarray(consistency_masks).shape != flow.shape[:-1]:
            raise ValueError("consistency_masks must match observed_flows")
        weights = np.ones(flow.shape[:-1], dtype=np.float32) if base_weights is None else np.asarray(base_weights, dtype=np.float32).copy()
        if weights.shape != flow.shape[:-1]:
            raise ValueError("base_weights must match observed_flows")
        depth = None if depths_m is None else np.asarray(depths_m)
        if depth is not None and depth.shape != flow.shape[:-1]:
            raise ValueError("depths_m must match observed_flows")

        output = np.zeros_like(weights, dtype=np.float32)
        photometric = np.ones_like(weights, dtype=np.float32)
        tile_maps = np.ones_like(weights, dtype=np.float32)
        repaired_flow = flow.copy()
        repaired_fraction: list[float] = []
        residuals: list[float | None] = []
        valid_fractions: list[float] = []
        for index in range(intervals):
            finite = np.isfinite(flow[index]).all(axis=-1)
            if consistency_masks is not None:
                finite &= np.asarray(consistency_masks[index], dtype=bool)
            if depth is not None:
                finite &= np.isfinite(depth[index]) & (depth[index] > 0.0)
            if frames is not None:
                photo, residual = self._photometric_weight(
                    np.asarray(frames[index]), np.asarray(frames[index + 1]), flow[index],
                    self.photometric_sigma, self.photometric_floor,
                )
                photometric[index] = photo
                tile_maps[index] = self._tile_weight(photo, finite)
                residuals.append(float(np.median(residual[finite])) if finite.any() else None)
            else:
                tile_maps[index] = self._tile_weight(weights[index], finite)
                residuals.append(None)
            if self.repair_enabled:
                median_u = cv2.medianBlur(flow[index, ..., 0].astype(np.float32), 5)
                median_v = cv2.medianBlur(flow[index, ..., 1].astype(np.float32), 5)
                median_flow = np.stack([median_u, median_v], axis=-1)
                discrepancy = np.linalg.norm(flow[index] - median_flow, axis=-1)
                local_scale = _robust_scale(discrepancy[finite], 0.25) if finite.any() else 0.25
                repair_mask = finite & np.isfinite(discrepancy) & (
                    discrepancy > max(self.repair_threshold_px, 3.0 * local_scale)
                )
                repaired_flow[index, repair_mask] = median_flow[repair_mask]
                repaired_fraction.append(float(np.mean(repair_mask)))
            else:
                repaired_fraction.append(0.0)
            fused = weights[index] * photometric[index] * tile_maps[index]
            fused = np.where(finite, fused, 0.0)
            output[index] = np.clip(fused, 0.0, 1.0)
            valid_fractions.append(float(np.mean(output[index] >= self.weight_floor)))
        return {
            "protocol": "flow-reliability-fusion-v1",
            "weights": output,
            "repaired_flows": repaired_flow,
            "photometric_weights": photometric,
            "tile_weights": tile_maps,
            "median_photometric_residual": residuals,
            "effective_fraction": valid_fractions,
            "mean_weight": float(np.mean(output)) if output.size else 0.0,
            "valid_fraction": float(np.mean(output >= self.weight_floor)) if output.size else 0.0,
            "photometric_available": frames is not None,
            "depth_validity_available": depth is not None,
            "repair_enabled": self.repair_enabled,
            "repaired_fraction": repaired_fraction,
        }


def calibrate_historical_flow_bias(
    *,
    full_flows: np.ndarray,
    history_ego_state: np.ndarray,
    history_count: int,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    roi_mask: np.ndarray,
    consistency_masks: np.ndarray | None = None,
    tile_size: int = 32,
    shrinkage: float = 0.35,
    max_correction_px: float = 1.5,
) -> dict[str, Any]:
    """Estimate a small systematic RAFT bias from known history ego motion.

    The calibration is candidate-blind: it uses only history frames and their
    recorded ego state.  A planar geometry model is used only to estimate a
    low-frequency residual field; robust tile medians and shrinkage prevent
    noisy history intervals from rewriting future motion.
    """
    flow = np.asarray(full_flows, dtype=np.float64)
    state = np.asarray(history_ego_state, dtype=np.float64)
    roi = np.asarray(roi_mask, dtype=bool)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("full_flows must have shape [T,H,W,2]")
    if state.ndim != 2 or len(state) < 2 or state.shape[1] < 3:
        return {"available": False, "reason": "history_state_missing", "corrected_flows": flow.astype(np.float32)}
    h, w = flow.shape[1:3]
    intervals = min(int(history_count) - 1, len(state) - 1, len(flow))
    if intervals < 1:
        return {"available": False, "reason": "history_intervals_missing", "corrected_flows": flow.astype(np.float32)}
    poses = [se2_to_transform(float(row[0]), float(row[1]), float(row[2])) for row in state[: intervals + 1]]
    camera_poses = [pose @ np.asarray(camera_to_ego, dtype=np.float64) for pose in poses]
    transforms = adjacent_camera_transforms(camera_poses)
    yy, xx = np.indices((h, w), dtype=np.float64)
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=0).reshape(3, -1)
    residuals: list[np.ndarray] = []
    valid_maps: list[np.ndarray] = []
    K = np.asarray(intrinsics, dtype=np.float64)
    for index in range(intervals):
        try:
            homography = ground_plane_homography(K, transforms[index], camera_poses[index])
        except (ValueError, np.linalg.LinAlgError):
            continue
        projected = homography @ pixels
        denominator = projected[2]
        valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
        next_x = projected[0] / np.where(valid, denominator, 1.0)
        next_y = projected[1] / np.where(valid, denominator, 1.0)
        valid &= (next_x >= 0.0) & (next_x <= w - 1) & (next_y >= 0.0) & (next_y <= h - 1)
        expected = np.stack([next_x - pixels[0], next_y - pixels[1]], axis=1).reshape(h, w, 2)
        valid = valid.reshape(h, w)
        valid &= roi & np.isfinite(flow[index]).all(axis=-1) & np.isfinite(expected).all(axis=-1)
        if consistency_masks is not None:
            valid &= np.asarray(consistency_masks[index], dtype=bool)
        residuals.append(flow[index] - expected)
        valid_maps.append(valid)
    if not residuals:
        return {"available": False, "reason": "geometry_invalid", "corrected_flows": flow.astype(np.float32)}
    residual = np.stack(residuals)
    valid = np.stack(valid_maps)
    correction = np.zeros((h, w, 2), dtype=np.float64)
    support = np.zeros((h, w), dtype=np.float64)
    tile = max(int(tile_size), 4)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            region = valid[:, y0:y1, x0:x1]
            values = residual[:, y0:y1, x0:x1][region]
            if len(values) < 8:
                continue
            med = np.median(values, axis=0)
            med = np.clip(med, -max_correction_px, max_correction_px)
            correction[y0:y1, x0:x1] = float(np.clip(shrinkage, 0.0, 1.0)) * med
            support[y0:y1, x0:x1] = float(len(values))
    corrected = flow.copy()
    future_start = max(int(history_count) - 1, 0)
    corrected[future_start:] -= correction[None, ...]
    return {
        "protocol": "historical-flow-bias-calibration-v1",
        "available": bool(np.any(support > 0.0)),
        "history_intervals_used": int(len(residuals)),
        "median_residual_px": float(np.median(np.linalg.norm(residual[valid], axis=-1))) if valid.any() else None,
        "median_correction_px": float(np.median(np.linalg.norm(correction[support > 0.0], axis=-1))) if np.any(support > 0.0) else 0.0,
        "corrected_fraction": float(np.mean(np.linalg.norm(correction, axis=-1) > 0.0)),
        "corrected_flows": corrected.astype(np.float32),
    }


def calibrate_historical_row_bias(
    *,
    full_flows: np.ndarray,
    history_ego_state: np.ndarray,
    history_count: int,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    roi_mask: np.ndarray,
    consistency_masks: np.ndarray | None = None,
    bands: int = 12,
    shrinkage: float = 0.15,
    max_correction_px: float = 0.8,
) -> dict[str, Any]:
    """Estimate a conservative, height-only flow bias from history.

    Unlike tile calibration this cannot memorize lateral image locations.  A
    robust median is estimated per image-row band and heavily shrunk before it
    is applied to future flow.  It is therefore useful as an audit of
    transferable low-frequency bias, not as a learned scene correction.
    """
    flow = np.asarray(full_flows, dtype=np.float64)
    state = np.asarray(history_ego_state, dtype=np.float64)
    roi = np.asarray(roi_mask, dtype=bool)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("full_flows must have shape [T,H,W,2]")
    if state.ndim != 2 or len(state) < 2 or state.shape[1] < 3:
        return {"available": False, "reason": "history_state_missing", "corrected_flows": flow.astype(np.float32)}
    h, w = flow.shape[1:3]
    intervals = min(int(history_count) - 1, len(state) - 1, len(flow))
    if intervals < 1:
        return {"available": False, "reason": "history_intervals_missing", "corrected_flows": flow.astype(np.float32)}
    poses = [se2_to_transform(float(row[0]), float(row[1]), float(row[2])) for row in state[: intervals + 1]]
    camera_poses = [pose @ np.asarray(camera_to_ego, dtype=np.float64) for pose in poses]
    transforms = adjacent_camera_transforms(camera_poses)
    yy, xx = np.indices((h, w), dtype=np.float64)
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=0).reshape(3, -1)
    residuals, valid_maps = [], []
    K = np.asarray(intrinsics, dtype=np.float64)
    for index in range(intervals):
        try:
            homography = ground_plane_homography(K, transforms[index], camera_poses[index])
        except (ValueError, np.linalg.LinAlgError):
            continue
        projected = homography @ pixels
        denominator = projected[2]
        valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
        next_x = projected[0] / np.where(valid, denominator, 1.0)
        next_y = projected[1] / np.where(valid, denominator, 1.0)
        valid &= (next_x >= 0.0) & (next_x <= w - 1) & (next_y >= 0.0) & (next_y <= h - 1)
        expected = np.stack([next_x - pixels[0], next_y - pixels[1]], axis=1).reshape(h, w, 2)
        valid = valid.reshape(h, w) & roi & np.isfinite(flow[index]).all(axis=-1)
        if consistency_masks is not None:
            valid &= np.asarray(consistency_masks[index], dtype=bool)
        residuals.append(flow[index] - expected)
        valid_maps.append(valid)
    if not residuals:
        return {"available": False, "reason": "geometry_invalid", "corrected_flows": flow.astype(np.float32)}
    residual = np.stack(residuals)
    valid = np.stack(valid_maps)
    n_bands = max(2, int(bands))
    correction = np.zeros((h, 2), dtype=np.float64)
    support = np.zeros(h, dtype=np.float64)
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    for band in range(n_bands):
        y0, y1 = int(round(band * h / n_bands)), int(round((band + 1) * h / n_bands))
        values = residual[:, y0:y1][valid[:, y0:y1]]
        if len(values) < 16:
            continue
        med = np.clip(np.median(values, axis=0), -max_correction_px, max_correction_px)
        correction[y0:y1] = shrink * med
        support[y0:y1] = len(values)
    corrected = flow.copy()
    future_start = max(int(history_count) - 1, 0)
    corrected[future_start:] -= correction[None, :, None, :]
    return {
        "protocol": "historical-row-bias-calibration-v1",
        "available": bool(np.any(support > 0.0)),
        "history_intervals_used": int(len(residuals)),
        "bands": n_bands,
        "median_residual_px": float(np.median(np.linalg.norm(residual[valid], axis=-1))) if valid.any() else None,
        "median_correction_px": float(np.median(np.linalg.norm(correction[support > 0.0], axis=-1))) if np.any(support > 0.0) else 0.0,
        "corrected_fraction": float(np.mean(np.linalg.norm(correction, axis=-1) > 0.0)),
        "corrected_flows": corrected.astype(np.float32),
    }
