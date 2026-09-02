"""Validated cached metric-depth input for the lightweight evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricDepthObservation:
    depths_m: np.ndarray
    confidence: np.ndarray
    intrinsics: np.ndarray
    source: str
    cache_path: str
    scale_divisor: float


def load_cached_metric_depth(
    record: dict[str, Any],
    geometry_config: dict[str, Any],
    *,
    expected_intervals: int,
    expected_size: tuple[int, int],
    expected_intrinsics: np.ndarray,
) -> MetricDepthObservation:
    cache_value = record.get("metric_depth_path")
    if not cache_value:
        raise ValueError(f"{record['sample_id']}: metric_depth_path is required")
    cache_path = Path(str(cache_value))
    if not cache_path.is_file():
        raise FileNotFoundError(f"metric depth cache not found: {cache_path}")
    divisor = float(geometry_config.get("depth_scale_divisor", 1.0))
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError("geometry.depth_scale_divisor must be positive")

    with np.load(cache_path) as cache:
        required = {"depth_m", "confidence", "intrinsics", "target_size"}
        missing = sorted(required.difference(cache.files))
        if missing:
            raise ValueError(f"metric depth cache is missing fields: {missing}")
        depths = cache["depth_m"].astype(np.float32) / divisor
        confidence = cache["confidence"].astype(np.float32)
        intrinsics = cache["intrinsics"].astype(np.float64)
        target_size = tuple(int(value) for value in cache["target_size"].tolist())
        if bool(geometry_config.get("require_cache_metadata_match", True)):
            if "camera_to_ego" not in cache.files:
                raise ValueError("metric depth cache is missing camera_to_ego")
            if not np.allclose(cache["camera_to_ego"], record["camera_to_ego"], atol=1e-6):
                raise ValueError("metric depth cache camera_to_ego does not match manifest")

    expected_shape = (expected_intervals, expected_size[1], expected_size[0])
    if depths.shape != expected_shape or confidence.shape != expected_shape:
        raise ValueError(
            f"metric depth and confidence must have shape {expected_shape}, "
            f"got {depths.shape} and {confidence.shape}"
        )
    if target_size != expected_size:
        raise ValueError(
            f"metric depth target_size {target_size} does not match image size {expected_size}"
        )
    if intrinsics.shape != (3, 3) or not np.allclose(
        intrinsics, expected_intrinsics, rtol=1e-5, atol=1e-4
    ):
        raise ValueError("metric depth intrinsics do not match optical-flow intrinsics")
    return MetricDepthObservation(
        depths_m=depths,
        confidence=confidence,
        intrinsics=intrinsics,
        source=str(record.get("metric_depth_source") or geometry_config.get("source") or "unknown"),
        cache_path=str(cache_path),
        scale_divisor=divisor,
    )


def metric_depth_reliability_masks(
    observation: MetricDepthObservation,
    observed_flows: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    confidence_quantile: float,
    observed_flow_quantile: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Return candidate-independent validity and per-interval diagnostics."""
    if not 0.0 <= confidence_quantile < 1.0:
        raise ValueError("geometry.confidence_quantile must be in [0,1)")
    if not 0.0 < observed_flow_quantile <= 1.0:
        raise ValueError("geometry.observed_flow_quantile must be in (0,1]")
    if min_depth_m < 0.0 or max_depth_m <= min_depth_m:
        raise ValueError("geometry depth range is invalid")
    observed = np.asarray(observed_flows, dtype=np.float32)
    if observed.shape[:3] != observation.depths_m.shape or observed.shape[-1] != 2:
        raise ValueError("metric depth and observed-flow shapes do not match")

    masks: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    for depth, confidence, flow in zip(
        observation.depths_m, observation.confidence, observed
    ):
        base = (
            np.isfinite(depth)
            & (depth > float(min_depth_m))
            & (depth < float(max_depth_m))
            & np.isfinite(confidence)
            & np.isfinite(flow).all(axis=-1)
        )
        confidence_floor = (
            float(np.quantile(confidence[base], confidence_quantile))
            if base.any()
            else float("inf")
        )
        mask = base & (confidence >= confidence_floor)
        flow_ceiling = float("inf")
        if mask.any() and observed_flow_quantile < 1.0:
            magnitude = np.linalg.norm(flow, axis=-1)
            flow_ceiling = float(np.quantile(magnitude[mask], observed_flow_quantile))
            mask &= magnitude <= flow_ceiling
        masks.append(mask)
        diagnostics.append(
            {
                "valid_fraction": float(mask.mean()),
                "confidence_floor": confidence_floor,
                "observed_flow_ceiling_px": flow_ceiling,
                "median_depth_m": float(np.median(depth[mask])) if mask.any() else float("nan"),
            }
        )
    return np.stack(masks), diagnostics
