"""Actor-level metric relative motion for event-causal WAM evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ActorRelativeTrack:
    """Metric actor observations expressed in each frame's ego coordinates."""

    actor_id: str
    class_label: str
    times_s: np.ndarray
    positions_ego_m: np.ndarray
    visibility: np.ndarray | None = None
    confidence: np.ndarray | None = None


@dataclass(frozen=True)
class ActorPixelTrack:
    """One actor anchor tracked in pixel coordinates across future frames."""

    actor_id: str
    class_label: str
    times_s: np.ndarray
    pixels_uv: np.ndarray
    visibility: np.ndarray | None = None
    confidence: np.ndarray | None = None


def validate_actor_future_window(
    times_s: np.ndarray,
    *,
    expected_frames: int = 8,
    expected_first_s: float = 0.5,
    expected_last_s: float = 4.0,
    tolerance_s: float = 0.10,
) -> np.ndarray:
    """Validate the reproducible 8-frame/4-second actor observation window."""
    times = np.asarray(times_s, dtype=np.float64)
    if times.shape != (int(expected_frames),) or not np.isfinite(times).all():
        raise ValueError(f"times_s must contain exactly {expected_frames} finite timestamps")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing")
    tolerance = max(float(tolerance_s), 0.0)
    if abs(float(times[0]) - float(expected_first_s)) > tolerance:
        raise ValueError("actor window must start at the expected first timestamp")
    if abs(float(times[-1]) - float(expected_last_s)) > tolerance:
        raise ValueError("actor window must end at the expected last timestamp")
    return times


def ground_contact_pixels_to_ego(
    pixels_uv: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_ego: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect camera rays with the local ego ground plane (z=0)."""
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(camera_to_ego, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_uv must have shape [N,2]")
    if K.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("intrinsics and camera_to_ego must have shapes [3,3] and [4,4]")
    homogeneous = np.column_stack([pixels, np.ones(len(pixels), dtype=np.float64)])
    rays_camera = (np.linalg.inv(K) @ homogeneous.T).T
    directions_ego = (transform[:3, :3] @ rays_camera.T).T
    origin_ego = transform[:3, 3]
    denominator = directions_ego[:, 2]
    scale = np.full(len(pixels), np.nan, dtype=np.float64)
    valid = np.isfinite(pixels).all(axis=1) & np.isfinite(denominator)
    valid &= np.abs(denominator) > 1e-8
    scale[valid] = -origin_ego[2] / denominator[valid]
    valid &= np.isfinite(scale) & (scale > 0.0)
    points = np.full((len(pixels), 3), np.nan, dtype=np.float64)
    points[valid] = origin_ego + scale[valid, None] * directions_ego[valid]
    valid &= np.isfinite(points).all(axis=1)
    return points, valid


def depth_pixels_to_ego(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_ego: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project per-pixel camera-z depth into the current ego frame."""
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(camera_to_ego, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depth.shape != (len(pixels),):
        raise ValueError("pixels_uv/depth_m must have shapes [N,2] and [N]")
    homogeneous = np.column_stack([pixels, np.ones(len(pixels), dtype=np.float64)])
    points_camera = (np.linalg.inv(K) @ homogeneous.T).T * depth[:, None]
    points = (transform[:3, :3] @ points_camera.T).T + transform[:3, 3]
    valid = np.isfinite(points).all(axis=1) & np.isfinite(depth) & (depth > 0.0)
    points[~valid] = np.nan
    return points, valid


def project_actor_pixel_track(
    track: ActorPixelTrack,
    intrinsics: np.ndarray,
    camera_to_ego: np.ndarray,
    *,
    depth_m: np.ndarray | None = None,
    depth_fallback_to_ground: bool = True,
) -> tuple[ActorRelativeTrack, dict[str, Any]]:
    """Project tracked actor anchors into per-frame metric ego coordinates.

    Intrinsics and camera extrinsics may be shared ``[3,3]``/``[4,4]`` arrays or
    supplied per frame as ``[T,3,3]``/``[T,4,4]``. Ground-plane intersection is
    the default scale source. Metric depth can override it where valid, with a
    ground fallback explicitly reported in the returned metadata.
    """
    times = np.asarray(track.times_s, dtype=np.float64)
    pixels = np.asarray(track.pixels_uv, dtype=np.float64)
    if times.ndim != 1 or pixels.shape != (len(times), 2):
        raise ValueError("times_s and pixels_uv must have shapes [T] and [T,2]")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing")
    visibility = (
        np.ones(len(times), dtype=bool)
        if track.visibility is None
        else np.asarray(track.visibility, dtype=bool)
    )
    confidence = (
        np.ones(len(times), dtype=np.float64)
        if track.confidence is None
        else np.asarray(track.confidence, dtype=np.float64)
    )
    if visibility.shape != (len(times),) or confidence.shape != (len(times),):
        raise ValueError("visibility and confidence must match times_s")
    if depth_m is not None:
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.shape != (len(times),):
            raise ValueError("depth_m must have shape [T]")
    else:
        depth = None

    def frame_value(value: np.ndarray, index: int, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape == shape:
            return array
        if array.shape == (len(times),) + shape:
            return array[index]
        raise ValueError(f"calibration must have shape {shape} or [{len(times)}, ...]")

    positions = np.full((len(times), 2), np.nan, dtype=np.float64)
    projection_valid = np.zeros(len(times), dtype=bool)
    projection_sources: list[str] = ["invalid"] * len(times)
    for index in range(len(times)):
        if not np.isfinite(pixels[index]).all() or not visibility[index]:
            continue
        K = frame_value(intrinsics, index, (3, 3))
        transform = frame_value(camera_to_ego, index, (4, 4))
        used_depth = depth is not None and np.isfinite(depth[index]) and depth[index] > 0.0
        if used_depth:
            projected, valid = depth_pixels_to_ego(
                pixels[index:index + 1],
                depth[index:index + 1],
                K,
                transform,
            )
            if valid[0]:
                positions[index] = projected[0, :2]
                projection_valid[index] = True
                projection_sources[index] = "metric_depth"
                continue
        if depth is None or depth_fallback_to_ground:
            projected, valid = ground_contact_pixels_to_ego(
                pixels[index:index + 1], K, transform
            )
            if valid[0]:
                positions[index] = projected[0, :2]
                projection_valid[index] = True
                projection_sources[index] = "ground_plane"

    relative_track = ActorRelativeTrack(
        actor_id=track.actor_id,
        class_label=track.class_label,
        times_s=times,
        positions_ego_m=positions,
        visibility=visibility & projection_valid,
        confidence=confidence,
    )
    metadata = {
        "protocol": "actor-pixel-projection-v1",
        "projection_valid": projection_valid.tolist(),
        "projection_sources": projection_sources,
        "metric_depth_fraction": float(np.mean(np.asarray(projection_sources) == "metric_depth"))
        if len(times) else 0.0,
        "ground_plane_fraction": float(np.mean(np.asarray(projection_sources) == "ground_plane"))
        if len(times) else 0.0,
        "candidate_bank_used": False,
        "future_action_used": False,
    }
    return relative_track, metadata


def _interval(center: float, sigma: float) -> dict[str, float]:
    radius = 1.645 * max(float(sigma), 1e-6)
    return {"q05": float(center - radius), "q50": float(center), "q95": float(center + radius)}


def _robust_polynomial_fit(
    times: np.ndarray,
    positions: np.ndarray,
    base_weights: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    origin = float(times[0])
    tau = times - origin
    design = np.column_stack([tau ** power for power in range(degree + 1)])
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    coefficients = np.zeros((degree + 1, 2), dtype=np.float64)
    for _ in range(8):
        root = np.sqrt(np.maximum(weights, 1e-8))[:, None]
        coefficients = np.linalg.lstsq(design * root, positions * root, rcond=None)[0]
        residual = np.linalg.norm(positions - design @ coefficients, axis=1)
        median = float(np.median(residual))
        scale = max(1.4826 * float(np.median(np.abs(residual - median))), 0.05)
        huber = np.minimum(1.0, 1.345 * scale / np.maximum(residual, 1e-8))
        updated = base_weights * huber
        if np.max(np.abs(updated - weights)) < 1e-4:
            weights = updated
            break
        weights = updated
    residual = positions - design @ coefficients
    rmse = float(np.sqrt(np.average(np.sum(residual**2, axis=1), weights=weights)))
    return coefficients, weights, rmse


def estimate_actor_relative_motion(
    track: ActorRelativeTrack,
    *,
    corridor_half_width_m: float = 1.25,
    minimum_samples: int = 3,
    minimum_span_s: float = 1.0,
) -> dict[str, Any]:
    """Fit relative distance and velocity without reading a future action."""
    times = np.asarray(track.times_s, dtype=np.float64)
    positions = np.asarray(track.positions_ego_m, dtype=np.float64)
    if times.ndim != 1 or positions.shape != (len(times), 2):
        raise ValueError("times_s and positions_ego_m must have shapes [T] and [T,2]")
    if len(times) and np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing")
    visibility = (
        np.ones(len(times), dtype=bool)
        if track.visibility is None
        else np.asarray(track.visibility, dtype=bool)
    )
    confidence = (
        np.ones(len(times), dtype=np.float64)
        if track.confidence is None
        else np.asarray(track.confidence, dtype=np.float64)
    )
    if visibility.shape != (len(times),) or confidence.shape != (len(times),):
        raise ValueError("visibility and confidence must match times_s")
    valid = visibility & np.isfinite(times) & np.isfinite(positions).all(axis=1)
    valid &= np.isfinite(confidence) & (confidence > 0.0)
    valid_count = int(valid.sum())
    span = float(times[valid][-1] - times[valid][0]) if valid_count else 0.0
    if valid_count < int(minimum_samples) or span < float(minimum_span_s):
        return {
            "protocol": "actor-relative-motion-v1",
            "actor_id": track.actor_id,
            "class_label": track.class_label,
            "available": False,
            "status": "abstain",
            "abstain_reason": "insufficient_temporal_support",
            "valid_samples": valid_count,
            "time_span_s": span,
            "observability": 0.0,
            "support": [],
        }

    fit_times = times[valid]
    fit_positions = positions[valid]
    fit_confidence = np.clip(confidence[valid], 0.01, 1.0)
    degree = 2 if valid_count >= 5 and span >= 2.0 else 1
    coefficients, weights, rmse = _robust_polynomial_fit(
        fit_times, fit_positions, fit_confidence, degree
    )
    tau = fit_times - fit_times[0]
    design = np.column_stack([tau ** power for power in range(degree + 1)])
    fitted = design @ coefficients
    derivative = np.zeros_like(fitted)
    for power in range(1, degree + 1):
        derivative += power * coefficients[power] * tau[:, None] ** (power - 1)

    support_fraction = float(valid_count / max(len(times), 1))
    median_confidence = float(np.median(fit_confidence))
    residual_factor = float(np.exp(-rmse / 1.5))
    observability = float(np.clip(support_fraction * median_confidence * residual_factor, 0.0, 1.0))
    status = "usable" if observability >= 0.55 else "uncertain"
    position_sigma = max(rmse, 0.10) / max(np.sqrt(valid_count), 1.0)
    velocity_sigma = max(rmse / max(span, 0.25), 0.10) / max(np.sqrt(valid_count), 1.0)

    rows: list[dict[str, Any]] = []
    ttc_values = []
    corridor_ttc_values = []
    time_to_corridor_values = []
    for time_s, point, velocity in zip(fit_times, fitted, derivative):
        d_long, d_lat = float(point[0]), float(point[1])
        closing_speed = float(-velocity[0])
        lateral_speed = float(velocity[1])
        inside_corridor = abs(d_lat) <= float(corridor_half_width_m) + 1e-6
        ttc = d_long / closing_speed if d_long > 0.0 and closing_speed > 0.25 else None
        if ttc is not None and np.isfinite(ttc) and ttc >= 0.0:
            ttc_values.append(float(ttc))
        else:
            ttc = None
        if inside_corridor:
            time_to_corridor = 0.0
        elif d_lat > corridor_half_width_m and lateral_speed < -0.05:
            time_to_corridor = (d_lat - corridor_half_width_m) / -lateral_speed
        elif d_lat < -corridor_half_width_m and lateral_speed > 0.05:
            time_to_corridor = (-corridor_half_width_m - d_lat) / lateral_speed
        else:
            time_to_corridor = None
        if time_to_corridor is not None and np.isfinite(time_to_corridor):
            time_to_corridor_values.append(float(time_to_corridor))
        else:
            time_to_corridor = None
        corridor_ttc = None
        if ttc is not None and (
            inside_corridor
            or (time_to_corridor is not None and time_to_corridor <= ttc)
        ):
            corridor_ttc = ttc
            corridor_ttc_values.append(float(ttc))
        rows.append({
            "time_s": float(time_s),
            "d_long_m": _interval(d_long, position_sigma),
            "d_lat_m": _interval(d_lat, position_sigma),
            "closing_speed_mps": _interval(closing_speed, velocity_sigma),
            "lateral_speed_mps": _interval(lateral_speed, velocity_sigma),
            "ttc_s": ttc,
            "corridor_conflict_ttc_s": corridor_ttc,
            "time_to_corridor_s": time_to_corridor,
            "inside_corridor": bool(inside_corridor),
        })

    return {
        "protocol": "actor-relative-motion-v1",
        "actor_id": track.actor_id,
        "class_label": track.class_label,
        "available": True,
        "status": status,
        "abstain_reason": None,
        "valid_samples": valid_count,
        "time_span_s": span,
        "support_fraction": support_fraction,
        "fit_degree": degree,
        "fit_rmse_m": rmse,
        "observability": observability,
        "minimum_ttc_s": min(ttc_values) if ttc_values else None,
        "minimum_corridor_conflict_ttc_s": (
            min(corridor_ttc_values) if corridor_ttc_values else None
        ),
        "minimum_time_to_corridor_s": (
            min(time_to_corridor_values) if time_to_corridor_values else None
        ),
        "support": rows,
        "candidate_bank_used": False,
        "future_action_used": False,
    }


def evaluate_relative_motion_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    dangerous_ttc_s: float = 4.0,
) -> dict[str, Any]:
    """Evaluate metric state accuracy and selective risk on actor gold labels."""
    records = list(rows)
    required_fields = (
        "predicted_distance_m", "reference_distance_m",
        "predicted_closing_speed_mps", "reference_closing_speed_mps",
    )

    leakage_rows = [
        row for row in records
        if bool(row.get("future_action_used", False))
        or bool(row.get("candidate_bank_used", False))
    ]

    def is_scored(row: dict[str, Any]) -> bool:
        if bool(row.get("abstain", False)) or bool(row.get("future_action_used", False)) or bool(row.get("candidate_bank_used", False)):
            return False
        try:
            return all(np.isfinite(float(row[field])) for field in required_fields)
        except (KeyError, TypeError, ValueError):
            return False

    scored = [row for row in records if is_scored(row)]
    distance_error = np.asarray([
        abs(float(row["predicted_distance_m"]) - float(row["reference_distance_m"]))
        for row in scored
    ], dtype=np.float64)
    speed_error = np.asarray([
        abs(float(row["predicted_closing_speed_mps"]) - float(row["reference_closing_speed_mps"]))
        for row in scored
    ], dtype=np.float64)

    def sign(value: float) -> int:
        return 1 if value > 0.25 else -1 if value < -0.25 else 0

    sign_correct = [
        sign(float(row["predicted_closing_speed_mps"]))
        == sign(float(row["reference_closing_speed_mps"]))
        for row in scored
    ]
    true_positive = false_positive = false_negative = 0
    for row in scored:
        predicted_ttc = row.get("predicted_ttc_s")
        reference_ttc = row.get("reference_ttc_s")
        predicted_danger = predicted_ttc is not None and float(predicted_ttc) <= dangerous_ttc_s
        reference_danger = reference_ttc is not None and float(reference_ttc) <= dangerous_ttc_s
        true_positive += int(predicted_danger and reference_danger)
        false_positive += int(predicted_danger and not reference_danger)
        false_negative += int(not predicted_danger and reference_danger)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    predicted_positive = true_positive + false_positive
    reference_positive = true_positive + false_negative
    precision_value = float(precision) if predicted_positive else None
    recall_value = float(recall) if reference_positive else None
    f1_value = (
        float(2.0 * precision * recall / max(precision + recall, 1e-12))
        if predicted_positive and reference_positive
        else None
    )

    ordered = sorted(scored, key=lambda row: float(row.get("observability", 0.0)), reverse=True)
    coverage_risk = []
    for target in (0.25, 0.50, 0.75, 1.00):
        count = int(np.ceil(target * len(ordered))) if ordered else 0
        selected = ordered[:count]
        errors = [
            abs(float(row["predicted_closing_speed_mps"]) - float(row["reference_closing_speed_mps"]))
            for row in selected
        ]
        coverage_risk.append({
            "coverage": float(count / max(len(records), 1)),
            "requested_scored_coverage": target,
            "closing_speed_mae_mps": float(np.mean(errors)) if errors else None,
        })
    return {
        "protocol": "actor-relative-motion-metrics-v1",
        "num_input": len(records),
        "num_scored": len(scored),
        "coverage": float(len(scored) / max(len(records), 1)),
        "num_leakage_excluded": len(leakage_rows),
        "formal_ready": not leakage_rows,
        "distance_mae_m": float(np.mean(distance_error)) if len(distance_error) else None,
        "closing_speed_mae_mps": float(np.mean(speed_error)) if len(speed_error) else None,
        "closing_sign_accuracy": float(np.mean(sign_correct)) if sign_correct else None,
        "dangerous_ttc_threshold_s": float(dangerous_ttc_s),
        "num_predicted_dangerous": int(predicted_positive),
        "num_reference_dangerous": int(reference_positive),
        "dangerous_ttc_precision": precision_value,
        "dangerous_ttc_recall": recall_value,
        "dangerous_ttc_f1": f1_value,
        "coverage_risk": coverage_risk,
    }
