"""Camera-calibration contracts and conservative missing-data handling.

The image-side evaluator needs calibration for metric ego-frame projection. A
WAM manifest often contains only rendered images, so this module makes that
gap explicit instead of silently inventing an identity camera.
"""

from __future__ import annotations

from typing import Any

import numpy as np


CALIBRATION_SCHEMA_VERSION = "camera-calibration-v1"


def _finite_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("calibration")
    payload = dict(nested) if isinstance(nested, dict) else {}
    # Top-level fields are the legacy IAC protocol. Nested fields take
    # precedence so an augmented WAM manifest can be merged without rewriting
    # the original source fields.
    for key in ("intrinsics", "camera_to_ego", "distortion", "image_size", "camera_channel"):
        if key in row and key not in payload:
            payload[key] = row[key]
    return payload


def calibration_status(row: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-readable calibration status without guessing values."""
    payload = _payload(row)
    reasons: list[str] = []
    intrinsics = _finite_matrix(payload.get("intrinsics"), (3, 3))
    camera_to_ego = _finite_matrix(payload.get("camera_to_ego"), (4, 4))

    if payload.get("intrinsics") is None:
        reasons.append("missing_intrinsics")
    elif intrinsics is None:
        reasons.append("invalid_intrinsics")
    else:
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            reasons.append("non_positive_focal_length")
        if not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-4):
            reasons.append("intrinsics_not_homogeneous")

    if payload.get("camera_to_ego") is None:
        reasons.append("missing_camera_to_ego")
    elif camera_to_ego is None:
        reasons.append("invalid_camera_to_ego")
    elif not np.allclose(camera_to_ego[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        reasons.append("camera_to_ego_not_homogeneous")

    distortion = payload.get("distortion")
    if distortion is not None:
        try:
            distortion_array = np.asarray(distortion, dtype=np.float64)
            if distortion_array.size and not np.all(np.isfinite(distortion_array)):
                reasons.append("invalid_distortion")
        except (TypeError, ValueError):
            reasons.append("invalid_distortion")

    image_size = payload.get("image_size")
    if image_size is not None:
        try:
            size = np.asarray(image_size, dtype=np.float64).reshape(-1)
            if size.size != 2 or not np.all(np.isfinite(size)) or np.any(size <= 0):
                reasons.append("invalid_image_size")
        except (TypeError, ValueError):
            reasons.append("invalid_image_size")

    invalid = any(reason.startswith("invalid_") or reason.endswith("homogeneous") or reason == "non_positive_focal_length" for reason in reasons)
    if invalid:
        status = "invalid"
    elif not reasons:
        status = "complete"
    elif len(reasons) < 2:
        status = "partial"
    else:
        status = "missing"
    return {
        "schema": str(payload.get("schema") or CALIBRATION_SCHEMA_VERSION),
        "status": status,
        "projectable": status == "complete",
        "projection_mode": "metric_ego" if status == "complete" else "image_plane_only",
        "source": payload.get("source"),
        "camera_channel": payload.get("camera_channel"),
        "reasons": reasons,
    }


def calibration_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Return normalized calibration values plus the explicit status."""
    payload = _payload(row)
    status = calibration_status(row)
    result: dict[str, Any] = {"schema": CALIBRATION_SCHEMA_VERSION, **status}
    for key in ("intrinsics", "camera_to_ego", "distortion", "image_size", "camera_channel", "source"):
        if payload.get(key) is not None:
            value = payload[key]
            result[key] = np.asarray(value).tolist() if isinstance(value, (list, tuple, np.ndarray)) else value
    return result
