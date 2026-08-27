"""Optional CoTracker3 point-track extraction for temporal geometry probes."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .relative_motion import ActorPixelTrack


@dataclass(frozen=True)
class PointTrackObservation:
    """Joint point tracks and model confidence for one image sequence."""

    tracks: np.ndarray  # [T, N, 2], pixel x/y
    visibility: np.ndarray  # [T, N]
    confidence: np.ndarray  # [T, N]
    query_points: np.ndarray  # [N, 2], pixel x/y in the first frame
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    query_frame_indices: np.ndarray | None = None


def actor_pixel_tracks_from_observation(
    observation: PointTrackObservation,
    actors: Sequence[Mapping[str, Any]],
    times_s: np.ndarray,
) -> list[ActorPixelTrack]:
    """Select detector-provided actor queries from a CoTracker observation.

    ``query_index`` refers to the actor anchor supplied to ``observe``. The
    association/detection system owns that mapping; this adapter only carries
    tracker visibility and confidence into the metric projection layer.
    """
    tracks = np.asarray(observation.tracks, dtype=np.float64)
    visibility = np.asarray(observation.visibility, dtype=np.float64)
    confidence = np.asarray(observation.confidence, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("observation.tracks must have shape [T,N,2]")
    if visibility.shape != tracks.shape[:2] or confidence.shape != tracks.shape[:2]:
        raise ValueError("observation visibility/confidence shape mismatch")
    if times.shape != (tracks.shape[0],) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be increasing and match tracker frames")
    output: list[ActorPixelTrack] = []
    for actor in actors:
        actor_id = str(actor.get("actor_id") or "")
        class_label = str(actor.get("class_label") or "unknown")
        if not actor_id:
            raise ValueError("each actor needs a non-empty actor_id")
        try:
            query_index = int(actor["query_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{actor_id}: query_index is required") from error
        if not 0 <= query_index < tracks.shape[1]:
            raise ValueError(f"{actor_id}: query_index is outside the observation")
        output.append(ActorPixelTrack(
            actor_id=actor_id,
            class_label=class_label,
            times_s=times.copy(),
            pixels_uv=tracks[:, query_index].copy(),
            visibility=visibility[:, query_index] > 0.5,
            confidence=confidence[:, query_index].copy(),
        ))
    return output


def _grid_points(
    height: int,
    width: int,
    *,
    grid_size: int,
    polygon_normalized: list[list[float]] | None,
) -> np.ndarray:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    polygon = None
    if polygon_normalized:
        polygon = np.asarray(
            [[float(x) * (width - 1), float(y) * (height - 1)] for x, y in polygon_normalized],
            dtype=np.float32,
        )
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError("polygon_normalized must have shape [N,2]")
    points = []
    for y in np.linspace(0.45 * (height - 1), 0.98 * (height - 1), grid_size):
        for x in np.linspace(0.05 * (width - 1), 0.95 * (width - 1), grid_size):
            if polygon is not None and cv2.pointPolygonTest(polygon, (float(x), float(y)), False) < 0:
                continue
            points.append([float(x), float(y)])
    if len(points) < 8:
        raise ValueError("road ROI produced fewer than eight CoTracker query points")
    return np.asarray(points, dtype=np.float32)


def validate_query_points(
    query_points: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Validate actor anchor queries in the resized image coordinate system."""
    points = np.asarray(query_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("query_points must have shape [N,2]")
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("query_points must contain finite points")
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] >= height)
    ):
        raise ValueError("query_points must lie inside target_size")
    return points


def validate_query_frame_indices(
    query_frame_indices: np.ndarray | None,
    *,
    num_queries: int,
    num_frames: int,
) -> np.ndarray:
    """Validate the frame where each CoTracker query first becomes visible."""
    if query_frame_indices is None:
        return np.zeros(int(num_queries), dtype=np.int64)
    query_frames = np.asarray(query_frame_indices, dtype=np.float64)
    if query_frames.shape != (int(num_queries),) or not np.isfinite(query_frames).all():
        raise ValueError("query_frame_indices must have shape [N]")
    if np.any(query_frames < 0) or np.any(query_frames >= int(num_frames)):
        raise ValueError("query_frame_indices must lie inside the video")
    if not np.equal(query_frames, np.floor(query_frames)).all():
        raise ValueError("query_frame_indices must be integers")
    return query_frames.astype(np.int64)


def actor_box_query_points(
    box_xyxy: np.ndarray,
    *,
    height: int,
    width: int,
    columns: int = 3,
    rows: int = 4,
) -> np.ndarray:
    """Sample body-region queries while avoiding the ground/background edge."""
    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box_xyxy must contain four finite values")
    x0, y0, x1, y1 = box
    if x1 - x0 < 2.0 or y1 - y0 < 2.0 or columns < 2 or rows < 2:
        raise ValueError("actor box or query grid is too small")
    xs = np.linspace(x0 + 0.20 * (x1 - x0), x1 - 0.20 * (x1 - x0), columns)
    ys = np.linspace(y0 + 0.15 * (y1 - y0), y0 + 0.75 * (y1 - y0), rows)
    return validate_query_points(
        np.asarray([[x, y] for y in ys for x in xs], dtype=np.float32),
        height=height,
        width=width,
    )


def aggregate_actor_region_tracks(
    observation: PointTrackObservation,
    anchor_query_point: np.ndarray,
    *,
    minimum_points: int = 3,
    confidence_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move an actor ground anchor with robust body-region track consensus."""
    tracks = np.asarray(observation.tracks, dtype=np.float64)
    visibility = np.asarray(observation.visibility, dtype=np.float64)
    confidence = np.asarray(observation.confidence, dtype=np.float64)
    source = np.asarray(observation.query_points, dtype=np.float64)
    anchor = np.asarray(anchor_query_point, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[1:] != (len(source), 2):
        raise ValueError("observation track/query shape mismatch")
    if visibility.shape != tracks.shape[:2] or confidence.shape != tracks.shape[:2]:
        raise ValueError("observation visibility/confidence shape mismatch")
    if anchor.shape != (2,) or not np.isfinite(anchor).all():
        raise ValueError("anchor_query_point must contain two finite values")
    output = np.full((tracks.shape[0], 2), np.nan, dtype=np.float64)
    output_visibility = np.zeros(tracks.shape[0], dtype=bool)
    output_confidence = np.zeros(tracks.shape[0], dtype=np.float64)
    for frame_index in range(tracks.shape[0]):
        valid = visibility[frame_index] > 0.5
        valid &= confidence[frame_index] >= float(confidence_threshold)
        valid &= np.isfinite(tracks[frame_index]).all(axis=1)
        if int(valid.sum()) < int(minimum_points):
            continue
        transform, inliers = cv2.estimateAffinePartial2D(
            source[valid], tracks[frame_index, valid], method=cv2.RANSAC,
            ransacReprojThreshold=3.0, maxIters=500, confidence=0.99,
        )
        if transform is None:
            displacement = np.median(tracks[frame_index, valid] - source[valid], axis=0)
            point = anchor + displacement
            inlier_fraction = 1.0
        else:
            point = transform[:, :2] @ anchor + transform[:, 2]
            inlier_fraction = float(np.mean(inliers)) if inliers is not None else 1.0
        width, height = observation.target_size
        if not np.isfinite(point).all() or not (0 <= point[0] < width and 0 <= point[1] < height):
            continue
        output[frame_index] = point
        output_visibility[frame_index] = True
        support_fraction = float(valid.mean())
        output_confidence[frame_index] = float(
            np.clip(np.median(confidence[frame_index, valid]) * support_fraction * inlier_fraction, 0.0, 1.0)
        )
    return output, output_visibility, output_confidence


class CoTrackerExtractor:
    """Lazy wrapper around the official CoTracker3 offline model.

    The dependency and checkpoint are intentionally loaded only when this
    backend is selected. The default IAC path remains RAFT-only.
    """

    def __init__(
        self,
        *,
        device: str,
        grid_size: int = 16,
        model_name: str = "cotracker3_offline",
        hub_repository: str = "facebookresearch/co-tracker",
        checkpoint: str | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.grid_size = int(grid_size)
        self.model_name = str(model_name)
        self.hub_repository = str(hub_repository)
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.model = torch.hub.load(
            self.hub_repository,
            self.model_name,
            pretrained=not bool(self.checkpoint),
            trust_repo=True,
            skip_validation=True,
        )
        if self.checkpoint:
            state_dict = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            target = self.model.model if hasattr(self.model, "model") else self.model
            target.load_state_dict(state_dict)
        self.model = self.model.to(device).eval()

    @staticmethod
    def _read_images(paths: list[str], target_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
        images: list[np.ndarray] = []
        source_size: tuple[int, int] | None = None
        for path in paths:
            image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            height, width = image.shape[:2]
            if source_size is None:
                source_size = (width, height)
            elif source_size != (width, height):
                raise ValueError("all frames in a video must have the same dimensions")
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if source_size is None:
            raise ValueError("video has no frames")
        return np.stack(images, axis=0), source_size

    def observe(
        self,
        frame_paths: list[str],
        *,
        target_size: tuple[int, int],
        polygon_normalized: list[list[float]] | None = None,
        query_points: np.ndarray | None = None,
        query_frame_indices: np.ndarray | None = None,
    ) -> PointTrackObservation:
        if len(frame_paths) < 2:
            raise ValueError("at least two video frames are required")
        frames, source_size = self._read_images(frame_paths, target_size)
        height, width = frames.shape[1:3]
        if query_points is None:
            query_points = _grid_points(
                height,
                width,
                grid_size=self.grid_size,
                polygon_normalized=polygon_normalized,
            )
        else:
            query_points = validate_query_points(
                query_points, height=height, width=width
            )
        query_frames = validate_query_frame_indices(
            query_frame_indices,
            num_queries=len(query_points),
            num_frames=len(frame_paths),
        )
        torch = self.torch
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)
        queries = torch.from_numpy(
            np.concatenate([query_frames[:, None].astype(np.float32), query_points], axis=1)
        ).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(video, queries)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("CoTracker output must contain tracks and visibility")
        tracks_output, visibility_output = output[0], output[1]
        confidence_output = output[2] if len(output) > 2 else visibility_output
        def batch_result(value: Any) -> np.ndarray:
            while isinstance(value, (tuple, list)):
                value = value[-1]
            array = value.detach().cpu().numpy().astype(np.float32)
            return array[0] if array.shape[0] == 1 else array
        tracks = batch_result(tracks_output)
        visibility = batch_result(visibility_output)
        confidence = batch_result(confidence_output)
        if tracks.shape[:2] != (len(frame_paths), len(query_points)):
            raise RuntimeError(f"unexpected CoTracker tracks shape: {tracks.shape}")
        return PointTrackObservation(
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            query_points=query_points,
            source_size=source_size,
            target_size=target_size,
            query_frame_indices=query_frames,
        )


def point_track_curvature_features(
    observation: PointTrackObservation,
    *,
    future_start_interval: int,
    confidence_threshold: float = 0.1,
) -> list[dict[str, Any]]:
    """Compute interpretable speed/curvature evidence from joint tracks.

    This is deliberately an evidence probe. It does not claim metric speed or
    replace the continuous trajectory decoder.
    """
    tracks = np.asarray(observation.tracks, dtype=np.float64)
    visibility = np.asarray(observation.visibility, dtype=np.float64)
    confidence = np.asarray(observation.confidence, dtype=np.float64)
    query = np.asarray(observation.query_points, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks must have shape [T,N,2]")
    if visibility.shape != tracks.shape[:2] or confidence.shape != tracks.shape[:2]:
        raise ValueError("visibility/confidence shape mismatch")
    width = max(int(observation.target_size[0]), 2)
    x = query[:, 0] / float(width - 1)
    central = (x >= 0.25) & (x < 0.75)
    left = x <= 0.40
    right = x >= 0.60
    features: list[dict[str, Any]] = []
    central_angles: list[float | None] = []
    for interval in range(tracks.shape[0] - 1):
        displacement = tracks[interval + 1] - tracks[interval]
        valid = visibility[interval] > 0.5
        valid &= visibility[interval + 1] > 0.5
        valid &= confidence[interval] >= float(confidence_threshold)
        valid &= confidence[interval + 1] >= float(confidence_threshold)
        finite = np.isfinite(displacement).all(axis=1)
        valid &= finite
        u = displacement[:, 0]
        v = displacement[:, 1]
        angle = np.arctan2(u, np.maximum(np.abs(v), 1e-3))
        def median_for(mask: np.ndarray) -> float | None:
            values = angle[valid & mask]
            return float(np.median(values)) if len(values) else None
        left_angle = median_for(left)
        right_angle = median_for(right)
        contrast = None
        if left_angle is not None and right_angle is not None:
            contrast = float(abs(np.arctan2(np.sin(right_angle - left_angle), np.cos(right_angle - left_angle))))
        central_angle = median_for(central)
        central_angles.append(central_angle)
        features.append({
            "interval_index": interval,
            "valid_fraction": float(valid.mean()),
            "left_support_fraction": float((valid & left).sum() / max(int(left.sum()), 1)),
            "right_support_fraction": float((valid & right).sum() / max(int(right.sum()), 1)),
            "central_support_fraction": float((valid & central).sum() / max(int(central.sum()), 1)),
            "median_flow_px": float(np.median(np.linalg.norm(displacement[valid], axis=1))) if valid.any() else 0.0,
            "curvature_lateral_contrast_rad": contrast,
            "central_flow_angle_rad": central_angle,
            "future_interval": interval >= int(future_start_interval),
        })
    for interval, item in enumerate(features):
        nearby = [central_angles[index] for index in (interval - 1, interval, interval + 1) if 0 <= index < len(central_angles)]
        transitions = [float(np.arctan2(np.sin(b - a), np.cos(b - a))) for a, b in zip(nearby[:-1], nearby[1:]) if a is not None and b is not None]
        item["curvature_temporal_turn_change_rad"] = float(np.median(np.abs(transitions))) if transitions else None
    return [item for item in features if item["future_interval"]]
