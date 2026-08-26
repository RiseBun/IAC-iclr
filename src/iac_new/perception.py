"""Optional lightweight semantic perception for the front-camera evaluator.

This module is deliberately an auxiliary constraint. It never produces the
trajectory posterior by itself; geometry and candidate-conditioned flow remain
the causal scoring signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def semantic_motion_summary(
    actor_masks: np.ndarray,
    dynamic_weights: np.ndarray,
    *,
    roi_mask: np.ndarray | None = None,
    dynamic_threshold: float = 0.5,
) -> list[dict[str, float | int | str | None]]:
    """Summarize semantic actor pixels against the geometric motion proxy.

    This is intentionally diagnostic. A SegFormer class label does not prove
    that an object is moving; the second factor is the shared rigid-flow
    residual weight. The output keeps those two evidence sources separate.
    """
    actors = np.asarray(actor_masks, dtype=bool)
    weights = np.asarray(dynamic_weights, dtype=np.float32)
    if actors.ndim != 3 or weights.shape != actors.shape:
        raise ValueError("actor_masks and dynamic_weights must both have shape [T,H,W]")
    roi = np.ones(actors.shape[1:], dtype=bool) if roi_mask is None else np.asarray(roi_mask, dtype=bool)
    if roi.shape != actors.shape[1:]:
        raise ValueError("roi_mask must have shape [H,W]")
    summaries: list[dict[str, float | int | str | None]] = []
    for index, (actor, weight) in enumerate(zip(actors, weights)):
        valid = roi & np.isfinite(weight)
        actor_valid = valid & actor
        actor_count = int(actor_valid.sum())
        dynamic_count = int((actor_valid & (weight < float(dynamic_threshold))).sum())
        static_count = int((actor_valid & (weight >= float(dynamic_threshold))).sum())
        summaries.append(
            {
                "interval_index": index,
                "actor_fraction": float(actor_valid.mean()) if valid.any() else 0.0,
                "actor_dynamic_fraction": float(dynamic_count / actor_count) if actor_count else None,
                "actor_static_fraction": float(static_count / actor_count) if actor_count else None,
                "actor_pixel_count": actor_count,
                "classification": (
                    "no_actor"
                    if not actor_count
                    else "likely_dynamic_actor"
                    if dynamic_count / actor_count >= float(dynamic_threshold)
                    else "mostly_static_or_unresolved_actor"
                ),
            }
        )
    return summaries


def project_ego_ground_to_image(
    points_ego: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame ground points and return pixels plus visibility."""
    points = np.asarray(points_ego, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ego must have shape [N,3]")
    camera_from_ego = np.linalg.inv(np.asarray(camera_to_ego, dtype=np.float64))
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float64)]
    points_camera = (camera_from_ego @ homogeneous.T).T[:, :3]
    projected = (np.asarray(intrinsics, dtype=np.float64) @ points_camera.T).T
    valid = np.isfinite(projected).all(axis=1) & (points_camera[:, 2] > 1e-6)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid] = projected[valid, :2] / projected[valid, 2:3]
    return pixels, valid


def trajectory_traversability(
    trajectory: np.ndarray,
    traversable_mask: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    *,
    half_width_m: float = 1.1,
    longitudinal_step_m: float = 0.5,
    lateral_samples: int = 5,
) -> dict[str, float | int | None]:
    """Measure how much of a candidate vehicle corridor lies on visible road."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    mask = np.asarray(traversable_mask, dtype=bool)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape [T,3]")
    centers = [np.zeros(3, dtype=np.float64)]
    previous = np.zeros(3, dtype=np.float64)
    for knot in trajectory:
        distance = float(np.linalg.norm(knot[:2] - previous[:2]))
        count = max(1, int(np.ceil(distance / max(longitudinal_step_m, 1e-3))))
        for alpha in np.linspace(0.0, 1.0, count + 1, endpoint=True)[1:]:
            center = previous + alpha * (knot - previous)
            centers.append(center)
        previous = knot
    ground_points = []
    for x_m, y_m, yaw_rad in centers[1:]:
        lateral = np.asarray([-np.sin(yaw_rad), np.cos(yaw_rad)], dtype=np.float64)
        for offset in np.linspace(-half_width_m, half_width_m, lateral_samples):
            xy = np.asarray([x_m, y_m]) + offset * lateral
            ground_points.append([xy[0], xy[1], 0.0])
    pixels, visible = project_ego_ground_to_image(
        np.asarray(ground_points), camera_to_ego, intrinsics
    )
    height, width = mask.shape
    inside = visible.copy()
    inside &= np.isfinite(pixels).all(axis=1)
    inside &= (pixels[:, 0] >= 0.0) & (pixels[:, 0] < width)
    inside &= (pixels[:, 1] >= 0.0) & (pixels[:, 1] < height)
    visible_count = int(inside.sum())
    if not visible_count:
        return {
            "traversable_fraction": None,
            "visible_corridor_samples": 0,
            "total_corridor_samples": len(ground_points),
        }
    xy = np.rint(pixels[inside]).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    return {
        "traversable_fraction": float(mask[xy[:, 1], xy[:, 0]].mean()),
        "visible_corridor_samples": visible_count,
        "total_corridor_samples": len(ground_points),
    }


@dataclass(frozen=True)
class PerceptionObservation:
    traversable_masks: np.ndarray
    actor_masks: np.ndarray
    class_maps: np.ndarray
    class_names: tuple[str, ...]
    backend: str
    model_id: str

    def future(self, future_start: int) -> "PerceptionObservation":
        return PerceptionObservation(
            traversable_masks=self.traversable_masks[future_start:],
            actor_masks=self.actor_masks[future_start:],
            class_maps=self.class_maps[future_start:],
            class_names=self.class_names,
            backend=self.backend,
            model_id=self.model_id,
        )


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return values.copy()
    padded = np.pad(values.astype(np.int32), int(radius), mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = 2 * int(radius) + 1
    counts = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return counts > 0


def temporal_road_consensus(
    frames: PerceptionObservation,
    observed_flows: np.ndarray,
    *,
    road_dilation_px: int = 4,
    actor_dilation_px: int = 3,
) -> PerceptionObservation:
    """Create candidate-independent interval masks stable across two frames."""
    flows = np.asarray(observed_flows, dtype=np.float32)
    road_frames = np.asarray(frames.traversable_masks, dtype=bool)
    actor_frames = np.asarray(frames.actor_masks, dtype=bool)
    if flows.ndim != 4 or flows.shape[-1] != 2:
        raise ValueError("observed_flows must have shape [T,H,W,2]")
    if road_frames.shape != actor_frames.shape or len(road_frames) != len(flows) + 1:
        raise ValueError("perception needs one more frame than optical-flow intervals")
    if road_frames.shape[1:3] != flows.shape[1:3]:
        raise ValueError("perception and optical flow spatial sizes differ")
    height, width = flows.shape[1:3]
    yy, xx = np.indices((height, width))
    stable_road = []
    actor_occlusion = []
    for index, flow in enumerate(flows):
        finite = np.isfinite(flow).all(axis=-1)
        next_x = np.rint(xx + np.where(finite, flow[..., 0], 0.0)).astype(np.int64)
        next_y = np.rint(yy + np.where(finite, flow[..., 1], 0.0)).astype(np.int64)
        inside = finite & (next_x >= 0) & (next_x < width) & (next_y >= 0) & (next_y < height)
        warped_next_road = np.zeros((height, width), dtype=bool)
        warped_next_actor = np.zeros((height, width), dtype=bool)
        next_road = _binary_dilate(road_frames[index + 1], road_dilation_px)
        next_actor = _binary_dilate(actor_frames[index + 1], actor_dilation_px)
        warped_next_road[inside] = next_road[next_y[inside], next_x[inside]]
        warped_next_actor[inside] = next_actor[next_y[inside], next_x[inside]]
        stable_road.append(
            _binary_dilate(road_frames[index], road_dilation_px) & warped_next_road
        )
        actor_occlusion.append(
            _binary_dilate(actor_frames[index], actor_dilation_px) | warped_next_actor
        )
    return PerceptionObservation(
        traversable_masks=np.stack(stable_road),
        actor_masks=np.stack(actor_occlusion),
        class_maps=np.asarray(frames.class_maps[:-1]),
        class_names=frames.class_names,
        backend=f"{frames.backend}_temporal_consensus",
        model_id=frames.model_id,
    )


class SegFormerPerception:
    """Cityscapes SegFormer wrapper with explicit class-name contracts."""

    def __init__(self, config: dict[str, Any], *, device: str) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "perception.backend=segformer requires the optional 'perception' dependencies"
            ) from error
        self.torch = torch
        self.device = device
        self.model_id = str(
            config.get("model_id", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        )
        local_files_only = bool(config.get("local_files_only", False))
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id, local_files_only=local_files_only
        )
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            self.model_id, local_files_only=local_files_only
        )
        self.model.to(device).eval()
        id2label = getattr(self.model.config, "id2label", {})
        self.class_names = tuple(str(id2label.get(index, index)) for index in range(len(id2label)))
        self.traversable_labels = {
            str(label).lower() for label in config.get("traversable_labels", ["road"])
        }
        self.actor_labels = {
            str(label).lower()
            for label in config.get(
                "actor_labels",
                ["car", "truck", "bus", "person", "rider", "bicycle", "motorcycle"],
            )
        }
        self.confidence_threshold = float(config.get("confidence_threshold", 0.55))
        self.backend = "segformer_cityscapes"

    def observe(
        self,
        frame_paths: list[str],
        *,
        target_size: tuple[int, int],
        intrinsics: np.ndarray,
        distortion: np.ndarray,
    ) -> PerceptionObservation:
        images = []
        for path in frame_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            if np.asarray(distortion).size:
                image = cv2.undistort(image, intrinsics, distortion, None, intrinsics)
            image = cv2.cvtColor(cv2.resize(image, target_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            images.append(image)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        logits = self.torch.nn.functional.interpolate(
            logits,
            size=(target_size[1], target_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = self.torch.softmax(logits, dim=1)
        confidence, class_maps = probabilities.max(dim=1)
        class_maps_np = class_maps.cpu().numpy().astype(np.int16)
        confidence_np = confidence.cpu().numpy().astype(np.float32)
        names = np.asarray([name.lower() for name in self.class_names], dtype=object)
        traversable_ids = np.flatnonzero(np.isin(names, list(self.traversable_labels)))
        actor_ids = np.flatnonzero(np.isin(names, list(self.actor_labels)))
        traversable = np.isin(class_maps_np, traversable_ids) & (
            confidence_np >= self.confidence_threshold
        )
        actors = np.isin(class_maps_np, actor_ids) & (
            confidence_np >= self.confidence_threshold
        )
        if not traversable_ids.size:
            raise ValueError(
                f"none of traversable_labels {sorted(self.traversable_labels)} exist in {self.model_id}"
            )
        return PerceptionObservation(
            traversable_masks=traversable,
            actor_masks=actors,
            class_maps=class_maps_np,
            class_names=self.class_names,
            backend=self.backend,
            model_id=self.model_id,
        )


def build_perception(config: dict[str, Any], *, device: str) -> SegFormerPerception | None:
    perception_config = config.get("perception", {})
    if not bool(perception_config.get("enabled", False)):
        return None
    backend = str(perception_config.get("backend", "segformer"))
    if backend != "segformer":
        raise ValueError(f"unsupported perception.backend: {backend}")
    return SegFormerPerception(perception_config, device=device)
