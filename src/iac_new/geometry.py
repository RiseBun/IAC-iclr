"""Camera-motion geometry from a road plane or per-frame metric depth."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def se2_to_transform(x_m: float, y_m: float, yaw_rad: float) -> np.ndarray:
    cosine = np.cos(float(yaw_rad))
    sine = np.sin(float(yaw_rad))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform[:3, 3] = [float(x_m), float(y_m), 0.0]
    return transform


def candidate_camera_poses(
    candidate_trajectory: np.ndarray, camera_to_ego: np.ndarray
) -> list[np.ndarray]:
    """Return camera-to-anchor poses for the anchor and every future knot."""
    trajectory = np.asarray(candidate_trajectory, dtype=np.float64)
    camera_to_ego = np.asarray(camera_to_ego, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(f"candidate trajectory must have shape [T,3], got {trajectory.shape}")
    if camera_to_ego.shape != (4, 4):
        raise ValueError(f"camera_to_ego must have shape [4,4], got {camera_to_ego.shape}")
    ego_poses = [np.eye(4, dtype=np.float64)] + [
        se2_to_transform(*knot) for knot in trajectory
    ]
    return [ego_to_anchor @ camera_to_ego for ego_to_anchor in ego_poses]


def adjacent_camera_transforms(camera_poses: Iterable[np.ndarray]) -> list[np.ndarray]:
    poses = [np.asarray(pose, dtype=np.float64) for pose in camera_poses]
    if len(poses) < 2 or any(pose.shape != (4, 4) for pose in poses):
        raise ValueError("at least two [4,4] camera poses are required")
    return [
        np.linalg.inv(next_pose) @ current_pose
        for current_pose, next_pose in zip(poses, poses[1:])
    ]


def ground_plane_in_camera(camera_to_anchor: np.ndarray) -> tuple[np.ndarray, float]:
    """Return n,d for the anchor ground plane n.T X + d = 0 in camera coordinates."""
    pose = np.asarray(camera_to_anchor, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"camera pose must have shape [4,4], got {pose.shape}")
    anchor_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    normal_camera = pose[:3, :3].T @ anchor_normal
    distance = float(anchor_normal @ pose[:3, 3])
    if abs(distance) < 1e-6:
        raise ValueError("camera height above the ground plane is zero")
    return normal_camera, distance


def ground_plane_homography(
    intrinsics: np.ndarray,
    next_camera_from_current: np.ndarray,
    current_camera_to_anchor: np.ndarray,
) -> np.ndarray:
    """Map current pixels to the next frame for a locally planar road."""
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(next_camera_from_current, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape [3,3], got {intrinsics.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"camera transform must have shape [4,4], got {transform.shape}")
    normal, distance = ground_plane_in_camera(current_camera_to_anchor)
    rotation = transform[:3, :3]
    translation = transform[:3, 3:4]
    normalized = rotation - translation @ normal.reshape(1, 3) / distance
    homography = intrinsics @ normalized @ np.linalg.inv(intrinsics)
    return homography / homography[2, 2]


def homography_flow(
    homography: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a homography to dense forward flow and an in-frame validity mask."""
    homography = np.asarray(homography, dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError(f"homography must have shape [3,3], got {homography.shape}")
    yy, xx = np.indices((height, width), dtype=np.float64)
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=0).reshape(3, -1)
    projected = homography @ pixels
    denominator = projected[2]
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
    next_x = projected[0] / np.where(valid, denominator, 1.0)
    next_y = projected[1] / np.where(valid, denominator, 1.0)
    valid &= np.isfinite(next_x) & np.isfinite(next_y)
    valid &= (next_x >= 0.0) & (next_x <= width - 1)
    valid &= (next_y >= 0.0) & (next_y <= height - 1)
    flow = np.stack([next_x - pixels[0], next_y - pixels[1]], axis=1)
    flow[~valid] = np.nan
    return flow.reshape(height, width, 2).astype(np.float32), valid.reshape(height, width)


def rigid_flow_from_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    next_camera_from_current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project current-frame z-depth through an ego-motion hypothesis."""
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(next_camera_from_current, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H,W], got {depth.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape [3,3], got {intrinsics.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"camera transform must have shape [4,4], got {transform.shape}")

    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=0).reshape(3, -1)
    flat_depth = depth.reshape(-1)
    points_current = np.linalg.inv(intrinsics) @ pixels
    points_current *= flat_depth.reshape(1, -1)
    points_next = transform[:3, :3] @ points_current + transform[:3, 3:4]
    next_z = points_next[2]
    valid = np.isfinite(flat_depth) & (flat_depth > 0.0) & (next_z > 1e-6)
    projected = intrinsics @ points_next
    next_x = projected[0] / np.where(valid, projected[2], 1.0)
    next_y = projected[1] / np.where(valid, projected[2], 1.0)
    valid &= np.isfinite(next_x) & np.isfinite(next_y)
    valid &= (next_x >= 0.0) & (next_x <= width - 1)
    valid &= (next_y >= 0.0) & (next_y <= height - 1)
    flow = np.stack([next_x - pixels[0], next_y - pixels[1]], axis=1)
    flow[~valid] = np.nan
    return flow.reshape(height, width, 2).astype(np.float32), valid.reshape(height, width)


def scale_intrinsics(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image dimensions must be positive")
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    scaled[0, :] *= target_width / source_width
    scaled[1, :] *= target_height / source_height
    return scaled
