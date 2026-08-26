#!/usr/bin/env python3
"""Render front-view direction panels plus an ego-frame maneuver support view."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from iac_new.maneuver import extract_maneuver


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resize_with_pad(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _scale_intrinsics(intrinsics: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    scaled[0, :] *= target_width / max(source_width, 1)
    scaled[1, :] *= target_height / max(source_height, 1)
    return scaled


def _project_ground(point: np.ndarray, camera_to_ego: np.ndarray, intrinsics: np.ndarray, size: tuple[int, int]) -> tuple[int, int] | None:
    width, height = size
    camera_from_ego = np.linalg.inv(np.asarray(camera_to_ego, dtype=np.float64))
    point_camera = camera_from_ego @ np.r_[np.asarray(point, dtype=np.float64), 1.0]
    if point_camera[2] <= 1e-5:
        return None
    projected = np.asarray(intrinsics, dtype=np.float64) @ point_camera[:3]
    if abs(float(projected[2])) <= 1e-8:
        return None
    x = float(projected[0] / projected[2])
    y = float(projected[1] / projected[2])
    if not (np.isfinite(x) and np.isfinite(y)) or x < 0 or x >= width or y < 0 or y >= height:
        return None
    return int(round(x)), int(round(y))


def _draw_ground_direction(
    image: np.ndarray,
    trajectory: np.ndarray,
    segment_index: int,
    color: tuple[int, int, int],
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    """Draw a trajectory segment after ground-plane projection into the image."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    index = min(max(int(segment_index), 0), len(trajectory) - 1)
    start = np.zeros(3, dtype=np.float64) if index == 0 else trajectory[index - 1]
    end = trajectory[index]
    start_pixel = _project_ground([start[0], start[1], 0.0], camera_to_ego, intrinsics, (image.shape[1], image.shape[0]))
    end_pixel = _project_ground([end[0], end[1], 0.0], camera_to_ego, intrinsics, (image.shape[1], image.shape[0]))
    if start_pixel is None or end_pixel is None:
        return
    cv2.arrowedLine(image, start_pixel, end_pixel, color, 3, cv2.LINE_AA, tipLength=0.28)


def _world_to_pixel(x: float, y: float, *, scale: float, origin: tuple[int, int]) -> tuple[int, int]:
    return int(round(origin[0] + y * scale)), int(round(origin[1] - x * scale))


def _draw_bev(
    size: tuple[int, int],
    action: np.ndarray,
    imagined: np.ndarray,
    support: list[dict[str, Any]] | None,
    action_maneuver: dict[str, Any],
    imagined_maneuver: dict[str, Any],
) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    max_x = max(8.0, float(np.max(action[:, 0])), float(np.max(imagined[:, 0])))
    max_y = max(3.0, float(np.max(np.abs(action[:, 1]))), float(np.max(np.abs(imagined[:, 1]))) + 1.5)
    scale = min((height - 110) / max_x, (width - 100) / (2.0 * max_y))
    origin = (width // 2, height - 58)
    for x in np.arange(0.0, max_x + 1.0, 2.0):
        p0 = _world_to_pixel(x, -max_y, scale=scale, origin=origin)
        p1 = _world_to_pixel(x, max_y, scale=scale, origin=origin)
        cv2.line(canvas, p0, p1, (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{x:.0f}m", (p0[0] + 3, p0[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (125, 125, 125), 1, cv2.LINE_AA)
    for y in np.arange(-max_y, max_y + 0.5, 1.0):
        cv2.line(canvas, _world_to_pixel(0.0, y, scale=scale, origin=origin), _world_to_pixel(max_x, y, scale=scale, origin=origin), (232, 232, 232), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (origin[0] - 10, origin[1] - 24), (origin[0] + 10, origin[1]), (35, 35, 35), -1)
    cv2.putText(canvas, "ego", (origin[0] + 15, origin[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)

    if support:
        left = []
        right = []
        for point, item in zip(imagined, support):
            x = float(point[0])
            left.append(_world_to_pixel(x, float(item["y_m"]["q05"]), scale=scale, origin=origin))
            right.append(_world_to_pixel(x, float(item["y_m"]["q95"]), scale=scale, origin=origin))
        if len(left) >= 2:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [np.asarray(left + right[::-1], dtype=np.int32)], (210, 235, 250))
            canvas = cv2.addWeighted(canvas, 0.58, overlay, 0.42, 0.0)
            cv2.polylines(canvas, [np.asarray(left, dtype=np.int32)], False, (210, 130, 40), 1, cv2.LINE_AA)
            cv2.polylines(canvas, [np.asarray(right, dtype=np.int32)], False, (210, 130, 40), 1, cv2.LINE_AA)

    def points(traj: np.ndarray) -> np.ndarray:
        return np.asarray([_world_to_pixel(float(row[0]), float(row[1]), scale=scale, origin=origin) for row in traj], dtype=np.int32)

    action_points = points(action)
    imagined_points = points(imagined)
    cv2.polylines(canvas, [action_points], False, (35, 55, 220), 4, cv2.LINE_AA)
    cv2.polylines(canvas, [imagined_points], False, (45, 185, 65), 4, cv2.LINE_AA)
    for row, point in zip(imagined, imagined_points):
        yaw = float(row[2])
        end = (int(round(point[0] + 18 * np.sin(yaw))), int(round(point[1] - 18 * np.cos(yaw))))
        cv2.arrowedLine(canvas, tuple(point), end, (45, 185, 65), 2, cv2.LINE_AA, tipLength=0.28)
    cv2.putText(canvas, "red=conditioned action | green=IAC image trajectory | blue=support", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "action: " + " -> ".join(action_maneuver["segment_types"]), (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 55, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "image: " + " -> ".join(imagined_maneuver["segment_types"]), (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 150, 55), 1, cv2.LINE_AA)
    return canvas


def render_group(group: dict[str, Any], rows: list[dict[str, Any]], output: Path, branch_mode: str) -> None:
    by_branch = {str(row["branch_id"]): row for row in rows}
    selected = next((branch for branch in group["branches"] if branch["branch_id"].endswith(f"branch={branch_mode}")), group["branches"][0])
    source = by_branch.get(selected["branch_id"], rows[0])
    history = list(source.get("history_images") or [])[-4:]
    future = list(source.get("future_images") or [])
    times = np.asarray(source.get("future_times_s"), dtype=np.float64)
    action = np.asarray(selected["executed_action"], dtype=np.float64)
    imagined = np.asarray(selected["imagined_future"], dtype=np.float64)
    action_maneuver = selected.get("action_maneuver") or extract_maneuver(action, times)
    imagined_maneuver = selected.get("imagined_maneuver") or extract_maneuver(imagined, times)
    source_intrinsics = np.asarray(source.get("camera_intrinsic", source.get("intrinsics")), dtype=np.float64)
    camera_to_ego = np.asarray(source["camera_to_ego"], dtype=np.float64)
    panel_w, panel_h = 300, 190
    left = np.full((panel_h * 2 + 42, panel_w * 4, 3), 22, dtype=np.uint8)
    cv2.putText(left, f"{group['counterfactual_group_id']} | {branch_mode} | action: {' -> '.join(action_maneuver['segment_types'])} | image: {' -> '.join(imagined_maneuver['segment_types'])}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)
    paths = history + future
    for index in range(min(8, len(paths))):
        image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            image = np.full((panel_h, panel_w, 3), 65, dtype=np.uint8)
        panel = _resize_with_pad(image, (panel_w, panel_h))
        if index < len(history):
            segment_index = 0
            label = f"history {index}"
        else:
            segment_index = min(index - len(history), len(imagined) - 1)
            label = f"future t={times[segment_index]:.2f}s"
        cv2.rectangle(panel, (0, 0), (panel_w, 28), (18, 18, 18), -1)
        cv2.putText(panel, label, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
        original_size = (int(image.shape[1]), int(image.shape[0]))
        panel_intrinsics = _scale_intrinsics(source_intrinsics, original_size, (panel_w, panel_h))
        _draw_ground_direction(panel, action, segment_index, (45, 60, 230), camera_to_ego, panel_intrinsics)
        _draw_ground_direction(panel, imagined, segment_index, (50, 200, 70), camera_to_ego, panel_intrinsics)
        yaw_action = float(action[segment_index, 2])
        yaw_image = float(imagined[segment_index, 2])
        cv2.putText(panel, f"A yaw={yaw_action:+.2f}", (7, panel_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (45, 60, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, f"I yaw={yaw_image:+.2f}", (7, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (50, 180, 65), 1, cv2.LINE_AA)
        row, col = index // 4, index % 4
        y0, x0 = 42 + row * panel_h, col * panel_w
        left[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel

    bev = _draw_bev((560, left.shape[0]), action, imagined, selected.get("imagined_support"), action_maneuver, imagined_maneuver)
    canvas = np.concatenate([left, bev], axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch-mode", default="logged")
    parser.add_argument("--max-scenes", type=int, default=10)
    parser.add_argument("--require-composition", action="store_true")
    args = parser.parse_args()
    rows = _read_jsonl(args.manifest)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["counterfactual_group_id"])].append(row)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    rendered = 0
    for group in matrix.get("groups", []):
        group_id = str(group["counterfactual_group_id"])
        group_rows = by_group.get(group_id, [])
        if not group_rows:
            continue
        branch = next((item for item in group["branches"] if item["branch_id"].endswith(f"branch={args.branch_mode}")), group["branches"][0])
        skeleton = branch.get("action_maneuver") or extract_maneuver(np.asarray(branch["executed_action"]), np.asarray(group_rows[0]["future_times_s"]))
        if args.require_composition and len(skeleton.get("segment_types", [])) < 2:
            continue
        render_group(group, group_rows, args.output_dir / f"scene_{rendered:03d}.png", args.branch_mode)
        rendered += 1
        if rendered >= args.max_scenes:
            break
    print(json.dumps({"rendered": rendered, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
