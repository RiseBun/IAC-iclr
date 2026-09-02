"""Small dependency-free trajectory-region visualizer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _world_to_pixel(x_m: float, y_m: float, scale: float, origin: tuple[int, int]) -> tuple[int, int]:
    return int(round(origin[0] + y_m * scale)), int(round(origin[1] - x_m * scale))


def render_trajectory_region(
    *,
    output_path: Path,
    sample_id: str,
    mode_summaries: list[dict[str, Any]],
    region: dict[str, Any],
    observability: list[dict[str, Any]],
    width: int = 960,
    height: int = 640,
) -> None:
    """Render an ego-frame trajectory support view and observability bars."""
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    plot_width = int(width * 0.68)
    origin = (plot_width // 2, height - 72)
    selected = [mode for mode in mode_summaries if mode.get("selected")]
    all_states = [state for mode in selected for state in mode["trajectory_states"]]
    max_forward = max([state["x_m"] for state in all_states] + [5.0])
    max_lateral = max([abs(state["y_m"]) for state in all_states] + [2.0])
    scale = min((plot_width - 80) / (2.0 * max_lateral), (height - 120) / max_forward)

    # Ground-plane grid and ego footprint.
    for forward in np.arange(0.0, max_forward + 1.0, 5.0):
        p0 = _world_to_pixel(forward, -max_lateral, scale, origin)
        p1 = _world_to_pixel(forward, max_lateral, scale, origin)
        cv2.line(canvas, p0, p1, (224, 224, 224), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{forward:.0f}m", (p0[0] + 4, p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (125, 125, 125), 1, cv2.LINE_AA)
    for lateral in np.arange(-max_lateral, max_lateral + 0.5, 1.0):
        p0 = _world_to_pixel(0.0, lateral, scale, origin)
        p1 = _world_to_pixel(max_forward, lateral, scale, origin)
        cv2.line(canvas, p0, p1, (232, 232, 232), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (origin[0] - 12, origin[1] - 28), (origin[0] + 12, origin[1]), (35, 35, 35), -1)
    cv2.putText(canvas, "ego", (origin[0] + 18, origin[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1, cv2.LINE_AA)

    # Draw an envelope for the selected modes, then the coupled mode curves.
    if selected:
        envelope_points = np.asarray(
            [origin]
            + [
                _world_to_pixel(state["x_m"], state["y_m"], scale, origin)
                for mode in selected
                for state in mode["trajectory_states"]
            ],
            dtype=np.int32,
        )
        envelope = cv2.convexHull(envelope_points)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [envelope], (220, 235, 250))
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0.0)
    for mode_index, mode in enumerate(selected):
        hue = int((mode_index * 67) % 180)
        color = tuple(int(value) for value in cv2.cvtColor(np.uint8([[[hue, 210, 220]]]), cv2.COLOR_HSV2BGR)[0, 0])
        points = np.asarray(
            [_world_to_pixel(state["x_m"], state["y_m"], scale, origin) for state in mode["trajectory_states"]],
            dtype=np.int32,
        )
        thickness = max(2, int(round(2.0 + 5.0 * mode["probability"])))
        cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
        for point, state in zip(points, mode["trajectory_states"]):
            arrow_length = 13
            yaw = float(state["yaw_rad"])
            arrow_end = (
                int(round(point[0] + arrow_length * np.sin(yaw))),
                int(round(point[1] - arrow_length * np.cos(yaw))),
            )
            cv2.arrowedLine(
                canvas,
                tuple(int(value) for value in point),
                arrow_end,
                color,
                1,
                cv2.LINE_AA,
                tipLength=0.3,
            )
        endpoint = tuple(int(value) for value in points[-1])
        cv2.putText(canvas, f"{mode['candidate_id']} p={mode['probability']:.2f}", (endpoint[0] + 6, endpoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)

    # Right panel: coupled variable ranges and interval observability.
    panel_x = plot_width + 18
    cv2.putText(canvas, "joint trajectory region", (panel_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
    y = 54
    cv2.putText(
        canvas,
        f"mass={float(region['selected_probability_mass']):.2f}",
        (panel_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    y += 20
    for mode in selected[:6]:
        speeds = mode["speed_range_mps"]
        cv2.putText(canvas, f"{mode['candidate_id']}: v {speeds[0]:.1f}-{speeds[1]:.1f} m/s", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
        y += 20
    y += 8
    cv2.putText(canvas, "observability by interval", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
    y += 20
    bar_width = max(40, width - panel_x - 28)
    for item in observability:
        value = max(0.0, min(1.0, float(item["effective_static_pixel_fraction"])))
        color = (65, 165, 75) if item["status"] == "good" else (55, 120, 215)
        cv2.rectangle(canvas, (panel_x, y), (panel_x + bar_width, y + 11), (222, 222, 222), -1)
        cv2.rectangle(canvas, (panel_x, y), (panel_x + int(bar_width * value), y + 11), color, -1)
        cv2.putText(canvas, f"{item['interval_index']} {item['role']} {value:.2f}", (panel_x, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (65, 65, 65), 1, cv2.LINE_AA)
        y += 42
        if y > height - 20:
            break
    cv2.putText(canvas, f"{sample_id} | ego-frame trajectory support (not full BEV)", (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 25, 25), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"failed to write visualization: {output_path}")


def safe_visualization_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)) + ".png"


def _read_aligned_image(
    path: str,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if np.asarray(distortion).size:
        image = cv2.undistort(image, intrinsics, distortion, None, intrinsics)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def _draw_sparse_flow(
    image: np.ndarray,
    flow: np.ndarray,
    support: np.ndarray,
    color: tuple[int, int, int],
    *,
    step: int = 48,
) -> None:
    height, width = support.shape
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            if not support[y, x] or not np.isfinite(flow[y, x]).all():
                continue
            dx, dy = (float(value) for value in flow[y, x])
            end = (int(round(x + dx)), int(round(y + dy)))
            cv2.arrowedLine(
                image,
                (x, y),
                end,
                color,
                1,
                cv2.LINE_AA,
                tipLength=0.25,
            )


def render_image_diagnostics(
    *,
    output_path: Path,
    sample_id: str,
    frame_paths: list[str],
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    target_size: tuple[int, int],
    observed_flows: np.ndarray,
    consistency_masks: np.ndarray | None,
    roi_mask: np.ndarray,
    future_start: int,
    future_dynamic_weights: np.ndarray,
    top_future_predicted_flows: np.ndarray,
    observability: list[dict[str, Any]],
) -> None:
    """Render the image evidence actually used by the trajectory probe."""
    width, height = target_size
    if len(frame_paths) != len(observed_flows) + 1:
        raise ValueError("frame paths must contain one more item than optical-flow intervals")
    if len(observability) != len(observed_flows):
        raise ValueError("observability must match optical-flow intervals")
    panels: list[np.ndarray] = []
    roi = np.asarray(roi_mask, dtype=bool)
    contours, _ = cv2.findContours(
        (roi.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for interval_index, observed in enumerate(observed_flows):
        panel = _read_aligned_image(
            frame_paths[interval_index], intrinsics, distortion, target_size
        )
        if interval_index < future_start:
            reliability = roi.astype(np.float32)
            if consistency_masks is not None:
                reliability *= consistency_masks[interval_index].astype(np.float32)
            weight_label = "ROI/FB reliability"
            predicted = None
        else:
            future_index = interval_index - future_start
            reliability = np.asarray(future_dynamic_weights[future_index], dtype=np.float32)
            weight_label = "rigid static weight"
            predicted = top_future_predicted_flows[future_index]

        heat = np.zeros_like(panel)
        heat[..., 1] = np.asarray(255.0 * np.clip(reliability, 0.0, 1.0), dtype=np.uint8)
        heat[..., 2] = np.asarray(255.0 * (1.0 - np.clip(reliability, 0.0, 1.0)), dtype=np.uint8)
        blend = cv2.addWeighted(panel, 0.68, heat, 0.32, 0.0)
        panel[roi] = blend[roi]
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

        _draw_sparse_flow(panel, observed, roi, (255, 255, 0))
        if predicted is not None:
            _draw_sparse_flow(
                panel, predicted, roi & (reliability >= 0.25), (0, 180, 255)
            )

        item = observability[interval_index]
        title = (
            f"{interval_index}: {item['role']}"
            + (" boundary" if item.get("is_boundary") else "")
            + f" | support={float(item['effective_static_pixel_fraction']):.2f}"
        )
        cv2.rectangle(panel, (0, 0), (width, 42), (18, 18, 18), -1)
        cv2.putText(
            panel,
            title,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{weight_label} | {item['status']}",
            (8, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)

    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    header_height = 44
    canvas = np.full((header_height + rows * height, columns * width, 3), 25, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{sample_id} | rigid-flow evidence: green=retained static evidence amber=suppressed/uncertain | cyan=observed orange=predicted",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    for index, panel in enumerate(panels):
        row = index // columns
        column = index % columns
        y0 = header_height + row * height
        x0 = column * width
        canvas[y0 : y0 + height, x0 : x0 + width] = panel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"failed to write image diagnostics: {output_path}")


def _project_ego_point(
    point_ego: np.ndarray, camera_to_ego: np.ndarray, intrinsics: np.ndarray, width: int, height: int
) -> tuple[int, int] | None:
    point_camera = np.linalg.inv(np.asarray(camera_to_ego, dtype=np.float64)) @ np.r_[point_ego, 1.0]
    if point_camera[2] <= 1e-6:
        return None
    projected = np.asarray(intrinsics, dtype=np.float64) @ point_camera[:3]
    x = float(projected[0] / projected[2])
    y = float(projected[1] / projected[2])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return int(round(x)), int(round(y))


def render_image_trajectory_overlay(
    *,
    output_path: Path,
    sample_id: str,
    frame_path: str,
    intrinsics: np.ndarray,
    source_intrinsics: np.ndarray | None = None,
    distortion: np.ndarray,
    target_size: tuple[int, int],
    camera_to_ego: np.ndarray,
    mode_summaries: list[dict[str, Any]],
    traversable_mask: np.ndarray | None,
    actor_mask: np.ndarray | None,
    show_drivable: bool = False,
    trajectory_half_width_m: float = 1.1,
) -> None:
    """Project selected ego-frame trajectory hypotheses onto the front image.

    The road mask is an internal soft prior, not a direction label.  It is
    therefore hidden by default so that the rendered arrows cannot be
    mistaken for a drivable-area prediction.
    """
    width, height = target_size
    panel = _read_aligned_image(
        frame_path,
        intrinsics if source_intrinsics is None else source_intrinsics,
        distortion,
        target_size,
    )
    if show_drivable and traversable_mask is not None:
        traversable = np.asarray(traversable_mask, dtype=bool)
        overlay = panel.copy()
        overlay[traversable] = (60, 190, 70)
        panel = cv2.addWeighted(panel, 0.68, overlay, 0.32, 0.0)
    if actor_mask is not None:
        actors = np.asarray(actor_mask, dtype=bool)
        contours, _ = cv2.findContours(
            actors.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(panel, contours, -1, (30, 40, 230), 2, cv2.LINE_AA)

    selected = [mode for mode in mode_summaries if mode.get("selected")]
    for mode_index, mode in enumerate(selected):
        color = (60, 220, 80) if mode_index == 0 else (220, 170, 30)
        points: list[tuple[int, int]] = []
        left_points: list[tuple[int, int]] = []
        right_points: list[tuple[int, int]] = []
        for state in mode["trajectory_states"]:
            projected = _project_ego_point(
                np.asarray([state["x_m"], state["y_m"], 0.0]),
                camera_to_ego,
                intrinsics,
                width,
                height,
            )
            if projected is not None:
                points.append(projected)
            yaw = float(state["yaw_rad"])
            center = np.asarray([state["x_m"], state["y_m"], 0.0], dtype=np.float64)
            lateral = np.asarray([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float64)
            left = _project_ego_point(
                center + float(trajectory_half_width_m) * lateral,
                camera_to_ego,
                intrinsics,
                width,
                height,
            )
            right = _project_ego_point(
                center - float(trajectory_half_width_m) * lateral,
                camera_to_ego,
                intrinsics,
                width,
                height,
            )
            if left is not None and right is not None:
                left_points.append(left)
                right_points.append(right)
        if len(points) >= 2:
            if len(left_points) == len(right_points) and len(left_points) >= 2:
                corridor = np.asarray(left_points + right_points[::-1], dtype=np.int32)
                corridor_overlay = panel.copy()
                cv2.fillPoly(corridor_overlay, [corridor], color)
                panel = cv2.addWeighted(panel, 0.78, corridor_overlay, 0.22, 0.0)
            cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, 4, cv2.LINE_AA)
            for point_index, point in enumerate(points):
                cv2.circle(panel, point, 4, color, -1, cv2.LINE_AA)
                if point_index > 0:
                    cv2.arrowedLine(
                        panel,
                        points[point_index - 1],
                        point,
                        color,
                        2,
                        cv2.LINE_AA,
                        tipLength=0.28,
                    )
            cv2.putText(
                panel,
                f"traj {mode['candidate_id']} p={float(mode['probability']):.2f}",
                (points[-1][0] + 6, points[-1][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.rectangle(panel, (0, 0), (width, 50), (18, 18, 18), -1)
    cv2.putText(panel, f"{sample_id} | projected ego-frame direction", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
    legend = "green=IAC direction arrow | red contour=dynamic/actor evidence"
    if show_drivable:
        legend += " | drivable=soft prior"
    cv2.putText(panel, legend, (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (215, 215, 215), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), panel):
        raise OSError(f"failed to write trajectory overlay: {output_path}")


def render_metric_depth_diagnostics(
    *,
    output_path: Path,
    sample_id: str,
    frame_paths: list[str],
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    target_size: tuple[int, int],
    depths_m: np.ndarray,
    confidence: np.ndarray,
    reliability_masks: np.ndarray,
    source: str,
) -> None:
    """Overlay metric depth and the exact candidate-independent support mask."""
    width, height = target_size
    depths = np.asarray(depths_m, dtype=np.float32)
    confidences = np.asarray(confidence, dtype=np.float32)
    masks = np.asarray(reliability_masks, dtype=bool)
    if not (depths.shape == confidences.shape == masks.shape):
        raise ValueError("depth, confidence, and reliability shapes must match")
    if len(frame_paths) != len(depths):
        raise ValueError("one current-frame image is required per depth interval")

    panels: list[np.ndarray] = []
    for index, (path, depth, conf, mask) in enumerate(
        zip(frame_paths, depths, confidences, masks)
    ):
        image = _read_aligned_image(path, intrinsics, distortion, target_size)
        finite = np.isfinite(depth) & (depth > 0.0)
        log_depth = np.zeros_like(depth, dtype=np.float32)
        if finite.any():
            low, high = np.quantile(np.log(depth[finite]), [0.02, 0.98])
            log_depth[finite] = np.clip(
                (np.log(depth[finite]) - low) / max(float(high - low), 1e-6), 0.0, 1.0
            )
        color = cv2.applyColorMap(
            np.asarray(255.0 * (1.0 - log_depth), dtype=np.uint8), cv2.COLORMAP_TURBO
        )
        blend = cv2.addWeighted(image, 0.45, color, 0.55, 0.0)
        panel = np.asarray(image * 0.28, dtype=np.uint8)
        panel[mask] = blend[mask]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        median_depth = float(np.median(depth[mask])) if mask.any() else float("nan")
        median_conf = float(np.median(conf[mask])) if mask.any() else float("nan")
        cv2.rectangle(panel, (0, 0), (width, 39), (18, 18, 18), -1)
        cv2.putText(
            panel,
            f"interval {index} | depth={median_depth:.1f}m | support={mask.mean():.2f}",
            (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"median confidence={median_conf:.2f}",
            (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA,
        )
        panels.append(panel)

    header_height = 44
    canvas = np.full((header_height + height, len(panels) * width, 3), 25, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{sample_id} | {source} | color=metric depth, dark=excluded evidence",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA,
    )
    for index, panel in enumerate(panels):
        canvas[header_height:, index * width : (index + 1) * width] = panel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"failed to write metric-depth diagnostics: {output_path}")
