#!/usr/bin/env python3
"""Render representative NAVSIM batch examples as image and BEV overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from iac_new.counterfactual import dense_counterfactual_trajectories


ROI = np.asarray([[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]])


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank(result: dict) -> int | None:
    if not result.get("valid"):
        return None
    return list(result["candidate_order"]).index("logged") + 1


def _camera_to_ego(record: dict) -> np.ndarray:
    if record.get("camera_to_ego") is not None:
        return np.asarray(record["camera_to_ego"], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(record["sensor2lidar_rotation"], dtype=np.float64)
    matrix[:3, 3] = np.asarray(record["sensor2lidar_translation"], dtype=np.float64)
    return matrix


def _project(trajectory: np.ndarray, camera_to_ego: np.ndarray, intrinsic: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.column_stack([trajectory[:, 0], trajectory[:, 1], np.zeros(len(trajectory)), np.ones(len(trajectory))])
    camera_points = (np.linalg.inv(camera_to_ego) @ points.T).T[:, :3]
    valid = camera_points[:, 2] > 0.05
    pixels = np.full((len(trajectory), 2), np.nan, dtype=np.float64)
    projected = (intrinsic @ camera_points[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels


def _draw_polyline(image: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int], thickness: int = 4, dashed: bool = False) -> None:
    for first, second in zip(pixels[:-1], pixels[1:]):
        if np.all(np.isfinite(first)) and np.all(np.isfinite(second)):
            if not dashed:
                cv2.line(image, tuple(np.round(first).astype(int)), tuple(np.round(second).astype(int)), color, thickness, cv2.LINE_AA)
            else:
                start = np.asarray(first, dtype=np.float64)
                end = np.asarray(second, dtype=np.float64)
                distance = float(np.linalg.norm(end - start))
                for offset in np.arange(0.0, max(distance, 1.0), 18.0):
                    a = start + (end - start) * (offset / max(distance, 1e-6))
                    b = start + (end - start) * (min(offset + 9.0, distance) / max(distance, 1e-6))
                    cv2.line(image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, thickness, cv2.LINE_AA)
    valid = pixels[np.all(np.isfinite(pixels), axis=1)]
    if len(valid):
        cv2.circle(image, tuple(np.round(valid[-1]).astype(int)), 8, color, -1, cv2.LINE_AA)


def _trajectory_corridor(trajectory: np.ndarray, half_width_m: float = 1.1) -> np.ndarray:
    centers = np.vstack([np.zeros((1, 3), dtype=np.float64), np.asarray(trajectory, dtype=np.float64)])
    lateral = np.column_stack([-np.sin(centers[:, 2]), np.cos(centers[:, 2])])
    left = centers[:, :2] + float(half_width_m) * lateral
    right = centers[:, :2] - float(half_width_m) * lateral
    return np.vstack([
        np.column_stack([left, np.zeros(len(left))]),
        np.column_stack([right[::-1], np.zeros(len(right))]),
    ])


def _draw_prediction_support(
    image: np.ndarray,
    trajectories: list[np.ndarray],
    camera_to_ego: np.ndarray,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.22,
) -> bool:
    """Fill the union of selected vehicle corridors without convex-hull bridging."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    drawn = False
    for trajectory in trajectories:
        corridor = _trajectory_corridor(trajectory)
        pixels = _project(corridor, camera_to_ego, intrinsic, width, height)
        if not np.all(np.isfinite(pixels)):
            continue
        polygon = np.round(pixels).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255, cv2.LINE_AA)
        drawn = True
    if not drawn:
        return False
    overlay = image.copy()
    overlay[mask > 0] = color
    image[:] = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
    return True


def _front_overlay(record: dict, result: dict, candidates: dict[str, np.ndarray], title: str) -> np.ndarray:
    image = cv2.imread(record["history_images"][-1])
    if image is None:
        raise FileNotFoundError(record["history_images"][-1])
    distortion = np.asarray(record.get("camera_distortion", []), dtype=np.float64)
    intrinsic = np.asarray(record["camera_intrinsic"], dtype=np.float64)
    if distortion.size:
        image = cv2.undistort(image, intrinsic, distortion, None, intrinsic)
    height, width = image.shape[:2]
    canvas = image.copy()
    polygon = np.round(ROI * [width, height]).astype(np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [polygon], (255, 190, 80))
    canvas = cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0.0)
    camera_to_ego = _camera_to_ego(record)
    selected = set(result.get("prediction_set_ids", [])) if result.get("valid") else set()
    support_trajectories = [
        candidates[candidate_id] for candidate_id in selected if candidate_id in candidates
    ]
    _draw_prediction_support(
        canvas, support_trajectories, camera_to_ego, intrinsic, (70, 200, 70), alpha=0.20
    )
    for candidate_id in selected:
        if candidate_id in {"logged", result.get("top_candidate_id")}:
            continue
        _draw_polyline(canvas, _project(candidates[candidate_id], camera_to_ego, intrinsic, width, height), (130, 180, 130), 2)
    if "logged" in candidates:
        _draw_polyline(canvas, _project(candidates["logged"], camera_to_ego, intrinsic, width, height), (40, 40, 230), 9)
    top_id = result.get("top_candidate_id") if result.get("valid") else None
    if top_id and top_id in candidates:
        _draw_polyline(canvas, _project(candidates[top_id], camera_to_ego, intrinsic, width, height), (230, 80, 30), 5, dashed=(top_id == "logged"))
    cv2.polylines(canvas, [polygon], True, (30, 190, 240), 3, cv2.LINE_AA)
    lines = [title, f"scene={result['scene_id']}", f"GT rank={_rank(result) or 'abstain'}  top={top_id or 'not scored'}", "green=prediction support  red=GT  blue=selected  yellow=measurement ROI"]
    y = 42
    for line in lines:
        cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 10, 10), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 245, 245), 2, cv2.LINE_AA)
        y += 34
    return canvas


def _bev(record: dict, result: dict, candidates: dict[str, np.ndarray], title: str) -> np.ndarray:
    size = 720
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    scale = 42.0
    origin = np.asarray([120.0, size - 90.0])
    def point(xy: np.ndarray) -> tuple[int, int]:
        return tuple(np.round(origin + np.asarray([xy[1] * scale, -xy[0] * scale])).astype(int))
    for x in range(0, 13):
        cv2.line(canvas, point(np.asarray([x, -4.5])), point(np.asarray([x, 4.5])), (220, 220, 220), 1)
    for y in range(-4, 5):
        cv2.line(canvas, point(np.asarray([0, y])), point(np.asarray([12, y])), (220, 220, 220), 1)
    cv2.arrowedLine(canvas, point(np.asarray([0, 0])), point(np.asarray([1.2, 0])), (40, 40, 40), 4, cv2.LINE_AA, tipLength=0.25)
    selected = set(result.get("prediction_set_ids", [])) if result.get("valid") else set()
    support_trajectories = []
    for candidate_id in selected:
        if candidate_id in candidates:
            support_trajectories.append(np.asarray(candidates[candidate_id][:, :2], dtype=np.float64))
    if support_trajectories:
        support_mask = np.zeros((size, size), dtype=np.uint8)
        for trajectory in support_trajectories:
            points = np.asarray([point(xy) for xy in np.vstack([np.zeros((1, 2)), trajectory])], dtype=np.int32)
            cv2.polylines(support_mask, [points], False, 255, max(1, int(round(2.2 * scale))), cv2.LINE_AA)
        overlay = canvas.copy()
        overlay[support_mask > 0] = (70, 200, 70)
        canvas = cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0.0)
        contours, _ = cv2.findContours(support_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (70, 180, 70), 2, cv2.LINE_AA)
    for candidate_id in selected:
        if candidate_id in {"logged", result.get("top_candidate_id")}:
            continue
        pts = np.asarray([point(xy) for xy in candidates[candidate_id][:, :2]], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (130, 180, 130), 2, cv2.LINE_AA)
    if "logged" in candidates:
        pts = np.asarray([point(xy) for xy in candidates["logged"][:, :2]], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (40, 40, 230), 8, cv2.LINE_AA)
    top_id = result.get("top_candidate_id") if result.get("valid") else None
    if top_id and top_id in candidates:
        pts = np.asarray([point(xy) for xy in candidates[top_id][:, :2]], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (230, 80, 30), 4, cv2.LINE_AA)
    cv2.putText(canvas, title, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "forward x (m)", (430, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2, cv2.LINE_AA)
    cv2.putText(canvas, "left y (m)", (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = {row["source_key"]: row for row in _read_jsonl(args.records)}
    results = _read_jsonl(args.results)
    valid = [row for row in results if row.get("valid")]
    best = next(row for row in valid if _rank(row) == 1)
    worst = max(valid, key=lambda row: _rank(row) or -1)
    abstain = next((row for row in results if not row.get("valid")), None)
    chosen = [("rank1", best), ("worst", worst)]
    if abstain is not None:
        chosen.append(("abstain", abstain))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, result in chosen:
        record = records[result["sample_id"]]
        reference = np.asarray(record["trajectory"], dtype=np.float64)
        candidates = {"logged": reference}
        for candidate_id, trajectory, _ in dense_counterfactual_trajectories(reference):
            candidates[candidate_id] = trajectory
        front = _front_overlay(record, result, candidates, f"NAVSIM batch example: {label}")
        bev = _bev(record, result, candidates, f"{label}: green support / red GT / blue selected")
        front = cv2.resize(front, (960, 560), interpolation=cv2.INTER_AREA)
        bev = cv2.resize(bev, (560, 560), interpolation=cv2.INTER_AREA)
        montage = np.full((560, 1520, 3), 235, dtype=np.uint8)
        montage[:, :960] = front
        montage[:, 960:] = cv2.resize(bev, (560, 560), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.output_dir / f"{label}.png"), montage)
        (args.output_dir / f"{label}.json").write_text(json.dumps({
            "sample_id": result["sample_id"], "scene_id": result["scene_id"], "rank": _rank(result),
            "top_candidate_id": result.get("top_candidate_id"), "valid": bool(result.get("valid")),
            "abstain_reasons": result.get("abstain_reasons", []),
        }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"examples": [label for label, _ in chosen], "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
