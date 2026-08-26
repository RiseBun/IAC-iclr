#!/usr/bin/env python3
"""Render NAVSIM image-side maneuver decisions frame by frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from iac_new.maneuver import extract_maneuver


COLORS = {
    "straight": (90, 190, 245),
    "left": (70, 190, 70),
    "right": (210, 130, 60),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reference_trajectory(record: dict[str, Any]) -> np.ndarray:
    if record.get("realized_future_ego_state") is not None:
        return np.asarray(record["realized_future_ego_state"], dtype=np.float64)[:, :3]
    if record.get("trajectory") is not None and not record.get("candidates"):
        return np.asarray(record["trajectory"], dtype=np.float64)
    for candidate in record["candidates"]:
        if str(candidate["candidate_id"]) == str(record["gt_candidate_id"]):
            return np.asarray(candidate["trajectory"], dtype=np.float64)
    raise KeyError(record["sample_id"])


def project(point: np.ndarray, camera_to_ego: np.ndarray, intrinsics: np.ndarray, shape: tuple[int, int]) -> tuple[int, int] | None:
    camera_point = np.linalg.inv(camera_to_ego) @ np.r_[point, 1.0]
    if camera_point[2] <= 1e-5:
        return None
    uv = intrinsics @ camera_point[:3]
    x, y = float(uv[0] / uv[2]), float(uv[1] / uv[2])
    height, width = shape
    return (int(round(x)), int(round(y))) if 0 <= x < width and 0 <= y < height else None


def draw_direction(image: np.ndarray, trajectory: np.ndarray, index: int, camera_to_ego: np.ndarray, intrinsics: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    # The event label is based on signed heading change (delta-yaw).  The
    # arrow must therefore show the vehicle heading at the state, not the
    # displacement from the ego origin to the next trajectory point.  Those
    # vectors can legitimately point to opposite sides during a turn.
    start = np.zeros(3, dtype=np.float64) if index == 0 else trajectory[index - 1]
    yaw = float(trajectory[index, 2])
    heading_length_m = 2.5
    end = np.asarray(
        [
            float(start[0]) + heading_length_m * np.cos(yaw),
            float(start[1]) + heading_length_m * np.sin(yaw),
            0.0,
        ],
        dtype=np.float64,
    )
    p0 = project(np.asarray([start[0], start[1], 0.0]), camera_to_ego, intrinsics, image.shape[:2])
    p1 = project(end, camera_to_ego, intrinsics, image.shape[:2])
    if p1 is None:
        p0 = (int(round(float(intrinsics[0, 2]))), image.shape[0] - 5)
        # Ego y is positive to the left; image x is positive to the right.
        # The minus sign is required when the ground projection is invalid.
        p1 = (
            int(round(p0[0] - 85.0 * np.sin(yaw))),
            int(round(p0[1] - 72.0 * np.cos(yaw))),
        )
    if p0 is None:
        # The ego origin is normally below the front-view image. Anchor the
        # first visible ground segment at the bottom centre of the frame.
        p0 = (int(round(float(intrinsics[0, 2]))), image.shape[0] - 5)
    cv2.arrowedLine(image, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.25)


def panel(path: str, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(path)
    if image is None:
        image = np.full((560, 960, 3), 70, dtype=np.uint8)
    original = (image.shape[1], image.shape[0])
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), original


def render(record: dict[str, Any], score: dict[str, Any], output: Path) -> None:
    predicted = np.asarray(score["decoder"]["trajectory"], dtype=np.float64)
    reference = reference_trajectory(record)
    support = (score.get("decoder") or {}).get("profile_support") or []
    times = np.asarray([item["time_s"] for item in support], dtype=np.float64) if len(support) == len(predicted) else np.asarray(record["future_times_s"], dtype=np.float64)[: len(predicted)]
    reference = reference[: len(predicted)]
    pred_event = extract_maneuver(predicted, times)
    ref_event = extract_maneuver(reference, times)
    history_paths = record.get("history_frame_paths", record.get("history_images", []))
    future_paths = record.get("future_frame_paths", record.get("future_images", []))
    paths = list(history_paths)[-4:] + list(future_paths)
    panel_size = (320, 186)
    grid = np.full((418, 1280, 3), 24, dtype=np.uint8)
    camera_to_ego = np.asarray(record["camera_to_ego"], dtype=np.float64)
    source_intrinsics = np.asarray(record.get("intrinsics", record.get("camera_intrinsic")), dtype=np.float64)
    for frame_index, path in enumerate(paths[:8]):
        image, original = panel(path, panel_size)
        intrinsics = source_intrinsics.copy()
        intrinsics[0] *= panel_size[0] / original[0]
        intrinsics[1] *= panel_size[1] / original[1]
        cv2.rectangle(image, (0, 0), (panel_size[0], 32), (18, 18, 18), -1)
        if frame_index < 4:
            title = f"history {frame_index + 1}"
        else:
            interval = frame_index - 4
            truth = ref_event["lateral_action"][interval]
            observed = pred_event["lateral_action"][interval]
            title = f"future {interval + 1}  GT:{truth}  IAC:{observed}"
            draw_direction(image, reference, interval, camera_to_ego, intrinsics, (35, 45, 225), 8)
            draw_direction(image, predicted, interval, camera_to_ego, intrinsics, (45, 210, 75), 4)
            border = (45, 180, 55) if truth == observed else (35, 35, 225)
            cv2.rectangle(image, (2, 2), (panel_size[0] - 3, panel_size[1] - 3), border, 3)
        cv2.putText(image, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1, cv2.LINE_AA)
        row, col = frame_index // 4, frame_index % 4
        grid[34 + row * panel_size[1] : 34 + (row + 1) * panel_size[1], col * panel_size[0] : (col + 1) * panel_size[0]] = image
    cv2.putText(grid, f"{record['sample_id']} | green=IAC heading | red=reference heading", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)

    timeline = np.full((418, 480, 3), 248, dtype=np.uint8)
    cv2.putText(timeline, "Direction event timeline", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    for row_index, (name, values) in enumerate((("GT", ref_event["lateral_action"]), ("IAC", pred_event["lateral_action"]))):
        y = 82 + row_index * 100
        cv2.putText(timeline, name, (18, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 2, cv2.LINE_AA)
        for index, value in enumerate(values):
            x = 78 + index * 95
            cv2.rectangle(timeline, (x, y), (x + 85, y + 58), COLORS[value], -1)
            cv2.putText(timeline, value, (x + 5, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(timeline, f"{times[index]:.1f}s", (x + 22, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
    matches = [a == b for a, b in zip(ref_event["lateral_action"], pred_event["lateral_action"])]
    accuracy = float(np.mean(matches))
    cv2.putText(timeline, f"interval accuracy: {accuracy:.1%}", (18, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(timeline, "Rule: heading change only", (18, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (65, 65, 65), 1, cv2.LINE_AA)
    cv2.putText(timeline, "Arrow: instantaneous ego heading", (18, 374), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (65, 65, 65), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate([grid, timeline], axis=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--sample-id", action="append", default=[], help="render only this sample id; repeatable")
    args = parser.parse_args()
    records = {row["sample_id"]: row for row in read_jsonl(args.manifest)}
    scores = [row for row in read_jsonl(args.scores) if row["sample_id"] in records]
    if args.sample_id:
        requested = set(args.sample_id)
        scores = [row for row in scores if row["sample_id"] in requested]
    ranked = []
    for score in scores:
        record = records[score["sample_id"]]
        event = extract_maneuver(reference_trajectory(record), np.asarray(record["future_times_s"]))
        priority = 0 if any(value != "straight" for value in event["lateral_action"]) else 1
        ranked.append((priority, score["sample_id"], score))
    for index, (_, sample_id, score) in enumerate(sorted(ranked)[: args.max_scenes]):
        render(records[sample_id], score, args.output_dir / f"event_{index:03d}_{sample_id}.png")
    print(json.dumps({"rendered": min(len(ranked), args.max_scenes), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
