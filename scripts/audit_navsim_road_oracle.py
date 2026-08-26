#!/usr/bin/env python3
"""Build a NAVSIM LiDAR road-support oracle and audit its boundaries.

This is deliberately independent of SegFormer and RAFT.  It projects ground
LiDAR returns into the anchor camera, dilates the sparse support into a soft
road corridor, and reports boundary observability.  The result is an upper
bound for the road-structure part of IAC rather than a dense semantic label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from iac_new.road_structure import extract_road_boundaries
from iac_new.geometry import scale_intrinsics
try:
    from scripts.audit_navsim_oracle_flow import (
        _frame_lookup,
        _sparse_depth,
        _transform,
        read_binary_pcd_xyz,
    )
except ModuleNotFoundError:
    from audit_navsim_oracle_flow import (
        _frame_lookup,
        _sparse_depth,
        _transform,
        read_binary_pcd_xyz,
    )


def _road_mask_from_lidar(ground_mask: np.ndarray, dilation_px: int) -> np.ndarray:
    value = np.asarray(ground_mask, dtype=bool)
    height, width = value.shape
    # Interpolate row-wise left/right ground support instead of dilating every
    # return. This avoids turning points near parked vehicles into a solid blob.
    rows: list[int] = []
    left: list[float] = []
    right: list[float] = []
    step = max(int(dilation_px), 2)
    for y in range(0, height, step):
        ys, xs = np.nonzero(value[max(0, y - step // 2):min(height, y + step // 2 + 1)])
        if len(xs) < 3:
            continue
        rows.append(y)
        left.append(float(np.quantile(xs, 0.08)))
        right.append(float(np.quantile(xs, 0.92)))
    if len(rows) < 2:
        return value
    row_arr = np.asarray(rows, dtype=np.float64)
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    support = np.zeros_like(value)
    y_min, y_max = int(row_arr.min()), int(row_arr.max())
    for y in range(y_min, min(height, y_max + step)):
        l = int(np.clip(np.interp(y, row_arr, left_arr), 0, width - 1))
        r = int(np.clip(np.interp(y, row_arr, right_arr), 0, width - 1))
        if r > l:
            support[y, l:r + 1] = True
    return support


def _render(path: Path, image_path: str, mask: np.ndarray, boundaries: dict[str, Any]) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return
    image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = image.copy()
    overlay[mask] = (60, 180, 60)
    image = cv2.addWeighted(image, 0.72, overlay, 0.28, 0.0)
    if boundaries.get("valid"):
        rows = np.asarray(boundaries["rows"], dtype=np.int64)
        left = np.asarray(boundaries["left_x"], dtype=np.int64)
        right = np.asarray(boundaries["right_x"], dtype=np.int64)
        for x_values, color in ((left, (255, 180, 40)), (right, (255, 180, 40))):
            points = np.column_stack([x_values, rows]).reshape(-1, 1, 2)
            cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
    cv2.putText(image, "green=LiDAR ground support  yellow=oracle boundaries", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "green=LiDAR ground support  yellow=oracle boundaries", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--visualize-count", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--dilation-px", type=int, default=7)
    parser.add_argument("--ground-z-min", type=float, default=-0.8)
    parser.add_argument("--ground-z-max", type=float, default=0.2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_records:
        rows = rows[: args.max_records]
    lookup_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    results = []
    for index, record in enumerate(rows):
        source_pkl = Path(record["source_pkl"])
        if source_pkl not in lookup_cache:
            lookup_cache[source_pkl] = _frame_lookup(source_pkl, record["camera"])
        lookup = lookup_cache[source_pkl]
        history = list(record.get("history_images") or [])
        all_images = history + list(record.get("future_images") or [])
        if not history or not all_images:
            continue
        anchor_path = history[-1]
        frame = lookup[Path(anchor_path).name]
        camera_row = frame["cams"][record["camera"]]
        camera_to_lidar = _transform(camera_row["sensor2lidar_rotation"], camera_row["sensor2lidar_translation"])
        points = read_binary_pcd_xyz(args.sensor_root / frame["lidar_path"])
        intrinsics = np.asarray(record["camera_intrinsic"], dtype=np.float64)
        source_image = cv2.imread(anchor_path, cv2.IMREAD_COLOR)
        if source_image is None:
            continue
        source_size = (int(source_image.shape[1]), int(source_image.shape[0]))
        intrinsics = scale_intrinsics(intrinsics, source_size, (args.width, args.height))
        _, sparse_ground = _sparse_depth(
            points,
            camera_to_lidar,
            intrinsics,
            (args.width, args.height),
            ground_z_range=(args.ground_z_min, args.ground_z_max),
        )
        road_mask = _road_mask_from_lidar(sparse_ground, args.dilation_px)
        boundaries = extract_road_boundaries(road_mask, row_step=4)
        item = {
            "source_key": record.get("source_key"),
            "scene_name": record.get("scene_name"),
            "anchor_image": anchor_path,
            "sparse_ground_fraction": float(np.mean(sparse_ground)),
            "road_support_fraction": float(np.mean(road_mask)),
            "boundary_confidence": float(boundaries.get("confidence", 0.0)),
            "boundary_valid": bool(boundaries.get("valid", False)),
            "boundary_descriptor": {
                "rows": boundaries.get("rows", []),
                "left_x": boundaries.get("left_x", []),
                "right_x": boundaries.get("right_x", []),
            },
        }
        if args.mask_dir is not None:
            args.mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = args.mask_dir / f"record_{index:03d}.npy"
            np.save(mask_path, road_mask.astype(np.uint8))
            item["road_mask_path"] = str(mask_path)
        results.append(item)
        if args.visualization_dir is not None and index < args.visualize_count:
            _render(args.visualization_dir / f"record_{index:03d}.png", anchor_path, road_mask, boundaries)
        print(json.dumps({"completed": len(results), "total": len(rows)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=True) for item in results) + "\n", encoding="utf-8")
    summary = {
        "protocol": "navsim-lidar-road-support-oracle-v1",
        "num_records": len(results),
        "valid_boundary_fraction": float(np.mean([item["boundary_valid"] for item in results])) if results else 0.0,
        "median_sparse_ground_fraction": float(np.median([item["sparse_ground_fraction"] for item in results])) if results else None,
        "median_road_support_fraction": float(np.median([item["road_support_fraction"] for item in results])) if results else None,
        "median_boundary_confidence": float(np.median([item["boundary_confidence"] for item in results])) if results else None,
    }
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
