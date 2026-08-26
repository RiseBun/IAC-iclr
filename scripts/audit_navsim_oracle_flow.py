#!/usr/bin/env python3
"""Audit RAFT against NAVSIM pose + lidar rigid-flow oracle."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from iac_new.flow import RaftFlowExtractor
from iac_new.flow_reliability import FlowReliabilityFusion
from iac_new.sea_raft_flow import SeaRaftFlowExtractor
from iac_new.geometry import rigid_flow_from_depth


_PCD_TYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("U", 1): "u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("I", 1): "i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
}


def read_binary_pcd_xyz(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    marker = b"DATA binary\n"
    marker_index = raw.find(marker)
    if marker_index < 0:
        raise ValueError(f"only binary PCD is supported: {path}")
    header = raw[:marker_index].decode("ascii").splitlines()
    values = {
        line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1]
        for line in header
        if " " in line
    }
    fields = values["FIELDS"].split()
    sizes = [int(value) for value in values["SIZE"].split()]
    types = values["TYPE"].split()
    counts = [
        int(value)
        for value in values.get("COUNT", " ".join("1" for _ in fields)).split()
    ]
    if any(count != 1 for count in counts):
        raise ValueError(f"multi-count PCD fields are unsupported: {path}")
    offsets = np.cumsum([0, *sizes[:-1]]).tolist()
    dtype = np.dtype(
        {
            "names": fields,
            "formats": [_PCD_TYPES[(kind, size)] for kind, size in zip(types, sizes)],
            "offsets": offsets,
            "itemsize": sum(sizes),
        }
    )
    payload = raw[marker_index + len(marker) :]
    points = np.frombuffer(payload, dtype=dtype, count=int(values["POINTS"]))
    return np.stack(
        [points[axis].astype(np.float64) for axis in ("x", "y", "z")], axis=1
    )


def _transform(rotation: Any, translation: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _camera_to_ego(frame: dict[str, Any], camera: str) -> np.ndarray:
    camera_row = frame["cams"][camera]
    sensor_to_lidar = _transform(
        camera_row["sensor2lidar_rotation"], camera_row["sensor2lidar_translation"]
    )
    lidar_to_ego = np.asarray(frame.get("lidar2ego", np.eye(4)), dtype=np.float64)
    return lidar_to_ego @ sensor_to_lidar


def _ego_to_global(frame: dict[str, Any]) -> np.ndarray:
    if "ego2global" in frame:
        pose = np.asarray(frame["ego2global"], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError("ego2global must have shape [4,4]")
        return pose
    translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    quaternion = np.asarray(frame["ego2global_rotation"], dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("NAVSIM ego pose has unexpected shape")
    qw, qx, qy, qz = quaternion
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    pose[:3, 3] = translation
    return pose


def _sparse_depth(
    points_lidar: np.ndarray,
    camera_to_lidar: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    *,
    ground_z_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    points = np.asarray(points_lidar, dtype=np.float64)
    lidar_to_camera = np.linalg.inv(camera_to_lidar)
    points_camera = (
        lidar_to_camera[:3, :3] @ points.T + lidar_to_camera[:3, 3:4]
    ).T
    depth = points_camera[:, 2]
    projected = (np.asarray(intrinsics, dtype=np.float64) @ points_camera.T).T
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (depth > 1.0) & (depth < 100.0)
    valid &= (u >= 0.08 * width) & (u < 0.92 * width)
    valid &= (v >= 0.42 * height) & (v < height)
    indices = np.flatnonzero(valid)
    u_pixel = np.rint(u[indices]).astype(np.int64).clip(0, width - 1)
    v_pixel = np.rint(v[indices]).astype(np.int64).clip(0, height - 1)
    z = depth[indices]
    flat = v_pixel * width + u_pixel
    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = flat_sorted[1:] != flat_sorted[:-1]
    selected = order[keep]
    selected_indices = indices[selected]
    depth_map = np.full((height, width), np.nan, dtype=np.float32)
    ground_map = np.zeros((height, width), dtype=bool)
    selected_u, selected_v = u_pixel[selected], v_pixel[selected]
    depth_map[selected_v, selected_u] = z[selected].astype(np.float32)
    ground = (
        (points[selected_indices, 2] >= float(ground_z_range[0]))
        & (points[selected_indices, 2] <= float(ground_z_range[1]))
        & (points[selected_indices, 0] > 1.0)
    )
    ground_map[selected_v[ground], selected_u[ground]] = True
    return depth_map, ground_map


def flow_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(observed).all(axis=-1)
        & np.isfinite(predicted).all(axis=-1)
    )
    count = int(valid.sum())
    if count < 10:
        return {"num_points": count, "median_epe_px": None}
    obs = observed[valid].astype(np.float64)
    pred = predicted[valid].astype(np.float64)
    error = obs - pred
    epe = np.linalg.norm(error, axis=1)
    obs_magnitude = np.linalg.norm(obs, axis=1)
    pred_magnitude = np.linalg.norm(pred, axis=1)
    informative = (obs_magnitude > 0.25) & (pred_magnitude > 0.25)
    cosine = np.full(count, np.nan, dtype=np.float64)
    cosine[informative] = np.sum(obs[informative] * pred[informative], axis=1) / (
        obs_magnitude[informative] * pred_magnitude[informative]
    )
    scale_denominator = np.sum(pred * pred, axis=1)
    scale = np.sum(obs * pred, axis=1) / np.maximum(scale_denominator, 1e-6)
    return {
        "num_points": count,
        "median_epe_px": float(np.median(epe)),
        "p75_epe_px": float(np.quantile(epe, 0.75)),
        "mean_epe_px": float(np.mean(epe)),
        "median_observed_flow_px": float(np.median(obs_magnitude)),
        "median_predicted_flow_px": float(np.median(pred_magnitude)),
        "median_signed_error_x_px": float(np.median(error[:, 0])),
        "median_signed_error_y_px": float(np.median(error[:, 1])),
        "median_direction_cosine": (
            float(np.nanmedian(cosine)) if np.isfinite(cosine).any() else None
        ),
        "median_observed_to_predicted_scale": float(
            np.median(scale[np.isfinite(scale)])
        ),
        "low_observed_flow_fraction": float(np.mean(obs_magnitude < 1.0)),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _frame_lookup(source_pkl: Path, camera: str) -> dict[str, dict[str, Any]]:
    with source_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    frames = payload if isinstance(payload, list) else payload.get("frames", [])
    return {
        Path(frame["cams"][camera]["data_path"]).name: frame
        for frame in frames
        if camera in frame.get("cams", {})
    }


def _render(
    output: Path,
    image_path: str,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return
    if distortion.size:
        image = cv2.undistort(image, intrinsics, distortion, None, intrinsics)
    image = cv2.resize(image, (observed.shape[1], observed.shape[0]))
    yy, xx = np.nonzero(mask)
    if len(xx) > 500:
        choice = np.linspace(0, len(xx) - 1, 500).astype(int)
        yy, xx = yy[choice], xx[choice]
    for y, x in zip(yy, xx):
        origin = (int(x), int(y))
        pixel = np.asarray([x, y], dtype=np.float64)
        obs_end = tuple(np.round(pixel + observed[y, x]).astype(int))
        pred_end = tuple(np.round(pixel + predicted[y, x]).astype(int))
        cv2.arrowedLine(image, origin, obs_end, (70, 210, 70), 1, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(image, origin, pred_end, (40, 40, 230), 1, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(image, "green=RAFT observed  red=pose+lidar oracle", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "green=RAFT observed  red=pose+lidar oracle", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 250, 250), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raft-model", choices=("small", "large"), default="large")
    parser.add_argument("--flow-backend", choices=("raft", "sea_raft"), default="raft")
    parser.add_argument("--sea-checkpoint")
    parser.add_argument("--fb-abs-threshold-px", type=float, default=1.5)
    parser.add_argument("--fb-relative-threshold", type=float, default=0.05)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--visualize-count", type=int, default=3)
    parser.add_argument("--ground-z-min", type=float, default=-0.8)
    parser.add_argument("--ground-z-max", type=float, default=0.2)
    parser.add_argument("--reliability-repair", action="store_true")
    parser.add_argument("--repair-threshold-px", type=float, default=2.0)
    parser.add_argument("--refinement-uncertainty", action="store_true")
    parser.add_argument("--blend-grid", default="", help="comma-separated alpha values for oracle rigid-flow blend audit")
    args = parser.parse_args()

    if args.flow_backend == "sea_raft" and not args.sea_checkpoint:
        parser.error("--sea-checkpoint is required with --flow-backend sea_raft")

    rows = _read_rows(args.records)
    if args.max_records:
        rows = rows[: args.max_records]
    if args.flow_backend == "sea_raft":
        extractor = SeaRaftFlowExtractor(
            checkpoint=args.sea_checkpoint, device=args.device, iters=12
        )
    else:
        extractor = RaftFlowExtractor(
            model_size=args.raft_model,
            device=args.device,
            updates=32,
            batch_size=4,
            forward_backward=True,
            fb_abs_threshold_px=float(args.fb_abs_threshold_px),
            fb_relative_threshold=float(args.fb_relative_threshold),
        )
    lookup_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    output_rows = []
    blend_grid = [float(value) for value in args.blend_grid.split(",") if value.strip()] if args.blend_grid else []
    blend_metrics: dict[float, list[dict[str, float | int | None]]] = {alpha: [] for alpha in blend_grid}
    for record_index, record in enumerate(rows):
        source_pkl = Path(record["source_pkl"])
        if source_pkl not in lookup_cache:
            lookup_cache[source_pkl] = _frame_lookup(source_pkl, record["camera"])
        lookup = lookup_cache[source_pkl]
        image_paths = list(record["history_images"]) + list(record["future_images"])
        frames = [lookup[Path(path).name] for path in image_paths]
        intrinsics = np.asarray(record["camera_intrinsic"], dtype=np.float64)
        distortion = np.asarray(record.get("camera_distortion", []), dtype=np.float64)
        observation = extractor.observe(
            image_paths, intrinsics, distortion, (args.width, args.height),
            return_uncertainty=bool(args.refinement_uncertainty) if args.flow_backend == "raft" else False,
        )
        repaired = None
        if args.reliability_repair:
            repaired = FlowReliabilityFusion(
                repair_enabled=True,
                repair_threshold_px=float(args.repair_threshold_px),
            ).estimate(
                observed_flows=observation.forward,
                frame_paths=image_paths,
                intrinsics=intrinsics,
                distortion=distortion,
                consistency_masks=observation.consistency_masks,
            )["repaired_flows"]
        intervals = []
        for interval_index, (current, following, observed) in enumerate(
            zip(frames[:-1], frames[1:], observation.forward)
        ):
            camera_row = current["cams"][record["camera"]]
            camera_to_lidar = _transform(
                camera_row["sensor2lidar_rotation"],
                camera_row["sensor2lidar_translation"],
            )
            points = read_binary_pcd_xyz(args.sensor_root / current["lidar_path"])
            depth, ground_mask = _sparse_depth(
                points,
                camera_to_lidar,
                observation.intrinsics,
                (args.width, args.height),
                ground_z_range=(args.ground_z_min, args.ground_z_max),
            )
            current_camera_to_global = (
                _ego_to_global(current)
                @ _camera_to_ego(current, record["camera"])
            )
            following_camera_to_global = (
                _ego_to_global(following)
                @ _camera_to_ego(following, record["camera"])
            )
            following_camera_from_current = (
                np.linalg.inv(following_camera_to_global) @ current_camera_to_global
            )
            predicted, geometry_valid = rigid_flow_from_depth(
                depth, observation.intrinsics, following_camera_from_current
            )
            lidar_mask = np.isfinite(depth) & geometry_valid
            consistency = observation.consistency_masks[interval_index]
            interval = {
                "interval_index": interval_index,
                "role": "history" if interval_index < 3 else "future",
                "start_timestamp_us": int(current["timestamp"]),
                "end_timestamp_us": int(following["timestamp"]),
                "dt_s": (int(following["timestamp"]) - int(current["timestamp"])) * 1e-6,
                "all_lidar": flow_metrics(observed, predicted, lidar_mask),
                "all_lidar_fb": flow_metrics(observed, predicted, lidar_mask & consistency),
                "ground_lidar": flow_metrics(observed, predicted, lidar_mask & ground_mask),
                "ground_lidar_fb": flow_metrics(
                    observed, predicted, lidar_mask & ground_mask & consistency
                ),
                "ground_lidar_fb_repaired": (
                    flow_metrics(repaired[interval_index], predicted, lidar_mask & ground_mask & consistency)
                    if repaired is not None else None
                ),
                "forward_backward_fraction": float(
                    consistency[lidar_mask].mean()
                ) if lidar_mask.any() else 0.0,
            }
            if blend_grid:
                blend_mask = lidar_mask & ground_mask & consistency
                for alpha in blend_grid:
                    blended = (1.0 - alpha) * observed + alpha * predicted
                    blend_metrics[alpha].append(flow_metrics(blended, predicted, blend_mask))
            if observation.refinement_uncertainty is not None:
                uncertainty = observation.refinement_uncertainty[interval_index]
                valid_uncertainty = np.isfinite(uncertainty) & lidar_mask & ground_mask & consistency
                if valid_uncertainty.any():
                    error = np.linalg.norm(observed - predicted, axis=-1)
                    u = uncertainty[valid_uncertainty].astype(np.float64)
                    e = error[valid_uncertainty].astype(np.float64)
                    interval["refinement_uncertainty"] = {
                        "median_px": float(np.median(u)),
                        "p95_px": float(np.quantile(u, 0.95)),
                        "error_median_px": float(np.median(e)),
                        "error_p90_px": float(np.quantile(e, 0.90)),
                        "high_uncertainty_error_median_px": float(np.median(e[u >= np.quantile(u, 0.75)])) if len(e) >= 8 else None,
                        "low_uncertainty_error_median_px": float(np.median(e[u <= np.quantile(u, 0.25)])) if len(e) >= 8 else None,
                    }
            intervals.append(interval)
            if (
                args.visualization_dir is not None
                and record_index < args.visualize_count
                and interval_index == 3
            ):
                _render(
                    args.visualization_dir / f"record_{record_index:03d}.png",
                    image_paths[interval_index],
                    intrinsics,
                    distortion,
                    observed,
                    predicted,
                    lidar_mask & ground_mask & consistency,
                )
        output_rows.append(
            {
                "source_key": record["source_key"],
                "scene_name": record["scene_name"],
                "source_pkl": str(source_pkl),
                "distortion": distortion.tolist(),
                "intervals": intervals,
            }
        )
        print(json.dumps({"completed": len(output_rows), "total": len(rows)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    future_ground = [
        interval["ground_lidar_fb"]
        for row in output_rows
        for interval in row["intervals"]
        if interval["role"] == "future"
        and interval["ground_lidar_fb"].get("median_epe_px") is not None
    ]
    future_ground_repaired = [
        interval["ground_lidar_fb_repaired"]
        for row in output_rows
        for interval in row["intervals"]
        if interval["role"] == "future"
        and interval.get("ground_lidar_fb_repaired")
        and interval["ground_lidar_fb_repaired"].get("median_epe_px") is not None
    ]
    direction_values = [
        row["median_direction_cosine"]
        for row in future_ground
        if row.get("median_direction_cosine") is not None
    ]
    summary = {
        "protocol": "navsim-pose-lidar-raft-oracle-v1",
        "num_records": len(output_rows),
        "num_scenes": len({row["scene_name"] for row in output_rows}),
        "num_future_ground_intervals": len(future_ground),
        "median_future_ground_epe_px": float(
            np.median([row["median_epe_px"] for row in future_ground])
        ) if future_ground else None,
        "p75_future_ground_epe_px": float(
            np.median([row["p75_epe_px"] for row in future_ground])
        ) if future_ground else None,
        "median_future_direction_cosine": (
            float(np.median(direction_values)) if direction_values else None
        ),
        "median_future_observed_to_predicted_scale": float(
            np.median([row["median_observed_to_predicted_scale"] for row in future_ground])
        ) if future_ground else None,
        "median_future_ground_points": float(
            np.median([row["num_points"] for row in future_ground])
        ) if future_ground else None,
        "median_future_low_flow_fraction": float(
            np.median([row["low_observed_flow_fraction"] for row in future_ground])
        ) if future_ground else None,
        "refinement_uncertainty_enabled": bool(args.refinement_uncertainty),
        "reliability_repair_enabled": bool(args.reliability_repair),
        "median_future_ground_epe_px_repaired": float(
            np.median([row["median_epe_px"] for row in future_ground_repaired])
        ) if future_ground_repaired else None,
        "p75_future_ground_epe_px_repaired": float(
            np.median([row["p75_epe_px"] for row in future_ground_repaired])
        ) if future_ground_repaired else None,
        "median_future_direction_cosine_repaired": (
            float(np.median([
                row["median_direction_cosine"] for row in future_ground_repaired
                if row.get("median_direction_cosine") is not None
            ])) if any(row.get("median_direction_cosine") is not None for row in future_ground_repaired) else None
        ),
        "oracle_blend_grid": {
            str(alpha): {
                "median_epe_px": float(np.median([
                    item["median_epe_px"] for item in values if item.get("median_epe_px") is not None
                ])) if any(item.get("median_epe_px") is not None for item in values) else None,
                "p75_epe_px": float(np.median([
                    item["p75_epe_px"] for item in values if item.get("p75_epe_px") is not None
                ])) if any(item.get("p75_epe_px") is not None for item in values) else None,
            }
            for alpha, values in blend_metrics.items()
        },
    }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
