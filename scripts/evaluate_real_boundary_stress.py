"""Stress-test road-boundary recovery on real NAVTRAIN images.

SegFormer's unmodified temporal mask is used only as a same-image pseudo-oracle;
it is never used by the production scorer.  The stress masks emulate far-range
occlusion and missing road evidence, allowing a controlled comparison between
the single-frame filter and the gated homography fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.flow import RaftFlowExtractor
from iac_new.perception import build_perception, temporal_road_consensus
from iac_new.protocol import read_jsonl, validate_record
from iac_new.road_structure import extract_road_boundaries
from iac_new.scoring import polygon_mask
from iac_new.temporal_geometry import HomographyBoundaryPropagator, RoadStateFilter


def _center_error(predicted: dict[str, Any], target: dict[str, Any], rows_subset: np.ndarray | None = None) -> float | None:
    if not predicted.get("valid") or not target.get("valid"):
        return None
    rows = np.asarray(target.get("rows", []), dtype=np.float64)
    if rows_subset is not None:
        rows = np.asarray(rows_subset, dtype=np.float64)
    if len(rows) < 2:
        return None
    target_left = np.interp(rows, target["rows"], target["left_x"])
    target_right = np.interp(rows, target["rows"], target["right_x"])
    target_center = 0.5 * (target_left + target_right)
    left_values = predicted.get("left_boundary", [])
    right_values = predicted.get("right_boundary", [])
    if not left_values and predicted.get("measurement", {}).get("boundaries", {}).get("valid"):
        measurement = predicted["measurement"]["boundaries"]
        left_values = [[x, y] for x, y in zip(measurement.get("left_x", []), measurement.get("rows", []))]
        right_values = [[x, y] for x, y in zip(measurement.get("right_x", []), measurement.get("rows", []))]
    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    if len(left) < 2 or len(right) < 2:
        return None
    pred_rows = np.asarray(left[:, 1], dtype=np.float64)
    pred_center = 0.5 * (left[:, 0] + np.interp(pred_rows, right[:, 1], right[:, 0]))
    expected = np.interp(pred_rows, rows, target_center)
    if rows_subset is not None:
        keep = np.isin(np.rint(pred_rows).astype(int), np.rint(rows_subset).astype(int))
        if not keep.any():
            return None
        pred_center = pred_center[keep]
        expected = expected[keep]
    return float(np.median(np.abs(pred_center - expected)))


def _run_variant(
    masks: np.ndarray,
    full_masks: np.ndarray,
    flows: np.ndarray,
    weights: np.ndarray,
    *,
    intrinsics: np.ndarray,
    camera_to_ego: np.ndarray,
    future_times_s: np.ndarray,
    propagation_method: str = "homography",
) -> dict[str, Any]:
    baseline = RoadStateFilter().update(masks)
    propagated = HomographyBoundaryPropagator(
        min_current_confidence_for_propagation=0.9,
        propagation_method=propagation_method,
        propagate_far_missing=(propagation_method != "flow_warp"),
    ).update(
        masks,
        intrinsics=intrinsics,
        camera_to_ego=camera_to_ego,
        observed_flows=flows,
        static_weights=weights,
        future_times_s=future_times_s,
    )
    full_boundaries = [extract_road_boundaries(mask) for mask in full_masks]
    base_valid = [item.get("valid", False) for item in baseline["states"]]
    prop_valid = [item.get("valid", False) for item in propagated["states"]]
    base_errors: list[float] = []
    prop_errors: list[float] = []
    for index, target in enumerate(full_boundaries):
        if index < len(baseline["states"]):
            error = _center_error(baseline["states"][index], target)
            if error is not None:
                base_errors.append(error)
        if index < len(propagated["states"]):
            error = _center_error(propagated["states"][index], target)
            if error is not None:
                prop_errors.append(error)
    # The caller can provide a degraded mask; report a second metric only on
    # rows where the degraded mask lost road support.
    missing_base_errors: list[float] = []
    missing_prop_errors: list[float] = []
    for index, target in enumerate(full_boundaries):
        if index >= len(masks):
            continue
        target_rows = np.asarray(target.get("rows", []), dtype=np.float64)
        if len(target_rows) < 2:
            continue
        missing_rows = [row for row in target_rows if not masks[index, int(round(row))].any()]
        if not missing_rows:
            continue
        if index < len(propagated["states"]):
            error = _center_error(propagated["states"][index], target, np.asarray(missing_rows))
            if error is not None:
                missing_prop_errors.append(error)
    return {
        "baseline_valid_states": int(sum(base_valid)),
        "propagated_valid_states": int(sum(prop_valid)),
        "baseline_center_error_px": float(np.median(base_errors)) if base_errors else None,
        "propagated_center_error_px": float(np.median(prop_errors)) if prop_errors else None,
        "propagation_applied_fraction": float(np.mean([
            bool(item.get("propagation_applied")) for item in propagated["propagation"]
        ])),
        "missing_row_count": int(sum(
            1 for index, target in enumerate(full_boundaries)
            for row in target.get("rows", [])
            if index < len(masks) and not masks[index, int(round(row))].any()
        )),
        "propagated_missing_center_error_px": float(np.median(missing_prop_errors)) if missing_prop_errors else None,
        "homography_use_fraction": float(np.mean([
            bool(item.get("homography_used")) for item in propagated["propagation"]
        ])),
        "flow_warp_use_fraction": float(np.mean([
            bool(item.get("flow_warp_used")) for item in propagated["propagation"]
        ])),
        "baseline_far_observability": float(np.mean([
            float(item.get("far_range_observability", 0.0)) for item in baseline["states"]
        ])) if baseline["states"] else None,
        "baseline_lateral_uncertainty_m": float(np.mean([
            float(item.get("lateral_uncertainty_m", 0.0)) for item in baseline["states"]
        ])) if baseline["states"] else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_rows = read_jsonl(args.manifest)[: args.max_samples]
    records = [validate_record(row, manifest_root=args.manifest.parent) for row in raw_rows]
    image_cfg = config["image"]
    width, height = int(image_cfg["width"]), int(image_cfg["height"])
    flow_cfg = config["flow"]
    extractor = RaftFlowExtractor(
        model_size=str(flow_cfg["model"]), device=args.device,
        updates=int(flow_cfg["updates"]), batch_size=int(flow_cfg["batch_size"]),
        forward_backward=True,
        fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
        fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
    )
    perception = build_perception(config, device=args.device)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        flow = extractor.observe(
            record["frame_paths"], record["intrinsics"], record["distortion"],
            (width, height), allow_mixed_source_sizes=True,
        )
        perception_observation = perception.observe(
            record["frame_paths"], target_size=(width, height),
            intrinsics=record["intrinsics"], distortion=record["distortion"],
        )
        perception_observation = temporal_road_consensus(
            perception_observation, flow.forward,
            road_dilation_px=int(config.get("perception", {}).get("road_dilation_px", 4)),
            actor_dilation_px=int(config.get("perception", {}).get("actor_dilation_px", 3)),
        )
        future_start = int(record["history_count"]) - 1
        full_masks = np.asarray(perception_observation.traversable_masks[future_start:], dtype=bool)
        observed = flow.forward[future_start:]
        consistency = flow.consistency_masks[future_start:]
        variants: dict[str, np.ndarray] = {"clean": full_masks.copy()}
        far = full_masks.copy()
        far[1:, : int(height * 0.62)] = False
        variants["far_missing"] = far
        full_missing = full_masks.copy()
        if len(full_missing) > 1:
            full_missing[1] = False
        variants["interval_missing"] = full_missing
        band = full_masks.copy()
        if len(band) > 1:
            band[1, int(height * 0.50): int(height * 0.72)] = False
        variants["partial_band"] = band
        item = {"sample_id": record["sample_id"], "variants": {}}
        for name, masks in variants.items():
            item["variants"][name] = _run_variant(
                masks, full_masks, observed, consistency.astype(np.float32),
                intrinsics=flow.intrinsics, camera_to_ego=record["camera_to_ego"],
                future_times_s=np.asarray(record["future_times_s"], dtype=np.float64),
            )
            item["variants"][f"{name}_flow_warp"] = _run_variant(
                masks, full_masks, observed, consistency.astype(np.float32),
                intrinsics=flow.intrinsics, camera_to_ego=record["camera_to_ego"],
                future_times_s=np.asarray(record["future_times_s"], dtype=np.float64),
                propagation_method="flow_warp",
            )
            item["variants"][f"{name}_global_flow_homography"] = _run_variant(
                masks, full_masks, observed, consistency.astype(np.float32),
                intrinsics=flow.intrinsics, camera_to_ego=record["camera_to_ego"],
                future_times_s=np.asarray(record["future_times_s"], dtype=np.float64),
                propagation_method="global_flow_homography",
            )
        rows.append(item)
        print(json.dumps({"completed": index + 1, "total": len(records)}), flush=True)
    summary = {
        "protocol": "real-image-road-boundary-stress-v1",
        "num_samples": len(rows),
        "pseudo_oracle": "unmodified SegFormer temporal road mask",
        "variants": {},
    }
    for name in ("clean", "far_missing", "interval_missing", "partial_band"):
        values = [row["variants"][name] for row in rows]
        summary["variants"][name] = {
            key: float(np.mean([item[key] for item in values if item[key] is not None])) if any(item[key] is not None for item in values) else None
            for key in ("baseline_valid_states", "propagated_valid_states", "baseline_center_error_px", "propagated_center_error_px", "missing_row_count", "propagated_missing_center_error_px", "propagation_applied_fraction", "homography_use_fraction")
        }
        flow_values = [row["variants"][f"{name}_flow_warp"] for row in rows]
        summary["variants"][f"{name}_flow_warp"] = {
            key: float(np.mean([item[key] for item in flow_values if item[key] is not None])) if any(item[key] is not None for item in flow_values) else None
            for key in ("baseline_valid_states", "propagated_valid_states", "baseline_center_error_px", "propagated_center_error_px", "missing_row_count", "propagated_missing_center_error_px", "propagation_applied_fraction", "homography_use_fraction", "flow_warp_use_fraction", "baseline_far_observability", "baseline_lateral_uncertainty_m")
        }
        global_values = [row["variants"][f"{name}_global_flow_homography"] for row in rows]
        summary["variants"][f"{name}_global_flow_homography"] = {
            key: float(np.mean([item[key] for item in global_values if item[key] is not None])) if any(item[key] is not None for item in global_values) else None
            for key in ("baseline_valid_states", "propagated_valid_states", "baseline_center_error_px", "propagated_center_error_px", "missing_row_count", "propagated_missing_center_error_px", "propagation_applied_fraction", "homography_use_fraction")
        }
    result = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=True) for item in rows) + "\n", encoding="utf-8")
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
