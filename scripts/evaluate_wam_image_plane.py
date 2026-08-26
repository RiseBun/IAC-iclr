#!/usr/bin/env python3
"""Evaluate WAM future frames without requiring generated-camera calibration.

The decoder sees only the last history image, generated future images, and a
candidate-independent image-plane ROI. Action candidates are used only after
decoding to compute the reciprocal consistency matrix.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.flow import RaftFlowExtractor
from iac_new.image_plane_decode import compare_image_plane_trajectory, decode_image_plane_motion
from iac_new.scoring import polygon_mask


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty WAM manifest: {path}")
    return rows


def _supported_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per generated branch: the action that generated it."""
    selected = []
    for row in rows:
        candidates = list(row.get("candidate_traj") or [])
        if not candidates:
            continue
        if int(row.get("candidate_index", -1)) != int(row.get("supported_candidate_index", -2)):
            continue
        selected.append(row)
    return selected


def _action_id(row: dict[str, Any]) -> str:
    meta = row.get("protocol_metadata") or {}
    return f"{row.get('twin_id', row.get('group_id'))}:{meta.get('condition_name', row.get('group_id'))}"


def _decode_branch(
    row: dict[str, Any],
    extractor: RaftFlowExtractor,
    *,
    width: int,
    height: int,
    polygon: list[list[float]],
) -> dict[str, Any]:
    history = [str(path) for path in row.get("history_images") or []]
    future = [str(path) for path in row.get("future_images") or []]
    if not history or not future:
        raise ValueError("history_images and future_images are required")
    missing = [path for path in history + future if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    flows, fb = extractor.observe_image_plane([history[-1]] + future, (width, height))
    roi = polygon_mask(height, width, polygon)
    decoded = decode_image_plane_motion(
        flows,
        roi,
        consistency_masks=fb,
        future_times_s=np.asarray((row.get("protocol_metadata") or {}).get("candidate_times_s"), dtype=np.float64),
    )
    action = np.asarray(row["candidate_traj"], dtype=np.float64)
    comparison = compare_image_plane_trajectory(np.asarray(decoded["trajectory"]), action)
    return {
        "branch_id": str(row.get("sample_id") or row.get("group_id")),
        "twin_id": str(row.get("twin_id") or ""),
        "condition_family": str((row.get("protocol_metadata") or {}).get("condition_family") or "unknown"),
        "condition_action_id": _action_id(row),
        "generated_future_images": future,
        "action_condition": action.tolist(),
        "decoded_image_plane": decoded,
        "self_comparison": comparison,
        "calibration": {"status": "missing", "projectable": False, "projection_mode": "image_plane_only"},
        "candidate_bank_used_by_decoder": False,
    }


def _pair_score(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    branches = [first, second]
    energy_matrix = []
    for image_branch in branches:
        row = []
        for action_branch in branches:
            row.append(1.0 - float(compare_image_plane_trajectory(
                np.asarray(image_branch["decoded_image_plane"]["trajectory"]),
                np.asarray(action_branch["action_condition"]),
            )["shape_compatibility"]))
        energy_matrix.append(row)
    own = np.asarray([energy_matrix[0][0], energy_matrix[1][1]], dtype=np.float64)
    cross = np.asarray([energy_matrix[0][1], energy_matrix[1][0]], dtype=np.float64)
    # A speed-only intervention has the same normalized path shape by design;
    # it must not be counted as a failed lateral/yaw consistency decision.
    action_shapes = [np.asarray(item["action_condition"], dtype=np.float64) for item in branches]
    shape_delta = float(np.mean(np.abs(
        action_shapes[0][:, 1:] - action_shapes[1][:, 1:]
    )))
    shape_observable = shape_delta > 1e-6
    # 0.5 is an explicit tie/indeterminate decision, rather than a failure.
    decision = np.where(own < cross - 1e-6, 1.0, np.where(own > cross + 1e-6, 0.0, 0.5))
    shape_cc = float(np.mean(decision)) if shape_observable else None
    margin = float(np.mean(cross - own)) if shape_observable else None
    action_forward = np.asarray([np.asarray(item["action_condition"])[-1, 0] for item in branches])
    decoded_forward = np.asarray([item["decoded_image_plane"]["forward_proxy_total"] for item in branches])
    speed_order_valid = bool(np.ptp(action_forward) > 1e-6 and np.ptp(decoded_forward) > 1e-6)
    speed_order_correct = bool(np.argsort(action_forward).tolist() == np.argsort(decoded_forward).tolist()) if speed_order_valid else None
    return {
        "protocol": "wam-image-plane-paired-cc-v1",
        "twin_id": first["twin_id"],
        "condition_family": first["condition_family"],
        "energy_matrix": energy_matrix,
        "shape_counterfactual_consistency": shape_cc,
        "shape_cc_margin": margin,
        "shape_observable": shape_observable,
        "speed_order_observable": speed_order_valid,
        "speed_order_correct": speed_order_correct,
        "speed_absolute_observable": False,
        "signed_lateral_yaw_validated": False,
        "orientation_sign_requires_calibration": True,
        "branches": branches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--model", choices=("small", "large"), default="large")
    args = parser.parse_args()
    rows = _supported_rows(read_rows(args.manifest))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("protocol_metadata") or {}
        groups[(str(row.get("twin_id") or ""), str(meta.get("condition_family") or "unknown"))].append(row)
    pairs = []
    for key, values in sorted(groups.items()):
        branch_by_group = {str(value.get("group_id")): value for value in values}
        if len(branch_by_group) == 2:
            pairs.append((key, list(branch_by_group.values())))
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    extractor = RaftFlowExtractor(
        model_size=args.model, device=args.device, updates=32, batch_size=4,
        forward_backward=True, fb_abs_threshold_px=1.5, fb_relative_threshold=0.05,
    )
    polygon = [[0.05, 1.0], [0.95, 1.0], [0.68, 0.50], [0.32, 0.50]]
    results = []
    errors = []
    started = time.perf_counter()
    for index, (_, pair) in enumerate(pairs, start=1):
        try:
            decoded = [
                _decode_branch(row, extractor, width=args.width, height=args.height, polygon=polygon)
                for row in pair
            ]
            results.append(_pair_score(decoded[0], decoded[1]))
        except Exception as error:
            errors.append({"pair": index, "error": str(error)})
        print(json.dumps({"completed": index, "total": len(pairs)}), flush=True)
    shape_cc = [row["shape_counterfactual_consistency"] for row in results if row["shape_counterfactual_consistency"] is not None]
    margins = [row["shape_cc_margin"] for row in results if row["shape_cc_margin"] is not None]
    speed = [row["speed_order_correct"] for row in results if row["speed_order_correct"] is not None]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_family[row["condition_family"]].append(row)
    summary = {
        "protocol": "wam-image-plane-paired-cc-v1",
        "manifest": str(args.manifest.resolve()),
        "num_manifest_rows": len(rows),
        "num_pairs": len(pairs),
        "num_scored_pairs": len(results),
        "num_error": len(errors),
        "mean_shape_counterfactual_consistency": float(np.mean(shape_cc)) if shape_cc else None,
        "median_shape_counterfactual_consistency": float(np.median(shape_cc)) if shape_cc else None,
        "mean_shape_cc_margin": float(np.mean(margins)) if margins else None,
        "shape_observable_pairs": len(shape_cc),
        "speed_order_accuracy": float(np.mean(speed)) if speed else None,
        "speed_order_pairs": len(speed),
        "speed_absolute_observable": False,
        "signed_lateral_yaw_validated": False,
        "orientation_sign_requires_calibration": True,
        "camera_calibration_required_for_signed_ego_cc": True,
        "candidate_bank_used_by_decoder": False,
        "by_condition_family": {
            family: {
                "pairs": len(values),
                "mean_shape_counterfactual_consistency": float(np.mean([v["shape_counterfactual_consistency"] for v in values if v["shape_counterfactual_consistency"] is not None])) if any(v["shape_counterfactual_consistency"] is not None for v in values) else None,
                "mean_shape_cc_margin": float(np.mean([v["shape_cc_margin"] for v in values if v["shape_cc_margin"] is not None])) if any(v["shape_cc_margin"] is not None for v in values) else None,
                "shape_observable_pairs": int(sum(v["shape_counterfactual_consistency"] is not None for v in values)),
                "speed_order_accuracy": float(np.mean([v["speed_order_correct"] for v in values if v["speed_order_correct"] is not None])) if any(v["speed_order_correct"] is not None for v in values) else None,
            }
            for family, values in sorted(by_family.items())
        },
        "elapsed_s": time.perf_counter() - started,
        "errors": errors,
        "results": results,
        "interpretation": "Image-plane output is a candidate-blind diagnostic. Speed is only rank-observable; signed lateral/yaw and metric ego trajectory require camera convention/calibration or an independent ego-state reference.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
