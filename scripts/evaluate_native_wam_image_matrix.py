#!/usr/bin/env python3
"""Decode native WAM future frames and score a shared action matrix.

This is the image-side half of the native WAM protocol.  A branch's action
condition is used only as a reference in the final cross-score; the decoder
sees history images and that branch's generated future images.  Consequently
the diagonal score is evidence that the future image itself supports the
conditioned action, rather than evidence that a candidate was selected from a
bank.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.action_image_matrix import (
    decoded_intervention_delta_matrix,
    decoded_trajectory_cross_matrix,
)
from iac_new.flow import RaftFlowExtractor
from iac_new.dino_features import DINOv2TemporalConsistency
from iac_new.maneuver import compare_maneuvers, extract_maneuver
from iac_new.perception import build_perception
from iac_new.protocol import validate_record
from iac_new.road_relative import compare_action_to_support

try:
    from scripts.evaluate_continuous_decoder import evaluate_record as decode_record
except ModuleNotFoundError:
    # Direct ``python scripts/...`` execution puts the scripts directory on
    # sys.path, while the test suite imports it as a package.
    from evaluate_continuous_decoder import evaluate_record as decode_record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _group_id(row: dict[str, Any]) -> str:
    value = row.get("counterfactual_group_id")
    if value is None or not str(value):
        raise ValueError("every branch needs counterfactual_group_id")
    return str(value)


def _branch_id(row: dict[str, Any]) -> str:
    value = row.get("branch_id")
    if value is None or not str(value):
        raise ValueError("every branch needs branch_id")
    return str(value)


def _times(row: dict[str, Any], count: int) -> tuple[list[float], list[float]]:
    future = np.asarray(row.get("future_times_s"), dtype=np.float64)
    if future.shape != (count,) or not np.all(np.isfinite(future)) or np.any(np.diff(future) <= 0.0) or np.any(future <= 0.0):
        raise ValueError(f"{_branch_id(row)}: future_times_s must be positive and strictly increasing")
    history = np.asarray(row.get("history_times_s", []), dtype=np.float64)
    if history.size == 0:
        # Native branch files often store future times relative to the anchor.
        # Use a synthetic monotonic history clock; only future relative times
        # enter the trajectory decoder.
        step = float(np.median(np.diff(future))) if len(future) > 1 else float(future[0])
        history = np.arange(-len(row["history_images"]) + 1, 1, dtype=np.float64) * step
    if history.shape != (len(row["history_images"],),) or not np.all(np.isfinite(history)) or np.any(np.diff(history) <= 0.0):
        raise ValueError(f"{_branch_id(row)}: history_times_s does not match history_images")
    # validate_record expects absolute future timestamps after the last history
    # timestamp.  Shift the relative future clock if native history timestamps
    # already end at zero or another arbitrary anchor.
    if future[0] <= history[-1]:
        future = future - future[0] + history[-1] + max(float(np.median(np.diff(future))), 1e-3)
    return history.tolist(), future.tolist()


def build_decoder_record(row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    history = list(row.get("history_images") or [])
    future = list(row.get("future_images") or [])
    if not history or not future:
        raise ValueError(f"{_branch_id(row)}: generated future_images are required")
    if len(history) < 2 or not future:
        raise ValueError(f"{_branch_id(row)}: native image protocol requires at least 2 history and 1 future frame")
    intrinsics = row.get("intrinsics", row.get("camera_intrinsic"))
    distortion = row.get("distortion", row.get("camera_distortion", []))
    if intrinsics is None or row.get("camera_to_ego") is None:
        raise ValueError(f"{_branch_id(row)}: intrinsics and camera_to_ego are required for image-plane decoding")
    history_times, future_times = _times(row, len(future))
    return {
        "sample_id": _branch_id(row),
        "scene_id": str(row.get("scene_name") or row.get("source_key") or _branch_id(row)),
        "history_frame_paths": history,
        "future_frame_paths": future,
        "history_times_s": history_times,
        "future_times_s": future_times,
        "intrinsics": intrinsics,
        "distortion": distortion,
        "camera_to_ego": row["camera_to_ego"],
        "candidates": candidates,
        "gt_candidate_id": _branch_id(row),
        "metadata": {
            "protocol": "wam-native-counterfactual-branch-v1",
            "source_key": row.get("source_key"),
            "counterfactual_group_id": _group_id(row),
            "branch_mode": row.get("branch_mode"),
            "future_images_source": row.get("future_images_source"),
            "realized_state_available": bool(row.get("realized_future_ego_state") is not None),
            "history_ego_state": row.get("history_ego_state"),
            "native_action_condition": row.get("native_action_condition", row.get("action_condition")),
            "action_trajectory": row.get("action_trajectory"),
            "realized_future_ego_state": row.get("realized_future_ego_state"),
            "task_success": row.get("task_success"),
            "road_oracle_mask_path": row.get("road_oracle_mask_path"),
        },
    }


def _candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        trajectory = row.get("action_trajectory", row.get("trajectory"))
        if trajectory is None:
            raise ValueError(f"{_branch_id(row)}: action_trajectory is required")
        result.append({"candidate_id": _branch_id(row), "trajectory": trajectory, "prior": 1.0})
    if len(result) < 2:
        raise ValueError("each counterfactual group needs at least two branches")
    return result


def _baseline_index(rows: list[dict[str, Any]]) -> int:
    """Use the native logged branch as the nuisance-scale baseline."""
    for index, row in enumerate(rows):
        if str(row.get("branch_mode") or "") == "logged":
            return index
    for index, row in enumerate(rows):
        if bool(row.get("realized_state_available")):
            return index
    raise ValueError("counterfactual group needs a logged or realized baseline branch")


def evaluate_groups(rows: list[dict[str, Any]], extractor: Any, config: dict[str, Any], perception: Any | None, manifest_root: Path, max_groups: int | None = None, dino: Any | None = None, road_oracle_masks: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_id(row)].append(row)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for group_index, (group_id, branches) in enumerate(sorted(grouped.items())):
        if max_groups is not None and group_index >= max_groups:
            break
        try:
            candidates = _candidates(branches)
            decoded_branches = []
            for row in branches:
                if road_oracle_masks and row.get("source_key") in road_oracle_masks:
                    row = {**row, "road_oracle_mask_path": road_oracle_masks[row["source_key"]]}
                raw_record = build_decoder_record(row, candidates)
                record = validate_record(raw_record, manifest_root=manifest_root)
                decoded = decode_record(
                    record, extractor, {**config, "allow_mixed_source_sizes": True}, perception, dino
                )
                decoded_branches.append({
                    "branch_id": _branch_id(row),
                    "condition_action_id": _branch_id(row),
                    "imagined_future": decoded["decoder"]["trajectory"],
                    "imagined_support": decoded["decoder"].get("profile_support"),
                    "executed_action": row.get("action_trajectory", row.get("trajectory")),
                    "valid": bool(decoded.get("valid", True)),
                    "abstain_reasons": list(decoded.get("abstain_reasons", [])),
                    "source_key": row.get("source_key"),
                    "road_relative_posterior": decoded.get("road_relative_posterior"),
                    "road_structure": decoded.get("road_structure"),
                    "history_speed_prior_mps": decoded.get("history_speed_prior_mps"),
                })
            times = np.asarray(branches[0]["future_times_s"], dtype=np.float64)
            for decoded_branch, source_row in zip(decoded_branches, branches):
                action_skeleton = extract_maneuver(
                    np.asarray(decoded_branch["executed_action"], dtype=np.float64), times
                )
                image_skeleton = extract_maneuver(
                    np.asarray(decoded_branch["imagined_future"], dtype=np.float64), times
                )
                decoded_branch["action_maneuver"] = action_skeleton
                decoded_branch["imagined_maneuver"] = image_skeleton
                decoded_branch["maneuver_consistency"] = compare_maneuvers(
                    action_skeleton, image_skeleton
                )
                if decoded_branch.get("road_relative_posterior") is not None:
                    decoded_branch["action_support"] = compare_action_to_support(
                        np.asarray(decoded_branch["executed_action"], dtype=np.float64),
                        decoded_branch["road_relative_posterior"],
                        times,
                    )
            action_cfg = config.get("action_image_score", {})
            scales = action_cfg.get("scales")
            matrix = decoded_trajectory_cross_matrix(decoded_branches, times, scales=scales)
            relative_matrix = decoded_intervention_delta_matrix(
                decoded_branches,
                times,
                baseline_index=_baseline_index(branches),
                scales=scales,
            )
            results.append({
                "counterfactual_group_id": group_id,
                "source_key": branches[0].get("source_key"),
                "branches": decoded_branches,
                "action_image_matrix": matrix,
                "intervention_delta_matrix": relative_matrix,
                "realized_state_cc_available": all(row.get("realized_future_ego_state") is not None for row in branches),
                "foresight_conditioned_success": {"status": "not_computed", "ready": all(row.get("realized_future_ego_state") is not None and row.get("task_success") is not None for row in branches), "reason": "image-only decoder output does not claim FCS; compute it only after independently realized state and task-success fields are attached for every branch"},
                "maneuver_consistency": {
                    "mean_score": float(np.mean([
                        branch["maneuver_consistency"]["score"] for branch in decoded_branches
                    ])),
                    "sequence_accuracy": float(np.mean([
                        branch["maneuver_consistency"]["sequence_match"] for branch in decoded_branches
                    ])),
                    "event_score": float(np.mean([
                        branch["maneuver_consistency"]["event_score"] for branch in decoded_branches
                    ])),
                },
                "road_relative_support": {
                    "mean_joint_support_coverage": float(np.mean([
                        branch.get("action_support", {}).get("joint_support_coverage", float("nan"))
                        for branch in decoded_branches
                    ])),
                    "mean_heading_support_coverage": float(np.mean([
                        branch.get("action_support", {}).get("heading_support_coverage", float("nan"))
                        for branch in decoded_branches
                    ])),
                },
            })
        except Exception as error:
            errors.append({"counterfactual_group_id": group_id, "error": str(error)})
    return results, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--road-oracle-manifest", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_jsonl(args.manifest)
    road_oracle_masks = None
    if args.road_oracle_manifest is not None:
        road_oracle_masks = {
            str(item.get("source_key")): str(item["road_mask_path"])
            for item in read_jsonl(args.road_oracle_manifest)
            if item.get("source_key") and item.get("road_mask_path")
        }
    flow_cfg = config["flow"]
    extractor = RaftFlowExtractor(
        model_size=str(flow_cfg["model"]), device=args.device,
        updates=int(flow_cfg["updates"]), batch_size=int(flow_cfg["batch_size"]),
        forward_backward=bool(flow_cfg["forward_backward"]),
        fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
        fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
    )
    perception = build_perception(config, device=args.device)
    dino = None
    dino_cfg = config.get("dino", {})
    if bool(dino_cfg.get("enabled", False)):
        dino = DINOv2TemporalConsistency(
            device=args.device,
            model_name=str(dino_cfg.get("model_name", "dinov2_vits14")),
            hub_dir=dino_cfg.get("hub_dir"),
            weight_floor=float(dino_cfg.get("weight_floor", 0.25)),
        )
    started = time.perf_counter()
    results, errors = evaluate_groups(rows, extractor, config, perception, args.manifest.parent, args.max_groups, dino, road_oracle_masks)
    evaluable = [row["action_image_matrix"] for row in results if row["action_image_matrix"]["num_evaluable"]]
    relative_evaluable = [row["intervention_delta_matrix"] for row in results if row["intervention_delta_matrix"]["num_evaluable"]]
    maneuver_rows = [row["maneuver_consistency"] for row in results]
    summary = {
        "protocol": "wam-native-image-decoded-action-matrix-v1",
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()),
        "num_input_branches": len(rows),
        "num_groups": len(results),
        "num_errors": len(errors),
        "num_evaluable": int(sum(row["num_evaluable"] for row in evaluable)),
        "num_abstain": int(sum(row["num_abstain"] for row in evaluable)),
        "coverage": (float(sum(row["num_evaluable"] for row in evaluable) / max(sum(row["num_branches"] for row in evaluable), 1)) if evaluable else 0.0),
        "diagonal_top1_accuracy": float(np.mean([row["diagonal_top1_accuracy"] for row in evaluable])) if evaluable else None,
        "mean_cc_margin": float(np.mean([row["mean_cc_margin"] for row in evaluable])) if evaluable else None,
        "intervention_delta_diagonal_top1_accuracy": float(np.mean([row["diagonal_top1_accuracy"] for row in relative_evaluable])) if relative_evaluable else None,
        "intervention_delta_mean_cc_margin": float(np.mean([row["mean_cc_margin"] for row in relative_evaluable])) if relative_evaluable else None,
        "intervention_delta_mean_reciprocal_rank": float(np.mean([row["mean_reciprocal_rank"] for row in relative_evaluable])) if relative_evaluable else None,
        "maneuver_mean_score": float(np.mean([row["mean_score"] for row in maneuver_rows])) if maneuver_rows else None,
        "maneuver_sequence_accuracy": float(np.mean([row["sequence_accuracy"] for row in maneuver_rows])) if maneuver_rows else None,
        "maneuver_event_score": float(np.mean([row["event_score"] for row in maneuver_rows])) if maneuver_rows else None,
        "realized_state_cc": "available only when every branch has independent realized_future_ego_state",
        "foresight_conditioned_success": "not computed by image-only output; requires realized future state and task_success for every branch",
        "elapsed_s": time.perf_counter() - started,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "groups": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
