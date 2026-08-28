#!/usr/bin/env python3
"""Evaluate image-decoded trajectories against independent native ego state."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.evaluate_continuous_decoder import evaluate_record
except ModuleNotFoundError:
    from evaluate_continuous_decoder import evaluate_record  # type: ignore[no-redef]
from iac_new.flow import RaftFlowExtractor
from iac_new.perception import build_perception
from iac_new.protocol import read_jsonl, validate_record, write_jsonl
from iac_new.state_protocol import states_to_trajectory
from iac_new.wam_metrics import ego_state_action_compatibility, normalized_ego_state_distance


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _native_to_decoder_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    history = list(row.get("history_images") or [])
    future = list(row.get("future_images") or [])
    if len(history) != 4 or len(future) < 1:
        raise ValueError(f"native row {index}: expected four history frames and future frames")
    times = np.asarray(row.get("future_times_s"), dtype=np.float64)
    realized = np.asarray(row.get("realized_future_ego_state"), dtype=np.float64)
    trajectory = np.asarray(row.get("trajectory"), dtype=np.float64)
    if times.shape != (len(future),) or realized.shape != (len(future), 5) or trajectory.shape != (len(future), 3):
        raise ValueError(f"native row {index}: future trajectory/state/time shapes do not match")
    history_times = np.arange(4, dtype=np.float64) * 0.5
    future_absolute = history_times[-1] + times
    return {
        "sample_id": str(row.get("source_key") or row.get("sample_id") or f"native_{index}"),
        "scene_id": str(row.get("scene_name") or row.get("scene_token") or f"native_{index}"),
        "history_frame_paths": history,
        "future_frame_paths": future,
        "history_times_s": history_times.tolist(),
        "future_times_s": future_absolute.tolist(),
        "intrinsics": row.get("camera_intrinsic") or row.get("intrinsics"),
        "distortion": row.get("camera_distortion") or row.get("distortion") or [],
        "camera_to_ego": row.get("camera_to_ego"),
        "history_ego_state": row.get("history_ego_state"),
        "history_times_s": row.get("history_times_s", [-1.5, -1.0, -0.5, 0.0]),
        "metric_depth_path": row.get("metric_depth_path"),
        # The generic image protocol requires a bank, but this native
        # validation never scores or selects from it. The duplicate is only a
        # schema placeholder and is explicitly excluded from decoder inputs.
        "candidates": [
            {"candidate_id": "native_realized", "trajectory": trajectory.tolist(), "prior": 1.0},
            {"candidate_id": "native_realized_schema_placeholder", "trajectory": trajectory.tolist(), "prior": 1.0},
        ],
        "gt_candidate_id": "native_realized",
        "native_realized_future_ego_state": realized.tolist(),
        "task_success": row.get("task_success"),
        "task_success_source": row.get("task_success_source"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--actor-weight", type=float, help="override perception.actor_weight")
    parser.add_argument("--disable-road-mask", action="store_true", help="disable semantic road weighting")
    args = parser.parse_args()
    config = _json(args.config)
    if args.actor_weight is not None:
        config.setdefault("perception", {})["actor_weight"] = float(args.actor_weight)
    if args.disable_road_mask:
        config.setdefault("perception", {})["use_traversable_mask"] = False
    raw = read_jsonl(args.manifest)
    if args.max_samples is not None:
        raw = raw[: args.max_samples]
    rows = [_native_to_decoder_row(row, index) for index, row in enumerate(raw)]
    records = [validate_record(row, manifest_root=args.manifest.parent) for row in rows]
    flow_cfg = config["flow"]
    extractor = RaftFlowExtractor(
        model_size=str(flow_cfg["model"]), device=args.device,
        updates=int(flow_cfg["updates"]), batch_size=int(flow_cfg["batch_size"]),
        forward_backward=bool(flow_cfg["forward_backward"]),
        fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
        fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
    )
    perception = build_perception(config, device=args.device)
    results = []
    errors = []
    started = time.perf_counter()
    for index, (record, native) in enumerate(zip(records, raw), start=1):
        try:
            result = evaluate_record(record, extractor, config, perception)
            decoded = np.asarray(result["decoder"]["trajectory"], dtype=np.float64)
            realized = np.asarray(native["realized_future_ego_state"], dtype=np.float64)
            realized_trajectory = states_to_trajectory(realized)
            times = np.asarray(record["future_times_s"], dtype=np.float64)
            result["realized_state_reference"] = {
                "source": native.get("state_reference_source") or "native_dataset_logged_ego_state",
                "compatibility": ego_state_action_compatibility(decoded, realized_trajectory, times),
                "normalized_distance": normalized_ego_state_distance(decoded, realized_trajectory, times),
                "task_success": native.get("task_success"),
                "task_success_source": native.get("task_success_source"),
                "independent_from_decoder": True,
            }
            results.append(result)
        except Exception as error:
            errors.append({"sample_id": record["sample_id"], "error": str(error)})
        print(json.dumps({"completed": index, "total": len(records)}), flush=True)
    compat = [row["realized_state_reference"]["compatibility"] for row in results]
    distances = [row["realized_state_reference"]["normalized_distance"] for row in results]
    task_labels = [row["realized_state_reference"]["task_success"] for row in results]
    report = {
        "protocol": "native-realized-state-image-decode-v1",
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()),
        "num_input": len(records),
        "num_scored": len(results),
        "num_error": len(errors),
        "mean_realized_state_compatibility": float(np.mean(compat)) if compat else None,
        "median_realized_state_compatibility": float(np.median(compat)) if compat else None,
        "mean_realized_state_normalized_distance": float(np.mean(distances)) if distances else None,
        "task_success_labels": sum(value is not None for value in task_labels),
        "foresight_conditioned_success": {
            "status": "unavailable",
            "reason": "native manifest has no explicit task_success labels" if not any(value is not None for value in task_labels) else "paired branches are required",
        },
        "independent_realized_state_used": True,
        "candidate_bank_used_by_decoder": False,
        "elapsed_s": time.perf_counter() - started,
        "errors": errors,
        "results": results,
    }
    write_jsonl(args.output, results)
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
