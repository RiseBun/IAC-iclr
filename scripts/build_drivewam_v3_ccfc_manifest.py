#!/usr/bin/env python3
"""Assemble paired v3 DriveWAM command branches for the Level-2 probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _action4(value: Any) -> list[list[float]]:
    """Runner emits [1,3,8]; Level-2 uses the four visual timestamps."""
    action = value[0] if isinstance(value, list) and len(value) == 1 else value
    return [[float(action[axis][idx]) for axis in range(3)] for idx in (1, 3, 5, 7)]


def _branch_rows(root: Path, index_map: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in sorted(root.glob("shard_*/manifest.json")):
        shard_id = int(manifest.parent.name.rsplit("_", 1)[-1])
        for generated in json.loads(manifest.read_text(encoding="utf-8")):
            # The runner restarts sample_index at zero in every shard.  The
            # immutable new_index_map uses one global index, so recover the
            # shard offset from the actual partition sizes.
            local_index = int(generated["sample_index"])
            offsets = (0, 187, 373, 559)
            mapped = index_map.get((shard_id, local_index))
            if mapped is None:
                mapped = index_map.get(offsets[shard_id] + local_index)
            if mapped is None:
                raise ValueError(f"runner sample is absent from new index map: shard={shard_id} index={local_index}")
            generated = dict(generated)
            generated["source_key"] = mapped["source_key"]
            rows.append(generated)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--new-index-map", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    private = {str(row["source_key"]): row for row in _jsonl(args.private_manifest)}
    index_rows = _jsonl(args.new_index_map)
    if index_rows and "shard" in index_rows[0]:
        index_map = {(int(row["shard"]), int(row["local_index"])): row for row in index_rows}
    else:
        index_map = {int(row["new_index"]): row for row in index_rows}
    by_branch = {
        "left": _branch_rows(args.left, index_map),
        "right": _branch_rows(args.right, index_map),
    }
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for branch, generated_rows in by_branch.items():
        seen: set[str] = set()
        for generated in generated_rows:
            source_key = str(generated["source_key"])
            if source_key in seen:
                raise ValueError(f"duplicate {branch} source key: {source_key}")
            seen.add(source_key)
            base = private.get(source_key)
            if base is None:
                raise ValueError(f"branch source key absent from private benchmark: {source_key}")
            action = _action4(generated["predicted_action_trajectory"])
            benchmark_id = str(base["benchmark_id"])
            history_dir = args.history_root / benchmark_id
            history = [str(history_dir / f"history_{idx:02d}.png") for idx in range(4)]
            row = {
                "sample_id": f"{source_key}::{branch}",
                "source_key": source_key,
                "counterfactual_group_id": source_key,
                "branch_role": branch,
                "history_fingerprint": source_key,
                "nuisance_seed": int(base["benchmark_id"].rsplit("-", 1)[-1]),
                "intervention_type": "navigation_command_onehot",
                # This is the intervention contract itself, not a negative
                # specificity control.  Leaving the field absent allows the
                # command-conditioned CCFC claim to be formally eligible.
                "history_frame_paths": history,
                "future_frame_paths": list(generated["future_images"]),
                "future_images": list(generated["future_images"]),
                "history_times_s": list(base.get("history_times_s", [-1.5, -1.0, -0.5, 0.0])),
                "future_times_s": [1.0, 2.0, 3.0, 4.0],
                "intrinsics": base["intrinsics"],
                "distortion": base.get("distortion", []),
                "camera_to_ego": base["camera_to_ego"],
                "history_ego_state": base.get("history_ego_state", []),
                "metadata": {
                    "benchmark_id": benchmark_id,
                    "stratum": base.get("stratum"),
                    "history_ego_state": base.get("history_ego_state", []),
                    "candidate_blind_image_branch": True,
                    "action_waypoint_used_by_image_branch": False,
                },
                "action_trajectory": action,
                "action_trajectory_source": "wam_action_head",
                "future_images_source": "wam_generated",
                "wam_model_id": "drivewam_navsim_diffusion_v1",
                "candidate_bank_used_by_decoder": False,
                "gt_candidate_id": "wam_action_head",
                "candidates": [
                    {"candidate_id": "wam_action_head", "prior": 1.0, "trajectory": action},
                    {"candidate_id": "zero_null", "prior": 0.1, "trajectory": [[0.0, 0.0, 0.0] for _ in action]},
                ],
                "command_override": branch,
                "lineage": {
                    "source_sample": generated["source_sample"],
                    "runner_branch": branch,
                    "same_history_seed": True,
                },
            }
            rows.append(row)
        counts[branch] = len(seen)
    if counts.get("left") != counts.get("right"):
        raise ValueError(f"unbalanced paired branches: {counts}")
    if any(row["counterfactual_group_id"] not in private for row in rows):
        raise ValueError("assembled row is not in private benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "pairs": counts["left"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
