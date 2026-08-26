#!/usr/bin/env python3
"""Attach explicitly aligned dataset ego states and task labels to a WAM manifest.

The annotation file is intentionally a separate artifact. Each row must carry
at least one explicit join key (``video_id``, ``sample_id``, ``source_key``, or
``scene_name`` + ``timestamp_us``) and canonical states with shape [T,5].
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _keys(row: dict[str, Any]) -> list[str]:
    keys = []
    for field in ("video_id", "sample_id", "source_key", "pair_id"):
        if row.get(field) is not None:
            keys.append(f"{field}:{row[field]}")
    if row.get("scene_name") is not None and row.get("timestamp_us") is not None:
        keys.append(f"scene_timestamp:{row['scene_name']}:{int(row['timestamp_us'])}")
    if row.get("scene_name") is not None:
        keys.append(f"scene_name:{row['scene_name']}")
    return keys


def _check_future_times(row: dict[str, Any], annotation: dict[str, Any], row_index: int) -> None:
    """Reject state joins whose logged future samples are on a different clock."""
    manifest_times = row.get("frame_times_s") or row.get("candidate_times_s")
    state_times = annotation.get("state_times_s")
    if manifest_times is None or state_times is None:
        return
    if len(manifest_times) != len(state_times):
        raise ValueError(
            f"annotation for row {row_index} has {len(state_times)} future timestamps, "
            f"but manifest expects {len(manifest_times)}"
        )
    errors = [abs(float(a) - float(b)) for a, b in zip(manifest_times, state_times)]
    if any(not math.isfinite(error) or error > 0.06 for error in errors):
        raise ValueError(
            f"annotation for row {row_index} is time-misaligned: "
            f"manifest={list(manifest_times)}, state={list(state_times)}, max_error={max(errors):.3f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nuplan", "navsim", "waymo"), required=True)
    parser.add_argument("--task-success-field", default="task_success")
    parser.add_argument("--allow-unmatched", action="store_true")
    args = parser.parse_args()
    annotations = {}
    for annotation in _read(args.annotations):
        for key in _keys(annotation):
            if key in annotations:
                raise ValueError(f"duplicate annotation join key: {key}")
            annotations[key] = annotation
    output_rows = []
    unmatched = []
    for index, row in enumerate(_read(args.manifest)):
        match = next((annotations[key] for key in _keys(row) if key in annotations), None)
        if match is None:
            unmatched.append(index)
            output_rows.append(row)
            continue
        history = match.get("history_ego_state")
        realized = match.get("realized_future_ego_state")
        if history is None or realized is None:
            raise ValueError(f"annotation for row {index} must contain history_ego_state and realized_future_ego_state")
        if len(history) == 0 or len(realized) == 0 or any(len(state) != 5 for state in history + realized):
            raise ValueError(f"annotation for row {index} has invalid canonical [T,5] state arrays")
        _check_future_times(row, match, index)
        enriched = dict(row)
        enriched.update({
            "dataset": args.dataset,
            "source_key": match.get("source_key") or row.get("source_key") or next(key for key in _keys(row) if key in annotations),
            "scene_name": match.get("scene_name") or row.get("scene_name"),
            "timestamp_us": match.get("timestamp_us") if match.get("timestamp_us") is not None else row.get("timestamp_us"),
            "history_ego_state": history,
            "realized_future_ego_state": realized,
            "state_times_s": match.get("state_times_s"),
            "task_success": match.get(args.task_success_field),
            "task_success_source": match.get("task_success_source", args.task_success_field),
            "state_reference_source": f"{args.dataset}_logged_ego_state",
            "state_annotation_key": next(key for key in _keys(row) if key in annotations),
        })
        output_rows.append(enriched)
    if unmatched and not args.allow_unmatched:
        raise SystemExit(f"unmatched manifest rows: {len(unmatched)}; use --allow-unmatched only for an audit artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in output_rows), encoding="utf-8")
    summary = {
        "protocol": "wam-ego-state-annotation-v1",
        "dataset": args.dataset,
        "manifest_rows": len(output_rows),
        "matched_rows": len(output_rows) - len(unmatched),
        "unmatched_rows": len(unmatched),
        "task_success_field": args.task_success_field,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
