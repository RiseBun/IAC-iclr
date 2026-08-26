#!/usr/bin/env python3
"""Normalize NAVSIM, Waymo, or NuPlan JSONL exports into state annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.state_protocol import navsim_pdm_success, navsim_states, nuplan_states, waymo_states


ADAPTERS = {"nuplan": nuplan_states, "navsim": navsim_states, "waymo": waymo_states}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--navsim-pdm-threshold", type=float, default=0.5)
    args = parser.parse_args()
    output = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        records = list(row.get("records") or [])
        history_count = int(row.get("history_count", 0))
        if history_count < 1 or history_count >= len(records):
            raise ValueError(f"line {line_number}: history_count must split non-empty history/future records")
        states = ADAPTERS[args.dataset](records, anchor_index=history_count - 1)
        task_success = row.get("task_success")
        task_source = row.get("task_success_source")
        if task_success is None and args.dataset == "navsim":
            task_success = navsim_pdm_success(row.get("pdm_score"), threshold=args.navsim_pdm_threshold)
            if task_success is not None:
                task_source = f"navsim_pdm_score>={args.navsim_pdm_threshold}"
        output.append({
            **{key: row[key] for key in ("source_key", "video_id", "sample_id", "pair_id", "scene_name", "timestamp_us") if row.get(key) is not None},
            "history_ego_state": states[:history_count].tolist(),
            "realized_future_ego_state": states[history_count:].tolist(),
            "state_times_s": row.get("future_times_s"),
            "task_success": task_success,
            "task_success_source": task_source,
            "state_reference_source": f"{args.dataset}_logged_ego_state",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({
        "protocol": "wam-ego-state-annotation-v1",
        "dataset": args.dataset,
        "rows": len(output),
        "task_success_rows": sum(row["task_success"] is not None for row in output),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
