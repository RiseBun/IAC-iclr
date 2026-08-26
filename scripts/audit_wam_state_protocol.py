#!/usr/bin/env python3
"""Audit calibration-free WAM state protocol readiness."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _has(row: dict, *keys: str) -> bool:
    return any(row.get(key) is not None for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-closed-loop", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("twin_id") or row.get("pair_id") or row.get("video_id") or "")].append(row)
    branch_counts = Counter()
    action_ready = history_state = realized_state = fcs_ready = 0
    missing = []
    for pair_id, branches in sorted(groups.items()):
        branch_counts[len(branches)] += 1
        if len(branches) < 2:
            missing.append({"pair_id": pair_id, "field": "branches", "reason": "same-history counterfactual pair needs >=2 branches"})
        for index, row in enumerate(branches):
            has_action = _has(row, "action_condition", "executed_action", "action", "candidates")
            has_future = _has(row, "generated_future", "imagined_future", "future_images")
            has_history_state = _has(row, "history_ego_state", "history_state")
            has_realized_state = _has(row, "realized_future_ego_state", "future_ego_state", "ego_state_future")
            has_success = row.get("task_success") is not None
            action_ready += int(has_action and has_future)
            history_state += int(has_history_state)
            realized_state += int(has_realized_state)
            fcs_ready += int(has_realized_state and has_success)
            for field, present, reason in (
                ("action_condition/generated_future", has_action and has_future, "needed for action-conditioned state response"),
                ("history_ego_state", has_history_state, "optional for score, required to document WAM input"),
                ("realized_future_ego_state", has_realized_state, "needed for closed-loop execution fidelity"),
                ("task_success", has_success, "needed for Foresight-Conditioned Success"),
            ):
                if not present:
                    missing.append({"pair_id": pair_id, "branch": index, "field": field, "reason": reason})
    result = {
        "protocol": "wam-ego-state-paired-v1",
        "rows": len(rows),
        "pairs": len(groups),
        "branch_count_distribution": dict(branch_counts),
        "action_response_ready_rows": action_ready,
        "history_ego_state_rows": history_state,
        "realized_future_ego_state_rows": realized_state,
        "foresight_conditioned_success_ready_rows": fcs_ready,
        "action_response_ready": bool(groups) and action_ready == len(rows) and all(len(v) >= 2 for v in groups.values()),
        "closed_loop_ready": bool(groups) and fcs_ready == len(rows) and all(len(v) >= 2 for v in groups.values()),
        "missing_or_advisory": missing[:100],
        "missing_count": len(missing),
        "note": "Missing calibration does not block state-space CC; missing realized future ego state blocks closed-loop/FCS claims.",
    }
    print(json.dumps(result, indent=2))
    if args.require_closed_loop and not result["closed_loop_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
