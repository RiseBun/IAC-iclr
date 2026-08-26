#!/usr/bin/env python3
"""Attach RAFT event posteriors to an existing decoded WAM branch report.

This adapter is for records where ``imagined_future`` is already the output of
the frozen IAC image probe. It does not run a second image model. It converts
decoded trajectories, action trajectories, and independent realized states to
the event contract consumed by ``evaluate_event_causal_metrics.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.event_metrics import action_trajectory_event_target
from iac_new.maneuver import extract_maneuver


def _posterior(trajectory: Any, times: np.ndarray) -> list[dict[str, Any]] | None:
    if trajectory is None:
        return None
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) != len(times):
        raise ValueError("trajectory must have shape [len(times),3]")
    return list(extract_maneuver(values, times)["event_posterior"])


def build_groups(report: dict[str, Any], *, times: np.ndarray) -> list[dict[str, Any]]:
    groups = []
    for group_index, group in enumerate(report.get("groups", [])):
        branches = []
        for branch_index, branch in enumerate(group.get("branches", [])):
            action = branch.get("executed_action", branch.get("action_condition"))
            imagined = branch.get("imagined_future")
            realized = branch.get("realized_future")
            if action is None or imagined is None:
                raise ValueError(f"branch {branch_index} is missing action or imagined_future")
            action_target = action_trajectory_event_target(np.asarray(action), times)
            imagined_posterior = _posterior(imagined, times)
            realized_target = _posterior(realized, times)
            branches.append({
                "branch_id": str(branch.get("branch_id", branch_index)),
                "condition_action_id": str(branch.get("condition_action_id", branch.get("branch_id", branch_index))),
                "generated_future_id": branch.get("generated_future_id", branch.get("branch_id", branch_index)),
                "imagined_event_source": "diagnostic_decoded_trajectory_adapter_v1",
                "imagined_event_posterior": imagined_posterior,
                "action_event_target": action_target,
                "realized_event_target": realized_target,
                "realized_event_source": "simulator_state" if realized is not None else None,
                "task_success": branch.get("task_success"),
                "valid": bool(branch.get("valid", True)),
                "abstain_reasons": list(branch.get("abstain_reasons", [])),
            })
        groups.append({
            "counterfactual_group_id": str(group.get("counterfactual_group_id", group_index)),
            "scene_id": group.get("scene_id"),
            "history_id": group.get("history_id"),
            "generation_seed": group.get("generation_seed"),
            "branches": branches,
        })
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-times", type=float, nargs="+", required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    groups = build_groups(report, times=np.asarray(args.future_times, dtype=np.float64))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group) + "\n")
    print(json.dumps({"groups": len(groups), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
