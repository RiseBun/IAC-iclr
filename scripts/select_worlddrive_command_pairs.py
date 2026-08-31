#!/usr/bin/env python3
"""Select pre-registered material native action pairs before pixel generation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--branch-a", default="command_0")
    parser.add_argument("--branch-b", default="command_2")
    parser.add_argument("--min-max-lateral-delta-m", type=float, default=1.0)
    parser.add_argument("--min-max-yaw-delta-rad", type=float, default=0.1)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(dict)
    for row in rows:
        grouped[str(row["counterfactual_group_id"])][str(row["branch_id"])] = row
    selected = []
    details = []
    for group_id, branches in sorted(grouped.items()):
        if args.branch_a not in branches or args.branch_b not in branches:
            raise ValueError(f"{group_id}: missing requested action pair")
        first = np.asarray(branches[args.branch_a]["stage2_trajectory"], dtype=np.float64)
        second = np.asarray(branches[args.branch_b]["stage2_trajectory"], dtype=np.float64)
        delta = second - first
        delta[:, 2] = (delta[:, 2] + np.pi) % (2 * np.pi) - np.pi
        max_lateral = float(np.max(np.abs(delta[:, 1])))
        max_yaw = float(np.max(np.abs(delta[:, 2])))
        passed = (
            max_lateral >= args.min_max_lateral_delta_m
            or max_yaw >= args.min_max_yaw_delta_rad
        )
        details.append({
            "counterfactual_group_id": group_id,
            "max_lateral_delta_m": max_lateral,
            "max_yaw_delta_rad": max_yaw,
            "passed": passed,
        })
        if passed:
            selected.extend((branches[args.branch_a], branches[args.branch_b]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
    report = {
        "protocol": "worlddrive-native-action-material-pair-gate-v1",
        "groups_input": len(grouped),
        "groups_selected": len(selected) // 2,
        "records_selected": len(selected),
        "pair_roles": [args.branch_a, args.branch_b],
        "thresholds": {
            "min_max_lateral_delta_m": args.min_max_lateral_delta_m,
            "min_max_yaw_delta_rad": args.min_max_yaw_delta_rad,
        },
        "selection_occurs_before_pixel_generation": True,
        "details": details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
