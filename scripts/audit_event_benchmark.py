#!/usr/bin/env python3
"""Audit formal benchmark readiness and optionally freeze a scene split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.event_benchmark import audit_event_benchmark_groups, scene_disjoint_split
from scripts.evaluate_event_causal_metrics import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", action="append", default=None)
    parser.add_argument("--minimum-branches", type=int, default=3)
    parser.add_argument("--minimum-pairwise-action-distance", type=float, default=0.05)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.6)
    parser.add_argument("--split-seed", default="event-causal-v1")
    parser.add_argument("--require-level", type=int, choices=(1, 2, 3))
    args = parser.parse_args()

    groups = read_jsonl(args.groups)
    result = audit_event_benchmark_groups(
        groups,
        dimensions=tuple(args.dimension or ["lateral"]),
        minimum_branches=args.minimum_branches,
        minimum_pairwise_action_distance=args.minimum_pairwise_action_distance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.split_output is not None:
        split = scene_disjoint_split(
            groups,
            calibration_fraction=args.calibration_fraction,
            seed=args.split_seed,
        )
        args.split_output.parent.mkdir(parents=True, exist_ok=True)
        args.split_output.write_text(json.dumps(split, indent=2), encoding="utf-8")

    summary = {
        key: result[key]
        for key in (
            "num_groups",
            "level1_ready_groups",
            "level2_ready_groups",
            "level3_ready_groups",
            "formal_event_cc_fcs_ready",
            "causal_closure_fui_ready",
        )
    }
    print(json.dumps(summary, indent=2))
    if args.require_level is not None:
        ready_key = {
            1: "level1_action_response_ready",
            2: "level2_event_cc_fcs_ready",
            3: "level3_fui_ready",
        }[args.require_level]
        if not all(row[ready_key] for row in result["rows"]):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
