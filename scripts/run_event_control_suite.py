#!/usr/bin/env python3
"""Run the benchmark together with causal sanity-check controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.event_benchmark import build_event_control_groups
from scripts.evaluate_event_causal_metrics import evaluate_groups, read_jsonl


def _evaluate(groups: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_groups(
        groups,
        dimensions=tuple(args.dimension or ["lateral"]),
        minimum_observability=args.minimum_observability,
        minimum_interval_coverage=args.minimum_interval_coverage,
        temperature=args.temperature,
        compatibility_threshold=args.compatibility_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", action="append", default=None)
    parser.add_argument("--minimum-observability", type=float, default=0.25)
    parser.add_argument("--minimum-interval-coverage", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--compatibility-threshold", type=float, default=0.70)
    args = parser.parse_args()

    groups = read_jsonl(args.groups)
    reports = {"observed": _evaluate(groups, args)}
    for control in ("oracle", "identical_future", "action_swap"):
        reports[control] = _evaluate(build_event_control_groups(groups, control), args)
    result = {
        "protocol": "event-causal-control-suite-v1",
        "reports": reports,
        "top1_accuracy": {
            name: report["event_cc"]["diagonal_top1_accuracy"]
            for name, report in reports.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["top1_accuracy"], indent=2))


if __name__ == "__main__":
    main()
