#!/usr/bin/env python3
"""Evaluate event CC and Event-FCS from explicit counterfactual event groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.event_benchmark import evaluate_fui_group
from iac_new.event_metrics import (
    event_counterfactual_matrix,
    event_foresight_conditioned_success,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _weighted_mean(results: list[dict[str, Any]], key: str) -> float | None:
    values = []
    weights = []
    for result in results:
        value = result.get(key)
        weight = int(result.get("num_evaluable", 0))
        if value is not None and weight > 0:
            values.append(float(value))
            weights.append(weight)
    return None if not weights else float(np.average(values, weights=weights))


def _plain_mean(results: list[dict[str, Any]], key: str) -> float | None:
    values = [float(result[key]) for result in results if result.get(key) is not None]
    return None if not values else float(np.mean(values))


def evaluate_groups(
    groups: list[dict[str, Any]],
    *,
    dimensions: tuple[str, ...],
    minimum_observability: float,
    minimum_interval_coverage: float,
    temperature: float,
    compatibility_threshold: float,
) -> dict[str, Any]:
    matrices = []
    episodes = []
    for group_index, group in enumerate(groups):
        group_id = str(group.get("counterfactual_group_id", group_index))
        branches = list(group.get("branches", []))
        matrix = event_counterfactual_matrix(
            branches,
            dimensions=dimensions,
            minimum_observability=minimum_observability,
            minimum_interval_coverage=minimum_interval_coverage,
            temperature=temperature,
        )
        matrix["counterfactual_group_id"] = group_id
        matrices.append(matrix)
        for branch_index, branch in enumerate(branches):
            episodes.append({
                "episode_id": str(branch.get("branch_id", f"{group_id}:{branch_index}")),
                "imagined_event_posterior": branch.get("imagined_event_posterior"),
                "realized_event_target": branch.get("realized_event_target"),
                "realized_event_source": branch.get("realized_event_source"),
                "task_success": branch.get("task_success"),
            })

    total_branches = sum(int(item["num_branches"]) for item in matrices)
    total_evaluable = sum(int(item["num_evaluable"]) for item in matrices)
    fcs = event_foresight_conditioned_success(
        episodes,
        compatibility_threshold=compatibility_threshold,
        dimensions=dimensions,
        minimum_observability=minimum_observability,
        minimum_interval_coverage=minimum_interval_coverage,
    )
    fui_groups = [
        evaluate_fui_group(group, dimensions=dimensions)
        for group in groups
        if group.get("fui_trials")
    ]
    fui = {
        "status": "computed" if fui_groups else "not_computed",
        "num_groups": len(fui_groups),
        "future_follow_rate": _plain_mean(fui_groups, "future_follow_rate"),
        "null_selection_change_rate": _plain_mean(
            fui_groups, "null_selection_change_rate"
        ),
        "fui_lift": _plain_mean(fui_groups, "fui_lift"),
        "groups": fui_groups,
    }
    if not fui_groups:
        fui["reason"] = (
            "FUI requires planner reruns under future permutations and null resamples"
        )
    return {
        "protocol": "event-causal-wam-evaluation-v1",
        "scope": "lateral-event-only" if dimensions == ("lateral",) else list(dimensions),
        "num_groups": len(groups),
        "num_branches": total_branches,
        "num_evaluable_branches": total_evaluable,
        "event_cc_coverage": 0.0 if not total_branches else total_evaluable / total_branches,
        "event_cc": {
            "diagonal_top1_accuracy": _weighted_mean(matrices, "diagonal_top1_accuracy"),
            "mean_reciprocal_rank": _weighted_mean(matrices, "mean_reciprocal_rank"),
            "mean_cc_margin": _weighted_mean(matrices, "mean_cc_margin"),
            "top1_lift_over_chance": _weighted_mean(matrices, "top1_lift_over_chance"),
            "cc_margin_lift_over_cyclic_swap": _weighted_mean(
                matrices, "cc_margin_lift_over_cyclic_swap"
            ),
        },
        "event_fcs": fcs,
        "groups": matrices,
        "fui": fui,
    }


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
    dimensions = tuple(args.dimension or ["lateral"])
    result = evaluate_groups(
        read_jsonl(args.groups),
        dimensions=dimensions,
        minimum_observability=args.minimum_observability,
        minimum_interval_coverage=args.minimum_interval_coverage,
        temperature=args.temperature,
        compatibility_threshold=args.compatibility_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "protocol", "scope", "num_groups", "num_branches", "event_cc_coverage", "event_cc"
    )}, indent=2))


if __name__ == "__main__":
    main()
