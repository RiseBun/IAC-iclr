#!/usr/bin/env python3
"""Compute paired CCFC lift over a specificity-control report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODES = ("metric", "scale_free", "arc_relative")


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    observed = json.loads(args.observed.read_text(encoding="utf-8"))
    control = json.loads(args.control.read_text(encoding="utf-8"))
    control_by_group = {row["counterfactual_group_id"]: row for row in control["reports"]}
    results = {}
    details = []
    for mode_index, mode in enumerate(MODES):
        pairs = []
        for row in observed["reports"]:
            group_id = row["counterfactual_group_id"]
            other = control_by_group.get(group_id)
            if other is None:
                continue
            observed_score = row["continuous_cfc"][mode].get("score")
            control_score = other["continuous_cfc"][mode].get("score")
            if observed_score is None or control_score is None:
                continue
            delta = float(observed_score - control_score)
            pairs.append(delta)
            details.append({
                "counterfactual_group_id": group_id,
                "mode": mode,
                "observed_score": observed_score,
                "control_score": control_score,
                "lift": delta,
            })
        values = np.asarray(pairs, dtype=np.float64)
        results[mode] = {
            "count": len(values),
            "mean_lift": None if not len(values) else float(values.mean()),
            "median_lift": None if not len(values) else float(np.median(values)),
            "positive_fraction": None if not len(values) else float(np.mean(values > 0)),
            "bootstrap_confidence_interval_95": None if not len(values) else _bootstrap(
                values, args.seed + mode_index
            ),
        }
    report = {
        "protocol": "ccfc-paired-specificity-control-lift-v1",
        "observed_claim_scope": observed.get("claim_scope"),
        "control_claim_scope": control.get("claim_scope"),
        "modes": results,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
