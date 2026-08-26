#!/usr/bin/env python3
"""Report logged-baseline CC and alternate-baseline controls.

The logged branch is the prespecified baseline. Alternate baselines are a
control for accidental gains caused by subtracting an arbitrary branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.action_image_matrix import decoded_intervention_delta_matrix


def _mode(branch: dict[str, Any]) -> str:
    return str(branch.get("branch_mode") or str(branch.get("branch_id", "")).split("::branch=")[-1])


def analyze(payload: dict[str, Any], times: np.ndarray) -> dict[str, Any]:
    groups = payload.get("groups") or []
    if not groups:
        raise ValueError("matrix output has no groups")
    modes = sorted({_mode(branch) for group in groups for branch in group.get("branches", [])})
    reports: dict[str, dict[str, float | int | None]] = {}
    for baseline_mode in modes:
        values = []
        missing = 0
        for group in groups:
            branches = group.get("branches", [])
            indices = [_mode(branch) for branch in branches]
            if baseline_mode not in indices:
                missing += 1
                continue
            values.append(decoded_intervention_delta_matrix(
                branches, times, baseline_index=indices.index(baseline_mode)
            ))
        reports[baseline_mode] = {
            "groups": len(values),
            "missing_groups": missing,
            "diagonal_top1_accuracy": float(np.mean([v["diagonal_top1_accuracy"] for v in values])) if values else None,
            "mean_cc_margin": float(np.mean([v["mean_cc_margin"] for v in values])) if values else None,
            "mean_reciprocal_rank": float(np.mean([v["mean_reciprocal_rank"] for v in values])) if values else None,
        }
    return {
        "protocol": "wam-intervention-baseline-control-v1",
        "baseline_definition": "logged branch is prespecified; alternate modes are controls",
        "future_times_s": times.tolist(),
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-times", type=float, nargs="+", required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.matrix.read_text(encoding="utf-8")), np.asarray(args.future_times, dtype=np.float64))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
