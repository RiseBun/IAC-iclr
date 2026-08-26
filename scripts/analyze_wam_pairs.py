#!/usr/bin/env python3
"""Evaluate paired WAM imagined-future/action records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.wam_metrics import (
    foresight_conditioned_success,
    paired_counterfactual_consistency,
    realized_state_counterfactual_consistency,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compatibility-threshold", type=float, default=0.70)
    args = parser.parse_args()
    results = []
    cc_values = []
    fcs_values = []
    realized_cc_values = []
    for line_number, line in enumerate(args.pairs.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        branches = list(row.get("branches") or [])
        if len(branches) < 2:
            raise ValueError(f"line {line_number}: a pair needs at least two branches")
        times = np.asarray(row.get("future_times_s"), dtype=np.float64)
        if times.ndim != 1 or times.size < 1:
            raise ValueError(f"line {line_number}: future_times_s is required")
        pair_results = []
        for index in range(1, len(branches)):
            cc = paired_counterfactual_consistency(
                branches[0], branches[index], times
            )
            if branches[0].get("realized_future") is not None and branches[index].get("realized_future") is not None:
                realized = realized_state_counterfactual_consistency(
                    np.asarray(branches[0]["imagined_future"]),
                    np.asarray(branches[index]["imagined_future"]),
                    np.asarray(branches[0]["realized_future"]),
                    np.asarray(branches[index]["realized_future"]),
                    times,
                )
                cc["realized_state_counterfactual_consistency"] = realized["realized_state_counterfactual_consistency"]
                cc["realized_state_response_alignment"] = realized["realized_state_response_alignment"]
                if realized["realized_state_counterfactual_consistency"] is not None:
                    realized_cc_values.append(float(realized["realized_state_counterfactual_consistency"]))
            pair_results.append({"branch_a": branches[0].get("branch_id", "0"), "branch_b": branches[index].get("branch_id", str(index)), **cc})
            if cc["counterfactual_consistency"] is not None:
                cc_values.append(float(cc["counterfactual_consistency"]))
        try:
            fcs = foresight_conditioned_success(
                branches,
                compatibility_threshold=args.compatibility_threshold,
                future_times_s=times,
            )
        except ValueError as error:
            fcs = {
                "status": "unavailable",
                "reason": str(error),
                "reference_kind": "realized_state_required",
            }
        fcs_values.append(float(fcs["foresight_conditioned_success"]) if fcs.get("foresight_conditioned_success") is not None else np.nan)
        results.append({"pair_id": row.get("pair_id"), "counterfactual_pairs": pair_results, "foresight_conditioned_success": fcs})
    summary = {
        "protocol": "wam-ego-state-paired-v1",
        "pairs": len(results),
        "mean_counterfactual_consistency": float(np.mean(cc_values)) if cc_values else None,
        "median_counterfactual_consistency": float(np.median(cc_values)) if cc_values else None,
        "mean_realized_state_counterfactual_consistency": float(np.mean(realized_cc_values)) if realized_cc_values else None,
        "mean_foresight_conditioned_success": float(np.nanmean(fcs_values)) if fcs_values and not np.all(np.isnan(fcs_values)) else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
