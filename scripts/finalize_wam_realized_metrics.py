#!/usr/bin/env python3
"""Add independent realized-state CC and FCS to decoded WAM groups."""
import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.wam_metrics import (
    foresight_conditioned_success,
    realized_state_counterfactual_consistency,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def native_to_image_clock(states, count):
    states = np.asarray(states, dtype=np.float64)
    if states.shape != (8, 5):
        raise ValueError(f"realized_future_ego_state must be [8,5], got {states.shape}")
    # DriveWAM outputs frames at native future points 2/4/6/8; a native
    # control manifest retains all eight points.
    indices = [1, 3, 5, 7] if count == 4 else list(range(8))
    return states[indices, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoded", required=True)
    ap.add_argument("--branches", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fcs-threshold", type=float, default=0.70)
    args = ap.parse_args()
    decoded = json.loads(Path(args.decoded).read_text(encoding="utf-8"))
    branches = {str(row["branch_id"]): row for row in read_jsonl(args.branches)}
    times = None
    all_fcs = []
    pair_scores = []
    for group in decoded.get("groups", []):
        enriched = []
        image_count = len(group["branches"][0]["imagined_future"])
        times = np.asarray([1.0, 2.0, 3.0, 4.0] if image_count == 4 else [0.5 * i for i in range(1, 9)], dtype=np.float64)
        for branch in group["branches"]:
            source = branches[str(branch["branch_id"])]
            realized = native_to_image_clock(source["realized_future_ego_state"], image_count)
            item = dict(branch)
            item["realized_future"] = realized.tolist()
            item["task_success"] = bool(source["task_success"])
            item["reference_kind"] = "realized_state"
            enriched.append(item)
        group["branches"] = enriched
        logged = next((item for item in enriched if str(branches[str(item["branch_id"])] .get("branch_mode")) == "logged"), enriched[0])
        for item in enriched:
            if item is logged:
                continue
            score = realized_state_counterfactual_consistency(
                np.asarray(logged["imagined_future"], dtype=np.float64),
                np.asarray(item["imagined_future"], dtype=np.float64),
                np.asarray(logged["realized_future"], dtype=np.float64),
                np.asarray(item["realized_future"], dtype=np.float64),
                times,
            )
            pair_scores.append(score)
        fcs = foresight_conditioned_success(
            enriched,
            compatibility_threshold=args.fcs_threshold,
            future_times_s=times,
        )
        group["realized_state_counterfactual_consistency"] = {
            "baseline_vs_branches": pair_scores[-(len(enriched) - 1):],
            "mean": float(np.mean([
                x["realized_state_counterfactual_consistency"]
                for x in pair_scores[-(len(enriched) - 1):]
            ])),
        }
        group["foresight_conditioned_success"] = fcs
        all_fcs.extend(enriched)
    fcs_global = foresight_conditioned_success(
        all_fcs,
        compatibility_threshold=args.fcs_threshold,
        future_times_s=times,
    )
    realized_values = [
        float(x["realized_state_counterfactual_consistency"])
        for x in pair_scores
        if x.get("realized_state_counterfactual_consistency") is not None
    ]
    decoded["summary"]["realized_state_counterfactual_consistency"] = {
        "available": True,
        "pairs": len(realized_values),
        "mean": float(np.mean(realized_values)) if realized_values else None,
        "median": float(np.median(realized_values)) if realized_values else None,
        "mean_foresight_realized_state_compatibility": float(np.mean([
            x["mean_foresight_realized_state_compatibility"] for x in pair_scores
        ])) if pair_scores else None,
        "mean_response_alignment": float(np.mean([
            x["realized_state_response_alignment"]["alignment_score"]
            for x in pair_scores
            if x["realized_state_response_alignment"].get("alignment_score") is not None
        ])) if pair_scores else None,
        "reference": "independent_navsim_pdm_kinematic_bicycle",
    }
    decoded["summary"]["foresight_conditioned_success"] = fcs_global
    decoded["summary"]["realized_future_clock"] = "native points 2/4/6/8 -> 1/2/3/4 s image clock"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decoded, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "realized_pairs": len(realized_values), "fcs": fcs_global}, indent=2))


if __name__ == "__main__":
    main()
