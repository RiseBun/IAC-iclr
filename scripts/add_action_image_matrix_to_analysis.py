#!/usr/bin/env python3
"""Add cross-scored action-image matrices to an existing paired WAM report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.action_image_matrix import decoded_trajectory_cross_matrix


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--decision-margin", type=float)
    args = parser.parse_args()

    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    times_by_group = {}
    for row in _read_jsonl(args.manifest):
        group_id = str(row.get("twin_id") or row.get("counterfactual_group_id"))
        times = row.get("frame_times_s") or row.get("future_times_s")
        if not group_id or times is None:
            raise ValueError("manifest rows require a group id and future frame times")
        previous = times_by_group.setdefault(group_id, times)
        if not np.allclose(previous, times):
            raise ValueError(f"{group_id}: branches have different future frame times")

    matrices = []
    for pair in report.get("results", []):
        group_id = str(pair.get("twin_id") or pair.get("group_id"))
        if group_id not in times_by_group:
            raise ValueError(f"{group_id}: no frame times in manifest")
        matrix = decoded_trajectory_cross_matrix(
            list(pair["branches"]),
            np.asarray(times_by_group[group_id], dtype=np.float64),
            temperature=args.temperature,
            decision_margin=args.decision_margin,
        )
        pair["action_image_matrix"] = matrix
        matrices.append(matrix)

    report["action_image_matrix_summary"] = {
        "protocol": "counterfactual-action-image-matrix-v1",
        "num_groups": len(matrices),
        "num_branches": int(sum(row["num_branches"] for row in matrices)),
        "num_evaluable": int(sum(row["num_evaluable"] for row in matrices)),
        "coverage": float(np.mean([row["coverage"] for row in matrices])) if matrices else None,
        "diagonal_top1_accuracy": float(
            np.mean([row["diagonal_top1_accuracy"] for row in matrices])
        ) if matrices else None,
        "mean_reciprocal_rank": float(
            np.mean([row["mean_reciprocal_rank"] for row in matrices])
        ) if matrices else None,
        "mean_cc_margin": float(
            np.mean([row["mean_cc_margin"] for row in matrices])
        ) if matrices else None,
        "mean_energy_margin": float(
            np.mean([row["mean_energy_margin"] for row in matrices])
        ) if matrices else None,
        "mean_pairwise_response_tv": float(
            np.mean([row["mean_pairwise_response_tv"] for row in matrices])
        ) if matrices else None,
        "evidence_source": "image_decoded_ego_trajectory",
        "decision_status": (
            "calibrated" if args.decision_margin is not None else "diagnostic_only"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["action_image_matrix_summary"], indent=2))


if __name__ == "__main__":
    main()
