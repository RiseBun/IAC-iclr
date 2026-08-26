#!/usr/bin/env python3
"""Analyze same-history WAM branches as a future-image x action matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.action_image_matrix import action_image_cross_matrix


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(row: dict[str, Any]) -> str:
    for field in ("sample_id", "video_id", "branch_id"):
        if row.get(field) is not None:
            return str(row[field])
    raise ValueError("row has no sample_id, video_id, or branch_id")


def _group_id(row: dict[str, Any]) -> str:
    for field in ("counterfactual_group_id", "history_id", "twin_id", "pair_id"):
        if row.get(field) is not None:
            return str(row[field])
    metadata = row.get("metadata") or {}
    for field in ("counterfactual_group_id", "history_id", "twin_id", "pair_id"):
        if metadata.get(field) is not None:
            return str(metadata[field])
    raise ValueError(f"{_key(row)} has no counterfactual group id")


def _condition_action_id(row: dict[str, Any]) -> str:
    for field in ("condition_action_id", "action_candidate_id", "gt_candidate_id"):
        if row.get(field) is not None:
            return str(row[field])
    supported = row.get("supported_candidate_index")
    candidates = list(row.get("candidates") or [])
    if supported is not None and candidates:
        candidate = candidates[int(supported)]
        return str(candidate.get("candidate_id", supported))
    raise ValueError(f"{_key(row)} has no condition action id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--decision-margin",
        type=float,
        help="Held-out calibrated probability margin; omit for diagnostic-only output.",
    )
    args = parser.parse_args()

    manifest_rows = _read_jsonl(args.manifest)
    score_by_key = {_key(row): row for row in _read_jsonl(args.scores)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_scores = []
    for manifest in manifest_rows:
        branch_id = _key(manifest)
        score = score_by_key.get(branch_id)
        if score is None:
            missing_scores.append(branch_id)
            continue
        grouped[_group_id(manifest)].append({
            "branch_id": branch_id,
            "condition_action_id": _condition_action_id(manifest),
            "candidate_scores": score.get("candidate_scores", []),
            "valid": bool(score.get("valid", True)),
            "abstain_reasons": list(score.get("abstain_reasons", [])),
        })

    results = []
    invalid_groups = []
    for group_id, branches in sorted(grouped.items()):
        try:
            results.append({
                "group_id": group_id,
                **action_image_cross_matrix(
                    branches,
                    temperature=args.temperature,
                    decision_margin=args.decision_margin,
                ),
            })
        except ValueError as error:
            invalid_groups.append({"group_id": group_id, "error": str(error)})

    evaluable = [row for row in results if row["num_evaluable"]]
    weights = np.asarray([row["num_evaluable"] for row in evaluable], dtype=np.float64)
    def weighted(field: str) -> float | None:
        values = np.asarray([row[field] for row in evaluable], dtype=np.float64)
        return float(np.average(values, weights=weights)) if values.size else None

    summary = {
        "protocol": "counterfactual-action-image-matrix-v1",
        "manifest": str(args.manifest),
        "scores": str(args.scores),
        "num_groups": len(results),
        "num_invalid_groups": len(invalid_groups),
        "num_branches": int(sum(row["num_branches"] for row in results)),
        "num_evaluable": int(sum(row["num_evaluable"] for row in results)),
        "num_abstain": int(sum(row["num_abstain"] for row in results)),
        "coverage": (
            float(sum(row["num_evaluable"] for row in results) / sum(row["num_branches"] for row in results))
            if results and sum(row["num_branches"] for row in results) else None
        ),
        "diagonal_top1_accuracy": weighted("diagonal_top1_accuracy"),
        "mean_reciprocal_rank": weighted("mean_reciprocal_rank"),
        "mean_cc_margin": weighted("mean_cc_margin"),
        "mean_energy_margin": weighted("mean_energy_margin"),
        "mean_pairwise_matched_accuracy": weighted("mean_pairwise_matched_accuracy"),
        "missing_scores": missing_scores,
        "invalid_groups": invalid_groups,
        "groups": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "groups"}, indent=2))


if __name__ == "__main__":
    main()
