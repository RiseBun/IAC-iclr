#!/usr/bin/env python3
"""Fail-closed audit for paired WAM counterfactual records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pair_ids = [str(row.get("pair_id") or "") for row in rows]
    duplicate_pair_ids = sorted(pair_id for pair_id, count in Counter(pair_ids).items() if pair_id and count > 1)
    missing = []
    branch_counts = []
    same_history = []
    same_task = []
    for index, row in enumerate(rows):
        branches = list(row.get("branches") or [])
        branch_counts.append(len(branches))
        if len(branches) < 2:
            missing.append({"row": index, "field": "branches", "reason": "at least two branches required"})
        for field in ("history_id", "task_id"):
            if row.get(field) is None:
                missing.append({"row": index, "field": field, "reason": "pair identity is required"})
        if row.get("history_id") is not None:
            same_history.append(True)
        if row.get("task_id") is not None:
            same_task.append(True)
        for branch_index, branch in enumerate(branches):
            for field in ("imagined_future", "executed_action"):
                if branch.get(field) is None:
                    missing.append({"row": index, "branch": branch_index, "field": field, "reason": "required for causal metric"})
            if branch.get("task_success") is None:
                missing.append({"row": index, "branch": branch_index, "field": "task_success", "reason": "required for FCS"})
    result = {
        "protocol": "wam-paired-future-action-v1",
        "rows": len(rows),
        "pair_ids_unique": not duplicate_pair_ids,
        "duplicate_pair_ids": duplicate_pair_ids,
        "min_branches": min(branch_counts) if branch_counts else 0,
        "max_branches": max(branch_counts) if branch_counts else 0,
        "causal_metric_ready": bool(rows) and not missing and not duplicate_pair_ids,
        "missing_or_invalid": missing[:50],
        "missing_count": len(missing),
        "note": "Rows without same-history counterfactual branches cannot support causal Counterfactual Consistency.",
    }
    print(json.dumps(result, indent=2))
    if not result["causal_metric_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
