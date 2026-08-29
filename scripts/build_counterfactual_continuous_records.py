#!/usr/bin/env python3
"""Join completed Level-1 image probes into Level-2 clear/risk records.

The join happens only after candidate-blind decoding.  The resulting JSONL is
the direct input to ``evaluate_counterfactual_continuous_alignment.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_records(manifest_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = {str(row.get("sample_id")): row for row in score_rows}
    records: list[dict[str, Any]] = []
    missing_scores: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        sample_id = str(row.get("sample_id") or "")
        score = scores.get(sample_id)
        if score is None:
            missing_scores.append(sample_id)
            continue
        group_id = str(row.get("counterfactual_group_id") or "")
        role = str(row.get("branch_role") or "")
        if not group_id or role not in {"clear", "risk"}:
            raise ValueError(f"{sample_id}: counterfactual_group_id and branch_role(clear/risk) are required")
        if score.get("decoder", {}).get("protocol") != "candidate-blind-continuous-trajectory-v1":
            raise ValueError(f"{sample_id}: score is not a candidate-blind continuous decoder output")
        if score.get("candidate_bank_used_by_decoder") is not False:
            raise ValueError(f"{sample_id}: decoder candidate-bank audit failed")
        if row.get("realized_future_ego_state") is not None or (row.get("metadata") or {}).get("realized_future_ego_state") is not None:
            raise ValueError(f"{sample_id}: realized future state must not enter Level-2 image records")
        record = {
            "sample_id": sample_id,
            "counterfactual_group_id": group_id,
            "branch_role": role,
            "history_fingerprint": row.get("history_fingerprint") or (row.get("metadata") or {}).get("history_fingerprint"),
            "wam_model_id": row.get("wam_model_id") or (row.get("metadata") or {}).get("wam_model_id"),
            "nuisance_seed": row.get("nuisance_seed") if row.get("nuisance_seed") is not None else (row.get("metadata") or {}).get("nuisance_seed"),
            "future_images_source": "wam_generated",
            "future_times_s": list(row.get("future_times_s") or []),
            "history_ego_state": list((row.get("metadata") or {}).get("history_ego_state") or row.get("history_ego_state") or []),
            "action_trajectory": row.get("action_trajectory"),
            "action_trajectory_source": row.get("action_trajectory_source") or "wam_action_head",
            "candidate_bank_used_by_decoder": False,
            "decoder": score["decoder"],
        }
        records.append(record)
        groups[group_id].append(record)
    issues = []
    for group_id, branches in sorted(groups.items()):
        roles = {str(branch["branch_role"]): branch for branch in branches}
        if set(roles) != {"clear", "risk"} or len(branches) != 2:
            issues.append({"counterfactual_group_id": group_id, "reason": "expected_exactly_clear_and_risk", "branch_count": len(branches)})
            continue
        clear, risk = roles["clear"], roles["risk"]
        for field in ("history_fingerprint", "wam_model_id", "nuisance_seed", "future_times_s"):
            if clear.get(field) != risk.get(field):
                issues.append({"counterfactual_group_id": group_id, "reason": f"mismatched_{field}"})
    if missing_scores or issues:
        raise ValueError(json.dumps({"missing_scores": missing_scores, "pair_issues": issues}, indent=2))
    return records, {"protocol": "counterfactual-continuous-record-assembly-v1", "groups": len(groups), "records": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, summary = build_records(_read(args.manifest), _read(args.scores))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in records), encoding="utf-8")
    print(json.dumps(summary | {"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
