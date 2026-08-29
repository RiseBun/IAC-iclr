#!/usr/bin/env python3
"""Fail-closed readiness audit for Level-2 continuous counterfactual records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_counterfactual_continuous_alignment import audit_pair, read_jsonl


def audit_records(rows: list[dict[str, Any]], *, require_eight_frame_four_second: bool = False) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("counterfactual_group_id"))].append(row)
    reports = []
    for group_id, branches in sorted(groups.items()):
        roles = {str(row.get("branch_role")): row for row in branches}
        issues: list[str] = []
        if set(roles) != {"clear", "risk"} or len(branches) != 2:
            issues.append("expected_exactly_one_clear_and_one_risk")
        else:
            issues.extend(audit_pair(group_id, roles))
        if require_eight_frame_four_second:
            for role, branch in roles.items():
                times = np.asarray(branch.get("future_times_s") or [], dtype=np.float64)
                if times.shape != (8,) or not np.all(np.isfinite(times)) or abs(float(times[-1]) - 4.0) > 0.05:
                    issues.append(f"{role}_not_eight_frames_four_seconds")
        reports.append({
            "counterfactual_group_id": group_id,
            "branches": len(branches),
            "ready": not issues,
            "issues": sorted(set(issues)),
        })
    ready = [item for item in reports if item["ready"]]
    return {
        "protocol": "counterfactual-continuous-readiness-audit-v1",
        "groups": len(reports),
        "records": len(rows),
        "ready_groups": len(ready),
        "invalid_groups": len(reports) - len(ready),
        "formal_level2_input_ready": bool(reports) and len(ready) == len(reports),
        "require_eight_frame_four_second": bool(require_eight_frame_four_second),
        "groups_detail": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-eight-frame-four-second", action="store_true")
    args = parser.parse_args()
    report = audit_records(read_jsonl(args.records), require_eight_frame_four_second=args.require_eight_frame_four_second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "groups_detail"}, indent=2, ensure_ascii=False))
    if not report["formal_level2_input_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
