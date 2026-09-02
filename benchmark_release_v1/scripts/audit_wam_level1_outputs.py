#!/usr/bin/env python3
"""Audit completed WAM future-image outputs before Level-1 conversion.

The audit is deliberately fail-closed: a directory of images is not enough.
Each row must retain lineage to a base sample, expose an independent action
head, and contain exactly the 8-frame/4-second output used by the metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_key(row: dict[str, Any]) -> str:
    return str(row.get("source_key") or (row.get("metadata") or {}).get("source_key") or "")


def _images(row: dict[str, Any]) -> list[str]:
    value = row.get("future_images")
    if value is None:
        value = row.get("generated_future_images")
    return value if isinstance(value, list) else []


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    expected_future_count: int = 8,
    check_files: bool = False,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    complete = 0
    for index, row in enumerate(rows):
        branch_id = str(row.get("branch_id") or "")
        prefix = f"row[{index}]"
        row_issues: list[str] = []
        if not branch_id:
            row_issues.append("missing_branch_id")
        elif branch_id in seen:
            row_issues.append("duplicate_branch_id")
        seen.add(branch_id)
        if not _source_key(row):
            row_issues.append("missing_source_key")
        if row.get("future_images_source") != "wam_generated":
            row_issues.append("future_images_source_is_not_wam_generated")
        if row.get("wam_generation_status") not in {None, "complete"}:
            row_issues.append("generation_is_not_complete")
        if not str(row.get("wam_model_id") or ""):
            row_issues.append("missing_wam_model_id")
        images = _images(row)
        if len(images) != expected_future_count or any(not str(path) for path in images):
            row_issues.append(f"future_images_must_have_{expected_future_count}_paths")
        if check_files:
            row_issues.extend("missing_image:" + str(path) for path in images if not Path(path).is_file())
        times = np.asarray(row.get("future_times_s"), dtype=np.float64)
        if times.shape != (expected_future_count,) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            row_issues.append("future_times_s_invalid")
        elif times[0] <= 0.0 or not (3.95 <= float(times[-1]) <= 4.05):
            row_issues.append("future_times_s_does_not_cover_0p5_to_4p0_seconds")
        action = row.get("action_condition", {}).get("trajectory")
        if action is None:
            action = row.get("action_trajectory")
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (expected_future_count, 3) or not np.all(np.isfinite(action_array)):
            row_issues.append("independent_action_head_trajectory_invalid")
        action_source = str(row.get("action_source") or (row.get("metadata") or {}).get("action_source") or "")
        if not action_source:
            row_issues.append("missing_independent_action_source")
        elif action_source in {"logged", "oracle", "proxy", "candidate"}:
            row_issues.append("action_source_is_not_independent")
        if row.get("realized_future_ego_state") is not None or (row.get("metadata") or {}).get("realized_future_ego_state") is not None:
            row_issues.append("realized_future_state_leakage")
        if row_issues:
            issues.append({"branch_id": branch_id or None, "issues": row_issues})
        else:
            complete += 1
    return {
        "protocol": "wam-generated-level1-output-audit-v1",
        "rows": len(rows),
        "complete_rows": complete,
        "invalid_rows": len(issues),
        "formal_level1_input_ready": bool(rows) and not issues,
        "issues": issues,
        "checked_files": bool(check_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    report = audit_rows(read_jsonl(args.generated), check_files=args.check_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not report["formal_level1_input_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
