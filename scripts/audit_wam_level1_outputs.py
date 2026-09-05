#!/usr/bin/env python3
"""Audit completed WAM future-image outputs before Level-1 conversion.

The audit is deliberately fail-closed: a directory of images is not enough.
Each row must retain lineage to a base sample, expose an independent action
head, and provide a native future axis with either 4 or 8 frames covering
approximately 4.0 seconds (history is separate and usually 4 frames).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ALLOWED_FUTURE_COUNTS = frozenset({4, 8})


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
    allowed_future_counts: frozenset[int] | set[int] = ALLOWED_FUTURE_COUNTS,
    expected_future_count: int | None = None,
    check_files: bool = False,
) -> dict[str, Any]:
    if expected_future_count is not None:
        if expected_future_count not in allowed_future_counts:
            raise ValueError(f"expected_future_count must be one of {sorted(allowed_future_counts)}")
        allowed = frozenset({expected_future_count})
    else:
        allowed = frozenset(allowed_future_counts)
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    complete = 0
    future_count_histogram: dict[str, int] = {str(count): 0 for count in sorted(allowed)}
    for index, row in enumerate(rows):
        branch_id = str(row.get("branch_id") or "")
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
        future_count = len(images)
        if future_count not in allowed or any(not str(path) for path in images):
            row_issues.append(f"future_images_must_have_one_of_{sorted(allowed)}_paths")
        else:
            future_count_histogram[str(future_count)] += 1
        if check_files:
            row_issues.extend("missing_image:" + str(path) for path in images if not Path(path).is_file())
        times = np.asarray(row.get("future_times_s"), dtype=np.float64)
        if times.shape != (future_count,) or not np.all(np.isfinite(times)) or (
            future_count >= 2 and np.any(np.diff(times) <= 0.0)
        ):
            row_issues.append("future_times_s_invalid")
        elif future_count in allowed and (times[0] <= 0.0 or not (3.95 <= float(times[-1]) <= 4.05)):
            row_issues.append("future_times_s_must_cover_about_4_seconds")
        action = row.get("action_condition", {}).get("trajectory")
        if action is None:
            action = row.get("action_trajectory")
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (future_count, 3) or not np.all(np.isfinite(action_array)):
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
        "allowed_future_counts": sorted(allowed),
        "future_count_histogram": future_count_histogram,
        "issues": issues,
        "checked_files": bool(check_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument(
        "--expected-future-count",
        type=int,
        choices=sorted(ALLOWED_FUTURE_COUNTS),
        default=None,
        help="pin exactly 4 or 8 future frames; default accepts either",
    )
    args = parser.parse_args()
    report = audit_rows(
        read_jsonl(args.generated),
        expected_future_count=args.expected_future_count,
        check_files=args.check_files,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not report["formal_level1_input_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
