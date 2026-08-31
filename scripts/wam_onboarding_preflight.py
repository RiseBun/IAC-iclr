#!/usr/bin/env python3
"""Fail-closed onboarding check for a new WAM adapter manifest.

The check separates three questions that are often conflated:
1. Can the generated images be measured (Level-1)?
2. Is there a controlled native action -> image intervention (Level-2)?
3. Is an independently realized rollout available (FCS)?

The manifest may use a model-native time grid.  A non-canonical grid is not
silently relabelled; it must declare ``continuous_resample`` in the capability
file before formal Level-2 scoring is allowed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("JSON manifest must contain a list")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _capability(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "native_action_head": False,
            "external_trajectory_control": False,
            "time_alignment": "unsupported",
            "independent_rollout": False,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("capability JSON must contain an object")
    return {
        "native_action_head": bool(value.get("native_action_head", False)),
        "external_trajectory_control": bool(value.get("external_trajectory_control", False)),
        "time_alignment": str(value.get("time_alignment", "unsupported")),
        "independent_rollout": bool(value.get("independent_rollout", False)),
    }


def audit_rows(
    rows: list[dict[str, Any]],
    capability: dict[str, Any],
    canonical_times: np.ndarray,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    native_grids: set[tuple[float, ...]] = set()
    image_ready = True
    realized_ready = True
    success_ready = True
    for index, row in enumerate(rows):
        group = str(row.get("counterfactual_group_id") or row.get("source_key") or "")
        if not group:
            issues.append({"row": index, "field": "counterfactual_group_id", "reason": "required"})
        groups[group].append(row)
        history = row.get("history_images") or []
        future = row.get("future_images") or []
        times = np.asarray(row.get("future_times_s") or [], dtype=np.float64)
        row_issues = []
        image_row_issues = []
        if not isinstance(history, list) or len(history) < 2:
            issue = ("history_images", "at least two frames required")
            row_issues.append(issue)
            image_row_issues.append(issue)
        if not isinstance(future, list) or not future:
            issue = ("future_images", "generated future frames required")
            row_issues.append(issue)
            image_row_issues.append(issue)
        if times.ndim != 1 or len(times) != len(future) or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            issue = ("future_times_s", "finite increasing timestamps must match future_images")
            row_issues.append(issue)
            image_row_issues.append(issue)
        else:
            native_grids.add(tuple(np.round(times, 6).tolist()))
        if not any(row.get(key) is not None for key in ("action_trajectory", "action_condition", "executed_action", "trajectory")):
            row_issues.append(("action_condition", "action trajectory/condition required"))
        if not any(row.get(key) is not None for key in ("realized_future_ego_state", "realized_future", "future_ego_state")):
            realized_ready = False
        if row.get("task_success") is None:
            success_ready = False
        for field, reason in row_issues:
            issues.append({"row": index, "field": field, "reason": reason})
        image_ready &= not image_row_issues

    branch_ready = all(len(members) >= 2 for members in groups.values()) and bool(groups)
    intervention_ready = capability["external_trajectory_control"] and all(
        bool(row.get("action_injection_verified", False)) for row in rows
    )
    time_exact = bool(native_grids) and all(
        len(grid) == len(canonical_times) and np.allclose(grid, canonical_times, atol=1e-6)
        for grid in native_grids
    )
    time_alignment = capability["time_alignment"]
    if time_alignment not in {"exact", "continuous_resample", "unsupported"}:
        issues.append({"field": "time_alignment", "reason": "must be exact, continuous_resample, or unsupported"})
        time_alignment = "unsupported"
    if not time_exact and time_alignment == "exact":
        issues.append({"field": "time_alignment", "reason": "declared exact but native grid differs from canonical grid"})
    if time_exact and time_alignment == "unsupported":
        issues.append({"field": "time_alignment", "reason": "canonical grid is exact but adapter did not declare alignment"})

    image_probe_ready = bool(rows) and image_ready
    action_response_ready = image_probe_ready and branch_ready and intervention_ready and time_alignment != "unsupported"
    formal_level2_ready = action_response_ready and capability["native_action_head"]
    fcs_ready = formal_level2_ready and capability["independent_rollout"] and realized_ready and success_ready
    return {
        "protocol": "iac-wam-onboarding-preflight-v1",
        "rows": len(rows),
        "groups": len(groups),
        "native_time_grids": [list(grid) for grid in sorted(native_grids)],
        "canonical_times_s": canonical_times.tolist(),
        "canonical_time_grid_exact": time_exact,
        "capability": capability,
        "image_probe_ready": image_probe_ready,
        "counterfactual_image_ready": action_response_ready,
        "formal_level2_ready": formal_level2_ready,
        "fcs_ready": fcs_ready,
        "next_action": (
            "run Level-1"
            if image_probe_ready and not formal_level2_ready
            else "run CCFC then independent rollout"
            if formal_level2_ready and not fcs_ready
            else "run full IAC report"
            if fcs_ready
            else "fix adapter contract and rerun preflight"
        ),
        "issues": issues[:200],
        "issue_count": len(issues),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capability-json", type=Path)
    parser.add_argument("--canonical-times", default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = np.asarray([float(value) for value in args.canonical_times.split(",") if value.strip()], dtype=np.float64)
    report = audit_rows(_read_manifest(args.manifest), _capability(args.capability_json), canonical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
