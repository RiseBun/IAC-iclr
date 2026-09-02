#!/usr/bin/env python3
"""Fail-closed audit for a paired, action-conditioned WAM benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _group_id(row: dict[str, Any]) -> str:
    return str(row.get("twin_id") or row.get("pair_id") or row.get("group_id") or row.get("source_key") or row.get("video_id") or "")


def _branch_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("branch_id") or row.get("video_id") or row.get("sample_id") or f"row-{index}")


def _history_fingerprint(row: dict[str, Any]) -> str | None:
    explicit = row.get("history_id") or row.get("history_source_key")
    if explicit:
        return str(explicit)
    images = row.get("history_images") or row.get("history_frame_paths")
    if not isinstance(images, list) or not images:
        return None
    return hashlib.sha256("\n".join(str(value) for value in images).encode()).hexdigest()


def _scene(row: dict[str, Any]) -> str | None:
    value = row.get("scene_name") or row.get("scene_id")
    return str(value) if value is not None else None


def audit(rows: list[dict[str, Any]], *, require_realized: bool, require_success: bool) -> dict[str, Any]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    branch_ids: list[str] = []
    for index, row in enumerate(rows):
        group = _group_id(row)
        branch = _branch_id(row, index)
        branch_ids.append(branch)
        if not group:
            issues.append({"row": index, "field": "pair_identity", "reason": "twin_id/pair_id/group_id/source_key/video_id required"})
        groups[group].append((index, row))
        history = row.get("history_images") or row.get("history_frame_paths")
        future = row.get("future_images") or row.get("future_frame_paths")
        times = row.get("future_times_s")
        if not isinstance(history, list) or len(history) < 2:
            issues.append({"row": index, "field": "history_images", "reason": "at least two history frames required"})
        if not isinstance(future, list) or len(future) < 1:
            issues.append({"row": index, "field": "future_images", "reason": "generated future images required"})
        if not isinstance(times, list) or len(times) != len(future) or any(float(times[i]) >= float(times[i + 1]) for i in range(len(times) - 1)):
            issues.append({"row": index, "field": "future_times_s", "reason": "finite increasing future timestamps matching future images required"})
        if not any(row.get(key) is not None for key in ("action_condition", "executed_action", "trajectory", "action_trajectory", "candidates")):
            issues.append({"row": index, "field": "action_condition", "reason": "action condition or candidate bank required"})
        if require_realized and not any(row.get(key) is not None for key in ("realized_future_ego_state", "realized_future", "future_ego_state")):
            issues.append({"row": index, "field": "realized_future_ego_state", "reason": "required for realized-state CC/FCS"})
        if require_success and row.get("task_success") is None:
            issues.append({"row": index, "field": "task_success", "reason": "required for FCS"})
    duplicate_branches = sorted(branch for branch, count in Counter(branch_ids).items() if count > 1)
    if duplicate_branches:
        issues.append({"field": "branch_id", "reason": "duplicate branch ids", "values": duplicate_branches})
    group_report = []
    for group, members in sorted(groups.items()):
        histories = {_history_fingerprint(row) for _, row in members}
        actions = [str(row.get("condition_action_id") or row.get("action_id") or row.get("branch_id") or row.get("video_id") or index) for index, row in members]
        scenes = {_scene(row) for _, row in members}
        group_report.append({
            "group_id": group,
            "branches": len(members),
            "history_fingerprints": len(histories - {None}),
            "action_ids_unique": len(actions) == len(set(actions)),
            "scene_ids": sorted(scene for scene in scenes if scene is not None),
        })
        if len(members) < 2:
            issues.append({"group_id": group, "field": "branches", "reason": "counterfactual group needs at least two branches"})
        if len(histories - {None}) > 1:
            issues.append({"group_id": group, "field": "history_images", "reason": "branches do not share identical history"})
        if len(actions) != len(set(actions)):
            issues.append({"group_id": group, "field": "action_condition_id", "reason": "action conditions must be unique within group"})
    realized_available = bool(rows) and all(
        any(row.get(key) is not None for key in ("realized_future_ego_state", "realized_future", "future_ego_state"))
        for row in rows
    )
    success_available = bool(rows) and all(row.get("task_success") is not None for row in rows)
    image_signal_fields = {"history_images", "future_images", "future_times_s", "action_condition"}
    image_probe_ready = bool(rows) and not any(issue.get("field") in image_signal_fields for issue in issues)
    counterfactual_fields = {"branches", "history_images", "action_condition_id", "branch_id", "pair_identity"}
    action_response_ready = image_probe_ready and not any(issue.get("field") in counterfactual_fields for issue in issues)
    return {
        "protocol": "iac-wam-benchmark-manifest-audit-v1",
        "rows": len(rows),
        "groups": len(groups),
        "group_report": group_report,
        "duplicate_branch_ids": duplicate_branches,
        "issues": issues[:200],
        "issue_count": len(issues),
        "image_probe_ready": image_probe_ready,
        "action_response_ready": action_response_ready,
        "realized_state_ready": realized_available and not any(issue["field"] == "realized_future_ego_state" for issue in issues),
        "fcs_ready": realized_available and success_available and not any(issue["field"] in {"realized_future_ego_state", "task_success"} for issue in issues),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--other-split", type=Path)
    parser.add_argument("--require-realized-state", action="store_true")
    parser.add_argument("--require-task-success", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.manifest)
    result = audit(rows, require_realized=args.require_realized_state, require_success=args.require_task_success)
    if args.other_split:
        other_scenes = {_scene(row) for row in _read(args.other_split)}
        current_scenes = {_scene(row) for row in rows}
        overlap = sorted(scene for scene in current_scenes & other_scenes if scene is not None)
        result["scene_overlap_with_other_split"] = overlap
        if overlap:
            result["issues"].append({"field": "scene_split", "reason": "scene overlap between calibration and holdout", "values": overlap})
            result["issue_count"] += len(overlap)
            result["action_response_ready"] = False
    result["status"] = "ok" if result["issue_count"] == 0 else "invalid"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["issue_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
