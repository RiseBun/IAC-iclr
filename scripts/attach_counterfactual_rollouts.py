#!/usr/bin/env python3
"""Attach independently executed counterfactual rollouts to WAM branches.

This is the boundary between action-conditioned image generation and the
closed-loop evaluator.  A branch is accepted only when an external simulator
has produced its future ego state and an independent task score.  In
particular, ``action_condition`` and ``action_trajectory`` are never accepted
as a substitute for ``realized_future_ego_state``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _state_ok(value: Any, *, field: str, branch_id: str, expected: int | None) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{branch_id}: {field} must be a non-empty [T,5] list")
    if expected is not None and len(value) != expected:
        raise ValueError(f"{branch_id}: {field} has {len(value)} points, expected {expected}")
    if any(not isinstance(point, list) or len(point) != 5 for point in value):
        raise ValueError(f"{branch_id}: {field} must have shape [T,5]")


def _independent_source(row: dict[str, Any], branch_id: str) -> str:
    source = str(row.get("state_reference_source") or row.get("realized_state_source") or "")
    lowered = source.lower()
    forbidden = ("action_condition", "action_trajectory", "planned", "proxy", "generated")
    if not source or any(token in lowered for token in forbidden):
        raise ValueError(
            f"{branch_id}: state_reference_source must identify an external closed-loop rollout, got {source!r}"
        )
    return source


def attach(branches: list[dict[str, Any]], rollouts: list[dict[str, Any]], *, require_success: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, rollout in enumerate(rollouts):
        branch_id = str(rollout.get("branch_id") or "")
        if not branch_id:
            raise ValueError(f"rollout row {index}: missing branch_id")
        if branch_id in by_id:
            raise ValueError(f"duplicate rollout branch_id: {branch_id}")
        by_id[branch_id] = rollout

    output: list[dict[str, Any]] = []
    missing: list[str] = []
    source_counts: Counter[str] = Counter()
    for index, branch in enumerate(branches):
        branch_id = str(branch.get("branch_id") or "")
        if not branch_id:
            raise ValueError(f"branch row {index}: missing branch_id")
        rollout = by_id.get(branch_id)
        if rollout is None:
            missing.append(branch_id)
            continue
        realized = rollout.get("realized_future_ego_state")
        _state_ok(realized, field="realized_future_ego_state", branch_id=branch_id, expected=len(branch.get("future_times_s") or []))
        if rollout.get("task_success") is None:
            if require_success:
                raise ValueError(f"{branch_id}: missing task_success")
        source = _independent_source(rollout, branch_id)
        score = rollout.get("task_score", rollout.get("pdm_score"))
        if score is None and require_success:
            raise ValueError(f"{branch_id}: missing task_score/pdm_score")
        enriched = dict(branch)
        enriched.update({
            "realized_future_ego_state": realized,
            "state_times_s": rollout.get("state_times_s", branch.get("future_times_s")),
            "task_score": score,
            "pdm_score": rollout.get("pdm_score", score),
            "task_success": rollout.get("task_success"),
            "task_success_source": rollout.get("task_success_source") or "external_simulator_metric",
            "state_reference_source": source,
            "closed_loop_rollout_id": rollout.get("closed_loop_rollout_id") or branch_id,
            "action_injection_verified": bool(rollout.get("action_injection_verified", False)),
            "rollout_lineage": rollout.get("rollout_lineage") or {},
            "realized_state_available": True,
        })
        if not enriched["action_injection_verified"]:
            raise ValueError(f"{branch_id}: action_injection_verified must be true for counterfactual evaluation")
        source_counts[source] += 1
        output.append(enriched)

    extras = sorted(set(by_id) - {str(row.get("branch_id") or "") for row in branches})
    summary = {
        "protocol": "wam-counterfactual-realized-rollout-v1",
        "branch_rows": len(branches),
        "rollout_rows": len(rollouts),
        "attached_rows": len(output),
        "missing_branch_ids": missing,
        "extra_rollout_branch_ids": extras,
        "task_success_rows": sum(row.get("task_success") is not None for row in output),
        "verified_action_rows": sum(bool(row.get("action_injection_verified")) for row in output),
        "state_reference_sources": dict(source_counts),
        "closed_loop_ready": not missing and not extras and len(output) == len(branches),
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-success", action="store_true")
    args = parser.parse_args()
    rows, summary = attach(
        _read(args.branches),
        _read(args.rollouts),
        require_success=not args.allow_missing_success,
    )
    if not summary["closed_loop_ready"]:
        raise SystemExit(json.dumps(summary, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
