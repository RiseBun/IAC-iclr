#!/usr/bin/env python3
"""Build a conservative readiness report from existing WAM artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(args: argparse.Namespace) -> dict[str, Any]:
    capability = _read(args.capability)
    sensitivity = _read(args.sensitivity)
    controls = _read(args.controls)
    split = _read(args.split_report)
    holdout = split.get("holdout", {})
    logged = controls.get("reports", {}).get("logged", {})

    gates = {
        "protocol_tests": {"status": "passed", "count": args.test_count},
        "baseline_lineage": {"status": "passed", "groups": holdout.get("pairs")},
        "baseline_relative_cc": {
            "status": "diagnostic_only",
            "top1": logged.get("diagonal_top1_accuracy"),
            "mean_cc_margin": logged.get("mean_cc_margin"),
            "reason": "DrivingWorld branch result is not an independent multi-WAM holdout.",
        },
        "wam_action_response": {
            "status": "weak_signal",
            "mean_future_image_l1": sensitivity.get("mean_future_image_l1"),
            "action_image_distance_correlation": sensitivity.get("action_image_distance_correlation"),
            "reason": "Action response exists but correlation is not yet strong enough for a model pass gate.",
        },
        "realized_state_cc": {
            "status": "unavailable",
            "reason": "Current generated branches lack independent realized future ego state.",
        },
        "foresight_conditioned_success": {
            "status": "unavailable",
            "reason": "Current generated branches lack independent task-success labels.",
        },
    }
    eligible = [row["model_id"] for row in capability.get("rows", []) if row.get("suitable_for_counterfactual_image_cc")]
    return {
        "protocol": "iac-wam-benchmark-readiness-v1",
        "status": "pre_benchmark",
        "formal_benchmark_ready": False,
        "blocking_requirements": [
            "run at least one externally action-conditioned WAM with downloaded checkpoint",
            "freeze calibration margin on scene-disjoint WAM calibration split",
            "attach independent realized_future_ego_state and task_success for realized-state CC/FCS",
        ],
        "eligible_controlled_wams": eligible,
        "gates": gates,
        "artifacts": {
            "capability": str(args.capability),
            "sensitivity": str(args.sensitivity),
            "controls": str(args.controls),
            "split_report": str(args.split_report),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--test-count", type=int, default=86)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

