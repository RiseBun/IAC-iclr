#!/usr/bin/env python3
"""Fail-closed left/right image response gate before CCFC.

DriveWAM's branch-invariant diagnostic was mean future-image L1 0.000831.
This gate only scores left-versus-right pairs that share a history.  It does
not decode trajectories and does not claim CCFC or FCS.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from analyze_wam_action_sensitivity import _action_distance, _image_distance
except ImportError:
    from scripts.analyze_wam_action_sensitivity import _action_distance, _image_distance


DRIVEWAM_FAIL_L1 = 0.000831
DEFAULT_MIN_L1 = 0.005


def _bootstrap_lower(values: np.ndarray, rng: np.random.Generator) -> float:
    if len(values) == 1:
        return float(values[0])
    samples = []
    for _ in range(1000):
        draw = rng.choice(values, size=len(values), replace=True)
        samples.append(float(np.mean(draw)))
    return float(np.quantile(samples, 0.025))


def score_gate(
    rows: list[dict[str, Any]],
    *,
    min_l1: float = DEFAULT_MIN_L1,
    image_size: tuple[int, int] = (256, 144),
    seed: int = 0,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    issues = []
    for index, row in enumerate(rows):
        group = str(row.get("counterfactual_group_id") or row.get("source_key") or "")
        mode = str(row.get("branch_mode") or "")
        if not group or mode not in {"logged", "left", "right"}:
            issues.append({"row": index, "reason": "need counterfactual_group_id and logged/left/right"})
            continue
        source = str(row.get("future_images_source") or "")
        if source != "wam_generated" and not source.endswith("_generated"):
            issues.append({"row": index, "reason": "future_images_source is not generated"})
            continue
        grouped[group][mode] = row
    pair_l1 = []
    pair_action = []
    details = []
    for group, branches in sorted(grouped.items()):
        if "left" not in branches or "right" not in branches:
            issues.append({"group": group, "reason": "missing left or right branch"})
            continue
        image_l1 = _image_distance(branches["left"]["future_images"], branches["right"]["future_images"], image_size)
        action = _action_distance(branches["left"], branches["right"])
        pair_l1.append(image_l1)
        pair_action.append(action)
        details.append({
            "counterfactual_group_id": group,
            "mean_future_image_l1": image_l1,
            "normalized_action_distance": action,
        })
    values = np.asarray(pair_l1, dtype=np.float64)
    lower = _bootstrap_lower(values, np.random.default_rng(seed)) if len(values) else None
    passed = bool(values.size) and lower is not None and lower > min_l1
    return {
        "protocol": "iac-action-response-gate-v1",
        "groups_with_left_right": len(details),
        "mean_left_right_image_l1": None if not values.size else float(values.mean()),
        "bootstrap_l1_lower_95": lower,
        "min_l1_threshold": min_l1,
        "drivewam_fail_reference_l1": DRIVEWAM_FAIL_L1,
        "mean_normalized_action_distance": None if not pair_action else float(np.mean(pair_action)),
        "passed": passed,
        "issues": issues,
        "pairs": details,
        "interpretation": (
            "pass: left/right generated futures are visually distinguishable above the DriveWAM fail reference"
            if passed
            else "fail: image response is too weak to enter CCFC; report wam_future_response_weak"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-l1", type=float, default=DEFAULT_MIN_L1)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = score_gate(rows, min_l1=args.min_l1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
