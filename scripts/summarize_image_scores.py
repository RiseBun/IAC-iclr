#!/usr/bin/env python3
"""Report decision-relevant image-side score diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.scores.read_text(encoding="utf-8").splitlines() if line]
    if args.split_manifest:
        selected_ids = {
            json.loads(line)["sample_id"]
            for line in args.split_manifest.read_text(encoding="utf-8").splitlines()
            if line
        }
        rows = [row for row in rows if row["sample_id"] in selected_ids]
    manifest_by_id = {}
    if args.manifest:
        manifest_by_id = {
            row["sample_id"]: row
            for row in (
                json.loads(line)
                for line in args.manifest.read_text(encoding="utf-8").splitlines()
                if line
            )
        }
    margins = []
    failures = []
    by_control: dict[str, list[bool]] = {}
    by_gt_id: dict[str, list[bool]] = {}
    feature_rows = []
    for row in rows:
        gt_id = row.get("gt_candidate_id")
        energies = {item["candidate_id"]: float(item["energy"]) for item in row["candidate_scores"]}
        other = min(value for key, value in energies.items() if key != gt_id)
        margin = other - energies[gt_id]
        margins.append(margin)
        control = str(row.get("metadata", {}).get("control_type") or "unknown")
        by_control.setdefault(control, []).append(bool(row["top1_correct"]))
        by_gt_id.setdefault(str(gt_id), []).append(bool(row["top1_correct"]))
        manifest = manifest_by_id.get(row["sample_id"])
        if manifest is not None and len(manifest["candidates"]) == 2:
            trajectories = {
                candidate["candidate_id"]: np.asarray(candidate["trajectory"], dtype=np.float64)
                for candidate in manifest["candidates"]
            }
            gt_traj = trajectories[str(gt_id)]
            other_id = next(key for key in trajectories if key != str(gt_id))
            other_traj = trajectories[other_id]
            feature_rows.append(
                {
                    "correct": bool(row["top1_correct"]),
                    "abs_forward_delta_m": abs(float(other_traj[-1, 0] - gt_traj[-1, 0])),
                    "abs_lateral_delta_m": abs(float(other_traj[-1, 1] - gt_traj[-1, 1])),
                    "abs_yaw_delta_rad": abs(float(other_traj[-1, 2] - gt_traj[-1, 2])),
                    "gt_is_farther": bool(gt_traj[-1, 0] > other_traj[-1, 0]),
                }
            )
        if not row["top1_correct"] or not row["valid"]:
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "control_type": control,
                    "gt": gt_id,
                    "top": row["top_candidate_id"],
                    "energy_margin_other_minus_gt": margin,
                    "valid": row["valid"],
                    "abstain_reasons": row["abstain_reasons"],
                }
            )
    result = {
        "rows": len(rows),
        "top1_accuracy": float(np.mean([row["top1_correct"] for row in rows])) if rows else None,
        "valid_fraction": float(np.mean([row["valid"] for row in rows])) if rows else None,
        "positive_energy_margin_fraction": float(np.mean(np.asarray(margins) > 0.0)) if margins else None,
        "median_energy_margin_other_minus_gt": float(np.median(margins)) if margins else None,
        "by_control": {
            key: {"rows": len(values), "top1_accuracy": float(np.mean(values))}
            for key, values in sorted(by_control.items())
        },
        "by_gt_candidate_id": {
            key: {"rows": len(values), "top1_accuracy": float(np.mean(values))}
            for key, values in sorted(by_gt_id.items())
        },
        "failures": failures,
    }
    if feature_rows:
        result["kinematic_diagnostics"] = {
            "gt_farther": _binary_group(feature_rows, "gt_is_farther"),
            "forward_delta_bins_m": _numeric_bins(
                feature_rows, "abs_forward_delta_m", [0.0, 1.0, 2.0, 4.0, float("inf")]
            ),
            "lateral_delta_bins_m": _numeric_bins(
                feature_rows, "abs_lateral_delta_m", [0.0, 0.25, 0.5, 1.0, float("inf")]
            ),
            "yaw_delta_bins_rad": _numeric_bins(
                feature_rows, "abs_yaw_delta_rad", [0.0, 0.02, 0.05, 0.1, float("inf")]
            ),
        }
    print(json.dumps(result, indent=2))


def _binary_group(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for value in (False, True):
        selected = [row["correct"] for row in rows if row[key] is value]
        result[str(value).lower()] = {
            "rows": len(selected),
            "top1_accuracy": float(np.mean(selected)) if selected else None,
        }
    return result


def _numeric_bins(rows: list[dict], key: str, edges: list[float]) -> list[dict]:
    result = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = [
            row["correct"] for row in rows if lower <= float(row[key]) < upper
        ]
        result.append(
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper if np.isfinite(upper) else None,
                "rows": len(selected),
                "top1_accuracy": float(np.mean(selected)) if selected else None,
            }
        )
    return result


if __name__ == "__main__":
    main()
