#!/usr/bin/env python3
"""Analyze calibrated dense posterior ranks and independent-support consistency."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.protocol import read_jsonl
from iac_new.calibrate import posterior_diagnostics
from iac_new.scoring import posterior_from_energies
from iac_new.support import (
    acceptable_set_metrics,
    classify_trajectory_candidates,
    counterfactual_consistency,
    independent_support_mask,
)


_DENSE_RE = re.compile(r"dense_s(?P<speed>[mp0-9]+)_y(?P<lateral>[mp0-9]+)_curv(?P<curvature>[mp0-9]+)")


def _decode(value: str) -> float:
    return float(value.replace("p", ".").replace("m", "-"))


def factors(candidate: dict[str, Any]) -> dict[str, float] | None:
    if candidate.get("counterfactual") is not None:
        return {str(k): float(v) for k, v in candidate["counterfactual"].items()}
    match = _DENSE_RE.fullmatch(str(candidate.get("candidate_id", "")))
    if not match:
        return None
    return {
        "speed_factor": _decode(match.group("speed")),
        "lateral_offset_m": _decode(match.group("lateral")),
        "curvature_offset_1pm": _decode(match.group("curvature")),
    }


def _rank_summary(values: list[float], top1: list[bool], count: int) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median_rank": None, "mean_rank": None, "rank_std": None, "top1_rate": None, "stability": None}
    ranks = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "rank_std": float(np.std(ranks)),
        "top1_rate": float(np.mean(top1)),
        # 1 means identical rank; 0 means the full bank range is occupied.
        "stability": float(max(0.0, 1.0 - np.std(ranks) / max(count - 1, 1))),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    x = _rankdata(np.asarray(left, dtype=np.float64))
    y = _rankdata(np.asarray(right, dtype=np.float64))
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lateral-tolerance-m", type=float, default=0.50)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.10)
    parser.add_argument("--speed-relative-tolerance", type=float, default=0.20)
    parser.add_argument("--curvature-tolerance-1pm", type=float, default=0.06)
    parser.add_argument("--plausible-lateral-tolerance-m", type=float, default=0.75)
    parser.add_argument("--plausible-yaw-tolerance-rad", type=float, default=0.14)
    parser.add_argument("--plausible-speed-relative-tolerance", type=float, default=0.25)
    parser.add_argument("--plausible-curvature-tolerance-1pm", type=float, default=0.08)
    args = parser.parse_args()
    score_rows = read_jsonl(args.scores)
    manifest_rows = {str(row["sample_id"]): row for row in read_jsonl(args.manifest)}
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    axis_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    axis_top1: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    monotonicity: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rank_rows = []
    cc_rows = []
    set_rows = []
    label_counts: defaultdict[str, int] = defaultdict(int)
    sensitivity_rows: dict[str, list[float]] = {"strict_0.5x": [], "default_1.0x": [], "loose_1.5x": []}
    support_definition = None
    for score_row in score_rows:
        sample_id = str(score_row["sample_id"])
        record = manifest_rows[sample_id]
        score_items = list(score_row["candidate_scores"])
        ids = [str(item["candidate_id"]) for item in score_items]
        energies = np.asarray([float(item["energy"]) for item in score_items], dtype=np.float64)
        priors = np.asarray([float(score_row.get("candidate_priors", {}).get(cid, 1.0)) for cid in ids], dtype=np.float64)
        probabilities = posterior_from_energies(energies, temperature=temperature, priors=priors)
        order = np.argsort(energies)
        ranks = np.empty(len(ids), dtype=int)
        ranks[order] = np.arange(1, len(ids) + 1)
        gt_id = score_row.get("gt_candidate_id")
        gt_rank = int(ranks[ids.index(str(gt_id))]) if gt_id is not None and str(gt_id) in ids else None
        rank_rows.append({"sample_id": sample_id, "gt_rank": gt_rank, "candidate_count": len(ids), "temperature": temperature})
        if gt_rank is not None:
            for axis in ("speed_factor", "lateral_offset_m", "curvature_offset_1pm"):
                monotonicity[axis][sample_id].append((0.0, float(gt_rank)))
        candidates = list(record["candidates"])
        candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
        for index, candidate_id in enumerate(ids):
            candidate = candidate_by_id[candidate_id]
            f = factors(candidate)
            if f is None or index >= len(ranks):
                continue
            for axis, value in f.items():
                key = f"{float(value):+.6f}"
                axis_values[axis][key].append(float(ranks[index]))
                axis_top1[axis][key].append(bool(ranks[index] == 1))
            # Isolate one perturbation axis at a time. This avoids attributing
            # an interaction effect to speed, lateral offset, or curvature.
            deltas = {
                "speed_factor": abs(float(f["speed_factor"]) - 1.0),
                "lateral_offset_m": abs(float(f["lateral_offset_m"])),
                "curvature_offset_1pm": abs(float(f["curvature_offset_1pm"])),
            }
            for axis, magnitude in deltas.items():
                other = [value for name, value in deltas.items() if name != axis]
                if all(value < 1e-12 for value in other):
                    monotonicity[axis][sample_id].append((magnitude, float(ranks[index])))
        labels, support_meta = classify_trajectory_candidates(
            candidates,
            str(gt_id),
            np.asarray(record["future_times_s"], dtype=np.float64),
            lateral_tolerance_m=args.plausible_lateral_tolerance_m,
            yaw_tolerance_rad=args.plausible_yaw_tolerance_rad,
            speed_relative_tolerance=args.plausible_speed_relative_tolerance,
            curvature_tolerance_1pm=args.plausible_curvature_tolerance_1pm,
        ) if gt_id is not None else ([], {})
        labels_by_id = {str(item["candidate_id"]): item for item in labels}
        if labels:
            aligned_labels = [labels_by_id[candidate_id] for candidate_id in ids]
        else:
            aligned_labels = [{"candidate_id": candidate_id, "label": "unknown", "acceptable": False, "support_distance": np.inf} for candidate_id in ids]
        strict_support_mask, strict_support_meta = independent_support_mask(
            candidates,
            str(gt_id),
            np.asarray(record["future_times_s"], dtype=np.float64),
            lateral_tolerance_m=args.lateral_tolerance_m,
            yaw_tolerance_rad=args.yaw_tolerance_rad,
            speed_relative_tolerance=args.speed_relative_tolerance,
            curvature_tolerance_1pm=args.curvature_tolerance_1pm,
        ) if gt_id is not None else (np.zeros(len(ids), dtype=bool), {})
        strict_by_id = {
            str(candidate["candidate_id"]): bool(value)
            for candidate, value in zip(candidates, strict_support_mask)
        }
        support_mask = np.asarray([strict_by_id[candidate_id] for candidate_id in ids], dtype=bool)
        prediction_set = np.asarray(
            [candidate_id in set(str(value) for value in score_row.get("prediction_set_ids", [])) for candidate_id in ids],
            dtype=bool,
        )
        if support_definition is None and strict_support_meta:
            support_definition = {
                "definition": strict_support_meta["definition"],
                "source": "logged trajectory only; independent of image-flow energy and posterior",
                "tolerances": strict_support_meta["tolerances"],
            }
        cc = counterfactual_consistency(probabilities, support_mask)
        cc.update({"sample_id": sample_id, "gt_rank": gt_rank})
        cc_rows.append(cc)
        set_metrics = acceptable_set_metrics(
            probabilities,
            aligned_labels,
            prediction_set=prediction_set,
        )
        set_metrics["sample_id"] = sample_id
        set_metrics["gt_rank"] = gt_rank
        set_rows.append(set_metrics)
        for item in aligned_labels:
            label_counts[str(item["label"])] += 1
        if gt_id is not None:
            for name, scale in (("strict_0.5x", 0.5), ("default_1.0x", 1.0), ("loose_1.5x", 1.5)):
                scaled_mask, _ = independent_support_mask(
                    candidates,
                    str(gt_id),
                    np.asarray(record["future_times_s"], dtype=np.float64),
                    lateral_tolerance_m=args.lateral_tolerance_m * scale,
                    yaw_tolerance_rad=args.yaw_tolerance_rad * scale,
                    speed_relative_tolerance=args.speed_relative_tolerance * scale,
                    curvature_tolerance_1pm=args.curvature_tolerance_1pm * scale,
                )
                sensitivity_rows[name].append(float(probabilities[scaled_mask].sum()))
    stability = {}
    bank_count = max((row["candidate_count"] for row in rank_rows), default=0)
    for axis in sorted(axis_values):
        stability[axis] = {
            value: _rank_summary(axis_values[axis][value], axis_top1[axis][value], bank_count)
            for value in sorted(axis_values[axis])
        }
    monotonic_summary = {}
    for axis, by_sample in sorted(monotonicity.items()):
        correlations = []
        for pairs in by_sample.values():
            correlation = _spearman(
                [float(pair[0]) for pair in pairs],
                [float(pair[1]) for pair in pairs],
            )
            if correlation is not None:
                correlations.append(correlation)
        monotonic_summary[axis] = {
            "eligible_samples": len(correlations),
            "mean_spearman_abs_perturbation_vs_rank": float(np.mean(correlations)) if correlations else None,
            "positive_fraction": float(np.mean(np.asarray(correlations) > 0.0)) if correlations else None,
        }
    result = {
        "protocol": "iac-new-image-v1",
        "temperature": temperature,
        "rows": len(score_rows),
        "posterior_validation": {
            "uncalibrated": posterior_diagnostics(score_rows, 1.0),
            "calibrated": posterior_diagnostics(score_rows, temperature),
        },
        "rank_summary": {
            "median_gt_rank": float(np.median([r["gt_rank"] for r in rank_rows if r["gt_rank"] is not None])) if any(r["gt_rank"] is not None for r in rank_rows) else None,
            "top1_rate": float(np.mean([r["gt_rank"] == 1 for r in rank_rows if r["gt_rank"] is not None])) if any(r["gt_rank"] is not None for r in rank_rows) else None,
        },
        "perturbation_rank_stability": stability,
        "perturbation_monotonicity": monotonic_summary,
        "counterfactual_consistency": {
            "trajectory_support_definition": support_definition,
            "mean_support_mass": float(np.mean([r["support_mass"] for r in cc_rows])) if cc_rows else None,
            "median_support_mass": float(np.median([r["support_mass"] for r in cc_rows])) if cc_rows else None,
            "support_coverage_rate": float(np.mean([r["support_mass"] > 0.5 for r in cc_rows])) if cc_rows else None,
            "mean_outside_support_mass": float(np.mean([r["outside_support_mass"] for r in cc_rows])) if cc_rows else None,
            "tolerance_sensitivity": {
                name: {
                    "mean_support_mass": float(np.mean(values)) if values else None,
                    "median_support_mass": float(np.median(values)) if values else None,
                }
                for name, values in sensitivity_rows.items()
            },
            "rows": cc_rows,
        },
        "acceptable_set_evaluation": {
            "definition": (
                "known_valid=logged trajectory; plausible=independent kinematic tube; "
                "unknown=not certified from front view; known_invalid=explicit oracle label"
            ),
            "label_counts": dict(sorted(label_counts.items())),
            "mean_acceptable_mass": float(np.mean([r["acceptable_mass"] for r in set_rows])) if set_rows else None,
            "median_acceptable_mass": float(np.median([r["acceptable_mass"] for r in set_rows])) if set_rows else None,
            "top_acceptable_rate": float(np.mean([r["top_is_acceptable"] for r in set_rows])) if set_rows else None,
            "mean_known_invalid_mass": float(np.mean([r["known_invalid_mass"] for r in set_rows])) if set_rows else None,
            "mean_unknown_mass": float(np.mean([r["unknown_mass"] for r in set_rows])) if set_rows else None,
            "prediction_set_acceptable_coverage": float(np.mean([r["prediction_set_covers_acceptable"] for r in set_rows])) if set_rows else None,
            "mean_prediction_set_acceptable_fraction": float(np.mean([r["prediction_set_acceptable_fraction"] for r in set_rows if r["prediction_set_acceptable_fraction"] is not None])) if any(r["prediction_set_acceptable_fraction"] is not None for r in set_rows) else None,
            "mean_prediction_set_unknown_fraction": float(np.mean([r["prediction_set_unknown_fraction"] for r in set_rows])) if set_rows else None,
            "rows": set_rows,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
