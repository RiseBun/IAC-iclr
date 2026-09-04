#!/usr/bin/env python3
"""Explain CFAC/CCFC failures and expose a longitudinal-scale-downweighted view.

This is an audit transform, not a replacement for the frozen benchmark score.
It reuses frozen Level-1 records and CCFC reports, preserving raw values while
adding shape-priority diagnostics.  The longitudinal weight is explicit and
recorded in the output so it cannot silently tune the benchmark.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SHAPE_FIELDS = ("lateral_speed_mps", "yaw_rate_radps", "curvature_1pm")
LONGITUDINAL_FIELDS = ("speed_mps", "acceleration_mps2")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _shape_longitudinal_view(record: dict[str, Any], longitudinal_weight: float) -> dict[str, Any]:
    comparison = record.get("comparison") or {}
    metrics = comparison.get("metrics") or {}
    tolerances = comparison.get("tolerances") or {}

    def normalized(fields: tuple[str, ...]) -> dict[str, float | None]:
        output: dict[str, float | None] = {}
        for field in fields:
            mae = (metrics.get(field) or {}).get("mae")
            tolerance = tolerances.get(field)
            output[field] = (
                None
                if mae is None or tolerance in (None, 0)
                else float(mae) / float(tolerance)
            )
        return output

    shape_norm = normalized(SHAPE_FIELDS)
    long_norm = normalized(LONGITUDINAL_FIELDS)
    shape_values = [float(value) for value in shape_norm.values() if value is not None]
    long_values = [float(value) for value in long_norm.values() if value is not None]
    shape_loss = _mean(shape_values)
    long_loss = _mean(long_values)
    weighted_loss = None
    if shape_loss is not None or long_loss is not None:
        shape_loss = 0.0 if shape_loss is None else shape_loss
        long_loss = 0.0 if long_loss is None else long_loss
        weighted_loss = (shape_loss + longitudinal_weight * long_loss) / (1.0 + longitudinal_weight)
    coverage = comparison.get("coverage")
    reasons: list[str] = []
    if comparison.get("status") != "ok":
        reasons.append("comparison_abstain_or_missing")
    elif coverage is not None and float(coverage) < 0.50:
        reasons.append("low_shape_coverage")
    shape_bad = [field for field, value in shape_norm.items() if value is not None and value > 1.0]
    long_bad = [field for field, value in long_norm.items() if value is not None and value > 1.0]
    if shape_bad:
        reasons.append("shape_residual_above_tolerance")
    if long_bad:
        reasons.append("longitudinal_scale_residual_above_tolerance")
    if not reasons:
        reasons.append("shape_within_tolerance")
    if shape_bad and long_bad:
        max_shape = max(shape_norm[field] for field in shape_bad)
        max_long = max(long_norm[field] for field in long_bad)
        dominant = "shape" if max_shape >= max_long else "longitudinal_scale"
    elif shape_bad:
        dominant = "shape"
    elif long_bad:
        dominant = "longitudinal_scale"
    elif "low_shape_coverage" in reasons:
        dominant = "coverage"
    else:
        dominant = "none"
    intervals = []
    for interval in comparison.get("per_interval") or []:
        errors = interval.get("absolute_errors") or {}
        interval_shape = {
            field: errors.get(field) for field in SHAPE_FIELDS if errors.get(field) is not None
        }
        interval_long = {
            field: errors.get(field) for field in LONGITUDINAL_FIELDS if errors.get(field) is not None
        }
        intervals.append({
            "time_s": interval.get("time_s"),
            "status": interval.get("status"),
            "shape_status": interval.get("shape_status"),
            "speed_status": interval.get("speed_status"),
            "observability": interval.get("observability"),
            "shape_absolute_errors": interval_shape,
            "longitudinal_absolute_errors": interval_long,
        })
    return {
        "sample_id": record.get("sample_id"),
        "status": comparison.get("status"),
        "coverage": coverage,
        "evaluable_intervals": comparison.get("evaluable_intervals"),
        "total_intervals": comparison.get("total_intervals"),
        "shape_normalized_mae": shape_norm,
        "longitudinal_normalized_mae": long_norm,
        "shape_score": None if shape_loss is None else float(np.exp(-shape_loss)),
        "longitudinal_score": None if long_loss is None else float(np.exp(-long_loss)),
        "shape_priority_score": None if weighted_loss is None else float(np.exp(-weighted_loss)),
        "longitudinal_weight": float(longitudinal_weight),
        "failure_reasons": reasons,
        "dominant_failure": dominant,
        "intervals": intervals,
    }


def _ccfc_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _load(path)
        for report in payload.get("reports") or []:
            if not isinstance(report, dict):
                continue
            cfc = report.get("continuous_cfc") or {}
            if not isinstance(cfc, dict):
                continue
            for mode, result in cfc.items():
                if not isinstance(result, dict):
                    continue
                subscores = result.get("subscores") or {}
                subscore_values = {
                    "direction": subscores.get("response_direction"),
                    "magnitude": subscores.get("response_magnitude"),
                    "temporal": subscores.get("response_temporal_alignment"),
                }
                present = {key: float(value) for key, value in subscore_values.items() if value is not None}
                lowest = min(present, key=present.get) if present else None
                metric_score = (cfc.get("metric") or {}).get("score")
                scale_free_score = (cfc.get("scale_free") or {}).get("score")
                arc_relative_score = (cfc.get("arc_relative") or {}).get("score")
                gap = (
                    None
                    if metric_score is None or scale_free_score is None
                    else float(scale_free_score) - float(metric_score)
                )
                reasons = []
                if result.get("status") != "ok":
                    reasons.append("ccfc_abstain_or_missing")
                if lowest is not None:
                    reasons.append(f"lowest_subscore_{lowest}")
                if gap is not None and gap >= 0.20:
                    reasons.append("metric_penalized_by_scale")
                if not reasons:
                    reasons.append("no_dominant_failure")
                rows.append({
                    "source_file": str(path),
                    "counterfactual_group_id": report.get("counterfactual_group_id"),
                    "scale_mode": mode,
                    "status": result.get("status"),
                    "coverage": result.get("coverage"),
                    "evaluable_intervals": result.get("evaluable_intervals"),
                    "total_intervals": result.get("total_intervals"),
                    "metric_score": metric_score,
                    "scale_free_score": scale_free_score,
                    "arc_relative_score": arc_relative_score,
                    "scale_free_minus_metric": gap,
                    "subscores": subscore_values,
                    "lowest_subscore": lowest,
                    "failure_reasons": reasons,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--ccfc", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--longitudinal-weight", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.0 < args.longitudinal_weight <= 1.0:
        raise SystemExit("--longitudinal-weight must be in (0, 1]")

    alignment = _load(args.alignment)
    level1_rows = [
        _shape_longitudinal_view(record, args.longitudinal_weight)
        for record in alignment.get("records") or []
    ]
    ccfc_paths: list[Path] = []
    for value in args.ccfc:
        matches = [Path(path) for path in glob.glob(value, recursive=True)]
        ccfc_paths.extend(matches if matches else [Path(value)])
    ccfc_paths = sorted({path for path in ccfc_paths if path.is_file()})
    ccfc_rows = _ccfc_rows(ccfc_paths)
    level1_summary = {
        "samples": len(level1_rows),
        "status_counts": dict(Counter(str(row.get("status")) for row in level1_rows)),
        "dominant_failure_counts": dict(Counter(str(row.get("dominant_failure")) for row in level1_rows)),
        "mean_coverage": _mean([float(row["coverage"]) for row in level1_rows if row.get("coverage") is not None]),
        "mean_shape_score": _mean([float(row["shape_score"]) for row in level1_rows if row.get("shape_score") is not None]),
        "mean_longitudinal_score": _mean([float(row["longitudinal_score"]) for row in level1_rows if row.get("longitudinal_score") is not None]),
        "mean_shape_priority_score": _mean([float(row["shape_priority_score"]) for row in level1_rows if row.get("shape_priority_score") is not None]),
    }
    ccfc_summary = {
        "rows": len(ccfc_rows),
        "mode_counts": dict(Counter(str(row.get("scale_mode")) for row in ccfc_rows)),
        "lowest_subscore_counts": dict(Counter(str(row.get("lowest_subscore")) for row in ccfc_rows)),
        "metric_penalized_by_scale": sum("metric_penalized_by_scale" in row.get("failure_reasons", []) for row in ccfc_rows),
        "mean_metric_score": _mean([float(row["metric_score"]) for row in ccfc_rows if row.get("metric_score") is not None]),
        "mean_scale_free_score": _mean([float(row["scale_free_score"]) for row in ccfc_rows if row.get("scale_free_score") is not None]),
        "mean_arc_relative_score": _mean([float(row["arc_relative_score"]) for row in ccfc_rows if row.get("arc_relative_score") is not None]),
    }
    payload = {
        "protocol": "cfac-ccfc-failure-diagnosis-and-shape-priority-audit-v1",
        "official_score_unchanged": True,
        "longitudinal_weight": float(args.longitudinal_weight),
        "source_alignment": str(args.alignment),
        "source_ccfc": [str(path) for path in ccfc_paths],
        "summary": {"level1": level1_summary, "ccfc": ccfc_summary},
        "level1_rows": level1_rows,
        "ccfc_rows": ccfc_rows,
        "interpretation": {
            "shape_priority_score": "exp(-(shape_normalized_mae + longitudinal_weight*longitudinal_normalized_mae)/(1+longitudinal_weight)); diagnostic only",
            "failure_reasons": "coverage, shape residual, and longitudinal scale residual are reported separately",
            "ccfc_scale_gap": "scale_free_score - metric_score; a large positive gap indicates metric scale penalty, not proof of model correctness",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
