#!/usr/bin/env python3
"""Separate WAM future-image response failures from IAC recovery failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def diagnose(iac_summary: dict[str, Any], wam_summary: dict[str, Any], *, heading_threshold: float = 0.98, lateral_threshold_m: float = 0.50, response_correlation_threshold: float = 0.20, response_ratio_threshold: float = 0.20) -> dict[str, Any]:
    iac_heading = iac_summary.get("mean_heading_cosine")
    iac_lateral = iac_summary.get("mean_lateral_abs_m")
    if iac_heading is None:
        iac_heading = iac_summary.get("mean_realized_state_compatibility")
    iac_pass = bool(
        iac_heading is not None and iac_lateral is not None
        and float(iac_heading) >= float(heading_threshold)
        and float(iac_lateral) <= float(lateral_threshold_m)
    )
    response_corr = wam_summary.get("action_image_distance_correlation")
    response_ratio = wam_summary.get("mean_response_ratio")
    wam_pass = bool(
        response_corr is not None and response_ratio is not None
        and float(response_corr) >= float(response_correlation_threshold)
        and float(response_ratio) >= float(response_ratio_threshold)
    )
    if iac_pass and wam_pass:
        diagnosis = "both_gates_pass"
    elif iac_pass:
        diagnosis = "wam_future_response_weak"
    elif wam_pass:
        diagnosis = "iac_recovery_failure"
    else:
        diagnosis = "both_or_unresolved"
    return {
        "protocol": "wam-iac-two-gate-diagnosis-v1",
        "diagnosis": diagnosis,
        "iac_gate": {
            "pass": iac_pass,
            "mean_heading_cosine_or_compatibility": iac_heading,
            "mean_lateral_abs_m": iac_lateral,
            "heading_threshold": float(heading_threshold),
            "lateral_threshold_m": float(lateral_threshold_m),
        },
        "wam_future_response_gate": {
            "pass": wam_pass,
            "action_image_distance_correlation": response_corr,
            "mean_response_ratio": response_ratio,
            "correlation_threshold": float(response_correlation_threshold),
            "ratio_threshold": float(response_ratio_threshold),
        },
        "interpretation": {
            "iac_gate": "candidate-blind image-to-trajectory recovery against independent realized state or logged trajectory",
            "wam_gate": "action intervention changes the generated future image; this is not yet a causal CC score",
            "formal_benchmark_ready": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iac-summary", type=Path, required=True)
    parser.add_argument("--wam-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(_load(args.iac_summary), _load(args.wam_summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
