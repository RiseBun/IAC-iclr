#!/usr/bin/env python3
"""Fit a global depth divisor against a frozen metric-depth reference split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--candidate-depth-dir", type=Path, required=True)
    parser.add_argument("--reference-depth-dir", type=Path, required=True)
    parser.add_argument("--reference-divisor", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reference_divisor <= 0.0:
        raise ValueError("reference divisor must be positive")
    ratios = []
    sample_ratios = []
    for line in args.split_manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        video_index = int(row["metadata"]["legacy_video_index"])
        candidate = np.load(args.candidate_depth_dir / f"video_{video_index:03d}.npz")[
            "depth_m"
        ].astype(np.float32)
        reference = np.load(args.reference_depth_dir / f"video_{video_index:03d}.npz")[
            "depth_m"
        ].astype(np.float32)
        valid = (
            np.isfinite(candidate)
            & np.isfinite(reference)
            & (candidate > 1.0)
            & (candidate < 100.0)
            & (reference > 1.0)
            & (reference < 100.0)
        )
        if not valid.any():
            continue
        ratio = float(np.median(candidate[valid] / reference[valid]))
        ratios.append(ratio)
        sample_ratios.append({"sample_id": row["sample_id"], "ratio": ratio})
    if not ratios:
        raise ValueError("no valid calibration depth pairs")
    raw_ratio = float(np.median(ratios))
    result = {
        "method": "median_per_clip_ratio_to_frozen_metric_reference",
        "calibration_samples": len(ratios),
        "reference_divisor": float(args.reference_divisor),
        "candidate_to_reference_raw_ratio": raw_ratio,
        "candidate_divisor": float(args.reference_divisor * raw_ratio),
        "sample_ratio_iqr": [
            float(np.quantile(ratios, 0.25)),
            float(np.quantile(ratios, 0.75)),
        ],
        "samples": sample_ratios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
