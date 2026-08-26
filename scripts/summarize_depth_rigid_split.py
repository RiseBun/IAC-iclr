#!/usr/bin/env python3
"""Summarize legacy depth-rigid-flow output on an iac_new split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    ids = {
        json.loads(line)["sample_id"]
        for line in args.split_manifest.read_text(encoding="utf-8").splitlines()
        if line
    }
    rows = [
        row
        for row in (
            json.loads(line)
            for line in args.scores.read_text(encoding="utf-8").splitlines()
            if line
        )
        if row["video_pair_id"] in ids
    ]
    margins = [float(row["supported_margin_px"]) for row in rows]
    result = {
        "scores": str(args.scores.resolve()),
        "split": str(args.split_manifest.resolve()),
        "rows": len(rows),
        "accuracy": float(np.mean([row["correct"] for row in rows])) if rows else None,
        "median_supported_margin_px": float(np.median(margins)) if rows else None,
        "negative_margin_count": int(np.sum(np.asarray(margins) < 0.0)),
        "failures": [
            {
                "sample_id": row["video_pair_id"],
                "supported_candidate_index": row["supported_candidate_index"],
                "predicted_candidate_index": row["predicted_candidate_index"],
                "margin_px": row["supported_margin_px"],
            }
            for row in rows
            if not row["correct"]
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
