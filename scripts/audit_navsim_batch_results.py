#!/usr/bin/env python3
"""Correctly summarize streaming NAVSIM results, including internal abstains."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path, required=True)
    parser.add_argument("--input-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _rows(args.results)
    previous = json.loads(args.previous_summary.read_text(encoding="utf-8"))
    valid = [row for row in rows if row.get("valid")]
    internal_abstain = [row for row in rows if not row.get("valid")]
    exceptions = list(previous.get("invalid_records") or [])
    ranks = []
    for row in valid:
        ranks.append(list(row["candidate_order"]).index("logged") + 1)
    internal_reasons = Counter(
        str(reason).split(":", 1)[-1]
        for row in internal_abstain
        for reason in row.get("abstain_reasons", [])
    )
    exception_reasons = Counter(
        "low_effective_pixels" if "effective pixels" in str(item.get("error", "")) else "other_exception"
        for item in exceptions
    )
    summary = {
        "protocol": "iac-new-image-v1-streaming-audited",
        "input_count": args.input_count,
        "returned_rows": len(rows),
        "valid_rows": len(valid),
        "internal_abstain_rows": len(internal_abstain),
        "exception_rows": len(exceptions),
        "total_abstain_rows": len(internal_abstain) + len(exceptions),
        "total_abstain_fraction": (len(internal_abstain) + len(exceptions)) / args.input_count,
        "top1_accuracy_valid": float(np.mean([row["top1_correct"] for row in valid])) if valid else None,
        "native_rank_mean_valid": float(np.mean(ranks)) if ranks else None,
        "native_rank_median_valid": float(np.median(ranks)) if ranks else None,
        "coverage_valid": float(np.mean([row["gt_in_prediction_set"] for row in valid])) if valid else None,
        "prediction_set_size_mean_valid": float(np.mean([row["prediction_set_size"] for row in valid])) if valid else None,
        "internal_abstain_reason_counts": dict(internal_reasons),
        "exception_reason_counts": dict(exception_reasons),
        "rank_histogram_valid": dict(Counter(str(rank) for rank in ranks)),
        "source_summary": str(args.previous_summary.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
