#!/usr/bin/env python3
"""Add a deterministic dense counterfactual trajectory bank to JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.counterfactual import densify_record


def _values(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-candidate-id")
    parser.add_argument("--speed-factors", default="0.85,1.0,1.15")
    parser.add_argument("--lateral-offsets-m", default="-0.75,-0.375,0,0.375,0.75")
    parser.add_argument("--curvature-offsets-1pm", default="-0.02,0,0.02")
    args = parser.parse_args()
    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(
                densify_record(
                    json.loads(line),
                    reference_candidate_id=args.reference_candidate_id,
                    speed_factors=_values(args.speed_factors),
                    lateral_offsets_m=_values(args.lateral_offsets_m),
                    curvature_offsets_1pm=_values(args.curvature_offsets_1pm),
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
