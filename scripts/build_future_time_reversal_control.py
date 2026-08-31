#!/usr/bin/env python3
"""Create a temporal-order control by reversing only generated future frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        future = list(row.get("future_frame_paths") or [])
        if len(future) != 8:
            raise ValueError(f"{row.get('sample_id')}: expected 8 future frames")
        original_id = str(row["sample_id"])
        row["sample_id"] = f"{original_id}::time_reversed"
        row["control_of_sample_id"] = original_id
        row["future_frame_paths"] = list(reversed(future))
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "specificity_control": "future_time_reversal",
            "future_timestamps_unchanged": True,
            "candidate_blind_image_branch": True,
        })
        row["metadata"] = metadata
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(rows), "control": "future_time_reversal"}, indent=2))


if __name__ == "__main__":
    main()
