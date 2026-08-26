#!/usr/bin/env python3
"""Join native IAC scores with source ego-state trajectories for event labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = {str(row["source_key"]).split(":")[-1]: row for row in rows(args.source)}
    output = []
    missing = []
    for score in rows(args.scores):
        key = str(score["sample_id"]).split(":")[-1]
        row = source.get(key)
        if row is None:
            missing.append(score["sample_id"])
            continue
        merged = dict(row)
        merged["sample_id"] = score["sample_id"]
        output.append(merged)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in output) + "\n", encoding="utf-8")
    print(json.dumps({"scores": len(rows(args.scores)), "matched": len(output), "missing": len(missing), "output": str(args.output)}))


if __name__ == "__main__":
    main()
