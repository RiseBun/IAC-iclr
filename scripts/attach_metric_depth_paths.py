#!/usr/bin/env python3
"""Attach explicit metric-depth cache paths to an existing JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--depth-cache-dir", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            index = int(row["metadata"]["legacy_video_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{row.get('sample_id')}: metadata.legacy_video_index is required"
            ) from error
        cache_path = (args.depth_cache_dir / f"video_{index:03d}.npz").resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        row["metric_depth_path"] = str(cache_path)
        row["metric_depth_source"] = args.source
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
