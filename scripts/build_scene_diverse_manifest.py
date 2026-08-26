#!/usr/bin/env python3
"""Select temporally spread, scene-balanced rows from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-scene", type=int, default=5)
    args = parser.parse_args()
    if args.per_scene < 1:
        raise SystemExit("--per-scene must be positive")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[str(row.get("scene_name") or row.get("scene_id"))].append(row)
    selected: list[dict] = []
    for scene_name in sorted(grouped):
        rows = sorted(grouped[scene_name], key=lambda row: (int(row.get("frame_idx", 0)), str(row.get("source_key", ""))))
        count = min(args.per_scene, len(rows))
        indices = [round(index * (len(rows) - 1) / max(count - 1, 1)) for index in range(count)]
        selected.extend(rows[index] for index in indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({
        "source": str(args.manifest.resolve()),
        "output": str(args.output.resolve()),
        "per_scene": args.per_scene,
        "records": len(selected),
        "scenes": len(grouped),
    }, indent=2))


if __name__ == "__main__":
    main()
