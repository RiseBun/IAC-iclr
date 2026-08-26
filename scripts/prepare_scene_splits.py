#!/usr/bin/env python3
"""Create deterministic, exactly sized scene-disjoint JSONL splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs=3, default=(30, 20, 50))
    parser.add_argument("--seed", default="iac-new-realvideo100-v1")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), []).append(row)
    if any(len(group) != 1 for group in by_scene.values()):
        raise ValueError("exact sample counts require one row per scene in this first experiment")
    if sum(args.sizes) != len(rows):
        raise ValueError("split sizes must sum to the number of rows")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['scene_id']}".encode("utf-8")
        ).hexdigest(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ("calibration", "validation", "test")
    start = 0
    summary = {"seed": args.seed, "source": str(args.manifest.resolve()), "splits": {}}
    for name, size in zip(names, args.sizes):
        selected = ordered[start : start + size]
        path = args.output_dir / f"{name}_{size}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected),
            encoding="utf-8",
        )
        summary["splits"][name] = {
            "path": str(path.resolve()),
            "rows": len(selected),
            "scenes": len({row["scene_id"] for row in selected}),
        }
        start += size
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
