#!/usr/bin/env python3
"""Create a balanced input partition for v3 rows without paired branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs-manifest", type=Path, required=True)
    ap.add_argument("--reuse-manifest", type=Path, required=True)
    ap.add_argument("--v3-inputs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--selection", choices=("new", "reused"), default="new")
    args = ap.parse_args()

    reused = {str(row["source_key"]) for row in read(args.reuse_manifest)}
    all_rows = read(args.inputs_manifest)
    missing = [
        row for row in all_rows
        if (str(row["source_key"]) not in reused) == (args.selection == "new")
    ]
    if not missing:
        raise SystemExit("no missing rows")
    # Keep each shard at most one row apart so the runner's local index is
    # deterministic and easy to join back to this map.
    for shard in range(4):
        (args.output / "shards" / f"shard_{shard}").mkdir(parents=True, exist_ok=True)
    mapping = []
    for local_global, row in enumerate(missing):
        shard = min(local_global * 4 // len(missing), 3)
        local_index = sum(1 for item in mapping if item["shard"] == shard)
        source = args.v3_inputs / f"sample_{int(row['sample_index']):06d}.pkl"
        target = args.output / "shards" / f"shard_{shard}" / f"sample_{local_index:06d}.pkl"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
        mapping.append({
            "global_missing_index": local_global,
            "source_key": row["source_key"],
            "source_sample": str(target),
            "v3_index": row["sample_index"],
            "shard": shard,
            "local_index": local_index,
        })
    (args.output / "index_map.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mapping), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps({"rows": len(mapping), "shards": {str(s): sum(r["shard"] == s for r in mapping) for s in range(4)}}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(mapping), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
