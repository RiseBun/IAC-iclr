#!/usr/bin/env python3
"""Merge reused and newly generated DriveWAM rows for benchmark v3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_lines(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_8x3(value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (3, 8):
        array = array.T
    if array.shape != (8, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid action shape {array.shape}")
    return array.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--new-outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = {row["source_key"]: row for row in read_lines(args.v3)}
    reuse = {row["source_key"]: row for row in read_lines(args.partition / "reuse_manifest.jsonl")}
    new_map = read_lines(args.partition / "new_index_map.jsonl")
    counts = []
    generated = {}
    offset = 0
    for shard in range(4):
        rows = json.loads((args.new_outputs / f"shard_{shard}" / "manifest.json").read_text(encoding="utf-8"))
        counts.append(len(rows))
        for row in rows:
            local = int(row["sample_index"])
            global_new = offset + local
            if global_new >= len(new_map):
                raise ValueError(f"shard {shard} sample {local} exceeds new map")
            mapping = new_map[global_new]
            key = mapping["source_key"]
            if key in generated:
                raise ValueError(f"duplicate generated source_key {key}")
            if not row.get("future_images") or len(row["future_images"]) != 4:
                raise ValueError(f"{key}: expected four generated future frames")
            generated[key] = {
                "sample_id": mapping["sample_id"],
                "benchmark_id": base[key].get("benchmark_id"),
                "source_key": key,
                "history_images": base[key]["history_images"],
                "future_images": [str(x) for x in row["future_images"]],
                "future_images_source": "wam_generated",
                "future_times_s": [1.0, 2.0, 3.0, 4.0],
                "action_trajectory": action_8x3(row.get("predicted_action_trajectory")),
                "action_trajectory_source": "drivewam_native_action_head",
                "wam_model_id": "drivewam_navsim_checkpoint_20260824",
                "model_revision": "drivewam_navsim_checkpoint_20260824",
                "seed": local,
                "lineage": {"shard": shard, "local_index": local, "new_index": global_new},
                "reuse": False,
            }
        offset += len(rows)
    if counts != [187, 186, 186, 186]:
        raise ValueError(f"unexpected shard counts {counts}")
    if len(generated) != len(new_map):
        raise ValueError(f"new outputs {len(generated)} != map {len(new_map)}")

    merged = []
    missing = []
    for row in sorted(base.values(), key=lambda item: int(item["benchmark_id"].rsplit("-", 1)[-1])):
        key = row["source_key"]
        value = reuse.get(key) or generated.get(key)
        if value is None:
            missing.append(key)
            continue
        merged.append(value)
    if missing or len(merged) != len(base):
        raise ValueError(f"missing merged rows: {len(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in merged), encoding="utf-8")
    audit = {
        "protocol": "iac-benchmark-v3-drivewam-generated-v1",
        "rows": len(merged),
        "reused_rows": len(reuse),
        "new_rows": len(generated),
        "native_future_times_s": [1.0, 2.0, 3.0, 4.0],
        "future_images_source": "wam_generated",
        "ground_truth_in_submission": False,
        "shard_counts": counts,
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
