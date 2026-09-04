#!/usr/bin/env python3
"""Partition v3 inputs into reusable v1 DriveWAM rows and new rows."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


def read_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"expected JSON array in {path}")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def action_8x3(value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (3, 8):
        array = array.T
    if array.shape != (8, 3) or not np.all(np.isfinite(array)):
        return None
    return array.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, action="append", required=True)
    parser.add_argument("--old-samples", type=Path, required=True)
    parser.add_argument("--v3-inputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    v3 = read_jsonl(args.v3)
    old_rows = []
    for manifest in args.old_manifest:
        old_rows.extend(read_jsonl(manifest))
    old_by_key = {}
    for row in old_rows:
        source = row.get("source_sample")
        try:
            with open(source, "rb") as handle:
                key = pickle.load(handle)["metadata"].get("source_key")
        except Exception:
            continue
        action = action_8x3(row.get("predicted_action_trajectory"))
        images = [str(x) for x in row.get("future_images") or []]
        if key and action is not None and len(images) >= 4 and all(Path(x).is_file() for x in images):
            old_by_key[key] = {"row": row, "action": action, "future_images": images}

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    reuse = []
    new_rows = []
    new_dir = out / "new_inputs"
    new_dir.mkdir(exist_ok=True)
    for row in v3:
        key = row["source_key"]
        previous = old_by_key.get(key)
        if previous is not None:
            old = previous["row"]
            reuse.append({
                "sample_id": row.get("sample_id", key),
                "benchmark_id": row.get("benchmark_id"),
                "source_key": key,
                "history_images": row.get("history_images", []),
                "future_images": previous["future_images"],
                "future_images_source": "wam_generated",
                "future_times_s": [1.0, 2.0, 3.0, 4.0][: len(previous["future_images"])],
                "action_trajectory": previous["action"],
                "action_trajectory_source": "drivewam_native_action_head",
                "wam_model_id": "drivewam_navsim_checkpoint_20260824",
                "model_revision": "drivewam_navsim_checkpoint_20260824",
                "seed": old.get("seed"),
                "lineage": {"reused_from": old.get("source_sample"), "source_key": key},
                "reuse": True,
            })
            continue
        index = len(new_rows)
        target = new_dir / f"sample_{index:06d}.pkl"
        source = args.v3_inputs / f"sample_{v3.index(row):06d}.pkl"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
        new_rows.append({"new_index": index, "source_key": key, "sample_id": row.get("sample_id", key), "source_sample": str(target), "v3_index": v3.index(row)})
    (out / "reuse_manifest.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in reuse), encoding="utf-8")
    (out / "new_index_map.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in new_rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"v3": len(v3), "reused": len(reuse), "new": len(new_rows)}, indent=2), encoding="utf-8")
    print(json.dumps({"v3": len(v3), "reused": len(reuse), "new": len(new_rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
