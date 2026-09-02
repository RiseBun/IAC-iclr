#!/usr/bin/env python3
"""Build a scene-disjoint, stratified Level-1 benchmark manifest.

The input manifests contain native realized future frames.  This script only
freezes the *measurement* benchmark; it does not make a causal claim about a
WAM.  Future images and logged states must remain on the private evaluation
side when the benchmark is distributed to WAM authors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STRATA = ("stop", "braking", "acceleration", "lateral_turn", "straight_cruise")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def scene_group(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset", "unknown"))
    if dataset.startswith("navsim"):
        # scene_name is the physical scene; log_name prevents accidental joins
        # when a source exporter uses a reused scene number.
        return f"navsim:{row.get('log_name', '')}:{row.get('scene_name', row.get('scene_token', ''))}"
    return f"waymo:{row.get('segment_id', row.get('scene_id', row.get('scene_name', '')))}"


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("source_key") or row.get("token"))


def classify_stratum(row: dict[str, Any]) -> str:
    states = np.asarray(row.get("realized_future_ego_state", []), dtype=np.float64)
    times = np.asarray(row.get("future_times_s", []), dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 3:
        return "unknown"
    x = states[:, 0]
    y = states[:, 1]
    yaw = np.unwrap(states[:, 2])
    dt = max(float(times[-1] - times[0]), 1e-6) if times.size >= states.shape[0] else 4.0
    speed = states[:, 3] if states.shape[1] >= 4 else np.gradient(x, max(dt / len(x), 1e-3))
    acceleration = float((speed[-1] - speed[0]) / dt)
    if float(np.max(np.abs(speed))) < 0.5:
        return "stop"
    if acceleration < -1.0:
        return "braking"
    if acceleration > 1.0:
        return "acceleration"
    if abs(float(y[-1] - y[0])) >= 1.5 or abs(float(yaw[-1] - yaw[0])) >= 0.12:
        return "lateral_turn"
    return "straight_cruise"


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if len(row.get("history_images") or []) != 4:
        errors.append("history_frames")
    if len(row.get("future_images") or []) != 8:
        errors.append("future_frames")
    if len(row.get("future_times_s") or []) != 8:
        errors.append("future_timestamps")
    future_times = row.get("future_times_s") or []
    if future_times and (float(future_times[-1]) < 3.95 or float(future_times[-1]) > 4.10):
        errors.append("horizon")
    if any(float(b) <= float(a) for a, b in zip(future_times, future_times[1:])):
        errors.append("timestamp_order")
    if len(row.get("realized_future_ego_state") or []) != 8:
        errors.append("future_state")
    calibration = row.get("camera_to_ego") or row.get("camera_calibration")
    if calibration is None:
        errors.append("calibration")
    return errors


def interval(row: dict[str, Any]) -> tuple[float, float]:
    if "anchor_timestamp_micros" in row:
        anchor = float(row["anchor_timestamp_micros"]) / 1e6
    else:
        anchor = float(row.get("timestamp_us", 0)) / 1e6
    return anchor - 1.5, anchor + 4.0


def add_metadata(row: dict[str, Any], *, split: str, seed: int, ordinal: int) -> dict[str, Any]:
    out = dict(row)
    out["scene_group"] = scene_group(row)
    out["stratum"] = classify_stratum(row)
    out["split"] = split
    out["benchmark_id"] = f"{split}-{ordinal:05d}"
    out["selection_seed"] = seed
    out["measurement_only"] = True
    out["future_visibility_policy"] = "private_reference; replace with WAM-generated future at evaluation"
    return out


def select_stratified(rows: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[classify_stratum(row)].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: stable_hash(row_id(r), seed))
    # Round-robin gives every available motion mode representation without
    # inventing rare braking/acceleration examples.
    selected: list[dict[str, Any]] = []
    while len(selected) < target:
        progressed = False
        for stratum in STRATA:
            bucket = buckets.get(stratum, [])
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break
    return selected


def choose_scene_disjoint(rows: list[dict[str, Any]], *, split: str, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[scene_group(row)].append(row)
    # NAVSIM has no frozen role, so make a deterministic 80/20 scene split.
    # Waymo already carries the audited role from the 200-row freeze.
    out: list[dict[str, Any]] = []
    for group, group_rows in groups.items():
        if str(group).startswith("navsim:"):
            is_dev = stable_hash(group, seed) % 5 == 0
            if (split == "dev_v1") == is_dev:
                out.extend(group_rows)
        else:
            # The frozen Waymo 200 contains eval and development windows from
            # the same segments.  Assign the *segment* to one split first,
            # then retain only the corresponding audited role below.  This is
            # what prevents a hidden scene leak across dev and benchmark.
            is_dev = stable_hash(group, seed) % 5 == 0
            if (split == "dev_v1") == is_dev:
                out.extend(group_rows)
    return out


def audit(rows: list[dict[str, Any]], rejected: Counter[str], source_counts: Counter[str], seed: int) -> dict[str, Any]:
    groups = [scene_group(r) for r in rows]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[scene_group(r)].append(r)
    overlap_pairs = 0
    for group_rows in by_group.values():
        ordered = sorted(group_rows, key=lambda r: interval(r)[0])
        for prev, cur in zip(ordered, ordered[1:]):
            if interval(cur)[0] <= interval(prev)[1]:
                overlap_pairs += 1
    datasets = Counter(str(r.get("dataset", "unknown")) for r in rows)
    strata = Counter(classify_stratum(r) for r in rows)
    return {
        "protocol": "iac-level1-benchmark-v1",
        "selection_seed": seed,
        "records": len(rows),
        "scene_groups": len(set(groups)),
        "dataset_counts": dict(sorted(datasets.items())),
        "stratum_counts": dict(sorted(strata.items())),
        "overlap_pairs_within_scene": overlap_pairs,
        "duplicate_ids": len(rows) - len({row_id(r) for r in rows}),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "source_candidate_counts": dict(sorted(source_counts.items())),
        "required_shape": {"history_frames": 4, "future_frames": 8, "future_horizon_s": 4.0},
        "causal_claim_allowed": False,
        "interpretation": "native future is a measurement reference; causal WAM scoring replaces it with generated future images",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim", type=Path, required=True)
    parser.add_argument("--waymo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-per-dataset", type=int, default=300)
    parser.add_argument("--dev-per-dataset", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_hashes = {str(path): sha256_file(path) for path in (args.navsim, args.waymo)}
    for path in (args.navsim, args.waymo):
        for row in read_jsonl(path):
            dataset = "navsim" if path == args.navsim else "waymo"
            source_counts[dataset] += 1
            errors = validate_row(row)
            if errors:
                rejected.update(errors)
                continue
            all_rows.append(row)

    # Remove exact duplicate IDs before any split or selection.
    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(all_rows, key=lambda r: (scene_group(r), interval(r)[0], row_id(r))):
        unique.setdefault(row_id(row), row)
    all_rows = list(unique.values())

    outputs: dict[str, list[dict[str, Any]]] = {}
    for split, target in (("dev_v1", args.dev_per_dataset), ("benchmark_v1", args.benchmark_per_dataset)):
        selected: list[dict[str, Any]] = []
        for dataset_name in ("navsim", "waymo"):
            candidates = [r for r in choose_scene_disjoint(all_rows, split=split, seed=args.seed)
                          if ("navsim" if str(r.get("dataset", "")).startswith("navsim") else "waymo") == dataset_name]
            if dataset_name == "waymo":
                wanted_role = "evaluation_nonoverlap" if split == "benchmark_v1" else "development_overlap"
                # Frozen v1 rows carry an audited role.  Newly materialized
                # Perception-v2 rows do not, so their segment-level hash split
                # above is the role and they remain eligible here.
                candidates = [r for r in candidates
                              if not r.get("metadata", {}).get("manifest_role")
                              or str(r.get("metadata", {}).get("manifest_role")) == wanted_role]
            chosen = select_stratified(candidates, target, args.seed + stable_hash(dataset_name, args.seed) % 10000)
            selected.extend(chosen)
        selected.sort(key=lambda r: (str(r.get("dataset")), scene_group(r), interval(r)[0], row_id(r)))
        outputs[split] = [add_metadata(r, split=split, seed=args.seed, ordinal=i) for i, r in enumerate(selected)]

    # Explicitly verify scene disjointness between dev and benchmark.
    dev_groups = {scene_group(r) for r in outputs["dev_v1"]}
    bench_groups = {scene_group(r) for r in outputs["benchmark_v1"]}
    if dev_groups & bench_groups:
        raise SystemExit(f"scene leakage between splits: {sorted(dev_groups & bench_groups)[:5]}")

    for split, rows in outputs.items():
        path = args.output_root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
        split_rejected = Counter(rejected)
        report = audit(rows, split_rejected, source_counts, args.seed)
        report.update({"split": split, "path": str(path.resolve()), "source_manifests": [str(args.navsim), str(args.waymo)], "source_manifest_sha256": source_hashes})
        (args.output_root / f"{split}.audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    combined = outputs["dev_v1"] + outputs["benchmark_v1"]
    combined_path = args.output_root / "all_v1.jsonl"
    combined_path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in combined), encoding="utf-8")
    summary = audit(combined, rejected, source_counts, args.seed)
    summary.update({"splits": {k: len(v) for k, v in outputs.items()}, "path": str(combined_path.resolve()), "source_manifest_sha256": source_hashes})
    (args.output_root / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
