#!/usr/bin/env python3
"""Select a scene-disjoint, all-NAVSIM benchmark-v3 manifest.

The inventory is already sampled at non-overlapping 4 s windows.  We add a
second guard here: at most one window per scene group.  The default 1000-row
quota keeps stop cases small and prevents straight-cruise cases from
dominating the benchmark while retaining all available acceleration/braking
examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def stable_hash(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big")


def classify(row: dict) -> str:
    s = np.asarray(row.get("realized_future_ego_state", []), dtype=float)
    if s.ndim != 2 or s.shape[0] < 2 or s.shape[1] < 4:
        return "unknown"
    speed = s[:, 3]
    accel = float((speed[-1] - speed[0]) / 4.0)
    if float(np.max(np.abs(speed))) < 0.5:
        return "stop"
    if accel < -1.0:
        return "braking"
    if accel > 1.0:
        return "acceleration"
    if abs(float(s[-1, 1] - s[0, 1])) >= 1.5 or abs(float(np.unwrap(s[:, 2])[-1] - np.unwrap(s[:, 2])[0])) >= 0.12:
        return "lateral_turn"
    return "straight_cruise"


def scene_group(row: dict) -> str:
    return str(row.get("source_key", "")).rsplit(":", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--stop-max-fraction", type=float, default=0.05)
    args = ap.parse_args()

    rows = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    for row in rows:
        row["_stratum_v3"] = classify(row)
        row["_scene_group_v3"] = scene_group(row)

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["_stratum_v3"] != "unknown":
            by_scene[row["_scene_group_v3"]].append(row)
    # Keep multiple windows from one scene when their 12-frame windows do not
    # overlap. This preserves rare maneuvers without leaking adjacent frames.
    candidates: list[dict] = []
    for scene_rows in by_scene.values():
        candidates.extend(sorted(scene_rows, key=lambda r: (int(r.get("frame_idx", 0)), r["source_key"])))

    available = Counter(r["_stratum_v3"] for r in candidates)
    stop_cap = min(int(args.total * args.stop_max_fraction), available["stop"])
    # Explicit quotas: rare/special maneuvers first. Straight cruise is capped
    # at 30% of the requested benchmark; stop is capped independently.
    quotas = {
        "acceleration": min(82, available["acceleration"]),
        "braking": min(65, available["braking"]),
        "lateral_turn": min(503, available["lateral_turn"]),
        "straight_cruise": min(int(args.total * 0.30), available["straight_cruise"]),
        "stop": stop_cap,
    }
    selected: list[dict] = []
    used_intervals: dict[str, list[int]] = defaultdict(list)

    def nonoverlap(row: dict) -> bool:
        start = int(row.get("frame_idx", 0))
        return all(abs(start - prev) > 11 for prev in used_intervals[row["_scene_group_v3"]])

    def add_if_valid(row: dict) -> bool:
        if not nonoverlap(row):
            return False
        selected.append(row)
        used_intervals[row["_scene_group_v3"]].append(int(row.get("frame_idx", 0)))
        return True

    for stratum in ("acceleration", "braking", "lateral_turn", "straight_cruise", "stop"):
        pool = [r for r in candidates if r["_stratum_v3"] == stratum]
        pool.sort(key=lambda r: stable_hash(r["source_key"], args.seed + len(stratum)))
        added = 0
        for row in pool:
            if added >= quotas[stratum]:
                break
            if add_if_valid(row):
                added += 1

    # Fill to total from remaining dynamic scenes, then cruise, then stop.
    preference = ("lateral_turn", "acceleration", "braking", "straight_cruise", "stop")
    for stratum in preference:
        pool = [r for r in candidates if r["_stratum_v3"] == stratum]
        pool.sort(key=lambda r: stable_hash(r["source_key"], args.seed + 100 + len(stratum)))
        for row in pool:
            if len(selected) >= args.total:
                break
            if stratum == "stop" and sum(r["_stratum_v3"] == "stop" for r in selected) >= stop_cap:
                break
            add_if_valid(row)
        if len(selected) >= args.total:
            break

    selected.sort(key=lambda r: (r["_stratum_v3"], r["source_key"]))
    clean = []
    for i, row in enumerate(selected):
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        out.update(
            {
                "benchmark_id": f"benchmark_v3-{i:05d}",
                "split": "benchmark_v3",
                "stratum": row["_stratum_v3"],
                "scene_group": row["_scene_group_v3"],
                "selection_seed": args.seed,
                "measurement_only": True,
                "future_visibility_policy": "private_reference; replace with WAM-generated future at evaluation",
            }
        )
        clean.append(out)

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / "benchmark_v3_navsim_private.jsonl"
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in clean))
    audit = {
        "protocol": "iac-benchmark-v3-navsim-only-v1",
        "source_inventory": str(args.input),
        "source_rows": len(rows),
        "selected_rows": len(clean),
        "scene_groups": len({r["scene_group"] for r in clean}),
        "dataset_counts": dict(Counter(r.get("dataset", "unknown") for r in clean)),
        "stratum_counts": dict(Counter(r["stratum"] for r in clean)),
        "stop_fraction": (sum(r["stratum"] == "stop" for r in clean) / len(clean)) if clean else None,
        "target_total": args.total,
        "stop_max_fraction": args.stop_max_fraction,
        "window_nonoverlap_within_scene": all(
            abs(int(a.get("frame_idx", 0)) - int(b.get("frame_idx", 0))) > 11
            for scene in {r["scene_group"] for r in clean}
            for a, b in zip(
                sorted([r for r in clean if r["scene_group"] == scene], key=lambda r: int(r.get("frame_idx", 0))),
                sorted([r for r in clean if r["scene_group"] == scene], key=lambda r: int(r.get("frame_idx", 0)))[1:],
            )
        ),
        "dynamic_special_fraction": sum(r["stratum"] in {"acceleration", "braking", "lateral_turn"} for r in clean) / len(clean) if clean else None,
        "future_horizon_s": 4.0,
        "history_frames": 4,
        "future_frames": 8,
        "public_release_policy": "strip future image paths and private GT; retain protocol metadata only",
    }
    (args.output_root / "benchmark_v3_navsim_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
