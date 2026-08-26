#!/usr/bin/env python3
"""Build a deterministic, scene-aware balanced NAVSIM maneuver manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from iac_new.maneuver import extract_maneuver


def read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=26)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    buckets = defaultdict(list)
    for row in read(args.input):
        trajectory = np.asarray(row["realized_future_ego_state"], dtype=np.float64)[:, :3]
        times = np.asarray(row["future_times_s"], dtype=np.float64)
        events = extract_maneuver(trajectory, times)["lateral_action"]
        turns = [event for event in events if event != "straight"]
        label = turns[0] if turns else "straight"
        item = dict(row)
        item["event_label"] = label
        item["event_protocol"] = "heading-change-v1-frozen-0.028rad"
        buckets[label].append(item)
    rng = np.random.default_rng(args.seed)
    selected = []
    for label in ("straight", "left", "right"):
        candidates = buckets[label]
        if len(candidates) < args.per_class:
            raise ValueError(f"{label} has only {len(candidates)} samples")
        # Deterministic shuffle, then round-robin across scenes to avoid one
        # long log dominating a class.
        order = rng.permutation(len(candidates))
        by_scene = defaultdict(list)
        for index in order:
            by_scene[candidates[index]["scene_name"]].append(candidates[index])
        chosen = []
        while len(chosen) < args.per_class:
            progressed = False
            for scene in sorted(by_scene):
                if by_scene[scene]:
                    chosen.append(by_scene[scene].pop())
                    progressed = True
                    if len(chosen) >= args.per_class:
                        break
            if not progressed:
                break
        selected.extend(chosen)
    selected.sort(key=lambda row: (row["event_label"], row["scene_name"], row["frame_idx"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in selected) + "\n", encoding="utf-8")
    print(json.dumps({"input": len(read(args.input)), "selected": len(selected), "counts": {label: sum(row["event_label"] == label for row in selected) for label in ("straight", "left", "right")}, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
