#!/usr/bin/env python3
"""Fail-closed audit for the independent 8-frame actor-motion manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.relative_motion import validate_actor_future_window


def audit(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[dict[str, Any]] = []
    chains: Counter[str] = Counter()
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        if row.get("protocol") != "actor-motion-reference-v1":
            errors.append({"row": index, "reason": "invalid_protocol"})
        if any(key in row for key in ("action_condition", "action_trajectory", "candidates", "future_action")):
            errors.append({"row": index, "reason": "action_leakage_field"})
        try:
            validate_actor_future_window(np.asarray(row.get("future_times_s"), dtype=np.float64))
        except (TypeError, ValueError) as error:
            errors.append({"row": index, "reason": "invalid_future_window", "detail": str(error)})
        if len(row.get("history_frame_paths") or []) != 4 or len(row.get("future_frame_paths") or []) != 8:
            errors.append({"row": index, "reason": "frame_count_mismatch"})
        tracks = row.get("actor_tracks") or []
        if not tracks:
            errors.append({"row": index, "reason": "missing_actor_tracks"})
        for track_index, track in enumerate(tracks):
            positions = np.asarray(track.get("positions_ego_m"), dtype=np.float64)
            visibility = np.asarray(track.get("visibility"), dtype=bool)
            if positions.shape != (8, 2) or visibility.shape != (8,):
                errors.append({"row": index, "track": track_index, "reason": "track_shape_mismatch"})
            if int(visibility.sum()) < 3:
                errors.append({"row": index, "track": track_index, "reason": "insufficient_visibility"})
        chains[str(row.get("chain_type"))] += 1
    return {
        "protocol": "actor-motion-reference-audit-v1",
        "path": str(path),
        "num_rows": len(rows),
        "counts_by_chain": dict(chains),
        "num_errors": len(errors),
        "formal_ready": not errors and bool(rows),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["formal_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
