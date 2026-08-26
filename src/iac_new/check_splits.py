"""Fail closed when calibration and test manifests overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .protocol import read_jsonl


def split_keys(rows: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    scenes = {str(row.get("scene_id") or row.get("scene_name") or "") for row in rows}
    samples = {str(row.get("sample_id") or "") for row in rows}
    frames = {
        str(frame)
        for row in rows
        for frame in (
            list(row.get("frame_paths") or [])
            + list(row.get("history_frame_paths") or [])
            + list(row.get("future_frame_paths") or [])
        )
    }
    scenes.discard("")
    samples.discard("")
    return scenes, samples, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    args = parser.parse_args()
    calibration = split_keys(read_jsonl(args.calibration))
    test = split_keys(read_jsonl(args.test))
    names = ("scene", "sample", "frame")
    overlaps = {
        name: sorted(left & right)
        for name, left, right in zip(names, calibration, test)
    }
    summary = {
        "calibration": str(args.calibration.resolve()),
        "test": str(args.test.resolve()),
        "overlap_counts": {name: len(values) for name, values in overlaps.items()},
        "examples": {name: values[:5] for name, values in overlaps.items()},
    }
    print(json.dumps(summary, indent=2))
    if any(overlaps.values()):
        raise SystemExit("calibration/test split overlap detected")


if __name__ == "__main__":
    main()
