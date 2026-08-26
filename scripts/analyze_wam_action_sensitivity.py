#!/usr/bin/env python3
"""Measure WAM branch response without using the IAC trajectory decoder.

This is a necessary diagnostic, not a semantic quality metric. For branches
sharing a history, it compares future-frame appearance changes with the
corresponding action-condition changes. A low image response despite a large
action intervention indicates that the WAM may be ignoring or clipping the
condition; it does not diagnose the IAC decoder.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image_distance(first: list[str], second: list[str], size: tuple[int, int]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("future image sequences must have equal nonzero length")
    values = []
    for first_path, second_path in zip(first, second):
        a = cv2.imread(first_path, cv2.IMREAD_COLOR)
        b = cv2.imread(second_path, cv2.IMREAD_COLOR)
        if a is None or b is None:
            raise FileNotFoundError(first_path if a is None else second_path)
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(b, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        values.append(float(np.mean(np.abs(a - b)) / 255.0))
    return float(np.mean(values))


def _action_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    a = np.asarray(first.get("action_trajectory", first.get("trajectory")), dtype=np.float64)
    b = np.asarray(second.get("action_trajectory", second.get("trajectory")), dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("action trajectories must have matching shape [T,3]")
    scale = np.asarray([5.0, 0.5, 0.08], dtype=np.float64)
    return float(np.sqrt(np.mean(((a - b) / scale) ** 2)))


def analyze(rows: list[dict[str, Any]], *, image_size: tuple[int, int] = (256, 144)) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row.get("counterfactual_group_id") or "")
        if not group:
            raise ValueError("every row needs counterfactual_group_id")
        grouped[group].append(row)
    pairs = []
    for group_id, branches in sorted(grouped.items()):
        if len(branches) < 2:
            continue
        for i, first in enumerate(branches):
            for j in range(i + 1, len(branches)):
                second = branches[j]
                image_distance = _image_distance(first["future_images"], second["future_images"], image_size)
                action_distance = _action_distance(first, second)
                pairs.append({
                    "counterfactual_group_id": group_id,
                    "first_branch_id": first.get("branch_id"),
                    "second_branch_id": second.get("branch_id"),
                    "first_branch_mode": first.get("branch_mode"),
                    "second_branch_mode": second.get("branch_mode"),
                    "normalized_action_distance": action_distance,
                    "mean_future_image_l1": image_distance,
                    "response_ratio": image_distance / max(action_distance, 1e-8),
                })
    if not pairs:
        raise ValueError("no same-history branch pairs found")
    actions = np.asarray([p["normalized_action_distance"] for p in pairs], dtype=np.float64)
    images = np.asarray([p["mean_future_image_l1"] for p in pairs], dtype=np.float64)
    correlation = float(np.corrcoef(actions, images)[0, 1]) if len(pairs) > 1 and np.std(actions) > 0 and np.std(images) > 0 else None
    return {
        "protocol": "wam-action-sensitivity-image-only-v1",
        "groups": len(grouped),
        "pairs": len(pairs),
        "mean_normalized_action_distance": float(actions.mean()),
        "mean_future_image_l1": float(images.mean()),
        "median_future_image_l1": float(np.median(images)),
        "mean_response_ratio": float(np.mean([p["response_ratio"] for p in pairs])),
        "action_image_distance_correlation": correlation,
        "interpretation": "low image distance for large action distance is evidence that the WAM condition may be ignored or clipped; this report does not use IAC trajectory decoding",
        "pairs_detail": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    args = parser.parse_args()
    report = analyze(read_jsonl(args.manifest), image_size=(args.width, args.height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs_detail"}, indent=2))


if __name__ == "__main__":
    main()
