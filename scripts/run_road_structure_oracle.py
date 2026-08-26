#!/usr/bin/env python3
"""Run a dependency-light synthetic oracle for road-structure evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.road_relative import compare_action_to_support, road_relative_posterior
from iac_new.road_structure import build_road_structure


def run(height: int = 96, width: int = 160, intervals: int = 4) -> dict:
    yy, xx = np.indices((height, width), dtype=np.float64)
    road = np.zeros((intervals, height, width), dtype=bool)
    for index in range(intervals):
        # A gently bending road whose center shifts right with image depth.
        center = width * 0.5 + 0.12 * (height - yy) + 0.0015 * (height - yy) ** 2
        half_width = 10.0 + 0.30 * (height - yy)
        road[index] = (xx >= center - half_width) & (xx <= center + half_width)
    foe = np.asarray([width * 0.56, height * 0.30])
    flow = 0.01 * np.stack([xx - foe[0], yy - foe[1]], axis=-1)
    flows = np.repeat(flow[None, ...], intervals, axis=0)
    static_weights = np.ones((intervals, height, width), dtype=np.float64)
    roi = np.ones((height, width), dtype=bool)
    evidence = build_road_structure(road, flows, static_weights, roi)
    times = np.linspace(0.8, 3.2, intervals)
    trajectory = np.stack([np.linspace(2.0, 8.0, intervals), np.linspace(0.0, 0.6, intervals), np.linspace(0.0, 0.08, intervals)], axis=1)
    posterior = road_relative_posterior(trajectory, times, observability=np.full(intervals, 0.9))
    support_score = compare_action_to_support(trajectory, posterior, times)
    return {"protocol": "road-structure-synthetic-oracle-v1", "evidence": evidence, "posterior": posterior, "self_support_score": support_score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(), indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
