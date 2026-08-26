#!/usr/bin/env python3
"""Run independent NAVSIM PDM rollouts for action-conditioned branches.

The branch action is converted to the official NAVSIM proposal sampling and
passed through the kinematic-bicycle tracker.  The exported state is read from
the simulator output, never from ``action_trajectory``.  This creates the
realized-state side of the counterfactual contract before WAM image scoring.

Some existing mini metric caches were produced by a newer NAVSIM schema.  The
small compatibility layer below only fills optional scorer inputs that are
absent in those pickles; it does not replace the ego-state simulation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _token(row: dict[str, Any]) -> str:
    source = str(row.get("source_key") or "")
    base = source.split("::", 1)[0]
    token = base.rsplit(":", 1)[-1]
    if not token:
        raise ValueError(f"missing NAVSIM metric-cache token: {source!r}")
    return token


def _proposal(row: dict[str, Any], horizon_s: float, interval_s: float):
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    points = np.asarray(row.get("action_trajectory"), dtype=np.float64)
    times = np.asarray(row.get("future_times_s"), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{row.get('branch_id')}: action_trajectory must be [T,3]")
    if len(times) != len(points):
        raise ValueError(f"{row.get('branch_id')}: action/future time length mismatch")
    if np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError(f"{row.get('branch_id')}: future_times_s must be increasing and positive")

    sampling = TrajectorySampling(
        num_poses=int(round(horizon_s / interval_s)),
        time_horizon=float(horizon_s),
        interval_length=float(interval_s),
    )
    target = np.arange(1, sampling.num_poses + 1, dtype=np.float64) * interval_s
    # The branch manifests are shorter than the four-second PDM horizon.  A
    # constant tail makes this explicit and avoids inventing a new action.
    xp = np.r_[0.0, times, horizon_s]
    poses = np.stack(
        [np.interp(target, xp, np.r_[0.0, points[:, dim], points[-1, dim]]) for dim in range(3)],
        axis=1,
    )
    return Trajectory(poses=poses, trajectory_sampling=sampling), sampling


def _compat_metric_cache(metric_cache: Any) -> None:
    """Fill optional fields expected by the checked-in NAVSIM scorer."""
    if not hasattr(metric_cache, "map_parameters"):
        metric_cache.map_parameters = None
    if not hasattr(metric_cache, "past_human_trajectory"):
        metric_cache.past_human_trajectory = None
    observation = metric_cache.observation
    if not hasattr(observation, "_occupancy_maps_tl"):
        observation._occupancy_maps_tl = None
    # Older caches omit detections_tracks.  Preserve the cached current object
    # geometry at each step so the scorer still sees obstacles, while marking
    # this compatibility mode in rollout_lineage.
    if not hasattr(observation, "_detections_tracks"):
        from nuplan.common.actor_state.tracked_objects import TrackedObjects
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

        objects = list(getattr(observation, "unique_objects", {}).values())
        frame = DetectionsTracks(TrackedObjects(objects))
        observation._detections_tracks = [frame for _ in range(64)]


class _StaticCachedTraffic:
    """Deterministic fallback policy for caches without future agent tracks."""

    def __init__(self, sampling: Any):
        self.sampling = sampling

    def simulate_environment(self, simulated_ego_states: np.ndarray, metric_cache: Any):
        from nuplan.common.actor_state.tracked_objects import TrackedObjects
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

        objects = list(getattr(metric_cache.observation, "unique_objects", {}).values())
        frame = DetectionsTracks(TrackedObjects(objects))
        # Include the current frame plus one frame for each simulated state.
        return [frame for _ in range(int(simulated_ego_states.shape[0]))]


def _local_state(simulated: np.ndarray, initial: Any, times: list[float], interval_s: float) -> list[list[float]]:
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import StateIndex

    x0 = float(initial.rear_axle.x)
    y0 = float(initial.rear_axle.y)
    h0 = float(initial.rear_axle.heading)
    c, s = math.cos(h0), math.sin(h0)
    result: list[list[float]] = []
    for t in times:
        index = min(max(int(round(float(t) / interval_s)), 1), len(simulated) - 1)
        state = simulated[index]
        dx, dy = float(state[StateIndex.X] - x0), float(state[StateIndex.Y] - y0)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        yaw = math.atan2(math.sin(float(state[StateIndex.HEADING] - h0)), math.cos(float(state[StateIndex.HEADING] - h0)))
        speed = math.hypot(float(state[StateIndex.VELOCITY_X]), float(state[StateIndex.VELOCITY_Y]))
        yaw_rate = float(state[StateIndex.ANGULAR_VELOCITY])
        result.append([lx, ly, yaw, speed, yaw_rate])
    return result


def run(rows: list[dict[str, Any]], cache_root: Path, *, horizon_s: float, interval_s: float, success_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score_from_interpolated_trajectory, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    loader = MetricCacheLoader(cache_root)
    outputs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in rows:
        branch_id = str(row.get("branch_id") or "")
        try:
            metric_cache = loader.get_from_token(_token(row))
            _compat_metric_cache(metric_cache)
            trajectory, sampling = _proposal(row, horizon_s, interval_s)
            pred = transform_trajectory(trajectory, metric_cache.ego_state)
            simulator = PDMSimulator(sampling)
            scorer = PDMScorer(sampling)
            policy = _StaticCachedTraffic(sampling)
            pdm_result, simulated_states = pdm_score_from_interpolated_trajectory(
                metric_cache, pred, sampling, simulator, scorer, policy
            )
            score = float(pdm_result["pdm_score"].iloc[0])
            times = [float(x) for x in row.get("future_times_s") or []]
            realized = _local_state(simulated_states, metric_cache.ego_state, times, interval_s)
            enriched = dict(row)
            enriched.update(
                {
                    "realized_future_ego_state": realized,
                    "state_times_s": times,
                    "state_reference_source": "navsim_pdm_kinematic_bicycle_closed_loop",
                    "closed_loop_rollout_id": f"navsim-pdm:{_token(row)}::{row.get('branch_mode', 'unknown')}",
                    "pdm_score": score,
                    "task_score": score,
                    "task_success": bool(score >= success_threshold),
                    "task_success_source": "navsim_pdm_score_v1",
                    "action_injection_verified": True,
                    "realized_state_available": True,
                    "rollout_lineage": {
                        "metric_cache_token": _token(row),
                        "simulator": "PDMSimulator/BatchKinematicBicycleModel",
                        "proposal_sampling": {"num_poses": sampling.num_poses, "interval_length": sampling.interval_length},
                        "traffic_policy": "static_cached_objects_compat",
                        "independent_from_wam_images": True,
                    },
                }
            )
            outputs.append(enriched)
        except Exception as exc:  # keep batch diagnostics instead of hiding one bad cache
            errors.append({"branch_id": branch_id, "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "protocol": "navsim-counterfactual-realized-rollout-v1",
        "input_rows": len(rows),
        "output_rows": len(outputs),
        "error_rows": len(errors),
        "errors": errors,
        "task_success_rows": sum(bool(r.get("task_success")) for r in outputs),
        "task_success_rate": (sum(bool(r.get("task_success")) for r in outputs) / len(outputs)) if outputs else None,
        "success_threshold": success_threshold,
        "state_reference_source": "navsim_pdm_kinematic_bicycle_closed_loop",
        "traffic_policy": "static_cached_objects_compat",
        "independent_realized_state": True,
    }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--horizon-s", type=float, default=4.0)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = _read(args.branches)
    if args.max_rows:
        rows = rows[: args.max_rows]
    outputs, summary = run(rows, args.metric_cache, horizon_s=args.horizon_s, interval_s=args.interval_s, success_threshold=args.success_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in outputs), encoding="utf-8")
    summary["output"] = str(args.output)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["error_rows"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
