#!/usr/bin/env python3
"""Build reproducible NAVSIM rollout staging branches from the 78-row base set.

This file is a staging contract only: it contains an action condition and the
native source identity, but no WAM-generated future images.  Realized state is
filled later by the independent PDM rollout script and is never exposed to an
image-side scorer.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_key(row: dict[str, Any]) -> str:
    key = row.get("source_key") or (row.get("metadata") or {}).get("source_key")
    if not key:
        raise ValueError(f"{row.get('sample_id', '<unknown>')}: missing canonical source_key")
    return str(key)


def _trajectory(row: dict[str, Any], candidate_id: str) -> np.ndarray:
    candidates = {str(item.get("candidate_id")): item for item in row.get("candidates") or []}
    item = candidates.get(candidate_id) or candidates.get(str(row.get("gt_candidate_id")))
    if item is None or item.get("trajectory") is None:
        raise ValueError(f"{row.get('sample_id', '<unknown>')}: candidate trajectory not found")
    result = np.asarray(item["trajectory"], dtype=np.float64)
    if result.shape != (8, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{row.get('sample_id', '<unknown>')}: expected finite [8,3] trajectory")
    return result


def _cache_token(row: dict[str, Any], pkl_root: Path | None) -> tuple[str | None, str | None]:
    """Resolve the cache anchor without guessing from timestamp proximity."""
    if pkl_root is None:
        return None, None
    source = _source_key(row)
    log_name = source.split(":", 2)[1]
    anchor_token = source.rsplit(":", 1)[-1]
    path = pkl_root / f"{log_name}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"native NAVSIM pkl not found: {path}")
    payload = pickle.load(path.open("rb"))
    frames = payload if isinstance(payload, list) else payload.get("frames", [])
    anchor = next((item for item in frames if str(item.get("token")) == anchor_token), None)
    if anchor is None or not anchor.get("sample_prev"):
        raise ValueError(f"{source}: sample_prev mapping is unavailable")
    return str(anchor["sample_prev"]), "native_pkl.sample_prev"


def _cached_tokens(cache_root: Path | None, log_name: str) -> set[str] | None:
    if cache_root is None:
        return None
    metadata = cache_root / "metadata"
    files = sorted(metadata.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"metric cache metadata is missing under {metadata}")
    tokens: set[str] = set()
    for line in files[0].read_text(encoding="utf-8").splitlines()[1:]:
        path = line.strip()
        if path and f"/{log_name}/" in path:
            tokens.add(Path(path).parent.name)
    return tokens


def build_rows(
    base_rows: list[dict[str, Any]],
    *,
    lateral_offset_m: float,
    yaw_offset_rad: float,
    modes: tuple[str, ...],
    candidate_id: str,
    model_id: str,
    pkl_root: Path | None,
    metric_cache: Path | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in base_rows:
        source_key = _source_key(row)
        cache_token, cache_token_source = _cache_token(row, pkl_root)
        if metric_cache is not None:
            log_name = source_key.split(":", 2)[1]
            cached = _cached_tokens(metric_cache, log_name)
            if cache_token not in (cached or set()):
                continue
        times = np.asarray(row.get("future_times_s"), dtype=np.float64)
        if times.shape != (8,) or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"{source_key}: future_times_s must be increasing [8]")
        base = _trajectory(row, candidate_id)
        for mode in modes:
            if mode not in {"logged", "left", "right"}:
                raise ValueError(f"unsupported branch mode: {mode}")
            trajectory = base.copy()
            if mode != "logged":
                sign = 1.0 if mode == "left" else -1.0
                progress = np.linspace(0.0, 1.0, len(trajectory))
                trajectory[:, 1] += sign * float(lateral_offset_m) * progress
                trajectory[:, 2] += sign * float(yaw_offset_rad) * progress
            branch_id = f"{source_key}::branch={mode}"
            metadata = dict(row.get("metadata") or {})
            metadata.update({
                "protocol": "navsim-rollout-staging-v1",
                "source_key": source_key,
                "branch_id": branch_id,
                "branch_mode": mode,
                "wam_model_id": model_id,
                "cache_token": cache_token,
                "cache_token_source": cache_token_source,
                "future_images_source": "wam_generated_pending",
                "action_source": "staging_candidate",
                "realized_future_state_available_to_image_branch": False,
            })
            staged = dict(row)
            staged.update({
                "sample_id": f"{row.get('sample_id', source_key)}::staging::{mode}",
                "source_key": source_key,
                "counterfactual_group_id": source_key,
                "branch_id": branch_id,
                "branch_mode": mode,
                "wam_model_id": model_id,
                "cache_token": cache_token,
                "cache_token_source": cache_token_source,
                "action_trajectory": trajectory.tolist(),
                "action_trajectory_source": "staging_candidate",
                "future_images": [],
                "future_images_source": "wam_generated_pending",
                "wam_generation_status": "pending",
                "realized_future_ego_state": None,
                "realized_state_available": False,
                "task_success": None,
                "metadata": metadata,
            })
            output.append(staged)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", default="logged,left,right")
    parser.add_argument("--candidate-id", default="logged")
    parser.add_argument("--lateral-offset-m", type=float, default=0.75)
    parser.add_argument("--yaw-offset-rad", type=float, default=0.12)
    parser.add_argument("--model-id", default="wam_pending")
    parser.add_argument("--pkl-root", type=Path, help="Native NAVSIM pkl directory for exact sample_prev cache mapping")
    parser.add_argument("--metric-cache", type=Path, help="Keep only branches with an exact cache token")
    args = parser.parse_args()
    rows = build_rows(
        _read(args.base),
        lateral_offset_m=args.lateral_offset_m,
        yaw_offset_rad=args.yaw_offset_rad,
        modes=tuple(x.strip() for x in args.modes.split(",") if x.strip()),
        candidate_id=args.candidate_id,
        model_id=args.model_id,
        pkl_root=args.pkl_root,
        metric_cache=args.metric_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"protocol": "navsim-rollout-staging-v1", "base_rows": len(_read(args.base)), "branch_rows": len(rows), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
