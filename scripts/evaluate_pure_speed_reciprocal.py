#!/usr/bin/env python3
"""Evaluate a candidate-blind pure-speed reciprocal real-video pilot.

The manifest owns the fast/slow condition labels and control type. Scores must
contain only candidate-blind image measurements. All primary statistics are
computed at twin level so the two rows of a reciprocal pair cannot inflate the
sample count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> str:
    return str(row.get("video_id") or row.get("branch_id") or row.get("sample_id") or "")


def _condition(row: dict[str, Any]) -> str:
    value = str(row.get("condition") or row.get("speed_condition") or row.get("label") or "").lower()
    if value in {"fast", "slow"}:
        return value
    raise ValueError(f"{_key(row)}: manifest condition must be fast or slow")


def _control(row: dict[str, Any]) -> str:
    return str(row.get("control_type") or row.get("control") or "clean").lower()


def _progress_scalar(score: dict[str, Any], manifest: dict[str, Any]) -> float:
    """Return one candidate-blind scalar: average forward progress rate."""
    times = np.asarray(manifest.get("future_times_s") or manifest.get("frame_times_s") or [], dtype=np.float64)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or times[-1] <= 0:
        raise ValueError(f"{_key(manifest)}: finite positive future/frame times are required")
    curve = score.get("predicted_progress_curve") or score.get("predicted_progress")
    if curve is not None:
        values = np.asarray(curve, dtype=np.float64)
        if values.shape != times.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{_key(score)}: predicted progress curve must match time axis")
        return float(values[-1] / times[-1])
    trajectory = score.get("predicted_trajectory") or score.get("trajectory")
    if trajectory is not None:
        values = np.asarray(trajectory, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(times) or values.shape[1] < 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{_key(score)}: predicted trajectory must be [T,>=1]")
        return float(values[-1, 0] / times[-1])
    motion = score.get("predicted_motion") or score.get("motion") or {}
    for field in ("forward_rate_mps", "forward_speed_mps", "speed_mps", "longitudinal_speed_mps"):
        if motion.get(field) is not None:
            value = float(motion[field])
            if not np.isfinite(value):
                raise ValueError(f"{_key(score)}: predicted speed is not finite")
            return value
    for field in ("predicted_speed_mps", "speed_mps", "forward_speed_mps"):
        if score.get(field) is not None:
            values = np.asarray(score[field], dtype=np.float64)
            if values.ndim == 0:
                value = float(values)
            elif values.shape == times.shape:
                value = float(np.mean(values))
            else:
                raise ValueError(f"{_key(score)}: predicted speed must be scalar or match time axis")
            if not np.isfinite(value):
                raise ValueError(f"{_key(score)}: predicted speed is not finite")
            return value
    raise ValueError(
        f"{_key(score)}: score needs predicted_progress_curve, predicted_trajectory, "
        "predicted_motion.forward_rate_mps, or predicted_speed_mps"
    )


def _bootstrap_mean(values: list[float], *, seed: int, draws: int = 10000) -> dict[str, Any]:
    if not values:
        return {"mean": None, "confidence_interval_95": None, "n_twins": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(np.mean(array)),
        "confidence_interval_95": [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))],
        "n_twins": int(len(array)),
    }


def _stable_seed(seed: int, control: str, offset: int = 0) -> int:
    digest = hashlib.blake2b(control.encode("utf-8"), digest_size=4).digest()
    return int(seed) + int.from_bytes(digest, "little") % 100000 + int(offset)


def evaluate(manifest_path: Path, scores_path: Path, *, seed: int = 0) -> dict[str, Any]:
    manifests = _read_jsonl(manifest_path)
    scores = _read_jsonl(scores_path)
    score_by_key = {_key(row): row for row in scores}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    for row in manifests:
        key = _key(row)
        if not key or key not in score_by_key:
            missing.append(key or "<missing-key>")
            continue
        condition = _condition(row)
        groups[(str(row.get("twin_id") or row.get("pair_id") or ""), _control(row))].append({
            "key": key,
            "condition": condition,
            "speed": _progress_scalar(score_by_key[key], row),
            "shape_score": score_by_key[key].get("shape_score"),
        })
    invalid = []
    pairs = []
    for (twin_id, control), branches in sorted(groups.items()):
        conditions = {row["condition"] for row in branches}
        if not twin_id or len(branches) != 2 or conditions != {"fast", "slow"}:
            invalid.append({"twin_id": twin_id, "control_type": control, "branches": len(branches)})
            continue
        fast = next(row for row in branches if row["condition"] == "fast")
        slow = next(row for row in branches if row["condition"] == "slow")
        pairs.append({
            "twin_id": twin_id,
            "control_type": control,
            "fast_score": float(fast["speed"]),
            "slow_score": float(slow["speed"]),
            "margin_fast_minus_slow": float(fast["speed"] - slow["speed"]),
            "correct": bool(fast["speed"] > slow["speed"]),
            "shape_scores": [fast["shape_score"], slow["shape_score"]],
        })
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_control[pair["control_type"]].append(pair)

    def control_report(control: str, values: list[dict[str, Any]]) -> dict[str, Any]:
        correct = [float(row["correct"]) for row in values]
        margins = [float(row["margin_fast_minus_slow"]) for row in values]
        report = {
            "pairs": len(values),
            "condition_accuracy": _bootstrap_mean(correct, seed=_stable_seed(seed, control)),
            "margin_fast_minus_slow": _bootstrap_mean(margins, seed=_stable_seed(seed, control, 1000)),
        }
        shape = [float(value) for row in values for value in row["shape_scores"] if value is not None]
        if shape:
            report["mean_shape_score"] = float(np.mean(shape))
        return report

    clean = control_report("clean", by_control.get("clean", []))
    controls = {name: control_report(name, values) for name, values in sorted(by_control.items()) if name != "clean"}
    summary = {
        "protocol": "pure-speed-reciprocal-real-video-v1",
        "manifest": str(manifest_path.resolve()),
        "scores": str(scores_path.resolve()),
        "candidate_blind_required": True,
        "manifest_rows": len(manifests),
        "scored_rows": len(manifests) - len(missing),
        "missing_scores": missing,
        "invalid_groups": invalid,
        "clean": clean,
        "controls": controls,
        "wrong_identity_rejection": (
            1.0 - float(controls["wrong_identity"]["condition_accuracy"]["mean"])
            if "wrong_identity" in controls and controls["wrong_identity"]["condition_accuracy"]["mean"] is not None
            else None
        ),
        "time_reversal_drop": (
            float(clean["condition_accuracy"]["mean"] - controls["time_reversed"]["condition_accuracy"]["mean"])
            if "time_reversed" in controls and clean["condition_accuracy"]["mean"] is not None
            and controls["time_reversed"]["condition_accuracy"]["mean"] is not None
            else None
        ),
        "results": pairs,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.scores, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"results"}}, indent=2))


if __name__ == "__main__":
    main()
