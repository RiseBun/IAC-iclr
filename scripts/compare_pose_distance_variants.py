#!/usr/bin/env python3
"""Ablate Level-1 pose normalization and ordered-curve diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.continuous_motion import constrained_dtw_distance, discrete_frechet_distance


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize(points: np.ndarray, mode: str) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    if mode == "x_end":
        scale = abs(float(result[-1, 0]))
    elif mode == "arc":
        scale = float(np.sum(np.linalg.norm(np.diff(result[:, :2], axis=0), axis=1)))
    else:
        raise ValueError(mode)
    if scale < 1e-8:
        return np.full_like(result, np.nan)
    result[:, :2] /= scale
    return result


def _metrics(image: np.ndarray, action: np.ndarray, mode: str) -> dict[str, float]:
    image_n = _normalize(image, mode)
    action_n = _normalize(action, mode)
    if not np.all(np.isfinite(image_n)) or not np.all(np.isfinite(action_n)):
        return {"pointwise_mae": float("nan"), "frechet": float("nan"), "dtw": float("nan"), "dtw_warp": float("nan")}
    dtw, warp = constrained_dtw_distance(image_n[:, :2], action_n[:, :2], window=1)
    return {
        "pointwise_mae": float(np.mean(np.abs(image_n - action_n))),
        "frechet": discrete_frechet_distance(image_n[:, :2], action_n[:, :2]),
        "dtw": dtw,
        "dtw_warp": warp,
    }


def _metrics_reversed_image(image: np.ndarray, action: np.ndarray, mode: str) -> dict[str, float]:
    """Reverse after normalization so the control tests temporal order only."""
    image_n = _normalize(image, mode)
    action_n = _normalize(action, mode)
    if not np.all(np.isfinite(image_n)) or not np.all(np.isfinite(action_n)):
        return {"pointwise_mae": float("nan"), "frechet": float("nan"), "dtw": float("nan"), "dtw_warp": float("nan")}
    image_n = image_n[::-1]
    dtw, warp = constrained_dtw_distance(image_n[:, :2], action_n[:, :2], window=1)
    return {
        "pointwise_mae": float(np.mean(np.abs(image_n - action_n))),
        "frechet": discrete_frechet_distance(image_n[:, :2], action_n[:, :2]),
        "dtw": dtw,
        "dtw_warp": warp,
    }


def _summary(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = values[0].keys()
    return {
        "mean": {key: float(np.nanmean([item[key] for item in values])) for key in keys},
        "median": {key: float(np.nanmedian([item[key] for item in values])) for key in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True, help="input_decoder_scores.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="input_manifest.jsonl")
    parser.add_argument("--filter-manifest", type=Path, help="optional JSONL defining sample_ids")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = {str(row["sample_id"]): row for row in _read_jsonl(args.scores) if row.get("valid", True)}
    manifests = {str(row["sample_id"]): row for row in _read_jsonl(args.manifest)}
    sample_ids = set(scores) & set(manifests)
    if args.filter_manifest:
        sample_ids &= {str(row["sample_id"]) for row in _read_jsonl(args.filter_manifest)}
    rows = []
    for sample_id in sorted(sample_ids):
        image = np.asarray(scores[sample_id]["decoder"]["trajectory"], dtype=np.float64)
        logged_id = str(manifests[sample_id].get("gt_candidate_id", "logged"))
        candidate = next(item for item in manifests[sample_id]["candidates"] if str(item["candidate_id"]) == logged_id)
        action = np.asarray(candidate["trajectory"], dtype=np.float64)
        if image.shape != action.shape or image.ndim != 2 or image.shape[1] != 3:
            continue
        rows.append({"sample_id": sample_id, "image": image, "action": action})
    if not rows:
        raise SystemExit("no intersecting valid samples")

    output: dict[str, Any] = {"protocol": "pose-distance-variant-ablation-v1", "records": len(rows), "variants": {}, "order_control": {}}
    for mode in ("x_end", "arc"):
        output["variants"][mode] = _summary([_metrics(item["image"], item["action"], mode) for item in rows])
        output["order_control"][f"reversed_image_{mode}"] = _summary([_metrics_reversed_image(item["image"], item["action"], mode) for item in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": output["protocol"], "records": len(rows), "variants_mean": {key: value["mean"] for key, value in output["variants"].items()}, "order_control_mean": {key: value["mean"] for key, value in output["order_control"].items()}}, indent=2))


if __name__ == "__main__":
    main()
