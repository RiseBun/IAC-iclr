"""Fit posterior temperature and conformal threshold on held-out rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import read_jsonl
from .scoring import finite_sample_quantile, posterior_from_energies


def candidate_energies(row: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    scores = list(row.get("candidate_scores") or [])
    if not scores:
        raise ValueError(f"{row.get('sample_id')}: candidate_scores are missing")
    return (
        np.asarray([float(item["energy"]) for item in scores], dtype=np.float64),
        [str(item["candidate_id"]) for item in scores],
    )


def candidate_priors(row: dict[str, Any], ids: list[str]) -> np.ndarray:
    mapping = dict(row.get("candidate_priors") or {})
    if not mapping:
        return np.ones(len(ids), dtype=np.float64)
    return np.asarray([float(mapping[candidate_id]) for candidate_id in ids])


def posterior_diagnostics(rows: list[dict[str, Any]], temperature: float) -> dict[str, float | int | None]:
    losses = []
    confidences = []
    correct = []
    brier = []
    for row in rows:
        gt_id = row.get("gt_candidate_id")
        if gt_id is None:
            continue
        energies, ids = candidate_energies(row)
        if str(gt_id) not in ids:
            continue
        probabilities = posterior_from_energies(
            energies, temperature=temperature, priors=candidate_priors(row, ids)
        )
        gt_index = ids.index(str(gt_id))
        losses.append(-np.log(max(float(probabilities[gt_index]), 1e-12)))
        top = int(np.argmax(probabilities))
        confidences.append(float(probabilities[top]))
        correct.append(float(top == gt_index))
        target = np.zeros(len(ids), dtype=np.float64)
        target[gt_index] = 1.0
        brier.append(float(np.mean(np.square(probabilities - target))))
    if not losses:
        return {"samples": 0, "nll": None, "brier": None, "ece": None, "top1_accuracy": None}
    ece = 0.0
    confidence_array = np.asarray(confidences)
    correct_array = np.asarray(correct)
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence_array >= lower) & (confidence_array < lower + 0.1 + 1e-12)
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence_array[mask].mean() - correct_array[mask].mean()))
    return {
        "samples": len(losses),
        "nll": float(np.mean(losses)),
        "brier": float(np.mean(brier)),
        "ece": float(ece),
        "top1_accuracy": float(np.mean(correct)),
    }


def fit_temperature(rows: list[dict[str, Any]]) -> float:
    values = np.geomspace(0.01, 10.0, 80)
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in values:
        losses = []
        for row in rows:
            gt_id = row.get("gt_candidate_id")
            if gt_id is None:
                continue
            energies, ids = candidate_energies(row)
            gt_index = ids.index(str(gt_id))
            probabilities = posterior_from_energies(
                energies,
                temperature=float(temperature),
                priors=candidate_priors(row, ids),
            )
            losses.append(-np.log(max(float(probabilities[gt_index]), 1e-12)))
        if losses and float(np.mean(losses)) < best_nll:
            best_nll = float(np.mean(losses))
            best_temperature = float(temperature)
    return best_temperature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=0.9)
    args = parser.parse_args()
    rows = read_jsonl(args.scores)
    temperature = fit_temperature(rows)
    nll_scores: list[float] = []
    for row in rows:
        gt_id = row.get("gt_candidate_id")
        if gt_id is None:
            continue
        energies, ids = candidate_energies(row)
        probabilities = posterior_from_energies(
            energies,
            temperature=temperature,
            priors=candidate_priors(row, ids),
        )
        nll_scores.append(-np.log(max(float(probabilities[ids.index(str(gt_id))]), 1e-12)))
    threshold = finite_sample_quantile(np.asarray(nll_scores), args.coverage)
    result = {
        "protocol": "iac-new-image-v1",
        "temperature": temperature,
        "target_coverage": float(args.coverage),
        "nll_threshold": threshold,
        "calibration_samples": len(nll_scores),
        "uncalibrated": posterior_diagnostics(rows, 1.0),
        "calibrated": posterior_diagnostics(rows, temperature),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
