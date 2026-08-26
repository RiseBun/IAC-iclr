"""Synthetic oracle for long-range road-boundary propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.temporal_geometry import HomographyBoundaryPropagator, RoadStateFilter


def _mask(h: int, w: int, offset: float, slope: float, curvature: float) -> np.ndarray:
    value = np.zeros((h, w), dtype=bool)
    for row in range(int(h * 0.30), h):
        z = 1.0 - row / max(h - 1, 1)
        center = 0.5 * w + offset + slope * z + curvature * z * z
        width = 0.34 * w + 0.10 * w * z
        left = int(np.clip(center - width / 2.0, 0, w - 1))
        right = int(np.clip(center + width / 2.0, left + 1, w))
        value[row, left:right] = True
    return value


def _center_error(state: dict, h: int, w: int, offset: float, slope: float, curvature: float) -> float | None:
    points = state.get("left_boundary", [])
    right = state.get("right_boundary", [])
    if len(points) < 2 or len(right) < 2:
        return None
    rows = np.asarray([p[1] for p in points], dtype=np.float64)
    center = 0.5 * (np.asarray([p[0] for p in points]) + np.asarray([p[0] for p in right]))
    z = 1.0 - rows / max(h - 1, 1)
    target = 0.5 * w + offset + slope * z + curvature * z * z
    return float(np.median(np.abs(center - target)))


def run_case(rng: np.random.Generator, *, missing_fraction: float, flow_noise_px: float) -> dict:
    h, w, t = 144, 256, 4
    offset, slope, curvature = rng.uniform(-12, 12), rng.uniform(-20, 20), rng.uniform(-18, 18)
    masks = np.stack([_mask(h, w, offset + 1.8 * i, slope, curvature) for i in range(t)])
    observed_masks = masks.copy()
    missing = rng.random(t) < missing_fraction
    missing[0] = False
    observed_masks[missing] = False
    flows = np.zeros((t, h, w, 2), dtype=np.float32)
    flows[..., 0] = 1.8 + rng.normal(0.0, flow_noise_px, size=(t, h, w))
    flows[..., 1] = rng.normal(0.0, flow_noise_px, size=(t, h, w))
    weights = np.ones((t, h, w), dtype=np.float32)
    K = np.asarray([[180.0, 0.0, w / 2], [0.0, 180.0, h / 2], [0.0, 0.0, 1.0]])
    camera = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]])
    propagated = HomographyBoundaryPropagator(min_current_confidence_for_propagation=0.9).update(
        observed_masks, intrinsics=K, camera_to_ego=camera,
        observed_flows=flows, static_weights=weights,
        history_ego_state=np.asarray([[0.0, 0.0, 0.0, 4.0, 0.0]]),
        history_times_s=np.asarray([0.0]), future_times_s=np.arange(1, t + 1, dtype=np.float64),
    )
    baseline = RoadStateFilter().update(observed_masks)
    errors_prop, errors_base = [], []
    for i, state in enumerate(propagated["states"]):
        truth = _center_error({"left_boundary": [[p[0], p[1]] for p in []]}, h, w, offset + 1.8 * i, slope, curvature)
        # Use direct centerline extraction from the known mask as the oracle.
        rows = np.flatnonzero(masks[i].any(axis=1))
        xs = [np.flatnonzero(masks[i, row]).mean() for row in rows]
        if state.get("left_boundary") and state.get("right_boundary"):
            pred_rows = np.asarray([p[1] for p in state["left_boundary"]])
            pred_center = 0.5 * (np.asarray([p[0] for p in state["left_boundary"]]) + np.asarray([p[0] for p in state["right_boundary"]]))
            gt_center = np.interp(pred_rows, rows, xs)
            errors_prop.append(float(np.median(np.abs(pred_center - gt_center))))
        base_state = baseline["states"][i]
        if base_state.get("measurement", {}).get("boundaries", {}).get("valid"):
            b = base_state["measurement"]["boundaries"]
            pred_rows = np.asarray(b["rows"], dtype=np.float64)
            pred_center = 0.5 * (np.asarray(b["left_x"]) + np.asarray(b["right_x"]))
            gt_center = np.interp(pred_rows, rows, xs)
            errors_base.append(float(np.median(np.abs(pred_center - gt_center))))
    return {
        "missing_fraction": missing_fraction,
        "flow_noise_px": flow_noise_px,
        "missing_intervals": int(missing.sum()),
        "propagation_valid": int(len(errors_prop)),
        "baseline_valid": int(len(errors_base)),
        "propagation_error_px": float(np.median(errors_prop)) if errors_prop else None,
        "baseline_error_px": float(np.median(errors_base)) if errors_base else None,
        "propagation_applied_fraction": float(np.mean([x.get("propagation_applied", False) for x in propagated["propagation"]])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=200)
    args = parser.parse_args()
    rows = [run_case(np.random.default_rng(1000 + i), missing_fraction=float([0.0, 0.25, 0.5][i % 3]), flow_noise_px=float([0.0, 0.5, 1.0][i % 3])) for i in range(args.cases)]
    result = {"protocol": "boundary-propagation-synthetic-oracle-v1", "num_cases": len(rows), "rows": rows}
    for missing in [0.0, 0.25, 0.5]:
        subset = [r for r in rows if r["missing_fraction"] == missing]
        result[f"missing_{missing}"] = {
            "baseline_valid": float(np.mean([r["baseline_valid"] for r in subset])),
            "propagation_valid": float(np.mean([r["propagation_valid"] for r in subset])),
            "baseline_error_px": float(np.nanmedian([r["baseline_error_px"] if r["baseline_error_px"] is not None else np.nan for r in subset])),
            "propagation_error_px": float(np.nanmedian([r["propagation_error_px"] if r["propagation_error_px"] is not None else np.nan for r in subset])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
