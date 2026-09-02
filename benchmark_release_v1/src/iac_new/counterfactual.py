"""Small, deterministic counterfactual trajectory banks."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def dense_counterfactual_trajectories(
    reference: np.ndarray,
    *,
    speed_factors: Iterable[float] = (0.85, 1.0, 1.15),
    lateral_offsets_m: Iterable[float] = (-0.75, -0.375, 0.0, 0.375, 0.75),
    curvature_offsets_1pm: Iterable[float] = (-0.02, 0.0, 0.02),
) -> list[tuple[str, np.ndarray, dict[str, float]]]:
    """Create smooth perturbations around one reference trajectory.

    Lateral perturbations grow with forward progress. Curvature increments
    accumulate over the perturbed path's arc length, avoiding discontinuities.
    """
    trajectory = np.asarray(reference, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("reference trajectory must have shape [T,3]")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("reference trajectory must be finite")
    progress = trajectory[:, 0] - trajectory[0, 0]
    progress = np.maximum(progress, 0.0)
    progress /= max(float(progress[-1]), 1e-6)
    progress[0] = 0.0
    # Include a zero origin in the smooth lateral profile: perturbations affect
    # the future path while preserving the anchor at t=0.
    generated: list[tuple[str, np.ndarray, dict[str, float]]] = []
    seen: set[tuple[float, ...]] = set()
    for speed_factor in speed_factors:
        for lateral_offset in lateral_offsets_m:
            for curvature_offset in curvature_offsets_1pm:
                candidate = trajectory.copy()
                candidate[:, 0] *= float(speed_factor)
                candidate[:, 1] += float(lateral_offset) * progress
                segment_lengths = np.linalg.norm(np.diff(candidate[:, :2], axis=0), axis=1)
                arc_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
                candidate[:, 2] += float(curvature_offset) * arc_length
                key = tuple(np.round(candidate.reshape(-1), 8).tolist())
                if key in seen:
                    continue
                seen.add(key)
                label = (
                    f"dense_s{speed_factor:.2f}_y{lateral_offset:+.3f}_"
                    f"curv{curvature_offset:+.3f}"
                ).replace("+", "p").replace("-", "m").replace(".", "p")
                generated.append((label, candidate, {
                    "speed_factor": float(speed_factor),
                    "lateral_offset_m": float(lateral_offset),
                    "curvature_offset_1pm": float(curvature_offset),
                }))
    return generated


def densify_record(
    row: dict[str, Any],
    *,
    reference_candidate_id: str | None = None,
    speed_factors: Iterable[float] = (0.85, 1.0, 1.15),
    lateral_offsets_m: Iterable[float] = (-0.75, -0.375, 0.0, 0.375, 0.75),
    curvature_offsets_1pm: Iterable[float] = (-0.02, 0.0, 0.02),
) -> dict[str, Any]:
    """Return a manifest row with original candidates plus dense controls."""
    speed_factors = tuple(float(value) for value in speed_factors)
    lateral_offsets_m = tuple(float(value) for value in lateral_offsets_m)
    curvature_offsets_1pm = tuple(float(value) for value in curvature_offsets_1pm)
    candidates = list(row.get("candidates") or [])
    if len(candidates) < 1:
        raise ValueError(f"{row.get('sample_id')}: at least one candidate is required")
    reference_id = str(reference_candidate_id or row.get("gt_candidate_id") or candidates[0]["candidate_id"])
    reference = next((candidate for candidate in candidates if str(candidate["candidate_id"]) == reference_id), None)
    if reference is None:
        raise ValueError(f"{row.get('sample_id')}: reference candidate {reference_id} is absent")
    dense = dense_counterfactual_trajectories(
        np.asarray(reference["trajectory"], dtype=np.float64),
        speed_factors=speed_factors,
        lateral_offsets_m=lateral_offsets_m,
        curvature_offsets_1pm=curvature_offsets_1pm,
    )
    existing = {str(candidate["candidate_id"]) for candidate in candidates}
    existing_trajectories = {
        tuple(np.round(np.asarray(candidate["trajectory"], dtype=np.float64).reshape(-1), 8).tolist())
        for candidate in candidates
    }
    added = []
    for label, trajectory, factors in dense:
        candidate_id = label
        trajectory_key = tuple(np.round(trajectory.reshape(-1), 8).tolist())
        if candidate_id in existing or trajectory_key in existing_trajectories:
            continue
        existing.add(candidate_id)
        existing_trajectories.add(trajectory_key)
        added.append({
            "candidate_id": candidate_id,
            "prior": 1.0,
            "trajectory": trajectory.tolist(),
            "counterfactual": factors,
            "parent_candidate_id": reference_id,
        })
    output = dict(row)
    # Preserve explicit labels in the manifest. The dense bank itself is only
    # a plausible kinematic neighborhood; it is not a collision oracle.
    labeled_candidates = []
    for candidate in candidates:
        item = dict(candidate)
        if str(item.get("candidate_id")) == reference_id:
            item.setdefault("feasibility_label", "known_valid")
        labeled_candidates.append(item)
    for item in added:
        item["feasibility_label"] = "plausible"
    output["candidates"] = labeled_candidates + added
    metadata = dict(row.get("metadata") or {})
    metadata["counterfactual_bank"] = {
        "reference_candidate_id": reference_id,
        "num_generated": len(added),
        "speed_factors": [float(value) for value in speed_factors],
        "lateral_offsets_m": [float(value) for value in lateral_offsets_m],
        "curvature_offsets_1pm": [float(value) for value in curvature_offsets_1pm],
    }
    output["metadata"] = metadata
    return output
