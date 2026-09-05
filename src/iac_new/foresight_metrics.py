"""Main WAM metrics for the visual-action benchmark.

The two primary quantities are deliberately output-only:

``CFAC`` (Cross-modal Future--Action Consistency)
    Does the imagined future motion agree with the native action motion?

``FAU`` (Foresight Action Utility)
    Do *both* imagined motion and native action agree with the observed future,
    relative to the same history?  A planner that never submits a future visual
    state cannot obtain FAU.

The module does not require an intervention API or a simulator.  Those are
optional audits and are kept outside the main score.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .continuous_motion import (
    IMAGE_PROFILE_SOURCES,
    history_only_motion_profile,
    image_motion_profile,
    trajectory_to_motion_profile,
)


SHAPE_FIELDS = ("lateral_speed_mps", "yaw_rate_radps", "curvature_1pm")
DEFAULT_DEADBANDS = {
    "lateral_speed_mps": 0.05,
    "yaw_rate_radps": 0.02,
    "curvature_1pm": 0.005,
}
DEFAULT_SCALES = {
    "lateral_speed_mps": 0.25,
    "yaw_rate_radps": 0.05,
    "curvature_1pm": 0.01,
}


def _rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(profile.get("rows") or [])
    if not rows:
        raise ValueError("motion profile has no rows")
    return rows


def _values(profile: dict[str, Any], field: str) -> np.ndarray:
    values = np.asarray([row.get(field, np.nan) for row in _rows(profile)], dtype=np.float64)
    return values


def _times(profile: dict[str, Any]) -> np.ndarray:
    values = np.asarray([row.get("time_s", np.nan) for row in _rows(profile)], dtype=np.float64)
    if not np.all(np.isfinite(values)) or values[0] <= 0 or np.any(np.diff(values) <= 0):
        raise ValueError("profile time_s must be positive and strictly increasing")
    return values


def _check_same_time_axis(profiles: Iterable[dict[str, Any]]) -> np.ndarray:
    axes = [_times(profile) for profile in profiles]
    reference = axes[0]
    if any(axis.shape != reference.shape or not np.allclose(axis, reference, atol=1e-6, rtol=0) for axis in axes[1:]):
        raise ValueError("all profiles must share the same future time axis")
    return reference


def _status(row: dict[str, Any]) -> str:
    return str(row.get("shape_status", row.get("status", "abstain")))


def _weights(profile: dict[str, Any], n: int) -> np.ndarray:
    values = np.asarray(
        [row.get("shape_observability", row.get("observability", 0.0)) for row in _rows(profile)],
        dtype=np.float64,
    )
    if values.shape != (n,):
        raise ValueError("observability must match profile rows")
    return np.clip(values, 0.0, 1.0)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        return None
    return float(np.average(values[finite], weights=weights[finite]))


def _direction_score(first: np.ndarray, second: np.ndarray, deadband: float) -> float | None:
    valid = np.isfinite(first) & np.isfinite(second)
    if not np.any(valid):
        return None
    a, b = first[valid], second[valid]
    a_sign = np.where(np.abs(a) <= deadband, 0, np.sign(a))
    b_sign = np.where(np.abs(b) <= deadband, 0, np.sign(b))
    return float(np.mean(a_sign == b_sign))


def _normalised_mae(first: np.ndarray, second: np.ndarray, scale: float, weights: np.ndarray | None = None) -> float | None:
    valid = np.isfinite(first) & np.isfinite(second)
    if not np.any(valid):
        return None
    errors = np.abs(first[valid] - second[valid])
    if weights is None:
        mean_error = float(np.mean(errors))
    else:
        local_weights = np.asarray(weights, dtype=np.float64)[valid]
        keep = np.isfinite(local_weights) & (local_weights > 0)
        if not np.any(keep):
            return None
        mean_error = float(np.average(errors[keep], weights=local_weights[keep]))
    return float(mean_error / max(float(scale), 1e-9))


def _same_time_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    valid = np.isfinite(first) & np.isfinite(second)
    if np.sum(valid) < 1:
        return None
    a, b = first[valid], second[valid]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _best_lag(first: np.ndarray, second: np.ndarray, max_lag: int = 1) -> tuple[int | None, float | None]:
    """Return diagnostic best lag; the main score remains same-time."""
    candidates: list[tuple[float, int]] = []
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            a, b = first[-lag:], second[:lag]
        elif lag > 0:
            a, b = first[:-lag], second[lag:]
        else:
            a, b = first, second
        valid = np.isfinite(a) & np.isfinite(b)
        if np.any(valid):
            candidates.append((float(np.mean(np.abs(a[valid] - b[valid]))), lag))
    if not candidates:
        return None, None
    error, lag = min(candidates)
    return int(lag), float(error)


def _component_alignment(
    first: np.ndarray,
    second: np.ndarray,
    *,
    field: str,
    weights: np.ndarray,
    deadbands: dict[str, float],
    scales: dict[str, float],
) -> dict[str, Any]:
    valid = np.isfinite(first) & np.isfinite(second) & (weights > 0)
    if not np.any(valid):
        return {"status": "abstain", "count": 0}
    local_weights = weights[valid]
    abs_error = np.abs(first[valid] - second[valid])
    mae = float(np.average(abs_error, weights=local_weights))
    scale = max(float(scales.get(field, DEFAULT_SCALES[field])), 1e-9)
    magnitude_score = float(np.exp(-mae / scale))
    direction = _direction_score(first[valid], second[valid], float(deadbands.get(field, DEFAULT_DEADBANDS[field])))
    cosine = _same_time_cosine(first[valid], second[valid])
    temporal_score = None if cosine is None else float((cosine + 1.0) / 2.0)
    lag, lag_mae = _best_lag(first[valid], second[valid])
    return {
        "status": "ok",
        "count": int(np.sum(valid)),
        "mae": mae,
        "normalised_mae": float(mae / scale),
        "direction_accuracy": direction,
        "same_time_cosine": cosine,
        "temporal_alignment": temporal_score,
        "best_lag_intervals": lag,
        "best_lag_mae": lag_mae,
        "direction_score": direction,
        "magnitude_score": magnitude_score,
    }


def _profile_from_submission(row: dict[str, Any], key: str, times: np.ndarray) -> dict[str, Any]:
    supplied_keys = {
        "imagined": ("imagined_motion_profile", "image_motion_profile", "imagined_profile"),
        "action": ("native_action_profile", "action_motion_profile", "action_profile"),
        "ground_truth": ("ground_truth_motion_profile", "ground_truth_profile"),
        "history": ("history_motion_profile",),
    }
    supplied = next((row.get(name) for name in supplied_keys[key] if row.get(name) is not None), None)
    if supplied is not None:
        return supplied
    trajectory_key = {
        "action": "action_trajectory",
        "ground_truth": "ground_truth_trajectory",
        "imagined": "imagined_trajectory",
        "history": None,
    }[key]
    if trajectory_key and row.get(trajectory_key) is not None:
        profile = trajectory_to_motion_profile(row[trajectory_key], times.tolist(), initial_speed_mps=row.get("initial_speed_mps"))
        profile["source"] = {
            "action": "native_action_head",
            "ground_truth": "ground_truth_future",
            "imagined": "image_only_candidate_blind_decoder",
        }[key]
        return profile
    raise ValueError(f"submission row is missing {key} profile")


def evaluate_cfac(
    imagined_profile: dict[str, Any],
    action_profile: dict[str, Any],
    *,
    deadbands: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    min_observability: float = 0.0,
) -> dict[str, Any]:
    """Evaluate one WAM sample's imagined-vs-native motion agreement.

    This is a consistency score, not a causal claim.  The image profile must
    have been produced without seeing the native action.
    """
    if imagined_profile.get("source") not in IMAGE_PROFILE_SOURCES and not imagined_profile.get("candidate_blind", False):
        raise ValueError("imagined_profile must declare an action-blind source")
    _check_same_time_axis((imagined_profile, action_profile))
    n = len(_rows(imagined_profile))
    weights = _weights(imagined_profile, n)
    weights = np.where(weights >= float(min_observability), weights, 0.0)
    deadbands = {**DEFAULT_DEADBANDS, **(deadbands or {})}
    scales = {**DEFAULT_SCALES, **(scales or {})}
    components: dict[str, dict[str, Any]] = {}
    for field in SHAPE_FIELDS:
        components[field] = _component_alignment(
            _values(imagined_profile, field),
            _values(action_profile, field),
            field=field,
            weights=weights,
            deadbands=deadbands,
            scales=scales,
        )
    shape_matrix = np.column_stack([_values(imagined_profile, field) for field in SHAPE_FIELDS])
    valid_intervals = np.any(np.isfinite(shape_matrix), axis=1) & (weights > 0)
    coverage = float(np.mean(valid_intervals))
    scores = []
    for component in components.values():
        if component.get("status") == "ok":
            parts = [component.get("direction_score"), component.get("magnitude_score"), component.get("temporal_alignment")]
            parts = [float(value) for value in parts if value is not None]
            if parts:
                scores.append(float(np.prod(parts) ** (1.0 / len(parts))))
    return {
        "protocol": "iac-cfac-v1",
        "metric": "CFAC",
        "definition": "imagined_motion_vs_native_action_motion_same_time",
        "status": "ok" if scores else "abstain",
        "score": float(np.mean(scores)) if scores else None,
        "coverage": coverage,
        "evaluable_intervals": int(np.sum(valid_intervals)),
        "total_intervals": int(n),
        "components": components,
        "claim": "consistency_only_no_causal_claim",
        "leakage_audit": {
            "action_visible_to_image_branch": False,
            "future_ground_truth_visible_to_image_branch": False,
        },
    }


def evaluate_fau(
    imagined_profile: dict[str, Any],
    action_profile: dict[str, Any],
    ground_truth_profile: dict[str, Any],
    *,
    history_profile: dict[str, Any] | None = None,
    scales: dict[str, float] | None = None,
    min_observability: float = 0.0,
) -> dict[str, Any]:
    """Evaluate foresight utility using *both* imagined and native outputs.

    ``fau_f`` is imagined-vs-real future and ``fau_a`` is native-action-vs-real
    future.  Errors are computed on residuals after subtracting the past-only
    history profile when it is provided; raw errors are retained for audit.
    """
    if imagined_profile.get("source") not in IMAGE_PROFILE_SOURCES and not imagined_profile.get("candidate_blind", False):
        raise ValueError("imagined_profile must declare an action-blind source")
    _check_same_time_axis((imagined_profile, action_profile, ground_truth_profile))
    n = len(_rows(imagined_profile))
    weights = _weights(imagined_profile, n)
    statuses = np.asarray([_status(item) for item in _rows(imagined_profile)], dtype=object)
    weights = np.where((weights >= float(min_observability)) & np.isin(statuses, ["usable", "uncertain"]), weights, 0.0)
    scales = {**DEFAULT_SCALES, **(scales or {})}
    baseline = history_profile
    components: dict[str, dict[str, Any]] = {}
    f_scores, a_scores, baseline_scores = [], [], []
    for field in SHAPE_FIELDS:
        imagined = _values(imagined_profile, field)
        action = _values(action_profile, field)
        truth = _values(ground_truth_profile, field)
        if baseline is not None:
            history = _values(baseline, field)
            imagined_eval, action_eval, truth_eval = imagined - history, action - history, truth - history
        else:
            imagined_eval, action_eval, truth_eval = imagined, action, truth
        scale = max(float(scales.get(field, DEFAULT_SCALES[field])), 1e-9)
        f_mae = _normalised_mae(imagined_eval, truth_eval, scale, weights)
        a_mae = _normalised_mae(action_eval, truth_eval, scale, weights)
        b_mae = _normalised_mae(np.zeros_like(truth_eval), truth_eval, scale, weights) if baseline is not None else None
        f_score = None if f_mae is None else float(np.exp(-f_mae))
        a_score = None if a_mae is None else float(np.exp(-a_mae))
        b_score = None if b_mae is None else float(np.exp(-b_mae))
        if f_score is not None:
            f_scores.append(f_score)
        if a_score is not None:
            a_scores.append(a_score)
        if b_score is not None:
            baseline_scores.append(b_score)
        components[field] = {
            "scale": scale,
            "fau_f_normalised_mae": f_mae,
            "fau_a_normalised_mae": a_mae,
            "fau_f_score": f_score,
            "fau_a_score": a_score,
            "history_baseline_score": b_score,
            "fau_f_relative_gain": None if f_mae is None or b_mae is None or b_mae < 1e-9 else float(np.clip(1.0 - f_mae / b_mae, -1.0, 1.0)),
            "fau_a_relative_gain": None if a_mae is None or b_mae is None or b_mae < 1e-9 else float(np.clip(1.0 - a_mae / b_mae, -1.0, 1.0)),
        }
    coverage = float(np.mean(weights > 0))
    fau_f = float(np.mean(f_scores)) if f_scores else None
    fau_a = float(np.mean(a_scores)) if a_scores else None
    fau = None if fau_f is None or fau_a is None else float(np.sqrt(max(fau_f, 0.0) * max(fau_a, 0.0)))
    return {
        "protocol": "iac-fau-v1",
        "metric": "FAU",
        "definition": "both_imagined_and_native_action_against_real_future_residual_to_history",
        "status": "ok" if fau is not None else "abstain",
        "score": fau,
        "fau_f": fau_f,
        "fau_a": fau_a,
        "history_baseline_score": float(np.mean(baseline_scores)) if baseline_scores else None,
        "coverage": coverage,
        "evaluable_intervals": int(np.sum(weights > 0)),
        "total_intervals": int(n),
        "components": components,
        "claim": "foresight_utility_not_causal_claim",
        "requires": ["future_visual_output", "native_action", "ground_truth_future"],
    }


def evaluate_submission_row(row: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one JSON object using precomputed Level-1 profiles."""
    times = np.asarray(row.get("future_times_s"), dtype=np.float64)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("future_times_s is required")
    imagined = _profile_from_submission(row, "imagined", times)
    action = _profile_from_submission(row, "action", times)
    cfac = evaluate_cfac(imagined, action)
    result: dict[str, Any] = {"sample_id": row.get("sample_id"), "cfac": cfac}
    if row.get("ground_truth_motion_profile") is not None or row.get("ground_truth_trajectory") is not None:
        truth = _profile_from_submission(row, "ground_truth", times)
        history = row.get("history_motion_profile")
        if history is None and row.get("history_ego_state") is not None:
            history = history_only_motion_profile(row["history_ego_state"], times.tolist(), history_times_s=row.get("history_times_s"))
        result["fau"] = evaluate_fau(imagined, action, truth, history_profile=history)
    else:
        result["fau"] = {"protocol": "iac-fau-v1", "metric": "FAU", "status": "unavailable", "reason": "ground_truth_future_not_submitted"}
    return result
