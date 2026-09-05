#!/usr/bin/env python3
"""Build a candidate-blind Level-1 manifest from generated WAM future frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_key(row: dict[str, Any]) -> str:
    return str(row.get("source_key") or (row.get("metadata") or {}).get("source_key") or "")


def _generated_images(row: dict[str, Any]) -> list[str]:
    value = row.get("future_images")
    if value is None:
        value = row.get("generated_future_images")
    if not isinstance(value, list) or not value or any(not str(item) for item in value):
        raise ValueError(f"{row.get('branch_id', '<unknown>')}: generated future images are required")
    return [str(item) for item in value]


ALLOWED_FUTURE_COUNTS = frozenset({4, 8})


def build_manifest(
    base_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    *,
    expected_history_count: int = 4,
    allowed_future_counts: frozenset[int] | set[int] = ALLOWED_FUTURE_COUNTS,
    expected_future_count: int | None = None,
    check_files: bool = False,
) -> list[dict[str, Any]]:
    if expected_future_count is not None:
        if expected_future_count not in allowed_future_counts:
            raise ValueError(f"expected_future_count must be one of {sorted(allowed_future_counts)}")
        allowed = frozenset({expected_future_count})
    else:
        allowed = frozenset(allowed_future_counts)
    base_by_key: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        key = _source_key(row)
        if not key:
            raise ValueError(f"base row {row.get('sample_id')}: source_key is required")
        if key in base_by_key:
            raise ValueError(f"duplicate base source_key: {key}")
        base_by_key[key] = row
    output: list[dict[str, Any]] = []
    for generated in generated_rows:
        branch_id = str(generated.get("branch_id") or "")
        if not branch_id:
            raise ValueError("every generated row needs branch_id")
        if generated.get("future_images_source") != "wam_generated":
            raise ValueError(f"{branch_id}: future_images_source must be wam_generated")
        if generated.get("wam_generation_status") not in {None, "complete"}:
            raise ValueError(f"{branch_id}: WAM generation is not complete")
        model_id = str(generated.get("wam_model_id") or "")
        if not model_id:
            raise ValueError(f"{branch_id}: wam_model_id is required")
        key = _source_key(generated)
        if key not in base_by_key:
            raise ValueError(f"{branch_id}: no base row for source_key {key}")
        base = base_by_key[key]
        history = list(base.get("history_frame_paths") or [])
        if len(history) != expected_history_count:
            raise ValueError(f"{branch_id}: base history must contain {expected_history_count} frames")
        future = _generated_images(generated)
        future_count = len(future)
        if future_count not in allowed:
            raise ValueError(
                f"{branch_id}: generated future must contain one of {sorted(allowed)} frames, got {future_count}"
            )
        if check_files:
            missing = [path for path in history + future if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"{branch_id}: missing image {missing[0]}")
        times = np.asarray(generated.get("future_times_s", base.get("future_times_s")), dtype=np.float64)
        if times.shape != (future_count,) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError(
                f"{branch_id}: future_times_s must be finite, increasing, and length {future_count}"
            )
        if times[0] <= 0.0 or not (3.95 <= float(times[-1]) <= 4.05):
            raise ValueError(f"{branch_id}: generated future must cover approximately 4.0 seconds")
        action = generated.get("action_condition", {}).get("trajectory")
        if action is None:
            action = generated.get("action_trajectory")
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (future_count, 3) or not np.all(np.isfinite(action_array)):
            raise ValueError(f"{branch_id}: action head trajectory must have shape [{future_count},3]")
        if generated.get("realized_future_ego_state") is not None or (generated.get("metadata") or {}).get("realized_future_ego_state") is not None:
            raise ValueError(f"{branch_id}: generated WAM output contains realized future state leakage")
        action_source = str(generated.get("action_source") or (generated.get("metadata") or {}).get("action_source") or "")
        if action_source in {"logged", "oracle", "proxy", "candidate"}:
            raise ValueError(f"{branch_id}: action source must be the independent WAM action head")
        candidates = list(base.get("candidates") or [])
        if len(candidates) < 2:
            raise ValueError(f"{branch_id}: base candidate bank must contain at least two entries")
        candidates.append({
            "candidate_id": "wam_action_head",
            "prior": 1.0,
            "trajectory": action_array.tolist(),
            "support_label": "independent_action_head_reference",
        })
        metadata = dict(base.get("metadata") or {})
        # Future realized state is deliberately removed: it is not an input to
        # the image probe and belongs only to Level-2/3 closed-loop scoring.
        metadata.pop("realized_future_ego_state", None)
        metadata.update({
            "protocol": f"wam-generated-level1-history4-future{future_count}-v1",
            "source_key": key,
            "branch_id": branch_id,
            "wam_model_id": model_id,
            "future_images_source": "wam_generated",
            "action_source": "wam_action_head",
            "action_trajectory_source": "wam_action_head",
            "candidate_blind_image_branch": True,
            "action_waypoint_used_by_image_branch": False,
            "realized_future_state_available_to_image_branch": False,
        })
        row = dict(base)
        row.update({
            "sample_id": f"{base.get('sample_id', key)}::generated::{branch_id}",
            "history_frame_paths": history,
            "future_frame_paths": future,
            "future_times_s": times.tolist(),
            "candidates": candidates,
            "gt_candidate_id": "wam_action_head",
            "future_images_source": "wam_generated",
            "wam_model_id": model_id,
            "branch_id": branch_id,
            "generated_future_id": branch_id,
            "action_trajectory": action_array.tolist(),
            "action_trajectory_source": "wam_action_head",
            "metadata": metadata,
        })
        # Preserve the paired-counterfactual lineage needed by Level-2. These
        # fields are metadata only and are never exposed to the image decoder.
        for field in ("counterfactual_group_id", "branch_role", "history_fingerprint", "nuisance_seed"):
            if field in generated:
                row[field] = generated[field]
        row.pop("realized_future_ego_state", None)
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument(
        "--expected-future-count",
        type=int,
        choices=sorted(ALLOWED_FUTURE_COUNTS),
        default=None,
        help="pin exactly 4 or 8 future frames; default accepts either",
    )
    args = parser.parse_args()
    rows = build_manifest(
        _read(args.base),
        _read(args.generated),
        expected_future_count=args.expected_future_count,
        check_files=args.check_files,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    future_counts = sorted({len(row["future_frame_paths"]) for row in rows})
    print(json.dumps({
        "protocol": "wam-generated-level1-history4-future4or8-v1",
        "rows": len(rows),
        "allowed_future_counts": sorted(ALLOWED_FUTURE_COUNTS),
        "observed_future_counts": future_counts,
        "wam_models": sorted({row["wam_model_id"] for row in rows}),
        "output": str(args.output.resolve()),
        "future_images_source": "wam_generated",
        "realized_future_state_exposed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
