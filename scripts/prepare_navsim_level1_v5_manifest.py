#!/usr/bin/env python3
"""Prepare a clean NAVSIM Level-1 manifest with overlap-aware evaluation roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _start(row: dict[str, Any]) -> int:
    indices = row.get("metadata", {}).get("native_window_frame_indices")
    if not isinstance(indices, list) or len(indices) != 12:
        raise ValueError(f"{row.get('sample_id')}: native_window_frame_indices must contain 12 values")
    return int(indices[0])


def _validate(row: dict[str, Any]) -> None:
    if len(row.get("history_frame_paths", [])) != 4:
        raise ValueError(f"{row.get('sample_id')}: expected 4 history frames")
    if len(row.get("future_frame_paths", [])) != 8:
        raise ValueError(f"{row.get('sample_id')}: expected 8 future frames")
    history_times = row.get("history_times_s", [])
    future_times = row.get("future_times_s", [])
    if len(history_times) != 4 or len(future_times) != 8:
        raise ValueError(f"{row.get('sample_id')}: timestamp count mismatch")
    if abs(float(history_times[-1])) > 1e-6:
        raise ValueError(f"{row.get('sample_id')}: history anchor must be t=0")
    if not (3.95 <= float(future_times[-1]) <= 4.05):
        raise ValueError(f"{row.get('sample_id')}: future horizon is not approximately 4 seconds")
    if any(b <= a for a, b in zip(future_times, future_times[1:])):
        raise ValueError(f"{row.get('sample_id')}: future timestamps are not increasing")
    if len(row.get("candidates", [])) == 0:
        raise ValueError(f"{row.get('sample_id')}: missing logged action candidate")


def _clean_row(row: dict[str, Any], *, role: str, overlap_count: int) -> dict[str, Any]:
    _validate(row)
    metadata = dict(row.get("metadata") or {})
    metadata.update({
        "manifest_role": role,
        "overlap_count_same_scene": int(overlap_count),
        "evaluation_warning": (
            "native_realized_future_validates_measurement_only; not a WAM-generated future"
        ),
        "candidate_blind_image_branch": True,
        "action_waypoint_used_by_image_branch": False,
    })
    logged = [candidate for candidate in row.get("candidates", []) if candidate.get("candidate_id") == "logged"]
    if len(logged) != 1:
        raise ValueError(f"{row.get('sample_id')}: expected exactly one logged candidate")
    cleaned = dict(row)
    # Keep the candidate bank for protocol compatibility. Level 1 code must use
    # only gt_candidate_id after image decoding, never expose candidates upstream.
    cleaned["candidates"] = list(row["candidates"])
    cleaned["gt_candidate_id"] = "logged"
    cleaned["metadata"] = metadata
    return cleaned


def _select_nonoverlap(rows: list[dict[str, Any]]) -> set[str]:
    """Greedily select the maximum number of non-overlapping 12-frame windows per scene."""
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), []).append(row)
    selected: set[str] = set()
    for scene_rows in by_scene.values():
        last_end = -10**18
        ordered = sorted(scene_rows, key=lambda row: (_start(row), str(row["sample_id"])))
        for row in ordered:
            start = _start(row)
            end = start + 11
            if start > last_end:
                selected.add(str(row["sample_id"]))
                last_end = end
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows = _read(args.input)
    if len(rows) != 78:
        raise ValueError(f"expected the frozen 78-row source manifest, got {len(rows)}")
    selected_ids = _select_nonoverlap(rows)
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), []).append(row)
    overlap_counts = {
        str(row["sample_id"]): sum(
            other is not row
            and _start(other) <= _start(row) + 11
            and _start(other) + 11 >= _start(row)
            for other in by_scene[str(row["scene_id"])]
        )
        for row in rows
    }

    prepared = []
    for row in rows:
        sample_id = str(row["sample_id"])
        role = "evaluation_nonoverlap" if sample_id in selected_ids else "development_overlap"
        prepared.append(_clean_row(row, role=role, overlap_count=overlap_counts[sample_id]))
    prepared.sort(key=lambda row: (str(row["scene_id"]), _start(row), str(row["sample_id"])))

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_path = args.output_root / "navsim_level1_v5_all_78.jsonl"
    eval_path = args.output_root / "navsim_level1_v5_eval_nonoverlap.jsonl"
    dev_path = args.output_root / "navsim_level1_v5_development_overlap.jsonl"
    for path, subset in (
        (all_path, prepared),
        (eval_path, [row for row in prepared if row["metadata"]["manifest_role"] == "evaluation_nonoverlap"]),
        (dev_path, [row for row in prepared if row["metadata"]["manifest_role"] == "development_overlap"]),
    ):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in subset),
            encoding="utf-8",
        )

    audit = {
        "protocol": "navsim-level1-v5-overlap-aware-v1",
        "source_manifest": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "input_records": len(rows),
        "all_records": len(prepared),
        "evaluation_nonoverlap_records": len(selected_ids),
        "development_overlap_records": len(rows) - len(selected_ids),
        "scene_count": len(by_scene),
        "evaluation_scene_count": len({row["scene_id"] for row in prepared if row["metadata"]["manifest_role"] == "evaluation_nonoverlap"}),
        "history_frames": 4,
        "future_frames": 8,
        "future_horizon_s": 4.0,
        "future_source": "navsim_native_realized",
        "causal_claim_allowed": False,
        "reason_causal_claim_blocked": "future images are logged realized frames, not WAM-generated futures",
        "files": {
            "all": str(all_path.resolve()),
            "evaluation_nonoverlap": str(eval_path.resolve()),
            "development_overlap": str(dev_path.resolve()),
        },
    }
    (args.output_root / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
