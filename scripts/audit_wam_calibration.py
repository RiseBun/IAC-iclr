#!/usr/bin/env python3
"""Audit and optionally augment WAM manifests with camera calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.calibration import CALIBRATION_SCHEMA_VERSION, calibration_payload, calibration_status


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"expected a JSON array or JSONL: {path}")
        return [dict(item) for item in value]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _keys(row: dict[str, Any]) -> set[str]:
    values = {
        row.get("video_id"), row.get("sample_id"), row.get("twin_id"),
        row.get("source_image"),
    }
    for field in ("history_images", "future_images", "history_frame_paths", "future_frame_paths"):
        for value in row.get(field) or []:
            values.add(value)
    return {str(value) for value in values if value is not None and str(value)}


def _ordered_keys(row: dict[str, Any]) -> list[str]:
    """Return deterministic, specific-first keys for calibration matching."""
    ordered: list[str] = []
    for field in ("video_id", "sample_id", "twin_id", "source_image"):
        value = row.get(field)
        if value is not None and str(value):
            ordered.append(str(value))
    for field in ("history_images", "future_images", "history_frame_paths", "future_frame_paths"):
        ordered.extend(str(value) for value in row.get(field) or [] if value is not None and str(value))
    return list(dict.fromkeys(ordered))


def _build_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = row.get("calibration") if isinstance(row.get("calibration"), dict) else row
        for key in _keys(row):
            index[key] = dict(payload)
    return index


def _augment(row: dict[str, Any], index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = dict(row)
    merged = dict(row.get("calibration") or {}) if isinstance(row.get("calibration"), dict) else {}
    matches = [key for key in _ordered_keys(row) if key in index]
    if matches:
        # The first match is the most specific key. Do not merge a second,
        # broader source path entry over it.
        merged = {**index[matches[0]], **merged}
    if merged:
        result["calibration"] = {**merged, "schema": CALIBRATION_SCHEMA_VERSION}
    status = calibration_payload(result)
    result["calibration"] = status
    result["calibration_status"] = calibration_status(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration-index", type=Path)
    parser.add_argument("--output", type=Path, help="write an augmented JSONL manifest")
    args = parser.parse_args()

    rows = _read_records(args.manifest)
    index = _build_index(_read_records(args.calibration_index)) if args.calibration_index else {}
    augmented = [_augment(row, index) for row in rows]
    counts: dict[str, int] = {key: 0 for key in ("complete", "partial", "missing", "invalid")}
    for row in augmented:
        counts[row["calibration_status"]["status"]] += 1
    summary = {
        "protocol": "wam-calibration-audit-v1",
        "manifest": str(args.manifest),
        "calibration_index": str(args.calibration_index) if args.calibration_index else None,
        "schema": CALIBRATION_SCHEMA_VERSION,
        "rows": len(augmented),
        "counts": counts,
        "projectable_fraction": counts["complete"] / len(augmented) if augmented else 0.0,
        "projection_policy": "metric_ego only when complete; image_plane_only otherwise",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in augmented:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        summary["augmented_manifest"] = str(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
