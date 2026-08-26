#!/usr/bin/env python3
"""Audit explicit WAM-to-dataset state join keys.

The audit is deliberately conservative: it reports only exact keys accepted
by ``attach_dataset_states`` and never proposes scene-name or image-similarity
matches as valid joins.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def explicit_keys(row: dict[str, Any]) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    for field in ("video_id", "sample_id", "source_key", "pair_id"):
        if row.get(field) is not None:
            keys.setdefault(field, []).append(f"{field}:{row[field]}")
    if row.get("scene_name") is not None and row.get("timestamp_us") is not None:
        keys.setdefault("scene_timestamp", []).append(
            f"scene_timestamp:{row['scene_name']}:{int(row['timestamp_us'])}"
        )
    if row.get("scene_name") is not None:
        keys.setdefault("scene_name", []).append(f"scene_name:{row['scene_name']}")
    return keys


def audit(wam_rows: list[dict[str, Any]], annotation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    wam_by_key: dict[str, set[int]] = {}
    annotation_by_key: dict[str, set[int]] = {}
    for index, row in enumerate(wam_rows):
        for values in explicit_keys(row).values():
            for key in values:
                wam_by_key.setdefault(key, set()).add(index)
    for index, row in enumerate(annotation_rows):
        for values in explicit_keys(row).values():
            for key in values:
                annotation_by_key.setdefault(key, set()).add(index)
    shared = sorted(set(wam_by_key).intersection(annotation_by_key))
    # A scene name alone is an advisory correlation, not an identity key.
    accepted = [key for key in shared if key.split(":", 1)[0] != "scene_name"]
    matched_wam = {index for key in accepted for index in wam_by_key[key]}
    matched_annotations = {index for key in accepted for index in annotation_by_key[key]}
    ambiguous = [
        {
            "key": key,
            "wam_rows": sorted(wam_by_key[key]),
            "annotation_rows": sorted(annotation_by_key[key]),
        }
        for key in accepted
        if len(annotation_by_key[key]) != 1
    ]
    key_types = Counter(key.split(":", 1)[0] for key in shared)
    return {
        "protocol": "explicit-state-join-audit-v1",
        "wam_rows": len(wam_rows),
        "annotation_rows": len(annotation_rows),
        "wam_key_count": len(wam_by_key),
        "annotation_key_count": len(annotation_by_key),
        "shared_exact_key_count": len(shared),
        "accepted_exact_key_count": len(accepted),
        "shared_key_types": dict(sorted(key_types.items())),
        "accepted_key_types": dict(sorted(Counter(key.split(":", 1)[0] for key in accepted).items())),
        "matched_wam_rows": len(matched_wam),
        "matched_annotation_rows": len(matched_annotations),
        "ambiguous_shared_keys": ambiguous,
        "lineage_join_ready": bool(accepted) and not ambiguous and len(matched_wam) == len(wam_rows),
        "closed_loop_join_ready": bool(accepted) and not ambiguous and len(matched_wam) == len(wam_rows),
        "fcs_join_ready": bool(shared) and not ambiguous and len(matched_wam) == len(wam_rows) and all(
            annotation_rows[index].get("realized_future_ego_state") is not None
            and annotation_rows[index].get("task_success") is not None
            for index in matched_annotations
        ),
        "policy": "exact video_id/sample_id/source_key/pair_id/scene_timestamp only; no scene-name-only or image-similarity joins",
        "unmatched_wam_rows": sorted(set(range(len(wam_rows))) - matched_wam),
        "unmatched_annotation_rows": sorted(set(range(len(annotation_rows))) - matched_annotations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wam", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-fcs", action="store_true")
    args = parser.parse_args()
    report = audit(read_jsonl(args.wam), read_jsonl(args.annotations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_fcs and not report["fcs_join_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
