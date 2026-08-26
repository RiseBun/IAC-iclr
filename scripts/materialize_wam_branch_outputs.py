#!/usr/bin/env python3
"""Attach generated WAM future frames to a native branch manifest.

The branch manifest is the source of truth for lineage and dataset state. A
WAM backend only supplies ``branch_id`` and generated future-image paths; it
must not overwrite history, actions, or realized-state annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _branch_key(row: dict[str, Any]) -> str:
    key = row.get("branch_id")
    if key is None or not str(key):
        raise ValueError("every row must have a non-empty branch_id")
    return str(key)


def _future_images(row: dict[str, Any]) -> list[str]:
    paths = row.get("future_images")
    if paths is None:
        paths = row.get("generated_future_images")
    if paths is None:
        paths = row.get("future_image_paths")
    if not isinstance(paths, list) or not paths or any(not str(path) for path in paths):
        raise ValueError(f"{_branch_key(row)}: generated future_images must be a non-empty list")
    return [str(path) for path in paths]


def materialize(
    branches: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    *,
    check_files: bool = False,
    allow_extra: bool = False,
) -> list[dict[str, Any]]:
    branch_by_key: dict[str, dict[str, Any]] = {}
    for row in branches:
        key = _branch_key(row)
        if key in branch_by_key:
            raise ValueError(f"duplicate branch_id in native manifest: {key}")
        branch_by_key[key] = row

    generated_by_key: dict[str, dict[str, Any]] = {}
    for row in generated:
        key = _branch_key(row)
        if key in generated_by_key:
            raise ValueError(f"duplicate branch_id in WAM output: {key}")
        paths = _future_images(row)
        if check_files:
            missing = [path for path in paths if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"{key}: generated image does not exist: {missing[0]}")
        generated_by_key[key] = row

    missing = sorted(set(branch_by_key) - set(generated_by_key))
    extra = sorted(set(generated_by_key) - set(branch_by_key))
    if missing:
        raise ValueError(f"WAM output is missing {len(missing)} branch(es), first: {missing[0]}")
    if extra and not allow_extra:
        raise ValueError(f"WAM output has {len(extra)} unknown branch(es), first: {extra[0]}")

    output: list[dict[str, Any]] = []
    for row in branches:
        key = _branch_key(row)
        generated_row = generated_by_key[key]
        merged = dict(row)
        merged["future_images"] = _future_images(generated_row)
        merged["future_images_source"] = "wam_generated"
        merged["wam_generation_status"] = "complete"
        if generated_row.get("wam_output_id") is not None:
            merged["wam_output_id"] = generated_row["wam_output_id"]
        if isinstance(generated_row.get("wam_generation_metadata"), dict):
            merged["wam_generation_metadata"] = generated_row["wam_generation_metadata"]
        output.append(merged)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--allow-extra", action="store_true")
    args = parser.parse_args()
    rows = materialize(
        read_jsonl(args.branches),
        read_jsonl(args.generated),
        check_files=args.check_files,
        allow_extra=args.allow_extra,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "protocol": "wam-native-counterfactual-branch-v1",
        "materialization": "generated_future_images_v1",
        "branch_rows": len(rows),
        "output": str(args.output.resolve()),
        "checked_files": bool(args.check_files),
    }, indent=2))


if __name__ == "__main__":
    main()
