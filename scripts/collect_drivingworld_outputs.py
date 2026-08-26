#!/usr/bin/env python3
"""Collect numbered DrivingWorld outputs into the immutable branch protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(rows: list[dict[str, Any]], *, generated_root: Path | None = None, experiment_name: str | None = None, require_files: bool = True) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        branch_id = str(row.get("branch_id") or "")
        if not branch_id:
            raise ValueError("every input row needs branch_id")
        input_dir = Path(str(row.get("input_dir") or ""))
        count = int(row.get("future_action_targets", 0))
        if not input_dir.is_dir() or count < 1:
            raise ValueError(f"{branch_id}: invalid input_dir or future target count")
        if generated_root is None:
            output_dir = input_dir
        else:
            output_dir = generated_root / (experiment_name or "") / f"sliding_{len(output)}"
        paths = [str((output_dir / f"{index}.png").resolve()) for index in range(15, 15 + count)]
        if require_files and any(not Path(path).is_file() for path in paths):
            missing = next(path for path in paths if not Path(path).is_file())
            raise FileNotFoundError(f"{branch_id}: generated output missing: {missing}")
        output.append({
            "branch_id": branch_id,
            "generated_future_images": paths,
            "wam_output_id": f"drivingworld:{branch_id}",
            "wam_generation_metadata": {
                "backend": "DrivingWorld",
                "context_frames": 15,
                "future_frames": count,
                "source_input_dir": str(input_dir.resolve()),
                "generated_output_dir": str(output_dir.resolve()),
                "placeholder_policy": row.get("placeholder_policy"),
            },
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    input_rows = read_jsonl(args.inputs)
    if args.max_records:
        input_rows = input_rows[: args.max_records]
    rows = collect(input_rows, generated_root=args.generated_root, experiment_name=args.experiment_name, require_files=not args.allow_missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"protocol": "drivingworld-output-manifest-v1", "branches": len(rows), "output": str(args.output.resolve()), "checked_files": not args.allow_missing}, indent=2))


if __name__ == "__main__":
    main()
