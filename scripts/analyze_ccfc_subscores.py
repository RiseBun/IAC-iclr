#!/usr/bin/env python3
"""Extract auditable CCFC direction/magnitude/temporal sub-scores.

The script never recomputes or rescales a CCFC result.  It only exposes the
components already emitted by the frozen scorer, so a low geometric mean can be
diagnosed without tuning the ruler on the benchmark set.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _iter_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(p) for p in glob.glob(value, recursive=True)]
        paths.extend(matches if matches else [Path(value)])
    return sorted(set(path for path in paths if path.is_file()))


def _reports(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    reports = payload.get("reports") or []
    if not isinstance(reports, list):
        return []
    rows: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        group = report.get("counterfactual_group_id")
        cfc = report.get("continuous_cfc") or {}
        if not isinstance(cfc, dict):
            continue
        for scale_mode, result in cfc.items():
            if not isinstance(result, dict):
                continue
            subscores = result.get("subscores") or {}
            rows.append(
                {
                    "source_file": str(path),
                    "counterfactual_group_id": group,
                    "scale_mode": scale_mode,
                    "status": result.get("status"),
                    "score": result.get("score"),
                    "coverage": result.get("coverage"),
                    "evaluable_intervals": result.get("evaluable_intervals"),
                    "total_intervals": result.get("total_intervals"),
                    "response_direction": subscores.get("response_direction"),
                    "response_magnitude": subscores.get("response_magnitude"),
                    "response_temporal_alignment": subscores.get("response_temporal_alignment"),
                }
            )
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="CCFC JSON files or recursive globs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in _iter_paths(args.paths):
        rows.extend(_reports(_load(path), path))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scale_mode"])].append(row)
    summary: dict[str, Any] = {}
    for scale_mode, values in sorted(grouped.items()):
        summary[scale_mode] = {
            "reports": len(values),
            "status_counts": {
                status: sum(row.get("status") == status for row in values)
                for status in sorted({row.get("status") for row in values})
            },
            "score_mean": _mean([float(row["score"]) for row in values if row.get("score") is not None]),
            "coverage_mean": _mean([float(row["coverage"]) for row in values if row.get("coverage") is not None]),
            "subscore_mean": {
                key: _mean([float(row[key]) for row in values if row.get(key) is not None])
                for key in ("response_direction", "response_magnitude", "response_temporal_alignment")
            },
        }

    payload = {
        "protocol": "ccfc-subscore-audit-v1",
        "source_files": [str(path) for path in _iter_paths(args.paths)],
        "rows": rows,
        "summary": summary,
        "note": "Descriptive audit only; no threshold or score was changed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
