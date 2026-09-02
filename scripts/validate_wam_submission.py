#!/usr/bin/env python3
"""Fail-closed audit of a WAM submission against the public split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.scorecard import validate_submission


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_submission(_read(args.submission), _read(args.public))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "issues"}, indent=2))
    if report["issues"]:
        print(json.dumps({"issues": report["issues"][:8]}, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
