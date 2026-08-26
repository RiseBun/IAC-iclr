#!/usr/bin/env python3
"""Score actor-relative estimates against independent metric annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.relative_motion import evaluate_relative_motion_metrics


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dangerous-ttc-s", type=float, default=4.0)
    args = parser.parse_args()

    report = evaluate_relative_motion_metrics(
        read_jsonl(args.records),
        dangerous_ttc_s=args.dangerous_ttc_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
