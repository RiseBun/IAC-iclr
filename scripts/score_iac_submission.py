#!/usr/bin/env python3
"""Build an IAC capability-stratified scorecard from a validated submission.

Optional CCFC/FAU/FCS cells are ``unavailable`` when their interface or evidence
is absent; the scorer never invents numbers or converts absence into zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.scorecard import build_model_scorecard, frozen_pilot_scorecard, validate_submission


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--measurements", type=Path, help="JSON object of cell measurements")
    parser.add_argument("--frozen-pilots", action="store_true", help="emit the official v1 pilot scoreboard")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frozen_pilots:
        report = frozen_pilot_scorecard()
    else:
        if args.public is None or args.submission is None:
            raise SystemExit("--public and --submission are required unless --frozen-pilots")
        public = _read(args.public)
        rows = _read(args.submission)
        audit = validate_submission(rows, public)
        if not audit["ready"]:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(2)
        capability = str(rows[0].get("capability") or "")
        model_id = str(rows[0].get("wam_model_id") or "unknown")
        measurements = {}
        if args.measurements is not None:
            measurements = json.loads(args.measurements.read_text(encoding="utf-8"))
        report = {
            "protocol": "iac-scorecard-v1",
            "benchmark": "benchmark_v1",
            "submission_rows": len(rows),
            "models": [
                build_model_scorecard(
                    model_id=model_id,
                    capability=capability,
                    measurements=measurements,
                )
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
