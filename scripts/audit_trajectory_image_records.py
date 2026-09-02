#!/usr/bin/env python3
"""Audit strict records used for trajectory-image consistency evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.wam_record import validate_trajectory_image_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors = []
    valid = 0
    for line_number, line in enumerate(args.records.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            validate_trajectory_image_record(json.loads(line))
            valid += 1
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            errors.append({"line": line_number, "error": str(error)})
    summary = {
        "protocol": "wam-trajectory-image-record-v1",
        "records": valid + len(errors),
        "valid": valid,
        "invalid": len(errors),
        "errors": errors,
        "status": "ok" if not errors else "invalid",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

