#!/usr/bin/env python3
"""Audit candidate WAM repositories before launching expensive rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.wam_adapters import inspect_known_wams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row.to_dict() for row in inspect_known_wams(args.home)]
    by_tier = {}
    for row in rows:
        by_tier.setdefault(row["evaluation_tier"], []).append(row["model_id"])
    payload = {
        "protocol": "iac-wam-capability-audit-v1",
        "selection_policy": "capability_tier_then_runtime_status",
        "formal_level2_candidates": [row["model_id"] for row in rows if row["formal_level2_eligible"]],
        "by_evaluation_tier": by_tier,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
