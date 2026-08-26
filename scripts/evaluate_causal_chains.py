#!/usr/bin/env python3
"""Audit and score the four interaction-level IAC causal chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.causal_chains import evaluate_causal_chain_records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-contrast", type=float, default=0.20)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    result = evaluate_causal_chain_records(
        read_jsonl(args.records),
        minimum_contrast=args.minimum_contrast,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {
        "protocol": result["protocol"],
        "num_pairs": result["num_pairs"],
        "num_evaluable_pairs": result["num_evaluable_pairs"],
        "counterfactual_coverage": result["counterfactual_coverage"],
        "four_chain_suite_ready": result["four_chain_suite_ready"],
        "macro_mean_causal_chain_score": result["macro_mean_causal_chain_score"],
        "overall": result["overall"],
    }
    print(json.dumps(summary, indent=2))
    if args.require_ready and not result["four_chain_suite_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
