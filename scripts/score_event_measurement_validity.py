#!/usr/bin/env python3
"""Score a frozen image-event probe against blinded human annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.measurement_validation import read_records, score_measurement_validity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-annotators", type=int, default=3)
    parser.add_argument("--consensus-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--minimum-probe-observability", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    annotations = [
        row for path in args.annotations for row in read_records(path)
    ]
    result = score_measurement_validity(
        read_records(args.private_key),
        annotations,
        minimum_annotators=args.minimum_annotators,
        consensus_fraction=args.consensus_fraction,
        minimum_probe_observability=args.minimum_probe_observability,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "num_items": result["num_items"],
        "num_annotations": result["num_annotations"],
        "num_scored_intervals": result["num_scored_intervals"],
        "human_agreement": result["human_agreement"],
        "probe_metrics": result["probe_metrics"],
        "per_source": result["per_source"],
    }, indent=2))


if __name__ == "__main__":
    main()
