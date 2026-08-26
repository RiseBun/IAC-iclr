#!/usr/bin/env python3
"""Build a mixed, opaque annotation pack from WAM videos and probe outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.measurement_validation import build_blind_annotation_pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("SOURCE_ID", "WAM_MANIFEST", "EVENT_GROUPS"),
        required=True,
        help="repeat for each WAM source; source ids remain only in the private key",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="measurement-validity-v1")
    parser.add_argument("--history-frame-count", type=int, default=2)
    args = parser.parse_args()
    sources = [
        {
            "source_id": source_id,
            "manifest_path": manifest,
            "event_groups_path": groups,
        }
        for source_id, manifest, groups in args.source
    ]
    result = build_blind_annotation_pack(
        sources,
        args.output_dir,
        seed=args.seed,
        history_frame_count=args.history_frame_count,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
