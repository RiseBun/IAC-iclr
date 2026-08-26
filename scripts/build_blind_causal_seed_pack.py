#!/usr/bin/env python3
"""Build an opaque human-annotation pack for four-chain risk seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.causal_annotation import build_blind_causal_seed_pack, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="iac-causal-seed-blind-v1")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--no-normalize-media", action="store_true")
    args = parser.parse_args()
    result = build_blind_causal_seed_pack(
        read_jsonl(args.candidates),
        args.output_dir,
        seed=args.seed,
        normalize_media=not args.no_normalize_media,
        media_size=(args.width, args.height),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
