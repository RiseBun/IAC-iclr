#!/usr/bin/env python3
"""Swap candidate-blind decoded futures across paired roles while keeping actions fixed."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role-a", default="clear")
    parser.add_argument("--role-b", default="risk")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("counterfactual_group_id"))].append(row)
    output = []
    for group_id, branches in sorted(grouped.items()):
        roles = {str(row.get("branch_role")): row for row in branches}
        if set(roles) != {args.role_a, args.role_b} or len(branches) != 2:
            raise ValueError(f"{group_id}: expected exactly {args.role_a}/{args.role_b}")
        first = copy.deepcopy(roles[args.role_a])
        second = copy.deepcopy(roles[args.role_b])
        first["decoder"], second["decoder"] = second["decoder"], first["decoder"]
        for row in (first, second):
            row["specificity_control"] = "paired_image_decoder_swap"
            row["sample_id"] = f"{row['sample_id']}::image_swap_control"
        output.extend((first, second))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "groups": len(grouped),
        "records": len(output),
        "control": "paired_image_decoder_swap",
        "pair_roles": [args.role_a, args.role_b],
    }, indent=2))


if __name__ == "__main__":
    main()
