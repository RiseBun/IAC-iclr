#!/usr/bin/env python3
"""Export native WorldDrive actions under same-history command interventions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from run_worlddrive_native_planner import (
    _history_digest,
    _history_tensor,
    _load_modules,
    _native_action,
    _read_sample,
    _sha256,
    _status_tensor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlddrive-root", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--cogvideox-root", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--command-indices", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    command_indices = tuple(dict.fromkeys(args.command_indices))
    if not command_indices or any(index not in range(4) for index in command_indices):
        raise ValueError("command indices must be unique values from 0 to 3")
    sys.path.insert(0, str(args.worlddrive_root.resolve()))
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    anchors, modules, vae = _load_modules(
        checkpoint=args.planner_checkpoint,
        anchors_path=args.anchors,
        cogvideox_root=args.cogvideox_root,
        device=device,
        dtype=dtype,
    )
    checkpoint_sha256 = _sha256(args.planner_checkpoint)
    anchors_sha256 = _sha256(args.anchors)
    output_rows = []
    with torch.inference_mode():
        for sample_index in range(args.samples):
            sample_path = args.sample_root / "drivewam_samples_logged" / f"sample_{sample_index:06d}.pkl"
            sample = _read_sample(sample_path)
            item_seed = args.seed + sample_index
            torch.manual_seed(item_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(item_seed)
            history = _history_tensor(sample["images"], device, dtype)
            visual = vae.encode(history).latent_dist.sample() * vae.config.scaling_factor
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            original_status = _status_tensor(sample, device, dtype)
            original_command = original_status[0, :4].float().cpu().tolist()
            group_id = f"worlddrive-command-{sample_index:06d}"

            for command_index in command_indices:
                status = original_status.clone()
                status[:, :4] = 0
                status[:, command_index] = 1
                native = _native_action(
                    visual=visual,
                    status=status,
                    anchors=anchors,
                    modules=modules,
                    device=device,
                    dtype=dtype,
                )
                output_rows.append({
                    "sample_id": f"{group_id}::command_{command_index}",
                    "sample_index": sample_index,
                    "source_key": sample["metadata"].get("source_key", group_id),
                    "counterfactual_group_id": group_id,
                    "branch_id": f"command_{command_index}",
                    "branch_role": f"command_index_{command_index}",
                    "intervention_type": "navigation_command_onehot",
                    "command_index": command_index,
                    "command_onehot": status[0, :4].float().cpu().tolist(),
                    "original_command_onehot": original_command,
                    "history_sha256": _history_digest(sample["images"]),
                    "ego_status": status[0].float().cpu().tolist(),
                    "native_action_head_recorded": True,
                    "native_selected_action": True,
                    "action_origin": "worlddrive_stage2_refined_planner",
                    "planner_checkpoint": str(args.planner_checkpoint.resolve()),
                    "planner_checkpoint_sha256": checkpoint_sha256,
                    "anchors_sha256": anchors_sha256,
                    "vae_seed": item_seed,
                    **native,
                })
            del history, visual, original_status
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "groups": args.samples,
        "rows": len(output_rows),
        "command_indices": list(command_indices),
        "same_history_within_group": True,
        "native_selected_action": True,
    }, indent=2))


if __name__ == "__main__":
    main()
