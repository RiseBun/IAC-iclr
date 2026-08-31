#!/usr/bin/env python3
"""Generate WorldDrive pixel futures from its recorded native stage-2 actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from run_worlddrive_action_response import (
    _history_digest,
    _history_tensor,
    _load_vae,
    _load_world_model,
    _read_sample,
    _save_future,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_actions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"native action manifest is empty: {path}")
    for row in rows:
        if not row.get("native_action_head_recorded"):
            raise ValueError(f"row is not a recorded native action: {row.get('sample_id')}")
        trajectory = np.asarray(row.get("stage2_trajectory"), dtype=np.float32)
        if trajectory.shape != (8, 3) or not np.isfinite(trajectory).all():
            raise ValueError(f"invalid native trajectory for {row.get('sample_id')}: {trajectory.shape}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlddrive-root", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--cogvideox-root", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--native-actions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("steps must be positive")
    for path in (
        args.worlddrive_root,
        args.world_model_checkpoint,
        args.cogvideox_root,
        args.anchors,
        args.native_actions,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(args.worlddrive_root.resolve()))

    rows = _read_actions(args.native_actions)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config, world_model, trajectory_encoder, adapters = _load_world_model(
        worlddrive_root=args.worlddrive_root,
        world_model_checkpoint=args.world_model_checkpoint,
        cogvideox_root=args.cogvideox_root,
        anchors_path=args.anchors,
        device=device,
        dtype=dtype,
    )
    vae = _load_vae(args.cogvideox_root, device, dtype)
    world_model_sha256 = _sha256(args.world_model_checkpoint)

    output_rows = []
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else nullcontext
    with inference_context():
        for ordinal, native in enumerate(rows):
            sample_index = int(native.get("sample_index", ordinal))
            sample_path = args.sample_root / "drivewam_samples_logged" / f"sample_{sample_index:06d}.pkl"
            sample = _read_sample(sample_path)
            history_sha256 = _history_digest(sample["images"])
            if history_sha256 != native["history_sha256"]:
                raise RuntimeError(f"history mismatch for native action {native['sample_id']}")

            item_seed = args.seed + sample_index
            torch.manual_seed(item_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(item_seed)
            history = _history_tensor(sample["images"], device, dtype)
            posterior = vae.encode(history).latent_dist
            visual = posterior.sample() * vae.config.scaling_factor
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            condition = adapters(visual)
            _, _, channels, height, width = condition.shape
            latent_frames = config.predict_frames // config.temporal_compression_ratio + 3
            generator = torch.Generator(device=device).manual_seed(item_seed)
            shared_noise = torch.randn(
                (1, latent_frames, channels, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )

            action = np.asarray(native["stage2_trajectory"], dtype=np.float32)
            action_tensor = torch.from_numpy(action).unsqueeze(0).to(device=device, dtype=dtype)
            action_embedding = trajectory_encoder(action_tensor, traj_train=False)
            generated = world_model.step_eval(
                condition,
                shared_noise,
                action_embedding,
                no_grad=True,
                guidance_scale=1,
                timesteps=args.steps,
            )
            future_paths = _save_future(
                latents=generated,
                vae=vae,
                output_dir=args.output_root / f"sample_{sample_index:06d}" / "native",
            )
            output_rows.append({
                "sample_id": native["sample_id"],
                "sample_index": sample_index,
                "source_key": native["source_key"],
                "history_sha256": history_sha256,
                "action_trajectory": action.tolist(),
                "action_origin": native["action_origin"],
                "native_action_head_recorded": True,
                "planner_checkpoint": native["planner_checkpoint"],
                "planner_checkpoint_sha256": native["planner_checkpoint_sha256"],
                "future_images": future_paths,
                "future_timestamps": [0.5 * index for index in range(1, 9)],
                "future_images_source": "worlddrive_generated",
                "model_id": "worlddrive_stage2_native_tadwm_1024",
                "world_model_checkpoint": str(args.world_model_checkpoint.resolve()),
                "world_model_checkpoint_sha256": world_model_sha256,
                "seed": item_seed,
                "diffusion_steps": args.steps,
                "action_injection_verified": True,
                "evidence_tier": "native_action_future_pair",
                "causal_claim_eligible": False,
            })
            del history, posterior, visual, condition, shared_noise, action_embedding, generated
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest.resolve()),
        "samples": len(output_rows),
        "native_action_head_recorded": True,
        "causal_claim_eligible": False,
    }, indent=2))


if __name__ == "__main__":
    main()
