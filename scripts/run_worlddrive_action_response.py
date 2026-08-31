#!/usr/bin/env python3
"""Generate fixed-noise WorldDrive futures for left/logged/right interventions.

This is deliberately a pre-CCFC action-response probe.  It exercises the
published trajectory-conditioned pixel world model without reading future
images or using the action candidate inside the image scorer.  The native
WorldDrive planner is audited separately before formal Level-2 admission.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler


BRANCHES = ("left", "logged", "right")


def _read_sample(path: Path) -> dict:
    with path.open("rb") as stream:
        sample = pickle.load(stream)
    required = {"images", "ego_status", "future_trajectory", "metadata"}
    missing = required.difference(sample)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    return sample


def _history_digest(images: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(images[:4]).tobytes()).hexdigest()


def _trajectory(sample: dict) -> np.ndarray:
    trajectory = np.asarray([row["pose"] for row in sample["future_trajectory"]], dtype=np.float32)
    if trajectory.shape != (8, 3):
        raise ValueError(f"expected 8x3 action trajectory, got {trajectory.shape}")
    return trajectory


def _history_tensor(images: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    frames = []
    for image in images[:4]:
        cropped = image[28:-28]
        frames.append(cv2.resize(cropped, (1024, 512), interpolation=cv2.INTER_AREA))
    history = np.asarray(frames, dtype=np.float32)
    history = np.concatenate([np.zeros_like(history), history], axis=0)
    tensor = torch.from_numpy(history).to(device=device, dtype=dtype)
    tensor = tensor.div_(255.0).sub_(0.5).mul_(2.0)
    return tensor.unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous()


def _load_world_model(
    *,
    worlddrive_root: Path,
    world_model_checkpoint: Path,
    cogvideox_root: Path,
    anchors_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    from navsim.agents.worlddrive import worlddrive_generator as generator_module
    from navsim.agents.worlddrive.worlddrive_adapters import Adapters
    from navsim.agents.worlddrive.worlddrive_generator import TrajWorldModel, TrajWorldModelConfig
    from navsim.agents.worlddrive.worlddrive_planner import TrajEncoder

    scheduler = CogVideoXDPMScheduler.from_pretrained(cogvideox_root, subfolder="scheduler")
    config = TrajWorldModelConfig(image_size_width=512, image_size_height=1024)
    with patch.object(generator_module.CogVideoXDPMScheduler, "from_pretrained", return_value=scheduler):
        world_model = TrajWorldModel(config)
    scale_model_input = world_model.scheduler.scale_model_input
    world_model.scheduler.scale_model_input = (
        lambda sample, timestep: scale_model_input(sample, timestep).to(dtype=dtype)
    )

    anchors = np.load(anchors_path)
    trajectory_encoder = TrajEncoder(traj_vocab=anchors)
    adapters = Adapters(inchannel_size=config.vae_embed_dim * 2, hidden_size=config.vae_embed_dim * 4)

    payload = torch.load(world_model_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = payload["model_state_dict"]
    world_state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if key.startswith("module.transformer.")
    }
    encoder_state = {
        key.removeprefix("module.traj_encoder."): value
        for key, value in state.items()
        if key.startswith("module.traj_encoder.")
    }
    adapter_state = {
        key.removeprefix("module.adapters."): value
        for key, value in state.items()
        if key.startswith("module.adapters.")
    }
    world_model.load_state_dict(world_state, strict=True)
    missing, unexpected = trajectory_encoder.load_state_dict(encoder_state, strict=False)
    if unexpected or set(missing).difference({"traj_vocab"}):
        raise RuntimeError(f"trajectory encoder mismatch: missing={missing}, unexpected={unexpected}")
    adapters.load_state_dict(adapter_state, strict=True)
    del payload, state, world_state, encoder_state, adapter_state
    gc.collect()

    for module in (world_model, trajectory_encoder, adapters):
        module.eval().to(device=device, dtype=dtype)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return config, world_model, trajectory_encoder, adapters


def _load_vae(cogvideox_root: Path, device: torch.device, dtype: torch.dtype):
    vae = AutoencoderKLCogVideoX.from_pretrained(
        cogvideox_root,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.eval()
    vae.enable_slicing()
    vae.enable_tiling()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def _save_future(
    *,
    latents: torch.Tensor,
    vae,
    output_dir: Path,
) -> list[str]:
    scaling = vae.config.scaling_factor
    prediction = latents[:, 2:].permute(0, 2, 1, 3, 4).contiguous() / scaling
    prediction = prediction.to(dtype=next(vae.parameters()).dtype)
    decoded = vae.decode(prediction).sample
    decoded = decoded.permute(0, 2, 3, 4, 1)[0].float().cpu().clamp(-1, 1)
    frames = ((decoded + 1.0) * 127.5).round().byte().numpy()
    if len(frames) == 9:
        frames = frames[1:]
    if len(frames) != 8:
        raise RuntimeError(f"expected 8 future frames after alignment, got {len(frames)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, rgb in enumerate(frames, start=1):
        path = output_dir / f"frame_{index:02d}.png"
        if not cv2.imwrite(str(path), rgb[..., ::-1]):
            raise RuntimeError(f"failed to write {path}")
        paths.append(str(path.resolve()))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlddrive-root", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--cogvideox-root", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    if args.samples < 1 or args.steps < 1:
        raise ValueError("samples and steps must be positive")
    for path in (args.worlddrive_root, args.world_model_checkpoint, args.cogvideox_root, args.anchors):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(args.worlddrive_root.resolve()))

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

    rows = []
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else nullcontext
    with inference_context():
        for sample_index in range(args.samples):
            samples = {
                branch: _read_sample(
                    args.sample_root / f"drivewam_samples_{branch}" / f"sample_{sample_index:06d}.pkl"
                )
                for branch in BRANCHES
            }
            digests = {_history_digest(sample["images"]) for sample in samples.values()}
            if len(digests) != 1:
                raise RuntimeError(f"sample {sample_index}: branch histories differ")

            history = _history_tensor(samples["logged"]["images"], device, dtype)
            posterior = vae.encode(history).latent_dist
            visual = posterior.sample() * vae.config.scaling_factor
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            condition = adapters(visual)
            _, _, channels, height, width = condition.shape
            latent_frames = config.predict_frames // config.temporal_compression_ratio + 3
            generator = torch.Generator(device=device).manual_seed(args.seed + sample_index)
            shared_noise = torch.randn(
                (1, latent_frames, channels, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )

            group_id = f"worlddrive-{sample_index:06d}"
            for branch in BRANCHES:
                action = _trajectory(samples[branch])
                action_tensor = torch.from_numpy(action).unsqueeze(0).to(device=device, dtype=dtype)
                action_embedding = trajectory_encoder(action_tensor, traj_train=False)
                generated = world_model.step_eval(
                    condition,
                    shared_noise.clone(),
                    action_embedding,
                    no_grad=True,
                    guidance_scale=1,
                    timesteps=args.steps,
                )
                future_paths = _save_future(
                    latents=generated,
                    vae=vae,
                    output_dir=args.output_root / f"sample_{sample_index:06d}" / branch,
                )
                rows.append({
                    "counterfactual_group_id": group_id,
                    "source_key": samples[branch]["metadata"].get("source_key", group_id),
                    "branch_mode": branch,
                    "history_sha256": next(iter(digests)),
                    "action_trajectory": action.tolist(),
                    "future_images": future_paths,
                    "future_timestamps": [0.5 * index for index in range(1, 9)],
                    "future_images_source": "worlddrive_generated",
                    "model_id": "worlddrive_tadwm_1024",
                    "checkpoint": str(args.world_model_checkpoint.resolve()),
                    "seed": args.seed + sample_index,
                    "diffusion_steps": args.steps,
                    "action_injection_verified": True,
                    "native_action_head_recorded": False,
                    "evidence_tier": "action_response_probe",
                })
                del generated, action_embedding
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del history, posterior, visual, condition, shared_noise
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest.resolve()),
        "samples": args.samples,
        "branches": len(rows),
        "steps": args.steps,
        "native_action_head_recorded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
