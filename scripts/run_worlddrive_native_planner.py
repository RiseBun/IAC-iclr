#!/usr/bin/env python3
"""Export WorldDrive's native stage-1 and stage-2 actions from fixed histories.

This script loads only the published planner/refiner checkpoint and the
CogVideoX VAE used by the official feature builder.  It does not read future
images or candidate actions when predicting.  Logged future trajectories are
stored only as post-hoc references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLCogVideoX


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _status_tensor(sample: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ego = sample["ego_status"]
    velocity = ego.get("ego_velocity", ego.get("velocity"))
    acceleration = ego.get("ego_acceleration", ego.get("acceleration"))
    if velocity is None or acceleration is None:
        raise ValueError(f"ego status lacks velocity/acceleration fields: {sorted(ego)}")
    parts = [
        np.asarray(ego["driving_command"], dtype=np.float32).reshape(-1),
        np.asarray(velocity, dtype=np.float32).reshape(-1),
        np.asarray(acceleration, dtype=np.float32).reshape(-1),
    ]
    status = np.concatenate(parts)
    if status.shape != (8,):
        raise ValueError(f"expected 8-D ego status, got {status.shape}")
    return torch.from_numpy(status).unsqueeze(0).to(device=device, dtype=dtype)


def _prefix_state(state: dict, prefix: str) -> dict:
    selected = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise RuntimeError(f"checkpoint contains no keys under {prefix!r}")
    return selected


def _load_modules(
    *,
    checkpoint: Path,
    anchors_path: Path,
    cogvideox_root: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    from navsim.agents.worlddrive.worlddrive_adapters import Adapters
    from navsim.agents.worlddrive.worlddrive_planner import TrajEncoder, TrajWorldPlanner
    from navsim.agents.worlddrive.worlddrive_refiner import TrajWorldRefiner

    anchors = np.load(anchors_path).astype(np.float32)
    if anchors.shape != (256, 8, 3):
        raise ValueError(f"expected anchors [256,8,3], got {anchors.shape}")

    modules = {
        "trajencoder": TrajEncoder(traj_vocab=anchors),
        "trajencoder_wm": TrajEncoder(traj_vocab=anchors),
        "adapters": Adapters(inchannel_size=32, hidden_size=64),
        "trajplanner": TrajWorldPlanner(with_wm_proj=True, gt_version=1),
        "trajrefiner": TrajWorldRefiner(),
    }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = payload["state_dict"]
    for name, module in modules.items():
        module.load_state_dict(_prefix_state(state, f"agent.{name}."), strict=True)
        module.eval().to(device=device, dtype=dtype)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    del payload, state

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
    return anchors, modules, vae


def _native_action(
    *,
    visual: torch.Tensor,
    status: torch.Tensor,
    anchors: np.ndarray,
    modules: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    trajencoder = modules["trajencoder"]
    trajencoder_wm = modules["trajencoder_wm"]
    adapters = modules["adapters"]
    planner = modules["trajplanner"]
    refiner = modules["trajrefiner"]

    visual_latents = adapters(visual)
    batch_size = visual_latents.shape[0]
    trajectory_embeddings = trajencoder(traj_train=True)[None].repeat(batch_size, 1, 1)
    anchor_tensor = torch.from_numpy(anchors).to(device=device, dtype=dtype)
    stage1, final_rewards, offsets, (visual_tokens, _) = planner(
        visual_latents,
        status,
        trajectory_embeddings,
        anchor_tensor,
        targets=None,
        eval_mode=True,
    )

    topk = 5
    topk_rewards, topk_indices = final_rewards.topk(k=topk, dim=1)
    selected = trajencoder_wm.traj_vocab[topk_indices]
    normalized = trajencoder_wm.norm_odo(selected)
    vocab_embeddings = trajencoder_wm.traj_vocab_encoder(normalized.reshape(batch_size, topk, -1))
    batch_indices = torch.arange(batch_size, device=device)[:, None]
    selected_offsets = offsets[batch_indices, topk_indices]
    offset_embeddings = trajencoder_wm.traj_offset_encoder(selected_offsets.reshape(batch_size, topk, -1))
    topk_trajectories = selected + selected_offsets.reshape(batch_size, topk, 8, 3)
    topk_embeddings = vocab_embeddings + offset_embeddings

    stage2, _, _, refine_weights = refiner(
        topk_trajectories,
        topk_embeddings,
        visual_tokens,
        None,
    )
    selected_stage2_rank = int(refine_weights[0, :, 0].argmax().item())
    return {
        "stage1_trajectory": stage1[0].float().cpu().tolist(),
        "stage2_trajectory": stage2[0].float().cpu().tolist(),
        "topk_trajectories": topk_trajectories[0].float().cpu().tolist(),
        "topk_anchor_indices": topk_indices[0].cpu().tolist(),
        "topk_stage1_rewards": topk_rewards[0].float().cpu().tolist(),
        "topk_refine_weights": refine_weights[0, :, 0].float().cpu().tolist(),
        "stage2_selected_topk_rank": selected_stage2_rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlddrive-root", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--cogvideox-root", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError("samples must be positive")
    for path in (args.worlddrive_root, args.planner_checkpoint, args.cogvideox_root, args.anchors):
        if not path.exists():
            raise FileNotFoundError(path)
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

    rows = []
    with torch.inference_mode():
        for sample_index in range(args.samples):
            sample_path = args.sample_root / "drivewam_samples_logged" / f"sample_{sample_index:06d}.pkl"
            sample = _read_sample(sample_path)
            torch.manual_seed(args.seed + sample_index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + sample_index)

            history = _history_tensor(sample["images"], device, dtype)
            visual = vae.encode(history).latent_dist.sample() * vae.config.scaling_factor
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            status = _status_tensor(sample, device, dtype)
            action = _native_action(
                visual=visual,
                status=status,
                anchors=anchors,
                modules=modules,
                device=device,
                dtype=dtype,
            )
            logged = np.asarray(
                [row["pose"] for row in sample["future_trajectory"]], dtype=np.float32
            )
            rows.append({
                "sample_id": f"worlddrive-native-{sample_index:06d}",
                "sample_index": sample_index,
                "source_key": sample["metadata"].get("source_key", f"sample-{sample_index:06d}"),
                "history_sha256": _history_digest(sample["images"]),
                "ego_status": status[0].float().cpu().tolist(),
                "native_action_head_recorded": True,
                "action_origin": "worlddrive_stage2_refined_planner",
                "planner_checkpoint": str(args.planner_checkpoint.resolve()),
                "planner_checkpoint_sha256": checkpoint_sha256,
                "anchors_sha256": anchors_sha256,
                "vae_seed": args.seed + sample_index,
                "logged_reference_trajectory": logged.tolist(),
                **action,
            })
            del history, visual, status
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "samples": len(rows),
        "native_action_head_recorded": True,
        "planner_checkpoint_sha256": checkpoint_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
