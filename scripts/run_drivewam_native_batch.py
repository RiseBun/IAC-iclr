#!/usr/bin/env python3
"""Run DriveWAM on native NAVSIM samples and decode future frames."""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


def _install_flash_attention_import_fallback() -> bool:
    """Make DriveWAM's optional flash-attn import harmless in torch mode.

    DriveWAM imports ``flash_attn`` unconditionally even when the evaluator is
    explicitly constructed with ``attn_mode='torch'``.  In that mode the model
    uses its own SDPA implementation, so registering an equivalent import-only
    fallback does not change the selected attention path or model weights.
    """
    try:
        import flash_attn  # noqa: F401
        return False
    except ImportError:
        import types
        import importlib.machinery
        import torch.nn.functional as F

        def flash_attn_func(query, key, value, *args, **kwargs):
            return F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            ).transpose(1, 2)

        module = types.ModuleType("flash_attn")
        module.flash_attn_func = flash_attn_func
        module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
        sys.modules["flash_attn"] = module
        interface = types.ModuleType("flash_attn_interface")
        interface.flash_attn_func = flash_attn_func
        interface.__spec__ = importlib.machinery.ModuleSpec("flash_attn_interface", loader=None)
        sys.modules["flash_attn_interface"] = interface
        return True


def _action_trajectory(batch_item, *, mode: str, expected_steps: int) -> torch.Tensor:
    """Build the explicit [T,3] pose condition used by the intervention probe."""
    values = []
    for item in list(batch_item.get("future_trajectory") or []):
        pose = item.get("pose") if isinstance(item, dict) else item
        if pose is not None and len(pose) >= 3:
            values.append([float(pose[0]), float(pose[1]), float(pose[2])])
    if not values:
        raise ValueError("native sample has no future_trajectory pose condition")
    values = np.asarray(values, dtype=np.float32)
    if len(values) < expected_steps:
        values = np.concatenate([values, np.repeat(values[-1:], expected_steps - len(values), axis=0)], axis=0)
    values = values[:expected_steps].copy()
    if mode in {"left", "right"}:
        sign = 1.0 if mode == "left" else -1.0
        progress = np.linspace(0.0, 1.0, expected_steps, dtype=np.float32)
        values[:, 1] += sign * 0.75 * progress
        values[:, 2] += sign * 0.12 * progress
    elif mode == "stop":
        values[:, :2] = 0.0
        values[:, 2] = 0.0
    return torch.from_numpy(values)


def decode_latents(evaluator, latents):
    mean = torch.tensor(evaluator.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(evaluator.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    z = latents * std + mean
    out = evaluator.vae.decode(z).sample
    out = ((out.float() / 2.0) + 0.5).clamp(0, 1)
    return (out * 255.0).round().to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--video-steps", type=int, default=4)
    ap.add_argument("--action-steps", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--intervention",
        choices=("predict", "logged", "left", "right", "stop"),
        default="predict",
        help="predict uses the repository default; other modes explicitly inject a pose condition",
    )
    ap.add_argument("--condition-chunk", type=int, choices=(0, 1), default=0)
    args = ap.parse_args()

    drivewam = os.environ.get("DRIVEWAM_ROOT", str(Path.cwd() / "third_party" / "DriveWAM"))
    tools = os.environ.get("IAC_TOOLS_ROOT", str(Path.cwd() / "tools"))
    iac_src = str(Path(__file__).resolve().parents[1] / "src")
    flash_fallback = _install_flash_attention_import_fallback()
    sys.path.insert(0, tools)
    sys.path.insert(0, drivewam)
    sys.path.insert(0, os.path.join(drivewam, "src"))
    sys.path.insert(0, iac_src)
    from configs import VA_CONFIGS
    from dataset.navsim_dataset import NavSimEpisodeDataset
    from navsim.eval import NavSimEvaluator
    if args.intervention != "predict":
        from iac_new.drivewam_adapter import DriveWAMIntervention, rollout_external_action

    config = VA_CONFIGS["navsim_cfg"]
    config.dataset_path = args.data
    config.wan22_pretrained_model_name_or_path = args.base
    config.enable_wandb = False
    # The 6B transformer and T5 encoder do not fit together on a 24GB card.
    # Keep the encoder on CPU; prompt embeddings are copied to the GPU once.
    config.text_encoder_device = torch.device("cpu")
    evaluator = NavSimEvaluator(config, checkpoint_path=args.checkpoint, attn_mode="torch", device=args.device)
    dataset = NavSimEpisodeDataset(config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(min(len(dataset), args.num_samples)):
        batch_item = dataset[idx]
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in batch_item.items()}
        with torch.inference_mode():
            if args.intervention == "predict":
                pred_actions, pred_latents = evaluator.predict(
                    batch, num_video_steps=args.video_steps, num_action_steps=args.action_steps, seed=idx
                )
                injection_verified = False
            else:
                # NavSimEpisodeDataset returns tensors and the sample path, but
                # intentionally drops the raw pose list.  Reload that immutable
                # native condition so intervention experiments never reconstruct
                # actions from model outputs or from a nearest-neighbour record.
                with open(batch_item["sample_path"], "rb") as sample_file:
                    raw_sample = pickle.load(sample_file)
                action = _action_trajectory(
                    raw_sample,
                    mode=args.intervention,
                    expected_steps=int(config.action_chunk_steps),
                )
                pred_actions, pred_latents = rollout_external_action(
                    evaluator,
                    batch,
                    action,
                    intervention=DriveWAMIntervention(args.condition_chunk),
                    num_video_steps=args.video_steps,
                    num_action_steps=args.action_steps,
                    seed=idx,
                )
                injection_verified = True
            decoded = decode_latents(evaluator, pred_latents)[0].cpu()  # C,T,H,W
        sample_out = out / f"sample_{idx:06d}"
        sample_out.mkdir(exist_ok=True)
        frame_paths = []
        for t in range(1, decoded.shape[1]):
            arr = decoded[:, t].permute(1, 2, 0).numpy()
            p = sample_out / f"future_{t:02d}.png"
            Image.fromarray(arr).save(p)
            frame_paths.append(str(p))
        traj_value = evaluator.actions_to_trajectory(pred_actions)
        traj = traj_value.detach().cpu().numpy().tolist() if torch.is_tensor(traj_value) else np.asarray(traj_value).tolist()
        rows.append({
            "sample_index": idx,
            "source_sample": batch_item["sample_path"],
            "future_images": frame_paths,
            "future_images_source": "drivewam_generated",
            "predicted_action_trajectory": traj,
            "branch_mode": args.intervention,
            "intervention_variant": (
                "repository_predict" if args.intervention == "predict"
                else f"external_pose_condition_chunk_{args.condition_chunk}"
            ),
            "action_injection_verified": injection_verified,
            "latent_shape": list(pred_latents.shape),
        })
        print(json.dumps(rows[-1])[:1200], flush=True)
        del pred_actions, pred_latents, decoded
        torch.cuda.empty_cache()
    (out / "manifest.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"output": str(out), "num_samples": len(rows)}, indent=2))
    if flash_fallback:
        print(json.dumps({"attention_import_fallback": "torch_sdpa", "model_attn_mode": "torch"}))


if __name__ == "__main__":
    main()
