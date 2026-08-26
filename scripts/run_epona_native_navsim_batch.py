#!/usr/bin/env python3
"""Run Epona on native NAVSIM camera frames as a cross-model protocol smoke."""
import argparse, json, os, pickle, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--sensor-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    epona = os.environ.get("EPONA_ROOT", str(Path.cwd() / "third_party" / "Epona"))
    sys.path.insert(0, epona)
    from models.model import TrainTransformersDiT
    from models.modules.tokenizer import VAETokenizer

    # Epona's bundled Config parser has a missing helper in this checkout;
    # the config is plain Python, so execute it into a namespace directly.
    from types import SimpleNamespace
    ns = {}
    exec(Path(f"{epona}/configs/dit_config_dcae_nuplan.py").read_text(), ns)
    cfg = SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})
    cfg.batch_size = 1
    cfg.vae_ckpt = args.vae
    cfg.resume_path = args.checkpoint
    cfg.num_sampling_steps = 4
    cfg.condition_frames = 10
    cfg.image_size = (512, 1024)
    cfg.temporal_patch_size = 6
    cfg.test_video_frames = 4
    cfg.device = args.device
    torch.cuda.set_device(torch.device(args.device))
    model = TrainTransformersDiT(cfg, load_path=args.checkpoint, local_rank=0, condition_frames=10).eval()
    tokenizer = VAETokenizer(cfg, 0)

    frames = pickle.load(open(args.pkl, "rb"))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, start in enumerate(range(0, len(frames) - 15, 5)):
        if idx >= args.num_samples: break
        window = frames[start:start + 15]
        anchor = window[9]
        imgs = []
        for f in window[:10]:
            rel = f["cams"]["CAM_F0"]["data_path"]
            p = Path(args.sensor_root) / rel
            im = cv2.imread(str(p))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (1024, 512), interpolation=cv2.INTER_AREA)
            imgs.append(torch.from_numpy(im).permute(2,0,1))
        images = torch.stack(imgs).float() / 255.0
        images = (images - 0.5) * 2.0
        # Epona consumes successive ego-frame pose/yaw conditions.
        abs_poses = torch.tensor([np.asarray(f["ego2global"], dtype=np.float32) for f in window[:11]])
        inv0 = torch.linalg.inv(abs_poses[9])
        abs_poses = inv0 @ abs_poses
        # step_eval expects one pose/yaw per latent frame plus one target
        # condition, i.e. 11 states for 10 image latents.
        rel = abs_poses[:11]
        pose = rel[:, :2, 3].unsqueeze(0)
        yaw = torch.atan2(rel[:, 1, 0], rel[:, 0, 0]).unsqueeze(0).unsqueeze(-1)
        images = images.unsqueeze(0).to(args.device)
        pose, yaw = pose.to(args.device), yaw.to(args.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latents = tokenizer.encode_to_z(images)
            pred_traj, pred_latent = model.step_eval(latents, pose, yaw, self_pred_traj=True)
            decoded = tokenizer.z_to_image(pred_latent, is_video=False)[0].float().cpu()
        p = out / f"sample_{idx:06d}_future.png"
        arr = (decoded.permute(1,2,0).numpy() * 255).clip(0,255).astype(np.uint8)
        cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        rows.append({"sample_index": idx, "frame_idx": int(anchor.get("frame_idx", start+9)), "future_image": str(p), "future_images_source": "epona_generated", "predicted_trajectory": np.asarray(pred_traj.detach().cpu()).tolist()})
        print(json.dumps(rows[-1])[:800], flush=True)
        del latents, pred_traj, pred_latent, decoded
        torch.cuda.empty_cache()
    (out / "manifest.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"output": str(out), "num_samples": len(rows)}, indent=2))


if __name__ == "__main__": main()
