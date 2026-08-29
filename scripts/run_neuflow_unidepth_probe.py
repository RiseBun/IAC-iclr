#!/usr/bin/env python3
"""Run a small, reproducible NeuFlow + UniDepth-L forward-progress probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_flow(backend: str, root: Path | None, checkpoint: Path | None, device: str):
    import torch

    if backend == "raft_large":
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        return raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval(), torch
    if backend != "neuflow":
        raise ValueError("flow backend must be neuflow or raft_large")
    if root is None or checkpoint is None:
        raise ValueError("neuflow-root and neuflow-checkpoint are required for NeuFlow")
    sys.path.insert(0, str(root))
    from NeuFlow.backbone_v7 import ConvBlock
    from NeuFlow.neuflow import NeuFlow

    model = NeuFlow().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=True)

    def fuse(conv, bn):
        fused = torch.nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                                conv.stride, conv.padding, conv.dilation, conv.groups,
                                bias=True).requires_grad_(False).to(conv.weight.device)
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        return fused

    for module in model.modules():
        if type(module) is ConvBlock:
            module.conv1 = fuse(module.conv1, module.norm1)
            module.conv2 = fuse(module.conv2, module.norm2)
            delattr(module, "norm1")
            delattr(module, "norm2")
            module.forward = module.forward_fuse
    model.eval().half()
    model.init_bhwd(1, 432, 768, device)
    return model, torch


def _load_depth(model_name: str, device: str):
    import torch
    from unidepth.models import UniDepthV2

    model = UniDepthV2.from_pretrained(model_name).to(device).eval()
    model.interpolation_mode = "bilinear"
    return model, torch


def _image(path: str, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _estimate_interval(flow: np.ndarray, depth: np.ndarray, K: np.ndarray, camera_to_ego: np.ndarray) -> tuple[float, str]:
    """Estimate metric ego progress from 3D-2D correspondences on the road."""
    h, w = depth.shape
    yy, xx = np.indices((h, w))
    roi = (yy > int(0.50 * h)) & (yy < int(0.96 * h))
    # Down-weight image borders and retain robust finite, positive depth.
    roi &= (xx > int(0.10 * w)) & (xx < int(0.90 * w))
    valid = roi & np.isfinite(flow[..., 1]) & np.isfinite(depth) & (depth > 1.0) & (depth < 120.0)
    if int(valid.sum()) < 100:
        return float("nan"), "insufficient_support"
    ys, xs = np.where(valid)
    if len(xs) > 2500:
        keep = np.linspace(0, len(xs) - 1, 2500, dtype=np.int64)
        xs, ys = xs[keep], ys[keep]
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    points = (np.linalg.inv(K) @ pixels) * depth[ys, xs].reshape(1, -1)
    image_points = np.stack([xs + flow[ys, xs, 0], ys + flow[ys, xs, 1]], axis=1).astype(np.float64)
    finite = np.all(np.isfinite(points), axis=0) & np.all(np.isfinite(image_points), axis=1)
    points = points[:, finite].T.astype(np.float32)
    image_points = image_points[finite].astype(np.float32)
    if len(points) < 100:
        return float("nan"), "insufficient_support"
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points, image_points, K.astype(np.float64), None,
        iterationsCount=100, reprojectionError=3.0, confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 50:
        return float("nan"), "pnp_failed"
    rotation, _ = cv2.Rodrigues(rvec)
    camera_translation = -(rotation.T @ tvec.reshape(3))
    ego_translation = np.asarray(camera_to_ego, dtype=np.float64)[:3, :3] @ camera_translation
    return float(abs(ego_translation[0])), "pnp"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from unidepth.utils.camera import Pinhole

    rows = _read_jsonl(args.manifest)[: args.limit]
    flow_model, flow_torch = _load_flow(args.flow_backend, args.neuflow_root, args.neuflow_checkpoint, args.device)
    depth_model, depth_torch = _load_depth(args.depth_model, args.device)
    estimates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        paths = row["future_frame_paths"]
        times = np.asarray(row["future_times_s"], dtype=np.float64)
        K = np.asarray(row["intrinsics"], dtype=np.float64)
        source_w = cv2.imread(paths[0], cv2.IMREAD_COLOR).shape[1]
        source_h = cv2.imread(paths[0], cv2.IMREAD_COLOR).shape[0]
        K[0] *= 768.0 / source_w
        K[1] *= 432.0 / source_h
        camera = Pinhole(K=depth_torch.from_numpy(K).float().unsqueeze(0).to(args.device))
        predicted: list[float] = []
        methods: list[str] = []
        for first, second in zip(paths[:-1], paths[1:]):
            bgr0, rgb0 = _image(first, width=768, height=432)
            bgr1, rgb1 = _image(second, width=768, height=432)
            if args.flow_backend == "neuflow":
                flow0 = flow_torch.from_numpy(bgr0).permute(2, 0, 1).unsqueeze(0).to(args.device).half()
                flow1 = flow_torch.from_numpy(bgr1).permute(2, 0, 1).unsqueeze(0).to(args.device).half()
                flow_args = (flow0, flow1)
            else:
                flow0 = flow_torch.from_numpy(rgb0).permute(2, 0, 1).unsqueeze(0).to(args.device).float()
                flow1 = flow_torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).to(args.device).float()
                flow_args = (flow0.div(127.5).sub(1.0), flow1.div(127.5).sub(1.0))
            with flow_torch.inference_mode():
                flow = flow_model(*flow_args, num_flow_updates=32)[-1][0].float().permute(1, 2, 0).cpu().numpy() if args.flow_backend == "raft_large" else flow_model(*flow_args)[-1][0].float().permute(1, 2, 0).cpu().numpy()
                depth_pred = depth_model.infer(
                    depth_torch.from_numpy(rgb0).permute(2, 0, 1), camera
                )["depth"].squeeze().float().cpu().numpy()
            estimate, method = _estimate_interval(flow, depth_pred, K, np.asarray(row["camera_to_ego"], dtype=np.float64))
            predicted.append(estimate)
            methods.append(method)
        predicted = np.asarray(predicted, dtype=np.float64)
        gt = np.asarray(
            row.get("realized_future_ego_state", row.get("metadata", {}).get("realized_future_ego_state", [])),
            dtype=np.float64,
        )
        gt_progress = np.diff(gt[:, 0]) if gt.ndim == 2 and len(gt) >= 2 else np.full(len(predicted), np.nan)
        common = np.isfinite(predicted) & np.isfinite(gt_progress)
        if not common.any():
            continue
        # Scale is intentionally not fit per sample: this is an out-of-domain probe.
        history = np.asarray(row.get("history_ego_state", row.get("metadata", {}).get("history_ego_state", [])), dtype=np.float64)
        history_speed = float(history[-1, 3]) if history.ndim == 2 and history.shape[0] and history.shape[1] > 3 else float("nan")
        dt = np.diff(times)[: len(predicted)] if len(times) >= 2 else np.full(len(predicted), np.nan)
        history_progress = history_speed * dt
        history_common = common & np.isfinite(history_progress)
        estimates.append({
            "sample_id": row.get("sample_id"),
            "n_intervals": int(common.sum()),
            "predicted_progress_m": predicted.tolist(),
            "estimation_methods": methods,
            "gt_progress_m": gt_progress.tolist(),
            "progress_mae_m": float(np.mean(np.abs(predicted[common] - gt_progress[common]))),
            "progress_sign_accuracy": float(np.mean(np.sign(predicted[common]) == np.sign(gt_progress[common]))),
            "history_progress_mae_m": (
                float(np.mean(np.abs(history_progress[history_common] - gt_progress[history_common])))
                if history_common.any() else None
            ),
        })
        print(f"[{index + 1}/{len(rows)}] {row.get('sample_id')} mae={estimates[-1]['progress_mae_m']:.3f}", flush=True)
    report = {
        "protocol": "navsim-neuflow-unidepth-l-forward-progress-probe-v1",
        "manifest": str(args.manifest.resolve()),
        "depth_model": args.depth_model,
        "flow_backend": args.flow_backend,
        "flow_checkpoint": str(args.neuflow_checkpoint),
        "num_requested": len(rows),
        "num_evaluable": len(estimates),
        "mean_progress_mae_m": float(np.mean([r["progress_mae_m"] for r in estimates])) if estimates else None,
        "median_progress_mae_m": float(np.median([r["progress_mae_m"] for r in estimates])) if estimates else None,
        "p90_progress_mae_m": float(np.quantile([r["progress_mae_m"] for r in estimates], 0.90)) if estimates else None,
        "mean_progress_sign_accuracy": float(np.mean([r["progress_sign_accuracy"] for r in estimates])) if estimates else None,
        "mean_history_progress_mae_m": float(np.mean([r["history_progress_mae_m"] for r in estimates if r["history_progress_mae_m"] is not None])) if estimates else None,
        "history_comparison": "last observed ego speed multiplied by each 0.5s interval; diagnostic only",
        "results": estimates,
        "formal_metric_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flow-backend", choices=("neuflow", "raft_large"), default="neuflow")
    parser.add_argument("--neuflow-root", type=Path)
    parser.add_argument("--neuflow-checkpoint", type=Path)
    parser.add_argument("--depth-model", default="lpiccinelli/unidepth-v2-vitl14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
