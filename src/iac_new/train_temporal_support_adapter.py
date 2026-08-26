#!/usr/bin/env python3
"""Train/evaluate a lightweight temporal support head on NAVSIM."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from iac_new.dino_features import DINOv2TemporalConsistency
from iac_new.temporal_support_adapter import TemporalSupportAdapter, denormalize_support, support_target_from_candidates


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _embed_rows(rows: list[dict], encoder: DINOv2TemporalConsistency, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings, targets = [], []
    for index, row in enumerate(rows):
        cache = cache_dir / f"{index:04d}.npz"
        if cache.exists():
            values = np.load(cache)
            embedding = values["embedding"]
            target = values["target"]
        else:
            paths = list(row["history_frame_paths"]) + list(row["future_frame_paths"])
            embedding = encoder.embed_global(paths)
            candidates = [candidate for candidate in row["candidates"] if candidate.get("feasibility_label") in {"known_valid", "plausible"}]
            target = support_target_from_candidates(candidates).numpy()
            np.savez_compressed(cache, embedding=embedding, target=target)
        embeddings.append(embedding)
        targets.append(target)
        print(json.dumps({"embedded": index + 1, "total": len(rows)}), flush=True)
    return np.stack(embeddings), np.stack(targets)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = denormalize_support(torch.from_numpy(prediction)).numpy()
    truth = denormalize_support(torch.from_numpy(target)).numpy()
    center_error = np.abs(pred[..., [0, 2, 4]] - truth[..., [0, 2, 4]])
    half_error = np.abs(pred[..., [1, 3, 5]] - truth[..., [1, 3, 5]])
    # Coverage is computed on the support target itself, not only the center.
    lo = pred[..., [0, 2, 4]] - pred[..., [1, 3, 5]]
    hi = pred[..., [0, 2, 4]] + pred[..., [1, 3, 5]]
    truth_lo = truth[..., [0, 2, 4]] - truth[..., [1, 3, 5]]
    truth_hi = truth[..., [0, 2, 4]] + truth[..., [1, 3, 5]]
    coverage = ((truth_lo >= lo) & (truth_hi <= hi)).mean()
    return {
        "center_mae": float(center_error.mean()),
        "lateral_center_mae_m": float(center_error[..., 0].mean()),
        "heading_center_mae_rad": float(center_error[..., 1].mean()),
        "curvature_center_mae_1pm": float(center_error[..., 2].mean()),
        "mean_half_width": float(pred[..., [1, 3, 5]].mean()),
        "support_interval_coverage": float(coverage),
        "p95_center_error": float(np.quantile(center_error, 0.95)),
        "p95_lateral_error_m": float(np.quantile(center_error[..., 0], 0.95)),
        "p95_curvature_error_1pm": float(np.quantile(center_error[..., 2], 0.95)),
        "mean_half_width_error": float(half_error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows = _rows(args.manifest)
    scenes = sorted({str(row["scene_id"]) for row in rows})
    if len(scenes) < 2:
        raise ValueError("scene-level holdout needs at least two scenes")
    train_scenes = set(scenes[: len(scenes) // 2])
    train_rows = [row for row in rows if str(row["scene_id"]) in train_scenes]
    holdout_rows = [row for row in rows if str(row["scene_id"]) not in train_scenes]
    encoder = DINOv2TemporalConsistency(device=args.device, model_name="dinov2_vits14")
    train_x, train_y = _embed_rows(train_rows, encoder, args.cache_dir / "train")
    holdout_x, holdout_y = _embed_rows(holdout_rows, encoder, args.cache_dir / "holdout")
    torch_device = torch.device(args.device)
    model = TemporalSupportAdapter(embedding_dim=int(train_x.shape[-1])).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    x = torch.from_numpy(train_x).to(torch_device)
    y = torch.from_numpy(train_y).to(torch_device)
    model.train()
    history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = F.smooth_l1_loss(prediction, y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    with torch.inference_mode():
        train_prediction = model(torch.from_numpy(train_x).to(torch_device)).cpu().numpy()
        holdout_prediction = model(torch.from_numpy(holdout_x).to(torch_device)).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "embedding_dim": int(train_x.shape[-1]), "train_scenes": sorted(train_scenes)}, args.output.with_suffix(".pt"))
    report = {
        "protocol": "navsim-temporal-support-adapter-v1",
        "manifest": str(args.manifest.resolve()),
        "train_scenes": sorted(train_scenes),
        "holdout_scenes": [scene for scene in scenes if scene not in train_scenes],
        "num_train": len(train_rows), "num_holdout": len(holdout_rows),
        "epochs": int(args.epochs), "embedding_dim": int(train_x.shape[-1]),
        "final_train_loss": history[-1],
        "train": _metrics(train_prediction, train_y),
        "holdout": _metrics(holdout_prediction, holdout_y),
        "note": "DINOv2 is frozen; only the GRU/MLP temporal support adapter is trained."
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
