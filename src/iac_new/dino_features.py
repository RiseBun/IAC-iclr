"""Lightweight DINOv2 temporal feature consistency for flow weighting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class DINOv2TemporalConsistency:
    """Compute soft semantic consistency after an optical-flow warp.

    DINOv2 is used only as an auxiliary reliability signal. The trajectory
    geometry remains estimated from RAFT flow; low semantic consistency lowers
    a pixel's weight instead of hard-masking it.
    """

    def __init__(
        self,
        *,
        device: str,
        model_name: str = "dinov2_vits14",
        hub_dir: str | None = None,
        weight_floor: float = 0.25,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.weight_floor = float(np.clip(weight_floor, 0.0, 1.0))
        if hub_dir is None:
            torch_home = Path.home() / ".cache" / "torch"
            cached = torch_home / "hub" / "facebookresearch_dinov2_main"
            hub_dir = str(cached) if cached.exists() else None
        # The server's NAVSIM environment is Python 3.9 while the cached
        # upstream hub checkout uses Python 3.10 union annotations. timm's
        # compatible DINOv2 implementation avoids that import mismatch.
        try:
            import timm
            self.model = timm.create_model(
                "vit_small_patch14_dinov2" if model_name == "dinov2_vits14" else model_name,
                pretrained=False,
                img_size=518,
            )
            checkpoint = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
            if checkpoint.exists() and model_name == "dinov2_vits14":
                self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
        except Exception:
            if hub_dir and Path(hub_dir).exists():
                self.model = torch.hub.load(hub_dir, model_name, source="local")
            else:
                self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.mean = mean
        self.std = std

    def _read(self, paths: list[str], target_size: tuple[int, int]) -> np.ndarray:
        images: list[np.ndarray] = []
        for path in paths:
            image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not images:
            raise ValueError("DINOv2 requires at least one frame")
        return np.stack(images, axis=0)

    def _features(self, images: np.ndarray) -> np.ndarray:
        torch = self.torch
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        tensor = torch.nn.functional.interpolate(tensor, size=(518, 518), mode="bilinear", align_corners=False)
        tensor = (tensor - self.mean) / self.std
        with torch.inference_mode():
            try:
                tokens = self.model.get_intermediate_layers(tensor, n=1, return_class_token=False)[0]
            except TypeError:
                tokens = self.model.get_intermediate_layers(tensor, n=1, return_prefix_tokens=False, norm=True)[0]
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise RuntimeError(f"unexpected DINOv2 patch token count: {tokens.shape}")
        features = tokens.reshape(len(images), side, side, tokens.shape[-1]).permute(0, 3, 1, 2)
        return features.float().cpu().numpy()

    def embed_global(
        self,
        frame_paths: list[str],
        *,
        target_size: tuple[int, int] = (256, 144),
    ) -> np.ndarray:
        """Return one frozen, normalized DINO embedding per frame."""
        images = self._read(frame_paths, target_size)
        spatial = self._features(images)
        pooled = spatial.mean(axis=(2, 3))
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norm, 1e-8)).astype(np.float32)

    def embed_global(
        self,
        frame_paths: list[str],
        *,
        target_size: tuple[int, int] = (256, 144),
    ) -> np.ndarray:
        """Return one frozen, normalized DINO embedding per frame.

        This is intentionally a small adapter interface: the backbone remains
        frozen and callers can train a lightweight temporal head on top of the
        frame-level embeddings without coupling the evaluator to DINO internals.
        """
        images = self._read(frame_paths, target_size)
        spatial = self._features(images)
        pooled = spatial.mean(axis=(2, 3))
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norm, 1e-8)).astype(np.float32)

    def observe(
        self,
        frame_paths: list[str],
        observed_flows: np.ndarray,
        *,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Return [T,H,W] semantic consistency weights in [weight_floor,1]."""
        flows = np.asarray(observed_flows, dtype=np.float32)
        if flows.ndim != 4 or flows.shape[-1] != 2:
            raise ValueError("observed_flows must have shape [T,H,W,2]")
        if len(frame_paths) != len(flows) + 1:
            raise ValueError("DINOv2 frames must contain one more frame than flows")
        width, height = target_size
        images = self._read(frame_paths, target_size)
        features = self._features(images)
        tensor = self.torch.from_numpy(features).to(self.device)
        tensor = self.torch.nn.functional.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        dense = tensor.permute(0, 2, 3, 1).float().cpu().numpy()
        yy, xx = np.indices((height, width), dtype=np.float32)
        weights = []
        for index, flow in enumerate(flows):
            map_x = xx + flow[..., 0]
            map_y = yy + flow[..., 1]
            next_feature = np.stack([
                cv2.remap(dense[index + 1, ..., channel], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                for channel in range(dense.shape[-1])
            ], axis=-1)
            current = dense[index]
            finite = np.isfinite(next_feature).all(axis=-1) & np.isfinite(current).all(axis=-1)
            cosine = np.sum(current * next_feature, axis=-1)
            norm = np.linalg.norm(next_feature, axis=-1) * np.linalg.norm(current, axis=-1)
            cosine = cosine / np.maximum(norm, 1e-6)
            consistency = np.clip(0.5 * (cosine + 1.0), 0.0, 1.0)
            consistency = np.where(finite, consistency, 0.0)
            weights.append(self.weight_floor + (1.0 - self.weight_floor) * consistency)
        return np.stack(weights, axis=0).astype(np.float32)
