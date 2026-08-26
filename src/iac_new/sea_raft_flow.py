"""SEA-RAFT flow adapter used as a drop-in IAC optical-flow backend."""

from __future__ import annotations

import sys
import os
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from .flow import FlowObservation, forward_backward_mask
from .geometry import scale_intrinsics


class SeaRaftFlowExtractor:
    def __init__(self, *, checkpoint: str, device: str, iters: int = 12) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.iters = int(iters)
        sea_root = Path(checkpoint).resolve().parents[1] / "SEA-RAFT"
        if not sea_root.exists():
            configured_root = os.environ.get("SEA_RAFT_ROOT")
            if configured_root is None:
                raise FileNotFoundError(
                    "SEA-RAFT source tree not found; set SEA_RAFT_ROOT"
                )
            sea_root = Path(configured_root)
        sys.path.insert(0, str(sea_root))
        sys.path.insert(0, str(sea_root / "core"))
        from raft import RAFT
        from safetensors.torch import load_file

        args = Namespace(
            use_var=True, var_min=0.0, var_max=10.0, pretrain="resnet34",
            initial_dim=64, block_dims=[64, 128, 256], radius=4, dim=128,
            num_blocks=2, iters=self.iters, image_size=[432, 960], scale=0,
            batch_size=1, epsilon=1e-8, lr=1e-4, wdecay=1e-5, dropout=0,
            clip=1.0, gamma=0.85, num_steps=10000, restore_ckpt=None,
            coarse_config=None, skip_pretrain=True,
        )
        model = RAFT(args)
        state = load_file(str(checkpoint), device="cpu")
        model.load_state_dict(state, strict=False)
        self.model = model.to(device).eval()

    @staticmethod
    def _read_images(paths: list[str], target_size: tuple[int, int]) -> list[np.ndarray]:
        width, height = target_size
        images = []
        for path in paths:
            image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return images

    def _infer_pairs(self, first: list[np.ndarray], second: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        outputs = []
        for left, right in zip(first, second):
            image1 = torch.from_numpy(left.copy()).permute(2, 0, 1).float()[None].to(self.device)
            image2 = torch.from_numpy(right.copy()).permute(2, 0, 1).float()[None].to(self.device)
            with torch.inference_mode():
                prediction = self.model(image1, image2, iters=self.iters, test_mode=True)["flow"][-1]
            outputs.append(prediction[0].permute(1, 2, 0).float().cpu().numpy())
        return np.stack(outputs, axis=0).astype(np.float32)

    def observe(
        self,
        frame_paths: list[str],
        intrinsics: np.ndarray,
        distortion: np.ndarray,
        target_size: tuple[int, int],
        allow_mixed_source_sizes: bool = False,
    ) -> FlowObservation:
        if len(frame_paths) < 2:
            raise ValueError("at least two video frames are required")
        images = self._read_images(frame_paths, target_size)
        forward = self._infer_pairs(images[:-1], images[1:])
        backward = self._infer_pairs(images[1:], images[:-1])
        width, height = target_size
        masks = np.stack([
            forward_backward_mask(fwd, bwd, absolute_threshold_px=1.5, relative_threshold=0.05)
            for fwd, bwd in zip(forward, backward)
        ], axis=0)
        source = cv2.imread(str(Path(frame_paths[0])), cv2.IMREAD_COLOR)
        source_size = (int(source.shape[1]), int(source.shape[0]))
        scaled = scale_intrinsics(np.asarray(intrinsics, dtype=np.float64), source_size, target_size)
        return FlowObservation(
            forward=forward, consistency_masks=masks, intrinsics=scaled,
            source_size=source_size, target_size=target_size,
        )
