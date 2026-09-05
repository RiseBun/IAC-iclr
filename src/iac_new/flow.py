"""Torchvision RAFT wrapper with optional forward-backward filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import scale_intrinsics


@dataclass(frozen=True)
class FlowObservation:
    forward: np.ndarray
    consistency_masks: np.ndarray | None
    intrinsics: np.ndarray
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    refinement_uncertainty: np.ndarray | None = None
    long_range_residual: np.ndarray | None = None


def forward_backward_mask(
    forward: np.ndarray,
    backward: np.ndarray,
    *,
    absolute_threshold_px: float,
    relative_threshold: float,
) -> np.ndarray:
    """Return pixels whose forward flow agrees with warped backward flow."""
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[2] != 2:
        raise ValueError(f"flow shape mismatch: {forward.shape} vs {backward.shape}")
    height, width = forward.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x = (xx + forward[..., 0]).astype(np.float32)
    map_y = (yy + forward[..., 1]).astype(np.float32)
    warped_backward = cv2.remap(
        backward.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    residual = np.linalg.norm(forward + warped_backward, axis=2)
    magnitude = np.linalg.norm(forward, axis=2) + np.linalg.norm(warped_backward, axis=2)
    threshold = float(absolute_threshold_px) + float(relative_threshold) * magnitude
    in_frame = (map_x >= 0.0) & (map_x <= width - 1) & (map_y >= 0.0) & (map_y <= height - 1)
    return in_frame & np.isfinite(residual) & (residual <= threshold)


def long_range_flow_residual(direct_flow: np.ndarray, two_step_flow: np.ndarray) -> np.ndarray:
    """Return the diagnostic discrepancy from a constant-rate two-step proxy."""
    direct = np.asarray(direct_flow, dtype=np.float32)
    two_step = np.asarray(two_step_flow, dtype=np.float32)
    if direct.shape != two_step.shape or direct.ndim != 3 or direct.shape[-1] != 2:
        raise ValueError("direct_flow and two_step_flow must have matching [H,W,2] shapes")
    return np.linalg.norm(direct - 0.5 * two_step, axis=-1).astype(np.float32)


class RaftFlowExtractor:
    def __init__(
        self,
        *,
        model_size: str,
        device: str,
        updates: int,
        batch_size: int,
        forward_backward: bool,
        fb_abs_threshold_px: float,
        fb_relative_threshold: float,
        checkpoint: str | None = None,
    ) -> None:
        import torch
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        if model_size == "large":
            model = raft_large(weights=None if checkpoint else Raft_Large_Weights.DEFAULT, progress=False)
        elif model_size == "small":
            model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
        else:
            raise ValueError("model_size must be 'small' or 'large'")
        if checkpoint:
            state = torch.load(str(checkpoint), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError("RAFT checkpoint must contain a state dict")
            model.load_state_dict(state, strict=True)
        self.torch = torch
        self.model = model.to(device).eval()
        self.model_size = model_size
        self.device = device
        self.updates = int(updates)
        self.batch_size = int(batch_size)
        self.use_forward_backward = bool(forward_backward)
        self.fb_abs_threshold_px = float(fb_abs_threshold_px)
        self.fb_relative_threshold = float(fb_relative_threshold)

    @staticmethod
    def _read_images(
        paths: list[str],
        intrinsics: np.ndarray,
        distortion: np.ndarray,
        target_size: tuple[int, int],
        allow_mixed_source_sizes: bool = False,
    ) -> tuple[list[np.ndarray], np.ndarray, tuple[int, int]]:
        images: list[np.ndarray] = []
        source_size: tuple[int, int] | None = None
        for path in paths:
            image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            height, width = image.shape[:2]
            if source_size is None:
                source_size = (width, height)
            elif source_size != (width, height) and not allow_mixed_source_sizes:
                raise ValueError("all frames in a video must have the same dimensions")
            if distortion.size:
                image = cv2.undistort(image, intrinsics, distortion, None, intrinsics)
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if source_size is None:
            raise ValueError("video has no frames")
        return images, scale_intrinsics(intrinsics, source_size, target_size), source_size

    def _infer_pairs(
        self, first: list[np.ndarray], second: list[np.ndarray],
        *, return_uncertainty: bool = False, uncertainty_tail: int = 8,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if len(first) != len(second):
            raise ValueError("RAFT pair lists must have equal length")
        outputs: list[np.ndarray] = []
        uncertainties: list[np.ndarray] = []
        torch = self.torch
        for start in range(0, len(first), self.batch_size):
            stop = min(start + self.batch_size, len(first))
            first_tensor = torch.stack(
                [torch.from_numpy(image.copy()).permute(2, 0, 1) for image in first[start:stop]]
            ).to(self.device).float().div_(127.5).sub_(1.0)
            second_tensor = torch.stack(
                [torch.from_numpy(image.copy()).permute(2, 0, 1) for image in second[start:stop]]
            ).to(self.device).float().div_(127.5).sub_(1.0)
            with torch.inference_mode():
                predictions = self.model(
                    first_tensor, second_tensor, num_flow_updates=self.updates
                )
            prediction = predictions[-1]
            outputs.append(prediction.cpu().permute(0, 2, 3, 1).numpy())
            if return_uncertainty:
                tail = predictions[-max(2, min(int(uncertainty_tail), len(predictions))):]
                stack = torch.stack(tail, dim=0)
                spread = torch.sqrt(torch.mean((stack - stack[-1:]) ** 2, dim=0))
                uncertainties.append(spread.cpu().numpy()[:, 0])
        final = np.concatenate(outputs, axis=0).astype(np.float32)
        uncertainty = None if not return_uncertainty else np.concatenate(uncertainties, axis=0).astype(np.float32)
        return final, uncertainty

    def observe(
        self,
        frame_paths: list[str],
        intrinsics: np.ndarray,
        distortion: np.ndarray,
        target_size: tuple[int, int],
        inference_size: tuple[int, int] | None = None,
        allow_mixed_source_sizes: bool = False,
        return_uncertainty: bool = False,
        uncertainty_tail: int = 8,
        long_range_consistency: bool = False,
    ) -> FlowObservation:
        if len(frame_paths) < 2:
            raise ValueError("at least two video frames are required")
        # The decoder coordinate system remains target_size.  An optional
        # larger inference canvas is isolated to RAFT, then every flow and
        # forward/backward mask is mapped back before geometry/scoring.
        inference_size = target_size if inference_size is None else tuple(inference_size)
        images, inference_intrinsics, source_size = self._read_images(
            frame_paths,
            np.asarray(intrinsics, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
            inference_size,
            allow_mixed_source_sizes=allow_mixed_source_sizes,
        )
        forward, forward_uncertainty = self._infer_pairs(
            images[:-1], images[1:], return_uncertainty=return_uncertainty,
            uncertainty_tail=uncertainty_tail,
        )
        long_range_residual = None
        if long_range_consistency and len(images) >= 3:
            # A two-step flow gives an independent temporal check.  For short
            # constant-rate intervals, half of the two-step displacement is a
            # conservative proxy for the first interval.  We export the
            # residual rather than hard-gating pixels; rotation, depth change,
            # and occlusion can all make the proxy invalid.
            skip_forward, _ = self._infer_pairs(images[:-2], images[2:])
            residuals = []
            for direct, skip in zip(forward[:-1], skip_forward):
                residuals.append(long_range_flow_residual(direct, skip))
            # The final interval has no following two-step measurement.  Keep
            # shape compatibility and mark it unavailable with NaNs.
            residuals.append(np.full(forward[-1].shape[:2], np.nan, dtype=np.float32))
            long_range_residual = np.stack(residuals, axis=0)
        masks = None
        backward = None
        if self.use_forward_backward:
            backward, _ = self._infer_pairs(images[1:], images[:-1])
        if inference_size != target_size:
            def resize_flow(value: np.ndarray) -> np.ndarray:
                height, width = target_size[1], target_size[0]
                resized = cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                resized[..., 0] *= width / float(inference_size[0])
                resized[..., 1] *= height / float(inference_size[1])
                return resized
            forward = np.stack([resize_flow(value) for value in forward], axis=0)
            if backward is not None:
                backward = np.stack([resize_flow(value) for value in backward], axis=0)
        if backward is not None:
            masks = np.stack(
                [forward_backward_mask(fwd, bwd, absolute_threshold_px=self.fb_abs_threshold_px,
                                        relative_threshold=self.fb_relative_threshold)
                 for fwd, bwd in zip(forward, backward)], axis=0)
        scaled_intrinsics = scale_intrinsics(
            np.asarray(intrinsics, dtype=np.float64), source_size, target_size
        )
        if inference_size != target_size and long_range_residual is not None:
            long_range_residual = np.stack([
                cv2.resize(value, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
                for value in long_range_residual
            ], axis=0)
        return FlowObservation(
            forward=forward,
            consistency_masks=masks,
            intrinsics=scaled_intrinsics,
            source_size=source_size,
            target_size=target_size,
            refinement_uncertainty=forward_uncertainty,
            long_range_residual=long_range_residual,
        )


def cuda_peak_memory_mb(torch_module: Any) -> float | None:
    if not torch_module.cuda.is_available():
        return None
    return float(torch_module.cuda.max_memory_allocated() / (1024.0**2))
