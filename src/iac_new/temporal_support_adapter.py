"""Small trainable temporal head for image-side trajectory support."""

from __future__ import annotations

import torch
from torch import nn


class TemporalSupportAdapter(nn.Module):
    """Predict interval support from frozen frame embeddings.

    Input is ``[batch, frames, embedding_dim]``.  The first four frames are
    history and the final four are the future clip, but the recurrent state is
    shared across all frames so the head can use long- and short-term context.
    Each output interval contains six normalized values:
    ``lateral_center, lateral_half_width, heading_center, heading_half_width,
    curvature_center, curvature_half_width``.
    """

    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 128, intervals: int = 4) -> None:
        super().__init__()
        self.intervals = int(intervals)
        self.input_norm = nn.LayerNorm(int(embedding_dim))
        self.temporal = nn.GRU(int(embedding_dim), int(hidden_dim), batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 6),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[1] < self.intervals:
            raise ValueError("embeddings must have shape [B,T,C] with enough future intervals")
        sequence, _ = self.temporal(self.input_norm(embeddings))
        outputs = self.head(sequence[:, -self.intervals :])
        # Half-widths must be non-negative; build a new tensor to keep
        # autograd free of in-place view mutations.
        centers = outputs[..., 0::2]
        half_widths = torch.nn.functional.softplus(outputs[..., 1::2])
        return torch.stack((centers, half_widths), dim=-1).reshape_as(outputs)


def support_target_from_candidates(candidates: list[dict], intervals: int = 4) -> torch.Tensor:
    """Build a normalized support target from NAVSIM plausible candidates."""
    import numpy as np

    trajectories = []
    for candidate in candidates:
        trajectory = np.asarray(candidate.get("trajectory"), dtype=np.float32)
        if trajectory.shape != (intervals, 3):
            continue
        trajectories.append(trajectory)
    if not trajectories:
        raise ValueError("candidate bank has no valid trajectories")
    values = np.stack(trajectories, axis=0)
    x = values[..., 0]
    yaw = values[..., 2]
    curvature = np.zeros_like(yaw)
    curvature[:, 0] = yaw[:, 0] / np.maximum(x[:, 0], 1.0)
    dx = np.maximum(x[:, 1:] - x[:, :-1], 0.5)
    curvature[:, 1:] = np.diff(yaw, axis=1) / dx
    components = (values[..., 1], yaw, curvature)
    scales = (5.0, 0.8, 0.1)
    target = []
    for component, scale in zip(components, scales):
        lo = np.min(component, axis=0) / scale
        hi = np.max(component, axis=0) / scale
        target.extend([0.5 * (lo + hi), 0.5 * (hi - lo)])
    # [center/half per component, intervals] -> [intervals, six fields].
    packed = np.stack(target, axis=0).T
    return torch.from_numpy(packed.astype(np.float32))


def denormalize_support(prediction: torch.Tensor) -> torch.Tensor:
    scales = prediction.new_tensor([5.0, 5.0, 0.8, 0.8, 0.1, 0.1])
    return prediction * scales
