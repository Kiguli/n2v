"""Concat the running activation with a stored (frozen) skip tensor."""

import torch
import torch.nn as nn


class ConcatWithFrozenSkip(nn.Module):
    """Concatenate a frozen (buffer) skip tensor to the current activation."""

    def __init__(self, skip: torch.Tensor, dim: int = -1):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("skip", skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip
        # Broadcast a feature-shaped skip (e.g. shape (D,) or (1, D)) up to
        # x's batch dimension. For an unmatched leading dim we unsqueeze
        # and expand rather than naively assuming skip.shape[0] is batch.
        if skip.ndim < x.ndim:
            skip = skip.unsqueeze(0)
            skip = skip.expand(x.shape[0], *skip.shape[1:])
        elif skip.shape[0] != x.shape[0]:
            # Same rank but mismatched batch — broadcast across batch.
            skip = skip.expand(x.shape[0], *skip.shape[1:])
        return torch.cat([x, skip], dim=self.dim)
