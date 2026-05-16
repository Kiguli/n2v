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
        if skip.shape[0] != x.shape[0]:
            skip = skip.expand(x.shape[0], *skip.shape[1:])
        return torch.cat([x, skip], dim=self.dim)
