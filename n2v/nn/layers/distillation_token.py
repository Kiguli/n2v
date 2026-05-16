"""Distillation token (DeiT): a second learnable token concatenated after CLS."""

import torch
import torch.nn as nn


class DistillationToken(nn.Module):
    """Prepend a learnable distillation token to an ``(B, L, D)`` sequence."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tok = self.token.expand(b, -1, -1)
        return torch.cat([tok, x], dim=1)
