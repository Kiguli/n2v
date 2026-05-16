"""MixFFN (SegFormer): Linear -> 3x3 depthwise conv -> GELU -> Linear."""

import torch
import torch.nn as nn


class MixFFN(nn.Module):
    """SegFormer Mix-FFN block (channel-mixing + spatial mixing)."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else 4 * dim)
        self.fc1 = nn.Linear(self.dim, self.hidden_dim)
        self.dwconv = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, groups=self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) sequence form expected
        x = self.fc1(x)
        b, l, d = x.shape
        h = w = int(l ** 0.5)
        x_2d = x.transpose(1, 2).reshape(b, d, h, w)
        x_2d = self.dwconv(x_2d)
        x = x_2d.flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)
