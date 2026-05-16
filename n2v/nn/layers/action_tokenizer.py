"""Action tokenizer: discretises a continuous action vector into bins."""

import torch
import torch.nn as nn


class ActionTokenizer(nn.Module):
    """Bin a continuous action vector ``a`` into integer token IDs.

    Uses uniform binning across each action dimension between
    ``min_action`` and ``max_action``.
    """

    def __init__(self, action_dim: int, n_bins: int, min_action: float = -1.0, max_action: float = 1.0):
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_bins = int(n_bins)
        self.min_action = float(min_action)
        self.max_action = float(max_action)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        clipped = a.clamp(self.min_action, self.max_action)
        normalised = (clipped - self.min_action) / (self.max_action - self.min_action)
        binned = (normalised * (self.n_bins - 1)).round().long()
        return binned
