"""AddWithFrozenSkip reachability: ``y = x + s`` for a constant skip ``s``.

Adding a constant to a set is the same as an affine map with weight=I,
bias=skip, so this routes through :mod:`linear_reach`.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _skip_vec(layer) -> np.ndarray:
    return layer.skip.detach().cpu().numpy().astype(np.float64).reshape(-1)


def _make_translation(skip: np.ndarray) -> nn.Linear:
    n = skip.size
    dummy = nn.Linear(n, n, bias=True)
    with torch.no_grad():
        dummy.weight.copy_(torch.eye(n).float())
        dummy.bias.copy_(torch.from_numpy(skip).float())
    return dummy


def add_with_frozen_skip_star(layer, input_stars: List[Star]) -> List[Star]:
    return linear_reach.linear_star(_make_translation(_skip_vec(layer)), input_stars)


def add_with_frozen_skip_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return linear_reach.linear_zono(_make_translation(_skip_vec(layer)), input_zonos)


def add_with_frozen_skip_box(layer, input_boxes: List[Box]) -> List[Box]:
    return linear_reach.linear_box(_make_translation(_skip_vec(layer)), input_boxes)
