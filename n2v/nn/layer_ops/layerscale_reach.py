"""LayerScale reachability: ``y = gamma * x`` with learnable per-channel ``gamma``.

This is elementwise affine, so the implementation routes through
:mod:`linear_reach` via a diagonal-Linear surrogate. Coverage matches
nnVLA's elementwise affine layers: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _gamma(layer) -> np.ndarray:
    return layer.gamma.detach().cpu().numpy().astype(np.float64).reshape(-1)


def _make_diag_linear(gamma: np.ndarray) -> nn.Linear:
    n = gamma.size
    dummy = nn.Linear(n, n, bias=False)
    with torch.no_grad():
        dummy.weight.copy_(torch.from_numpy(np.diag(gamma)).float())
    return dummy


def layerscale_star(layer, input_stars: List[Star]) -> List[Star]:
    g = _gamma(layer)
    return linear_reach.linear_star(_make_diag_linear(g), input_stars)


def layerscale_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    g = _gamma(layer)
    return linear_reach.linear_zono(_make_diag_linear(g), input_zonos)


def layerscale_box(layer, input_boxes: List[Box]) -> List[Box]:
    g = _gamma(layer)
    return linear_reach.linear_box(_make_diag_linear(g), input_boxes)


def layerscale_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    g = _gamma(layer)
    return linear_reach.linear_hexatope(_make_diag_linear(g), input_sets)


def layerscale_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    g = _gamma(layer)
    return linear_reach.linear_octatope(_make_diag_linear(g), input_sets)
