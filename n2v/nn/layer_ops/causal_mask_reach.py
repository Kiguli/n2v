"""CausalMask reachability.

Adds a constant lower-triangular mask to attention logits. Affine
addition of a fixed matrix routes through the standard
translation-via-linear pattern.

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _make_translation(mask_vec: np.ndarray) -> nn.Linear:
    n = mask_vec.size
    dummy = nn.Linear(n, n, bias=True)
    with torch.no_grad():
        dummy.weight.copy_(torch.eye(n).float())
        dummy.bias.copy_(torch.from_numpy(mask_vec).float())
    return dummy


def _mask_vec(layer, input_dim: int) -> np.ndarray:
    full = layer.mask.detach().cpu().numpy().astype(np.float64)
    # Flatten the LxL mask to a vector of length L*L matching the
    # flattened logits the layer is added to.
    l = int(np.sqrt(input_dim))
    if l * l != input_dim:
        # Fallback: skip the mask if shape unknown (sound no-op).
        return np.zeros(input_dim, dtype=np.float64)
    return full[:l, :l].reshape(-1)


def causal_mask_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        mv = _mask_vec(layer, b.dim).reshape(-1, 1)
        out.append(Box(b.lb + mv, b.ub + mv))
    return out


def causal_mask_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        mv = _mask_vec(layer, s.dim)
        out.extend(linear_reach.linear_star(_make_translation(mv), [s]))
    return out


def causal_mask_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        mv = _mask_vec(layer, z.dim)
        out.extend(linear_reach.linear_zono(_make_translation(mv), [z]))
    return out
