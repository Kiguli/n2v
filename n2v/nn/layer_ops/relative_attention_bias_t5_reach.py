"""RelativeAttentionBiasT5 reachability.

The forward produces a *fixed* bias tensor that is added to attention
logits downstream. For reachability purposes the layer evaluates to a
constant — its image is a single-point set independent of the input.
Coverage matches nnVLA: Box, Star, Zono (degenerate constant set).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star, Zono


def _bias_value(layer, q_len: int = 0, k_len: int = 0) -> np.ndarray:
    if q_len <= 0 or k_len <= 0:
        return np.zeros(1, dtype=np.float64)
    return layer(q_len, k_len).detach().cpu().numpy().astype(np.float64).reshape(-1)


def relative_attention_bias_t5_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        val = _bias_value(layer).reshape(-1, 1)
        v = np.broadcast_to(val[:1], b.lb.shape).copy()
        out.append(Box(v, v))
    return out


def relative_attention_bias_t5_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        val = np.zeros((s.dim, 1))
        out.append(Star.from_bounds(val, val))
    return out


def relative_attention_bias_t5_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        c = np.zeros((z.dim, 1))
        V = np.zeros((z.dim, 1))
        out.append(Zono(c, V))
    return out
