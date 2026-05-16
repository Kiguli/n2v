"""RelativePositionBiasTable (Swin) reachability.

Like the T5 bias the forward produces a fixed bias tensor; its
reachable image is a single-point (constant) set.
Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star, Zono


def relative_position_bias_table_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        v = np.zeros_like(b.lb)
        out.append(Box(v, v))
    return out


def relative_position_bias_table_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        val = np.zeros((s.dim, 1))
        out.append(Star.from_bounds(val, val))
    return out


def relative_position_bias_table_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        c = np.zeros((z.dim, 1))
        V = np.zeros((z.dim, 1))
        out.append(Zono(c, V))
    return out
