"""RelativePositionBiasTable (Swin) reachability.

Like the T5 bias the forward produces a fixed bias tensor; its
reachable image is a single-point (constant) set.
Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.image_star import ImageStar


def relative_position_bias_table_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        v = np.zeros_like(b.lb)
        out.append(Box(v, v))
    return out


def relative_position_bias_table_star(layer, input_stars: List[Star]) -> List[Star]:
    """Constant-set output. Preserves ImageStar shape if applicable."""
    out: List[Star] = []
    for s in input_stars:
        is_image = isinstance(s, ImageStar)
        dim = s.to_star().dim if is_image else s.dim
        val = np.zeros((dim, 1))
        new_star = Star.from_bounds(val, val)
        if is_image:
            new_star = new_star.to_image_star(s.height, s.width, s.num_channels)
        out.append(new_star)
    return out


def relative_position_bias_table_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        c = np.zeros((z.dim, 1))
        V = np.zeros((z.dim, 1))
        out.append(Zono(c, V))
    return out


def relative_position_bias_table_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    """Constant set: a degenerate Hexatope at zero."""
    out: List[Hexatope] = []
    for s in input_sets:
        z = np.zeros((s.dim, 1))
        out.append(Hexatope.from_bounds(z, z))
    return out


def relative_position_bias_table_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    """Constant set: a degenerate Octatope at zero."""
    out: List[Octatope] = []
    for s in input_sets:
        z = np.zeros((s.dim, 1))
        out.append(Octatope.from_bounds(z, z))
    return out
