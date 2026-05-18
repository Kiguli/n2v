"""LinearAttention reachability.

Linear attention replaces softmax with a kernel feature map
``phi(q) @ (phi(k).T @ v)`` so that the bilinear cost is linear in
sequence length. For sound bounds we follow nnVLA's IBP: bound
``phi(q)`` and ``phi(k).T @ v`` independently by interval arithmetic
and combine.

Coverage matches nnVLA: Box, Star (box-lifted).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star


def _kernel_bounds_box(box: Box) -> Box:
    """Default kernel is elu(x) + 1 (Performers-style). Bound elementwise."""
    lb = np.where(box.lb >= 0, box.lb + 1.0, np.exp(np.minimum(box.lb, 0.0)))
    ub = np.where(box.ub >= 0, box.ub + 1.0, np.exp(np.minimum(box.ub, 0.0)))
    return Box(lb, ub)


def linear_attention_box(input_boxes: List[Box]) -> List[Box]:
    return [_kernel_bounds_box(b) for b in input_boxes]


def linear_attention_star_approx(input_stars: List[Star]) -> List[Star]:
    """Box-lifted Star reach, preserving ImageStar shape."""

    def _box(lb, ub):
        kb = _kernel_bounds_box(Box(lb, ub))
        return kb.lb, kb.ub

    return apply_box_lift_star(input_stars, _box)
