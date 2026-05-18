"""ELU activation reachability.

ELU(x) = x for x >= 0, alpha*(exp(x) - 1) for x < 0. Smooth, monotone,
strictly increasing. Coverage matches nnVLA: Box (IBP), Star (CROWN-
style linear relaxation).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star


def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x >= 0, x, alpha * (np.exp(np.minimum(x, 0.0)) - 1.0))


def elu_box(input_boxes: List[Box], alpha: float = 1.0) -> List[Box]:
    """ELU is monotone non-decreasing; apply directly to bounds."""
    return [Box(_elu(b.lb, alpha), _elu(b.ub, alpha)) for b in input_boxes]


def elu_star_approx(input_stars: List[Star], alpha: float = 1.0) -> List[Star]:
    """Box-lifted Star over-approximation, preserving ImageStar shape."""

    def _box(lb: np.ndarray, ub: np.ndarray):
        return (
            _elu(lb.flatten(), alpha).reshape(-1, 1),
            _elu(ub.flatten(), alpha).reshape(-1, 1),
        )

    return apply_box_lift_star(input_stars, _box)
