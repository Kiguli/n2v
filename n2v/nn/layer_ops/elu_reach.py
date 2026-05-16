"""ELU activation reachability.

ELU(x) = x for x >= 0, alpha*(exp(x) - 1) for x < 0. Smooth, monotone,
strictly increasing. Coverage matches nnVLA: Box (IBP), Star (CROWN-
style linear relaxation).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar


def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x >= 0, x, alpha * (np.exp(np.minimum(x, 0.0)) - 1.0))


def elu_box(input_boxes: List[Box], alpha: float = 1.0) -> List[Box]:
    """ELU is monotone non-decreasing; apply directly to bounds."""
    return [Box(_elu(b.lb, alpha), _elu(b.ub, alpha)) for b in input_boxes]


def elu_star_approx(input_stars: List[Star], alpha: float = 1.0) -> List[Star]:
    """Box-lifted Star over-approximation.

    Sound but coarse: every neuron is bound by an axis-aligned interval
    ``[ELU(l), ELU(u)]``. Tighter linear relaxations can replace this
    later (see nnVLA ``elu/methods/crown.py``).
    """
    output: List[Star] = []
    for s in input_stars:
        base = s.to_star() if isinstance(s, ImageStar) else s
        lb, ub = base.estimate_ranges()
        out_lb = _elu(lb.flatten(), alpha).reshape(-1, 1)
        out_ub = _elu(ub.flatten(), alpha).reshape(-1, 1)
        out = Star.from_bounds(out_lb, out_ub)
        if isinstance(s, ImageStar):
            out = out.to_image_star(s.height, s.width, s.num_channels)
        output.append(out)
    return output
