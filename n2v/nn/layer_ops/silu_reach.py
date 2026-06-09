"""SiLU / Swish activation reachability.

SiLU(x) = x * sigmoid(x). Smooth, non-monotone (small dip near x ≈ -1.28).
Coverage matches nnVLA: Box (IBP, dip-aware), Star (CROWN-style approx).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star


_SILU_X_MIN = -1.2785          # x where SiLU attains its global min
_SILU_F_MIN = -0.2784646        # SiLU(_SILU_X_MIN). Rounded AWAY from zero
                                # from the true minimum -0.27846454 so the
                                # box floor is a true lower bound. The prior
                                # -0.2784645 was rounded TOWARD zero by
                                # ~4.3e-8, making the box floor strictly
                                # above the true min (T0-3 / audit C5).


def _silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def silu_box(input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        lb = b.lb.flatten()
        ub = b.ub.flatten()
        fl = _silu(lb)
        fu = _silu(ub)
        contains_min = (lb <= _SILU_X_MIN) & (ub >= _SILU_X_MIN)
        out_lb = np.where(contains_min, _SILU_F_MIN, np.minimum(fl, fu))
        out_ub = np.maximum(fl, fu)
        out.append(Box(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)))
    return out


def silu_star_approx(input_stars: List[Star]) -> List[Star]:
    """Box-lifted Star reach, preserving ImageStar shape."""

    def _box(lb: np.ndarray, ub: np.ndarray):
        box = silu_box([Box(lb, ub)])[0]
        return box.lb, box.ub

    return apply_box_lift_star(input_stars, _box)
