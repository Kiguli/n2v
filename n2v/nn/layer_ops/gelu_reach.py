"""GELU activation reachability.

GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

Non-monotone (small dip near x = -0.75). Box reach uses the global
minimum at the inflection ``x_min ≈ -0.7517916``; Star reach uses a
sound linear over-approximation that contains the function on the
neuron's bound interval.

Coverage matches nnVLA: Box (IBP), Star (CROWN-style approx).
"""

from __future__ import annotations

from math import erf, sqrt
from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar


_GELU_X_MIN = -0.7517916  # x where GELU attains its global min
_GELU_F_MIN = -0.16997    # GELU(_GELU_X_MIN) ≈ -0.169966...


def _gelu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def gelu_box(input_boxes: List[Box]) -> List[Box]:
    """Sound Box reach for GELU, accounting for the small left-side dip."""
    out: List[Box] = []
    for b in input_boxes:
        lb = b.lb.flatten()
        ub = b.ub.flatten()
        fl = _gelu(lb)
        fu = _gelu(ub)
        # If interval brackets the global min, lower bound is f_min.
        contains_min = (lb <= _GELU_X_MIN) & (ub >= _GELU_X_MIN)
        out_lb = np.where(contains_min, _GELU_F_MIN, np.minimum(fl, fu))
        out_ub = np.maximum(fl, fu)
        out.append(Box(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)))
    return out


def gelu_star_approx(input_stars: List[Star]) -> List[Star]:
    """Box-lifted Star reach. See module docstring for soundness."""
    output: List[Star] = []
    for s in input_stars:
        base = s.to_star() if isinstance(s, ImageStar) else s
        lb, ub = base.estimate_ranges()
        # Reuse the Box helper for the dip-aware bounding.
        box = gelu_box([Box(lb, ub)])[0]
        out = Star.from_bounds(box.lb, box.ub)
        if isinstance(s, ImageStar):
            out = out.to_image_star(s.height, s.width, s.num_channels)
        output.append(out)
    return output
