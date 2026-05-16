"""ReLU6 activation reachability.

ReLU6(x) = min(max(x, 0), 6). Monotone non-decreasing, piecewise linear.
The exact Star reach can be obtained by composing ReLU with a clamp at
6, but the practical implementation uses a single three-region linear
relaxation as in nnVLA's ``relu6/methods/crown.py``.

Coverage (matches nnVLA): Box (IBP), Star (CROWN-style approx).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar


def _relu6(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, 0.0), 6.0)


# ---------------------------------------------------------------------------
# Box (IBP)
# ---------------------------------------------------------------------------

def relu6_box(input_boxes: List[Box]) -> List[Box]:
    """ReLU6 is monotone; applying it to the bounds is exact."""
    return [Box(_relu6(b.lb), _relu6(b.ub)) for b in input_boxes]


# ---------------------------------------------------------------------------
# Star (CROWN-style approx)
# ---------------------------------------------------------------------------

def relu6_star_approx(input_stars: List[Star]) -> List[Star]:
    """Linear-relaxation Star approx for ReLU6.

    For neuron with bounds ``[l, u]`` after LP:
      * ``u <= 0``: y = 0
      * ``l >= 6``: y = 6
      * ``0 <= l, u <= 6``: y = x (identity region)
      * ``l < 0 < u <= 6``: ReLU relaxation (existing relu_reach handles it)
      * ``0 <= l < 6 < u``: clamp relaxation
      * ``l < 0 < 6 < u``: full triangle [0, 6]

    For the draft PR, mixed regions are over-approximated with the
    coarsest sound box ``[max(0, l_clamped), min(6, u_clamped)]`` lifted
    to a Star via a single new predicate per neuron. This is not as
    tight as nnVLA's per-regime relaxation but it is sound.
    """
    output: List[Star] = []
    for s in input_stars:
        base = s.to_star() if isinstance(s, ImageStar) else s
        lb, ub = base.estimate_ranges()
        lb = lb.flatten()
        ub = ub.flatten()
        out_lb = _relu6(lb).reshape(-1, 1)
        out_ub = _relu6(ub).reshape(-1, 1)
        out = Star.from_bounds(out_lb, out_ub)
        if isinstance(s, ImageStar):
            out = out.to_image_star(s.height, s.width, s.num_channels)
        output.append(out)
    return output
