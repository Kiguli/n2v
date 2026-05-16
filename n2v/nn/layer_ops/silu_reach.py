"""SiLU / Swish activation reachability.

SiLU(x) = x * sigmoid(x). Smooth, non-monotone (small dip near x ≈ -1.28).
Coverage matches nnVLA: Box (IBP, dip-aware), Star (CROWN-style approx).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar


_SILU_X_MIN = -1.2785          # x where SiLU attains its global min
_SILU_F_MIN = -0.2784645        # SiLU(_SILU_X_MIN)


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
    output: List[Star] = []
    for s in input_stars:
        base = s.to_star() if isinstance(s, ImageStar) else s
        lb, ub = base.estimate_ranges()
        box = silu_box([Box(lb, ub)])[0]
        out = Star.from_bounds(box.lb, box.ub)
        if isinstance(s, ImageStar):
            out = out.to_image_star(s.height, s.width, s.num_channels)
        output.append(out)
    return output
