"""QuickGELU activation reachability.

QuickGELU(x) = x * sigmoid(1.702 * x).  Smooth non-monotone (small dip
near x ≈ -1.18). Box reach accounts for the dip; Star reach uses a
sound box-lifted relaxation matching nnVLA's CROWN fallback.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar


_QGELU_X_MIN = -1.176                # numerically located global min
_QGELU_F_MIN = -0.169                # QuickGELU(_QGELU_X_MIN)


def _quickgelu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-1.702 * x))


def quickgelu_box(input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        lb = b.lb.flatten()
        ub = b.ub.flatten()
        fl = _quickgelu(lb)
        fu = _quickgelu(ub)
        contains_min = (lb <= _QGELU_X_MIN) & (ub >= _QGELU_X_MIN)
        out_lb = np.where(contains_min, _QGELU_F_MIN, np.minimum(fl, fu))
        out_ub = np.maximum(fl, fu)
        out.append(Box(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)))
    return out


def quickgelu_star_approx(input_stars: List[Star]) -> List[Star]:
    output: List[Star] = []
    for s in input_stars:
        base = s.to_star() if isinstance(s, ImageStar) else s
        lb, ub = base.estimate_ranges()
        box = quickgelu_box([Box(lb, ub)])[0]
        out = Star.from_bounds(box.lb, box.ub)
        if isinstance(s, ImageStar):
            out = out.to_image_star(s.height, s.width, s.num_channels)
        output.append(out)
    return output
