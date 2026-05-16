"""Sparsemax reachability.

Sparsemax(x) is the Euclidean projection onto the probability simplex.
Like softmax, every output lies in [0, 1] and sums to 1 along the
softmax axis. The sound box reach is simply ``[0, 1]`` per coordinate.

Coverage matches nnVLA: Box, Star (box-lifted).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star


def sparsemax_box(input_boxes: List[Box]) -> List[Box]:
    return [
        Box(np.zeros_like(b.lb), np.ones_like(b.ub))
        for b in input_boxes
    ]


def sparsemax_star_approx(input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        n = s.dim
        out.append(Star.from_bounds(np.zeros((n, 1)), np.ones((n, 1))))
    return out
