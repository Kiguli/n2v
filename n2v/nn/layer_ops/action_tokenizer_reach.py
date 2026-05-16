"""ActionTokenizer reachability.

ActionTokenizer maps continuous actions to integer token IDs via
uniform binning. The output is a discrete-valued tensor; for sound
bounds we lift to the integer range ``[0, n_bins - 1]`` per dimension.

Coverage matches nnVLA: Box + Star (box-lifted), no Zono (output is
integer-valued so a zonotope generator basis is not meaningful).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star


def action_tokenizer_box(layer, input_boxes: List[Box]) -> List[Box]:
    n_bins = int(layer.n_bins) - 1
    out: List[Box] = []
    for b in input_boxes:
        out.append(
            Box(np.zeros_like(b.lb), np.full_like(b.ub, n_bins, dtype=np.float64))
        )
    return out


def action_tokenizer_star_approx(layer, input_stars: List[Star]) -> List[Star]:
    n_bins = int(layer.n_bins) - 1
    out: List[Star] = []
    for s in input_stars:
        n = s.dim
        out.append(Star.from_bounds(np.zeros((n, 1)), np.full((n, 1), float(n_bins))))
    return out
