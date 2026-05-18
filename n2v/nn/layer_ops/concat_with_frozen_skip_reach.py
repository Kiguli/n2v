"""ConcatWithFrozenSkip reachability: concatenate input set with a constant.

For a Box, this stacks the bounds. For Star/Zono, it pads the basis or
generator matrix with constant rows containing the frozen skip values.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star, Zono


def _skip_vec(layer) -> np.ndarray:
    return layer.skip.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)


def concat_with_frozen_skip_box(layer, input_boxes: List[Box]) -> List[Box]:
    skip = _skip_vec(layer)
    out: List[Box] = []
    for b in input_boxes:
        out.append(Box(np.vstack([b.lb, skip]), np.vstack([b.ub, skip])))
    return out


def concat_with_frozen_skip_star(layer, input_stars: List[Star]) -> List[Star]:
    skip = _skip_vec(layer)
    out: List[Star] = []
    for s in input_stars:
        m = skip.shape[0]
        n_var = s.V.shape[1] - 1
        skip_block = np.hstack([skip, np.zeros((m, n_var))])
        new_V = np.vstack([s.V, skip_block])
        out.append(Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub))
    return out


def concat_with_frozen_skip_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    skip = _skip_vec(layer)
    out: List[Zono] = []
    for z in input_zonos:
        m = skip.shape[0]
        new_c = np.vstack([z.c, skip])
        new_V = np.vstack([z.V, np.zeros((m, z.V.shape[1]))])
        out.append(Zono(new_c, new_V))
    return out
