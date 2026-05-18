"""CLSToken reachability: prepend a learnable token to a sequence.

Concatenation with a constant token vector routes through the same
pattern as :mod:`concat_with_frozen_skip_reach` (with the skip placed
*before* the running activation instead of after).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star, Zono


def _token(layer) -> np.ndarray:
    return layer.token.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)


def cls_token_box(layer, input_boxes: List[Box]) -> List[Box]:
    tok = _token(layer)
    return [Box(np.vstack([tok, b.lb]), np.vstack([tok, b.ub])) for b in input_boxes]


def cls_token_star(layer, input_stars: List[Star]) -> List[Star]:
    tok = _token(layer)
    out: List[Star] = []
    for s in input_stars:
        m = tok.shape[0]
        n_var = s.V.shape[1] - 1
        prepend = np.hstack([tok, np.zeros((m, n_var))])
        new_V = np.vstack([prepend, s.V])
        out.append(Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub))
    return out


def cls_token_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    tok = _token(layer)
    out: List[Zono] = []
    for z in input_zonos:
        m = tok.shape[0]
        new_c = np.vstack([tok, z.c])
        new_V = np.vstack([np.zeros((m, z.V.shape[1])), z.V])
        out.append(Zono(new_c, new_V))
    return out
