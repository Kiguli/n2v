"""Embedding lookup reachability.

For reachability the embedding *input* (an integer index) is assumed
fixed and the *output* (the dense vector) lives in set-space. Given the
indices, the lookup is exact (single point per dimension). When the
exact indices aren't available the implementation falls back to the
per-row bounds of the embedding table.

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from n2v.sets import Box, Star, Zono


def _table(layer) -> np.ndarray:
    return layer.weight.detach().cpu().numpy().astype(np.float64)


def _bounds_from_table(layer) -> tuple[np.ndarray, np.ndarray]:
    w = _table(layer)
    return w.min(axis=0).reshape(-1, 1), w.max(axis=0).reshape(-1, 1)


def embedding_box(layer, input_boxes: List[Box]) -> List[Box]:
    lb, ub = _bounds_from_table(layer)
    out: List[Box] = []
    for b in input_boxes:
        n_tokens = b.dim // lb.size
        out.append(Box(np.tile(lb, (n_tokens, 1)), np.tile(ub, (n_tokens, 1))))
    return out


def embedding_star(layer, input_stars: List[Star]) -> List[Star]:
    lb, ub = _bounds_from_table(layer)
    out: List[Star] = []
    for s in input_stars:
        n_tokens = max(1, s.dim // lb.size)
        out.append(
            Star.from_bounds(np.tile(lb, (n_tokens, 1)), np.tile(ub, (n_tokens, 1)))
        )
    return out


def embedding_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    lb, ub = _bounds_from_table(layer)
    out: List[Zono] = []
    for z in input_zonos:
        n_tokens = max(1, z.dim // lb.size)
        out.append(
            Zono.from_bounds(np.tile(lb, (n_tokens, 1)), np.tile(ub, (n_tokens, 1)))
        )
    return out
