"""LayerScale reachability: ``y = gamma * x`` with learnable per-channel ``gamma``.

``LayerScale.forward`` broadcasts a length-``dim`` ``gamma`` across the
last axis of its input — i.e. for an ``(L, dim)`` sequence each of the
``L`` tokens is scaled by the same ``gamma``.

The reach applies the tiled diagonal directly to each set
representation via :mod:`_row_affine` — O(n * nVar), no dense
``(n, n)`` matrix (Copilot review, PR #12: the previous ``np.diag``
surrogate was O(n^2) and OOMed at transformer scale).

Coverage matches nnVLA: Box, Star, Zono, Hexatope, Octatope.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops._row_affine import apply_row_affine


def _gamma_tiled(layer, input_dim: int) -> np.ndarray:
    """Return a length-``input_dim`` diagonal scale vector.

    ``layer.gamma`` has length ``dim``; we tile it across ``L = input_dim //
    dim`` tokens. Raises ``ValueError`` if ``input_dim`` is not a multiple
    of ``dim``.
    """
    gamma = layer.gamma.detach().cpu().numpy().astype(np.float64).reshape(-1)
    dim = gamma.size
    if input_dim % dim != 0:
        raise ValueError(
            f"LayerScale flat input dim {input_dim} is not a multiple of "
            f"dim={dim}. The concrete forward broadcasts gamma across the "
            f"last axis."
        )
    L = input_dim // dim
    return np.tile(gamma, L)


def _apply(layer, input_sets: List) -> List:
    out = []
    for s in input_sets:
        out.append(apply_row_affine(s, _gamma_tiled(layer, s.dim)))
    return out


def layerscale_star(layer, input_stars: List[Star]) -> List[Star]:
    return _apply(layer, input_stars)


def layerscale_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return _apply(layer, input_zonos)


def layerscale_box(layer, input_boxes: List[Box]) -> List[Box]:
    return _apply(layer, input_boxes)


def layerscale_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    return _apply(layer, input_sets)


def layerscale_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    return _apply(layer, input_sets)
