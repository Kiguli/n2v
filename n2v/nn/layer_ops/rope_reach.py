"""Rotary Positional Embedding (RoPE) reachability.

For a fixed sequence position the rotation is an exact linear map per
token: the feature pairs ``(i, i + d/2)`` rotate by position-dependent
angles. In the canonical row-operator form (see :mod:`_row_affine`)::

    y = A (*) x + B (*) x[perm]

where for flat index ``pos*d + i`` (``i < d/2``, ``j = i + d/2``)::

    A[pos*d + i] = cos[pos, i]      B[pos*d + i] = -sin[pos, i]
    A[pos*d + j] = cos[pos, i]      B[pos*d + j] = +sin[pos, i]
    perm swaps the two halves of each token: pos*d+i <-> pos*d+j

This applies in O(n * nVar) without materialising the previous dense
``(L*d, L*d)`` rotation matrix (Copilot review, PR #12: O(n^2) and
infeasible at realistic transformer sizes).

Coverage matches nnVLA: Box, Star, Zono, Hexatope, Octatope.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops._row_affine import apply_row_affine


def _rotation_vectors(
    layer, dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(A, B, perm)`` for the RoPE rotation on a flat (L*d) vector.

    Raises ``ValueError`` if the flat input dim is not a multiple of
    ``layer.dim``, or if the implied sequence length exceeds the
    precomputed ``max_len``. Silently collapsing extra positions to
    zero would be unsound — the concrete forward would either index
    out of range or use unintended encodings.
    """
    cos = layer.cos.detach().cpu().numpy().astype(np.float64)
    sin = layer.sin.detach().cpu().numpy().astype(np.float64)
    d = layer.dim
    l_max = cos.shape[0]
    if dim % d != 0:
        raise ValueError(
            f"RoPE flat input dim {dim} is not a multiple of layer.dim={d}."
        )
    L = dim // d
    if L > l_max:
        raise ValueError(
            f"RoPE sequence length {L} exceeds layer.max_len={l_max}. The "
            f"concrete forward has no encodings for positions beyond max_len."
        )
    half = d // 2

    # Per-token index template.
    i_idx = np.arange(half)
    j_idx = i_idx + half

    A = np.empty(dim, dtype=np.float64)
    B = np.empty(dim, dtype=np.float64)
    perm = np.empty(dim, dtype=np.int64)
    for pos in range(L):
        c = cos[pos, :half]
        s = sin[pos, :half]
        base = pos * d
        # First half: y_i = c_i * x_i - s_i * x_j
        A[base + i_idx] = c
        B[base + i_idx] = -s
        perm[base + i_idx] = base + j_idx
        # Second half: y_j = c_i * x_j + s_i * x_i
        A[base + j_idx] = c
        B[base + j_idx] = s
        perm[base + j_idx] = base + i_idx
    return A, B, perm


def _apply(layer, input_sets: List) -> List:
    out = []
    for s in input_sets:
        A, B, perm = _rotation_vectors(layer, s.dim)
        out.append(apply_row_affine(s, A, B, perm))
    return out


def rope_star(layer, input_stars: List[Star]) -> List[Star]:
    return _apply(layer, input_stars)


def rope_box(layer, input_boxes: List[Box]) -> List[Box]:
    return _apply(layer, input_boxes)


def rope_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return _apply(layer, input_zonos)


def rope_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    return _apply(layer, input_sets)


def rope_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    return _apply(layer, input_sets)
