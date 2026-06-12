"""Dense-matrix-free application of structured row-linear operators.

Several transformer layers apply linear maps whose matrices are huge but
trivially structured:

* LayerScale: ``y = g (*) x``                       (diagonal)
* RoPE:       ``y = A (*) x + B (*) x[perm]``       (block 2x2 rotations)

Materialising these as dense ``(n, n)`` matrices (the previous
``np.diag`` / explicit rotation-matrix construction) is O(n^2) memory
and time -- for a flattened ``L * d_model`` transformer activation this
OOMs long before realistic sizes (Copilot review, PR #12).

Every operator handled here has the canonical form::

    y = A (*) x + B (*) x[perm]

with elementwise coefficient vectors ``A``, ``B`` and an index
permutation ``perm`` (LayerScale: ``B = None``). This module applies
that form directly to each set representation in O(n * nVar):

* Star/ImageStar: ``V' = A[:, None] * V + B[:, None] * V[perm, :]`` --
  the exact linear image (the centre column transforms identically);
  constraints and predicate bounds are untouched.
* Zono/ImageZono: same row arithmetic on ``c`` and ``V``.
* Box: per-coordinate interval arithmetic. ``x[i]`` and ``x[perm[i]]``
  are distinct coordinates of an axis-aligned box, hence independent,
  so the interval sum of the two sign-adjusted terms is exact.
* Hexatope/Octatope: same row arithmetic on ``center``/``generators``
  with the constraint kernel deep-copied -- exactly what their
  ``affine_map`` does (Bak et al. Thm 4/6), minus the dense matmul.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


def _rows(M: np.ndarray, A: np.ndarray, B: Optional[np.ndarray],
          perm: Optional[np.ndarray]) -> np.ndarray:
    """Apply ``y = A (*) x + B (*) x[perm]`` to every column of ``M``."""
    out = A[:, None] * M
    if B is not None:
        out = out + B[:, None] * M[perm, :]
    return out


def _interval(lb: np.ndarray, ub: np.ndarray, A: np.ndarray,
              B: Optional[np.ndarray], perm: Optional[np.ndarray]):
    """Exact interval image of ``y = A (*) x + B (*) x[perm]`` on a box."""
    a_lo = np.minimum(A * lb, A * ub)
    a_hi = np.maximum(A * lb, A * ub)
    if B is None:
        return a_lo, a_hi
    lb_p, ub_p = lb[perm], ub[perm]
    b_lo = np.minimum(B * lb_p, B * ub_p)
    b_hi = np.maximum(B * lb_p, B * ub_p)
    return a_lo + b_lo, a_hi + b_hi


def apply_row_affine(s, A: np.ndarray, B: Optional[np.ndarray] = None,
                     perm: Optional[np.ndarray] = None):
    """Apply ``y = A (*) x + B (*) x[perm]`` to a reachable set.

    ``A`` (and ``B``/``perm`` when given) must be length ``s.dim``.
    Returns a set of the same type as ``s``.
    """
    A = np.asarray(A, dtype=np.float64).reshape(-1)
    if B is not None:
        B = np.asarray(B, dtype=np.float64).reshape(-1)
        perm = np.asarray(perm, dtype=np.int64).reshape(-1)
        if B.size != A.size or perm.size != A.size:
            raise ValueError(
                f"row-affine operator size mismatch: A={A.size}, "
                f"B={B.size}, perm={perm.size}"
            )

    if isinstance(s, ImageStar):
        flat = s.to_star()
        out = Star(
            _rows(flat.V, A, B, perm), flat.C, flat.d,
            flat.predicate_lb, flat.predicate_ub,
        )
        return out.to_image_star(s.height, s.width, s.num_channels)

    if isinstance(s, Star):
        return Star(
            _rows(s.V, A, B, perm), s.C, s.d,
            s.predicate_lb, s.predicate_ub,
        )

    if isinstance(s, ImageZono):
        c = _rows(s.c, A, B, perm)
        V = _rows(s.V, A, B, perm)
        return ImageZono(c, V, s.height, s.width, s.num_channels)

    if isinstance(s, Zono):
        return Zono(_rows(s.c, A, B, perm), _rows(s.V, A, B, perm))

    if isinstance(s, Box):
        lb = np.asarray(s.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(s.ub, dtype=np.float64).reshape(-1)
        lo, hi = _interval(lb, ub, A, B, perm)
        return Box(lo.reshape(-1, 1), hi.reshape(-1, 1))

    if isinstance(s, Hexatope):
        center = np.asarray(s.center, dtype=np.float64).reshape(-1)
        new_center = A * center
        if B is not None:
            new_center = new_center + B * center[perm]
        new_gens = _rows(s.generators, A, B, perm)
        return Hexatope(new_center, new_gens, s.dcs.copy())

    if isinstance(s, Octatope):
        center = np.asarray(s.center, dtype=np.float64).reshape(-1)
        new_center = A * center
        if B is not None:
            new_center = new_center + B * center[perm]
        new_gens = _rows(s.generators, A, B, perm)
        return Octatope(new_center, new_gens, s.utvpi.copy())

    raise TypeError(
        f"apply_row_affine: unsupported set type {type(s).__name__}"
    )
