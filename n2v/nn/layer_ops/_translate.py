"""O(n) constant translation of reachable sets: ``y = x + t``.

The ViT position-embedding add reduces to adding a constant vector.
Routing that through a dense identity ``nn.Linear(n, n)`` costs O(n^2)
memory/time and OOMs at transformer-flattened sizes (Copilot review).
A translation only moves the set's centre:

* Star/ImageStar: add to the centre column of ``V`` (generators,
  constraints and predicate bounds are unchanged -- exact image).
* Zono/ImageZono: add to ``c``.
* Box: add to both bounds.
"""

from __future__ import annotations

import numpy as np

from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


def translate_set(s, t: np.ndarray):
    """Return a new set equal to ``{x + t : x in s}``.

    ``t`` must be a flat vector of length ``s.dim`` (HWC order for the
    Image variants, matching their flat layout).
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.size != s.dim:
        raise ValueError(
            f"translate_set: translation length {t.size} does not match "
            f"set dim {s.dim}"
        )

    if isinstance(s, ImageStar):
        new_V = s.V.copy()
        new_V[..., 0] = new_V[..., 0] + t.reshape(
            s.height, s.width, s.num_channels)
        return ImageStar(
            new_V, s.C, s.d, s.predicate_lb, s.predicate_ub,
            s.height, s.width, s.num_channels,
        )

    if isinstance(s, Star):
        new_V = s.V.copy()
        new_V[:, 0] = new_V[:, 0] + t
        return Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub)

    if isinstance(s, ImageZono):
        return ImageZono(
            s.c + t.reshape(-1, 1), s.V.copy(),
            s.height, s.width, s.num_channels,
        )

    if isinstance(s, Zono):
        return Zono(s.c + t.reshape(-1, 1), s.V.copy())

    if isinstance(s, Box):
        return Box(s.lb + t.reshape(-1, 1), s.ub + t.reshape(-1, 1))

    raise TypeError(
        f"translate_set: unsupported set type {type(s).__name__}"
    )
