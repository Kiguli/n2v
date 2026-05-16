"""Reachability oracle framework.

The single load-bearing helper is :func:`assert_set_contains_pushforward`,
which checks that the *pushforward* of an input set through a concrete
PyTorch operation lies inside the reachable output set produced by an n2v
layer reachability routine.

Soundness intuition
-------------------
A reachability operation ``reach: (layer, set_in) -> set_out`` is sound iff
for every concrete input point ``x in set_in``, the value ``layer(x)`` is
contained in ``set_out``. The oracle Monte-Carlo-checks this by sampling
many points and verifying containment in (the union of) the produced
output sets.

This is a *necessary* but not *sufficient* condition for soundness, and is
intentionally cheaper than a formal proof. It catches dispatcher routing
bugs, basis-matrix transpose mistakes, and constraint-sign errors with
high probability.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Union

import numpy as np

from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

SetType = Union[Star, Zono, Box, ImageStar, ImageZono]


def sample_from_set(input_set: SetType, n_samples: int) -> np.ndarray:
    """Sample ``n_samples`` concrete points from ``input_set``.

    Returns
    -------
    np.ndarray
        Array of shape ``(dim, k)`` where ``k <= n_samples``. For Box/Zono
        the helper always returns exactly ``n_samples`` columns; for Star
        it may return fewer if rejection sampling fails for the requested
        budget.
    """
    if isinstance(input_set, (Star, ImageStar)):
        base = input_set if not isinstance(input_set, ImageStar) else input_set.to_star()
        return base.sample(n_samples)

    if isinstance(input_set, (Zono, ImageZono)):
        base = input_set if not isinstance(input_set, ImageZono) else input_set.to_zono()
        # Sample alpha ~ U[-1, 1]^k and compute c + V @ alpha.
        n_gen = base.V.shape[1]
        alpha = np.random.uniform(-1.0, 1.0, size=(n_gen, n_samples))
        return base.c + base.V @ alpha

    if isinstance(input_set, Box):
        return input_set.sample(n_samples)

    raise TypeError(f"Unsupported set type for sampling: {type(input_set).__name__}")


def contains(output_set: SetType, point: np.ndarray, atol: float = 1e-5) -> bool:
    """Check whether ``point`` lies inside ``output_set`` (with tolerance)."""
    point = np.asarray(point, dtype=np.float64).reshape(-1, 1)

    if isinstance(output_set, ImageStar):
        return output_set.to_star().contains(point)
    if isinstance(output_set, ImageZono):
        return output_set.to_zono().contains(point)
    if isinstance(output_set, Star):
        return output_set.contains(point)
    if isinstance(output_set, Zono):
        return output_set.contains(point)
    if isinstance(output_set, Box):
        lb = output_set.lb.reshape(-1, 1)
        ub = output_set.ub.reshape(-1, 1)
        return bool(np.all(point >= lb - atol) and np.all(point <= ub + atol))

    raise TypeError(f"Unsupported set type for containment: {type(output_set).__name__}")


def _contains_in_any(output_sets: Sequence[SetType], point: np.ndarray, atol: float) -> bool:
    return any(contains(s, point, atol=atol) for s in output_sets)


def assert_set_contains_pushforward(
    layer_fn: Callable[[np.ndarray], np.ndarray],
    input_set: SetType,
    output_sets: Iterable[SetType],
    n_samples: int = 512,
    atol: float = 1e-5,
    min_required: int | None = None,
) -> None:
    """Sample ``n_samples`` points from ``input_set``, push them through
    ``layer_fn``, and assert every pushforward point lies inside at least
    one of ``output_sets``.

    Parameters
    ----------
    layer_fn
        Callable that maps a single concrete input point ``x`` (1D
        np.ndarray of length ``dim_in``) to a 1D np.ndarray of length
        ``dim_out``.
    input_set
        Input reachable set (Star/Zono/Box or their Image variants).
    output_sets
        Iterable of reachable output sets to test containment against.
    n_samples
        Target number of Monte-Carlo samples.
    atol
        Absolute tolerance for box-style containment checks.
    min_required
        If given, accept the test if at least this many sample points
        were successfully drawn (useful for high-dimensional Stars where
        rejection sampling may yield fewer).
    """
    output_list: List[SetType] = list(output_sets)
    if not output_list:
        raise AssertionError("Oracle received no output sets to check against.")

    samples = sample_from_set(input_set, n_samples)
    if samples.size == 0:
        raise AssertionError("Sampling produced no points for the input set.")

    k = samples.shape[1]
    if min_required is not None and k < min_required:
        raise AssertionError(
            f"Oracle sampled only {k} points (< min_required={min_required})."
        )

    misses = 0
    examples: list[np.ndarray] = []
    for i in range(k):
        x = samples[:, i]
        y = np.asarray(layer_fn(x), dtype=np.float64).reshape(-1)
        if not _contains_in_any(output_list, y, atol):
            misses += 1
            if len(examples) < 3:
                examples.append(y)

    if misses > 0:
        raise AssertionError(
            f"Pushforward containment failed for {misses}/{k} samples. "
            f"First miss(es): {examples!r}"
        )
