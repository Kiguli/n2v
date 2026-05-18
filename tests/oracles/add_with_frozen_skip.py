"""Pushforward-containment oracle for AddWithFrozenSkip reachability."""

from __future__ import annotations

import numpy as np
import torch

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import add_with_frozen_skip_reach
from n2v.nn.layers import AddWithFrozenSkip

from tests.oracles import assert_set_contains_pushforward


def _concrete(layer: AddWithFrozenSkip):
    layer.eval()

    def _apply(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return layer(torch.from_numpy(x).float()).detach().cpu().numpy()

    return _apply


def test_add_with_frozen_skip_exactness():
    skip = torch.tensor([1.0, -2.0, 3.0])
    layer = AddWithFrozenSkip(skip=skip)
    inp = Box(np.array([[-1.0], [-1.0], [-1.0]]), np.array([[1.0], [1.0], [1.0]]))
    out = add_with_frozen_skip_reach.add_with_frozen_skip_box(layer, [inp])
    assert_set_contains_pushforward(_concrete(layer), inp, out, n_samples=128)
    # Affine + constant: bounds should be exact.
    np.testing.assert_allclose(out[0].lb.flatten(), np.array([0.0, -3.0, 2.0]))
    np.testing.assert_allclose(out[0].ub.flatten(), np.array([2.0, -1.0, 4.0]))


def test_add_with_frozen_skip_star():
    skip = torch.tensor([0.5, -0.25])
    layer = AddWithFrozenSkip(skip=skip)
    inp = Star.from_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    out = add_with_frozen_skip_reach.add_with_frozen_skip_star(layer, [inp])
    assert_set_contains_pushforward(_concrete(layer), inp, out, n_samples=128)
