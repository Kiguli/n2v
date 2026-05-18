"""Pushforward-containment oracle for LayerScale reachability."""

from __future__ import annotations

import numpy as np
import torch

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import layerscale_reach
from n2v.nn.layers import LayerScale

from tests.oracles import assert_set_contains_pushforward


def _concrete_layerscale(layer: LayerScale):
    layer.eval()

    def _apply(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return layer(torch.from_numpy(x).float()).detach().cpu().numpy()

    return _apply


def test_layerscale_box_oracle():
    layer = LayerScale(dim=3, init_value=2.5)
    inp = Box(np.array([[-1.0], [0.0], [1.0]]), np.array([[0.0], [1.0], [2.0]]))
    out = layerscale_reach.layerscale_box(layer, [inp])
    assert_set_contains_pushforward(_concrete_layerscale(layer), inp, out, n_samples=128)


def test_layerscale_star_oracle():
    layer = LayerScale(dim=3, init_value=-0.5)
    inp = Star.from_bounds(np.zeros((3, 1)), np.ones((3, 1)))
    out = layerscale_reach.layerscale_star(layer, [inp])
    assert_set_contains_pushforward(_concrete_layerscale(layer), inp, out, n_samples=128)
