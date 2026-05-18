"""Pushforward-containment oracle for RoPE reachability."""

from __future__ import annotations

import numpy as np
import torch

from n2v.sets import Box, Star
from n2v.nn.layer_ops import rope_reach
from n2v.nn.layers import RoPE

from tests.oracles import assert_set_contains_pushforward


def _flatten_rope(layer: RoPE, dim: int):
    # RoPE expects (..., L, D); the helper flattens an (L*D,) input.
    layer.eval()
    L = max(1, dim // layer.dim)

    def _apply(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(x).float().reshape(1, L, layer.dim)
            y = layer(t)
        return y.detach().cpu().numpy().reshape(-1)

    return _apply


def test_rope_box_oracle():
    layer = RoPE(dim=4, max_len=8)
    lb = np.zeros((8, 1))
    ub = np.ones((8, 1))
    inp = Box(lb, ub)
    out = rope_reach.rope_box(layer, [inp])
    assert_set_contains_pushforward(
        _flatten_rope(layer, dim=8), inp, out, n_samples=128
    )


def test_rope_star_oracle():
    layer = RoPE(dim=4, max_len=8)
    lb = np.zeros((8, 1))
    ub = np.ones((8, 1))
    inp = Star.from_bounds(lb, ub)
    out = rope_reach.rope_star(layer, [inp])
    assert_set_contains_pushforward(
        _flatten_rope(layer, dim=8), inp, out, n_samples=128
    )
