"""Pushforward-containment oracle for CausalMask reachability."""

from __future__ import annotations

import numpy as np
import torch

from n2v.sets import Box, Star
from n2v.nn.layer_ops import causal_mask_reach
from n2v.nn.layers import CausalMask

from tests.oracles import assert_set_contains_pushforward


def _concrete_mask(layer: CausalMask, L: int):
    layer.eval()

    def _apply(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = torch.from_numpy(x).float().reshape(L, L)
            return layer(logits).detach().cpu().numpy().reshape(-1)

    return _apply


def test_causal_mask_box_oracle():
    L = 3
    layer = CausalMask(max_len=L, fill_value=-1e9)
    inp = Box(np.zeros((L * L, 1)), np.ones((L * L, 1)))
    out = causal_mask_reach.causal_mask_box(layer, [inp])
    assert_set_contains_pushforward(_concrete_mask(layer, L), inp, out, n_samples=64)


def test_causal_mask_star_oracle():
    L = 3
    layer = CausalMask(max_len=L, fill_value=-1e9)
    inp = Star.from_bounds(np.zeros((L * L, 1)), np.ones((L * L, 1)))
    out = causal_mask_reach.causal_mask_star(layer, [inp])
    assert_set_contains_pushforward(_concrete_mask(layer, L), inp, out, n_samples=64)
