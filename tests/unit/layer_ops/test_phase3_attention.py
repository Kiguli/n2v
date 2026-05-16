"""Phase 3 sanity tests: attention reachability shapes and basic soundness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from n2v.sets import Box, Star
from n2v.nn.layer_ops import (
    softmax_attention_reach,
    causal_mask_reach,
    sparsemax_reach,
    relative_attention_bias_t5_reach,
    relative_position_bias_table_reach,
    linear_attention_reach,
)


def _box_2d(rows: int, cols: int) -> Box:
    n = rows * cols
    return Box(np.zeros((n, 1)), np.ones((n, 1)))


def test_softmax_attention_box():
    q = _box_2d(3, 4)
    k = _box_2d(3, 4)
    v = _box_2d(3, 4)
    out = softmax_attention_reach.softmax_attention_box([q], [k], [v], l_q=3, d_v=4)
    assert len(out) == 1
    # Sum-to-one + values in [0, 1] implies output bounded by [0, 1] in each dim.
    assert np.all(out[0].lb >= 0.0 - 1e-9)
    assert np.all(out[0].ub <= 1.0 + 1e-9)


def test_sparsemax_box():
    b = Box(np.array([[0.0], [1.0], [2.0]]), np.array([[1.0], [2.0], [3.0]]))
    out = sparsemax_reach.sparsemax_box([b])
    np.testing.assert_array_equal(out[0].lb, np.zeros((3, 1)))
    np.testing.assert_array_equal(out[0].ub, np.ones((3, 1)))


def test_linear_attention_box():
    b = Box(np.array([[-1.0], [0.0], [1.0]]), np.array([[0.0], [1.0], [2.0]]))
    out = linear_attention_reach.linear_attention_box([b])
    assert len(out) == 1
    assert np.all(out[0].lb <= out[0].ub)


def test_relative_bias_t5_box():
    # Using a fake layer just to satisfy the function signature; box reach
    # returns a constant set independent of layer details.
    class _Fake:
        def __call__(self, *args, **kwargs):
            return torch.zeros(1, 1, 1, 1)

    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = relative_attention_bias_t5_reach.relative_attention_bias_t5_box(_Fake(), [b])
    assert out[0].dim == 4


def test_relative_position_bias_table_box():
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = relative_position_bias_table_reach.relative_position_bias_table_box(None, [b])
    assert out[0].dim == 4
