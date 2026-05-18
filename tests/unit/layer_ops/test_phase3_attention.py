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
from n2v.nn.layers import RelativeAttentionBiasT5, RelativePositionBiasTable


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
    q = _box_2d(2, 4)
    k = _box_2d(2, 4)
    v = _box_2d(2, 4)
    out = linear_attention_reach.linear_attention_box([q], [k], [v], l_q=2, d_v=4)
    assert len(out) == 1
    assert np.all(out[0].lb <= out[0].ub)


def test_linear_attention_box_legacy_raises():
    """Single-input call was unsound and should fail loudly."""
    b = Box(np.array([[-1.0], [0.0], [1.0]]), np.array([[0.0], [1.0], [2.0]]))
    with pytest.raises(NotImplementedError, match="Q/K/V"):
        linear_attention_reach.linear_attention_box([b])


def test_relative_bias_t5_box():
    """Box reach uses the layer's learned bias table extrema."""
    layer = RelativeAttentionBiasT5(num_buckets=8, max_distance=16, n_heads=2)
    # Force a non-zero embedding so we can verify the extrema are used.
    with torch.no_grad():
        layer.relative_attention_bias.weight.data.uniform_(-0.5, 0.5)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = relative_attention_bias_t5_reach.relative_attention_bias_t5_box(layer, [b])
    assert out[0].dim == 4
    # Output should be the trained extrema, not the zero constant.
    w = layer.relative_attention_bias.weight.detach().cpu().numpy()
    np.testing.assert_allclose(out[0].lb, np.full((4, 1), float(w.min())))
    np.testing.assert_allclose(out[0].ub, np.full((4, 1), float(w.max())))


def test_relative_position_bias_table_box():
    layer = RelativePositionBiasTable(window_size=2, n_heads=2)
    with torch.no_grad():
        layer.bias_table.data.uniform_(-0.3, 0.3)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = relative_position_bias_table_reach.relative_position_bias_table_box(layer, [b])
    assert out[0].dim == 4
    t = layer.bias_table.detach().cpu().numpy()
    np.testing.assert_allclose(out[0].lb, np.full((4, 1), float(t.min())))
    np.testing.assert_allclose(out[0].ub, np.full((4, 1), float(t.max())))
