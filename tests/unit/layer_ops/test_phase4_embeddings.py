"""Sanity tests for the ViT token/position layers (CLSToken, PositionalEncoding)."""

from __future__ import annotations

import numpy as np
import pytest

from n2v.sets import Box
from n2v.nn.layer_ops import positional_encoding_reach, cls_token_reach
from n2v.nn.layers import PositionalEncoding, CLSToken


def test_positional_encoding_box():
    layer = PositionalEncoding(dim=4, max_len=16)
    b = Box(np.zeros((8, 1)), np.ones((8, 1)))  # L=2 tokens of dim 4
    out = positional_encoding_reach.positional_encoding_box(layer, [b])
    assert out[0].dim == 8


def test_positional_encoding_non_multiple_dim_raises():
    """Copilot review: an input dim not a multiple of the model dim is
    not a valid (L, dim) sequence; the concrete forward would raise, so
    the reach must too (rather than translating a truncated PE row)."""
    layer = PositionalEncoding(dim=16, max_len=64)
    b = Box(np.zeros((60, 1)), np.ones((60, 1)))  # 60 not a multiple of 16
    with pytest.raises(ValueError, match="not a multiple of the model dim"):
        positional_encoding_reach.positional_encoding_box(layer, [b])


def test_positional_encoding_beyond_max_len_raises():
    layer = PositionalEncoding(dim=4, max_len=2)
    b = Box(np.zeros((12, 1)), np.ones((12, 1)))  # L=3 > max_len=2
    with pytest.raises(ValueError, match="no encoding"):
        positional_encoding_reach.positional_encoding_box(layer, [b])


def test_cls_token_prepends_dim():
    layer = CLSToken(dim=4)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = cls_token_reach.cls_token_box(layer, [b])
    assert out[0].dim == 8
