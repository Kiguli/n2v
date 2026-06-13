"""Sanity tests for the ViT token/position layers (CLSToken, PositionalEncoding)."""

from __future__ import annotations

import numpy as np

from n2v.sets import Box
from n2v.nn.layer_ops import positional_encoding_reach, cls_token_reach
from n2v.nn.layers import PositionalEncoding, CLSToken


def test_positional_encoding_box():
    layer = PositionalEncoding(dim=4, max_len=16)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = positional_encoding_reach.positional_encoding_box(layer, [b])
    assert out[0].dim == 4


def test_cls_token_prepends_dim():
    layer = CLSToken(dim=4)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = cls_token_reach.cls_token_box(layer, [b])
    assert out[0].dim == 8
