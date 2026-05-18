"""Phase 4 sanity tests: embeddings, positional encodings, tokens."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import (
    embedding_reach,
    positional_encoding_reach,
    rope_reach,
    cls_token_reach,
    distillation_token_reach,
)
from n2v.nn.layers import PositionalEncoding, RoPE, CLSToken, DistillationToken


def test_embedding_box():
    """4 token indices × 4 embed_dim = 16 output features."""
    layer = nn.Embedding(num_embeddings=10, embedding_dim=4)
    layer.eval()
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))  # 4 tokens
    out = embedding_reach.embedding_box(layer, [b])
    assert out[0].dim == 16


def test_positional_encoding_box():
    layer = PositionalEncoding(dim=4, max_len=16)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = positional_encoding_reach.positional_encoding_box(layer, [b])
    assert out[0].dim == 4


def test_rope_box():
    layer = RoPE(dim=4, max_len=16)
    b = Box(np.zeros((8, 1)), np.ones((8, 1)))
    out = rope_reach.rope_box(layer, [b])
    assert out[0].dim == 8


def test_cls_token_prepends_dim():
    layer = CLSToken(dim=4)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = cls_token_reach.cls_token_box(layer, [b])
    assert out[0].dim == 8


def test_distillation_token_prepends_dim():
    layer = DistillationToken(dim=4)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = distillation_token_reach.distillation_token_box(layer, [b])
    assert out[0].dim == 8
