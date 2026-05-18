"""Phase 5 sanity tests: TiedLinear, Conv2DTranspose, heads, OpenMax."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import (
    tied_linear_reach,
    conv2d_transpose_reach,
    action_head_reach,
    action_tokenizer_reach,
    openmax_reach,
)
from n2v.nn.layers import (
    TiedLinear,
    ActionHead,
    ActionTokenizer,
    OpenMax,
)


def test_tied_linear_star():
    source = nn.Linear(3, 2)
    layer = TiedLinear(source=source, bias=True)
    lb = np.zeros((3, 1))
    ub = np.ones((3, 1))
    out = tied_linear_reach.tied_linear_star(layer, [Star.from_bounds(lb, ub)])
    assert out[0].dim == 2


def test_conv2d_transpose_box():
    layer = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
    layer.eval()
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))  # 1x2x2 input
    out = conv2d_transpose_reach.conv2d_transpose_box(layer, [b])
    # Output of stride-2 transpose over 2x2 is 4x4 = 16
    assert out[0].dim == 16


def test_action_head_box():
    layer = ActionHead(in_features=5, action_dim=3)
    out = action_head_reach.action_head_box(layer, [Box(np.zeros((5, 1)), np.ones((5, 1)))])
    assert out[0].dim == 3


def test_action_tokenizer_box():
    layer = ActionTokenizer(action_dim=3, n_bins=10, min_action=-1.0, max_action=1.0)
    out = action_tokenizer_reach.action_tokenizer_box(layer, [Box(-np.ones((3, 1)), np.ones((3, 1)))])
    assert out[0].dim == 3
    np.testing.assert_array_equal(out[0].lb, np.zeros((3, 1)))
    np.testing.assert_array_equal(out[0].ub, np.full((3, 1), 9.0))


def test_openmax_box():
    layer = OpenMax(num_classes=5, alpha=3)
    out = openmax_reach.openmax_box(layer, [Box(np.zeros((5, 1)), np.ones((5, 1)))])
    assert out[0].dim == 6  # num_classes + 1
