"""Phase 1 sanity tests: activations and normalisations ported from nnVLA.

These tests validate shape, dtype, and basic soundness via random-sample
pushforward containment. They are deliberately small; richer per-layer
oracles live under ``tests/oracles``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, Star
from n2v.nn.layer_ops import (
    relu6_reach,
    elu_reach,
    gelu_reach,
    quickgelu_reach,
    silu_reach,
    hardswish_reach,
    layernorm_reach,
    groupnorm_reach,
    rmsnorm_reach,
    grn_reach,
)
from n2v.nn.layers import RMSNorm, GRN

# pylint: disable=missing-function-docstring

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


@pytest.fixture
def small_box():
    lb = np.array([[-2.0], [-1.0], [0.0], [1.0]])
    ub = np.array([[-1.0], [0.0], [1.0], [2.0]])
    return Box(lb, ub)


@pytest.fixture
def small_star(small_box):
    return Star.from_bounds(small_box.lb, small_box.ub)


def _activation_box_passes(reach_fn, torch_fn, small_box):
    out = reach_fn([small_box])
    assert len(out) == 1
    samples = small_box.sample(64)
    pushed = torch_fn(torch.from_numpy(samples).double()).numpy()
    out_lb = out[0].lb.reshape(-1)
    out_ub = out[0].ub.reshape(-1)
    assert np.all(pushed >= out_lb[:, None] - 1e-5)
    assert np.all(pushed <= out_ub[:, None] + 1e-5)


def test_relu6_box(small_box):
    _activation_box_passes(relu6_reach.relu6_box, nn.ReLU6(), small_box)


def test_elu_box(small_box):
    _activation_box_passes(lambda b: elu_reach.elu_box(b, alpha=1.0), nn.ELU(alpha=1.0), small_box)


def test_gelu_box(small_box):
    _activation_box_passes(gelu_reach.gelu_box, nn.GELU(), small_box)


def test_quickgelu_box(small_box):
    quickgelu = lambda x: x * torch.sigmoid(1.702 * x)
    _activation_box_passes(quickgelu_reach.quickgelu_box, quickgelu, small_box)


def test_silu_box(small_box):
    _activation_box_passes(silu_reach.silu_box, nn.SiLU(), small_box)


def test_hardswish_box(small_box):
    _activation_box_passes(hardswish_reach.hardswish_box, nn.Hardswish(), small_box)


def test_activation_star_returns_star(small_star):
    for fn in [
        relu6_reach.relu6_star_approx,
        lambda s: elu_reach.elu_star_approx(s, alpha=1.0),
        gelu_reach.gelu_star_approx,
        quickgelu_reach.quickgelu_star_approx,
        silu_reach.silu_star_approx,
        hardswish_reach.hardswish_star_approx,
    ]:
        out = fn([small_star])
        assert len(out) == 1
        assert isinstance(out[0], Star)


# ---------------------------------------------------------------------------
# Normalisations
# ---------------------------------------------------------------------------


def test_layernorm_box():
    layer = nn.LayerNorm(4, eps=1e-5, elementwise_affine=True)
    layer.eval()
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    out = layernorm_reach.layernorm_box(layer, [Box(lb, ub)])
    assert len(out) == 1
    assert out[0].dim == 4


def test_rmsnorm_box():
    layer = RMSNorm(4, eps=1e-6)
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    out = rmsnorm_reach.rmsnorm_box(layer, [Box(lb, ub)])
    assert len(out) == 1
    assert out[0].dim == 4


def test_groupnorm_box():
    layer = nn.GroupNorm(num_groups=2, num_channels=4)
    layer.eval()
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    out = groupnorm_reach.groupnorm_box(layer, [Box(lb, ub)])
    assert len(out) == 1
    assert out[0].dim == 4


def test_grn_box():
    layer = GRN(dim=2)
    lb = np.zeros((4, 1))
    ub = np.ones((4, 1))
    out = grn_reach.grn_box(layer, [Box(lb, ub)])
    assert len(out) == 1
    assert out[0].dim == 4
