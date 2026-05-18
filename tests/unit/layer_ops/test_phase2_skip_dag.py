"""Phase 2 sanity tests: LayerScale, DropPath, frozen-skip and DAG ops."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import (
    layerscale_reach,
    drop_path_reach,
    add_with_frozen_skip_reach,
    concat_with_frozen_skip_reach,
    dag_add_reach,
    dag_concat_reach,
    concat2d_reach,
    selective_feature_fusion_reach,
)
from n2v.nn.layers import (
    LayerScale,
    DropPath,
    AddWithFrozenSkip,
    ConcatWithFrozenSkip,
)


@pytest.fixture
def small_box():
    lb = np.array([[-1.0], [0.0], [1.0]])
    ub = np.array([[0.0], [1.0], [2.0]])
    return Box(lb, ub)


def test_layerscale_box(small_box):
    layer = LayerScale(dim=3, init_value=2.0)
    out = layerscale_reach.layerscale_box(layer, [small_box])
    assert len(out) == 1
    np.testing.assert_allclose(out[0].lb.flatten(), 2.0 * small_box.lb.flatten())
    np.testing.assert_allclose(out[0].ub.flatten(), 2.0 * small_box.ub.flatten())


def test_drop_path_is_identity(small_box):
    layer = DropPath(drop_prob=0.5)
    layer.eval()
    out = drop_path_reach.drop_path_box(layer, [small_box])
    np.testing.assert_array_equal(out[0].lb, small_box.lb)
    np.testing.assert_array_equal(out[0].ub, small_box.ub)


def test_add_with_frozen_skip(small_box):
    layer = AddWithFrozenSkip(skip=torch.tensor([1.0, 2.0, 3.0]))
    out = add_with_frozen_skip_reach.add_with_frozen_skip_box(layer, [small_box])
    np.testing.assert_allclose(out[0].lb.flatten(), small_box.lb.flatten() + np.array([1, 2, 3]))


def test_concat_with_frozen_skip(small_box):
    layer = ConcatWithFrozenSkip(skip=torch.tensor([[7.0], [8.0]]))
    out = concat_with_frozen_skip_reach.concat_with_frozen_skip_box(layer, [small_box])
    assert out[0].dim == 5


def test_dag_add_two_streams(small_box):
    other = Box(np.array([[1.0], [1.0], [1.0]]), np.array([[2.0], [2.0], [2.0]]))
    out = dag_add_reach.dag_add_box([small_box], [[other]])
    assert len(out) == 1
    np.testing.assert_allclose(out[0].lb.flatten(), small_box.lb.flatten() + np.array([1, 1, 1]))


def test_dag_concat_two_streams(small_box):
    other = Box(np.array([[10.0]]), np.array([[11.0]]))
    out = dag_concat_reach.dag_concat_box([small_box], [[other]])
    assert out[0].dim == 4


def test_concat2d_two_streams(small_box):
    other = Box(np.array([[-3.0]]), np.array([[-2.0]]))
    out = concat2d_reach.concat2d_box([small_box], [[other]])
    assert out[0].dim == 4


def test_sff_two_streams(small_box):
    other = Box(np.array([[-5.0], [-5.0], [-5.0]]), np.array([[-4.0], [-4.0], [-4.0]]))
    out = selective_feature_fusion_reach.selective_feature_fusion_box(
        [small_box], [[other]]
    )
    np.testing.assert_allclose(
        out[0].lb.flatten(), np.array([-5.0, -5.0, -5.0])
    )
    np.testing.assert_allclose(
        out[0].ub.flatten(), np.array([0.0, 1.0, 2.0])
    )
