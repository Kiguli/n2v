"""Integration tests for the multi-input dispatcher in n2v.nn.reach.

These tests build small fx-traceable modules whose forward consumes
multiple input ports (DagAdd / DagConcat / SoftmaxAttention), trace
them, and run ``_handle_multi_input_op`` end-to-end. They verify that
the layer-op helpers actually fire — not just that they exist.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from n2v.sets import Box, Star
from n2v.nn.layers import DagAdd, DagConcat, SoftmaxAttention
from n2v.nn.reach import _handle_multi_input_op


def _make_node_values(*streams: List) -> dict:
    """Helper: build a node_values dict matching positional ``streams``."""
    return {f"arg{i}": s for i, s in enumerate(streams)}


def _make_fake_node(module_name: str, n_inputs: int) -> fx.Node:
    """Build a real fx.Node with ``n_inputs`` placeholder args.

    The dispatcher only reads ``node.args`` / ``node.kwargs`` and does
    ``isinstance(arg, fx.Node)`` checks, so the node op type is
    irrelevant — but ``call_module`` requires the named module to exist
    on the root, while ``call_function`` does not. We therefore build a
    ``call_function`` node with ``operator.add`` as a placeholder target.
    No GraphModule is needed since we only need the fx.Node.
    """
    import operator

    g = fx.Graph()
    placeholders = [g.placeholder(f"arg{i}") for i in range(n_inputs)]
    op_node = g.create_node(
        "call_function", operator.add, tuple(placeholders), {}, name=module_name
    )
    g.output(op_node)
    return op_node


def test_dagadd_box_routes_through_multi_input_dispatcher():
    module = DagAdd()
    node = _make_fake_node("dagadd", n_inputs=2)
    a = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))
    b = Box(np.array([[2.0], [3.0]]), np.array([[2.0], [3.0]]))
    node_values = {"arg0": [a], "arg1": [b]}
    out = _handle_multi_input_op(module, node, node_values, set_type=Box)
    assert out is not None and len(out) == 1
    np.testing.assert_allclose(out[0].lb.flatten(), [2.0, 3.0])
    np.testing.assert_allclose(out[0].ub.flatten(), [3.0, 4.0])


def test_dagconcat_box_routes_through_multi_input_dispatcher():
    module = DagConcat()
    node = _make_fake_node("dagconcat", n_inputs=2)
    a = Box(np.array([[0.0]]), np.array([[1.0]]))
    b = Box(np.array([[2.0], [3.0]]), np.array([[2.5], [3.5]]))
    node_values = {"arg0": [a], "arg1": [b]}
    out = _handle_multi_input_op(module, node, node_values, set_type=Box)
    assert out is not None and out[0].dim == 3


def test_softmax_attention_raises_without_d_head():
    module = SoftmaxAttention()  # d_head left None
    node = _make_fake_node("attn", n_inputs=3)
    q = Box(np.zeros((4, 1)), np.ones((4, 1)))
    k = Box(np.zeros((4, 1)), np.ones((4, 1)))
    v = Box(np.zeros((4, 1)), np.ones((4, 1)))
    node_values = {"arg0": [q], "arg1": [k], "arg2": [v]}
    with pytest.raises(ValueError, match="d_head"):
        _handle_multi_input_op(module, node, node_values, set_type=Box)


def test_softmax_attention_with_d_head_runs():
    module = SoftmaxAttention(d_head=2)
    node = _make_fake_node("attn", n_inputs=3)
    # L_q = L_k = 2, d_v = 2 → flat dim 4.
    q = Box(np.zeros((4, 1)), np.ones((4, 1)))
    k = Box(np.zeros((4, 1)), np.ones((4, 1)))
    v = Box(np.zeros((4, 1)), np.ones((4, 1)))
    node_values = {"arg0": [q], "arg1": [k], "arg2": [v]}
    out = _handle_multi_input_op(module, node, node_values, set_type=Box)
    assert out is not None and out[0].dim == 4
    assert np.all(out[0].lb >= -1e-9)
    assert np.all(out[0].ub <= 1.0 + 1e-9)


def test_single_input_layer_returns_none_from_multi_input_dispatcher():
    """A layer with only one input port must NOT consume the multi-input route."""
    module = nn.Linear(3, 2)  # single-input
    node = _make_fake_node("lin", n_inputs=1)
    inp = Box(np.zeros((3, 1)), np.ones((3, 1)))
    node_values = {"arg0": [inp]}
    out = _handle_multi_input_op(module, node, node_values, set_type=Box)
    assert out is None
