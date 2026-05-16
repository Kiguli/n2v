"""Pooler reachability: take CLS row, apply Linear + tanh.

This composite is decomposed via the dispatcher's :mod:`linear_reach` +
:mod:`tanh_reach` after slicing the first token. The helper here is a
recursive forwarder for the case when the layer is opaque.
"""

from __future__ import annotations

from typing import List


def pooler_passthrough(layer, input_sets: List, method: str = "exact", **kwargs):
    from n2v.nn.layer_ops.dispatcher import reach_layer
    current = input_sets
    current = reach_layer(layer.dense, current, method, **kwargs)
    current = reach_layer(layer.activation, current, method, **kwargs)
    return current
