"""MixFFN reachability.

n2v relies on torch.fx to decompose MixFFN into its primitives
(``Linear -> Conv2d -> GELU -> Linear``). This module exists only to
provide a recursive fallback when an explicit ``MixFFN`` module appears
as an opaque node in the trace; in that case we walk its sub-modules
manually using the dispatcher.
"""

from __future__ import annotations

from typing import List

from n2v.nn.layer_ops.dispatcher import reach_layer  # local import to avoid cycle


def mix_ffn_passthrough(layer, input_sets: List, method: str = "exact", **kwargs):
    current = input_sets
    for sub in [layer.fc1, layer.dwconv, layer.act, layer.fc2]:
        current = reach_layer(sub, current, method, **kwargs)
    return current
