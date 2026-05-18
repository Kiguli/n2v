"""MixFFN reachability.

n2v relies on torch.fx to decompose MixFFN into its primitives
(``Linear -> Conv2d -> GELU -> Linear``). This module exists only to
provide a recursive fallback when an explicit ``MixFFN`` module appears
as an opaque node in the trace; in that case we walk its sub-modules
manually using the dispatcher.
"""

from __future__ import annotations

from typing import List


def mix_ffn_passthrough(layer, input_sets: List, method: str = "exact", **kwargs):
    """Recurse through MixFFN's sub-modules using the dispatcher.

    Imports ``reach_layer`` lazily — ``dispatcher.py`` imports
    ``mix_ffn_reach`` at module load, so importing ``reach_layer`` at
    file load would create a circular import.
    """
    from n2v.nn.layer_ops.dispatcher import reach_layer  # local import

    current = input_sets
    for sub in [layer.fc1, layer.dwconv, layer.act, layer.fc2]:
        current = reach_layer(sub, current, method, **kwargs)
    return current
