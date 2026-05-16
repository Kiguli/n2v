"""EfficientAttentionSR reachability (Spatial Reduction Attention).

SRA reduces the K/V sequence length via a strided convolution, then
applies softmax attention to the reduced representation. For
reachability the SR part is a Conv2d (handled by conv2d_reach) and the
attention part by softmax_attention_reach; this module is a thin
wrapper that recursively dispatches its sub-modules.
"""

from __future__ import annotations

from typing import List


def efficient_attention_sr_passthrough(layer, input_sets: List, method: str = "exact", **kwargs):
    from n2v.nn.layer_ops.dispatcher import reach_layer
    current = input_sets
    for sub_name in ("sr", "proj_q", "proj_k", "proj_v", "attention", "proj_out"):
        sub = getattr(layer, sub_name, None)
        if sub is not None:
            current = reach_layer(sub, current, method, **kwargs)
    return current
