"""LayerNorm reachability.

LayerNorm: ``y = (x - mu) / sqrt(var + eps) * gamma + beta`` where mean
and variance are taken across the last axis of ``x``.

Since the normalisation is non-linear, this implementation uses interval
bounds on the mean and variance to derive a sound axis-aligned box on
the output, then lifts that into a Star (for Star reach) by
``Star.from_bounds``. This is the conservative fallback nnVLA's CROWN
backend uses when slope-zero rectangles are unavoidable.

Coverage matches nnVLA: Box (IBP), Star (CROWN/IBP fallback).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch.nn as nn

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star
from n2v.nn.layer_ops._norm_utils import affine_after_norm, normalised_interval


def _ln_params(layer: nn.LayerNorm):
    eps = float(layer.eps)
    weight = layer.weight.detach().cpu().numpy().astype(np.float64) if layer.weight is not None else None
    bias = layer.bias.detach().cpu().numpy().astype(np.float64) if layer.bias is not None else None
    return weight, bias, eps


def layernorm_box(layer: nn.LayerNorm, input_boxes: List[Box]) -> List[Box]:
    weight, bias, eps = _ln_params(layer)
    out: List[Box] = []
    for b in input_boxes:
        norm_lb, norm_ub = normalised_interval(b.lb, b.ub, eps=eps)
        out_lb, out_ub = affine_after_norm(norm_lb, norm_ub, weight, bias)
        out.append(Box(out_lb, out_ub))
    return out


def layernorm_star_approx(layer: nn.LayerNorm, input_stars: List[Star]) -> List[Star]:
    weight, bias, eps = _ln_params(layer)

    def _box(lb, ub):
        norm_lb, norm_ub = normalised_interval(lb, ub, eps=eps)
        return affine_after_norm(norm_lb, norm_ub, weight, bias)

    return apply_box_lift_star(input_stars, _box)
