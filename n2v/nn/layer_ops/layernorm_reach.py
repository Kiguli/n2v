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
from n2v.sets.image_star import ImageStar
from n2v.nn.layer_ops._image_shape import apply_box_lift_star
from n2v.nn.layer_ops._layernorm_star import predicate_preserving_norm_star
from n2v.nn.layer_ops._norm_utils import (
    affine_after_norm,
    interval_mean_var,
    normalised_interval,
)


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
    """Predicate-preserving Star reach for LayerNorm.

    Subtracts the input mean exactly (linear, preserves predicates) and
    applies the scale ``1/sigma`` via a midpoint affine map plus a per-
    feature slack predicate. The fallback box-lift is used when the
    input has no predicate basis (e.g. a degenerate Star).
    """
    weight, bias, eps = _ln_params(layer)
    output: List[Star] = []
    for s in input_stars:
        is_image = isinstance(s, ImageStar)
        base = s.to_star() if is_image else s
        if base.V is None or base.V.size == 0:
            # Fall through to box-lift; nothing to preserve.
            lb, ub = base.estimate_ranges()
            norm_lb, norm_ub = normalised_interval(lb, ub, eps=eps)
            out_lb, out_ub = affine_after_norm(norm_lb, norm_ub, weight, bias)
            new_star = Star.from_bounds(out_lb, out_ub)
        else:
            lb, ub = base.estimate_ranges()
            _, _, var_lb, var_ub = interval_mean_var(lb, ub)
            sigma_lb = float(np.sqrt(float(var_lb) + eps))
            sigma_ub = float(np.sqrt(float(var_ub) + eps))
            new_star = predicate_preserving_norm_star(
                base,
                sigma_bounds=(sigma_lb, sigma_ub),
                weight=weight,
                bias=bias,
                subtract_mean=True,
            )
        if is_image:
            new_star = new_star.to_image_star(s.height, s.width, s.num_channels)
        output.append(new_star)
    return output
