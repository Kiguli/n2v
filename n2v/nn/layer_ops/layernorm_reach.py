"""LayerNorm reachability.

LayerNorm: ``y = (x - mu) / sqrt(var + eps) * gamma + beta`` where mean
and variance are taken over each ``normalized_shape``-sized last-axis
group of ``x``. For transformer inputs the flattened set typically
holds an ``(L, D)`` sequence as ``L*D`` flat features, in which case
LayerNorm operates *per token* (each ``D``-element chunk independently).

The reach paths split the flat input into ``L`` groups of size ``D``,
apply the interval / predicate-preserving normalisation per group, and
re-stack. Inputs whose flat dim is not a multiple of ``D`` are
rejected with a clear error.

Coverage matches nnVLA: Box (IBP), Star (predicate-preserving).
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
    shape = layer.normalized_shape
    if isinstance(shape, (tuple, list)):
        D = int(np.prod(shape))
    else:
        D = int(shape)
    return weight, bias, eps, D


def _layernorm_interval_per_group(lb: np.ndarray, ub: np.ndarray, D: int, eps: float):
    """Bound LayerNorm output by normalising each ``D``-element group."""
    flat_lb = lb.reshape(-1)
    flat_ub = ub.reshape(-1)
    if flat_lb.size % D != 0:
        raise ValueError(
            f"LayerNorm flat input dim {flat_lb.size} is not divisible by "
            f"normalized_shape={D}. The concrete forward normalises each "
            f"D-element group independently, so the input must hold an "
            f"(L, D) sequence flattened to L*D features."
        )
    L = flat_lb.size // D
    chunks_lb = flat_lb.reshape(L, D)
    chunks_ub = flat_ub.reshape(L, D)
    out_lb = np.zeros_like(chunks_lb)
    out_ub = np.zeros_like(chunks_ub)
    for i in range(L):
        n_lb, n_ub = normalised_interval(chunks_lb[i], chunks_ub[i], eps=eps)
        out_lb[i] = n_lb.reshape(-1)
        out_ub[i] = n_ub.reshape(-1)
    return out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)


def _broadcast_affine(weight: np.ndarray | None, bias: np.ndarray | None, L: int):
    """Tile per-feature weight/bias across ``L`` groups."""
    w = np.tile(weight, L) if weight is not None else None
    b = np.tile(bias, L) if bias is not None else None
    return w, b


def layernorm_box(layer: nn.LayerNorm, input_boxes: List[Box]) -> List[Box]:
    weight, bias, eps, D = _ln_params(layer)
    out: List[Box] = []
    for b in input_boxes:
        norm_lb, norm_ub = _layernorm_interval_per_group(b.lb, b.ub, D, eps)
        L = norm_lb.size // D
        w_b, b_b = _broadcast_affine(weight, bias, L)
        out_lb, out_ub = affine_after_norm(norm_lb, norm_ub, w_b, b_b)
        out.append(Box(out_lb, out_ub))
    return out


def layernorm_star_approx(layer: nn.LayerNorm, input_stars: List[Star]) -> List[Star]:
    """Predicate-preserving Star reach for LayerNorm.

    Subtracts the per-group input mean exactly (linear, preserves
    predicates) and applies the ``1/sigma`` scale via a midpoint
    affine map plus a per-feature slack predicate. Currently uses a
    single conservative ``[sigma_lb, sigma_ub]`` interval over the
    whole flattened input — refining to per-group sigma bounds is a
    tightening follow-up.
    """
    weight, bias, eps, D = _ln_params(layer)
    output: List[Star] = []
    for s in input_stars:
        is_image = isinstance(s, ImageStar)
        base = s.to_star() if is_image else s
        if base.V is None or base.V.size == 0:
            lb, ub = base.estimate_ranges()
            norm_lb, norm_ub = _layernorm_interval_per_group(lb, ub, D, eps)
            L = norm_lb.size // D
            w_b, b_b = _broadcast_affine(weight, bias, L)
            out_lb, out_ub = affine_after_norm(norm_lb, norm_ub, w_b, b_b)
            new_star = Star.from_bounds(out_lb, out_ub)
        else:
            lb, ub = base.estimate_ranges()
            if lb.size % D != 0:
                raise ValueError(
                    f"LayerNorm flat input dim {lb.size} is not divisible "
                    f"by normalized_shape={D}."
                )
            L = lb.size // D
            _, _, var_lb, var_ub = interval_mean_var(lb, ub)
            sigma_lb = float(np.sqrt(np.asarray(var_lb).item() + eps))
            sigma_ub = float(np.sqrt(np.asarray(var_ub).item() + eps))
            w_b, b_b = _broadcast_affine(weight, bias, L)
            new_star = predicate_preserving_norm_star(
                base,
                sigma_bounds=(sigma_lb, sigma_ub),
                weight=w_b,
                bias=b_b,
                subtract_mean=True,
            )
        if is_image:
            new_star = new_star.to_image_star(s.height, s.width, s.num_channels)
        output.append(new_star)
    return output
