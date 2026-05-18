"""GroupNorm reachability.

GroupNorm partitions channels into ``num_groups`` groups and applies
LayerNorm independently within each group. The per-group bounds use the
same interval-mean / interval-variance derivation as :mod:`layernorm_reach`,
applied to the per-group sub-vectors of the input.

Coverage matches nnVLA: Box (IBP), Star (CROWN/IBP fallback).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch.nn as nn

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star
from n2v.nn.layer_ops._norm_utils import affine_after_norm, normalised_interval


def _gn_params(layer: nn.GroupNorm):
    eps = float(layer.eps)
    num_groups = int(layer.num_groups)
    num_channels = int(layer.num_channels)
    weight = layer.weight.detach().cpu().numpy().astype(np.float64) if layer.weight is not None else None
    bias = layer.bias.detach().cpu().numpy().astype(np.float64) if layer.bias is not None else None
    return num_groups, num_channels, weight, bias, eps


def _groupnorm_interval(lb: np.ndarray, ub: np.ndarray, num_groups: int, num_channels: int, eps: float):
    """Sound interval reach for GroupNorm applied to an ``(C, ...)`` input.

    The input bounds are reshaped per-channel; channels are split into
    ``num_groups`` groups and each group is bounded by the LayerNorm
    interval helper.
    """
    lb = lb.reshape(-1)
    ub = ub.reshape(-1)
    if lb.size % num_channels != 0:
        raise ValueError(
            f"GroupNorm input length {lb.size} not divisible by num_channels={num_channels}"
        )
    spatial = lb.size // num_channels
    lb_c = lb.reshape(num_channels, spatial)
    ub_c = ub.reshape(num_channels, spatial)

    channels_per_group = num_channels // num_groups
    out_lb = np.zeros_like(lb_c)
    out_ub = np.zeros_like(ub_c)

    for g in range(num_groups):
        start = g * channels_per_group
        end = start + channels_per_group
        group_lb = lb_c[start:end].reshape(-1)
        group_ub = ub_c[start:end].reshape(-1)
        n_lb, n_ub = normalised_interval(group_lb, group_ub, eps=eps)
        out_lb[start:end] = n_lb.reshape(channels_per_group, spatial)
        out_ub[start:end] = n_ub.reshape(channels_per_group, spatial)

    return out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)


def groupnorm_box(layer: nn.GroupNorm, input_boxes: List[Box]) -> List[Box]:
    num_groups, num_channels, weight, bias, eps = _gn_params(layer)
    out: List[Box] = []
    for b in input_boxes:
        norm_lb, norm_ub = _groupnorm_interval(b.lb, b.ub, num_groups, num_channels, eps)
        # weight/bias are per-channel; broadcast across spatial.
        if weight is not None or bias is not None:
            spatial = norm_lb.size // num_channels
            w_b = np.repeat(weight, spatial) if weight is not None else None
            b_b = np.repeat(bias, spatial) if bias is not None else None
            norm_lb, norm_ub = affine_after_norm(norm_lb, norm_ub, w_b, b_b)
        out.append(Box(norm_lb, norm_ub))
    return out


def groupnorm_star_approx(layer: nn.GroupNorm, input_stars: List[Star]) -> List[Star]:
    num_groups, num_channels, weight, bias, eps = _gn_params(layer)

    def _box(lb, ub):
        norm_lb, norm_ub = _groupnorm_interval(lb, ub, num_groups, num_channels, eps)
        if weight is not None or bias is not None:
            spatial = norm_lb.size // num_channels
            w_b = np.repeat(weight, spatial) if weight is not None else None
            b_b = np.repeat(bias, spatial) if bias is not None else None
            norm_lb, norm_ub = affine_after_norm(norm_lb, norm_ub, w_b, b_b)
        return norm_lb, norm_ub

    return apply_box_lift_star(input_stars, _box)
