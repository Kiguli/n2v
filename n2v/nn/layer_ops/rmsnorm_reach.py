"""RMSNorm reachability.

RMSNorm: ``y = x / sqrt(mean(x^2) + eps) * gamma``. Like LayerNorm,
this is non-affine; we use interval bounds on ``rms`` to derive a sound
axis-aligned output box.

Coverage matches nnVLA: Box (IBP), Star (CROWN/IBP fallback).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar
from n2v.nn.layer_ops._norm_utils import affine_after_norm


def _rms_params(layer):
    eps = float(getattr(layer, "eps", 1e-6))
    weight = None
    if getattr(layer, "weight", None) is not None:
        weight = layer.weight.detach().cpu().numpy().astype(np.float64)
    return weight, eps


def _rms_interval(lb: np.ndarray, ub: np.ndarray, eps: float):
    lb = lb.reshape(-1).astype(np.float64)
    ub = ub.reshape(-1).astype(np.float64)
    # bound on x^2 elementwise
    sq_lb = np.where(np.sign(lb) == np.sign(ub), np.minimum(lb ** 2, ub ** 2), 0.0)
    sq_ub = np.maximum(lb ** 2, ub ** 2)
    rms_lb = float(np.sqrt(sq_lb.mean() + eps))
    rms_ub = float(np.sqrt(sq_ub.mean() + eps))
    s_lb = 1.0 / rms_ub
    s_ub = 1.0 / rms_lb
    # y_i ∈ s * x_i ; s > 0 always, so monotone in x sign-wise
    cands = np.stack([
        s_lb * lb, s_lb * ub, s_ub * lb, s_ub * ub
    ])
    out_lb = cands.min(axis=0)
    out_ub = cands.max(axis=0)
    return out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)


def rmsnorm_box(layer, input_boxes: List[Box]) -> List[Box]:
    weight, eps = _rms_params(layer)
    out: List[Box] = []
    for b in input_boxes:
        norm_lb, norm_ub = _rms_interval(b.lb, b.ub, eps=eps)
        out_lb, out_ub = affine_after_norm(norm_lb, norm_ub, weight, None)
        out.append(Box(out_lb, out_ub))
    return out


def rmsnorm_star_approx(layer, input_stars: List[Star]) -> List[Star]:
    weight, eps = _rms_params(layer)
    output: List[Star] = []
    for s in input_stars:
        base = s.to_star() if isinstance(s, ImageStar) else s
        lb, ub = base.estimate_ranges()
        norm_lb, norm_ub = _rms_interval(lb, ub, eps=eps)
        out_lb, out_ub = affine_after_norm(norm_lb, norm_ub, weight, None)
        out = Star.from_bounds(out_lb, out_ub)
        if isinstance(s, ImageStar):
            out = out.to_image_star(s.height, s.width, s.num_channels)
        output.append(out)
    return output
