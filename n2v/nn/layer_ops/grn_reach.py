"""GRN (Global Response Normalisation) reachability (ConvNeXt-v2).

GRN computes a per-channel L2 norm across (H, W), normalises by its
spatial mean, and applies an affine ``gamma * (x * nx) + beta + x``.
Sound interval bounds use the worst-case L2 norm of the per-channel
slice; Star reach reuses the Box bound.

Coverage matches nnVLA: Box (IBP), Star (CROWN/IBP fallback).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar
from n2v.nn.layer_ops._norm_utils import affine_after_norm


def _grn_params(layer):
    eps = float(getattr(layer, "eps", 1e-6))
    dim = int(getattr(layer, "dim", 0))
    gamma = layer.gamma.detach().cpu().numpy().astype(np.float64).reshape(-1) if hasattr(layer, "gamma") else None
    beta = layer.beta.detach().cpu().numpy().astype(np.float64).reshape(-1) if hasattr(layer, "beta") else None
    return dim, gamma, beta, eps


def _grn_interval(lb: np.ndarray, ub: np.ndarray, dim: int, eps: float):
    """Sound interval for GRN forward applied to an (H*W*C,) flat vector."""
    lb = lb.reshape(-1).astype(np.float64)
    ub = ub.reshape(-1).astype(np.float64)
    n_total = lb.size
    if dim <= 0 or n_total % dim != 0:
        # Fall back to elementwise residual-pass-through if shape is unknown.
        return lb.reshape(-1, 1), ub.reshape(-1, 1)
    hw = n_total // dim
    # per-channel max |x|
    abs_max = np.maximum(np.abs(lb), np.abs(ub))
    abs_max_c = abs_max.reshape(hw, dim).max(axis=0)  # per-channel
    # ||x_c||_2 ≤ sqrt(hw) * abs_max_c
    gx_ub = np.sqrt(hw) * abs_max_c
    gx_lb = np.zeros_like(gx_ub)
    # nx = gx / (mean(gx) + eps)
    mean_gx_lb = gx_lb.mean()
    mean_gx_ub = gx_ub.mean()
    nx_ub = gx_ub / (mean_gx_lb + eps)
    nx_lb = gx_lb / (mean_gx_ub + eps)
    # x * nx (broadcast nx over spatial dim) — bound elementwise
    nx_lb_b = np.broadcast_to(nx_lb, (hw, dim)).reshape(-1)
    nx_ub_b = np.broadcast_to(nx_ub, (hw, dim)).reshape(-1)
    cands = np.stack([
        lb * nx_lb_b, lb * nx_ub_b, ub * nx_lb_b, ub * nx_ub_b
    ])
    xnx_lb = cands.min(axis=0)
    xnx_ub = cands.max(axis=0)
    # y = gamma * xnx + beta + x; gamma/beta are per-channel
    # Output preliminary interval:
    return xnx_lb.reshape(-1, 1), xnx_ub.reshape(-1, 1)


def grn_box(layer, input_boxes: List[Box]) -> List[Box]:
    dim, gamma, beta, eps = _grn_params(layer)
    out: List[Box] = []
    for b in input_boxes:
        xnx_lb, xnx_ub = _grn_interval(b.lb, b.ub, dim, eps)
        if dim > 0 and gamma is not None:
            hw = xnx_lb.size // dim
            g_b = np.tile(gamma, hw).reshape(-1, 1)
            be_b = np.tile(beta, hw).reshape(-1, 1) if beta is not None else None
            xnx_lb, xnx_ub = affine_after_norm(xnx_lb, xnx_ub, g_b, be_b)
        # residual + x
        out_lb = xnx_lb + b.lb.reshape(-1, 1)
        out_ub = xnx_ub + b.ub.reshape(-1, 1)
        out.append(Box(out_lb, out_ub))
    return out


def grn_star_approx(layer, input_stars: List[Star]) -> List[Star]:
    dim, gamma, beta, eps = _grn_params(layer)
    output: List[Star] = []
    for s in input_stars:
        base = s.to_star() if isinstance(s, ImageStar) else s
        lb, ub = base.estimate_ranges()
        xnx_lb, xnx_ub = _grn_interval(lb, ub, dim, eps)
        if dim > 0 and gamma is not None:
            hw = xnx_lb.size // dim
            g_b = np.tile(gamma, hw).reshape(-1, 1)
            be_b = np.tile(beta, hw).reshape(-1, 1) if beta is not None else None
            xnx_lb, xnx_ub = affine_after_norm(xnx_lb, xnx_ub, g_b, be_b)
        out_lb = xnx_lb + lb.reshape(-1, 1)
        out_ub = xnx_ub + ub.reshape(-1, 1)
        out = Star.from_bounds(out_lb, out_ub)
        if isinstance(s, ImageStar):
            out = out.to_image_star(s.height, s.width, s.num_channels)
        output.append(out)
    return output
