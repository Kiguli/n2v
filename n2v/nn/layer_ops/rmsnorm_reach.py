"""RMSNorm reachability.

RMSNorm: ``y = x / sqrt(mean(x^2) + eps) * gamma``. Operates over the
last axis (a ``normalized_shape``-sized group) of ``x``. For
transformer inputs flattened from ``(L, D)`` the layer normalises each
``D``-element chunk independently; the reach paths mirror that
structure by chunking the flat input into ``L`` groups of size ``D``.

Coverage matches nnVLA: Box (IBP), Star (predicate-preserving).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar
from n2v.nn.layer_ops._image_shape import apply_box_lift_star
from n2v.nn.layer_ops._layernorm_star import predicate_preserving_norm_star
from n2v.nn.layer_ops._norm_utils import affine_after_norm


def _rms_params(layer):
    eps = float(getattr(layer, "eps", 1e-6))
    weight = None
    if getattr(layer, "weight", None) is not None:
        weight = layer.weight.detach().cpu().numpy().astype(np.float64)
    D = int(getattr(layer, "normalized_shape", weight.size if weight is not None else 0))
    if D == 0:
        raise ValueError("RMSNorm wrapper must expose normalized_shape or a weight tensor.")
    return weight, eps, D


def _rms_interval_chunk(chunk_lb: np.ndarray, chunk_ub: np.ndarray, eps: float):
    sq_lb = np.where(
        np.sign(chunk_lb) == np.sign(chunk_ub),
        np.minimum(chunk_lb ** 2, chunk_ub ** 2),
        0.0,
    )
    sq_ub = np.maximum(chunk_lb ** 2, chunk_ub ** 2)
    rms_lb = float(np.sqrt(sq_lb.mean() + eps))
    rms_ub = float(np.sqrt(sq_ub.mean() + eps))
    s_lb = 1.0 / rms_ub
    s_ub = 1.0 / rms_lb
    cands = np.stack([s_lb * chunk_lb, s_lb * chunk_ub, s_ub * chunk_lb, s_ub * chunk_ub])
    return cands.min(axis=0), cands.max(axis=0)


def _rms_interval_per_group(lb: np.ndarray, ub: np.ndarray, D: int, eps: float):
    flat_lb = lb.reshape(-1).astype(np.float64)
    flat_ub = ub.reshape(-1).astype(np.float64)
    if flat_lb.size % D != 0:
        raise ValueError(
            f"RMSNorm flat input dim {flat_lb.size} is not divisible by "
            f"normalized_shape={D}. The concrete forward normalises each "
            f"D-element group independently."
        )
    L = flat_lb.size // D
    out_lb = np.zeros_like(flat_lb)
    out_ub = np.zeros_like(flat_ub)
    for i in range(L):
        start = i * D
        end = start + D
        c_lb, c_ub = _rms_interval_chunk(flat_lb[start:end], flat_ub[start:end], eps)
        out_lb[start:end] = c_lb
        out_ub[start:end] = c_ub
    return out_lb.reshape(-1, 1), out_ub.reshape(-1, 1), L


def _broadcast_weight(weight: Optional[np.ndarray], L: int):
    return np.tile(weight, L) if weight is not None else None


def rmsnorm_box(layer, input_boxes: List[Box]) -> List[Box]:
    weight, eps, D = _rms_params(layer)
    out: List[Box] = []
    for b in input_boxes:
        norm_lb, norm_ub, L = _rms_interval_per_group(b.lb, b.ub, D, eps)
        w_b = _broadcast_weight(weight, L)
        out_lb, out_ub = affine_after_norm(norm_lb, norm_ub, w_b, None)
        out.append(Box(out_lb, out_ub))
    return out


def rmsnorm_star_approx(layer, input_stars: List[Star]) -> List[Star]:
    """Predicate-preserving Star reach for RMSNorm.

    Bounds ``rms(x)`` per ``normalized_shape``-sized group with
    interval arithmetic, then applies the scale ``1/rms`` via a
    midpoint affine map plus per-feature slack predicates. Uses a
    single conservative ``[rms_lb, rms_ub]`` interval across all groups
    for the predicate-preserving path; per-group sigma tightening is
    a follow-up.
    """
    weight, eps, D = _rms_params(layer)
    output: List[Star] = []
    for s in input_stars:
        is_image = isinstance(s, ImageStar)
        base = s.to_star() if is_image else s
        if base.V is None or base.V.size == 0:
            lb, ub = base.estimate_ranges()
            norm_lb, norm_ub, L = _rms_interval_per_group(lb, ub, D, eps)
            w_b = _broadcast_weight(weight, L)
            out_lb, out_ub = affine_after_norm(norm_lb, norm_ub, w_b, None)
            new_star = Star.from_bounds(out_lb, out_ub)
        else:
            lb, ub = base.estimate_ranges()
            if lb.size % D != 0:
                raise ValueError(
                    f"RMSNorm flat input dim {lb.size} is not divisible by "
                    f"normalized_shape={D}."
                )
            L = lb.size // D
            # Bound x^2 conservatively across the whole input for sigma
            # interval; tighter per-group bounds are a follow-up.
            lb_flat = lb.reshape(-1)
            ub_flat = ub.reshape(-1)
            sq_lb = np.where(
                np.sign(lb_flat) == np.sign(ub_flat),
                np.minimum(lb_flat ** 2, ub_flat ** 2),
                0.0,
            )
            sq_ub = np.maximum(lb_flat ** 2, ub_flat ** 2)
            rms_lb = float(np.sqrt(sq_lb.mean() + eps))
            rms_ub = float(np.sqrt(sq_ub.mean() + eps))
            w_b = _broadcast_weight(weight, L)
            new_star = predicate_preserving_norm_star(
                base,
                sigma_bounds=(rms_lb, rms_ub),
                weight=w_b,
                bias=None,
                subtract_mean=False,
            )
        if is_image:
            new_star = new_star.to_image_star(s.height, s.width, s.num_channels)
        output.append(new_star)
    return output
