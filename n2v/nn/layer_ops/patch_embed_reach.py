"""PatchEmbed reachability — Conv2d -> flatten(2) -> transpose(1, 2).

PatchEmbed maps a ``(B, C, H, W)`` image into a ``(B, L, embed_dim)``
sequence where ``L = (H // patch_size) * (W // patch_size)``. The
implementation is:

    x = self.proj(x)           # (B, embed_dim, H/p, W/p)
    return x.flatten(2).transpose(1, 2)  # (B, L, embed_dim)

We treat PatchEmbed as an fx leaf (see ``n2v/nn/_tracer.py``) and
implement reach by:

1. Running the existing Conv2d reach helper on the input set. For
   ImageZono / ImageStar inputs this yields an output set in CHW layout
   (channel-major) with the new ``(H/p, W/p, embed_dim)`` spatial shape.

2. Permuting the flat representation from channel-major
   ``(c, h, w)`` to token-major ``(h*w, c)``. After the permutation the
   set carries the **(L, embed_dim)** layout the downstream transformer
   expects when it operates on a flat ``Star/Zono`` of dim ``L * embed_dim``.

This is a sound, exact-affine reach (Conv2d + permutation) and lands as
the keystone for the ViT integration test (``tests/integration/
test_minimal_vit.py``).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops import conv2d_reach


def _channel_major_to_token_major_index(
    n_channels: int, height: int, width: int,
) -> np.ndarray:
    """Build a permutation that takes a flat ``(C, H, W)`` (channel-major)
    layout to ``(H*W, C)`` (token-major) layout.

    The output index ``perm[i]`` gives the source row in the channel-major
    set that should occupy position ``i`` in the token-major layout. For
    ``C=2, H=2, W=2`` the channel-major flat order is
    ``[c0_h0_w0, c0_h0_w1, c0_h1_w0, c0_h1_w1, c1_h0_w0, ...]`` and the
    token-major target is
    ``[c0_h0_w0, c1_h0_w0, c0_h0_w1, c1_h0_w1, c0_h1_w0, c1_h1_w0, ...]``.
    """
    # channel-major index: i_cm = c * (H*W) + h * W + w
    # token-major index:   i_tm = (h * W + w) * C + c
    # We want perm[i_tm] = i_cm.
    perm = np.empty(n_channels * height * width, dtype=np.int64)
    for c in range(n_channels):
        for h in range(height):
            for w in range(width):
                i_cm = c * (height * width) + h * width + w
                i_tm = (h * width + w) * n_channels + c
                perm[i_tm] = i_cm
    return perm


def _permute_rows_flat(
    flat_centre: np.ndarray, flat_generators: np.ndarray, perm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute rows of a flat (centre, generators) pair."""
    return flat_centre[perm], flat_generators[perm]


def _patch_embed_image_zono(layer: nn.Module, input_image_zono: ImageZono) -> Zono:
    """Apply Conv2d via ``conv2d_zono`` then flatten to token-major.

    ``conv2d_zono`` returns an ImageZono with ``(H, W, C, n_gen)`` in V
    and HWC-row-major flat positions. PatchEmbed's forward is
    ``proj(x).flatten(2).transpose(1, 2)`` which produces a
    ``(B, L=H*W, embed_dim=C)`` sequence -- i.e. token-major flat order
    ``(token_idx) * C + c`` for ``token_idx = h * W + w``.

    This token-major flat layout EQUALS the HWC-row-major flat layout
    (worked out: ``h*W*C + w*C + c == (h*W + w) * C + c``), so simply
    calling ``ImageZono.to_zono()`` produces the correct flat Zono. No
    permutation is needed.
    """
    proj: nn.Conv2d = layer.proj  # type: ignore[attr-defined]
    conv_out = conv2d_reach.conv2d_zono(proj, [input_image_zono])
    assert len(conv_out) == 1
    out_set = conv_out[0]
    if isinstance(out_set, ImageZono):
        return out_set.to_zono()
    return out_set


def _patch_embed_image_star(layer: nn.Module, input_image_star: ImageStar) -> Star:
    """Apply Conv2d via ``conv2d_star`` then flatten to token-major.

    See ``_patch_embed_image_zono`` for the layout argument.
    ``ImageStar.to_star()`` flattens in HWC order which matches the
    desired token-major layout.
    """
    proj: nn.Conv2d = layer.proj  # type: ignore[attr-defined]
    star_out = conv2d_reach.conv2d_star(proj, [input_image_star])
    assert len(star_out) == 1
    out_set = star_out[0]
    if isinstance(out_set, ImageStar):
        return out_set.to_star()
    return out_set


def patch_embed_zono(layer: nn.Module, input_sets: List) -> List[Zono]:
    """Sound Zono reach for PatchEmbed.

    Expects each input set to be an :class:`ImageZono` (with explicit
    spatial shape); a flat :class:`Zono` cannot recover the H/W needed
    by Conv2d and will raise.
    """
    out: List[Zono] = []
    for s in input_sets:
        if not isinstance(s, ImageZono):
            raise TypeError(
                f"PatchEmbed Zono reach requires ImageZono input "
                f"(needs H/W/C); got {type(s).__name__}."
            )
        out.append(_patch_embed_image_zono(layer, s))
    return out


def patch_embed_star(layer: nn.Module, input_sets: List, **kwargs) -> List[Star]:
    """Sound Star reach for PatchEmbed.

    Expects each input set to be an :class:`ImageStar`.
    """
    out: List[Star] = []
    for s in input_sets:
        if not isinstance(s, ImageStar):
            raise TypeError(
                f"PatchEmbed Star reach requires ImageStar input "
                f"(needs H/W/C); got {type(s).__name__}."
            )
        out.append(_patch_embed_image_star(layer, s))
    return out


def patch_embed_box(layer: nn.Module, input_sets: List) -> List[Box]:
    """Box reach for PatchEmbed.

    Box does not carry explicit ``(H, W, C)``, but PatchEmbed.proj is a
    fully-affine Conv2d so we can recover a sound (loose) Box by lifting
    each input Box to an ImageZono using the layer's known
    ``in_channels`` and the implied image size from the Box's flat dim,
    running ``patch_embed_zono``, and taking the IBP envelope of the
    result.

    Raises if the flat Box dim is not ``in_channels * h * w`` for some
    square ``h == w``; non-square inputs need explicit ImageZono
    construction at the call site.
    """
    from n2v.sets.image_zono import ImageZono
    from n2v.sets import Zono

    proj: nn.Conv2d = layer.proj  # type: ignore[attr-defined]
    in_c = int(proj.in_channels)
    out: List[Box] = []
    for box in input_sets:
        flat = box.dim
        if flat % in_c != 0:
            raise NotImplementedError(
                f"PatchEmbed Box reach: flat input dim {flat} is not "
                f"divisible by in_channels={in_c}; cannot infer H/W."
            )
        n_pixels = flat // in_c
        # Assume square image (the common case; explicit ImageZono is
        # available for non-square).
        side = int(round(n_pixels ** 0.5))
        if side * side != n_pixels:
            raise NotImplementedError(
                f"PatchEmbed Box reach: cannot infer a square image of "
                f"{n_pixels} pixels; use ImageZono input for non-square."
            )
        zono_in = ImageZono.from_bounds(
            box.lb, box.ub, height=side, width=side, num_channels=in_c,
        )
        zono_out = _patch_embed_image_zono(layer, zono_in)
        lb, ub = zono_out.get_bounds()
        out.append(Box(
            np.asarray(lb).reshape(-1, 1), np.asarray(ub).reshape(-1, 1),
        ))
    return out


def _hex_oct_lb_ub(s):
    """Fast IBP (lb, ub) for Hex/Oct -- their zero-arg ``estimate_ranges``."""
    lb, ub = s.estimate_ranges()
    return np.asarray(lb).reshape(-1, 1), np.asarray(ub).reshape(-1, 1)


def patch_embed_hexatope(layer: nn.Module, input_sets: List):
    """Sound (box-lifted) Hexatope reach for PatchEmbed.

    Lifts each Hexatope to its IBP box envelope, runs ``patch_embed_box``
    (which infers a square image shape), then constructs a fresh
    Hexatope from the result. Loose but sound.
    """
    from n2v.sets import Hexatope

    out = []
    for h in input_sets:
        lb, ub = _hex_oct_lb_ub(h)
        box_in = Box(lb, ub)
        box_out = patch_embed_box(layer, [box_in])[0]
        out.append(Hexatope.from_bounds(box_out.lb, box_out.ub))
    return out


def patch_embed_octatope(layer: nn.Module, input_sets: List):
    """Sound (box-lifted) Octatope reach for PatchEmbed.

    Same box-lift pattern as ``patch_embed_hexatope``.
    """
    from n2v.sets import Octatope

    out = []
    for o in input_sets:
        lb, ub = _hex_oct_lb_ub(o)
        box_in = Box(lb, ub)
        box_out = patch_embed_box(layer, [box_in])[0]
        out.append(Octatope.from_bounds(box_out.lb, box_out.ub))
    return out
