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
    """Apply Conv2d via ``conv2d_zono`` then permute to token-major."""
    proj: nn.Conv2d = layer.proj  # type: ignore[attr-defined]
    conv_out = conv2d_reach.conv2d_zono(proj, [input_image_zono])
    assert len(conv_out) == 1
    out_set = conv_out[0]
    # ``conv2d_zono`` returns an ImageZono with (height, width, num_channels)
    # in the layout matching nnv's ``(H, W, C)`` (HWC) flatten convention.
    out_h = out_set.height
    out_w = out_set.width
    out_c = out_set.num_channels
    # Build the permutation that maps the underlying HWC-flat layout to the
    # token-major (H*W, C) layout the downstream transformer expects.
    perm = np.empty(out_h * out_w * out_c, dtype=np.int64)
    for h in range(out_h):
        for w in range(out_w):
            for c in range(out_c):
                # Source position in HWC-major flat:
                i_src = h * (out_w * out_c) + w * out_c + c
                # Target position in token-major (token = h*w, feature = c):
                i_dst = (h * out_w + w) * out_c + c
                perm[i_dst] = i_src
    # In this canonical layout HWC-flat order already matches token-major
    # order (token_idx = h*out_w + w, feature = c iterating last). So perm
    # is the identity. Keep the construction for clarity; if the
    # underlying conv reach changes layout, only the permutation block
    # below needs updating.
    new_c = out_set.c[perm]
    new_V = out_set.V[perm]
    # Return as a flat Zono (drops the spatial structure; downstream
    # transformer ops operate on (L, embed_dim) flat).
    return Zono(new_c, new_V)


def _patch_embed_image_star(layer: nn.Module, input_image_star: ImageStar) -> Star:
    """Apply Conv2d via ``conv2d_star`` then permute to token-major."""
    proj: nn.Conv2d = layer.proj  # type: ignore[attr-defined]
    star_out = conv2d_reach.conv2d_star(proj, [input_image_star])
    assert len(star_out) == 1
    out_set = star_out[0]
    if isinstance(out_set, ImageStar):
        out_h = out_set.height
        out_w = out_set.width
        out_c = out_set.num_channels
        # Permute V's rows to token-major. Same construction as the
        # ImageZono path -- with HWC underlying flatten the permutation
        # collapses to identity, but kept for layout-change robustness.
        perm = np.empty(out_h * out_w * out_c, dtype=np.int64)
        for h in range(out_h):
            for w in range(out_w):
                for c in range(out_c):
                    perm[(h * out_w + w) * out_c + c] = (
                        h * (out_w * out_c) + w * out_c + c
                    )
        new_V = out_set.V[perm]
        return Star(new_V, out_set.C, out_set.d,
                    out_set.predicate_lb, out_set.predicate_ub)
    # If conv2d_star happened to return a flat Star already, the identity
    # is the correct permutation.
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
    """Box reach via conv2d_box; raises because conv2d_box requires
    image dimensions that Box does not carry."""
    raise NotImplementedError(
        "PatchEmbed Box reach requires explicit image dimensions. "
        "Use ImageZono or ImageStar input set so the Conv2d helper "
        "can recover H/W/C."
    )
