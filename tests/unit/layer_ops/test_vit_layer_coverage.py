"""Set-type coverage tests for every layer / primitive this PR touched.

The ViT integration test (``tests/integration/test_minimal_vit.py``)
only exercises the Zono path because that is what the benchmark needs.
This file pins **every** set type for each layer/primitive added or
modified by this PR, so future regressions on Box / Star / Hexatope /
Octatope are caught by CI:

  * PatchEmbed           — Box, Star, Zono, Hexatope, Octatope
  * LayerNorm (single-token)  — Box, Star, Zono, Hexatope, Octatope
  * GELU (erf + tanh)    — Box, Star, Zono, Hexatope, Octatope
  * Linear (per-token block-tile) — Box, Star, Zono, Hexatope, Octatope
  * fx ``operator.add`` (set + constant)  — Star, Zono, Box, Hex, Oct
  * fx ``operator.getitem`` (tensor slice) — Star, Zono, Box, Hex, Oct
  * SoftmaxAttention multi-input  — Box, Star, Zono, Hex, Oct

Each test instantiates the relevant layer / op, builds a small input
set of the right type, runs the reach helper directly via the
dispatcher, and asserts shape + finite bounds + box containment of
one concrete forward sample where applicable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono, Hexatope, Octatope
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops import dispatcher
from n2v.nn.layer_ops import (
    layernorm_reach,
    gelu_reach,
    linear_reach,
    patch_embed_reach,
)
from n2v.nn.layers import PatchEmbed


# ----------------------------- helpers ---------------------------------------


def _flat_box(lb_vec: np.ndarray, ub_vec: np.ndarray) -> Box:
    return Box(lb_vec.reshape(-1, 1).astype(np.float64),
               ub_vec.reshape(-1, 1).astype(np.float64))


def _flat_star(box: Box) -> Star:
    return Star.from_bounds(box.lb, box.ub)


def _flat_zono(box: Box) -> Zono:
    return Zono.from_bounds(box.lb, box.ub)


def _flat_hex(box: Box) -> Hexatope:
    return Hexatope.from_bounds(box.lb, box.ub)


def _flat_oct(box: Box) -> Octatope:
    return Octatope.from_bounds(box.lb, box.ub)


def _bounds_of(s):
    """Fast IBP (lb, ub). Hex/Oct use estimate_ranges (no solver kwarg)."""
    if isinstance(s, (Hexatope, Octatope)):
        lb, ub = s.estimate_ranges()
    elif hasattr(s, "get_bounds"):
        lb, ub = s.get_bounds()
    else:
        lb, ub = s.get_ranges()
    return np.asarray(lb).flatten(), np.asarray(ub).flatten()


# ----------------------------- LayerNorm -------------------------------------


def test_layernorm_box_star_zono_hex_oct_single_token():
    layer = nn.LayerNorm(4)
    layer.eval()
    lb = np.array([-1.0, -0.5, 0.0, 0.5])
    ub = np.array([0.0, 0.5, 1.0, 1.5])
    box = _flat_box(lb, ub)
    for set_obj, helper in (
        (box, lambda: layernorm_reach.layernorm_box(layer, [box])),
        (_flat_star(box), lambda: layernorm_reach.layernorm_star_approx(layer, [_flat_star(box)])),
        (_flat_zono(box), lambda: layernorm_reach.layernorm_zono(layer, [_flat_zono(box)])),
        (_flat_hex(box), lambda: layernorm_reach.layernorm_hexatope(layer, [_flat_hex(box)])),
        (_flat_oct(box), lambda: layernorm_reach.layernorm_octatope(layer, [_flat_oct(box)])),
    ):
        out = helper()
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        assert np.all(np.isfinite(lb_o)) and np.all(np.isfinite(ub_o))
        assert lb_o.size == 4


# ----------------------------- GELU (erf + tanh) -----------------------------


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_box_star_zono_hex_oct(approximate):
    layer = nn.GELU(approximate=approximate)
    layer.eval()
    lb = np.array([-2.0, -0.5, 0.0])
    ub = np.array([-1.0, 0.5, 1.0])
    box = _flat_box(lb, ub)

    for set_in in (
        box, _flat_star(box), _flat_zono(box), _flat_hex(box), _flat_oct(box),
    ):
        out = dispatcher.reach_layer(layer, [set_in], "approx")
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        assert lb_o.size == 3
        assert np.all(np.isfinite(lb_o)) and np.all(np.isfinite(ub_o))
        # Soundness: a sample at x_lb / x_ub through the true forward must
        # be inside the reach bounds.
        with torch.no_grad():
            x = torch.from_numpy(np.stack([lb, ub])).double()
            true_out = layer(x.float()).double().numpy()
        assert np.all(true_out.min(axis=0) >= lb_o - 1e-5)
        assert np.all(true_out.max(axis=0) <= ub_o + 1e-5)


# ----------------------------- Linear block-tile (per-token) -----------------


def test_linear_block_tile_across_all_set_types():
    """nn.Linear applied to a sequence-flattened (L*D_in,) input must
    block-tile the weight so each token's D_in chunk is mapped
    independently. Pins the fix in linear_reach._maybe_block_tile_linear
    across Box / Star / Zono / Hex / Oct.
    """
    torch.manual_seed(0)
    layer = nn.Linear(3, 2, bias=True)
    layer.eval()
    L = 4
    in_dim = L * 3  # L tokens of D_in=3
    lb = np.zeros(in_dim)
    ub = np.ones(in_dim)
    box = _flat_box(lb, ub)

    for set_in, fn in (
        (box, linear_reach.linear_box),
        (_flat_star(box), linear_reach.linear_star),
        (_flat_zono(box), linear_reach.linear_zono),
        (_flat_hex(box), linear_reach.linear_hexatope),
        (_flat_oct(box), linear_reach.linear_octatope),
    ):
        out = fn(layer, [set_in])
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        assert lb_o.size == L * 2  # L tokens of D_out=2


# ----------------------------- PatchEmbed ------------------------------------


@pytest.fixture
def patch_embed_layer():
    torch.manual_seed(0)
    return PatchEmbed(in_channels=1, embed_dim=2, patch_size=2)


def test_patch_embed_zono(patch_embed_layer):
    img_size = 4
    box = _flat_box(
        np.zeros(img_size * img_size).astype(np.float64),
        np.ones(img_size * img_size).astype(np.float64),
    )
    zono_in = ImageZono.from_bounds(
        box.lb, box.ub, height=img_size, width=img_size, num_channels=1,
    )
    out = patch_embed_reach.patch_embed_zono(patch_embed_layer, [zono_in])
    assert len(out) == 1
    lb_o, ub_o = out[0].get_bounds()
    expected_dim = ((img_size // 2) ** 2) * 2  # n_patches * embed_dim
    assert lb_o.size == expected_dim


def test_patch_embed_star(patch_embed_layer):
    img_size = 4
    star_in = ImageStar.from_bounds(
        np.zeros((img_size * img_size, 1)),
        np.ones((img_size * img_size, 1)),
        height=img_size, width=img_size, num_channels=1,
    )
    out = patch_embed_reach.patch_embed_star(patch_embed_layer, [star_in])
    assert len(out) == 1
    expected_dim = ((img_size // 2) ** 2) * 2
    assert out[0].dim == expected_dim


def test_patch_embed_box(patch_embed_layer):
    """Box reach for PatchEmbed lifts to ImageZono internally."""
    img_size = 4
    box = _flat_box(
        np.zeros(img_size * img_size), np.ones(img_size * img_size),
    )
    out = patch_embed_reach.patch_embed_box(patch_embed_layer, [box])
    assert len(out) == 1
    expected_dim = ((img_size // 2) ** 2) * 2
    assert out[0].dim == expected_dim


def test_patch_embed_hexatope(patch_embed_layer):
    img_size = 4
    box = _flat_box(
        np.zeros(img_size * img_size), np.ones(img_size * img_size),
    )
    hex_in = Hexatope.from_bounds(box.lb, box.ub)
    out = patch_embed_reach.patch_embed_hexatope(patch_embed_layer, [hex_in])
    assert len(out) == 1
    expected_dim = ((img_size // 2) ** 2) * 2
    lb_o, ub_o = _bounds_of(out[0])
    assert lb_o.size == expected_dim


def test_patch_embed_octatope(patch_embed_layer):
    img_size = 4
    box = _flat_box(
        np.zeros(img_size * img_size), np.ones(img_size * img_size),
    )
    oct_in = Octatope.from_bounds(box.lb, box.ub)
    out = patch_embed_reach.patch_embed_octatope(patch_embed_layer, [oct_in])
    assert len(out) == 1
    expected_dim = ((img_size // 2) ** 2) * 2
    lb_o, ub_o = _bounds_of(out[0])
    assert lb_o.size == expected_dim


# ----------------------------- fx operator.add (set + constant) -------------


def test_fx_add_set_plus_constant_all_set_types():
    """End-to-end via a tiny model: ``x + buffer`` (pos-embed-style add).
    The fx call_function handler must accept Star / Zono / Box /
    Hexatope / Octatope inputs and produce the correctly-translated
    output set.
    """
    from n2v.nn import NeuralNetwork

    class AddConst(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("c", torch.tensor([1.0, 2.0]))

        def forward(self, x):
            return x + self.c

    model = AddConst().eval()
    box = _flat_box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    for set_in, ctor in (
        (box, Box),
        (_flat_star(box), Star),
        (_flat_zono(box), Zono),
        (_flat_hex(box), Hexatope),
        (_flat_oct(box), Octatope),
    ):
        out = NeuralNetwork(model).reach(set_in, method="approx")
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        # Output = input + [1, 2]. Inner dim 0: [1, 2]; dim 1: [2, 3].
        np.testing.assert_allclose(lb_o, np.array([1.0, 2.0]), atol=1e-9)
        np.testing.assert_allclose(ub_o, np.array([2.0, 3.0]), atol=1e-9)


# ----------------------------- fx operator.getitem (tensor slice) ----------


def test_fx_getitem_slice_all_set_types():
    """End-to-end via a tiny model: ``x[:, 0]`` extracts the first token
    of a sequence-flattened reach set. Pins the fx call_function
    operator.getitem handler across all five set types.
    """
    from n2v.nn import NeuralNetwork

    class SliceFirstToken(nn.Module):
        n_tokens = 2  # picked up by the getitem inference fallback

        def __init__(self):
            super().__init__()

        def forward(self, x):
            # Pretend x is (B=1, L=2, D=3) flattened to (1, 6).
            x = x.view(1, self.n_tokens, 3)
            return x[:, 0]

    model = SliceFirstToken().eval()
    box = _flat_box(
        np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
        np.array([0.5, 1.5, 2.5, 10.5, 11.5, 12.5]),
    )
    for set_in in (
        box, _flat_star(box), _flat_zono(box),
        _flat_hex(box), _flat_oct(box),
    ):
        out = NeuralNetwork(model).reach(
            set_in, method="approx", n_tokens=2,
        )
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        # First token in the flat layout is the first D=3 chunk.
        np.testing.assert_allclose(lb_o, np.array([0.0, 1.0, 2.0]), atol=1e-9)
        np.testing.assert_allclose(ub_o, np.array([0.5, 1.5, 2.5]), atol=1e-9)


# ----------------------------- SoftmaxAttention multi-input ----------------


@pytest.mark.parametrize("set_kind", ["Box", "Star", "Zono", "Hex", "Oct"])
def test_softmax_attention_multi_input_all_set_types(set_kind):
    """End-to-end: a minimal model whose forward calls SoftmaxAttention on
    three projected views of the input. With N2VTracer (commit 6878285)
    treating SoftmaxAttention as an fx leaf, ``_handle_multi_input_op``
    dispatches Box/Star/Zono/Hex/Oct streams via the box-lifted helper.

    Pins the multi-input dispatcher routes for every set type, including
    the new Hex/Oct branches added in this PR.
    """
    from n2v.nn import NeuralNetwork
    from n2v.nn.layers import SoftmaxAttention

    class _AttnModel(nn.Module):
        def __init__(self, d_head: int = 2):
            super().__init__()
            self.q_proj = nn.Linear(d_head, d_head, bias=False)
            self.k_proj = nn.Linear(d_head, d_head, bias=False)
            self.v_proj = nn.Linear(d_head, d_head, bias=False)
            self.attn = SoftmaxAttention(d_head=d_head)
            # Identity projections so we can reason about the output range.
            with torch.no_grad():
                self.q_proj.weight.copy_(torch.eye(d_head))
                self.k_proj.weight.copy_(torch.eye(d_head))
                self.v_proj.weight.copy_(torch.eye(d_head))

        def forward(self, x):
            return self.attn(self.q_proj(x), self.k_proj(x), self.v_proj(x))

    model = _AttnModel(d_head=2).eval()
    box = _flat_box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    ctors = {
        "Box": lambda: box,
        "Star": lambda: _flat_star(box),
        "Zono": lambda: _flat_zono(box),
        "Hex": lambda: _flat_hex(box),
        "Oct": lambda: _flat_oct(box),
    }
    set_in = ctors[set_kind]()

    out = NeuralNetwork(model).reach(set_in, method="approx")
    assert len(out) == 1
    lb_o, ub_o = _bounds_of(out[0])
    # With V in [0, 1] and softmax rows summing to 1, the output lies in
    # the convex hull of V's columns, i.e. each output coordinate is
    # bounded by the corresponding column's [min, max]. For our identity
    # projections that means lb >= 0 and ub <= 1.
    assert np.all(lb_o >= 0.0 - 1e-6)
    assert np.all(ub_o <= 1.0 + 1e-6)
