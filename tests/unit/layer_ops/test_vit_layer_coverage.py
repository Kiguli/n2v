"""Set-type coverage tests for the ViT layers / primitives in this PR.

The ViT integration test (``tests/integration/test_minimal_vit.py``)
only exercises the Zono path because that is what the benchmark needs.
This file pins the **Box / Star / Zono** reach for each ViT layer and
primitive added by this PR, so future regressions are caught by CI:

  * PatchEmbed           — Box, Star, Zono
  * LayerNorm (single-token)  — Box, Star, Zono
  * GELU (erf + tanh)    — Box, Star, Zono
  * Linear (per-token block-tile) — Box, Star, Zono (+ base Hex/Oct)
  * fx ``operator.add`` (set + constant / set + set) — Box, Star, Zono
  * fx ``operator.getitem`` (tensor slice) — Box, Star, Zono
  * SoftmaxAttention multi-input — Box, Star, Zono

Each test instantiates the relevant layer / op, builds a small input
set of the right type, runs the reach helper via the dispatcher, and
asserts shape + finite bounds + Monte-Carlo containment of concrete
forward samples where applicable.
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


def test_layernorm_box_star_zono_single_token():
    """Audit N7/N11: previously asserted only ``isfinite`` -- a vacuous
    ``[-inf, +inf]`` envelope would pass. Now: assert Monte-Carlo
    containment of 32 random forward samples in the reach bounds.
    """
    layer = nn.LayerNorm(4)
    layer.eval()
    lb = np.array([-1.0, -0.5, 0.0, 0.5])
    ub = np.array([0.0, 0.5, 1.0, 1.5])
    box = _flat_box(lb, ub)
    for set_in, helper in (
        (box, lambda: layernorm_reach.layernorm_box(layer, [box])),
        (_flat_star(box), lambda: layernorm_reach.layernorm_star_approx(layer, [_flat_star(box)])),
        (_flat_zono(box), lambda: layernorm_reach.layernorm_zono(layer, [_flat_zono(box)])),
    ):
        out = helper()
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        assert np.all(np.isfinite(lb_o)) and np.all(np.isfinite(ub_o))
        assert lb_o.size == 4

    # N11/M4: Monte-Carlo concrete-forward containment (Box path; others
    # cover the same forward via their box envelopes).
    pytest.assert_reach_contains_forward(
        layer, lb, ub,
        lambda lay, sets: layernorm_reach.layernorm_box(lay, sets),
        n_samples=32, input_shape=(1, 4),
    )


# ----------------------------- GELU (erf + tanh) -----------------------------


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_box_star_zono(approximate):
    layer = nn.GELU(approximate=approximate)
    layer.eval()
    lb = np.array([-2.0, -0.5, 0.0])
    ub = np.array([-1.0, 0.5, 1.0])
    box = _flat_box(lb, ub)

    for set_in in (
        box, _flat_star(box), _flat_zono(box),
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
        # Audit I8: pass expected_n_tokens=L so the helper verifies the
        # inferred block-tile count instead of silently inferring.
        out = fn(layer, [set_in], expected_n_tokens=L)
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        assert lb_o.size == L * 2  # L tokens of D_out=2

    # Audit N3/N11: assert per-token concrete-forward correspondence.
    # A transposed ``kron(W, I_L)`` (vs the correct ``kron(I_L, W)``)
    # would pass the shape assertion above but fail this MC check.
    pytest.assert_reach_contains_forward(
        layer,
        np.zeros(in_dim), np.ones(in_dim),
        lambda lay, sets: linear_reach.linear_box(
            lay, sets, expected_n_tokens=L,
        ),
        n_samples=32, input_shape=(1, L, 3),
    )


def test_patch_embed_box_multichannel_requires_image_shape_audit_I2():
    """PR-1 audit I2: PatchEmbed Box reach with in_channels > 1 must
    REFUSE to infer layout. The previous code silently called
    ``ImageZono.from_bounds`` with HWC semantics, mis-permuting a
    CHW-flat input. Fix: require explicit image_shape; raise otherwise.
    """
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach as pe

    layer = PatchEmbed(
        in_channels=3, embed_dim=4, patch_size=2,
    ).eval()
    # 4x4x3 = 48 elements
    box = _flat_box(np.zeros(48), np.ones(48))
    with pytest.raises(NotImplementedError, match="in_channels=3.*image_shape"):
        pe.patch_embed_box(layer, [box])


def test_patch_embed_box_with_image_shape_hwc_matches_imagezono():
    """When ``image_shape`` is given, the Box path must produce the same
    IBP envelope as routing the equivalent ImageZono directly."""
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach as pe

    torch.manual_seed(0)
    layer = PatchEmbed(
        in_channels=3, embed_dim=4, patch_size=2,
    ).eval()
    flat_lb = np.zeros(48)
    flat_ub = np.ones(48)
    box = _flat_box(flat_lb, flat_ub)
    box_out = pe.patch_embed_box(
        layer, [box],
        image_shape=(4, 4, 3), image_layout="HWC",
    )[0]
    zono_in = ImageZono.from_bounds(
        flat_lb, flat_ub, height=4, width=4, num_channels=3,
    )
    zono_out = pe.patch_embed_zono(layer, [zono_in])[0]
    lb_z, ub_z = zono_out.get_bounds()
    np.testing.assert_allclose(
        np.asarray(box_out.lb).flatten(),
        np.asarray(lb_z).flatten(), atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(box_out.ub).flatten(),
        np.asarray(ub_z).flatten(), atol=1e-9,
    )


def test_patch_embed_box_chw_layout_differs_from_hwc_for_multichannel():
    """Audit I2: a CHW-flat input must be permuted to HWC before the
    conv reach to match PyTorch forward semantics. This test asserts
    that the CHW path differs from the HWC path when the per-channel
    bounds differ (so the permutation is observable).
    """
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach as pe

    torch.manual_seed(0)
    layer = PatchEmbed(
        in_channels=3, embed_dim=4, patch_size=2,
    ).eval()
    # 2x2x3 = 12 elements. Construct a per-channel-distinct CHW-flat
    # box: channel 0 = [0, 0.1], channel 1 = [0.4, 0.5], channel 2 = [0.8, 0.9].
    chw_lb = np.array(
        [0.0, 0.0, 0.0, 0.0,  # channel 0 (4 pixels)
         0.4, 0.4, 0.4, 0.4,  # channel 1
         0.8, 0.8, 0.8, 0.8], # channel 2
    )
    chw_ub = chw_lb + 0.1
    box = _flat_box(chw_lb, chw_ub)
    out_hwc = pe.patch_embed_box(
        layer, [box], image_shape=(2, 2, 3), image_layout="HWC",
    )[0]
    out_chw = pe.patch_embed_box(
        layer, [box], image_shape=(2, 2, 3), image_layout="CHW",
    )[0]
    # The two must produce different reaches.
    assert not np.allclose(out_hwc.lb, out_chw.lb), (
        "HWC and CHW reaches identical -- permutation is not happening."
    )


def test_patch_embed_box_non_square_image_explicit_shape_works():
    """Audit I3: non-square images must be supported via image_shape
    kwarg. (The pre-fix code silently inferred a square side from
    pixel count and mis-shaped non-square images.)
    """
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach as pe

    # 2x8 image, 2x2 patches, 1 channel = 16 elements (pixel count IS a
    # perfect square, so the buggy code would silently use a 4x4 image).
    layer = PatchEmbed(
        in_channels=1, embed_dim=4, patch_size=2,
    ).eval()
    box = _flat_box(np.zeros(16), np.ones(16))
    out = pe.patch_embed_box(
        layer, [box], image_shape=(2, 8, 1), image_layout="HWC",
    )
    assert len(out) == 1
    # 2x8 image / (2x2 patch) = 1x4 = 4 tokens of dim 4 = 16 elements
    assert out[0].dim == 16


def test_linear_block_tile_mismatch_raises_audit_I8():
    """PR-1 audit I8: when the dispatcher (or test) declares
    ``expected_n_tokens`` and the divisibility-inferred ``L`` disagrees,
    the helper must raise ``NotImplementedError`` instead of silently
    verifying a different function.
    """
    layer = nn.Linear(3, 2, bias=True).eval()
    box = _flat_box(np.zeros(12), np.ones(12))   # divisibility says L=4
    with pytest.raises(NotImplementedError, match="disagrees with"):
        linear_reach.linear_box(layer, [box], expected_n_tokens=3)


def test_linear_block_tile_no_n_tokens_warns_audit_I8():
    """Audit I8: silent block-tile inference (L > 1 without an explicit
    n_tokens signal) must emit a ``UserWarning`` so users can audit
    when their reach is doing per-token tiling."""
    layer = nn.Linear(3, 2, bias=True).eval()
    box = _flat_box(np.zeros(12), np.ones(12))
    with pytest.warns(UserWarning, match="without an explicit n_tokens"):
        out = linear_reach.linear_box(layer, [box])
        assert out[0].dim == 8


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
    for set_in in (
        box,
        _flat_star(box),
        _flat_zono(box),
        _flat_hex(box),
        _flat_oct(box),
    ):
        out = NeuralNetwork(model).reach(set_in, method="approx")
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        # Output = input + [1, 2]. Inner dim 0: [1, 2]; dim 1: [2, 3].
        np.testing.assert_allclose(lb_o, np.array([1.0, 2.0]), atol=1e-9)
        np.testing.assert_allclose(ub_o, np.array([2.0, 3.0]), atol=1e-9)


# ------------- Activation min-location numerical drift (audit I1) ------------


def test_activation_box_floor_is_below_true_min_on_narrow_intervals():
    """PR-1 audit I1: GELU/SiLU min-location constants must be set so that
    narrow boxes which BRACKET the true argmin produce a reach lower bound
    BELOW the true minimum on the box (sound), not above it (unsound).

    The previous constants used the true argmin rounded to only 4-7 digits
    -- e.g. ``_GELU_TANH_X_MIN = -0.7517`` vs true ``-0.7524614``. The
    point check ``(lb <= x_min) & (ub >= x_min)`` then missed narrow
    boxes that bracketed the true argmin but excluded the rounded
    constant, producing an above-floor lower bound.

    Counterexamples (each verified ≥ 1e-12 unsound on the buggy code):
        * GELU erf:  Box [-0.75179155, -0.75179]
        * GELU tanh: Box [-0.7530, -0.7520]

    Pin: reach lb must be ≤ scipy's bounded minimum on the box.
    """
    import scipy.optimize as opt
    from math import erf, tanh as math_tanh, sqrt
    from n2v.nn.layer_ops.gelu_reach import gelu_box, gelu_tanh_box

    def _gelu_erf(x):
        return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))

    def _gelu_tanh(x):
        return 0.5 * x * (1.0 + math_tanh(
            sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        ))

    cases = [
        ("gelu_erf",  gelu_box,      _gelu_erf,  -0.75179155, -0.75179),
        ("gelu_tanh", gelu_tanh_box, _gelu_tanh, -0.75300, -0.75200),
    ]
    for name, reach_fn, fwd, a, b in cases:
        box = Box(
            np.array([[a]], dtype=np.float64),
            np.array([[b]], dtype=np.float64),
        )
        out = reach_fn([box])[0]
        lb = float(np.asarray(out.lb).flatten()[0])
        res = opt.minimize_scalar(
            fwd, bounds=(a, b), method="bounded",
            options={"xatol": 1e-14},
        )
        assert lb <= res.fun + 1e-14, (
            f"{name}: Box [{a}, {b}] reach lb = {lb!r} is ABOVE true min "
            f"{res.fun!r} (delta = {lb - res.fun:.3e}). I1 unsoundness."
        )


# ------- CLSToken / ConcatWithFrozenSkip Hex+Oct coverage (audit I7) --------


# ----------------------------- F.gelu approximate kwarg leak (audit C1) -----


def test_fx_add_set_plus_set_two_stream_audit_N12():
    """PR-1 audit N12: the existing operator.add tests only exercise
    set+const. The set+set branch (two independent reach streams added
    at a residual) was untested in test_vit_layer_coverage. A bug in
    ``_add_sets`` (e.g. sign flip) would not be caught by any other
    test in this file. Pin: a model ``y = a(x) + b(x)`` must MC-contain
    forward samples.
    """
    class TwoStreamAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 3, bias=False)
            self.b = nn.Linear(3, 3, bias=False)
            torch.manual_seed(7)
            # Non-trivial weights so a sign flip would show up.
            with torch.no_grad():
                self.a.weight.copy_(torch.tensor(
                    [[0.5, -0.2, 0.0],
                     [0.0, 0.3, 0.1],
                     [-0.1, 0.0, 0.4]],
                ))
                self.b.weight.copy_(torch.tensor(
                    [[0.1, 0.0, -0.2],
                     [-0.3, 0.2, 0.0],
                     [0.0, -0.1, 0.5]],
                ))

        def forward(self, x):
            return self.a(x) + self.b(x)

    from n2v.nn import NeuralNetwork
    model = TwoStreamAdd().eval()
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])

    def _reach(layer, sets):
        return NeuralNetwork(model).reach(sets[0], method="approx")

    pytest.assert_reach_contains_forward(
        model, lb, ub, _reach, n_samples=32, input_shape=(1, 3),
    )


def test_patch_embed_box_mc_containment_audit_N2():
    """Audit N2: the existing PatchEmbed tests assert shape/finiteness
    only -- a constant-output reach would pass. Add concrete-forward
    MC containment with image_shape=(2, 2, 1) (square but explicit so
    no warning fires; in_channels=1 keeps HWC==CHW so layout is moot).
    """
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach

    layer = PatchEmbed(
        in_channels=1, embed_dim=4, patch_size=2,
    ).eval()
    lb = np.zeros(4)
    ub = np.full(4, 0.5)

    pytest.assert_reach_contains_forward(
        layer, lb, ub,
        lambda lay, sets: patch_embed_reach.patch_embed_box(
            lay, sets, image_shape=(2, 2, 1), image_layout="HWC",
        ),
        n_samples=24, input_shape=(1, 1, 2, 2),
    )


def test_fx_f_gelu_approximate_tanh_kwarg_preserved():
    """PR-1 audit C1: ``F.gelu(x, approximate='tanh')`` must route to the
    tanh-form floor (-0.170041), not the erf-form floor (-0.169972).

    The buggy implementation had ``F.gelu: nn.GELU`` in
    ``FUNCTION_TO_MODULE_CLS``, and ``_function_node_to_module`` consulted
    the dict first via ``cls()`` -- silently dropping ``approximate='tanh'``
    and instantiating ``nn.GELU(approximate='none')``. The reach then routed
    through the erf-form (floor -0.169972), an above-floor lower bound that
    excludes true tanh-form outputs near the dip at x ~ -0.7517 -> unsound.

    Pin: a Box bracketing the dip must produce a lower bound <= the
    tanh-form floor.
    """
    import torch.nn.functional as F  # local: matches the production import path
    from n2v.nn import NeuralNetwork
    from n2v.nn.layer_ops.gelu_reach import _GELU_TANH_F_MIN

    class GeluTanhFn(nn.Module):
        def forward(self, x):
            return F.gelu(x, approximate="tanh")

    model = GeluTanhFn().eval()
    box = _flat_box(np.array([-1.0]), np.array([-0.5]))
    out = NeuralNetwork(model).reach(box, method="approx")
    assert len(out) == 1
    lb = float(np.asarray(out[0].lb).flatten()[0])
    assert lb <= _GELU_TANH_F_MIN + 1e-9, (
        f"F.gelu approximate='tanh' floor leaked: lb={lb!r}, "
        f"expected <= {_GELU_TANH_F_MIN!r} (audit C1)."
    )


def test_fx_f_gelu_approximate_none_default_still_works():
    """Companion: ``F.gelu(x)`` (no kwarg) must route to the erf form."""
    import torch.nn.functional as F
    from n2v.nn import NeuralNetwork
    from n2v.nn.layer_ops.gelu_reach import _GELU_F_MIN

    class GeluDefault(nn.Module):
        def forward(self, x):
            return F.gelu(x)

    model = GeluDefault().eval()
    box = _flat_box(np.array([-1.0]), np.array([-0.5]))
    out = NeuralNetwork(model).reach(box, method="approx")
    lb = float(np.asarray(out[0].lb).flatten()[0])
    assert lb <= _GELU_F_MIN + 1e-9


# ------------------ SoftmaxAttention l_q from Q stream (audit T1-5) ----------


def test_fx_softmax_attention_cross_shaped_l_q_from_q_stream_audit_T1_5():
    """Audit T1-5: ``l_q`` (the number of output query rows) must derive
    from the Q stream's flat dim, not V's. For self-attention the two
    coincide, masking the bug; for a cross-shaped call (L_q != L_kv) the
    previous code tiled L_kv output rows, producing a wrong-SHAPED reach
    set (values were still convex-hull-of-V sound, but the row count was
    wrong and downstream layers would verify a different layout).

    Model: 2 tokens of d_head=2; Q = first token only (L_q=1), K = V =
    both tokens (L_kv=2). Expected reach output: 1 row of the column-wise
    hull of V's two tokens -> flat dim 2 (pre-fix: dim 4).
    """
    from n2v.nn import NeuralNetwork
    from n2v.nn.layers import SoftmaxAttention

    class CrossShapedAttn(nn.Module):
        n_tokens = 2

        def __init__(self):
            super().__init__()
            self.attn = SoftmaxAttention(d_head=2)

        def forward(self, x):
            x = x.view(1, 2, 2)
            q = x[:, 0]            # one token: (1, 2)
            q = q.view(1, 1, 2)    # back to (B, L_q=1, d)
            return self.attn(q, x, x)

    model = CrossShapedAttn().eval()
    # Token 0 in [0, 0.1]^2, token 1 in [0.5, 0.6] x [0.8, 0.9].
    box = _flat_box(
        np.array([0.0, 0.0, 0.5, 0.8]),
        np.array([0.1, 0.1, 0.6, 0.9]),
    )
    out = NeuralNetwork(model).reach(box, method="approx", n_tokens=2)
    assert len(out) == 1
    lb_o, ub_o = _bounds_of(out[0])
    assert lb_o.size == 2, (
        f"T1-5: expected L_q*d_v = 1*2 = 2 output dims, got {lb_o.size} "
        f"(l_q derived from the wrong stream)."
    )
    # Convex-hull-of-V bounds: column-wise min/max over the two tokens.
    np.testing.assert_allclose(lb_o, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(ub_o, [0.6, 0.9], atol=1e-9)

    # MC containment: random samples through the true forward.
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(24):
            x = rng.uniform(
                [0.0, 0.0, 0.5, 0.8], [0.1, 0.1, 0.6, 0.9],
            ).astype(np.float32)
            y = model(torch.from_numpy(x).reshape(1, 4)).numpy().flatten()
            assert np.all(lb_o - 1e-6 <= y) and np.all(y <= ub_o + 1e-6)


# ------------------ SoftmaxAttention Q/K/V kwargs binding (audit C2) ----------


def test_fx_softmax_attention_kwargs_query_value_key_order():
    """PR-1 audit C2 + math-audit 1d: SoftmaxAttention Q/K/V binding must
    NOT depend on Python's kwargs insertion order. The wrapper signature
    is ``forward(q, k, v, attn_mask=None)``; calling
    ``self.attn(q=..., v=..., k=...)`` (kwargs deliberately NOT in
    declared order) must produce the same reach as the positional call.

    To make misbinding detectable, K and V are produced by DIFFERENT
    linear maps (V is scaled by 3), so an insertion-order bind that
    swaps k/v changes the hull bound observably.
    """
    from n2v.nn import NeuralNetwork
    from n2v.nn.layers.softmax_attention import SoftmaxAttention

    d_head = 2

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = nn.Linear(d_head, d_head, bias=False)
            self.v_proj = nn.Linear(d_head, d_head, bias=False)
            with torch.no_grad():
                self.k_proj.weight.copy_(torch.eye(d_head))
                self.v_proj.weight.copy_(3.0 * torch.eye(d_head))
            self.attn = SoftmaxAttention(d_head=d_head)

    class AttnKwargReordered(_Base):
        def forward(self, x):
            return self.attn(q=x, v=self.v_proj(x), k=self.k_proj(x))

    class AttnPositional(_Base):
        def forward(self, x):
            return self.attn(x, self.k_proj(x), self.v_proj(x))

    box = _flat_box(
        np.array([0.0, 0.1]),
        np.array([0.5, 0.6]),
    )
    out_kw = NeuralNetwork(AttnKwargReordered().eval()).reach(
        box, method="approx",
    )
    out_pos = NeuralNetwork(AttnPositional().eval()).reach(
        box, method="approx",
    )
    np.testing.assert_allclose(
        np.asarray(out_kw[0].lb).flatten(),
        np.asarray(out_pos[0].lb).flatten(),
        atol=1e-9,
        err_msg="kwargs reordering produced different reach (audit C2).",
    )
    np.testing.assert_allclose(
        np.asarray(out_kw[0].ub).flatten(),
        np.asarray(out_pos[0].ub).flatten(),
        atol=1e-9,
        err_msg="kwargs reordering produced different reach (audit C2).",
    )
    # The V stream is 3x the input box, so the hull ub must reflect V
    # (ub ~ 1.8), not K (ub ~ 0.6) -- pins that v landed in the V slot.
    assert np.asarray(out_pos[0].ub).flatten()[0] > 1.0


def test_fx_softmax_attention_constant_v_buffer_raises_math_audit_1d():
    """Math-audit 1d: a constant (get_attr) V buffer previously vanished
    from the stream list and a traced attn_mask slid into the V slot --
    executed counterexample: true output 10.0, reach [0, 0]. Now any
    non-traced q/k/v raises, and any non-None attn_mask raises.
    """
    from n2v.nn import NeuralNetwork
    from n2v.nn.layers import SoftmaxAttention

    class ConstV(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(1, 1, bias=False)
            self.wk = nn.Linear(1, 1, bias=False)
            self.wm = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.wq.weight.fill_(1.0)
                self.wk.weight.fill_(1.0)
                self.wm.weight.fill_(0.0)
            self.register_buffer("v_const", torch.tensor([[10.0]]))
            self.attn = SoftmaxAttention(d_head=1)

        def forward(self, x):
            return self.attn(self.wq(x), self.wk(x), self.v_const, self.wm(x))

    box = _flat_box(np.array([0.0]), np.array([0.1]))
    with pytest.raises(NotImplementedError, match="not a traced reach stream"):
        NeuralNetwork(ConstV().eval()).reach(box, method="approx")


# ------------- fx set+set add: Minkowski soundness (math-audit P0) ----------


def test_fx_add_star_residual_with_box_lifted_branch_math_audit_P0():
    """Math-audit P0 repro 1: ``z = Wx + gelu(x)`` with ``W = -I``. The
    GELU Star path box-lifts to fresh predicates; the previous
    shared-predicate sum ``V1 + V2`` assumed a correlation that does
    not exist and the true output at x = (-1,-1) escaped the reach
    upper bound by 0.011. The fx add route now uses the Minkowski sum,
    which is unconditionally sound.
    """
    from n2v.nn import NeuralNetwork

    class NegSkipGelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.w.weight.copy_(-torch.eye(2))
            self.act = nn.GELU()

        def forward(self, x):
            return self.w(x) + self.act(x)

    model = NegSkipGelu().eval()
    star = Star.from_bounds(
        np.array([[-1.0], [-1.0]]), np.array([[1.0], [1.0]]),
    )
    out = NeuralNetwork(model).reach(star, method="approx")
    assert len(out) == 1
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(32):
            x = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            y = model(torch.from_numpy(x).unsqueeze(0)).numpy().flatten()
            assert out[0].contains(y.reshape(-1, 1)), (
                f"true output {y} escapes the residual-add reach "
                f"(math-audit P0)."
            )
    # The witness from the executed counterexample specifically:
    with torch.no_grad():
        y_w = model(torch.tensor([[-1.0, -1.0]])).numpy().flatten()
    assert out[0].contains(y_w.reshape(-1, 1))


def test_fx_add_star_exact_split_cross_product_math_audit_P0():
    """Math-audit P0 repro 2: ``relu(x + (-1)) + relu(x + 1)`` under
    ``method='exact'``. The previous index-``zip`` pairing of the two
    ReLU split lists discarded the second operand's region constraints
    and the entire true segment z in (0, 2) vanished from the reach
    union -- a violated safety property would have verified SAFE. The
    cross-product Minkowski pairing must contain every true output.
    """
    import torch.nn.functional as F
    from n2v.nn import NeuralNetwork

    class TwoRelu(nn.Module):
        def forward(self, x):
            return F.relu(x + (-1.0)) + F.relu(x + 1.0)

    model = TwoRelu().eval()
    star = Star.from_bounds(np.array([[-2.0]]), np.array([[2.0]]))
    out = NeuralNetwork(model).reach(star, method="exact")
    assert len(out) >= 1
    with torch.no_grad():
        for xv in (-2.0, -0.5, 0.0, 0.5, 2.0):
            z = float(model(torch.tensor([[xv]])).item())
            contained = any(
                s.contains(np.array([[z]])) for s in out
            )
            assert contained, (
                f"true z({xv}) = {z} not contained in any output star "
                f"(math-audit P0 repro 2)."
            )


# ------- Norm Star predicate-bounds guard (math-audit Finding 1) ------------


def test_layernorm_star_constraint_only_predicates_raises_math_audit_F1():
    """Math-audit Finding 1: a Star constrained only by C@alpha <= d
    (predicate_lb/ub = None) previously had [-1, 1] silently imposed --
    unsound when feasible alpha lies outside that box (executed escape
    0.188). The norm Star path must now refuse such stars.
    """
    layer = nn.LayerNorm(4, eps=1.0, elementwise_affine=False).eval()
    V = np.hstack([np.zeros((4, 1)), np.eye(4)])
    # alpha in [-5, 5]^4 encoded ONLY via constraints.
    C = np.vstack([np.eye(4), -np.eye(4)])
    d = 5.0 * np.ones((8, 1))
    star = Star(V, C, d, None, None)
    with pytest.raises(NotImplementedError, match="predicate_lb"):
        layernorm_reach.layernorm_star_approx(layer, [star])


# ------------ set+const ImageStar 4D layout guard (math-audit item 2) -------


def test_fx_add_image_star_plus_scalar_const_copilot():
    """Copilot review: a scalar broadcast add (``x + 1.0``) on an
    ImageStar must tile the scalar across (H, W, C) -- the previous
    ``const.reshape(H, W, C)`` raised ValueError on a shape-(1,)
    constant instead of producing the sound translation.
    """
    from n2v.nn import NeuralNetwork

    class AddScalar(nn.Module):
        def forward(self, x):
            return x + 1.0

    star = ImageStar.from_bounds(
        np.zeros((8, 1)), np.ones((8, 1)),
        height=2, width=2, num_channels=2,
    )
    out = NeuralNetwork(AddScalar().eval()).reach(star, method="approx")[0]
    lo, ub = out.get_ranges()
    np.testing.assert_allclose(np.asarray(lo).flatten(), np.ones(8), atol=1e-9)
    np.testing.assert_allclose(np.asarray(ub).flatten(), 2 * np.ones(8), atol=1e-9)


def test_fx_add_image_star_plus_4d_nchw_const_raises_math_audit_2():
    """Math-audit item 2: a 4D NCHW constant (torch's natural parameter
    shape ``(1, C, H, W)``) added to an ImageStar was reshaped
    flat-order into (H, W, C), silently scrambling channels for C > 1.
    Layout is ambiguous -- the handler must refuse 4D constants.
    """
    from n2v.nn import NeuralNetwork

    class AddNCHWConst(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "c", torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2),
            )

        def forward(self, x):
            return x + self.c

    star = ImageStar.from_bounds(
        np.zeros((8, 1)), np.ones((8, 1)),
        height=2, width=2, num_channels=2,
    )
    with pytest.raises(NotImplementedError, match="layout-ambiguous"):
        NeuralNetwork(AddNCHWConst().eval()).reach(star, method="approx")


# ----------------------------- fx operator.getitem (tensor slice) ----------


def test_fx_add_set_plus_constant_image_star_4d_V():
    """Audit spot-check: the operator.add set+const handler for ImageStar
    previously wrote ``new_V[:, 0:1] = ...`` which slices the W axis of the
    4D V tensor ``(H, W, C, n_var+1)`` rather than the centre column.

    For an ImageStar input the slice produced shape ``(H, 1, C, n_var+1)``
    while const_flat was ``(flat_dim, 1)`` — numpy broadcast either crashed
    or silently mis-aligned. The fix indexes ``new_V[..., 0]`` (last axis,
    centre column) and reshapes the constant to ``(H, W, C)``.

    This test pins the fixed behaviour: a per-element constant add on a
    (H=2, W=2, C=2) ImageStar produces the correct per-element output.
    """
    from n2v.nn import NeuralNetwork
    from n2v.sets.image_star import ImageStar

    class IdAddConst(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "c",
                torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ).reshape(2, 2, 2),
            )

        def forward(self, x):
            return x + self.c

    model = IdAddConst().eval()
    star = ImageStar.from_bounds(
        np.zeros((8, 1)), np.ones((8, 1)),
        height=2, width=2, num_channels=2,
    )
    out = NeuralNetwork(model).reach(star, method="approx")
    assert len(out) == 1
    lb, ub = out[0].get_ranges()
    lb = np.asarray(lb).flatten()
    ub = np.asarray(ub).flatten()
    # Expected: input is [0, 1] elementwise; add [1, 2, 3, 4, 5, 6, 7, 8]
    # element-wise; output is [1..8, 2..9].
    np.testing.assert_allclose(lb, [1, 2, 3, 4, 5, 6, 7, 8], atol=1e-9)
    np.testing.assert_allclose(ub, [2, 3, 4, 5, 6, 7, 8, 9], atol=1e-9)


def test_fx_getitem_image_star_4d_V_flattens_before_slice():
    """PR-1 audit I6: ``operator.getitem`` on an ImageStar previously did
    ``s.V[row_start:row_end]`` which slices the FIRST axis (H) of the 4D
    V tensor (H, W, C, nVar+1), producing a 4D slice that
    ``Star(...)`` rejects with ``ValueError: too many values to unpack``.

    Fix: flatten via ``to_star()`` (HWC-row-major == token-major) BEFORE
    row-slicing. Pin: an ImageStar carrying a 2x2 image with C=2 (so dim
    = 8) and a model that selects the first token (``x[:, 0]``) must
    return a Star with dim = 2 (D = C since L = H*W = 4), not raise.
    """
    from n2v.nn import NeuralNetwork

    class SliceFirstToken(nn.Module):
        # H*W=4 tokens, C=2 channels -> token-major flat layout.
        n_tokens = 4

        def forward(self, x):
            x = x.view(1, 4, 2)
            return x[:, 0]

    model = SliceFirstToken().eval()
    image_star = ImageStar.from_bounds(
        np.zeros((8, 1)), np.ones((8, 1)),
        height=2, width=2, num_channels=2,
    )
    out = NeuralNetwork(model).reach(image_star, method="approx", n_tokens=4)
    assert len(out) == 1
    # The first token is the (h=0, w=0) pixel -> rows 0..1 of the HWC-flat
    # layout = the first 2 channels of pixel (0,0).
    lb, ub = out[0].get_ranges()
    lb = np.asarray(lb).flatten()
    ub = np.asarray(ub).flatten()
    assert lb.shape == (2,), f"expected dim 2, got {lb.shape}"
    np.testing.assert_allclose(lb, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(ub, [1.0, 1.0], atol=1e-9)


def test_fx_getitem_non_trivial_batch_index_raises_audit_N1():
    """PR-1 audit N1: ``_handle_getitem`` previously stripped
    ``index[0]`` unconditionally so ``x[1, 0]`` was silently treated as
    ``x[:, 0]`` -- the reach would select the wrong token of an
    arbitrary batch element. n2v only supports batch-1 inputs, but a
    non-trivial batch index in the user's model means the model is
    ambiguous; we should raise rather than silently rewrite.
    """
    from n2v.nn import NeuralNetwork

    class BadBatchSlice(nn.Module):
        n_tokens = 2

        def forward(self, x):
            x = x.view(1, 2, 3)
            return x[0, 0]  # explicit batch index 0 — not slice(None)

    model = BadBatchSlice().eval()
    box = _flat_box(
        np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
        np.array([0.5, 1.5, 2.5, 10.5, 11.5, 12.5]),
    )
    with pytest.raises(NotImplementedError, match="batch axis"):
        NeuralNetwork(model).reach(box, method="approx", n_tokens=2)


def test_fx_getitem_negative_token_idx():
    """Audit spot-check: getitem with negative token_idx (e.g. ``x[:, -1]``
    -- the canonical "select last token / CLS / DistillationToken" pattern)
    previously produced an EMPTY reach because
    ``row_start = token_idx * D = -D`` and ``row_end = 0`` made ``s.V[-D:0]``
    an empty slice.

    The fix normalises negative indices via ``token_idx + L`` and raises on
    out-of-range. This test pins both behaviours.
    """
    from n2v.nn import NeuralNetwork

    class SliceLast(nn.Module):
        n_tokens = 2

        def forward(self, x):
            x = x.view(1, 2, 3)
            return x[:, -1]

    model = SliceLast().eval()
    box = _flat_box(
        np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
        np.array([0.5, 1.5, 2.5, 10.5, 11.5, 12.5]),
    )
    out = NeuralNetwork(model).reach(box, method="approx", n_tokens=2)
    # x[:, -1] selects the LAST token: rows 3-5 of the flat layout.
    np.testing.assert_allclose(
        out[0].lb.flatten(), np.array([10.0, 11.0, 12.0]), atol=1e-9,
    )
    np.testing.assert_allclose(
        out[0].ub.flatten(), np.array([10.5, 11.5, 12.5]), atol=1e-9,
    )

    class BadSlice(nn.Module):
        n_tokens = 2

        def forward(self, x):
            x = x.view(1, 2, 3)
            return x[:, 5]  # out of range

    with pytest.raises(NotImplementedError, match="out of range"):
        NeuralNetwork(BadSlice().eval()).reach(
            box, method="approx", n_tokens=2,
        )


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

    # Audit N4/N11: tighten beyond ``in [0,1]`` by checking concrete-forward
    # containment against the actual model.  Use the strict-inside box
    # ``[0.3, 0.7]`` so a stub returning ``Box([0],[1])`` would still pass
    # the previous assertion but fail to contain forward samples outside
    # [0.3, 0.7].  We run this MC check on Box only since it's the only
    # set type that exercises the full set-of-streams path identically to
    # the forward.
    if set_kind == "Box":
        strict_lb = np.array([0.3, 0.4])
        strict_ub = np.array([0.6, 0.7])

        def _attn_reach(layer, sets):
            return NeuralNetwork(model).reach(sets[0], method="approx")

        pytest.assert_reach_contains_forward(
            model, strict_lb, strict_ub, _attn_reach,
            n_samples=24,
            input_shape=(1, 2),
        )


def test_linear_star_image_star_l_gt_1_flattens_and_rewraps_dive_review():
    """Deep-dive review: an L > 1 ImageStar through ``linear_star``
    previously crashed (``_block_apply`` assumed 2D V). It now flattens
    via ``to_star`` (HWC == per-pixel block order), applies the
    blockwise map, and re-wraps with the new channel count when the
    block count equals the pixel count.
    """
    from n2v.nn.layer_ops import linear_reach

    torch.manual_seed(0)
    layer = nn.Linear(3, 2, bias=True).eval()  # per-pixel: C=3 -> 2
    ims = ImageStar.from_bounds(
        np.zeros((2, 2, 3)), np.ones((2, 2, 3)),
        height=2, width=2, num_channels=3,
    )
    out = linear_reach.linear_star(layer, [ims], expected_n_tokens=4)[0]
    assert isinstance(out, ImageStar)
    assert (out.height, out.width, out.num_channels) == (2, 2, 2)
    # MC containment: per-pixel forward must lie inside the reach.
    flat = out.to_star()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(16):
            x = rng.uniform(0, 1, (2, 2, 3)).astype(np.float32)
            # forward applies Linear over the channel axis per pixel
            y = layer(torch.from_numpy(x)).numpy().reshape(-1, 1)
            assert flat.contains(y), "per-pixel forward escaped reach"


