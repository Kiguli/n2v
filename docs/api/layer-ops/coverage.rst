Layer Coverage Matrix
=====================

Per-layer set-type support after the porting work. The columns are the
five set types implemented in :mod:`n2v.sets`: Box, Star (and
ImageStar), Zono (and ImageZono), Hexatope, Octatope.

Legend
------

- ✅ — implemented and routed through the dispatcher
- ✨ — implemented as a *predicate-preserving* Star approximation that
  carries the input Star's predicates through, then adds per-feature
  slack predicates for the non-affine portion of the layer
- ⚠ — implemented as a sound *box-lifted* Star approximation (looser
  than the predicate-preserving variants; tightening tracked as
  follow-up work)
- — — not soundly applicable to this layer type

Pre-existing Layers
-------------------

============================  ====  ====  ====  ========  ========
Layer                         Box   Star  Zono  Hexatope  Octatope
============================  ====  ====  ====  ========  ========
``nn.Linear``                 ✅    ✅    ✅    ✅        ✅
``nn.Conv1d``                 ✅    ✅    ✅    —         —
``nn.Conv2d``                 —     ✅    ✅    —         —
``nn.BatchNorm1d/2d``         ✅    ✅    ✅    ✅        ✅
``nn.Flatten``                ✅    ✅    ✅    ✅        ✅
``nn.MaxPool2d``              —     ✅    ✅    —         —
``nn.AvgPool2d``              —     ✅    ✅    —         —
GlobalAvgPool                 —     ✅    ✅    —         —
``nn.ReLU``                   ✅    ✅    ✅    ✅        ✅
``nn.LeakyReLU``              ✅    ✅    ✅    —         —
``nn.Sigmoid``                ✅    ✅    ✅    —         —
``nn.Tanh``                   ✅    ✅    ✅    —         —
Sign                          ✅    ✅    ✅    —         —
Pad                           —     ✅    ✅    —         —
Reduce                        ✅    ✅    ✅    —         —
Upsample                      —     ✅    ✅    —         —
============================  ====  ====  ====  ========  ========

Phase 1 — Activations & Normalisation
-------------------------------------

============================  ====  ====  ====
Layer                         Box   Star  Zono
============================  ====  ====  ====
``nn.ReLU6``                  ✅    ⚠     —
``nn.ELU``                    ✅    ⚠     —
``nn.GELU``                   ✅    ⚠     —
QuickGELU                     ✅    ⚠     —
``nn.SiLU``                   ✅    ⚠     —
``nn.Hardswish``              ✅    ⚠     —
``nn.LayerNorm``              ✅    ✨    —
RMSNorm                       ✅    ✨    —
``nn.GroupNorm``              ✅    ✨    —
GRN                           ✅    ✨    —
============================  ====  ====  ====

Phase 2 — MLP / Skip / DAG
--------------------------

==========================  ====  ====  ====  ========  ========
Layer                       Box   Star  Zono  Hexatope  Octatope
==========================  ====  ====  ====  ========  ========
LayerScale                  ✅    ✅    ✅    ✅        ✅
DropPath                    ✅    ✅    ✅    —         —
AddWithFrozenSkip           ✅    ✅    ✅    ✅        ✅
ConcatWithFrozenSkip        ✅    ✅    ✅    —         —
DagAdd (multi-input)        ✅    —     —     —         —
DagConcat (multi-input)     ✅    —     —     —         —
Concat2D (multi-input)      ✅    —     —     —         —
SelectiveFeatureFusion      ✅    —     —     —         —
MixFFN                      via dispatcher (recursive sub-module pass)
==========================  ====  ====  ====  ========  ========

Phase 3 — Attention
-------------------

==========================  ====  ====  ====  ========  ========
Layer                       Box   Star  Zono  Hexatope  Octatope
==========================  ====  ====  ====  ========  ========
SoftmaxAttention            ✅    ⚠     —     —         —
Sparsemax                   ✅    ⚠     —     —         —
LinearAttention             ✅    ⚠     —     —         —
SparseAttention             ✅    ⚠     —     —         —
CrossAttention              ✅    ⚠     —     —         —
GroupedQueryAttention       ✅    ⚠     —     —         —
MultiQueryAttention         ✅    ⚠     —     —         —
EfficientAttentionSR        via dispatcher (decomposed)
CausalMask                  ✅    ✅    ✅    ✅        ✅
RelativeAttentionBiasT5     ✅    ✅    ✅    ✅        ✅
RelativePositionBiasTable   ✅    ✅    ✅    ✅        ✅
==========================  ====  ====  ====  ========  ========

Phase 4 — Embeddings & Tokens
-----------------------------

==========================  ====  ====  ====  ========  ========
Layer                       Box   Star  Zono  Hexatope  Octatope
==========================  ====  ====  ====  ========  ========
``nn.Embedding``            ✅    ✅    ✅    —         —
SegmentEmbedding            ✅    ✅    ✅    ✅        ✅
PositionalEncoding          ✅    ✅    ✅    ✅        ✅
RoPE                        ✅    ✅    ✅    ✅        ✅
CLSToken                    ✅    ✅    ✅    —         —
DistillationToken           ✅    ✅    ✅    —         —
PatchEmbed                  via dispatcher (decomposed)
OverlapPatchEmbed           via dispatcher (decomposed)
==========================  ====  ====  ====  ========  ========

Phase 5 — Conv Variants & Specialty
-----------------------------------

==========================  ====  ====  ====  ========  ========
Layer                       Box   Star  Zono  Hexatope  Octatope
==========================  ====  ====  ====  ========  ========
TiedLinear                  ✅    ✅    ✅    ✅        ✅
ActionHead                  ✅    ✅    ✅    ✅        ✅
``nn.ConvTranspose2d``      ✅    ✅    ✅    ✅        ✅
DepthwiseConv               ✅    ✅    ✅    —         —
ConvTokenEmbedding          ✅    ✅    ✅    —         —
ActionTokenizer             ✅    ⚠     —     —         —
OpenMax                     ✅    ⚠     —     —         —
Pooler                      via dispatcher (decomposed)
ProjectionHead              via dispatcher (decomposed)
==========================  ====  ====  ====  ========  ========

Notes on Approximation Quality
------------------------------

The Star reach for the new ports falls into one of three categories.

**Exact / preserves predicates (✅).** Affine ports — linear maps,
translations, rotations, concatenations with constants — route through
:mod:`linear_reach`'s exact Star reach, so the input predicates carry
through to the output basis matrix unchanged. This is the same
guarantee n2v already provides for ``nn.Linear`` and ``nn.BatchNorm``.

**Predicate-preserving with slack (✨).** The four normalisation layers
(``LayerNorm``, ``RMSNorm``, ``GroupNorm``, ``GRN``) cannot be made
fully affine in the input, so we split the layer into an *exact* linear
part (mean subtraction for LayerNorm/GroupNorm, the ``+ x`` residual
for GRN) and a *bounded* nonlinear part (the ``1/sigma`` scale factor
or the ``gamma * (x * nx)`` correction). The linear part carries the
input predicates exactly; the nonlinear part is encoded as per-feature
slack predicates bounded to ``[-1, 1]`` and added as new columns in
the output basis. See :mod:`n2v.nn.layer_ops._layernorm_star`.

**Box-lifted (⚠).** The smooth activations (``GELU``, ``QuickGELU``,
``SiLU``, ``HardSwish``, ``ReLU6``, ``ELU``) and the attention family
(``SoftmaxAttention``, ``Sparsemax``, ``LinearAttention``, the
attention variants) currently compute an axis-aligned interval bound
and lift it to a fresh Star via ``Star.from_bounds``. This is sound
but discards predicate dependence. The ``sigmoid_reach`` template (a
three-region CROWN relaxation) is a clean way to upgrade these to
predicate-preserving in a follow-up. For ``SoftmaxAttention`` the
predicate-preserving variant is research-grade (see nnVLA
``softmax_attention/methods/star.py``).

Probabilistic Verification Scope
--------------------------------

n2v's probabilistic verification path (``method='probabilistic'`` and
``method='hybrid'`` on :class:`~n2v.nn.NeuralNetwork.reach`) uses
:class:`~n2v.sets.ProbabilisticBox` together with conformal-inference
surrogates in :mod:`n2v.probabilistic`. ProbabilisticBox is a special
case: rather than dispatching to per-layer reach functions, the
verifier evaluates a *surrogate* model on calibration samples, so a
layer only needs to be forward-callable in PyTorch — its set
reachability is bypassed entirely.

**All ported layers are forward-callable in PyTorch and therefore
compatible with the probabilistic path.** No per-layer ProbabilisticBox
reach functions are required (and none are implemented). If you build a
transformer with the new layers and call
``model.reach(input_set, method='probabilistic')``, you get conformal
coverage guarantees on the output without any new code paths firing.

For the **deterministic** ``method='exact'`` and ``method='approx'``
paths the per-layer set support shown in the tables above governs
behaviour, and ``NotImplementedError`` is raised for any unsupported
(layer, set type) pair.

Adding a New Layer
------------------

Two paths are supported:

1. **Edit the dispatcher** — add a new ``<name>_reach.py`` module with
   the per-set-type functions, then add ``isinstance`` branches in the
   corresponding ``_reach_layer_<set_type>`` router inside
   :mod:`~n2v.nn.layer_ops.dispatcher`. This is the original pattern
   and remains supported.

2. **Use the declarative registry** — implement the reach function and
   decorate it with ``@register(LayerCls, SetCls)`` from
   :mod:`n2v.nn.layer_ops.registry`. The dispatcher consults the
   registry as a fallback after its ``isinstance`` chains, so a
   registered handler fires whenever no built-in branch matches.

The registry path is intended for third-party layers and new
contributions that don't need to touch ``dispatcher.py``. See
``tests/unit/layer_ops/test_registry.py`` for end-to-end examples.
