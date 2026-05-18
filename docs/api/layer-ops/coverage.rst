Layer Coverage Matrix
=====================

Per-layer set-type support after the nnVLA port. The columns are the
five set types implemented in :mod:`n2v.sets`: Box, Star (and
ImageStar), Zono (and ImageZono), Hexatope, Octatope.

Legend
------

- ✅ — implemented and routed through the dispatcher
- ⚠ — implemented as a sound box-lifted approximation (looser than the
  tightest known relaxation; tracked in the draft PR description as a
  follow-up tightening)
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
``nn.LayerNorm``              ✅    ⚠     —
RMSNorm                       ✅    ⚠     —
``nn.GroupNorm``              ✅    ⚠     —
GRN                           ✅    ⚠     —
============================  ====  ====  ====

Phase 2 — MLP / Skip / DAG
--------------------------

============================  ====  ====  ====
Layer                         Box   Star  Zono
============================  ====  ====  ====
LayerScale                    ✅    ✅    ✅
DropPath                      ✅    ✅    ✅
AddWithFrozenSkip             ✅    ✅    ✅
ConcatWithFrozenSkip          ✅    ✅    ✅
DagAdd (multi-input)          ✅    —     —
DagConcat (multi-input)       ✅    —     —
Concat2D (multi-input)        ✅    —     —
SelectiveFeatureFusion        ✅    —     —
MixFFN (decomposed)           via dispatcher
============================  ====  ====  ====

Phase 3 — Attention
-------------------

============================  ====  ====  ====
Layer                         Box   Star  Zono
============================  ====  ====  ====
SoftmaxAttention              ✅    ⚠     —
Sparsemax                     ✅    ⚠     —
LinearAttention               ✅    ⚠     —
SparseAttention               ✅    ⚠     —
CrossAttention                ✅    ⚠     —
GroupedQueryAttention         ✅    ⚠     —
MultiQueryAttention           ✅    ⚠     —
EfficientAttentionSR          via dispatcher (decomposed)
CausalMask                    ✅    ✅    ✅
RelativeAttentionBiasT5       ✅    ✅    ✅
RelativePositionBiasTable     ✅    ✅    ✅
============================  ====  ====  ====

Phase 4 — Embeddings & Tokens
-----------------------------

============================  ====  ====  ====
Layer                         Box   Star  Zono
============================  ====  ====  ====
``nn.Embedding``              ✅    ✅    ✅
SegmentEmbedding              ✅    ✅    ✅
PositionalEncoding            ✅    ✅    ✅
RoPE                          ✅    ✅    ✅
CLSToken                      ✅    ✅    ✅
DistillationToken             ✅    ✅    ✅
PatchEmbed (decomposed)       via dispatcher
OverlapPatchEmbed (decomposed) via dispatcher
============================  ====  ====  ====

Phase 5 — Conv Variants & Specialty
-----------------------------------

============================  ====  ====  ====  ========  ========
Layer                         Box   Star  Zono  Hexatope  Octatope
============================  ====  ====  ====  ========  ========
TiedLinear                    ✅    ✅    ✅    ✅        ✅
ActionHead                    ✅    ✅    ✅    ✅        ✅
``nn.ConvTranspose2d``        ✅    ✅    ✅    —         —
DepthwiseConv                 ✅    ✅    ✅    —         —
ConvTokenEmbedding            ✅    ✅    ✅    —         —
ActionTokenizer               ✅    ⚠     —     —         —
OpenMax                       ✅    ⚠     —     —         —
Pooler                        via dispatcher (decomposed)
ProjectionHead                via dispatcher (decomposed)
============================  ====  ====  ====  ========  ========

Notes
-----

The ⚠ entries are sound but use a box-lifted Star approximation. Each
is tracked in the draft PR description with a pointer to the
corresponding nnVLA ``methods/star.py`` implementation; tightening
these is the main follow-up before the PR is marked ready for review.

**Update (post-integration pass)**: ``LayerNorm`` and ``RMSNorm`` now
ship a *predicate-preserving* Star reach that subtracts the input mean
exactly and adds a per-feature slack predicate for the scale interval
(see :mod:`n2v.nn.layer_ops._layernorm_star`). The remaining ⚠ entries
for ``GELU`` / ``QuickGELU`` / ``SiLU`` / ``HardSwish`` / ``GroupNorm``
/ ``GRN`` / ``SoftmaxAttention`` / ``Sparsemax`` are still box-lifted.

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

**All layers ported in this work are forward-callable in PyTorch and
therefore compatible with the probabilistic path.** No per-layer
ProbabilisticBox reach functions are required (and none are
implemented). If you build a transformer with the new layers and call
``model.reach(input_set, method='probabilistic')``, you get conformal
coverage guarantees on the output without any new code paths firing.

For the **deterministic** ``method='exact'`` and ``method='approx'``
paths the per-layer set support shown in the tables above governs
behaviour, and ``NotImplementedError`` is raised for any unsupported
(layer, set type) pair.
