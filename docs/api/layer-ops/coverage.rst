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
