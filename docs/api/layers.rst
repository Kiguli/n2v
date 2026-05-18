Layer Wrappers
==============

``n2v.nn.layers`` provides thin ``nn.Module`` wrappers for layer types
not present in ``torch.nn`` (RMSNorm, GRN, RoPE, CausalMask,
LayerScale, DropPath, SoftmaxAttention, ...). The wrappers exist so
users can build models in PyTorch with the same names this port
follows.

Most single-input wrappers are detected by the dispatcher via
``isinstance`` and route through :func:`~n2v.nn.layer_ops.dispatcher.reach_layer`:
``RMSNorm``, ``GRN``, ``LayerScale``, ``DropPath``, ``CausalMask``,
``RoPE``, ``CLSToken``, ``DistillationToken``, ``TiedLinear``,
``OpenMax``, ``ActionHead``, ``ActionTokenizer``, ``Pooler``,
``ProjectionHead``, ``PositionalEncoding``,
``RelativeAttentionBiasT5``, ``RelativePositionBiasTable``,
``AddWithFrozenSkip``, ``ConcatWithFrozenSkip``.

Multi-input wrappers (``DagAdd``, ``DagConcat``, ``Concat2D``,
``SelectiveFeatureFusion``, ``SoftmaxAttention``) are detected by the
*graph-level* multi-input dispatcher in :mod:`n2v.nn.reach`. They are
not invoked through the single-input ``reach_layer`` path; calling
``reach_layer`` on one of these raises ``NotImplementedError``.

Wrappers that are intentionally NOT routed by the dispatcher and should
appear only inside a :class:`torch.fx` trace decomposed into primitives
(``Linear`` + ``Conv2d`` + activation + ...): ``MixFFN``,
``ParallelResidual``, ``PatchEmbed``, ``OverlapPatchEmbed``. These are
verified through their decomposed primitives, not as opaque modules.

Importing a wrapper does **not** affect dispatch on its own —
dispatcher routing is via the ``isinstance`` chains and graph-level
multi-input hook.

.. automodule:: n2v.nn.layers
   :members:
   :undoc-members: False

Available Wrappers
------------------

.. autoclass:: n2v.nn.layers.RMSNorm
   :members:

.. autoclass:: n2v.nn.layers.GRN
   :members:

.. autoclass:: n2v.nn.layers.LayerScale
   :members:

.. autoclass:: n2v.nn.layers.DropPath
   :members:

.. autoclass:: n2v.nn.layers.SoftmaxAttention
   :members:

.. autoclass:: n2v.nn.layers.CausalMask
   :members:

.. autoclass:: n2v.nn.layers.RoPE
   :members:

.. autoclass:: n2v.nn.layers.CLSToken
   :members:

.. autoclass:: n2v.nn.layers.DistillationToken
   :members:

.. autoclass:: n2v.nn.layers.TiedLinear
   :members:

.. autoclass:: n2v.nn.layers.OpenMax
   :members:

.. autoclass:: n2v.nn.layers.ActionHead
   :members:

.. autoclass:: n2v.nn.layers.ActionTokenizer
   :members:

.. autoclass:: n2v.nn.layers.PatchEmbed
   :members:

.. autoclass:: n2v.nn.layers.OverlapPatchEmbed
   :members:

.. autoclass:: n2v.nn.layers.ParallelResidual
   :members:

.. autoclass:: n2v.nn.layers.MixFFN
   :members:

.. autoclass:: n2v.nn.layers.Pooler
   :members:

.. autoclass:: n2v.nn.layers.ProjectionHead
   :members:

.. autoclass:: n2v.nn.layers.SelectiveFeatureFusion
   :members:

.. autoclass:: n2v.nn.layers.SegmentEmbedding
   :members:

.. autoclass:: n2v.nn.layers.PositionalEncoding
   :members:

.. autoclass:: n2v.nn.layers.RelativeAttentionBiasT5
   :members:

.. autoclass:: n2v.nn.layers.RelativePositionBiasTable
   :members:

.. autoclass:: n2v.nn.layers.DagAdd
   :members:

.. autoclass:: n2v.nn.layers.DagConcat
   :members:

.. autoclass:: n2v.nn.layers.Concat2D
   :members:

.. autoclass:: n2v.nn.layers.AddWithFrozenSkip
   :members:

.. autoclass:: n2v.nn.layers.ConcatWithFrozenSkip
   :members:
