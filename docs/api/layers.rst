Layer Wrappers
==============

``n2v.nn.layers`` provides thin ``nn.Module`` wrappers for layer types
not present in ``torch.nn`` (RMSNorm, GRN, RoPE, CausalMask,
LayerScale, DropPath, SoftmaxAttention, ...). These wrappers exist so
that users can build models in PyTorch using the same names nnVLA
exposes and n2v's :mod:`~n2v.nn.layer_ops.dispatcher` will route to the
correct reachability implementation by ``isinstance``.

Importing a wrapper does **not** affect dispatch on its own —
registration is done by the dispatcher's ``isinstance`` chains.

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
