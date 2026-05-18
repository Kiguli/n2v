DAG & Skip
==========

Multi-input and skip-connection layers.

The single-input ports (``LayerScale``, ``DropPath``, the FrozenSkip
helpers) route through :func:`~n2v.nn.layer_ops.dispatcher.reach_layer`
as standard layers.

The multi-input ports (``DagAdd``, ``DagConcat``, ``Concat2D``,
``SelectiveFeatureFusion``) take a primary stream plus an ``extras``
list of additional input-port streams. End-to-end dispatch of these
wrappers through :class:`~n2v.nn.NeuralNetwork.reach` requires the
graph-level multi-input hook in :mod:`n2v.nn.reach._handle_multi_input_op`,
which is wired up for ``Box`` but has open corner cases for ``Star``
under certain fx traces — see the PR description for the current
status. The per-op helpers documented below operate correctly when
invoked directly with the right stream layout.

LayerScale & DropPath
---------------------

.. automodule:: n2v.nn.layer_ops.layerscale_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.drop_path_reach
   :members:
   :undoc-members: False

Frozen Skip
-----------

.. automodule:: n2v.nn.layer_ops.add_with_frozen_skip_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.concat_with_frozen_skip_reach
   :members:
   :undoc-members: False

DAG Multi-Input
---------------

.. automodule:: n2v.nn.layer_ops.dag_add_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.dag_concat_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.concat2d_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.selective_feature_fusion_reach
   :members:
   :undoc-members: False

MixFFN
------

.. automodule:: n2v.nn.layer_ops.mix_ffn_reach
   :members:
   :undoc-members: False
