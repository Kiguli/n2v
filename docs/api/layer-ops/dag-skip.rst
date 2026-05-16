DAG & Skip
==========

Multi-input and skip-connection layers. ``DagAdd`` / ``DagConcat`` /
``Concat2D`` / ``SelectiveFeatureFusion`` accept multiple input streams
via the dispatcher's ``extras`` parameter; the others (LayerScale,
DropPath, FrozenSkip helpers, MixFFN) are single-input.

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
