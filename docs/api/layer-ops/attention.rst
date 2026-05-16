Attention
=========

Attention layers added during the nnVLA port. SoftmaxAttention and its
variants form the core; the auxiliary modules (CausalMask, relative
bias tables, Sparsemax) attach to them via the dispatcher's
single-input routes.

SoftmaxAttention and Variants
-----------------------------

.. automodule:: n2v.nn.layer_ops.softmax_attention_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.cross_attention_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.grouped_query_attention_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.multi_query_attention_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.sparse_attention_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.linear_attention_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.efficient_attention_sr_reach
   :members:
   :undoc-members: False

Masks & Biases
--------------

.. automodule:: n2v.nn.layer_ops.causal_mask_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.relative_attention_bias_t5_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.relative_position_bias_table_reach
   :members:
   :undoc-members: False
