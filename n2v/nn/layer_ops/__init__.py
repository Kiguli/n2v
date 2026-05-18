"""
Layer operation modules - reachability for specific layer types.

Internal modules that implement reachability for individual layer types
across different set representations (Star, Zono, Box, Hexatope, Octatope).

These are typically accessed through the high-level NeuralNetwork.reach() API
rather than directly.
"""

# Existing layer ops (pre-port).
from n2v.nn.layer_ops import linear_reach
from n2v.nn.layer_ops import relu_reach
from n2v.nn.layer_ops import flatten_reach
from n2v.nn.layer_ops import conv2d_reach
from n2v.nn.layer_ops import conv1d_reach
from n2v.nn.layer_ops import maxpool2d_reach
from n2v.nn.layer_ops import avgpool2d_reach
from n2v.nn.layer_ops import global_avgpool_reach
from n2v.nn.layer_ops import batchnorm_reach
from n2v.nn.layer_ops import pad_reach
from n2v.nn.layer_ops import reduce_reach
from n2v.nn.layer_ops import upsample_reach
from n2v.nn.layer_ops import leakyrelu_reach
from n2v.nn.layer_ops import sigmoid_reach
from n2v.nn.layer_ops import tanh_reach
from n2v.nn.layer_ops import sign_reach

# Phase 1 ports — activations & normalisation.
from n2v.nn.layer_ops import relu6_reach
from n2v.nn.layer_ops import elu_reach
from n2v.nn.layer_ops import gelu_reach
from n2v.nn.layer_ops import quickgelu_reach
from n2v.nn.layer_ops import silu_reach
from n2v.nn.layer_ops import hardswish_reach
from n2v.nn.layer_ops import layernorm_reach
from n2v.nn.layer_ops import rmsnorm_reach
from n2v.nn.layer_ops import groupnorm_reach
from n2v.nn.layer_ops import grn_reach

# Phase 2 ports — MLP / skip / DAG.
from n2v.nn.layer_ops import layerscale_reach
from n2v.nn.layer_ops import drop_path_reach
from n2v.nn.layer_ops import add_with_frozen_skip_reach
from n2v.nn.layer_ops import concat_with_frozen_skip_reach
from n2v.nn.layer_ops import dag_add_reach
from n2v.nn.layer_ops import dag_concat_reach
from n2v.nn.layer_ops import concat2d_reach
from n2v.nn.layer_ops import selective_feature_fusion_reach
from n2v.nn.layer_ops import mix_ffn_reach

# Phase 3 ports — attention.
from n2v.nn.layer_ops import softmax_attention_reach
from n2v.nn.layer_ops import causal_mask_reach
from n2v.nn.layer_ops import sparsemax_reach
from n2v.nn.layer_ops import relative_attention_bias_t5_reach
from n2v.nn.layer_ops import relative_position_bias_table_reach
from n2v.nn.layer_ops import linear_attention_reach
from n2v.nn.layer_ops import efficient_attention_sr_reach
from n2v.nn.layer_ops import sparse_attention_reach
from n2v.nn.layer_ops import cross_attention_reach
from n2v.nn.layer_ops import grouped_query_attention_reach
from n2v.nn.layer_ops import multi_query_attention_reach

# Phase 4 ports — embeddings & tokens.
from n2v.nn.layer_ops import embedding_reach
from n2v.nn.layer_ops import segment_embedding_reach
from n2v.nn.layer_ops import positional_encoding_reach
from n2v.nn.layer_ops import rope_reach
from n2v.nn.layer_ops import cls_token_reach
from n2v.nn.layer_ops import distillation_token_reach

# Phase 5 ports — conv variants & specialty.
from n2v.nn.layer_ops import tied_linear_reach
from n2v.nn.layer_ops import conv2d_transpose_reach
from n2v.nn.layer_ops import depthwise_conv_reach
from n2v.nn.layer_ops import conv_token_embedding_reach
from n2v.nn.layer_ops import action_head_reach
from n2v.nn.layer_ops import action_tokenizer_reach
from n2v.nn.layer_ops import openmax_reach
from n2v.nn.layer_ops import pooler_reach
from n2v.nn.layer_ops import projection_head_reach

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.layer_ops.registry import register, lookup

__all__ = [
    # Pre-existing
    "linear_reach", "relu_reach", "flatten_reach", "conv2d_reach", "conv1d_reach",
    "maxpool2d_reach", "avgpool2d_reach", "global_avgpool_reach",
    "batchnorm_reach", "pad_reach", "reduce_reach", "upsample_reach",
    "leakyrelu_reach", "sigmoid_reach", "tanh_reach", "sign_reach",
    # Phase 1
    "relu6_reach", "elu_reach", "gelu_reach", "quickgelu_reach", "silu_reach",
    "hardswish_reach", "layernorm_reach", "rmsnorm_reach", "groupnorm_reach",
    "grn_reach",
    # Phase 2
    "layerscale_reach", "drop_path_reach", "add_with_frozen_skip_reach",
    "concat_with_frozen_skip_reach", "dag_add_reach", "dag_concat_reach",
    "concat2d_reach", "selective_feature_fusion_reach", "mix_ffn_reach",
    # Phase 3
    "softmax_attention_reach", "causal_mask_reach", "sparsemax_reach",
    "relative_attention_bias_t5_reach", "relative_position_bias_table_reach",
    "linear_attention_reach", "efficient_attention_sr_reach",
    "sparse_attention_reach", "cross_attention_reach",
    "grouped_query_attention_reach", "multi_query_attention_reach",
    # Phase 4
    "embedding_reach", "segment_embedding_reach", "positional_encoding_reach",
    "rope_reach", "cls_token_reach", "distillation_token_reach",
    # Phase 5
    "tied_linear_reach", "conv2d_transpose_reach", "depthwise_conv_reach",
    "conv_token_embedding_reach", "action_head_reach", "action_tokenizer_reach",
    "openmax_reach", "pooler_reach", "projection_head_reach",
    # Entry points
    "reach_layer", "register", "lookup",
]
