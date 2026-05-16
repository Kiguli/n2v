"""Transposed Conv2d reachability.

A transposed convolution is the gradient operator of an ordinary
convolution and is itself affine in the input. The implementation
materialises the equivalent dense ``W^T`` matrix for the configured
kernel/stride/padding and forwards to :mod:`linear_reach`.

Because ``W^T`` can be large, this helper is intended for small feature
maps (test, verification of small ViT/U-Net decoders).

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _explicit_linear(layer: nn.ConvTranspose2d, input_shape: tuple) -> nn.Linear:
    """Materialise ConvTranspose2d as an nn.Linear by enumerating one-hot inputs."""
    c_in, h_in, w_in = input_shape
    n_in = c_in * h_in * w_in
    with torch.no_grad():
        eye = torch.eye(n_in).reshape(n_in, c_in, h_in, w_in)
        out = layer(eye)
        out_flat = out.reshape(n_in, -1).T  # (n_out, n_in)
        n_out = out_flat.shape[0]
        dense = nn.Linear(n_in, n_out, bias=True)
        dense.weight.copy_(out_flat.float())
        bias = layer(torch.zeros(1, c_in, h_in, w_in)).reshape(-1).float()
        dense.bias.copy_(bias)
    return dense


def _input_shape_from_layer(layer, input_dim: int) -> tuple:
    c_in = int(layer.in_channels)
    side = int(np.sqrt(input_dim // c_in))
    return c_in, side, side


def conv2d_transpose_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        shape = _input_shape_from_layer(layer, s.dim)
        out.extend(linear_reach.linear_star(_explicit_linear(layer, shape), [s]))
    return out


def conv2d_transpose_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        shape = _input_shape_from_layer(layer, b.dim)
        out.extend(linear_reach.linear_box(_explicit_linear(layer, shape), [b]))
    return out


def conv2d_transpose_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        shape = _input_shape_from_layer(layer, z.dim)
        out.extend(linear_reach.linear_zono(_explicit_linear(layer, shape), [z]))
    return out
