"""
Linear layer reachability operations.

Works directly with PyTorch nn.Linear layers.

T1-7 (ViT enable): nn.Linear in a transformer is applied PER TOKEN to a
``(B, L, D_in)`` sequence, producing ``(B, L, D_out)``. After
``PatchEmbed`` and ``CLSToken`` the reach set is flat with dim
``L * D_in``. ``layer.weight`` is ``(D_out, D_in)`` -- not
``(L * D_out, L * D_in)`` -- so the existing ``zono.affine_map(W, b)``
call raises a shape mismatch. We block-tile the weight (and bias) to
``(L * D_out, L * D_in)`` (Kronecker product with the identity) when the
input flat dim is a multiple of ``layer.in_features`` greater than 1.
"""

import torch.nn as nn
import numpy as np
from typing import List, Tuple
from n2v.sets import Star, Zono, Box, Hexatope, Octatope


def _maybe_block_tile_linear(
    W: np.ndarray, b: np.ndarray | None, input_flat_dim: int,
) -> Tuple[np.ndarray, np.ndarray | None, int]:
    """If ``input_flat_dim > layer.in_features`` and divides cleanly, return
    a block-diagonal expansion that applies ``W`` to each ``in_features``
    chunk independently. Otherwise return ``W, b`` unchanged.

    Returns ``(W_tiled, b_tiled, L)`` where ``L`` is the number of token
    blocks (1 when no tiling is needed).
    """
    in_features = W.shape[1]
    if input_flat_dim == in_features:
        return W, b, 1
    if input_flat_dim % in_features != 0:
        # Caller will raise a clear shape error.
        return W, b, 0
    L = input_flat_dim // in_features
    # Kronecker product I_L (X) W gives a block-diagonal layout that
    # applies W independently to each of the L token chunks.
    W_tiled = np.kron(np.eye(L, dtype=W.dtype), W)
    if b is not None:
        b_tiled = np.tile(b.reshape(-1), L)
    else:
        b_tiled = None
    return W_tiled, b_tiled, L


def linear_star(layer: nn.Linear, input_stars: List[Star]) -> List[Star]:
    """
    Exact reachability for Linear layer using Star sets.

    Args:
        layer: PyTorch nn.Linear layer
        input_stars: List of input Star sets

    Returns:
        List of output Star sets
    """
    W = layer.weight.detach().cpu().numpy()  # (out_features, in_features)
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_stars = []
    for star in input_stars:
        W_use, b_use, _ = _maybe_block_tile_linear(W, b, star.dim)
        if b_use is not None:
            b_reshaped = b_use.reshape(-1, 1)
            output_star = star.affine_map(W_use, b_reshaped)
        else:
            output_star = star.affine_map(W_use)
        output_stars.append(output_star)

    return output_stars


def linear_zono(layer: nn.Linear, input_zonos: List[Zono]) -> List[Zono]:
    """
    Exact reachability for Linear layer using Zonotopes.

    Args:
        layer: PyTorch nn.Linear layer
        input_zonos: List of input Zonotopes

    Returns:
        List of output Zonotopes
    """
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_zonos = []
    for zono in input_zonos:
        W_use, b_use, _ = _maybe_block_tile_linear(W, b, zono.dim)
        if b_use is not None:
            b_reshaped = b_use.reshape(-1, 1)
            output_zono = zono.affine_map(W_use, b_reshaped)
        else:
            output_zono = zono.affine_map(W_use)
        output_zonos.append(output_zono)

    return output_zonos


def linear_box(layer: nn.Linear, input_boxes: List[Box]) -> List[Box]:
    """
    Exact reachability for Linear layer using Boxes.

    Args:
        layer: PyTorch nn.Linear layer
        input_boxes: List of input Boxes

    Returns:
        List of output Boxes
    """
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_boxes = []
    for box in input_boxes:
        if b is not None:
            b_reshaped = b.reshape(-1, 1)
            output_box = box.affine_map(W, b_reshaped)
        else:
            output_box = box.affine_map(W)
        output_boxes.append(output_box)

    return output_boxes


def linear_hexatope(layer: nn.Linear, input_hexatopes: List[Hexatope]) -> List[Hexatope]:
    """
    Exact reachability for Linear layer using Hexatopes.

    Args:
        layer: PyTorch nn.Linear layer
        input_hexatopes: List of input Hexatopes

    Returns:
        List of output Hexatopes
    """
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_hexatopes = []
    for hexatope in input_hexatopes:
        if b is not None:
            b_reshaped = b.reshape(-1, 1)
            output_hexatope = hexatope.affine_map(W, b_reshaped)
        else:
            output_hexatope = hexatope.affine_map(W)
        output_hexatopes.append(output_hexatope)

    return output_hexatopes


def linear_octatope(layer: nn.Linear, input_octatopes: List[Octatope]) -> List[Octatope]:
    """
    Exact reachability for Linear layer using Octatopes.

    Args:
        layer: PyTorch nn.Linear layer
        input_octatopes: List of input Octatopes

    Returns:
        List of output Octatopes
    """
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_octatopes = []
    for octatope in input_octatopes:
        if b is not None:
            b_reshaped = b.reshape(-1, 1)
            output_octatope = octatope.affine_map(W, b_reshaped)
        else:
            output_octatope = octatope.affine_map(W)
        output_octatopes.append(output_octatope)

    return output_octatopes
