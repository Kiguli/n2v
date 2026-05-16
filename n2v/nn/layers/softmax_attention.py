"""SoftmaxAttention wrapper module.

Wraps the canonical scaled dot-product attention so that n2v's dispatcher
can detect the op via ``isinstance``. The forward path delegates to
``torch.nn.functional.scaled_dot_product_attention`` for parity with
PyTorch built-ins.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttention(nn.Module):
    """Plain scaled dot-product softmax attention.

    Forward signature matches ``F.scaled_dot_product_attention``:
    ``forward(q, k, v, attn_mask=None)``.
    """

    def __init__(self, d_head: int | None = None):
        super().__init__()
        self.d_head = d_head

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d_head = self.d_head if self.d_head is not None else q.size(-1)
        scale = 1.0 / math.sqrt(d_head)
        logits = (q @ k.transpose(-1, -2)) * scale
        if attn_mask is not None:
            logits = logits + attn_mask
        attn = F.softmax(logits, dim=-1)
        return attn @ v
