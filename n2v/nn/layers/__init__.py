"""User-facing layer wrappers for the ViT building blocks.

These thin ``nn.Module`` wrappers let users build Vision-Transformer
models in PyTorch using the names n2v recognises, and the dispatcher
detects them via ``isinstance``. Standard ``torch.nn`` layers used by a
ViT (``nn.Linear``, ``nn.Conv2d``, ``nn.LayerNorm``, ``nn.GELU``) are
handled directly by the dispatcher and need no wrapper.

Importing a wrapper does **not** affect dispatch -- registration is done
by ``n2v.nn.layer_ops.dispatcher`` via ``isinstance`` chains.
"""

from n2v.nn.layers.softmax_attention import SoftmaxAttention
from n2v.nn.layers.cls_token import CLSToken
from n2v.nn.layers.patch_embed import PatchEmbed
from n2v.nn.layers.pooler import Pooler
from n2v.nn.layers.positional_encoding import PositionalEncoding

__all__ = [
    "SoftmaxAttention",
    "CLSToken",
    "PatchEmbed",
    "Pooler",
    "PositionalEncoding",
]
