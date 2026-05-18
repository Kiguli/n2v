"""MixFFN reachability — explicitly *not* implemented as a passthrough.

``MixFFN.forward`` performs a sequence-to-image reshape/transpose
before the depthwise Conv2d and the inverse afterward. A naive serial
sub-module pass ``fc1 → dwconv → act → fc2`` omits those reshapes and
either passes wrong-shaped sets into ``Conv2d`` or verifies a
different operation entirely.

n2v relies on torch.fx tracing to decompose MixFFN into its primitives
(``Linear`` + reshape + ``Conv2d`` + ``GELU`` + reshape + ``Linear``),
which gives each step a correctly-shaped input set. If the fx tracer
ever passes an opaque ``MixFFN`` module to the dispatcher, this helper
now raises so the unsoundness is loud rather than silent.
"""

from __future__ import annotations

from typing import List


def mix_ffn_passthrough(layer, input_sets: List, method: str = "exact", **kwargs):
    raise NotImplementedError(
        "MixFFN reach is not implemented as a passthrough: the inner "
        "sequence-to-image reshape between Linear and Conv2d would be "
        "lost, silently changing the verified operation. Rely on "
        "torch.fx to decompose MixFFN into primitives, or split the "
        "block manually in your model definition."
    )
