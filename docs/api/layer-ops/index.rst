Layer Operations
================

Per-layer reachability functions that take a PyTorch layer plus a list
of input sets and return a list of output sets. These are wired into
the unified :func:`~n2v.nn.layer_ops.dispatcher.reach_layer` entry point
so :class:`~n2v.nn.NeuralNetwork` can compute reachability over any
supported architecture.

How Dispatch Works
------------------

:func:`~n2v.nn.layer_ops.dispatcher.reach_layer` inspects the
*input-set type* (Star / ImageStar, Zono / ImageZono, Box, Hexatope,
Octatope) and routes to a private ``_reach_layer_<set_type>`` function.
Each of these routers contains an ``isinstance(layer, ...)`` chain that
selects the correct per-layer module from
:mod:`n2v.nn.layer_ops`.

The dispatcher also handles a few control-flow / shape ops directly
(``nn.Identity``, ``nn.Dropout``, ``nn.Sequential``, and ONNX operators
like ``OnnxNeg`` / ``OnnxTranspose`` / ``OnnxCast``), recursing into
sub-layers as needed.

To add a new layer:

1. Add a new module ``n2v/nn/layer_ops/<name>_reach.py`` exposing
   ``<name>_box``, ``<name>_star``, ``<name>_zono`` etc., matching the
   set types that the underlying operation supports soundly.
2. Add an ``isinstance`` branch in the appropriate set-type router(s)
   inside :mod:`~n2v.nn.layer_ops.dispatcher`.
3. Add unit tests under ``tests/unit/layer_ops/`` and (recommended) an
   oracle scenario under ``tests/oracles/<name>.py``.

Dispatcher Entry Point
----------------------

.. autofunction:: n2v.nn.layer_ops.dispatcher.reach_layer

Per-Category Reference
----------------------

.. toctree::
   :maxdepth: 2

   activations
   normalisation
   linear-conv
   attention
   dag-skip
   pooling-shape
   embeddings
   specialty
   coverage
