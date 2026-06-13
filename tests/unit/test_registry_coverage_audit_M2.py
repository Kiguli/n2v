"""Registry-coverage CI artefact + completeness check (PR-1 audit M2).

The PR-1 workflow audit's completeness critic flagged the missing
``__all__`` x set-type cross-product check as the single highest-leverage
"missed category". Without a mechanical guard, every new wrapper that
lands in ``n2v.nn.layers.__all__`` can silently miss a dispatcher branch
for one or more set types -- repeating the I5 / I7 regressions where
``N2VTracer`` leaf-treated wrappers that had no handler and any call
fell through ``_registry_lookup`` -> ``None`` -> ``NotImplementedError``.

This test introspects ``n2v.nn.layer_ops.dispatcher`` source and builds
a ``{wrapper: {set_type: bool}}`` coverage matrix. It then:

1. Asserts every wrapper in ``__all__`` has AT LEAST ONE dispatcher
   branch (the I5 / I7 hard failure mode -- N2VTracer would leaf-treat
   a wrapper with zero branches).
2. Reports partial-coverage cells (wrapper has Box but not Star, etc.)
   as a WARNING in the test output. We don't fail on partial coverage
   because some wrappers are deliberately Box-only per nnVLA catalog
   (DagAdd, Concat2D, SelectiveFeatureFusion -- explicitly
   marked in dispatcher source).
3. Emits a printable coverage table for human review (also written to
   ``coverage_matrix.txt`` under pytest's tmp_path so CI can attach it
   without touching the source tree).

The coverage check is intentionally cheap: source-level grep, no
imports of every wrapper, no instantiation. Runs in <100ms.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest


# Set types whose dispatcher branches we audit. The ViT layer wrappers
# in this PR target Box / Star / Zono; the pre-existing Hexatope /
# Octatope set types remain available for the base layers but are not
# part of the ViT wrapper surface, so they are not audited here.
_SET_TYPES = ("star", "zono", "box")

# Wrappers known to be Box-only per the nnVLA catalog. (None of the ViT
# wrappers in this PR are Box-only.)
_BOX_ONLY = set()

# Wrappers intentionally kept OUT of the N2VTracer leaf list because fx
# decomposes them into primitives the dispatcher handles directly.
_TRACER_EXCLUDED = set()

# Wrappers dispatched via a subclass-of-stdlib-base isinstance check
# rather than a dedicated ``_X`` alias. (None in this PR -- the ViT
# wrappers all have dedicated dispatcher branches.)
_VIA_PARENT_CLASS = {}


def _load_dispatcher_source() -> str:
    from n2v.nn.layer_ops import dispatcher
    return pathlib.Path(dispatcher.__file__).read_text(encoding="utf-8")


def _build_import_alias_map(src: str) -> dict[str, str]:
    """Map ``_X`` dispatcher-local alias -> wrapper class name.

    Example: ``from n2v.nn.layers.patch_embed import PatchEmbed as _PatchEmbed``
    -> ``{'_PatchEmbed': 'PatchEmbed'}``.
    """
    mapping: dict[str, str] = {}
    pattern = re.compile(
        r"from n2v\.nn\.layers\.\w+ import (\w+) as (_\w+)",
    )
    for cls, alias in pattern.findall(src):
        mapping[alias] = cls
    return mapping


def _function_source(src: str, fn_name: str) -> str:
    """Slice the source of a top-level function ``fn_name`` out of ``src``.

    Returns '' if not found.
    """
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            lines = src.splitlines()
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    return ""


def _branches_in(fn_src: str, alias_map: dict[str, str]) -> set[str]:
    """Return the set of wrapper class names referenced by
    ``isinstance(layer, _X)`` (or ``isinstance(layer, (_X, ...))``) in
    the given function source.
    """
    out: set[str] = set()
    # isinstance(layer, _X)
    for alias in re.findall(
        r"isinstance\(\s*layer\s*,\s*(_\w+)\s*\)", fn_src,
    ):
        if alias in alias_map:
            out.add(alias_map[alias])
    # isinstance(layer, (_X, _Y, ...))
    for tup in re.findall(
        r"isinstance\(\s*layer\s*,\s*\(([^)]+)\)\s*\)", fn_src,
    ):
        for alias in re.findall(r"_\w+", tup):
            if alias in alias_map:
                out.add(alias_map[alias])
    return out


def _coverage_matrix() -> tuple[dict[str, dict[str, bool]], dict[str, str]]:
    """Build the ``{wrapper: {set_type: bool}}`` matrix and the alias map."""
    src = _load_dispatcher_source()
    alias_map = _build_import_alias_map(src)

    matrix: dict[str, dict[str, bool]] = {}

    # Initialise rows from n2v.nn.layers.__all__.
    from n2v.nn import layers as _layers
    wrappers = list(getattr(_layers, "__all__", []))
    for w in wrappers:
        matrix[w] = {t: False for t in _SET_TYPES}

    for t in _SET_TYPES:
        fn_src = _function_source(src, f"_reach_layer_{t}")
        if not fn_src:
            pytest.skip(
                f"could not locate _reach_layer_{t} in dispatcher source"
            )
        present = _branches_in(fn_src, alias_map)
        for w in wrappers:
            if w in present:
                matrix[w][t] = True

    return matrix, alias_map


def _format_matrix(matrix: dict[str, dict[str, bool]]) -> str:
    """Render an ASCII coverage table sorted by wrapper name."""
    col_w = max(len(w) for w in matrix) + 2
    header_cells = [t.capitalize() for t in _SET_TYPES]
    lines = []
    lines.append(
        "Wrapper".ljust(col_w) + " | " + " ".join(
            c.center(5) for c in header_cells
        )
    )
    lines.append("-" * (col_w + 3 + 6 * len(header_cells)))
    for w in sorted(matrix):
        row = matrix[w]
        cells = []
        for t in _SET_TYPES:
            cells.append(("OK" if row[t] else "--").center(5))
        lines.append(w.ljust(col_w) + " | " + " ".join(cells))
    return "\n".join(lines)


def test_every_leaf_wrapper_has_at_least_one_dispatcher_branch_audit_M2(
    tmp_path,
):
    """PR-1 audit M2 / completeness critic missed-category #2.

    Catches the I5/I7-class regression: a wrapper enters ``__all__``
    (and therefore the N2VTracer leaf list) but the dispatcher has
    zero branches for it. Every set-type call then raises
    ``NotImplementedError`` via ``_registry_lookup``.

    Asserts the floor: at least one set type per wrapper. Partial
    coverage (e.g. Box only) is flagged via a printed warning but does
    not fail -- some wrappers are deliberately Box-only per nnVLA
    catalog.
    """
    matrix, _ = _coverage_matrix()
    table = _format_matrix(matrix)
    # Save the matrix as a CI artefact in the pytest temp dir (Copilot
    # review: writing into the source tree leaves untracked files and
    # breaks hermetic/read-only test environments). CI can collect it
    # from the printed path below.
    out_path = tmp_path / "coverage_matrix.txt"
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\ncoverage matrix written to: {out_path}")

    missing_completely: list[str] = []
    partial: list[tuple[str, list[str]]] = []
    for w, row in matrix.items():
        if w in _TRACER_EXCLUDED or w in _VIA_PARENT_CLASS:
            continue
        covered = [t for t in _SET_TYPES if row[t]]
        if not covered:
            missing_completely.append(w)
        elif w not in _BOX_ONLY and set(covered) != set(_SET_TYPES):
            uncovered = [t for t in _SET_TYPES if not row[t]]
            partial.append((w, uncovered))

    # Print the table so ``pytest -s`` and CI logs surface it.
    print("\nDispatcher coverage matrix:\n" + table)
    if partial:
        print("\nPartial coverage (intended for Box-only wrappers per "
              "nnVLA catalog; otherwise consider adding):")
        for w, uncovered in partial:
            print(f"  {w}: missing {uncovered}")

    assert not missing_completely, (
        f"PR-1 audit M2: the following wrappers are in "
        f"``n2v.nn.layers.__all__`` (and therefore N2VTracer leaves) "
        f"but have NO dispatcher branch for ANY set type. Every reach "
        f"call routes through ``_registry_lookup`` -> ``None`` -> "
        f"``NotImplementedError``. Either add a dispatcher branch "
        f"(see I5 / I7), add the class to the N2VTracer "
        f"``_EXCLUDED`` set so fx decomposes it into primitives, or "
        f"register it in ``_VIA_PARENT_CLASS`` if it dispatches via a "
        f"stdlib base class.\n"
        f"  Missing: {missing_completely}"
    )


def test_via_parent_class_claims_match_actual_subclass_audit_M2():
    """Audit M2 sub-check: every wrapper claimed in ``_VIA_PARENT_CLASS``
    must actually be a subclass of the claimed base. Without this,
    someone could whitelist a wrapper that doesn't actually dispatch
    via a stdlib base, re-introducing the regression we are guarding
    against.
    """
    import torch.nn  # noqa: F401 -- needed for getattr by name
    from n2v.nn import layers as _layers

    for wrapper_name, base_path in _VIA_PARENT_CLASS.items():
        wrapper_cls = getattr(_layers, wrapper_name, None)
        assert wrapper_cls is not None, (
            f"_VIA_PARENT_CLASS lists {wrapper_name!r} but it is not "
            f"exported from n2v.nn.layers."
        )
        module_path, _, attr = base_path.rpartition(".")
        module = __import__(module_path, fromlist=[attr])
        base_cls = getattr(module, attr)
        assert issubclass(wrapper_cls, base_cls), (
            f"_VIA_PARENT_CLASS claims {wrapper_name} dispatches via "
            f"{base_path}, but {wrapper_name} is NOT a subclass of "
            f"{base_cls}. Update the whitelist or fix the dispatch."
        )
