"""``Layout``'s index, store and sidecar accessors: the path algebra of a built index.

Pure path functions, no IO. The properties asserted here are the ones the stages that
use them rely on:

- Composite keying: the index node hangs under ``final/<recon12>/``, so a build-recipe
  edit resolves to a fresh index dir beside the same corpus, while a corpus edit
  resolves to a fresh ``final_dir`` with no index beneath it. Neither stale pairing is
  addressable at all, which is stronger than asserting against it.
- English-once as a path rule: ``index_lang_dir(..., variant, "en")`` returns the same
  path for every variant, so reuse needs no copy or symlink step that a later stage
  could forget, and it stays true part by part because the part axis lives inside that
  one collapsed path.
- The part level is a required, explicit argument: ``index_shard_dir`` takes a
  keyword-only ``part`` with no default, so a caller that has not thought about parts
  cannot silently address part 0 of a ten-part cell.
- The sidecar sits outside ``final_dir``: ``url`` and ``date`` carry no chunker, merge,
  translation or reconcile semantics, so the sidecar must not be re-keyed by them.
- The vector cell is the same algebra one node over: ``vectors_cell_dir`` hangs under
  the packing node, is keyed by the per-leg encode hash rather than ``index_hash``,
  collapses ``en`` to one ``_shared`` cell by the same rule, and stops at the cell. The
  ``block-NNNNN`` level belongs to ``preprocess.vectors`` just as ``plaid-00000``
  belongs to ``preprocess.index``; that the two halves of the split build agree on it
  is checked in ``tests/preprocess/test_vector_cell_seam_small.py``.
"""

from __future__ import annotations

import pytest

from ragtime.common import Layout
from ragtime.common.passage_store import RENDERINGS

pytestmark = pytest.mark.small

_FAMILY = "e2e"
_CHUNKER = "8fbe879560a4ab" + "0" * 50
_RECON = "f8f20fe2cf17ab" + "0" * 50
_INDEX = "1234567890ab" + "c" * 52
_PACK = "0123456789ab" + "d" * 52
_ENCODE = "6b9c6bdb667e" + "e" * 52


def _layout(tmp_path) -> Layout:
    return Layout(run_dir=tmp_path, base=tmp_path, family=_FAMILY, chunker_hash=_CHUNKER)


def test_index_dir_hangs_under_the_final_corpus_node(tmp_path) -> None:
    layout = _layout(tmp_path)
    assert layout.index_dir(_RECON, _INDEX) == (
        layout.final_dir(_RECON) / "index" / _INDEX[:12]
    )


def test_a_recipe_edit_is_a_fresh_index_beside_the_same_corpus(tmp_path) -> None:
    """Only the index level moves when the build recipe changes."""
    layout = _layout(tmp_path)
    a = layout.index_dir(_RECON, _INDEX)
    b = layout.index_dir(_RECON, "9" * 64)
    assert a != b
    assert a.parent == b.parent == layout.final_dir(_RECON) / "index"


def test_a_corpus_edit_is_a_fresh_final_dir_with_no_index_beneath_it(tmp_path) -> None:
    """A new ``recon12`` cannot resolve to the previous corpus's index."""
    layout = _layout(tmp_path)
    old = layout.index_dir(_RECON, _INDEX)
    new = layout.index_dir("a" * 64, _INDEX)
    assert new != old
    assert not str(new).startswith(str(layout.final_dir(_RECON)))
    assert layout.final_dir("a" * 64) in new.parents


def test_english_shard_dir_is_one_shared_path_for_every_variant(tmp_path) -> None:
    """The path scheme is itself the English-once reuse mechanism."""
    layout = _layout(tmp_path)
    paths = {
        str(layout.index_lang_dir(_RECON, _INDEX, variant, "en")) for variant in RENDERINGS
    }
    assert len(paths) == 1, paths
    shared = layout.index_lang_dir(_RECON, _INDEX, None, "en")
    assert str(shared) in paths
    assert shared == layout.index_dir(_RECON, _INDEX) / "_shared" / "en"
    # the part axis does not break it: every part of every variant is inside that one
    # shared path, never a sibling of it.
    for part in (0, 7):
        assert {
            str(layout.index_shard_dir(_RECON, _INDEX, variant, "en", part=part))
            for variant in (*RENDERINGS, None)
        } == {str(shared / f"part-{part:05d}")}


def test_non_english_shard_dirs_are_per_variant_and_need_a_variant(tmp_path) -> None:
    layout = _layout(tmp_path)
    dirs = {
        variant: layout.index_lang_dir(_RECON, _INDEX, variant, "zh") for variant in RENDERINGS
    }
    assert len(set(map(str, dirs.values()))) == len(RENDERINGS)
    for variant, path in dirs.items():
        assert path == layout.index_dir(_RECON, _INDEX) / variant / "zh"
    with pytest.raises(ValueError, match="needs a variant"):
        layout.index_lang_dir(_RECON, _INDEX, None, "zh")
    with pytest.raises(ValueError, match="needs a variant"):
        layout.index_shard_dir(_RECON, _INDEX, None, "zh", part=0)


def test_shard_dir_is_a_named_part_under_the_language_cell(tmp_path) -> None:
    """The part axis: zero-padded names, so on-disk order is part order."""
    from ragtime.common.layout import INDEX_PART_GLOB, index_part_name

    layout = _layout(tmp_path)
    cell = layout.index_lang_dir(_RECON, _INDEX, "omt", "zh")
    assert layout.index_shard_dir(_RECON, _INDEX, "omt", "zh", part=0) == cell / "part-00000"
    assert layout.index_shard_dir(_RECON, _INDEX, "omt", "zh", part=12) == cell / "part-00012"
    # every part, including a lone one, is named: a reader's glob and the writer's name
    # come from the same literal, so a differently spelled part is unconstructible
    assert index_part_name(0) == "part-00000"
    assert [index_part_name(i) for i in range(11)] == sorted(
        index_part_name(i) for i in range(11)
    )
    assert INDEX_PART_GLOB == "part-*"


def test_shard_dir_refuses_to_default_the_part(tmp_path) -> None:
    """``part`` is keyword-only with no default: a caller must say which part it means."""
    import inspect

    parameter = inspect.signature(Layout.index_shard_dir).parameters["part"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        _layout(tmp_path).index_shard_dir(_RECON, _INDEX, "omt", "zh")  # type: ignore[call-arg]


def test_manifest_and_passage_store_paths(tmp_path) -> None:
    layout = _layout(tmp_path)
    assert layout.index_manifest_path(_RECON, _INDEX) == (
        layout.index_dir(_RECON, _INDEX) / "manifest.json"
    )
    # The rendering store is derived from the four tables beside it, so it is keyed by
    # the same composite recon hash: a v1 store cannot be served over a v2 inventory.
    assert layout.passage_store_path(_RECON, None) == layout.final_dir(_RECON) / "passages.lmdb"
    assert layout.passage_store_path(_RECON, None) != layout.passage_store_path("b" * 64, None)


def test_the_vector_cell_hangs_under_the_packing_node(tmp_path) -> None:
    """The vectors of a packing live with that packing's passage table, one level in.

    A re-pack therefore resolves to a fresh node holding no vectors, since they describe
    passages that no longer exist, while an ``index_build`` edit that moves no encode key
    leaves them where they are. That is the reason encode and assemble are split.
    """
    layout = _layout(tmp_path)
    cell = layout.vectors_cell_dir(_RECON, _PACK, "dense", _ENCODE, "omt", "zh")
    assert cell == (
        layout.final_passages_path(_RECON, _PACK).parent
        / "vectors"
        / "dense"
        / _ENCODE[:12]
        / "omt"
        / "zh"
    )
    assert cell == layout.vectors_dir(_RECON, _PACK, "dense", _ENCODE) / "omt" / "zh"
    # A different packing is a disjoint tree; the legacy (pre-``packing``) node is a third.
    assert cell != layout.vectors_cell_dir(_RECON, "9" * 64, "dense", _ENCODE, "omt", "zh")
    legacy = layout.vectors_cell_dir(_RECON, None, "dense", _ENCODE, "omt", "zh")
    assert legacy != cell
    assert legacy.parents[4] == layout.final_dir(_RECON)


def test_the_vector_node_is_keyed_by_leg_and_by_the_per_leg_encode_hash(tmp_path) -> None:
    """Two legs give two nodes, and a moved encode recipe a third, never an overwrite."""
    layout = _layout(tmp_path)
    nodes = {
        str(layout.vectors_dir(_RECON, _PACK, leg, _ENCODE))
        for leg in ("dense", "sparse", "late_interaction")
    }
    assert len(nodes) == 3
    assert layout.vectors_dir(_RECON, _PACK, "dense", _ENCODE) != layout.vectors_dir(
        _RECON, _PACK, "dense", "f" * 64
    )


def test_english_vector_cells_collapse_to_one_shared_path(tmp_path) -> None:
    """One node over, English is encoded once by the same path scheme.

    The rule is ``index_lang_dir``'s, spelled the same way, so a reader who
    has understood one has understood the other.
    """
    layout = _layout(tmp_path)
    paths = {
        str(layout.vectors_cell_dir(_RECON, _PACK, "sparse", _ENCODE, variant, "en"))
        for variant in RENDERINGS
    }
    assert len(paths) == 1, paths
    shared = layout.vectors_cell_dir(_RECON, _PACK, "sparse", _ENCODE, None, "en")
    assert str(shared) in paths
    assert shared == layout.vectors_dir(_RECON, _PACK, "sparse", _ENCODE) / "_shared" / "en"
    # Non-English does not collapse, and has no variant-independent form at all.
    assert len({
        str(layout.vectors_cell_dir(_RECON, _PACK, "sparse", _ENCODE, variant, "es"))
        for variant in RENDERINGS
    }) == len(RENDERINGS)
    with pytest.raises(ValueError, match="needs a variant"):
        layout.vectors_cell_dir(_RECON, _PACK, "sparse", _ENCODE, None, "es")


def test_layout_owns_no_block_level_inside_the_vector_cell(tmp_path) -> None:
    """``block-00000`` belongs to ``preprocess.vectors``, as ``plaid-00000`` does to index.

    ``Layout`` owns every level down to the unit a stage fans over; the sub-division a
    stage alone can enumerate stays with that stage. Asserted so that no second,
    competing block-path scheme grows here.
    """
    from ragtime.preprocess.vectors import block_dir_name

    layout = _layout(tmp_path)
    assert not [name for name in dir(Layout) if "block" in name]
    cell = layout.vectors_cell_dir(_RECON, _PACK, "dense", _ENCODE, "omt", "zh")
    assert (cell / block_dir_name(0)).parent == cell
