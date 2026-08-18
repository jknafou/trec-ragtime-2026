"""The vectorize to assemble seam: one cell path, resolved by one ``Layout`` method.

The defect this file exists for: the two halves of the split index build were written
against a ``Layout`` method that did not exist yet, and each half guessed a different name
and a different signature. The encode side wanted ``vectors_cell_dir(reconcile_hash,
encode_hash, leg, variant, source_lang)`` and the assemble side wanted
``vector_cell_dir(reconcile_hash, *, encode_hash, variant, source_lang)``. Both test suites
passed, because each installed its own stand-in for the missing method. Two halves that
agree separately and not with each other is what a per-module test cannot see.

So every assertion here imports both adapters and compares what they resolve, rather than
comparing either against a path spelled out in this file. The stand-ins are gone, checked
below by reading the modules' source, so nothing is left that could answer the question
locally.

This is path algebra only: no IO, no encoder, no engine.
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout
from ragtime.common.passage_store import RENDERINGS
from ragtime.preprocess import assemble as asm
from ragtime.preprocess import vectorize as vzr
from ragtime.preprocess import vectors as vec
from ragtime.preprocess.assemble import AssembleAdapter, AssembleCtx
from ragtime.preprocess.index import (
    DENSE_LEG,
    LEGS,
    IndexShardSpec,
    index_build_options,
    leg_encode_hash,
)
from ragtime.preprocess.vectorize import VectorizeAdapter

pytestmark = pytest.mark.small

_RECON = "f8f20fe2cf17" + "0" * 52
_PACK = "0123456789ab" + "c" * 52
_IDX = "a" * 64
_LANGS = ("en", "es")


def _cfg() -> Any:
    """The minimum both adapters read: the languages and the hashed ``index_build`` recipe."""
    return types.SimpleNamespace(
        run_id="e2e-original",
        languages=_LANGS,
        blocks={
            "chunker": {"config": {"tokenizer_id": "BAAI/bge-m3@abc", "token_budget": 512}},
            "index_build": {
                "config": {
                    "dense_model": "BAAI/bge-m3",
                    "sparse_model": "omai-research/milco-650m",
                    "spine_model": "hltcoe/plaidx-large-neuclir-mtd",
                    "plaid_nbits": 2,
                    "plaid_part_passages": 16,
                    "encode_block_passages": 8,
                    "assemble_device": "cpu",
                }
            },
            "execution": {},
        },
    )


def _layout(tmp_path: Path) -> Layout:
    return Layout(run_dir=tmp_path, base=tmp_path, family="e2e", chunker_hash="c" * 64)


class _Seam:
    """Both halves, built from one cfg, each resolving cells its own way."""

    def __init__(self, tmp_path: Path, pack_hash: str | None = _PACK) -> None:
        cfg = _cfg()
        self.cfg = cfg
        self.opts = index_build_options(cfg)
        self.layout = _layout(tmp_path)
        self.pack_hash = pack_hash
        self.encode_hashes = {leg: leg_encode_hash(self.opts, leg) for leg in LEGS}
        self.vectorize = VectorizeAdapter(
            recon_hash=_RECON,
            pack_hash=pack_hash,
            encode_hashes=dict(self.encode_hashes),
            base=str(tmp_path),
        )
        self.assemble = AssembleAdapter(
            recon_hash=_RECON, pack_hash=pack_hash, idx_hash=_IDX, base=str(tmp_path)
        )
        self.ctx = AssembleCtx(
            opts=self.opts,
            layout=self.layout,
            recon_hash=_RECON,
            pack_hash=pack_hash,
            idx_hash=_IDX,
            encode_hashes=dict(self.encode_hashes),
        )

    def by_encode(self, leg: str, variant: str | None, lang: str) -> Path:
        """The cell the encode half writes its blocks into."""
        return self.vectorize.cell_dir(self.layout, leg, variant, lang)

    def by_assemble(self, leg: str, variant: str | None, lang: str) -> Path:
        """The cell the assemble half streams those blocks out of."""
        return self.assemble.cell_dir(
            self.ctx, IndexShardSpec(variant=variant, source_lang=lang), leg
        )


_CELLS: tuple[tuple[str | None, str], ...] = (
    (None, "en"),
    *((variant, "en") for variant in RENDERINGS),
    *((variant, "es") for variant in RENDERINGS),
)


# --------------------------------------------------------------------------- #
# The one thing neither half could check alone.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leg", LEGS)
@pytest.mark.parametrize(("variant", "lang"), _CELLS)
def test_both_halves_resolve_the_same_cell(
    tmp_path: Path, leg: str, variant: str | None, lang: str
) -> None:
    """The cell written to is the cell read from, for every ``(leg, variant, lang)``."""
    seam = _Seam(tmp_path)
    assert seam.by_encode(leg, variant, lang) == seam.by_assemble(leg, variant, lang)


def test_the_two_halves_go_through_the_same_layout_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just equal paths but one resolver, so the two cannot drift apart later.

    Both adapters are routed through a single patched ``Layout.vectors_cell_dir``. A half
    that had kept its own path arithmetic would sail past the equality test above.
    """
    seam = _Seam(tmp_path)
    calls: list[tuple[Any, ...]] = []

    def _spy(self: Layout, *args: Any) -> Path:
        calls.append(args)
        return Path("/sentinel")

    monkeypatch.setattr(Layout, "vectors_cell_dir", _spy)
    assert seam.by_encode(DENSE_LEG, "omt", "es") == Path("/sentinel")
    assert seam.by_assemble(DENSE_LEG, "omt", "es") == Path("/sentinel")
    # And with the same arguments, in the same order.
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0] == (
        _RECON,
        _PACK,
        DENSE_LEG,
        leg_encode_hash(seam.opts, DENSE_LEG),
        "omt",
        "es",
    )


# --------------------------------------------------------------------------- #
# English is encoded once, and the block level lives inside that collapse.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leg", LEGS)
def test_english_collapses_to_one_shared_cell_in_both_halves(tmp_path: Path, leg: str) -> None:
    """Every rendering names the same English directory, in both halves.

    A per-rendering English path would either encode English three times, or let three
    renderings write three disagreeing copies of what is meant to be one tree. No census or
    id-set check downstream would notice either.
    """
    seam = _Seam(tmp_path)
    paths = {
        str(resolve(leg, variant, "en"))
        for resolve in (seam.by_encode, seam.by_assemble)
        for variant in (None, *RENDERINGS)
    }
    assert len(paths) == 1, paths
    shared = seam.by_encode(leg, None, "en")
    assert str(shared) in paths
    assert shared.parts[-2:] == ("_shared", "en")


@pytest.mark.parametrize("leg", LEGS)
def test_the_block_level_is_strictly_inside_the_english_collapse(
    tmp_path: Path, leg: str
) -> None:
    """``block-00000`` is a sub-division of the shared cell, not a sibling of it.

    ``Layout`` owns every level down to the cell and ``preprocess.vectors`` owns the block
    name, the same division as ``preprocess.index`` owning ``plaid-00000`` inside an index
    part. If the block level sat beside the collapse instead of inside it, English being
    encoded once would hold for the directory and fail for every file in it.
    """
    seam = _Seam(tmp_path)
    shared = seam.by_encode(leg, None, "en")
    blocks = {
        seam.by_encode(leg, variant, "en") / vec.block_dir_name(3)
        for variant in (None, *RENDERINGS)
    }
    assert blocks == {shared / vec.block_dir_name(3)}
    assert shared in next(iter(blocks)).parents


# --------------------------------------------------------------------------- #
# Non-English does not collapse: the renderings are three separate encodes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leg", LEGS)
def test_the_three_renderings_of_a_translated_language_are_three_cells(
    tmp_path: Path, leg: str
) -> None:
    seam = _Seam(tmp_path)
    for resolve in (seam.by_encode, seam.by_assemble):
        paths = {str(resolve(leg, variant, "es")) for variant in RENDERINGS}
        assert len(paths) == len(RENDERINGS), paths
    # A translated language has no variant-free cell. The shared unit is English and
    # nothing else, so ``None`` there is a defect rather than a default.
    with pytest.raises(ValueError, match="needs a variant"):
        seam.layout.vectors_cell_dir(_RECON, _PACK, leg, "e" * 64, None, "es")


# --------------------------------------------------------------------------- #
# The cell hangs under the packing node, so a re-pack owns none of these vectors.
# --------------------------------------------------------------------------- #
def test_the_cell_sits_under_the_pack_node_beside_the_passage_table(tmp_path: Path) -> None:
    """The vectors are the passages of one packing, encoded, so they live with them."""
    seam = _Seam(tmp_path)
    passages = seam.layout.final_passages_path(_RECON, _PACK)
    cell = seam.by_encode(DENSE_LEG, "omt", "es")
    assert passages.parent in cell.parents
    assert seam.layout.final_dir(_RECON) in cell.parents
    assert _PACK[:12] in cell.parts


@pytest.mark.parametrize("leg", LEGS)
def test_a_repack_resolves_to_a_disjoint_tree_in_both_halves(tmp_path: Path, leg: str) -> None:
    """A different ``pack12`` shares no directory with this one below the final node.

    That is the argument for splitting the build, stated as a path fact. A re-pack finds no
    vectors it could mistake for its own, while an assemble-only edit, which moves neither
    ``pack12`` nor any encode key, finds all of them.
    """
    a = _Seam(tmp_path)
    b = _Seam(tmp_path, pack_hash="9" * 64)
    for variant, lang in _CELLS:
        old = a.by_encode(leg, variant, lang)
        new = b.by_encode(leg, variant, lang)
        assert old != new
        assert new == b.by_assemble(leg, variant, lang)
        assert not str(new).startswith(str(old.parent))
    # The node from before the ``packing`` block is a third, equally disjoint tree.
    legacy = _Seam(tmp_path, pack_hash=None).by_encode(leg, "omt", "es")
    assert legacy != a.by_encode(leg, "omt", "es")
    # That node is ``final_dir``, because there was no ``passages/<pack12>`` level then, so
    # it carries no pack level that could collide with one.
    assert legacy.parents[2] == a.layout.final_dir(_RECON) / "vectors" / leg
    assert legacy.parents[4] == a.layout.final_dir(_RECON)


def test_an_assemble_only_recipe_change_leaves_every_cell_where_it_was(
    tmp_path: Path,
) -> None:
    """``plaid_nbits`` is an assemble key, so the vectors it re-reads stay where they are."""
    seam = _Seam(tmp_path)
    cfg = _cfg()
    cfg.blocks["index_build"]["config"]["plaid_nbits"] = 4
    moved = index_build_options(cfg)
    for leg in LEGS:
        assert leg_encode_hash(moved, leg) == seam.encode_hashes[leg]


# --------------------------------------------------------------------------- #
# The stand-ins that hid the defect are gone.
# --------------------------------------------------------------------------- #
def test_layout_exposes_exactly_one_vector_cell_accessor() -> None:
    """One name, one signature, and the assemble half's guessed spelling does not exist."""
    assert hasattr(Layout, "vectors_cell_dir")
    assert not hasattr(Layout, "vector_cell_dir")
    parameters = list(inspect.signature(Layout.vectors_cell_dir).parameters)
    assert parameters == [
        "self",
        "reconcile_hash",
        "pack_hash",
        "leg",
        "encode_hash",
        "variant",
        "source_lang",
    ]
    # The block level is not Layout's; ``preprocess.vectors.block_dir_name`` owns it, the
    # same division as ``plaid-00000`` inside an index part.
    assert not [name for name in dir(Layout) if "block" in name]


@pytest.mark.parametrize("module", [vzr, asm])
def test_neither_half_guards_the_seam_with_a_getattr_shim(module: Any) -> None:
    """No ``getattr(layout, ...)``, no fallback, no locally built vector path.

    Both halves used to reach the method through ``getattr`` with a loud error, which is
    what let each of them ship against a name the other did not use. Now that the method
    exists, a ``getattr`` could only hide the next such divergence.
    """
    source = Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")
    assert "getattr(layout" not in source
    assert "VECTORS_CELL_METHOD" not in source
    # No local re-definition of the seam under either spelling. The one route is a call on a
    # ``Layout`` instance, which is how every other artefact path in these modules resolves.
    assert "def vectors_cell_dir" not in source
    assert "def vector_cell_dir" not in source
    assert ".vectors_cell_dir(" in source
