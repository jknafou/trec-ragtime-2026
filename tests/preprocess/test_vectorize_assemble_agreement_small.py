"""Vectorize and assemble have to cut the corpus the same way.

This is the second seam the split index build got self-consistently wrong on each side.
``test_vector_cell_seam_small.py`` covers the first, where the two halves resolved a
``Layout`` method that did not exist under two different spellings.

Here the two halves cut a part on two different orders. ``vectorize`` pins block, and
therefore part, membership to the row order of ``passages.parquet``, which is what makes its
read a slice. ``assemble.validate`` re-derived the same part from the
``(token_count, passage_id)`` order instead. The two coincide only where ``token_count`` is
monotone in row order, which held for every small fixture written until then and does not
hold on the corpus: ``token_count`` decreases 228,834 times in the row order of the 460,326
real Spanish passages. Every cell with more than one part, 23 for en, 20 for es, 21 for zh
and 14 for ru, failed ``validate``, returned to ``pending/`` and eventually poisoned into
``failed/``, where ``saturate.drive`` stops. The assemble suite could not see any of it.

So the assertions here span both halves and run end to end. The blocks are written by the
real ``VectorizeAdapter`` with real legs, fake clients and the real work queue, read by the
real ``AssembleAdapter``, and the id set each half believes a part holds is compared against
a third derivation taken straight from the parquet table. Neither half is checked against a
constant spelled out in this file.

The corpus fixture comes from ``test_index_small`` and the harness pieces from the two
halves' own suites, imported rather than rewritten. A second description of the corpus, or
of a leg writer, would be a second thing that can drift.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout
from ragtime.common import io as common_io
from ragtime.orchestration import saturate
from ragtime.preprocess import vectors as vectors_mod
from ragtime.preprocess.assemble import AssembleAdapter
from ragtime.preprocess.index import (
    DENSE_LEG,
    LEGS,
    default_legs,
    index_build_options,
    index_hash,
    leg_encode_hash,
)
from ragtime.preprocess.vectorize import VectorizeAdapter
from tests.preprocess.test_assemble_small import _fake_writers
from tests.preprocess.test_index_small import (
    _RECON,
    _cfg,
    _ctx,
    _fixture_docs,
    _tables,
    _write,
)

pytestmark = pytest.mark.small

#: The shared fixture has 16 passages per language. At 4 per block and 8 per part that is
#: 4 blocks in 2 parts of 2 whole blocks. The cell has to have more than one part: with a
#: single part, part 0 is the whole language under either order and the two look identical.
_BLOCK = 4
_PART = 8
_LANGS = ("en", "es", "ru", "zh")


# --------------------------------------------------------------------------- #
# Harness: the real encode half, then the real assemble half, over one corpus.
# --------------------------------------------------------------------------- #
class _Both:
    """One corpus, vectorised by the real encode half and read by the real assemble half."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _cfg(tmp_path)
        cfg.blocks["index_build"]["config"]["encode_block_passages"] = _BLOCK
        cfg.blocks["index_build"]["config"]["plaid_part_passages"] = _PART
        cfg.blocks["execution"] = {"vectorize_blocks_per_task": 1}
        self.cfg = cfg
        self.opts = index_build_options(cfg)
        self.layout: Layout = _write(tmp_path, _tables(_fixture_docs()), cfg)
        self.legs = tuple(default_legs())
        self.encode_hashes = {leg: leg_encode_hash(self.opts, leg) for leg in LEGS}

        self.vectorize = VectorizeAdapter(
            recon_hash=_RECON,
            pack_hash=None,
            encode_hashes=dict(self.encode_hashes),
            base=str(tmp_path),
            legs=self.legs,
        )
        self.assemble = AssembleAdapter(
            recon_hash=_RECON,
            pack_hash=None,
            idx_hash=index_hash(cfg),
            base=str(tmp_path),
        )
        self._drain_vectorize(tmp_path)
        _fake_writers(monkeypatch)  # no faiss, seismic or pylate: this is about ids
        self.assemble_ctx = self.assemble.bringup(cfg)
        self.receipts = {
            (spec.variant, spec.source_lang, spec.part): self._assemble_one(tmp_path, spec)
            for spec in self.assemble.shard_specs(cfg)
        }

    # -- the encode half, through the real work-queue -------------------------- #
    def _drain_vectorize(self, tmp_path: Path) -> None:
        wq = saturate.queue_for(self.cfg, self.vectorize, base=tmp_path)
        saturate.seed(self.cfg, self.vectorize, wq)
        ctx = _ctx(self.layout, self.cfg, self.legs)
        while True:
            shard = saturate.workqueue.claim(wq.pending, wq.running)
            if shard is None:
                break
            out = self.vectorize.work(ctx, shard)
            assert self.vectorize.validate(out), f"vectorize.validate failed: {shard.name}"
            saturate.workqueue.mark_done(shard, wq.done, "corpus")

    # -- the assemble half ----------------------------------------------------- #
    def _assemble_one(self, tmp_path: Path, spec: Any) -> Path:
        running = tmp_path / "assemble-wq" / "running"
        running.mkdir(parents=True, exist_ok=True)
        shard = running / spec.name
        shard.write_text(json.dumps(spec.payload()) + "\n", encoding="utf-8")
        return self.assemble.work(self.assemble_ctx, shard)

    # -- three independent answers to "what is in part p?" --------------------- #
    def table_ids(self, lang: str) -> list[str]:
        """The language's ``passage_id``s in table row order, derived from nothing else."""
        return [
            str(row["passage_id"])
            for row in common_io.iter_parquet(
                self.layout.final_passages_path(_RECON, None),
                columns=["lang", "passage_id"],
            )
            if row["lang"] == lang
        ]

    def token_order_ids(self, lang: str) -> list[str]:
        """The same language in the ``(token_count, passage_id)`` order, the wrong cut."""
        rows = [
            (int(row["token_count"] or 0), str(row["passage_id"]))
            for row in common_io.iter_parquet(
                self.layout.final_passages_path(_RECON, None),
                columns=["lang", "passage_id", "token_count"],
            )
            if row["lang"] == lang
        ]
        return [passage for _, passage in sorted(rows)]

    def vectorized_part_ids(
        self, leg: str, variant: str | None, lang: str, part: int
    ) -> set[str]:
        """The ids ``vectorize`` wrote for part ``p``, read off its own block id maps.

        Assembled the way ``stream_part`` assembles a part, over the contiguous run of whole
        blocks that :class:`~ragtime.preprocess.vectors.BlockPlan` gives it, so this is the
        encode half's answer rather than a restatement of the table.
        """
        plan = vectors_mod.BlockPlan(
            passages=len(self.table_ids(lang)), block_passages=_BLOCK, part_passages=_PART
        )
        cell = self.vectorize.cell_dir(self.layout, leg, variant, lang)
        ids: set[str] = set()
        for block in plan.blocks_of_part(part):
            rows = common_io.read_parquet(
                cell / vectors_mod.block_dir_name(block) / vectors_mod.idmap_filename()
            )
            ids |= {str(r["passage_id"]) for r in rows}
        return ids

    def expected_part_ids(self, lang: str, part: int) -> frozenset[str]:
        """The ids ``assemble`` expects for part ``p``, re-derived from the table its way."""
        return self.assemble.expected_ids_at(
            self.layout.final_passages_path(_RECON, None),
            lang,
            part=part,
            parts=2,
            part_size=_PART,
        )


@pytest.fixture
def both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Both:
    return _Both(tmp_path, monkeypatch)


# --------------------------------------------------------------------------- #
# The fixture has to be able to show the defect, or nothing below means anything.
# --------------------------------------------------------------------------- #
def test_the_shared_corpus_fixture_is_non_monotone_in_token_count(both: _Both) -> None:
    """At least one language's ``token_count`` decreases somewhere in row order.

    Without that the row order and the token order are the same sequence, every assertion in
    this file passes against either cut, and that is how the defect shipped.
    """
    non_monotone = []
    for lang in _LANGS:
        counts = [
            int(row["token_count"] or 0)
            for row in common_io.iter_parquet(
                both.layout.final_passages_path(_RECON, None),
                columns=["lang", "token_count"],
            )
            if row["lang"] == lang
        ]
        if any(b < a for a, b in pairwise(counts)):
            non_monotone.append(lang)
    assert non_monotone, "the fixture cannot distinguish row order from token order"
    # And for those languages the two orders really do cut part 0 differently.
    assert any(
        set(both.table_ids(lang)[:_PART]) != set(both.token_order_ids(lang)[:_PART])
        for lang in non_monotone
    )


# --------------------------------------------------------------------------- #
# The agreement: what one half wrote is what the other half expects.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("part", [0, 1])
@pytest.mark.parametrize(("variant", "lang"), [(None, "en"), ("omt", "es"), ("original", "zh")])
def test_the_part_vectorize_wrote_is_the_part_assemble_expects(
    both: _Both, variant: str | None, lang: str, part: int
) -> None:
    """Three independent derivations of one part's id set, all equal.

    The first is the blocks the encode half published, unioned over the part's block run.
    The second is the assemble half's own re-derivation through ``expected_ids_at``, which
    is what ``validate`` compares against. The third is a slice of the table's row order
    taken here from the parquet file by neither half.
    """
    table_slice = set(both.table_ids(lang)[part * _PART : (part + 1) * _PART])
    expected = both.expected_part_ids(lang, part)

    assert set(expected) == table_slice
    for leg in LEGS:
        assert both.vectorized_part_ids(leg, variant, lang, part) == table_slice, leg


def test_the_expected_part_is_not_the_token_order_slice(both: _Both) -> None:
    """Where the two orders disagree about part 0, the row order is the one that wins.

    A validator cutting on the token order would satisfy every other test in this file, since
    both halves would still be internally consistent against their own order, and fail only
    here. The languages are discovered rather than listed: which of the fixture's four are
    non-monotone is a property of the shared corpus fixture, and a hard-coded list would
    empty this test the day that fixture changed. The assertion that at least one language
    qualifies is what keeps it honest.
    """
    divergent = [
        lang
        for lang in _LANGS
        if set(both.token_order_ids(lang)[:_PART]) != set(both.table_ids(lang)[:_PART])
    ]
    assert divergent, "no language distinguishes the two orders: this test proves nothing"
    for lang in divergent:
        variant = None if lang == "en" else "original"
        row_slice = set(both.table_ids(lang)[:_PART])
        assert set(both.expected_part_ids(lang, 0)) == row_slice, lang
        assert both.vectorized_part_ids(DENSE_LEG, variant, lang, 0) == row_slice, lang


# --------------------------------------------------------------------------- #
# And so the chain completes: every part of every cell validates.
# --------------------------------------------------------------------------- #
def test_every_part_of_every_multi_part_cell_validates(both: _Both) -> None:
    """Over the whole plan, no shard returns to ``pending/``.

    This is what failed in the real run: an assemble stage that never completes, while
    `merge`'s own checks still pass, so the artefact tree looks unfinished rather than wrong.
    """
    specs = both.assemble.shard_specs(both.cfg)
    assert specs and all(spec.parts == 2 for spec in specs)
    for spec in specs:
        receipt = both.receipts[(spec.variant, spec.source_lang, spec.part)]
        assert both.assemble.validate(receipt) is True, spec.name


def test_merge_publishes_a_manifest_over_the_vectorized_corpus(both: _Both) -> None:
    """The stage's final check of coverage, disjointness and cross-rendering identity passes
    over parts cut by the other half rather than by a fixture."""
    manifest = both.assemble.merge(both.cfg, list(both.receipts.values()))
    assert manifest.exists()
