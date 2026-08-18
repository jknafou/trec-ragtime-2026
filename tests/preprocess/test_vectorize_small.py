"""The vectorize half of the index build, with no cluster and no real checkpoints.

There are no qrels and no dev set, so nothing here measures retrieval quality. Every
assertion is a structural property of the artefact tree: block membership, within-block
order, cross-rendering identity, resume, and the handover to assemble through
``preprocess.vectors``.

The corpus fixture is ``test_index_small``'s four-language ``_Doc``, ``_tables`` and
``_write``, imported rather than rebuilt, because the two stages read the same five final
tables and a second fixture would be a second description of the corpus that can drift. The
fake clients come from there for the same reason. The legs are the real ``FaissDenseLeg``,
``SeismicSparseLeg`` and ``PlaidLateInteractionLeg``: vectorize only ever calls
``Leg.encode_docs``, and the engine libraries are imported inside ``writer``, ``open`` and
``search``, which this stage never touches, so the real encode call sites run and each leg's
representation reaches its own ``VectorFormat``.

Nothing at the seam is stubbed. The cell path comes from the real
``ragtime.common.Layout.vectors_cell_dir``, the same method ``preprocess.assemble`` reads
these blocks back through. A suite that can supply its own path scheme is a suite that stays
green while the two halves of the seam disagree, which is how that defect shipped once
already. ``test_vector_cell_seam_small.py`` covers the two halves resolving one directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ragtime.common import Layout
from ragtime.common import io as common_io
from ragtime.common.passage_store import iter_final_passages
from ragtime.orchestration import saturate
from ragtime.preprocess import vectors as vectors_mod
from ragtime.preprocess.index import (
    DENSE_LEG,
    LATE_INTERACTION_LEG,
    LEGS,
    SPARSE_LEG,
    IndexIntegrityError,
    _assert_encode_device,
    default_legs,
    index_build_options,
    leg_encode_hash,
)
from ragtime.preprocess.vectorize import (
    CellBlockPlan,
    VectorBlockSpec,
    VectorizeAdapter,
    VectorizeTaskSpec,
    blocks_per_task,
    plan_cell_blocks,
    store_dtype,
)
from tests.preprocess.test_index_small import (
    _RECON,
    _cfg,
    _ctx,
    _Doc,
    _fixture_docs,
    _tables,
    _write,
)

pytestmark = pytest.mark.small


# --------------------------------------------------------------------------- #
# Build helper: seed the real work-queue, drain it, validate every task.
# --------------------------------------------------------------------------- #
class _Built:
    def __init__(self, cfg, layout, adapter, ctx, legs, wq, receipts, root) -> None:
        self.cfg = cfg
        self.layout = layout
        self.adapter = adapter
        self.ctx = ctx
        self.legs = legs
        self.wq = wq
        self.receipts = receipts
        self.root = root

    def cell(self, leg: str, variant: str | None, lang: str) -> Path:
        return self.adapter.cell_dir(self.layout, leg, variant, lang)

    def block_ids(self, leg: str, variant: str | None, lang: str, block: int) -> list[str]:
        rows = common_io.read_parquet(
            self.cell(leg, variant, lang)
            / vectors_mod.block_dir_name(block)
            / vectors_mod.idmap_filename()
        )
        return [str(r["passage_id"]) for r in rows]


def _adapter(cfg, tmp_path: Path, legs) -> VectorizeAdapter:
    opts = index_build_options(cfg)
    return VectorizeAdapter(
        recon_hash=_RECON,
        pack_hash=None,
        encode_hashes={leg: leg_encode_hash(opts, leg) for leg in LEGS},
        base=str(tmp_path),
        legs=tuple(legs),
    )


def _make_cfg(tmp_path: Path, *, block_passages: int, width: int = 1, langs=None):
    cfg = _cfg(tmp_path)
    cfg.blocks["index_build"]["config"]["encode_block_passages"] = int(block_passages)
    cfg.blocks["execution"] = {"vectorize_blocks_per_task": int(width)}
    if langs is not None:
        cfg.languages = tuple(langs)
    return cfg


def _build(
    tmp_path: Path,
    *,
    block_passages: int = 4,
    width: int = 1,
    docs=None,
    langs=None,
    legs=None,
) -> _Built:
    """Seed the real work queue, drain it through ``work``, and ``validate`` every task."""
    cfg = _make_cfg(tmp_path, block_passages=block_passages, width=width, langs=langs)
    layout = _write(tmp_path, _tables(docs or _fixture_docs()), cfg)
    legs = tuple(default_legs() if legs is None else legs)
    adapter = _adapter(cfg, tmp_path, legs)
    wq = saturate.queue_for(cfg, adapter, base=tmp_path)
    saturate.seed(cfg, adapter, wq)
    ctx = _ctx(layout, cfg, legs)
    receipts: list[Path] = []
    while True:
        shard = saturate.workqueue.claim(wq.pending, wq.running)
        if shard is None:
            break
        out = adapter.work(ctx, shard)
        assert adapter.validate(out), f"validate failed for {shard.name}"
        saturate.workqueue.mark_done(shard, wq.done, "corpus")
        receipts.append(out)
    return _Built(cfg, layout, adapter, ctx, legs, wq, receipts, tmp_path)


def _table_ids(layout: Layout, lang: str) -> list[str]:
    """The language's ``passage_id``s in table row order, which is what a block is cut on."""
    return [
        str(row["passage_id"])
        for row in common_io.iter_parquet(
            layout.final_passages_path(_RECON, None), columns=["lang", "passage_id"]
        )
        if row["lang"] == lang
    ]


def _tree_digest(root: Path) -> dict[str, str]:
    """``relative path -> sha256`` for every file under ``root``."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------- #
# V01, the plan: what a block is
# --------------------------------------------------------------------------- #
def test_block_membership_is_the_tables_row_order_slice(tmp_path: Path) -> None:
    """Block ``b`` is rows ``[b*B, (b+1)*B)`` of the language, in table row order."""
    build = _build(tmp_path, block_passages=4)
    for lang in ("en", "es", "ru", "zh"):
        rows = _table_ids(build.layout, lang)
        variant = None if lang == "en" else "original"
        for block in range(len(rows) // 4):
            assert build.block_ids(DENSE_LEG, variant, lang, block) != []
            assert set(build.block_ids(DENSE_LEG, variant, lang, block)) == set(
                rows[block * 4 : (block + 1) * 4]
            ), (lang, block)


def test_within_a_block_the_order_is_token_count_then_passage_id(tmp_path: Path) -> None:
    """Inside a block, the stored ordinal order is the ``(token_count, passage_id)`` order.

    ``Batcher`` sorts by length and its sort is stable, so bucketing an already-ordered list
    preserves that order. The ordinal order is therefore a function of two columns of the
    one text-free table, and so the same in every rendering.
    """
    build = _build(tmp_path, block_passages=4)
    keys = {
        r["passage_id"]: (int(r["token_count"] or 0), str(r["passage_id"]))
        for r in common_io.iter_parquet(
            build.layout.final_passages_path(_RECON, None),
            columns=["passage_id", "token_count"],
        )
    }
    for block in range(4):
        ids = build.block_ids(DENSE_LEG, "original", "ru", block)
        assert [keys[i] for i in ids] == sorted(keys[i] for i in ids)


def test_block_membership_is_identical_across_renderings(tmp_path: Path) -> None:
    """The property every cross-rendering comparison rests on, read off the artefact."""
    build = _build(tmp_path, block_passages=4)
    for lang in ("es", "ru", "zh"):
        for block in range(4):
            sets = {
                variant: frozenset(build.block_ids(DENSE_LEG, variant, lang, block))
                for variant in ("original", "omt", "omt_opus")
            }
            assert len(set(sets.values())) == 1, (lang, block, {k: len(v) for k, v in sets.items()})


def test_the_ranged_read_yields_exactly_the_unranged_streams_rows(tmp_path: Path) -> None:
    """The pushdown is a read optimisation: same rows, same order, same text.

    The whole cell's blocks, concatenated in block order, are compared against
    ``iter_final_passages`` with no ``document_range``. The range changes which rows reach
    the co-walk and never touches ``compose_passage_text``.
    """
    build = _build(tmp_path, block_passages=4)
    reference = [
        r["passage_id"]
        for r in iter_final_passages(build.layout, _RECON, pack_hash=None)
        if r["lang"] == "zh"
    ]
    got: list[str] = []
    for block in range(4):
        got.extend(build.block_ids(DENSE_LEG, "original", "zh", block))
    assert sorted(got) == sorted(reference)
    assert len(got) == len(set(got))


def test_a_document_straddling_a_block_boundary_is_split_not_duplicated(
    tmp_path: Path,
) -> None:
    """The boundary case the read plan is built around: three passages in one document,
    with a block of two.

    Both tasks read that document, because its row range is shared, so a defect here shows
    up as a duplicated or missing passage rather than as a crash.
    """
    docs = [
        _Doc("eng-docs/0000001", "en", ["A one.", "B two.", "C three."], " ", [(0,), (1,), (2,)]),
        _Doc("eng-docs/0000002", "en", ["D four.", "E five."], " ", [(0,), (1,)]),
    ]
    build = _build(tmp_path, block_passages=2, docs=docs, langs=("en",))
    rows = _table_ids(build.layout, "en")
    assert len(rows) == 5
    got = [build.block_ids(DENSE_LEG, None, "en", b) for b in range(3)]
    assert [len(x) for x in got] == [2, 2, 1]  # only the last block is short
    assert sorted(i for block in got for i in block) == sorted(rows)


def test_a_language_with_no_passages_still_publishes_one_empty_block(tmp_path: Path) -> None:
    """A cell that contributed no block at all would be invisible to every census."""
    docs = [_Doc("eng-docs/0000001", "en", ["A one.", "B two."], " ", [(0,), (1,)])]
    build = _build(tmp_path, block_passages=4, docs=docs, langs=("en", "es"))
    for variant in ("original", "omt", "omt_opus"):
        census = vectors_mod.read_cell_census(build.cell(DENSE_LEG, variant, "es"))
        assert (census.blocks, census.passages) == (1, 0)
    build.adapter.merge(build.cfg, build.receipts)


def test_plan_cell_blocks_records_the_document_row_each_block_starts_in(
    tmp_path: Path,
) -> None:
    """The seed-time plan is in document coordinates, which is what makes the read a slice."""
    docs = [
        _Doc("eng-docs/0000001", "en", ["A one.", "B two.", "C three."], " ", [(0,), (1,), (2,)]),
        _Doc("eng-docs/0000002", "en", ["D four.", "E five."], " ", [(0,), (1,)]),
    ]
    cfg = _make_cfg(tmp_path, block_passages=2, langs=("en",))
    layout = _write(tmp_path, _tables(docs), cfg)
    plans = plan_cell_blocks(layout, _RECON, None, block_passages=2, langs=("en",))
    plan = plans["en"]
    assert isinstance(plan, CellBlockPlan)
    assert (plan.passages, plan.blocks, plan.end_document) == (5, 3, 2)
    # Blocks 0 and 1 both begin inside document 0, which holds three passages, and the
    # second coordinate is the ordinal that document's first passage carries.
    assert plan.block_starts == ((0, 0), (0, 0), (1, 3))


# --------------------------------------------------------------------------- #
# V02: the task is a width, not a boundary
# --------------------------------------------------------------------------- #
def test_task_width_is_read_from_execution_and_defaults_to_one(tmp_path: Path) -> None:
    assert blocks_per_task(_make_cfg(tmp_path, block_passages=4, width=7)) == 7
    cfg = _make_cfg(tmp_path, block_passages=4)
    cfg.blocks["execution"] = {}
    assert blocks_per_task(cfg) == 1


def test_two_task_widths_produce_byte_identical_trees(tmp_path: Path) -> None:
    """Widening the claim re-fans the work. It moves no boundary and no byte."""
    narrow = _build(tmp_path / "narrow", block_passages=4, width=1)
    wide = _build(tmp_path / "wide", block_passages=4, width=3)
    root_n = narrow.layout.final_dir(_RECON) / "vectors"
    root_w = wide.layout.final_dir(_RECON) / "vectors"
    assert _tree_digest(root_n) == _tree_digest(root_w)
    assert len(narrow.receipts) > len(wide.receipts)  # fewer, wider claims


def test_a_wider_rerun_over_a_narrow_tree_resumes_every_block(tmp_path: Path) -> None:
    """The unit of ``_SUCCESS`` is the block, so two widths resume each other."""
    first = _build(tmp_path, block_passages=4, width=1)
    root = first.layout.final_dir(_RECON) / "vectors"
    before = _tree_digest(root)
    cfg = _make_cfg(tmp_path, block_passages=4, width=4)
    adapter = _adapter(cfg, tmp_path, first.legs)
    wq = saturate.queue_for(cfg, adapter, base=tmp_path)
    # A fresh queue for the same stage key: the artefact tree is the checkpoint, not the queue.
    for path in wq.done.glob("*"):
        path.unlink()
    saturate.seed(cfg, adapter, wq)
    # The client records every forward pass, so nothing being re-encoded is observed at the
    # model call rather than inferred from the mtimes the first assertion already covers.
    encoded_before = len(first.ctx.dense.calls)
    while (shard := saturate.workqueue.claim(wq.pending, wq.running)) is not None:
        out = adapter.work(first.ctx, shard)
        assert adapter.validate(out)
        saturate.workqueue.mark_done(shard, wq.done, "corpus")
    assert _tree_digest(root) == before  # not one byte, not one mtime
    assert len(first.ctx.dense.calls) == encoded_before


# --------------------------------------------------------------------------- #
# V03: specs and their strictness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("missing", ["block", "blocks"])
def test_block_payload_without_the_block_axis_fails_loudly(missing: str) -> None:
    payload = VectorBlockSpec(variant="omt", source_lang="es", block=3, blocks=9).payload()
    payload.pop(missing)
    with pytest.raises(IndexIntegrityError, match="block axis"):
        VectorBlockSpec.from_payload(payload)


def test_block_payload_round_trips(tmp_path: Path) -> None:
    spec = VectorBlockSpec(variant=None, source_lang="en", block=7, blocks=23)
    assert VectorBlockSpec.from_payload(spec.payload()) == spec
    assert spec.name == "_shared_en_b00007"
    assert spec.text_rendering == "original"


@pytest.mark.parametrize("missing", ["cell_passages", "block_passages", "start_ordinal"])
def test_task_payload_without_its_ordinal_frame_fails_loudly(missing: str) -> None:
    task = VectorizeTaskSpec(
        blocks=(VectorBlockSpec("omt", "es", 0, 2), VectorBlockSpec("omt", "es", 1, 2)),
        cell_passages=9,
        block_passages=8,
        start_ordinal=0,
        ranges=(),
    )
    payload = task.payload()
    payload.pop(missing)
    with pytest.raises(IndexIntegrityError):
        VectorizeTaskSpec.from_payload(payload)


def test_a_task_covers_one_cell_and_consecutive_blocks() -> None:
    with pytest.raises(IndexIntegrityError, match="covers one cell"):
        VectorizeTaskSpec(
            blocks=(VectorBlockSpec("omt", "es", 0, 2), VectorBlockSpec("omt", "ru", 1, 2)),
            cell_passages=9,
            block_passages=8,
            start_ordinal=0,
            ranges=(),
        )
    with pytest.raises(IndexIntegrityError, match="consecutive blocks"):
        VectorizeTaskSpec(
            blocks=(VectorBlockSpec("omt", "es", 0, 4), VectorBlockSpec("omt", "es", 2, 4)),
            cell_passages=30,
            block_passages=8,
            start_ordinal=0,
            ranges=(),
        )


# --------------------------------------------------------------------------- #
# V04: the leg set and the cell key
# --------------------------------------------------------------------------- #
def test_a_leg_set_that_is_not_LEGS_is_unconstructible(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, block_passages=4)
    opts = index_build_options(cfg)
    with pytest.raises(ValueError, match="exactly these legs"):
        VectorizeAdapter(
            recon_hash=_RECON,
            pack_hash=None,
            encode_hashes={leg: leg_encode_hash(opts, leg) for leg in LEGS},
            legs=tuple(default_legs()[:2]),
        )


def test_an_adapter_without_every_legs_encode_hash_is_unconstructible() -> None:
    with pytest.raises(ValueError, match="encode hash"):
        VectorizeAdapter(recon_hash=_RECON, pack_hash=None, encode_hashes={DENSE_LEG: "x"})


def test_an_assemble_only_change_moves_no_cell_and_an_encode_change_moves_one(
    tmp_path: Path,
) -> None:
    """The artefact key is where the split lives: ``plaid_nbits`` re-encodes nothing."""
    base = _make_cfg(tmp_path, block_passages=4)
    assemble_only = _make_cfg(tmp_path, block_passages=4)
    assemble_only.blocks["index_build"]["config"]["plaid_nbits"] = 4
    encode_change = _make_cfg(tmp_path, block_passages=4)
    encode_change.blocks["index_build"]["config"]["sparse_max_length"] = 1024

    def cells(cfg) -> dict[str, str]:
        layout = Layout(run_dir=tmp_path, base=tmp_path, family="f", chunker_hash="c" * 64)
        adapter = _adapter(cfg, tmp_path, default_legs())
        return {leg: str(adapter.cell_dir(layout, leg, "omt", "es")) for leg in LEGS}

    assert cells(assemble_only) == cells(base)
    changed = cells(encode_change)
    assert changed[SPARSE_LEG] != cells(base)[SPARSE_LEG]
    assert changed[DENSE_LEG] == cells(base)[DENSE_LEG]
    assert changed[LATE_INTERACTION_LEG] == cells(base)[LATE_INTERACTION_LEG]


def test_english_is_one_shared_cell_for_every_rendering(tmp_path: Path) -> None:
    """The path scheme itself is what builds English once: no copy, no per-rendering rebuild."""
    build = _build(tmp_path, block_passages=4)
    shared = build.cell(DENSE_LEG, None, "en")
    assert "_shared" in shared.parts
    for variant in ("original", "omt", "omt_opus"):
        assert build.cell(DENSE_LEG, variant, "en") == shared
    # One English cell exists on disk. ``_shared/en`` holds the blocks and no rendering
    # directory beside it holds an ``en`` cell at all, so the reuse needs no copy, no
    # symlink and no query-time fallback that a later stage could forget.
    leg_root = shared.parent.parent
    assert sorted(p.name for p in (leg_root / "_shared").iterdir() if p.is_dir()) == ["en"]
    for variant in ("original", "omt", "omt_opus"):
        assert not (leg_root / variant / "en").exists()
    blocks = [
        p for p in (leg_root / "_shared" / "en").glob(vectors_mod.BLOCK_GLOB) if p.is_dir()
    ]
    assert len(blocks) == 4


def test_the_store_dtype_comes_from_the_hashed_recipe(tmp_path: Path) -> None:
    opts = index_build_options(_make_cfg(tmp_path, block_passages=4))
    assert store_dtype(opts, DENSE_LEG) == "float32"
    assert store_dtype(opts, LATE_INTERACTION_LEG) == "float32"
    assert store_dtype(opts, SPARSE_LEG) == "float32"


# --------------------------------------------------------------------------- #
# V05: validate
# --------------------------------------------------------------------------- #
def _tamper_idmap(block_dir: Path, ids: list[str]) -> None:
    rows = common_io.read_parquet(block_dir / vectors_mod.idmap_filename())
    for row, passage_id in zip(rows, ids, strict=False):
        row["passage_id"] = passage_id
    common_io.write_parquet_stream(
        block_dir / vectors_mod.idmap_filename(),
        rows,
        schema=vectors_mod.idmap_schema(),
        skip_if_done=False,
    )


def test_validate_rejects_an_idmap_that_is_not_the_tables_block(tmp_path: Path) -> None:
    build = _build(tmp_path, block_passages=4)
    receipt = next(p for p in build.receipts if "_shared_en_b00000" in p.name)
    report = json.loads(receipt.read_text(encoding="utf-8").strip())
    block_dir = Path(report["cells"][DENSE_LEG]) / vectors_mod.block_dir_name(0)
    _tamper_idmap(block_dir, ["eng-docs/0007421#p9"])
    assert build.adapter.validate(receipt) is False


def test_validate_rejects_a_task_seeded_under_a_stale_block_count(tmp_path: Path) -> None:
    """A prefix-indexing plan passes every per-block check. Only the implied count sees it."""
    build = _build(tmp_path, block_passages=4)
    receipt = build.receipts[0]
    report = json.loads(receipt.read_text(encoding="utf-8").strip())
    report["blocks"] = 99
    common_io.write_jsonl(receipt, [report], skip_if_done=False)
    assert build.adapter.validate(receipt) is False


def test_validate_rejects_an_unpublished_leg(tmp_path: Path) -> None:
    build = _build(tmp_path, block_passages=4)
    receipt = build.receipts[0]
    report = json.loads(receipt.read_text(encoding="utf-8").strip())
    block = int(report["built"][0]["block"])
    marker = common_io.success_marker(
        Path(report["cells"][SPARSE_LEG]) / vectors_mod.block_dir_name(block)
    )
    marker.unlink()
    assert build.adapter.validate(receipt) is False


# --------------------------------------------------------------------------- #
# V06: merge
# --------------------------------------------------------------------------- #
def test_merge_accepts_a_complete_build(tmp_path: Path) -> None:
    build = _build(tmp_path, block_passages=4)
    build.adapter.merge(build.cfg, build.receipts)  # not raising is the assertion


def test_merge_raises_when_a_rendering_holds_different_ids_for_one_block(
    tmp_path: Path,
) -> None:
    build = _build(tmp_path, block_passages=4)
    receipt = next(p for p in build.receipts if "_omt_es_b00000" in p.name)
    report = json.loads(receipt.read_text(encoding="utf-8").strip())
    report["built"][0]["id_digest"] = "0" * 64
    common_io.write_jsonl(receipt, [report], skip_if_done=False)
    with pytest.raises(IndexIntegrityError, match="block membership differs"):
        build.adapter.merge(build.cfg, build.receipts)


def test_merge_raises_when_a_cell_lost_a_block(tmp_path: Path) -> None:
    build = _build(tmp_path, block_passages=4)
    receipts = [p for p in build.receipts if "_omt_es_b00003" not in p.name]
    with pytest.raises(IndexIntegrityError, match="the corpus implies"):
        build.adapter.merge(build.cfg, receipts)


def test_merge_raises_when_a_published_cell_is_short_on_disk(tmp_path: Path) -> None:
    """The census is read off the blocks' own witnesses, not off the task reports."""
    build = _build(tmp_path, block_passages=4)
    victim = build.cell(DENSE_LEG, "omt", "es") / vectors_mod.block_dir_name(3)
    common_io.success_marker(victim).unlink()
    with pytest.raises(vectors_mod.VectorIntegrityError):
        build.adapter.merge(build.cfg, build.receipts)


# --------------------------------------------------------------------------- #
# V07: the handover from vectorize to assemble
# --------------------------------------------------------------------------- #
def test_a_published_cell_streams_as_one_assemble_part(tmp_path: Path) -> None:
    """What ``assemble`` does: census, then ``stream_part``, then a gapless concatenation.

    It also covers the no-text property one level down: the only per-passage table a block
    writes is ``preprocess.index``'s id-map schema verbatim, which has no text column.
    """
    build = _build(tmp_path, block_passages=4)
    cell = build.cell(DENSE_LEG, "original", "zh")
    census = vectors_mod.read_cell_census(cell)
    assert (census.blocks, census.passages, census.leg) == (4, 16, DENSE_LEG)
    assert census.encode_hash == build.adapter.encode_hashes[DENSE_LEG]
    assert census.dtype == "float32"
    streamed: list[str] = []
    cursor = 0
    for block in vectors_mod.stream_part(cell, part=0, part_passages=16):
        assert block.ordinals.start == cursor
        cursor = block.ordinals.stop
        streamed.extend(ref.passage_id for ref in block.refs)
        assert len(block.reps) == len(block.refs)
    assert cursor == 16
    assert sorted(streamed) == sorted(_table_ids(build.layout, "zh"))
    names = {f.name for f in (cell / vectors_mod.block_dir_name(0)).iterdir() if f.is_file()}
    assert "vectors.npy" in names and vectors_mod.idmap_filename() in names
    assert [f.name for f in vectors_mod.idmap_schema()] == [
        "ordinal",
        "passage_id",
        "document_id",
        "source_lang",
    ]


def test_blocks_must_divide_the_assemble_part_evenly(tmp_path: Path) -> None:
    """Caught at seed time rather than at the first assemble, which is 82 GPU-hours of
    encoding later."""
    cfg = _make_cfg(tmp_path, block_passages=3)
    cfg.blocks["index_build"]["config"]["plaid_part_passages"] = 8
    _write(tmp_path, _tables(_fixture_docs()), cfg)
    adapter = _adapter(cfg, tmp_path, default_legs())
    with pytest.raises(vectors_mod.BlockGeometryError, match="whole number of blocks"):
        list(adapter.shards(cfg))


def test_the_shard_plan_is_three_renderings_plus_one_shared_english(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, block_passages=4)
    _write(tmp_path, _tables(_fixture_docs()), cfg)
    adapter = _adapter(cfg, tmp_path, default_legs())
    shards = list(adapter.shards(cfg))
    assert len({s.name for s in shards}) == len(shards)
    # Three renderings by three non-English languages by four blocks, plus four shared
    # English blocks.
    assert len(shards) == 3 * 3 * 4 + 4
    variants = {VectorizeTaskSpec.from_payload(s.payload).variant for s in shards}
    assert variants == {None, "original", "omt", "omt_opus"}


def test_the_stage_key_moves_with_the_encode_recipe_only(tmp_path: Path) -> None:
    base = _adapter(_make_cfg(tmp_path, block_passages=4), tmp_path, default_legs())
    assemble_only = _make_cfg(tmp_path, block_passages=4)
    assemble_only.blocks["index_build"]["config"]["plaid_nbits"] = 4
    encode_change = _make_cfg(tmp_path, block_passages=8)
    assert _adapter(assemble_only, tmp_path, default_legs()).stage == base.stage
    assert _adapter(encode_change, tmp_path, default_legs()).stage != base.stage
    assert base.template == "workqueue_worker.sbatch"
    assert isinstance(base, saturate.StageAdapter)


def test_every_block_carries_exactly_the_three_legs(tmp_path: Path) -> None:
    build = _build(tmp_path, block_passages=4)
    for report_path in build.receipts:
        report = json.loads(report_path.read_text(encoding="utf-8").strip())
        assert sorted(report["cells"]) == sorted(LEGS)
        for entry in report["built"]:
            assert sorted(entry["legs"]) == sorted(LEGS)
            for leg, path in entry["legs"].items():
                assert common_io.is_done(Path(path)), (leg, path)


def test_counters_are_emitted_for_every_leg_and_language(tmp_path: Path) -> None:
    build = _build(tmp_path, block_passages=4)
    ids = {record.metric_id for record in build.adapter.stats.records()}
    assert {"vectorize.blocks", "vectorize.passages", "vectorize.encoded_passages"} <= ids


# --------------------------------------------------------------------------- #
# The device the recipe declares is the device this worker has.
#
# `assemble_device` had a bring-up check and `encode_device` did not, while the encoders
# resolve theirs as `"cuda" if torch.cuda.is_available() else "cpu"`. A mis-scheduled
# vectorize worker therefore did not fail: it encoded on CPU and published the blocks under
# a leg_encode_hash whose body says "cuda". Nothing downstream can see that, because the
# census, the id sets and the manifest are all satisfied by CPU-encoded vectors. The chain
# runs two templates, vectorize on the GPU one and assemble on the CPU one, so a mis-wired
# stage-to-template mapping lands exactly here.
# --------------------------------------------------------------------------- #
def test_a_declared_cuda_encode_on_a_worker_without_cuda_fails_at_bringup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check sits in ``bringup``, so it fires before a client is built or a shard claimed."""
    from ragtime.config import ConfigError

    cfg = _make_cfg(tmp_path, block_passages=4)
    # The shipped declaration and the default, not an exotic configuration.
    assert index_build_options(cfg).encode_device == "cuda"
    adapter = _adapter(cfg, tmp_path, default_legs())

    try:
        import torch

        available = bool(torch.cuda.is_available())
    except ImportError:  # the torch-free chunk environment, itself a failure on this path
        available = False

    if available:  # pragma: no cover - depends on the node having a card
        assert _assert_encode_device(index_build_options(cfg)) == "cuda"
    else:
        # The message names the launch as well as the fault. `RAGTIME_GPU_MODEL=none` is
        # the assemble template's own signature, which turns "why is this node CPU?" into
        # a question the message has already answered.
        monkeypatch.setenv("RAGTIME_GPU_MODEL", "none")
        monkeypatch.setenv("SLURMD_NODENAME", "cpu046")
        with pytest.raises(ConfigError, match="encode_device") as excinfo:
            adapter.bringup(cfg)
        message = str(excinfo.value)
        assert "gpu=none" in message and "node=cpu046" in message
        assert "workqueue_worker" in message  # names the template to schedule under


def test_a_declared_cpu_encode_needs_no_gpu_and_no_torch(tmp_path: Path) -> None:
    """The guard fires only on the declared device, and a cpu run does not need torch."""
    cfg = _make_cfg(tmp_path, block_passages=4)
    cfg.blocks["index_build"]["config"]["encode_device"] = "cpu"
    assert _assert_encode_device(index_build_options(cfg)) == "cpu"
