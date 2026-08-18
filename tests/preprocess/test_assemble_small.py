"""Assemble: stored vector blocks into FAISS, Seismic and PLAID, without an encoder.

Four properties make the split build safe, and this file tries to break each of them.

The read is sequential and the id map is the concatenation. One part's blocks reach the
writer in ascending order, once each, with gapless part-local ordinals, and the published
``idmap.parquet`` is the concatenation of those blocks' id maps. That is also the part slice
of the passage table's row order, re-derived from the table, which is what
``preprocess.vectorize`` cuts on. The ``(token_count, passage_id)`` order is the within-block
order only, and this fixture's ``token_count`` is non-monotone in row order so the two orders
can never drift back into agreeing: see :func:`_token_count`.

Provenance comes from the hashes, not from a client. A cell whose blocks record a different
``encode_hash``, leg, language or rendering, or whose geometry implies a different part plan,
is refused before anything is written, and no encoder is ever built.

No text table is opened. The whole stage runs to a published manifest with
``documents.parquet``, ``sentences.parquet`` and both ``translations/*.parquet`` replaced by
unreadable bytes, and the module names no accessor that could reach them.

Every integrity check the fused index adapter had survives the move: exactly ``LEGS``,
per-part id-set equality, part disjointness, the on-disk census, cross-rendering part
membership, and the manifest with its per-part ``worker`` field.

The structural tests drive the real adapter with a recording ``LegWriter`` stand-in, so no
engine library is imported. One test drives the real writers end to end where faiss, seismic
and pylate are installed, since the reason for importing them is not to re-implement their
correctness here.
"""

from __future__ import annotations

import ast
import json
import shutil
import types
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragtime.common import Layout
from ragtime.common import io as common_io
from ragtime.common.passage_store import RENDERINGS
from ragtime.common.schemas import final_passage_arrow_schema
from ragtime.config import all_hashes
from ragtime.orchestration.run_identity import run_family
from ragtime.preprocess import assemble as asm
from ragtime.preprocess import vectors as vec
from ragtime.preprocess.assemble import AssembleAdapter
from ragtime.preprocess.index import (
    DENSE_LEG,
    IDMAP_FILENAME,
    LATE_INTERACTION_LEG,
    LEGS,
    SPARSE_LEG,
    IndexIntegrityError,
    IndexShardSpec,
    index_build_options,
    leg_encode_hash,
)

pytestmark = pytest.mark.small

_RECON = "f8f20fe2cf17" + "0" * 52
_IDX = "a" * 64
#: The two hashed geometry knobs, small but real. 24 passages per language gives 2 assemble
#: parts, 16 and a short 8, over 3 encode blocks of 8, so a part is a contiguous run of whole
#: blocks: part 0 is blocks 0 and 1, part 1 is block 2. The sequential read rests on that.
#: 16 is also the size the index suite found a real PLAID codec and a real Seismic build to
#: need, see its ``_PASSAGES_PER_LANG``, so part 0 exercises both.
_PART = 16
_BLOCK = 8
_PER_LANG = 24
_LANGS = ("en", "es")
_DIM = 8
_LI_VECTORS = 3  # token vectors per passage on the late-interaction leg


# --------------------------------------------------------------------------- #
# Fixture: a text-free passage table + one vector cell per (leg, variant, lang).
# --------------------------------------------------------------------------- #
def _cfg() -> Any:
    """A minimal cfg carrying what the adapter reads. No packing block is needed."""
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
                    "plaid_part_passages": _PART,
                    "encode_block_passages": _BLOCK,
                    "assemble_device": "cpu",
                }
            },
            "execution": {},
        },
    )


def _token_count(i: int) -> int:
    """A token count that is non-monotone in table row order.

    ``7`` is coprime with :data:`_PER_LANG`, so row ``i``'s count walks a full permutation of
    ``1..24`` and every prefix of the row order is a different set from the same-length prefix
    of the ``(token_count, passage_id)`` order.

    A monotone fixture -- ``i + 1``, the token count rising with row position -- makes the two
    orders the same sequence and hides ``expected_ids_at`` validating a row-order artefact
    against the token order. On the real corpus ``token_count`` decreases 228,834 times in row
    order over 460,326 Spanish passages, so a multi-part cell that confuses the two fails
    ``validate`` and returns to ``pending/``. Do not make this monotone.
    """
    return 1 + (i * 7) % _PER_LANG


def _passage_rows() -> list[dict[str, Any]]:
    """The one text-free table, in row order, which is the order the build cuts on.

    ``token_count`` is non-monotone in that order, see :func:`_token_count`, so a part's
    row-order slice and its ``(token_count, passage_id)`` slice are different sets and the
    two halves of the build cannot agree here by accident.

    ``sentence_ids`` name sentences that exist in no table, so a stage that tried to compose
    text here could not. That makes the no-text property structural rather than asserted.
    """
    rows: list[dict[str, Any]] = []
    for lang in _LANGS:
        for i in range(_PER_LANG):
            doc = f"{lang}-docs/{i:04d}"
            rows.append(
                {
                    "passage_id": f"{doc}#p0",
                    "document_id": doc,
                    "lang": lang,
                    "sentence_ids": [f"{doc}#s0"],
                    "token_count": _token_count(i),
                    "is_oversized": False,
                    "paragraph_index": [0],
                }
            )
    return rows


def _pinned(lang: str) -> list[tuple[str, str]]:
    """``[(passage_id, document_id)]`` of one language in the table's row order.

    Block ``b``, and therefore part ``p``, which is a whole run of blocks, is a slice of this
    list rather than of the token order.
    """
    return [
        (r["passage_id"], r["document_id"]) for r in _passage_rows() if r["lang"] == lang
    ]


def _token_order(lang: str) -> list[str]:
    """One language's ``passage_id``s in the ``(token_count, passage_id)`` order.

    It is here so the tests can assert that it differs from the row order. It is the cut the
    fused ``IndexAdapter`` made, and the wrong answer for this stage.
    """
    return [
        r["passage_id"]
        for r in sorted(
            (r for r in _passage_rows() if r["lang"] == lang),
            key=lambda r: (int(r["token_count"]), str(r["passage_id"])),
        )
    ]


def _block_refs(lang: str, block: int) -> list[tuple[str, str]]:
    """Block ``block``'s refs as ``vectorize`` writes them: a row-order slice, then sorted.

    Membership is the row-order slice ``[b*B, (b+1)*B)`` and the order within it is
    ``(token_count, passage_id)``, the batch-composition key. Both matter: the id set is what
    ``validate`` re-derives, and the sequence is what the published ``idmap.parquet`` has to
    be the concatenation of.
    """
    counts = {r["passage_id"]: int(r["token_count"]) for r in _passage_rows()}
    window = _pinned(lang)[block * _BLOCK : (block + 1) * _BLOCK]
    return sorted(window, key=lambda ref: (counts[ref[0]], ref[0]))


def _part_refs(lang: str, part: int) -> list[tuple[str, str]]:
    """Part ``part``'s refs: the concatenation of its blocks', in block order."""
    per_part = _PART // _BLOCK
    return [
        ref
        for block in range(part * per_part, (part + 1) * per_part)
        for ref in _block_refs(lang, block)
    ]


def _reps(leg: str, lo: int, n: int) -> list[Any]:
    """Deterministic per-leg reps. Dense is unit-norm because ``_FaissWriter`` asserts it."""
    if leg == DENSE_LEG:
        out = []
        for i in range(n):
            raw = np.asarray([((lo + i + j) % 7) + 1 for j in range(_DIM)], dtype="float32")
            out.append((raw / np.linalg.norm(raw)).tolist())
        return out
    if leg == SPARSE_LEG:
        return [{3: 0.25, 7 + lo + i: 0.5} for i in range(n)]
    return [
        np.arange(
            (lo + i) * 100, (lo + i) * 100 + _LI_VECTORS * _DIM, dtype="float32"
        ).reshape(_LI_VECTORS, _DIM)
        / 1000.0
        for i in range(n)
    ]


def _cell_dir(layout: Layout, opts: Any, leg: str, variant: str | None, lang: str) -> Path:
    """Where ``(leg, variant, lang)``'s blocks live, through the real ``Layout`` method.

    A fixture that installed its own path scheme when ``Layout`` lacked one used to stand in
    here, and it let this suite pass while the encode half wrote its blocks somewhere else
    entirely. The fixture writer and the adapter now resolve the same method with the same
    arguments, so the only way for them to disagree is for ``Layout`` itself to be wrong,
    which ``test_vector_cell_seam_small.py`` checks against the other half.
    """
    return layout.vectors_cell_dir(
        _RECON, None, leg, leg_encode_hash(opts, leg), variant, lang
    )


def _cells(cfg: Any) -> list[tuple[str | None, str]]:
    """``(variant, source_lang)`` cells: English once, and every rendering otherwise."""
    out: list[tuple[str | None, str]] = [(None, "en")]
    out.extend((variant, "es") for variant in RENDERINGS)
    return out


def _write_fixture(tmp_path: Path, cfg: Any) -> Layout:
    """The passage table and every leg's vector blocks, at the paths the adapter resolves."""
    layout = Layout(
        run_dir=tmp_path,
        base=tmp_path,
        family=run_family(cfg),
        chunker_hash=all_hashes(cfg)["chunker"],
    )
    common_io.write_parquet_stream(
        layout.final_passages_path(_RECON, None),
        _passage_rows(),
        schema=final_passage_arrow_schema(),
    )
    opts = index_build_options(cfg)
    for leg in LEGS:
        for variant, lang in _cells(cfg):
            _write_cell(layout, opts, leg, variant, lang)
    return layout


def _write_cell(
    layout: Layout,
    opts: Any,
    leg: str,
    variant: str | None,
    lang: str,
    *,
    block_encode_hash: str | None = None,
    block_lang: str | None = None,
    block_variant: str | None = "<unset>",
    passages: int | None = None,
) -> Path:
    """Write one cell's blocks. The keyword overrides exist to forge a mismatched cell.

    What they do not move is the cell's path, which is always the real ``leg_encode_hash``
    one, so a forged witness sits exactly where the shard looks. That is the only way to
    exercise the hash check: a forged path would just be an absent cell.
    """
    n = _PER_LANG if passages is None else passages
    cell = _cell_dir(layout, opts, leg, variant, lang)
    plan = vec.BlockPlan(passages=n, block_passages=_BLOCK, part_passages=_PART)
    counts = {r["passage_id"]: int(r["token_count"]) for r in _passage_rows()}
    rows = _pinned(lang)[:n]
    for block in range(plan.blocks):
        lo, hi = plan.block_bounds(block)
        # Exactly as ``vectorize`` writes a block: membership is the row-order slice
        # ``plan.block_bounds`` gives it, and the order within it is
        # ``(token_count, passage_id)``. The two differ in this fixture, because the counts
        # are non-monotone in row order, so a fixture writing the token-order slice instead
        # would be writing a different block, which is the confusion under test.
        refs = sorted(rows[lo:hi], key=lambda ref: (counts[ref[0]], ref[0]))
        vec.write_block(
            cell,
            block=block,
            plan=plan,
            fmt=vec.format_for(leg),
            encode_hash=block_encode_hash or leg_encode_hash(opts, leg),
            lang=lang if block_lang is None else block_lang,
            variant=variant if block_variant == "<unset>" else block_variant,
            refs=[vec.PassageRef(p, d) for p, d in refs],
            reps=_reps(leg, lo, hi - lo),
        )
    return cell


# --------------------------------------------------------------------------- #
# A recording LegWriter stand-in, so no engine library is imported.
# --------------------------------------------------------------------------- #
_LOG: list[tuple[str, str, list[int]]] = []


class _FakeWriter:
    """The ``LegWriter`` contract, ``add`` and ``finish``, plus a call log."""

    def __init__(self, out_dir: Path, opts: Any) -> None:
        self.out_dir = Path(out_dir)
        self.opts = opts
        # ``.<leg>.tmp-<pid>`` is the private temp directory the adapter publishes from.
        self.leg = self.out_dir.name.lstrip(".").split(".tmp-")[0]
        self.rows: dict[int, Any] = {}

    def add(self, ordinals: Any, reps: Any) -> None:
        _LOG.append((self.leg, "add", [int(o) for o in ordinals]))
        for ordinal, rep in zip(ordinals, reps, strict=True):
            self.rows[int(ordinal)] = rep

    def finish(self) -> None:
        _LOG.append((self.leg, "finish", sorted(self.rows)))
        (self.out_dir / "engine.json").write_text(
            json.dumps(sorted(self.rows)), encoding="utf-8"
        )


@pytest.fixture(autouse=True)
def _clear_log() -> None:
    _LOG.clear()


def _fake_writers(monkeypatch: pytest.MonkeyPatch) -> None:
    for leg in LEGS:
        monkeypatch.setitem(asm._WRITER_OF, leg, _FakeWriter)


def _adapter(tmp_path: Path) -> AssembleAdapter:
    return AssembleAdapter(
        recon_hash=_RECON, pack_hash=None, idx_hash=_IDX, base=str(tmp_path)
    )


def _run_one(
    adapter: AssembleAdapter, ctx: Any, wq_dir: Path, spec: IndexShardSpec
) -> Path:
    """Write the shard payload where ``saturate`` would, then call ``work`` directly."""
    running = wq_dir / "running"
    running.mkdir(parents=True, exist_ok=True)
    shard = running / spec.name
    shard.write_text(json.dumps(spec.payload()) + "\n", encoding="utf-8")
    return adapter.work(ctx, shard)


def _run_all(adapter: AssembleAdapter, cfg: Any, tmp_path: Path) -> list[Path]:
    ctx = adapter.bringup(cfg)
    wq = tmp_path / "wq"
    return [_run_one(adapter, ctx, wq, spec) for spec in adapter.shard_specs(cfg)]


# --------------------------------------------------------------------------- #
# The plan: the unit is the PLAID part, and English is planned once.
# --------------------------------------------------------------------------- #
def test_the_assemble_unit_is_the_plaid_part(tmp_path: Path) -> None:
    cfg = _cfg()
    _write_fixture(tmp_path, cfg)
    adapter = _adapter(tmp_path)

    assert adapter.part_passages(cfg) == _PART
    assert adapter.parts_for(cfg, "es") == 2  # 24 passages over parts of 16
    specs = adapter.shard_specs(cfg)
    # Three renderings with two Spanish parts each, plus two shared English parts.
    assert len(specs) == len(RENDERINGS) * 2 + 2
    shared = [s for s in specs if s.source_lang == "en"]
    assert [s.variant for s in shared] == [None, None]
    assert {s.parts for s in specs} == {2}


def test_shard_payloads_round_trip_through_the_index_shard_spec(tmp_path: Path) -> None:
    cfg = _cfg()
    _write_fixture(tmp_path, cfg)
    adapter = _adapter(tmp_path)
    for shard in adapter.shards(cfg):
        spec = IndexShardSpec.from_payload(shard.payload)
        assert spec.name == shard.name


# --------------------------------------------------------------------------- #
# The read is sequential, and the id map is the concatenation.
# --------------------------------------------------------------------------- #
def test_a_part_is_read_as_a_gapless_ascending_run_of_whole_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg()
    _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    ctx = adapter.bringup(cfg)
    spec = IndexShardSpec(variant="omt", source_lang="es", part=0, parts=2)
    _run_one(adapter, ctx, tmp_path / "wq", spec)

    for leg in LEGS:
        adds = [ordinals for name, kind, ordinals in _LOG if name == leg and kind == "add"]
        # One add per block of the part, ascending and gapless. Part 0 is two blocks of 8.
        assert adds == [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], leg


def test_part_idmap_is_the_concatenation_and_equals_the_table_slice(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    ctx = adapter.bringup(cfg)

    for part, expected in ((0, _part_refs("es", 0)), (1, _part_refs("es", 1))):
        spec = IndexShardSpec(variant="omt", source_lang="es", part=part, parts=2)
        _run_one(adapter, ctx, tmp_path / "wq", spec)
        shard_dir = layout.index_shard_dir(_RECON, _IDX, "omt", "es", part=part)
        for leg in LEGS:
            rows = common_io.read_parquet(shard_dir / leg / IDMAP_FILENAME)
            assert [r["ordinal"] for r in rows] == list(range(len(expected)))
            assert [(r["passage_id"], r["document_id"]) for r in rows] == expected
            assert {r["source_lang"] for r in rows} == {"es"}
        # And it is the same slice validation re-derives from the passage table.
        assert adapter.expected_ids_at(
            layout.final_passages_path(_RECON, None),
            "es",
            part=part,
            parts=2,
            part_size=_PART,
        ) == {p for p, _ in expected}


def test_the_fixture_token_counts_are_non_monotone_in_row_order() -> None:
    """The property the fixture needs, asserted so it cannot be refactored away.

    With a monotone ``token_count`` the table's row order and its
    ``(token_count, passage_id)`` order are the same sequence and a validator cutting either
    one passes, which is how ``expected_ids_at`` shipped cutting the wrong one.
    """
    for lang in _LANGS:
        rows = [r for r in _passage_rows() if r["lang"] == lang]
        counts = [int(r["token_count"]) for r in rows]
        assert any(b < a for a, b in pairwise(counts)), lang
        # And the two orders really are different sequences, not the same one re-sorted.
        assert [r["passage_id"] for r in rows] != _token_order(lang)


@pytest.mark.parametrize("part", [0, 1])
def test_expected_part_ids_are_the_row_order_slice_not_the_token_order(
    tmp_path: Path, part: int
) -> None:
    """``validate``'s expectation is the cut ``vectorize`` made: rows, not token keys.

    The two are different sets here, as they are on the real corpus, where ``token_count``
    decreases 228,834 times in the row order of the 460,326 Spanish passages. So this fails
    against a validator that slices the token order, which is the defect it exists for.
    """
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    adapter = _adapter(tmp_path)

    expected = adapter.expected_ids_at(
        layout.final_passages_path(_RECON, None), "es", part=part, parts=2, part_size=_PART
    )

    row_order = {p for p, _ in _pinned("es")[part * _PART : (part + 1) * _PART]}
    token_order = set(_token_order("es")[part * _PART : (part + 1) * _PART])
    assert expected == row_order
    assert expected != token_order


def test_validate_accepts_every_part_of_a_multi_part_cell(tmp_path: Path, monkeypatch) -> None:
    """Every part of a cell with more than one part validates.

    This used to return False for both parts, because the id sets it compared were two
    different slices of two different orders. The shard went back to ``pending/`` and after
    ``k_max`` attempts the cell landed in ``failed/``, where ``saturate.drive`` stops. Every
    real cell has more than one part, 23 for en, 20 for es, 21 for zh and 14 for ru, so this
    is the whole assemble stage rather than an edge case.
    """
    cfg = _cfg()
    _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    ctx = adapter.bringup(cfg)

    for spec in adapter.shard_specs(cfg):
        assert spec.parts == 2, spec.name  # a multi-part cell, or this test proves nothing
        assert adapter.validate(_run_one(adapter, ctx, tmp_path / "wq", spec)) is True


def test_resume_republishes_nothing_and_counts_the_skip(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg()
    _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    ctx = adapter.bringup(cfg)
    spec = IndexShardSpec(variant="omt", source_lang="es", part=0, parts=2)
    _run_one(adapter, ctx, tmp_path / "wq", spec)
    _LOG.clear()

    out = _run_one(adapter, ctx, tmp_path / "wq", spec)

    assert _LOG == []  # nothing re-read, nothing re-written
    assert adapter.validate(out) is True
    assert adapter.stats.total("index.leg_skipped_resume") == float(len(LEGS))


# --------------------------------------------------------------------------- #
# Provenance comes from the hashes, and no encoder exists to compare against.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"block_encode_hash": "f" * 64}, "encode_hash"),
        ({"block_lang": "ru"}, "lang"),
        ({"block_variant": "omt_opus"}, "variant"),
    ],
)
def test_a_cell_that_is_not_this_shards_is_refused(
    tmp_path: Path, monkeypatch, override: dict[str, Any], message: str
) -> None:
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    opts = index_build_options(cfg)
    # Re-write the dense cell of (omt, es) with one forged field.
    cell = _cell_dir(layout, opts, DENSE_LEG, "omt", "es")
    shutil.rmtree(cell)
    _write_cell(layout, opts, DENSE_LEG, "omt", "es", **override)

    adapter = _adapter(tmp_path)
    ctx = adapter.bringup(cfg)
    spec = IndexShardSpec(variant="omt", source_lang="es", part=0, parts=2)
    with pytest.raises(IndexIntegrityError, match=message):
        _run_one(adapter, ctx, tmp_path / "wq", spec)


def test_a_cell_vectorized_under_a_different_plan_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """A cell holding fewer passages implies fewer parts, and assembling it anyway would
    publish a part that looks whole over a prefix of the language."""
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    opts = index_build_options(cfg)
    for leg in LEGS:
        cell = _cell_dir(layout, opts, leg, "omt", "es")
        shutil.rmtree(cell)
        _write_cell(layout, opts, leg, "omt", "es", passages=8)

    adapter = _adapter(tmp_path)
    ctx = adapter.bringup(cfg)
    spec = IndexShardSpec(variant="omt", source_lang="es", part=0, parts=2)
    with pytest.raises(IndexIntegrityError, match="stale plan"):
        _run_one(adapter, ctx, tmp_path / "wq", spec)


def test_no_encoder_is_ever_built(tmp_path: Path, monkeypatch) -> None:
    """``build_clients`` is patched to raise, and a whole part still assembles."""
    from ragtime.serving import registry

    def _boom(*_a, **_kw):  # pragma: no cover - reached only if the stage regresses
        raise AssertionError("assemble must never build a model client")

    monkeypatch.setattr(registry, "build_clients", _boom)
    cfg = _cfg()
    _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    ctx = adapter.bringup(cfg)
    out = _run_one(
        adapter,
        ctx,
        tmp_path / "wq",
        IndexShardSpec(variant="omt", source_lang="es", part=0, parts=2),
    )
    assert adapter.validate(out) is True
    # The context carries no client field at all, so this is a property of the type.
    assert not {f for f in AssembleAdapter.__dataclass_fields__} & {"dense", "milco", "mtd"}
    assert not hasattr(ctx, "dense")


def test_the_module_never_names_a_model_or_a_text_table() -> None:
    """The accessors that could reach text, or a client, do not appear in the source.

    The test below runs the whole stage against unreadable text tables. This is the static
    counterpart, so an edit that adds a text read fails here with the reason attached rather
    than only wherever the corrupt bytes happen to be opened.
    """
    tree = ast.parse(Path(asm.__file__).read_text(encoding="utf-8"))
    # Every identifier the module uses: names, attributes and imports. This walks the AST
    # rather than scanning for substrings, because the docstrings discuss these accessors,
    # and a scan that could not tell prose from code would force those explanations out of
    # the file to keep itself passing.
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    used |= {
        part
        for node in ast.walk(tree)
        if isinstance(node, ast.alias)
        for part in (node.name.split(".") + [node.asname or ""])
    }
    used |= {
        node.module.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    for forbidden in (
        "documents_path",
        "final_sentences_path",
        "final_translations_path",
        "iter_final_passages",
        "compose_passage_text",
        "PassageStore",
        "build_clients",
        "encode_docs",
        "encode_query",
    ):
        assert forbidden not in used, forbidden


# --------------------------------------------------------------------------- #
# No text table is opened, shown by making them unreadable.
# --------------------------------------------------------------------------- #
def _poison_text_tables(layout: Layout) -> list[Path]:
    """Replace every text table with bytes no reader can open, and mark them published."""
    paths = [
        layout.documents_path(),
        layout.final_sentences_path(_RECON),
        *(layout.final_translations_path(_RECON, v) for v in RENDERINGS if v != "original"),
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"NOT A PARQUET FILE - opening this must never happen\n")
        common_io.success_marker(path).write_bytes(b"")
    return paths


def test_the_whole_stage_runs_with_every_text_table_unreadable(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    poisoned = _poison_text_tables(layout)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)

    outs = _run_all(adapter, cfg, tmp_path)
    assert all(adapter.validate(out) for out in outs)
    manifest = adapter.merge(cfg, outs)

    assert manifest.exists()
    # The poisoned files are untouched: nothing rewrote or repaired them either.
    for path in poisoned:
        assert path.read_bytes().startswith(b"NOT A PARQUET FILE")


# --------------------------------------------------------------------------- #
# validate: the checks the index adapter had, kept.
# --------------------------------------------------------------------------- #
def _one_part(tmp_path: Path, monkeypatch) -> tuple[AssembleAdapter, Layout, Path, Path]:
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    ctx = adapter.bringup(cfg)
    out = _run_one(
        adapter,
        ctx,
        tmp_path / "wq",
        IndexShardSpec(variant="omt", source_lang="es", part=0, parts=2),
    )
    shard_dir = layout.index_shard_dir(_RECON, _IDX, "omt", "es", part=0)
    return adapter, layout, out, shard_dir


def test_validate_accepts_a_complete_part(tmp_path: Path, monkeypatch) -> None:
    adapter, _layout, out, _shard = _one_part(tmp_path, monkeypatch)
    assert adapter.validate(out) is True


def test_validate_rejects_a_missing_part_census(tmp_path: Path, monkeypatch) -> None:
    adapter, _layout, out, shard_dir = _one_part(tmp_path, monkeypatch)
    (shard_dir / "part.json").unlink()
    assert adapter.validate(out) is False


def test_validate_rejects_an_unpublished_leg(tmp_path: Path, monkeypatch) -> None:
    adapter, _layout, out, shard_dir = _one_part(tmp_path, monkeypatch)
    common_io.success_marker(shard_dir / LATE_INTERACTION_LEG).unlink()
    assert adapter.validate(out) is False


def test_validate_rejects_a_shifted_part_boundary(tmp_path: Path, monkeypatch) -> None:
    """The id set is re-derived from the table, so a leg holding the wrong slice fails even
    when its own id map is internally consistent."""
    adapter, _layout, out, shard_dir = _one_part(tmp_path, monkeypatch)
    rows = common_io.read_parquet(shard_dir / DENSE_LEG / IDMAP_FILENAME)
    rows[0]["passage_id"] = _pinned("es")[20][0]  # a passage that belongs to part 1
    from ragtime.preprocess.index import idmap_arrow_schema

    common_io.write_parquet_stream(
        shard_dir / DENSE_LEG / IDMAP_FILENAME,
        rows,
        schema=idmap_arrow_schema(),
        skip_if_done=False,
    )
    assert adapter.validate(out) is False


# --------------------------------------------------------------------------- #
# merge: coverage, disjointness, the census, cross-rendering membership, the manifest.
# --------------------------------------------------------------------------- #
def test_merge_publishes_one_manifest_with_the_splits_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "4242424")
    adapter = _adapter(tmp_path)
    outs = _run_all(adapter, cfg, tmp_path)

    manifest = json.loads(adapter.merge(cfg, outs).read_text(encoding="utf-8").strip())

    assert manifest["legs"] == list(LEGS)
    assert manifest["index_hash"] == _IDX
    # The vector keys are the split's provenance record: one per leg, and distinct.
    opts = index_build_options(cfg)
    assert manifest["leg_encode_hash"] == {
        leg: leg_encode_hash(opts, leg) for leg in LEGS
    }
    assert len(set(manifest["leg_encode_hash"].values())) == len(LEGS)
    for variant in RENDERINGS:
        entry = manifest["variants"][variant]
        assert entry["passages"] == len(_LANGS) * _PER_LANG
        # English is one build, named by every rendering's manifest rather than rebuilt.
        assert entry["shards"]["en"]["path"] == str(
            layout.index_lang_dir(_RECON, _IDX, None, "en")
        )
        assert entry["shards"]["es"]["parts"] == 2
        assert entry["shards"]["es"]["part_passages"] == _PART
        # Per-part worker provenance survived the move.
        assert all(
            part["worker"].get("job_id") == "4242424"
            for part in entry["shards"]["es"]["shard_parts"]
        )
    # The per-leg config hash is the same across renderings by construction.
    assert manifest["leg_config_hash"] == {
        leg: manifest["variants"]["omt"]["shards"]["es"]["shard_parts"][0]["legs"][leg][
            "config_hash"
        ]
        for leg in LEGS
    }


def test_merge_refuses_a_missing_part(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg()
    _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    outs = _run_all(adapter, cfg, tmp_path)
    with pytest.raises(IndexIntegrityError, match="no assemble output"):
        adapter.merge(cfg, outs[:-1])


def test_merge_refuses_overlapping_parts(tmp_path: Path, monkeypatch) -> None:
    """Two parts covering the same id sum to more than their union. A union that looks
    complete on its own would not catch that."""
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    outs = _run_all(adapter, cfg, tmp_path)
    from ragtime.preprocess.index import idmap_arrow_schema

    part1 = layout.index_shard_dir(_RECON, _IDX, "omt", "es", part=1)
    for leg in LEGS:
        rows = common_io.read_parquet(part1 / leg / IDMAP_FILENAME)
        rows[0]["passage_id"] = _pinned("es")[0][0]  # already covered by part 0
        common_io.write_parquet_stream(
            part1 / leg / IDMAP_FILENAME,
            rows,
            schema=idmap_arrow_schema(),
            skip_if_done=False,
        )
    with pytest.raises(IndexIntegrityError, match="disjoint|covers"):
        adapter.merge(cfg, outs)


def test_merge_refuses_a_cell_that_lost_a_part_on_disk(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    outs = _run_all(adapter, cfg, tmp_path)
    shutil.rmtree(layout.index_shard_dir(_RECON, _IDX, "omt", "es", part=1))
    with pytest.raises(IndexIntegrityError):
        adapter.merge(cfg, outs)


def test_merge_proves_part_membership_is_identical_across_renderings(
    tmp_path: Path, monkeypatch
) -> None:
    """Swap one id between two parts of one rendering. The union, the counts and the
    disjointness all still hold, and only the per-part digest differs, which is the property
    every cross-rendering comparison rests on."""
    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    _fake_writers(monkeypatch)
    adapter = _adapter(tmp_path)
    outs = _run_all(adapter, cfg, tmp_path)
    from ragtime.preprocess.index import idmap_arrow_schema

    dirs = [layout.index_shard_dir(_RECON, _IDX, "omt", "es", part=p) for p in (0, 1)]
    for leg in LEGS:
        rows0 = common_io.read_parquet(dirs[0] / leg / IDMAP_FILENAME)
        rows1 = common_io.read_parquet(dirs[1] / leg / IDMAP_FILENAME)
        rows0[15]["passage_id"], rows1[0]["passage_id"] = (
            rows1[0]["passage_id"],
            rows0[15]["passage_id"],
        )
        for path, rows in ((dirs[0], rows0), (dirs[1], rows1)):
            common_io.write_parquet_stream(
                path / leg / IDMAP_FILENAME,
                rows,
                schema=idmap_arrow_schema(),
                skip_if_done=False,
            )
    with pytest.raises(IndexIntegrityError, match="part membership differs"):
        adapter.merge(cfg, outs)


def test_an_assembly_with_the_wrong_leg_set_is_unconstructible(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly these legs"):
        AssembleAdapter(
            recon_hash=_RECON,
            pack_hash=None,
            idx_hash=_IDX,
            base=str(tmp_path),
            legs=(DENSE_LEG, SPARSE_LEG),
        )


# --------------------------------------------------------------------------- #
# The sparse round trip is exact. It is the one leg whose writer re-derives its input.
# --------------------------------------------------------------------------- #
def test_sparse_reps_round_trip_through_the_writers_own_conversion() -> None:
    from ragtime.serving.sparse_milco import seismic_components

    weights = {3: 0.25, 17: 0.5, 100: 0.125}
    components, values = seismic_components(weights)
    (rebuilt,) = asm._reps_for(SPARSE_LEG, [(components, values)])
    assert seismic_components(rebuilt) == (components, values)
    # The other two legs are handed their stored form untouched.
    dense = np.zeros((2, 3), dtype="float32")
    assert asm._reps_for(DENSE_LEG, dense) is dense


# --------------------------------------------------------------------------- #
# The real writers, skipped where the engine libraries are not installed.
# --------------------------------------------------------------------------- #
def test_real_engines_assemble_one_part_and_reopen_it(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    pytest.importorskip("seismic")
    pytest.importorskip("pylate")
    from ragtime.preprocess.index import (
        FaissDenseLeg,
        PlaidLateInteractionLeg,
        SeismicSparseLeg,
        plaid_part_count,
    )

    cfg = _cfg()
    layout = _write_fixture(tmp_path, cfg)
    adapter = _adapter(tmp_path)
    ctx = adapter.bringup(cfg)
    spec = IndexShardSpec(variant="omt", source_lang="es", part=0, parts=2)
    out = _run_one(adapter, ctx, tmp_path / "wq", spec)

    assert adapter.validate(out) is True
    shard_dir = layout.index_shard_dir(_RECON, _IDX, "omt", "es", part=0)
    reader_ctx = types.SimpleNamespace(opts=ctx.opts)

    dense = FaissDenseLeg().open(shard_dir / DENSE_LEG, reader_ctx)
    assert dense.ntotal == _PART
    # Self-retrieval through the stored vectors, which shows that ordinal 0's vector is at
    # ordinal 0. There are no qrels, so it says nothing about retrieval quality.
    query = np.asarray(_reps(DENSE_LEG, 0, 1)[0], dtype="float32").reshape(1, -1)
    hits = FaissDenseLeg().search(dense, reader_ctx, query, 1)
    assert hits[0][0] == 0

    sparse = SeismicSparseLeg().open(shard_dir / SPARSE_LEG, reader_ctx)
    got = SeismicSparseLeg().search(sparse, reader_ctx, {3: 0.25, 7: 0.5}, 2)
    assert got and all(0 <= ordinal < _PART for ordinal, _ in got)

    # One PLAID part per assemble part, by construction: the part size bounds the buffer.
    assert plaid_part_count(shard_dir / LATE_INTERACTION_LEG) == 1
    parts = PlaidLateInteractionLeg().open(shard_dir / LATE_INTERACTION_LEG, reader_ctx)
    assert len(parts) == 1
    assert faiss is not None
