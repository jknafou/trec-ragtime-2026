"""The on-disk vector block format, where vectorize hands over to assemble.

Three properties carry the format. What ``write_block`` wrote is what ``read_block_dir``
returns, per leg, including the empty and single-passage blocks a corpus-scale remainder
produces. A part is a sequential concatenation: ``stream_part`` yields a part's blocks in
order with gapless part-local ordinals, and the concatenation of their refs is the part's
slice of the cell. And the census never fans short: a missing witness, a missing
``_SUCCESS``, a witness in the wrong directory, a disagreeing plan, a lost block or a short
block that is not the last all raise, rather than returning a subset that looks whole.

The reps are hand-built and no engine is loaded, since the formats are files rather than
indexes. The real ``seismic_components`` conversion does run, because the sparse format
stores its output verbatim and that is the point of choosing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ragtime.common.io import is_done, read_parquet
from ragtime.preprocess import vectors as vec
from ragtime.preprocess.index import (
    DENSE_LEG,
    IDMAP_FILENAME,
    LATE_INTERACTION_LEG,
    SPARSE_LEG,
    idmap_arrow_schema,
)

pytestmark = pytest.mark.small

_LANG = "zh"
_VARIANT = "omt"
_HASH = "0123456789ab"


# --------------------------------------------------------------------------- #
# Fixture builders: deterministic reps, one shape per leg.
# --------------------------------------------------------------------------- #
def _refs(lo: int, n: int) -> list[vec.PassageRef]:
    out: list[vec.PassageRef] = []
    for i in range(n):
        doc = f"{_LANG}-docs/{(lo + i) // 3:07d}"
        out.append(vec.PassageRef(f"{doc}#p{(lo + i) % 3}", doc))
    return out


def _dense_reps(lo: int, n: int, dim: int = 4) -> list[list[float]]:
    return [[float(lo + i) + j / 8.0 for j in range(dim)] for i in range(n)]


def _sparse_reps(lo: int, n: int) -> list[dict[int, float]]:
    # Deliberately not in sorted key order. seismic_components sorts by the component
    # string, and the stored file carries that order, not the encoder's dict order.
    return [{7 + lo + i: 0.5, 3: 0.25, 100: 0.125} for i in range(n)]


def _li_reps(lo: int, n: int, dim: int = 3) -> list[np.ndarray]:
    # Ragged by construction, 1 to 3 vectors per passage. The cumulative-offset
    # reconstruction is the format, so a uniform fixture would test nothing.
    return [
        np.arange((lo + i) * 10, (lo + i) * 10 + (1 + i % 3) * dim, dtype="float32").reshape(
            1 + i % 3, dim
        )
        for i in range(n)
    ]


_REPS = {DENSE_LEG: _dense_reps, SPARSE_LEG: _sparse_reps, LATE_INTERACTION_LEG: _li_reps}


def _write(cell: Path, plan: vec.BlockPlan, leg: str, block: int, **kw) -> Path:
    lo, hi = plan.block_bounds(block)
    return vec.write_block(
        cell,
        block=block,
        plan=plan,
        fmt=kw.pop("fmt", None) or vec.format_for(leg),
        encode_hash=_HASH,
        lang=_LANG,
        variant=_VARIANT,
        refs=_refs(lo, hi - lo),
        reps=_REPS[leg](lo, hi - lo),
        **kw,
    )


def _assert_reps_equal(leg: str, got, want) -> None:
    if not want:  # an empty block: every leg returns an empty payload and claims no shape
        assert len(got) == 0
        return
    if leg == DENSE_LEG:
        np.testing.assert_allclose(np.asarray(got), np.asarray(want, dtype="float32"))
    elif leg == SPARSE_LEG:
        assert len(got) == len(want)
        for (components, values), weights in zip(got, want, strict=True):
            from ragtime.serving.sparse_milco import seismic_components

            exp_c, exp_v = seismic_components(weights)
            assert components == exp_c
            np.testing.assert_allclose(values, exp_v)
    else:
        assert len(got) == len(want)
        for a, b in zip(got, want, strict=True):
            np.testing.assert_allclose(np.asarray(a, dtype="float32"), b, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------- #
# The arithmetic: blocks divide parts evenly, or there is no plan.
# --------------------------------------------------------------------------- #
def test_plan_refuses_a_part_that_is_not_whole_blocks() -> None:
    with pytest.raises(vec.BlockGeometryError, match="whole number of blocks"):
        vec.BlockPlan(passages=100, block_passages=8, part_passages=12)


@pytest.mark.parametrize(
    ("block_passages", "part_passages"), [(0, 8), (8, 0), (-1, 8), (8, -8)]
)
def test_plan_refuses_non_positive_sizes(block_passages: int, part_passages: int) -> None:
    with pytest.raises(vec.BlockGeometryError):
        vec.BlockPlan(passages=10, block_passages=block_passages, part_passages=part_passages)


def test_plan_arithmetic_on_a_ragged_tail() -> None:
    # Three parts and a ragged remainder: 2 blocks per part, 5 blocks, the last one short.
    plan = vec.BlockPlan(passages=37, block_passages=8, part_passages=16)
    assert (plan.blocks, plan.parts, plan.blocks_per_part) == (5, 3, 2)
    assert [plan.block_bounds(b) for b in range(5)] == [
        (0, 8), (8, 16), (16, 24), (24, 32), (32, 37)
    ]
    assert [plan.block_passage_count(b) for b in range(5)] == [8, 8, 8, 8, 5]
    assert [plan.part_of_block(b) for b in range(5)] == [0, 0, 1, 1, 2]
    assert [list(plan.blocks_of_part(p)) for p in range(3)] == [[0, 1], [2, 3], [4]]
    # Part-local ordinals restart at 0 for every part and are gapless inside one.
    assert [plan.part_ordinals(b) for b in range(5)] == [
        range(8), range(8, 16), range(8), range(8, 16), range(5)
    ]
    assert [plan.part_passage_count(p) for p in range(3)] == [16, 16, 5]


def test_plan_range_checks_are_loud() -> None:
    plan = vec.BlockPlan(passages=10, block_passages=8, part_passages=8)
    with pytest.raises(vec.BlockGeometryError, match="outside this cell's plan"):
        plan.block_bounds(2)
    with pytest.raises(vec.BlockGeometryError, match="outside this cell's plan"):
        plan.blocks_of_part(2)


def test_an_empty_cell_still_has_one_block_and_one_part() -> None:
    plan = vec.BlockPlan(passages=0, block_passages=8, part_passages=8)
    assert (plan.blocks, plan.parts) == (1, 1)
    assert plan.block_bounds(0) == (0, 0)
    assert plan.part_ordinals(0) == range(0)


# --------------------------------------------------------------------------- #
# Round trip, per leg, including the degenerate blocks.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leg", [DENSE_LEG, SPARSE_LEG, LATE_INTERACTION_LEG])
@pytest.mark.parametrize("passages", [16, 1, 0])
def test_round_trip_per_leg(tmp_path: Path, leg: str, passages: int) -> None:
    plan = vec.BlockPlan(passages=passages, block_passages=8, part_passages=16)
    cell = tmp_path / "cell"
    for block in range(plan.blocks):
        _write(cell, plan, leg, block)
    census = vec.read_cell_census(cell)
    assert census.blocks == plan.blocks
    assert census.passages == passages
    assert (census.leg, census.lang, census.variant, census.encode_hash) == (
        leg, _LANG, _VARIANT, _HASH,
    )

    for block in range(plan.blocks):
        lo, hi = plan.block_bounds(block)
        got = vec.read_block_dir(cell / vec.block_dir_name(block))
        assert got.block == block
        assert got.ordinals == range(hi - lo)
        assert list(got.refs) == _refs(lo, hi - lo)
        _assert_reps_equal(leg, got.reps, _REPS[leg](lo, hi - lo))


@pytest.mark.parametrize("leg", [DENSE_LEG, SPARSE_LEG, LATE_INTERACTION_LEG])
def test_a_block_is_published_atomically_and_resume_is_a_no_op(tmp_path: Path, leg: str) -> None:
    plan = vec.BlockPlan(passages=8, block_passages=8, part_passages=8)
    cell = tmp_path / "cell"
    published = _write(cell, plan, leg, 0)
    assert is_done(published)
    assert not list(cell.glob(".*tmp*"))
    before = {p.name: p.stat().st_mtime_ns for p in published.rglob("*")}
    again = _write(cell, plan, leg, 0)
    assert again == published
    assert {p.name: p.stat().st_mtime_ns for p in published.rglob("*")} == before


def test_a_block_published_by_a_racing_worker_leaves_no_temp_behind(tmp_path: Path) -> None:
    """Two workers can both pass the skip check, and the loser leaves nothing behind.

    Its temp directory is dot-prefixed, so the census glob would never see it and nothing
    downstream would ever clean it up.
    """
    plan = vec.BlockPlan(passages=8, block_passages=8, part_passages=8)
    cell = tmp_path / "cell"
    winner = _write(cell, plan, DENSE_LEG, 0)
    loser_tmp = cell / f".{vec.block_dir_name(0)}.tmp-999"
    loser_tmp.mkdir(parents=True)
    (loser_tmp / "vectors.npy").write_bytes(b"junk")
    assert vec._publish_block_dir(loser_tmp, winner) == winner
    assert not loser_tmp.exists()
    assert (winner / "vectors.npy").read_bytes() != b"junk"


@pytest.mark.parametrize("leg", [DENSE_LEG, SPARSE_LEG, LATE_INTERACTION_LEG])
def test_two_writes_of_one_block_are_byte_identical(tmp_path: Path, leg: str) -> None:
    """A rebuilt block is byte-identical, so no witness may carry a timestamp."""
    plan = vec.BlockPlan(passages=13, block_passages=8, part_passages=16)
    a = _write(tmp_path / "a", plan, leg, 1)
    b = _write(tmp_path / "b", plan, leg, 1)
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    record = json.loads((a / vec.BLOCK_FILENAME).read_text(encoding="utf-8").strip())
    assert not [k for k in record if "time" in k or "date" in k or k == "ts"]


def test_witness_records_the_contracted_fields(tmp_path: Path) -> None:
    plan = vec.BlockPlan(passages=13, block_passages=8, part_passages=16)
    cell = tmp_path / "cell"
    _write(cell, plan, LATE_INTERACTION_LEG, 1)
    record = json.loads(
        (cell / vec.block_dir_name(1) / vec.BLOCK_FILENAME).read_text(encoding="utf-8").strip()
    )
    assert record == {
        "block": 1,
        "blocks": 2,
        "block_passages": 8,
        "passages": 5,
        "lang": _LANG,
        "variant": _VARIANT,
        "leg": LATE_INTERACTION_LEG,
        "encode_hash": _HASH,
        "dtype": vec.DEFAULT_LATE_INTERACTION_DTYPE,
        # Only the late-interaction leg carries a leaf-vector count.
        "vectors": sum(1 + i % 3 for i in range(5)),
        "bytes": record["bytes"],
    }
    assert record["bytes"] > 0


def test_dense_and_sparse_witnesses_carry_no_vector_count(tmp_path: Path) -> None:
    plan = vec.BlockPlan(passages=4, block_passages=8, part_passages=8)
    for leg in (DENSE_LEG, SPARSE_LEG):
        cell = tmp_path / leg
        _write(cell, plan, leg, 0)
        record = json.loads(
            (cell / vec.block_dir_name(0) / vec.BLOCK_FILENAME).read_text(encoding="utf-8").strip()
        )
        assert record["vectors"] is None
        assert vec.read_cell_census(cell).vectors is None


# --------------------------------------------------------------------------- #
# The id map is `preprocess.index`'s, verbatim, and has no text column.
# --------------------------------------------------------------------------- #
def test_idmap_is_the_index_schema_verbatim(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    plan = vec.BlockPlan(passages=5, block_passages=8, part_passages=8)
    block = _write(tmp_path / "cell", plan, DENSE_LEG, 0)
    assert vec.idmap_filename() == IDMAP_FILENAME
    with pq.ParquetFile(str(block / IDMAP_FILENAME)) as pf:
        assert pf.schema_arrow == idmap_arrow_schema()
    rows = read_parquet(block / IDMAP_FILENAME)
    assert [r["ordinal"] for r in rows] == list(range(5))
    assert {r["source_lang"] for r in rows} == {_LANG}
    # This is structural rather than a discipline: there is no column a text could sit in.
    assert set(rows[0]) == {"ordinal", "passage_id", "document_id", "source_lang"}


# --------------------------------------------------------------------------- #
# Sequential streaming: the property the design exists for.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leg", [DENSE_LEG, SPARSE_LEG, LATE_INTERACTION_LEG])
def test_stream_part_is_a_gapless_concatenation(tmp_path: Path, leg: str) -> None:
    plan = vec.BlockPlan(passages=37, block_passages=8, part_passages=16)
    cell = tmp_path / "cell"
    for block in range(plan.blocks):
        _write(cell, plan, leg, block)

    for part in range(plan.parts):
        lo, hi = plan.part_bounds(part)
        seen: list[vec.PassageRef] = []
        cursor = 0
        blocks: list[int] = []
        for read in vec.stream_part(cell, part=part, part_passages=plan.part_passages):
            assert read.ordinals.start == cursor  # gapless, part-local, in order
            assert len(read.refs) == len(read.ordinals)
            cursor = read.ordinals.stop
            blocks.append(read.block)
            seen.extend(read.refs)
            block_lo, _ = plan.block_bounds(read.block)
            _assert_reps_equal(leg, read.reps, _REPS[leg](block_lo, len(read.refs)))
        assert blocks == list(plan.blocks_of_part(part))  # ascending, each exactly once
        assert cursor == hi - lo
        assert seen == _refs(lo, hi - lo)  # the concatenation is the part's slice


def test_stream_part_reads_each_block_file_exactly_once(tmp_path: Path, monkeypatch) -> None:
    plan = vec.BlockPlan(passages=32, block_passages=8, part_passages=16)
    cell = tmp_path / "cell"
    for block in range(plan.blocks):
        _write(cell, plan, DENSE_LEG, block)
    opened: list[str] = []
    real = vec.read_block_dir
    monkeypatch.setattr(
        vec, "read_block_dir", lambda d, **kw: (opened.append(Path(d).name), real(d, **kw))[1]
    )
    list(vec.stream_part(cell, part=1, part_passages=16))
    assert opened == [vec.block_dir_name(2), vec.block_dir_name(3)]


def test_stream_part_rejects_a_part_outside_the_plan(tmp_path: Path) -> None:
    plan = vec.BlockPlan(passages=8, block_passages=8, part_passages=8)
    cell = tmp_path / "cell"
    _write(cell, plan, DENSE_LEG, 0)
    with pytest.raises(vec.BlockGeometryError, match="outside this cell's plan"):
        list(vec.stream_part(cell, part=1, part_passages=8))


def test_stream_part_rejects_a_part_size_that_is_not_whole_blocks(tmp_path: Path) -> None:
    plan = vec.BlockPlan(passages=8, block_passages=8, part_passages=8)
    cell = tmp_path / "cell"
    _write(cell, plan, DENSE_LEG, 0)
    with pytest.raises(vec.BlockGeometryError, match="whole number of blocks"):
        list(vec.stream_part(cell, part=0, part_passages=12))


# --------------------------------------------------------------------------- #
# The census never fans short.
# --------------------------------------------------------------------------- #
def _cell(tmp_path: Path, *, passages: int = 20, leg: str = DENSE_LEG) -> tuple[Path, vec.BlockPlan]:
    plan = vec.BlockPlan(passages=passages, block_passages=8, part_passages=16)
    cell = tmp_path / "cell"
    for block in range(plan.blocks):
        _write(cell, plan, leg, block)
    return cell, plan


def test_census_requires_at_least_one_block(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(vec.VectorIntegrityError, match="no block-"):
        vec.read_cell_census(tmp_path / "empty")


def test_census_refuses_a_block_without_its_witness(tmp_path: Path) -> None:
    cell, _ = _cell(tmp_path)
    (cell / vec.block_dir_name(1) / vec.BLOCK_FILENAME).unlink()
    with pytest.raises(vec.VectorIntegrityError, match="abandoned attempt"):
        vec.read_cell_census(cell)


def test_census_refuses_an_unpublished_block(tmp_path: Path) -> None:
    cell, _ = _cell(tmp_path)
    (cell / f"{vec.block_dir_name(1)}._SUCCESS").unlink()
    with pytest.raises(vec.VectorIntegrityError, match="_SUCCESS"):
        vec.read_cell_census(cell)


def test_census_refuses_a_witness_in_the_wrong_directory(tmp_path: Path) -> None:
    cell, _ = _cell(tmp_path)
    witness = cell / vec.block_dir_name(1) / vec.BLOCK_FILENAME
    record = json.loads(witness.read_text(encoding="utf-8").strip())
    record["block"] = 0
    witness.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(vec.VectorIntegrityError, match="claims block 0"):
        vec.read_cell_census(cell)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("blocks", 9),
        ("block_passages", 4),
        ("leg", SPARSE_LEG),
        ("encode_hash", "deadbeefcafe"),
        ("lang", "ru"),
        ("variant", "original"),
        ("dtype", "float16"),
    ],
)
def test_census_requires_unanimity_on_the_whole_plan(tmp_path: Path, key: str, value) -> None:
    cell, _ = _cell(tmp_path)
    witness = cell / vec.block_dir_name(1) / vec.BLOCK_FILENAME
    record = json.loads(witness.read_text(encoding="utf-8").strip())
    record[key] = value
    witness.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(vec.VectorIntegrityError, match="disagree about the cell's plan"):
        vec.read_cell_census(cell)


def test_census_refuses_a_cell_missing_a_block(tmp_path: Path) -> None:
    """Why a reader may never glob: 2 of 3 blocks looks exactly like a complete cell."""
    cell, plan = _cell(tmp_path)
    assert plan.blocks == 3
    import shutil

    shutil.rmtree(cell / vec.block_dir_name(1))
    (cell / f"{vec.block_dir_name(1)}._SUCCESS").unlink()
    with pytest.raises(vec.VectorIntegrityError, match="assembled short"):
        vec.read_cell_census(cell)


def test_census_refuses_a_short_block_that_is_not_the_last(tmp_path: Path) -> None:
    cell, _ = _cell(tmp_path)
    witness = cell / vec.block_dir_name(0) / vec.BLOCK_FILENAME
    record = json.loads(witness.read_text(encoding="utf-8").strip())
    record["passages"] = 7
    witness.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(vec.VectorIntegrityError, match="Only the last block"):
        vec.read_cell_census(cell)


def test_read_block_rejects_an_idmap_shorter_than_its_witness(tmp_path: Path) -> None:
    from ragtime.common.io import write_parquet_stream

    cell, _ = _cell(tmp_path, passages=8)
    block = cell / vec.block_dir_name(0)
    write_parquet_stream(
        block / IDMAP_FILENAME,
        [
            {"ordinal": 0, "passage_id": "p", "document_id": "d", "source_lang": _LANG},
        ],
        schema=idmap_arrow_schema(),
        skip_if_done=False,
    )
    with pytest.raises(vec.VectorIntegrityError, match="1 rows but the witness claims 8"):
        vec.read_block_dir(block)


def test_read_block_rejects_vectors_shorter_than_the_witness(tmp_path: Path) -> None:
    cell, _ = _cell(tmp_path, passages=8)
    block = cell / vec.block_dir_name(0)
    with open(block / "vectors.npy", "wb") as f:
        np.save(f, np.zeros((3, 4), dtype="<f4"), allow_pickle=False)
    with pytest.raises(vec.VectorIntegrityError, match="3 vectors but the witness claims 8"):
        vec.read_block_dir(block)


def test_late_interaction_rejects_a_bin_that_does_not_match_its_counts(tmp_path: Path) -> None:
    cell, _ = _cell(tmp_path, passages=8, leg=LATE_INTERACTION_LEG)
    block = cell / vec.block_dir_name(0)
    path = block / "vectors.bin"
    path.write_bytes(path.read_bytes()[:-2] + b"\x00")  # one byte short of a whole row
    with pytest.raises(vec.VectorIntegrityError, match="not a whole number of vectors"):
        vec.read_block_dir(block)


# --------------------------------------------------------------------------- #
# Writer guards, and the dtype parameter.
# --------------------------------------------------------------------------- #
def test_write_block_refuses_a_wrong_sized_block(tmp_path: Path) -> None:
    plan = vec.BlockPlan(passages=20, block_passages=8, part_passages=16)
    with pytest.raises(vec.BlockGeometryError, match="holds 8 passages, got 3"):
        vec.write_block(
            tmp_path / "cell",
            block=0,
            plan=plan,
            fmt=vec.format_for(DENSE_LEG),
            encode_hash=_HASH,
            lang=_LANG,
            variant=_VARIANT,
            refs=_refs(0, 3),
            reps=_dense_reps(0, 3),
        )


def test_write_block_refuses_mismatched_refs_and_reps(tmp_path: Path) -> None:
    plan = vec.BlockPlan(passages=8, block_passages=8, part_passages=8)
    with pytest.raises(vec.BlockGeometryError, match="address different rows"):
        vec.write_block(
            tmp_path / "cell",
            block=0,
            plan=plan,
            fmt=vec.format_for(DENSE_LEG),
            encode_hash=_HASH,
            lang=_LANG,
            variant=_VARIANT,
            refs=_refs(0, 8),
            reps=_dense_reps(0, 7),
        )


def test_write_block_refuses_a_block_outside_the_plan(tmp_path: Path) -> None:
    plan = vec.BlockPlan(passages=8, block_passages=8, part_passages=8)
    with pytest.raises(vec.BlockGeometryError, match="outside this cell's plan"):
        vec.write_block(
            tmp_path / "cell",
            block=1,
            plan=plan,
            fmt=vec.format_for(DENSE_LEG),
            encode_hash=_HASH,
            lang=_LANG,
            variant=_VARIANT,
            refs=_refs(0, 8),
            reps=_dense_reps(0, 8),
        )


def test_default_dtypes_are_the_documented_ones() -> None:
    assert vec.format_for(DENSE_LEG).dtype == "float32"
    assert vec.format_for(SPARSE_LEG).dtype == "float32"
    assert vec.format_for(LATE_INTERACTION_LEG).dtype == "float16"


def test_late_interaction_dtype_is_a_parameter_and_the_witness_decides_the_read(
    tmp_path: Path,
) -> None:
    """The fp16 default may move, and a block already written stays readable as written."""
    plan = vec.BlockPlan(passages=8, block_passages=8, part_passages=8)
    cell = tmp_path / "cell"
    _write(cell, plan, LATE_INTERACTION_LEG, 0, fmt=vec.format_for(LATE_INTERACTION_LEG, dtype="float32"))
    census = vec.read_cell_census(cell)
    assert census.dtype == "float32"
    assert census.format().dtype == "float32"
    got = vec.read_block_dir(cell / vec.block_dir_name(0))
    assert got.reps[0].dtype == np.dtype("<f4")
    # fp32 is exact for this fixture, so the round trip is bit for bit.
    for a, b in zip(got.reps, _li_reps(0, 8), strict=True):
        np.testing.assert_array_equal(np.asarray(a, dtype="float32"), b)


def test_fp16_storage_is_the_default_and_halves_the_bytes(tmp_path: Path) -> None:
    plan = vec.BlockPlan(passages=8, block_passages=8, part_passages=8)
    half = _write(tmp_path / "16", plan, LATE_INTERACTION_LEG, 0)
    full = _write(
        tmp_path / "32", plan, LATE_INTERACTION_LEG, 0,
        fmt=vec.format_for(LATE_INTERACTION_LEG, dtype="float32"),
    )
    assert (half / "vectors.bin").stat().st_size * 2 == (full / "vectors.bin").stat().st_size


def test_dense_dtype_is_float32_by_default_and_stored_little_endian(tmp_path: Path) -> None:
    plan = vec.BlockPlan(passages=4, block_passages=8, part_passages=8)
    block = _write(tmp_path / "cell", plan, DENSE_LEG, 0)
    arr = np.load(block / "vectors.npy", allow_pickle=False)
    assert arr.dtype == np.dtype("<f4")


def test_sparse_refuses_a_second_dtype() -> None:
    with pytest.raises(vec.BlockGeometryError, match="Seismic's own type"):
        vec.format_for(SPARSE_LEG, dtype="float16")


def test_unknown_leg_and_dtype_are_refused() -> None:
    with pytest.raises(vec.BlockGeometryError, match="unknown leg"):
        vec.format_for("bm25")
    with pytest.raises(vec.BlockGeometryError, match="unknown store dtype"):
        vec.format_for(DENSE_LEG, dtype="bfloat16")


def test_late_interaction_keeps_a_zero_vector_passage(tmp_path: Path) -> None:
    """A passage with no token vectors is counted as 0 rather than dropped. Dropping it
    would shift every offset after it."""
    plan = vec.BlockPlan(passages=3, block_passages=8, part_passages=8)
    cell = tmp_path / "cell"
    reps = [np.ones((2, 3), dtype="float32"), np.zeros((0, 3), dtype="float32"), np.full((1, 3), 7.0, dtype="float32")]
    vec.write_block(
        cell,
        block=0,
        plan=plan,
        fmt=vec.format_for(LATE_INTERACTION_LEG),
        encode_hash=_HASH,
        lang=_LANG,
        variant=_VARIANT,
        refs=_refs(0, 3),
        reps=reps,
    )
    got = vec.read_block_dir(cell / vec.block_dir_name(0))
    assert [len(r) for r in got.reps] == [2, 0, 1]
    np.testing.assert_array_equal(np.asarray(got.reps[2], dtype="float32"), reps[2])
    counts = read_parquet(cell / vec.block_dir_name(0) / vec.COUNTS_FILENAME)
    assert [r["n_vectors"] for r in counts] == [2, 0, 1]
    assert [r["passage_id"] for r in counts] == [r.passage_id for r in _refs(0, 3)]


def test_sparse_stores_seismic_components_order_not_the_encoder_dict_order(
    tmp_path: Path,
) -> None:
    from ragtime.serving.sparse_milco import seismic_components

    plan = vec.BlockPlan(passages=1, block_passages=8, part_passages=8)
    cell = tmp_path / "cell"
    weights = {7: 0.5, 3: 0.25, 100: 0.125}
    vec.write_block(
        cell,
        block=0,
        plan=plan,
        fmt=vec.format_for(SPARSE_LEG),
        encode_hash=_HASH,
        lang=_LANG,
        variant=_VARIANT,
        refs=_refs(0, 1),
        reps=[weights],
    )
    stored = read_parquet(cell / vec.block_dir_name(0) / "vectors.parquet")[0]
    assert stored["components"] == seismic_components(weights)[0] == ["100", "3", "7"]
