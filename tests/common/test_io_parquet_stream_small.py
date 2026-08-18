"""``iter_parquet`` / ``write_parquet_stream``: the constant-memory Parquet pair.

The merged corpus spine is Parquet with zstd compression. What is covered here:

- byte-determinism: the same records written twice are the same bytes, so the
  artifact tree can serve as the checkpoint;
- round-trip fidelity: ``text`` and every ``sentence_char_spans`` slice survive
  byte-identically, which is what the verbatim NFC span commit grounds on;
- streaming: neither writer nor reader materialises the whole file (many row
  groups, many read batches, and the writer accepts a one-pass generator);
- the ``write_jsonl`` durability contract (temp, rename, ``_SUCCESS``, skip-if-done).
"""

from __future__ import annotations

import hashlib

import pytest

from ragtime.common.io import (
    PARQUET_COMPRESSION,
    PARQUET_USE_DICTIONARY,
    is_done,
    iter_parquet,
    iter_parquet_batches,
    read_parquet,
    write_parquet_stream,
)

pytestmark = pytest.mark.small


def _schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("passage_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("sentence_ids", pa.list_(pa.string())),
            pa.field("sentence_char_spans", pa.list_(pa.list_(pa.int64(), 2))),
            pa.field("token_count", pa.int64()),
        ]
    )


def _record(i: int) -> dict:
    """A passage-shaped record whose text exercises the Unicode edge cases."""
    sents = [f"Sentence {i} café — résumé.", "第二个 句子。", f"Тест номер {i}."]
    text = " ".join(sents)
    spans, pos = [], 0
    for j, s in enumerate(sents):
        if j:
            pos += 1
        spans.append([pos, pos + len(s)])
        pos += len(s)
    return {
        "passage_id": f"eng-docs/{i:07d}#p0",
        "text": text,
        "sentence_ids": [f"eng-docs/{i:07d}#p0#s{j}" for j in range(len(sents))],
        "sentence_char_spans": spans,
        "token_count": len(text.split()),
    }


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_parquet_stream_is_byte_deterministic(tmp_path) -> None:
    """Two writes of the same records produce an identical sha256."""
    recs = [_record(i) for i in range(50)]
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_parquet_stream(a, iter(recs), schema=_schema(), row_group_size=7)
    write_parquet_stream(b, iter(recs), schema=_schema(), row_group_size=7)
    assert _sha256(a) == _sha256(b)
    assert a.stat().st_size == b.stat().st_size
    assert is_done(a) and is_done(b)
    assert not list(tmp_path.glob("*.tmp-*"))  # no temp left behind


def test_writer_options_are_pinned_constants_not_pyarrow_defaults(tmp_path) -> None:
    """The determinism contract is pinned in the file metadata: zstd, no dictionary."""
    import pyarrow.parquet as pq

    out = tmp_path / "pinned.parquet"
    write_parquet_stream(out, (_record(i) for i in range(20)), schema=_schema(), row_group_size=5)
    md = pq.ParquetFile(str(out)).metadata
    assert md.num_row_groups == 4  # row_group_size honoured, not one giant group
    for rg in range(md.num_row_groups):
        for c in range(md.num_columns):
            col = md.row_group(rg).column(c)
            assert col.compression.lower() == PARQUET_COMPRESSION
            # ids are globally unique, so dictionary encoding buys nothing here.
            assert PARQUET_USE_DICTIONARY is False
            assert "PLAIN_DICTIONARY" not in [str(e) for e in col.encodings]


def test_round_trip_preserves_text_and_every_sentence_span_byte_identically(tmp_path) -> None:
    """``text`` and every ``text[start:end]`` slice survive verbatim."""
    recs = [_record(i) for i in range(30)]
    out = tmp_path / "rt.parquet"
    write_parquet_stream(out, iter(recs), schema=_schema(), row_group_size=8)
    got = list(iter_parquet(out, batch_size=4))
    assert got == recs  # every field, every value, same order
    for src, dst in zip(recs, got, strict=True):
        assert dst["text"].encode() == src["text"].encode()
        for (s0, e0), (s1, e1) in zip(
            src["sentence_char_spans"], dst["sentence_char_spans"], strict=True
        ):
            assert (s0, e0) == (s1, e1)
            assert dst["text"][s1:e1] == src["text"][s0:e0]
            assert dst["text"][s1:e1].encode() == src["text"][s0:e0].encode()


def test_iter_parquet_streams_in_batches_without_materializing_the_file(tmp_path) -> None:
    """The reader is constant-memory: many small batches over a multi-row-group file."""
    n = 200
    out = tmp_path / "big.parquet"
    write_parquet_stream(out, (_record(i) for i in range(n)), schema=_schema(), row_group_size=10)

    batches = list(iter_parquet_batches(out, batch_size=16))
    assert len(batches) > 1  # streamed, not one whole-table read
    assert max(len(b) for b in batches) <= 16  # bounded working set
    assert sum(len(b) for b in batches) == n
    # the flattened stream is the file order, and it is lazy (first row before the end).
    it = iter_parquet(out, batch_size=16)
    assert next(it)["passage_id"] == _record(0)["passage_id"]
    assert [r["passage_id"] for r in it][-1] == _record(n - 1)["passage_id"]
    # column pruning reads only what was asked for (the id-column scan path).
    pruned = next(iter_parquet(out, columns=["passage_id"], batch_size=64))
    assert pruned.keys() == {"passage_id"}


def test_write_parquet_stream_consumes_a_one_pass_generator(tmp_path) -> None:
    """Records may be a generator: the writer never re-iterates or lists them."""
    consumed: list[int] = []

    def gen():
        for i in range(25):
            consumed.append(i)
            yield _record(i)

    out = tmp_path / "gen.parquet"
    write_parquet_stream(out, gen(), schema=_schema(), row_group_size=10)
    assert consumed == list(range(25))  # exactly one pass
    assert len(list(iter_parquet(out))) == 25


def test_schema_pins_column_order_regardless_of_record_key_order(tmp_path) -> None:
    """Shard records with shuffled keys still produce the pinned column order/bytes."""
    recs = [_record(i) for i in range(12)]
    shuffled = [{k: r[k] for k in reversed(list(r))} for r in recs]
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_parquet_stream(a, iter(recs), schema=_schema())
    write_parquet_stream(b, iter(shuffled), schema=_schema())
    assert _sha256(a) == _sha256(b)
    assert list(iter_parquet(a)) == list(iter_parquet(b)) == recs


def test_stream_writer_keeps_the_success_and_skip_if_done_contract(tmp_path) -> None:
    out = tmp_path / "nested" / "resume.parquet"
    write_parquet_stream(out, iter([_record(0)]), schema=_schema())
    assert is_done(out)
    mtime = out.stat().st_mtime_ns
    write_parquet_stream(out, iter([_record(99)]), schema=_schema())  # already done -> no-op
    assert out.stat().st_mtime_ns == mtime
    assert list(iter_parquet(out)) == [_record(0)]
    write_parquet_stream(out, iter([_record(99)]), schema=_schema(), skip_if_done=False)
    assert list(iter_parquet(out)) == [_record(99)]


def test_empty_record_stream_writes_a_readable_done_artifact(tmp_path) -> None:
    out = tmp_path / "empty.parquet"
    write_parquet_stream(out, iter([]), schema=_schema())
    assert is_done(out)
    assert list(iter_parquet(out)) == []
    assert read_parquet(out) == []


def test_stream_writer_rejects_non_finite_floats_and_leaves_no_partial_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        write_parquet_stream(tmp_path / "bad.parquet", iter([{"score": float("nan")}]))
    with pytest.raises(ValueError, match="non-finite"):
        write_parquet_stream(tmp_path / "bad2.parquet", iter([{"score": float("inf")}]))
    # a failed write is a non-event: no artifact, no _SUCCESS, no temp (still resumable).
    assert not list(tmp_path.iterdir())


def test_a_failing_record_stream_leaves_no_artifact_or_temp(tmp_path) -> None:
    """Mid-stream failure (after a row group was already flushed) is still a non-event."""

    def gen():
        yield from (_record(i) for i in range(12))
        raise RuntimeError("shard read blew up")

    out = tmp_path / "partial.parquet"
    with pytest.raises(RuntimeError, match="blew up"):
        write_parquet_stream(out, gen(), schema=_schema(), row_group_size=5)
    assert not out.exists() and not is_done(out)
    assert not list(tmp_path.glob("*.tmp-*"))
