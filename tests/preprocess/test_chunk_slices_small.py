"""Per-shard slice files, and their equivalence to reading the whole language file.

The shard read path used to stream the whole 1.29 GB language gz from byte 0 on every
worker, at about 36 us per document skipped, which came to roughly 258 GB of concurrent
reads from the shared filesystem per shard round on a 200-task array. ``seed`` now writes
each shard's documents into a small gz of their own once, and ``work`` reads only that.

The property that matters is equivalence: a slice yields exactly the records
``read_native_slice(file, start, end)`` yields, in the same order and with the same
``lang``, so shard outputs produced by the old path stay valid and a re-run reproduces
them. Splitting is also idempotent and resumable, and a missing slice falls back to the old
path with a log line rather than silently.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
from pathlib import Path

import pytest

from ragtime.common import Layout, iter_parquet
from ragtime.common.io import is_done, success_marker
from ragtime.config import all_hashes
from ragtime.orchestration import saturate
from ragtime.preprocess import corpus
from ragtime.preprocess.chunk import ChunkAdapter, _ChunkCtx

pytestmark = pytest.mark.small

_FAM = "e2e"


def _docs(n: int, stem: str = "eng-docs") -> list[dict]:
    """Native records carrying the scripts where a verbatim copy can go wrong."""
    out = []
    for i in range(n):
        text = f"doc {i} café: 第{i}行|segunda oración"  # accents and CJK
        out.append(
            {"id": f"{stem}/{i:07d}", "text": text, "url": f"http://x/{i}", "date": "2026-01-01"}
        )
    return out


def _write_gz(path: Path, rows: list[dict], *, blank_lines: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if blank_lines:
                f.write("\n")  # a blank line is not a document and takes no document index
    return path


# --------------------------------------------------------------------------- #
# Reading a slice gives the same records as read_native_slice, one for one.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("blank_lines", [False, True])
def test_slice_reads_equal_read_native_slice_for_every_range(
    tmp_path: Path, blank_lines: bool
) -> None:
    src = _write_gz(tmp_path / "raw" / "eng-docs.jsonl.gz", _docs(20), blank_lines=blank_lines)
    out_dir = tmp_path / "slices"
    ranges = [(0, 3), (3, 8), (8, 9), (9, 17), (17, 20)]  # first and last shard included

    written = corpus.write_native_slices(src, ranges, out_dir)
    assert len(written) == len(ranges)

    seen: list[dict] = []
    for start, end in ranges:
        sl = out_dir / corpus.slice_filename(src, start, end)
        assert is_done(sl)  # written temp-then-rename with a _SUCCESS, like every artefact
        expected = list(corpus.read_native_slice(src, start, end))
        got = list(corpus.read_native(sl))
        assert got == expected  # same records, same order, same lang tagging
        assert len(got) == end - start  # whole documents, none dropped or split
        seen.extend(got)
    # The slices together are the whole file in file order, with no gap and no duplicate.
    assert seen == list(corpus.read_native(src))


def test_slice_filename_keeps_the_language_prefix_and_the_shard_name() -> None:
    name = corpus.slice_filename("eng-docs.jsonl.gz", 0, 5000)
    assert name == "eng-docs_000000000_000005000.jsonl.gz"
    assert corpus.lang_of(name) == corpus.lang_of("eng-docs.jsonl.gz") == "en"
    # The slice name is the queue's shard name plus the suffix, so there is one source for
    # the name and no separate key in the payload.
    assert name == corpus.shard_stem("eng-docs.jsonl.gz", 0, 5000) + corpus.SLICE_SUFFIX


def test_a_range_past_end_of_file_writes_nothing(tmp_path: Path) -> None:
    src = _write_gz(tmp_path / "eng-docs.jsonl.gz", _docs(4))
    written = corpus.write_native_slices(src, [(0, 4), (4, 9)], tmp_path / "slices")
    assert [p.name for p in written] == [corpus.slice_filename(src, 0, 4)]


# --------------------------------------------------------------------------- #
# A completed slice is not rewritten on a second pass.
# --------------------------------------------------------------------------- #
def test_split_twice_rewrites_nothing_and_is_byte_deterministic(tmp_path: Path) -> None:
    src = _write_gz(tmp_path / "spa-docs.jsonl.gz", _docs(12, "spa-docs"))
    out_dir = tmp_path / "slices"
    ranges = [(0, 5), (5, 12)]

    first = corpus.write_native_slices(src, ranges, out_dir)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in first}

    assert corpus.write_native_slices(src, ranges, out_dir) == []  # nothing rewritten
    for p in first:
        assert (p.read_bytes(), p.stat().st_mtime_ns) == before[p.name]

    # A fresh split of the same input is byte-identical, because the gzip header pins
    # mtime to 0.
    other = tmp_path / "slices2"
    for p in corpus.write_native_slices(src, ranges, other):
        assert p.read_bytes() == before[p.name][0]
    assert not list(out_dir.glob(".*"))  # no staging files left behind


def test_truncated_or_missing_slices_are_regenerated(tmp_path: Path) -> None:
    src = _write_gz(tmp_path / "rus-docs.jsonl.gz", _docs(12, "rus-docs"))
    out_dir = tmp_path / "slices"
    ranges = [(0, 4), (4, 8), (8, 12)]
    corpus.write_native_slices(src, ranges, out_dir)

    truncated = out_dir / corpus.slice_filename(src, 4, 8)  # a preempted write, no _SUCCESS
    truncated.write_bytes(truncated.read_bytes()[:12])
    success_marker(truncated).unlink()
    missing = out_dir / corpus.slice_filename(src, 8, 12)
    missing.unlink()
    success_marker(missing).unlink()

    written = corpus.write_native_slices(src, ranges, out_dir)
    assert {p.name for p in written} == {truncated.name, missing.name}  # only the broken two
    for start, end in ranges:
        sl = out_dir / corpus.slice_filename(src, start, end)
        assert list(corpus.read_native(sl)) == list(corpus.read_native_slice(src, start, end))


# --------------------------------------------------------------------------- #
# Through the real adapter and driver, the slice path and the fallback path agree.
# --------------------------------------------------------------------------- #
def _stage(base: Path, cfg, docs_by_stem: dict[str, list[dict]]) -> Layout:
    ch = all_hashes(cfg)["chunker"]
    layout = Layout(run_dir=base, base=base, family=_FAM, chunker_hash=ch)
    raw = layout.corpus_raw_dir(_FAM, ch)
    for stem, rows in docs_by_stem.items():
        _write_gz(raw / f"{stem}.jsonl.gz", rows)
    return layout


def _run_lifecycle(
    base: Path, cfg, docs, fake_segmenter, fake_tokenizer, monkeypatch, *, drop_slices: bool
) -> Path:
    layout = _stage(base, cfg, docs)
    ch = all_hashes(cfg)["chunker"]
    adapter = ChunkAdapter(base=str(base))
    ctx = _ChunkCtx(
        segmenter=fake_segmenter,
        tokenizer=fake_tokenizer,
        raw_dir=layout.corpus_raw_dir(_FAM, ch),
        batch_size=4,
        slices_dir=layout.corpus_slices_dir(_FAM, ch),
    )
    monkeypatch.setattr(ChunkAdapter, "bringup", lambda self, _cfg: ctx)
    wq = saturate.queue_for(cfg, adapter, base=str(base))
    saturate.seed(cfg, adapter, wq)
    if drop_slices:  # a queue seeded before slices existed has none
        shutil.rmtree(layout.corpus_slices_dir(_FAM, ch))
    saturate.run_worker(cfg, adapter, wq, backoff_s=0.0, max_iters=200)
    saturate.drive(cfg, adapter, wq, poll_s=0.0, max_polls=5)
    return layout.sentences_path()


def test_work_uses_the_slice_and_the_missing_slice_fallback_is_identical_and_logged(
    tmp_path, make_cfg, fake_segmenter, fake_tokenizer, monkeypatch, caplog
) -> None:
    cfg = make_cfg(token_budget=20, overlap_frac=0.15)
    cfg.blocks["execution"]["corpus_shards"] = 8  # several shards per file
    docs = {
        "eng-docs": [
            {"id": f"eng-docs/{i:07d}", "text": "a b c|d e|f g h i", "url": "u", "date": "d"}
            for i in range(12)
        ],
        "spa-docs": [
            {"id": f"spa-docs/{i:07d}", "text": "uno dos|tres cuatro cinco", "url": "u", "date": "d"}
            for i in range(9)
        ],
    }

    with caplog.at_level(logging.WARNING, logger="ragtime.preprocess.chunk"):
        sliced = _run_lifecycle(
            tmp_path / "sliced", cfg, docs, fake_segmenter, fake_tokenizer, monkeypatch,
            drop_slices=False,
        )
    assert "preprocess.chunk.slice.fallback" not in caplog.text  # the fast path was taken
    # Every shard got its own slice file, named after the shard.
    ch = all_hashes(cfg)["chunker"]
    slices = Layout(
        run_dir=tmp_path / "sliced", base=tmp_path / "sliced", family=_FAM, chunker_hash=ch
    ).corpus_slices_dir(_FAM, ch)
    assert len(list(slices.glob("*.jsonl.gz"))) > 2

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="ragtime.preprocess.chunk"):
        fallback = _run_lifecycle(
            tmp_path / "fallback", cfg, docs, fake_segmenter, fake_tokenizer, monkeypatch,
            drop_slices=True,
        )
    assert "preprocess.chunk.slice.fallback" in caplog.text  # the slow path is visible

    # What a shard sees is unchanged either way, so shard outputs written before slices
    # existed stay valid and a re-run reproduces them byte for byte.
    assert list(iter_parquet(sliced)) == list(iter_parquet(fallback))
    assert sliced.read_bytes() == fallback.read_bytes()
