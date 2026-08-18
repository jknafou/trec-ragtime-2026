"""Byte-exact corpus reader over ordinary multi-line gzip JSONL.

The RAGTIME2 corpus files are newline-delimited JSON, so this reader is ``gzip.open``
plus a per-line ``json.loads``; the concatenated-single-line quirk lives only in the
topics file and is absorbed by ``common.topics``. Records are yielded as written, in file
order, carrying the original ``id`` that is threaded through chunk, translate and index.

Native records ``{id, text, url, date}`` have no ``lang`` field, so it is tagged here from
the filename (``{eng,spa,rus,zho}-docs.jsonl.gz`` maps to ``en``/``es``/``ru``/``zh``); the
doc-level ``*-trans.jsonl.gz`` records already carry their own.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common.io import concat_files, is_done

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "DOCS_GLOB",
    "SLICE_SUFFIX",
    "TRANS_GLOB",
    "count_docs",
    "lang_of",
    "read_native",
    "read_native_slice",
    "read_trans",
    "shard_stem",
    "slice_filename",
    "write_native_slices",
]

# The corpus filename convention mapped to the two-letter native language tag.
_STEM_TO_LANG = {"eng": "en", "spa": "es", "rus": "ru", "zho": "zh"}
DOCS_GLOB = "*-docs.jsonl.gz"
TRANS_GLOB = "*-trans.jsonl.gz"
SLICE_SUFFIX = ".jsonl.gz"  # a slice file is itself an ordinary native corpus file
# Slice files are a short-lived build intermediate read exactly once, so compression is
# tuned for write speed rather than size: the sequential split pass is on the critical
# path of the corpus build, the few GB it costs on disk are not.
_SLICE_COMPRESSLEVEL = 1


def lang_of(path: str | Path) -> str:
    """Derive the native language tag from a corpus filename (``eng-docs...`` -> ``en``).

    Works for both ``{lang}-docs.jsonl.gz`` and ``{lang}-trans.jsonl.gz``. Raises on an
    unrecognized stem rather than guessing.
    """
    stem = Path(path).name.split("-", 1)[0]
    try:
        return _STEM_TO_LANG[stem]
    except KeyError:
        raise ValueError(
            f"cannot derive language from corpus filename {Path(path).name!r} "
            f"(expected one of {sorted(_STEM_TO_LANG)})"
        ) from None


def _read_lines(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_native(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield native doc records ``{id, text, url, date, lang}`` in file order.

    The ``lang`` tag is added from the filename, since the native record carries none; no
    existing field is renamed, dropped or coerced. This is the dict shape ``chunk``
    consumes directly.
    """
    lang = lang_of(path)
    for rec in _read_lines(path):
        rec["lang"] = lang
        yield rec


def read_native_slice(path: str | Path, start: int, end: int) -> Iterator[dict[str, Any]]:
    """Yield native docs ``[start, end)`` by 0-based doc index: the doc-balanced shard slice.

    Streams the gzip sequentially, discarding lines before ``start`` and stopping at ``end``,
    so a worker holds only its shard's docs in memory. Same record shape as
    :func:`read_native`; documents are never split across a shard boundary.

    This is the slow path, kept as the semantic reference and as ``ChunkAdapter.work``'s
    fallback for a queue seeded before slice files existed. It decompresses from byte 0 on
    every call, so every worker would stream the whole language file to reach its offset.
    :func:`write_native_slices` materialises the same ranges once so each worker reads only
    its own small file, and yields these exact records in this exact order.
    """
    lang = lang_of(path)
    i = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if i >= end:
                break
            if i >= start:
                rec = json.loads(line)
                rec["lang"] = lang
                yield rec
            i += 1


def shard_stem(docs_file: str | Path, start: int, end: int) -> str:
    """Return ``<stem>_<start:09d>_<end:09d>``, the shared shard and slice name for a doc range.

    One naming source for the work-queue shard file (``ChunkAdapter.shards``) and its
    materialised slice file, so ``work`` can find its slice from the payload alone.
    Zero-padded offsets keep lexicographic order equal to numeric order, which
    ``ChunkAdapter.merge`` relies on to restore global doc order.
    """
    stem = Path(docs_file).name.removesuffix(SLICE_SUFFIX)
    return f"{stem}_{start:09d}_{end:09d}"


def slice_filename(docs_file: str | Path, start: int, end: int) -> str:
    """The per-shard slice filename, e.g. ``eng-docs_000000000_000005000.jsonl.gz``.

    Keeps the ``{eng,spa,rus,zho}`` prefix first so the slice is an ordinary native corpus
    file: :func:`lang_of` derives the same language from it as from the parent file, and
    :func:`read_native` reads it with no special case.
    """
    return shard_stem(docs_file, start, end) + SLICE_SUFFIX


def write_native_slices(
    path: str | Path,
    ranges: Sequence[tuple[int, int]],
    out_dir: str | Path,
    *,
    compresslevel: int = _SLICE_COMPRESSLEVEL,
) -> list[Path]:
    """Materialise each ``[start, end)`` doc range of ``path`` as its own gz, in one pass.

    Without this, every worker would stream a whole multi-GB language file to skip to its own
    offset. The ``seed`` role runs once, on one machine, before any worker; it reads each
    language file once and writes out the shard slices as their boundaries are crossed, after
    which a worker does a full sequential read of its own small slice.

    Each doc line is copied verbatim (stripped of surrounding whitespace, exactly as
    :func:`read_native_slice` strips it before ``json.loads``) and never re-serialized, so
    reading a slice with :func:`read_native` yields the same dicts, in the same order, as
    ``read_native_slice(path, start, end)``. Blank lines are skipped and do not consume a doc
    index, matching the reader's counting. The gzip header is written with ``mtime=0`` and an
    empty filename so a re-split produces identical bytes.

    A slice is finalised through ``common.io.concat_files`` (atomic rename plus ``_SUCCESS``),
    so a slice with its marker is skipped and a truncated one is regenerated, making this safe
    to re-run after a preempted seed. If every range is already done the file is not opened.
    Returns the slices written by this call.

    The slices duplicate the raw corpus on disk. They land under the family and
    chunker-hash-keyed ``Layout.corpus_slices_dir``, so they are keyed, shared and cleaned
    with the rest of the corpus artifacts.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [
        (start, end, out / slice_filename(path, start, end))
        for start, end in sorted(ranges)
        if not is_done(out / slice_filename(path, start, end))
    ]
    written: list[Path] = []
    if not todo:
        return written

    gz: Any = None
    raw: Any = None
    staging: Path | None = None
    dst: Path | None = None

    def _open(target: Path) -> None:
        nonlocal gz, raw, staging, dst
        dst = target
        staging = target.with_name(f".{target.name}.building-{os.getpid()}")
        # Not a context manager: the handle spans many loop iterations, opened at a range's
        # first doc and closed by _finalize at its last (or by the outer finally on error).
        # filename="" and mtime=0 pin the gzip header so a re-split is byte-identical; the
        # staging name carries a pid and must not leak into it.
        raw = open(staging, "wb")  # noqa: SIM115
        gz = gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=compresslevel, mtime=0
        )

    def _finalize() -> None:
        """Close the open slice and publish it atomically (temp, rename, ``_SUCCESS``)."""
        nonlocal gz, raw, staging, dst
        if gz is None:
            return
        gz.close()
        raw.close()
        assert dst is not None and staging is not None
        concat_files(dst, [staging])  # the shared atomic writer owns rename and _SUCCESS
        staging.unlink(missing_ok=True)
        written.append(dst)
        gz = raw = staging = dst = None

    j = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            i = 0  # 0-based doc index (blank lines are not docs), as read_native_slice counts
            for line in f:
                line = line.strip()
                if not line:
                    continue
                while j < len(todo) and i >= todo[j][1]:  # this range is complete
                    _finalize()
                    j += 1
                if j >= len(todo):
                    break  # every requested range is materialised; stop decompressing
                start, _end, target = todo[j]
                if i >= start:
                    if gz is None:
                        _open(target)
                    gz.write((line + "\n").encode("utf-8"))
                i += 1
        _finalize()  # the last range ends at EOF
    finally:
        if gz is not None:  # an error mid-slice leaves no published file and no marker
            gz.close()
            raw.close()
            if staging is not None:
                staging.unlink(missing_ok=True)
    return written


def count_docs(path: str | Path) -> int:
    """Count the non-blank native doc lines, the seed's per-file doc balance input."""
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def read_trans(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield doc-level organizer-MT records ``{id, text, lang}`` in file order.

    The ``id`` matches the native record's ``id``; nothing is added or coerced, since the
    record already carries its own ``lang``. The organizer-MT file is not one of the corpus
    renderings: both MT arms are our own (NLLB and OPUS-MT). This reader stays because the
    shipped file is a useful external reference point.
    """
    yield from _read_lines(path)
