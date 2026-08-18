"""Artifact IO: atomic, resumable, byte-stable readers and writers.

Every writer here is resumable: the payload goes to a temp file in the same
directory, is ``fsync``'d, atomically renamed over the target, the parent
directory is ``fsync``'d, and a ``_SUCCESS`` marker is written last. Re-running
over a completed artifact is a no-op, so a resume recomputes only missing cells.

- JSONL uses the stdlib ``json`` module with ``ensure_ascii=False`` and
  ``separators=(",", ":")`` for byte-stable output, and rejects non-finite floats
  before serializing, since the stdlib would emit invalid ``NaN``/``Infinity``.
- Parquet uses ``pyarrow`` with every writer option pinned as an explicit constant
  (see :data:`PARQUET_COMPRESSION` and friends) so the bytes do not track a pyarrow
  default. ``read_parquet``/``write_parquet`` materialize a whole small table;
  ``iter_parquet``/``write_parquet_stream`` are the constant-memory pair for
  corpus-scale artifacts such as the multi-million-document passage spine.
- ``lmdb_open`` is the passage-store environment factory. Query-time opens are
  ``readonly=True, lock=False``, which is safe because the store is immutable once
  built.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only (pyarrow is imported lazily)
    import pyarrow as pa

__all__ = [
    "DEFAULT_MAP_SIZE",
    "DEV_MARKER",
    "PARQUET_BATCH_SIZE",
    "PARQUET_COMPRESSION",
    "PARQUET_COMPRESSION_LEVEL",
    "PARQUET_DATA_PAGE_SIZE",
    "PARQUET_ROW_GROUP_SIZE",
    "PARQUET_USE_DICTIONARY",
    "PARQUET_VERSION",
    "align_document_rows",
    "concat_files",
    "concat_parquet",
    "ensure_finite",
    "is_dev",
    "is_done",
    "iter_parquet",
    "iter_parquet_batches",
    "iter_parquet_range",
    "lmdb_open",
    "mark_dev",
    "parquet_row_group_sizes",
    "parquet_row_group_span",
    "read_jsonl",
    "read_parquet",
    "success_marker",
    "write_jsonl",
    "write_lines",
    "write_parquet",
    "write_parquet_stream",
]

# JSONL byte-stability knobs (validator-safe, native non-ASCII preserved).
_JSON_SEP = (",", ":")

# LMDB virtual map reservation. LMDB reserves this as a sparse mmap up front, not
# as physical bytes, so it is sized generously for the full multi-rendering corpus
# to avoid MDB_MAP_FULL mid-build; on-disk size grows with real data.
DEFAULT_MAP_SIZE: int = 1 << 40  # 1 TiB virtual reservation


# --------------------------------------------------------------------------- #
# Finite-float guard + success markers
# --------------------------------------------------------------------------- #
def ensure_finite(obj: Any) -> Any:
    """Recursively reject NaN / +-Inf; return ``obj`` unchanged if clean."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float cannot be serialized: {obj!r}")
    elif isinstance(obj, dict):
        for v in obj.values():
            ensure_finite(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            ensure_finite(v)
    return obj


def success_marker(path: str | os.PathLike[str]) -> Path:
    """The ``_SUCCESS`` companion marker for a single-file artifact ``path``."""
    p = Path(path)
    return p.with_name(p.name + "._SUCCESS")


def is_done(path: str | os.PathLike[str]) -> bool:
    """True if the artifact's ``_SUCCESS`` marker exists (skip-if-done)."""
    return success_marker(path).exists()


#: Marks a tree as produced by the development harness rather than by a run.
#: A sibling of ``_SUCCESS``, kept here because this module owns marker semantics.
DEV_MARKER = "_DEV_RUN.json"


def mark_dev(run_dir: str | os.PathLike[str], meta: Mapping[str, Any]) -> Path:
    """Stamp ``run_dir`` as a dev tree, before any artifact is written.

    Containment is already structural, since a dev path cannot be parsed as a
    production cell key. The marker adds a second layer that survives a file being
    copied out of the dev tree, so a hand-promoted artifact still carries its
    provenance; ``meta`` records the injected inputs and their sha256. Written with
    the same atomic temp-then-rename as every other artifact here.
    """
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    target = d / DEV_MARKER
    tmp = d / f".{DEV_MARKER}.tmp-{os.getpid()}"
    tmp.write_text(json.dumps(dict(meta), indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    return target


def is_dev(path: str | os.PathLike[str]) -> bool:
    """True if ``path`` or any ancestor is marked as a dev tree.

    Walks upward because the marker sits at the dev-run root while callers hold a
    leaf artifact path.
    """
    p = Path(path)
    return any((cand / DEV_MARKER).exists() for cand in (p, *p.parents))


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a rename is durable across a crash."""
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via temp -> fsync -> rename -> fsync(dir)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _write_success(path: Path) -> None:
    marker = success_marker(path)
    with open(marker, "wb") as f:
        f.write(b"")
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(marker.parent)


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a newline-delimited JSON file into a list of dicts (skips blank lines)."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[Any],
    *,
    skip_if_done: bool = True,
) -> Path:
    """Atomically write ``records`` as byte-stable JSONL with a ``_SUCCESS`` marker.

    Each record may be a dict (or any JSON-serializable value) or an object with a
    ``to_dict()`` method. Non-finite floats are rejected before writing. If the
    artifact is already done and ``skip_if_done``, this is a no-op and the existing
    file's bytes and mtime are left untouched.
    """
    p = Path(path)
    if skip_if_done and is_done(p):
        return p
    lines: list[str] = []
    for rec in records:
        obj = rec.to_dict() if hasattr(rec, "to_dict") else rec
        lines.append(json.dumps(ensure_finite(obj), ensure_ascii=False, separators=_JSON_SEP))
    data = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _atomic_write_bytes(p, data)
    _write_success(p)
    return p


def write_lines(
    path: str | os.PathLike[str],
    lines: Iterable[str],
    *,
    skip_if_done: bool = True,
) -> Path:
    """Atomically write ``lines`` as a newline-terminated text artifact + ``_SUCCESS``.

    The non-JSON sibling of :func:`write_jsonl`, with the same durability and resume
    contract. It exists because the Task-2 TREC run is a six-column text file rather
    than JSONL.

    Each line is written verbatim followed by ``\\n``. An embedded newline is rejected,
    since a TREC row containing one would silently become two rows.
    """
    p = Path(path)
    if skip_if_done and is_done(p):
        return p
    out: list[str] = []
    for line in lines:
        if "\n" in line or "\r" in line:
            raise ValueError(f"embedded newline in a text artifact line: {line!r}")
        out.append(line)
    data = ("\n".join(out) + ("\n" if out else "")).encode("utf-8")
    _atomic_write_bytes(p, data)
    _write_success(p)
    return p


def concat_files(
    path: str | os.PathLike[str],
    sources: Sequence[str | os.PathLike[str]],
    *,
    skip_if_done: bool = True,
    buffer_size: int = 16 * 1024 * 1024,
) -> Path:
    """Stream-concatenate ``sources``' raw bytes into one artifact, atomically.

    The constant-memory merge writer: bytes are copied file by file in the given
    order through a fixed ``buffer_size`` buffer, into a temp sibling that is
    ``fsync``'d, renamed over the target, then dir-``fsync``'d and marked
    ``_SUCCESS`` - the durability and resume semantics of :func:`write_jsonl`.
    Callers own both the ordering and the format guarantee, for example that each
    source is byte-stable JSONL with a trailing newline so raw concatenation equals
    a parse and rewrite.
    """
    p = Path(path)
    if skip_if_done and is_done(p):
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as out:
        for src in sources:
            with open(src, "rb") as f:
                shutil.copyfileobj(f, out, buffer_size)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, p)
    _fsync_dir(p.parent)
    _write_success(p)
    return p


# --------------------------------------------------------------------------- #
# Parquet (pyarrow)
#
# The corpus spine and the translate tables are written as Parquet with zstd, which
# is both smaller and faster to scan than JSONL at this scale. Every writer option
# below is pinned explicitly rather than left to a pyarrow default, so the same
# records always produce the same bytes and the artifact tree stays a valid
# checkpoint across library versions.
# --------------------------------------------------------------------------- #
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_USE_DICTIONARY = False  # ids are globally unique, so a dictionary has no reuse
PARQUET_VERSION = "2.6"
PARQUET_DATA_PAGE_SIZE = 1 << 20  # 1 MiB pages
PARQUET_ROW_GROUP_SIZE = 20_000  # rows per row group, the writer's memory bound
PARQUET_BATCH_SIZE = 10_000  # rows per read batch, iter_parquet's memory bound


def _parquet_writer_options() -> dict[str, Any]:
    """The pinned ``ParquetWriter`` options that keep output byte-deterministic."""
    return {
        "compression": PARQUET_COMPRESSION,
        "compression_level": PARQUET_COMPRESSION_LEVEL,
        "use_dictionary": PARQUET_USE_DICTIONARY,
        "version": PARQUET_VERSION,
        "data_page_size": PARQUET_DATA_PAGE_SIZE,
        "write_statistics": True,
    }


def _reject_nonfinite_table(table: pa.Table) -> None:
    """Raise on NaN or +-Inf in any top-level float column.

    The Parquet counterpart of :func:`ensure_finite`, vectorized as one Arrow kernel
    per float column because the row-by-row Python walk does not scale to corpus-size
    tables. Nulls are skipped, and floats nested inside a list or struct column are
    not checked; no shared schema has one.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    for name, col in zip(table.column_names, table.columns, strict=True):
        if not pa.types.is_floating(col.type):
            continue
        if pc.all(pc.is_finite(col), min_count=0).as_py() is False:
            raise ValueError(f"non-finite float cannot be serialized: column {name!r}")


def read_parquet(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a whole Parquet file into a list of row dicts.

    For small whole-table artifacts only; corpus-scale readers such as the passage
    spine and the translate tables must use :func:`iter_parquet` instead.
    """
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pylist()


def parquet_row_group_sizes(path: str | os.PathLike[str]) -> list[int]:
    """Row count of each row group, in file order, read from the footer only.

    Parquet's footer is already an index, so this is a metadata-only open and no
    column data is decompressed. Plan contiguous row-group ranges with this and read
    them with :func:`iter_parquet_batches`'s ``row_groups`` argument.
    """
    import pyarrow.parquet as pq

    with pq.ParquetFile(str(path)) as pf:
        md = pf.metadata
        return [md.row_group(i).num_rows for i in range(md.num_row_groups)]


def iter_parquet_batches(
    path: str | os.PathLike[str],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = PARQUET_BATCH_SIZE,
    row_groups: Sequence[int] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Stream a Parquet file as batches of row dicts, in constant memory.

    Wraps ``ParquetFile.iter_batches``: at most ``batch_size`` rows are materialized
    at a time, whatever the file's size. ``columns`` prunes the read to the columns
    the caller needs and is worth passing whenever the whole record is not required.

    ``row_groups`` prunes the read to a subset of the file's row groups, resolved
    through the footer index rather than by scanning to an offset. This is what lets
    a work-queue shard be a contiguous row range over a shared artifact with no
    pre-split copy on disk. Indices must be derived from cumulative
    :func:`parquet_row_group_sizes` rather than from ``row // PARQUET_ROW_GROUP_SIZE``,
    since row groups are not guaranteed uniform. An empty sequence yields nothing.
    """
    import pyarrow.parquet as pq

    cols = list(columns) if columns else None
    rgs = None if row_groups is None else [int(i) for i in row_groups]
    if rgs is not None and not rgs:
        return
    with pq.ParquetFile(str(path)) as pf:
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols, row_groups=rgs):
            yield batch.to_pylist()


def iter_parquet(
    path: str | os.PathLike[str],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = PARQUET_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Stream a Parquet file one row dict at a time, in constant memory.

    The corpus-scale counterpart of :func:`read_parquet`, which materializes the
    whole table. Row order is the file's stored order, so a spine written in shard
    order is read back in shard order.
    """
    for batch in iter_parquet_batches(path, columns=columns, batch_size=batch_size):
        yield from batch


# --------------------------------------------------------------------------- #
# Row-range algebra over co-ordered tables.
#
# The corpus is a set of tables written in one global document order, and every
# corpus-scale consumer wants the same two things: read rows [a, b) of a table
# without scanning to the offset, and translate a row range of one table into the
# matching row range of a co-ordered table covering only some of the documents.
# They live here because `common.passage_store` needs them and `common` may not
# import `preprocess`; the `preprocess` names delegate onto these.
# --------------------------------------------------------------------------- #
#: Rows per batch when scanning one narrow id column to plan or align ranges. Larger
#: than the read default because such a scan touches a single column.
ID_SCAN_BATCH = 100_000


def parquet_row_group_span(
    path: str | os.PathLike[str], row_start: int, row_end: int
) -> tuple[list[int], int]:
    """``(row_group_indices, first_row_of_the_first_group)`` covering ``[start, end)``.

    Derived from cumulative :func:`parquet_row_group_sizes` rather than from
    ``row // PARQUET_ROW_GROUP_SIZE``, since row groups are not guaranteed uniform:
    a concatenated artifact's trailing group is short. Metadata only; the footer is
    read and no column data is touched.
    """
    sizes = parquet_row_group_sizes(path)
    groups: list[int] = []
    first_row = 0
    pos = 0
    for g, n in enumerate(sizes):
        lo, hi = pos, pos + n
        pos = hi
        if hi <= row_start or lo >= row_end:
            continue
        if not groups:
            first_row = lo
        groups.append(g)
    return groups, first_row


def iter_parquet_range(
    path: str | os.PathLike[str],
    row_start: int,
    row_end: int,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = PARQUET_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Stream rows ``[row_start, row_end)`` of ``path``, footer-indexed.

    Only the row groups overlapping the range are opened, and the leading and
    trailing rows of the boundary groups are dropped positionally. An empty or
    inverted range yields nothing.
    """
    if row_end <= row_start:
        return
    groups, first_row = parquet_row_group_span(path, row_start, row_end)
    if not groups:
        return
    pos = first_row
    for batch in iter_parquet_batches(
        path, columns=columns, batch_size=batch_size, row_groups=groups
    ):
        for row in batch:
            if row_start <= pos < row_end:
                yield row
            pos += 1
            if pos >= row_end:
                return


def _document_ids(path: str | os.PathLike[str]) -> Iterator[str]:
    """Stream a table's ``document_id`` column in file order; the ordinal is the row index."""
    for batch in iter_parquet_batches(
        path, columns=["document_id"], batch_size=ID_SCAN_BATCH
    ):
        for row in batch:
            yield row["document_id"]


def align_document_rows(
    parent_path: str | os.PathLike[str],
    subtable_path: str | os.PathLike[str],
    boundary_ordinals: Sequence[int],
) -> list[int]:
    """The first row of ``subtable_path`` at or after each parent-row ordinal.

    ``parent_path`` is a table holding every document exactly once in the global
    document order; ``subtable_path`` is a document-ordered table whose documents are
    a subsequence of the parent's, because an English document has no merge map and no
    translation row, a non-English one has no identity row, and a document may have no
    passages at all. A boundary document is therefore often absent from the sub-table.

    Both streams are in document order, so one forward co-walk assigns every sub-table
    row its document ordinal, and a boundary becomes the first sub-table row whose
    document ordinal is at least the boundary's. That is well defined whether or not
    the boundary document itself appears, and equals the sub-table's row count when no
    such row exists, which is the normal case for a trailing shard owning no rows here.

    ``boundary_ordinals`` must be non-decreasing, since the walk is forward-only. Two
    forward column scans, with no random access and no id-to-ordinal dictionary. A
    sub-table document missing from ``parent_path`` is a hard error: the tables are not
    from the same corpus build, and shifting a range would hand a reader another
    document's rows.
    """
    n = len(boundary_ordinals)
    if n == 0:
        return []
    for a, b in pairwise(boundary_ordinals):
        if b < a:
            raise ValueError(
                f"boundary ordinals must be non-decreasing (the co-walk is forward-only); "
                f"got {list(boundary_ordinals)}"
            )
    total = sum(parquet_row_group_sizes(subtable_path))
    starts: list[int] = []
    docs = _document_ids(parent_path)
    cur_doc: str | None = None
    cur_ord = -1
    k = 0
    i = 0
    for batch in iter_parquet_batches(
        subtable_path, columns=["document_id"], batch_size=ID_SCAN_BATCH
    ):
        if k >= n:
            break
        for row in batch:
            doc = row["document_id"]
            if doc != cur_doc:
                # Advance the document walk to this row's document (forward only).
                while True:
                    nxt = next(docs, None)
                    if nxt is None:
                        raise LookupError(
                            f"{subtable_path!s}: document {doc!r} is not in "
                            f"{parent_path!s} (or is out of order); the two tables are "
                            "not from the same corpus build"
                        )
                    cur_ord += 1
                    if nxt == doc:
                        break
                cur_doc = doc
                # This is the document's first row, so it opens every boundary this
                # document is at or past.
                while k < n and boundary_ordinals[k] <= cur_ord:
                    starts.append(i)
                    k += 1
            i += 1
            if k >= n:
                break
    while k < n:
        starts.append(total)  # trailing boundaries own no rows of this sub-table
        k += 1
    return starts


def write_parquet_stream(
    path: str | os.PathLike[str],
    records: Iterable[Mapping[str, Any]],
    *,
    schema: pa.Schema | None = None,
    skip_if_done: bool = True,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
) -> Path:
    """Stream row dicts into one Parquet file: constant memory, atomic, ``_SUCCESS``.

    The Parquet twin of :func:`concat_files`. ``records`` may be a generator over an
    arbitrarily large corpus; only ``row_group_size`` rows are held at once, then
    flushed through ``pq.ParquetWriter`` with the pinned
    :func:`_parquet_writer_options`. Durability and resume semantics match
    :func:`write_jsonl`.

    ``schema`` pins the column order and types, which is what makes the output
    byte-deterministic for a given record stream; pass it explicitly for any artifact
    another stage reads. Keys absent from the schema are dropped and schema fields
    absent from a record become null. When ``schema`` is None it is inferred from the
    first buffered chunk, which is adequate for small uniform tables.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if row_group_size < 1:
        raise ValueError(f"row_group_size must be >= 1, got {row_group_size}")
    p = Path(path)
    if skip_if_done and is_done(p):
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}")

    writer: Any = None
    buf: list[Mapping[str, Any]] = []

    def _flush() -> None:
        nonlocal writer, schema, buf
        table = pa.Table.from_pylist(buf, schema=schema)  # type: ignore[arg-type]
        _reject_nonfinite_table(table)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(str(tmp), schema, **_parquet_writer_options())
        writer.write_table(table, row_group_size=row_group_size)
        buf = []

    complete = False
    try:
        for rec in records:
            buf.append(rec)
            if len(buf) >= row_group_size:
                _flush()
        if buf or writer is None:
            _flush()  # trailing partial row group, or an empty artifact (header+footer)
        complete = True
    finally:
        if writer is not None:
            writer.close()
        if not complete:
            # A failed write leaves no half-file behind: the temp is never renamed and
            # no _SUCCESS exists, so the artifact stays missing and resumable.
            tmp.unlink(missing_ok=True)
    # fsync the payload before the rename, mirroring _atomic_write_bytes: the writer
    # only closes into the page cache, so without this a crash could leave a durable
    # _SUCCESS vouching for a truncated parquet.
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, p)
    _fsync_dir(p.parent)
    _write_success(p)
    return p


def write_parquet(
    path: str | os.PathLike[str],
    records: Sequence[dict[str, Any]],
    *,
    skip_if_done: bool = True,
) -> Path:
    """Atomically write a whole small table of row dicts to Parquet + ``_SUCCESS``.

    Thin wrapper over :func:`write_parquet_stream` with an inferred schema, so both
    paths share one set of pinned writer options and therefore the same bytes. Use
    :func:`write_parquet_stream` with an explicit schema for anything corpus-scale.
    No-op if already done and ``skip_if_done``.
    """
    return write_parquet_stream(
        path,
        list(records),
        skip_if_done=skip_if_done,
        row_group_size=max(len(records), 1),
    )


def concat_parquet(
    path: str | os.PathLike[str],
    sources: Sequence[str | os.PathLike[str]],
    *,
    schema: pa.Schema | None = None,
    skip_if_done: bool = True,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    batch_size: int = PARQUET_BATCH_SIZE,
) -> Path:
    """Concatenate Parquet shard files into one artifact, Arrow-native.

    The Parquet counterpart of :func:`concat_files`, which does not apply here
    because Parquet carries a binary footer, so a raw byte copy of two files is not
    a valid third file. Same atomic rename, dir-fsync, ``_SUCCESS`` and
    no-op-if-done contract; rows are appended in the given order through the pinned
    :func:`_parquet_writer_options`.

    Two properties are load-bearing:

    - Staying Arrow-native (``iter_batches`` into ``write_table``) is an order of
      magnitude faster than a ``to_pylist`` round-trip, which is why corpus-scale
      merges use this rather than :func:`write_parquet_stream` over row dicts.
    - The same rows arriving as eight shard files or five must produce the same
      checksum, so that a non-semantic knob like the shard count does not change the
      artifact hash. Batches are therefore buffered across source boundaries and
      flushed in exact ``row_group_size`` chunks, each combined into one contiguous
      chunk per column before writing.

    ``schema`` pins the output columns and types; when omitted the first source's
    schema is used and every later source must match it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if row_group_size < 1:
        raise ValueError(f"row_group_size must be >= 1, got {row_group_size}")
    p = Path(path)
    if skip_if_done and is_done(p):
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}")

    writer: Any = None
    out_schema = schema
    buf: list[Any] = []  # pending pa.RecordBatch, in order
    buf_rows = 0

    def _open(sch: pa.Schema) -> Any:
        return pq.ParquetWriter(str(tmp), sch, **_parquet_writer_options())

    def _flush(*, final: bool) -> None:
        """Write whole ``row_group_size`` groups out of the buffer (all of it if final)."""
        nonlocal buf, buf_rows
        while buf and (final or buf_rows >= row_group_size):
            table = pa.Table.from_batches(buf, schema=out_schema)
            n = min(row_group_size, table.num_rows)
            # One contiguous chunk per column, so the bytes depend only on the row
            # sequence and not on how the rows were split into shards.
            writer.write_table(
                table.slice(0, n).combine_chunks(), row_group_size=row_group_size
            )
            rest = table.slice(n)
            buf = rest.to_batches() if rest.num_rows else []
            buf_rows = rest.num_rows

    complete = False
    try:
        for src in sources:
            with pq.ParquetFile(str(src)) as pf:
                if out_schema is None:
                    out_schema = pf.schema_arrow
                if writer is None:
                    writer = _open(out_schema)
                for batch in pf.iter_batches(batch_size=batch_size):
                    buf.append(batch)
                    buf_rows += batch.num_rows
                    _flush(final=False)
        if writer is None:  # no sources at all: write an empty but valid artifact
            if out_schema is None:
                raise ValueError(
                    "concat_parquet: no sources and no schema to write an empty file"
                )
            writer = _open(out_schema)
        _flush(final=True)
        complete = True
    finally:
        if writer is not None:
            writer.close()
        if not complete:
            tmp.unlink(missing_ok=True)
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, p)
    _fsync_dir(p.parent)
    _write_success(p)
    return p


# --------------------------------------------------------------------------- #
# LMDB
# --------------------------------------------------------------------------- #
def lmdb_open(
    path: str | os.PathLike[str],
    *,
    readonly: bool = False,
    map_size: int = DEFAULT_MAP_SIZE,
    lock: bool | None = None,
):
    """Open (or create) an LMDB environment at ``path``.

    Query-time callers pass ``readonly=True``, after which ``lock`` defaults to
    ``False``. That is safe because the passage store is immutable once built, and
    lock files are unreliable on Lustre and NFS. A writable open creates the
    directory and uses LMDB's default locking.
    """
    import lmdb

    p = Path(path)
    if lock is None:
        lock = not readonly
    if not readonly:
        p.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(p),
        map_size=map_size,
        readonly=readonly,
        lock=lock,
        subdir=True,
        create=not readonly,
        max_dbs=0,
    )
