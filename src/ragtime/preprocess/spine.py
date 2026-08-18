"""Document-atomic row-range sharding over the co-ordered corpus tables.

The corpus is stored as tables written in one document order: ``documents.parquet`` (one
row per document), ``sentences.parquet`` (one row per sentence) and the merge map (one row
per non-English sentence). Every post-chunk substage shards them the same way:

1. Plan contiguous, document-atomic row ranges over a work table (:func:`plan_shards`). A
   shard owns whole documents, never a document's tail, because the downstream rules
   (merge direction, unit membership) are per-document and a seam would group the two
   halves inconsistently.
2. Align each range to ``documents.parquet``'s own row range (:func:`align_documents`), so
   a worker reads only its own documents' text instead of streaming a multi-GB table to its
   offset. The work table's boundary document ids are present in ``documents.parquet`` and
   in the same order, so one forward pass over that file's ``document_id`` column resolves
   every boundary exactly.
3. Read a shard's rows back (:func:`iter_row_range`) through Parquet's footer index, so
   nothing is pre-split onto disk.

Both planning passes project a single column and run once per build, in ``saturate.seed``;
they write nothing.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ragtime.common.io import (
    PARQUET_BATCH_SIZE,
    iter_parquet_batches,
    iter_parquet_range,
    parquet_row_group_sizes,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

__all__ = [
    "ShardRange",
    "align_documents",
    "group_by_document",
    "iter_row_range",
    "load_document_texts",
    "plan_shards",
]

# Rows per batch when scanning a single id column to plan boundaries; larger than the read
# default because the scan touches one narrow column and runs once per build.
_SCAN_BATCH = 100_000


@dataclass(frozen=True, slots=True)
class ShardRange:
    """One claimable unit: a document-atomic row range over a work table.

    ``row_start``/``row_end`` are absolute, half-open row indices into the work table;
    ``doc_row_start``/``doc_row_end`` the matching half-open range in
    ``documents.parquet``. The pair makes a worker's read self-contained: it never scans
    either file from row 0, and it holds only its own documents' text.
    """

    row_start: int
    row_end: int
    doc_row_start: int = 0
    doc_row_end: int = 0

    @property
    def name(self) -> str:
        """``rows_<start:012d>_<end:012d>``, whose lexicographic order is work-table row order.

        The shard-name sort restores global row order at ``merge`` time, so the zero-padding
        width and the numeric suffix are part of the contract.
        """
        return f"rows_{self.row_start:012d}_{self.row_end:012d}"

    def payload(self) -> dict[str, int]:
        """The JSON payload ``work`` reads back (opaque to the driver)."""
        return {
            "row_start": self.row_start,
            "row_end": self.row_end,
            "doc_row_start": self.doc_row_start,
            "doc_row_end": self.doc_row_end,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ShardRange:
        """Rebuild a range from its payload (the ``work``-side counterpart)."""
        return cls(
            row_start=int(payload["row_start"]),
            row_end=int(payload["row_end"]),
            doc_row_start=int(payload["doc_row_start"]),
            doc_row_end=int(payload["doc_row_end"]),
        )


def plan_shards(
    work_path: str | os.PathLike[str], *, n_shards: int
) -> tuple[list[ShardRange], list[str]]:
    """Cut ``work_path`` into ~``n_shards`` contiguous, document-atomic row ranges.

    Returns ``(ranges, boundary_documents)``: the ranges with their document columns still
    unset (``align_documents`` fills them), and the ``document_id`` that opens each range,
    in order. It makes one column-projected pass over ``document_id``.

    A boundary is proposed every ``ceil(total / n_shards)`` rows and then snapped outward to
    the next document change, so a document is never split. An empty table yields no ranges.
    """
    ranges: list[ShardRange] = []
    boundary_docs: list[str] = []
    total = sum(parquet_row_group_sizes(work_path))
    if total == 0:
        return ranges, boundary_docs
    per = max(1, math.ceil(total / max(1, n_shards)))

    start = 0
    start_doc: str | None = None
    target = per
    prev: str | None = None
    i = 0
    for batch in iter_parquet_batches(
        work_path, columns=["document_id"], batch_size=_SCAN_BATCH
    ):
        for row in batch:
            doc = row["document_id"]
            if start_doc is None:
                start_doc = doc
            if i >= target and prev is not None and doc != prev:
                ranges.append(ShardRange(row_start=start, row_end=i))
                boundary_docs.append(start_doc)
                start, start_doc, target = i, doc, i + per
            prev = doc
            i += 1
    ranges.append(ShardRange(row_start=start, row_end=total))
    boundary_docs.append(start_doc if start_doc is not None else "")
    return ranges, boundary_docs


def align_documents(
    documents_path: str | os.PathLike[str],
    ranges: Sequence[ShardRange],
    boundary_docs: Sequence[str],
) -> list[ShardRange]:
    """Fill each range's ``documents.parquet`` row range in one forward column scan.

    ``documents.parquet`` holds every document exactly once, in the same order the work
    table's documents appear, so each range's opening ``document_id`` is found and the
    matches come out in order. A boundary that is not found is an error: it means the two
    tables are no longer from the same build, and shifting a range would hand a worker
    another document's text.
    """
    if not ranges:
        return []
    wanted = list(boundary_docs)
    total_docs = sum(parquet_row_group_sizes(documents_path))
    starts: list[int] = []
    k = 0
    i = 0
    for batch in iter_parquet_batches(
        documents_path, columns=["document_id"], batch_size=_SCAN_BATCH
    ):
        for row in batch:
            if k < len(wanted) and row["document_id"] == wanted[k]:
                starts.append(i)
                k += 1
            i += 1
        if k == len(wanted):
            break
    if k != len(wanted):
        raise LookupError(
            f"{documents_path!s}: shard boundary document {wanted[k]!r} not found: the "
            "work table and documents.parquet are not from the same corpus build"
        )
    ends = [*starts[1:], total_docs]
    return [
        ShardRange(
            row_start=r.row_start, row_end=r.row_end, doc_row_start=a, doc_row_end=b
        )
        for r, a, b in zip(ranges, starts, ends, strict=True)
    ]


def iter_row_range(
    path: str | os.PathLike[str],
    row_start: int,
    row_end: int,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = PARQUET_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Stream rows ``[row_start, row_end)`` of ``path``, footer-indexed and constant memory.

    A delegate to :func:`~ragtime.common.io.iter_parquet_range`: the footer arithmetic lives
    in ``common.io`` because ``common.passage_store`` needs it too and ``common`` may not
    import ``preprocess``.
    """
    yield from iter_parquet_range(
        path, row_start, row_end, columns=columns, batch_size=batch_size
    )


def load_document_texts(
    documents_path: str | os.PathLike[str], shard: ShardRange
) -> dict[str, str]:
    """Return ``{document_id: text}`` for exactly this shard's documents.

    Bounded by the shard rather than the corpus (roughly ``total_docs / n_shards``
    documents), which is why a range is aligned onto ``documents.parquet`` instead of the
    file being scanned per document.
    """
    return {
        row["document_id"]: row["text"]
        for row in iter_row_range(
            documents_path,
            shard.doc_row_start,
            shard.doc_row_end,
            columns=["document_id", "text"],
        )
    }


def group_by_document(
    rows: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[str, list[Mapping[str, Any]]]]:
    """Group a document-ordered row stream into ``(document_id, rows)`` in constant memory.

    Relies on the tables' contract that a document's rows are contiguous, so no buffering
    beyond the current document is needed.
    """
    cur_id: str | None = None
    cur: list[Mapping[str, Any]] = []
    for row in rows:
        did = row["document_id"]
        if did != cur_id:
            if cur_id is not None:
                yield cur_id, cur
            cur_id, cur = did, []
        cur.append(row)
    if cur_id is not None:
        yield cur_id, cur
