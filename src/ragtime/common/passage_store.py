"""The three-rendering passage store: by-id text over original, omt and omt_opus.

A passage has no stored text. On disk it is an ordered list of final ``sentence_id``s
(``final/<recon12>/passages.parquet``), and its text is derived from the spine every
time it is built:

- ``documents.parquet``                              ``document_id -> text``
- ``final/<recon12>/sentences.parquet``              ``sentence_id -> (document_id, start, end)``
- ``final/<recon12>/translations/<variant>.parquet`` ``sentence_id -> English text``,
  for non-English sentences only

English has no translation, and the artifact says so. Under
``reconcile.store_identity_translations: false`` the translation tables carry no row for
an English sentence at all: an English passage's ``omt`` and ``omt_opus`` text is its
``original`` text, resolved from the native span here rather than read back from a stored
identity copy. Two consequences follow, both structural rather than conventional:

1. English is byte-identical across all three renderings by construction. It was not
   under stored identity rows, where ``original`` was the single contiguous document
   slice while ``omt`` was a space join over the segments, so any English passage whose
   source separated two sentences by something other than one space rendered differently.
2. A stored English translation that is not its own source span is unconstructible, since
   the row it would live in does not exist. What is checked here instead is the converse:
   a sentence with no translation row must be English (:data:`IDENTITY_LANG`), and a
   non-English passage with no rows raises rather than silently falling back to its
   native, untranslated text.

Two renderings, two different composition rules (:func:`compose_passage_text`):

- ``original`` is one contiguous slice ``documents.text[first.start : last.end]`` via
  :func:`schemas.passage_native_text`, never a space join over the members. A join
  fabricates a separator the source never had, which for CJK text is the common case,
  and that separator would land both in the text the retriever encodes and in the
  verbatim NFC span-commit target, so a claim quoting across a sentence boundary would
  fail to ground.
- ``omt`` and ``omt_opus`` are a space join over the member sentences' translations, for
  every source language, because those segments are English.

This module is the one home of both rules; ``preprocess.index``'s encode step calls
:func:`compose_passage_text` rather than re-deriving either.

Two store implementations behind one interface:

- :class:`PassageStore`, an in-memory dict for fixtures, tests and bounded slices.
- :class:`LmdbPassageStore`, the production memory-mapped by-id store, built off the
  final tables by :meth:`LmdbPassageStore.build_from_final`. Query-time opens are
  ``readonly=True, lock=False``, safe because the store is immutable once built, and a
  store opened twice in one process yields two handles over one reference-counted
  environment: LMDB forbids a second environment on the same path, and two would be two
  mappings that can disagree, so the second holder shares rather than opens.

Both expose ``passage(passage_id) -> Passage`` and ``render(passage_id, passage_lang)
-> str``. ``render`` is total over the three renderings for any stored id: ``original``
is always the native passage text, and an ``omt`` or ``omt_opus`` rendering that was not
stored resolves to ``original``, so a lookup never raises for a valid rendering name.
``passage_lang`` selects the reading, decoupled from the searched index; the doc-id for
citation is always recovered via ``ids.doc_id_of``.

Corpus scale is a hard constraint: the build streams all four tables with
``io.iter_parquet`` and an explicit column list, and joins them as a forward-only co-walk
in document order, so peak memory is one document's text plus that document's own
sentence and translation rows. Nothing ever materializes the multi-million-row passage or
sentence tables, or the documents table's text column.

A slice of the corpus should cost a slice rather than the corpus (:class:`FinalRange`
with :func:`plan_final_ranges`). Constant memory is not the same property as bounded
work: without a range, :func:`iter_final_passages` opens all five tables at row 0 and
composes every passage of every document, so a caller wanting one shard's worth filtered
after composition and paid for the whole corpus each time, which made finer sharding
progressively worse. A :class:`FinalRange` names one ``documents.parquet`` row range plus
the row range each co-ordered table holds for exactly those documents, resolved once by a
forward co-walk and then read through Parquet's footer index, so a ranged stream touches
only its own row groups and composes the same bytes the un-ranged stream yields for those
documents.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

from . import io as _io
from .layout import Layout
from .schemas import Passage, passage_native_text

__all__ = [
    "IDENTITY_LANG",
    "PASSAGES_TABLE",
    "RENDERINGS",
    "SENTENCES_TABLE",
    "FinalRange",
    "LmdbPassageStore",
    "PassageCompositionError",
    "PassageStore",
    "compose_passage_text",
    "iter_final_passages",
    "plan_final_ranges",
    "translations_table",
]

# The three renderings. The translation axis is the MT tier: `omt` uses NLLB-200-3.3B and
# `omt_opus` uses OPUS-MT. This tuple must equal `config.schema.KNOB_VALUES`, which cannot
# be imported here because `common` sits below `config` on the dependency spine; the
# agreement is asserted by a test in tests/config instead.
RENDERINGS: tuple[str, ...] = ("original", "omt", "omt_opus")

#: The corpus language that is the MT target, and therefore the one language for which a
#: translation would be an identity copy of its own source span. Named once here, beside
#: :data:`RENDERINGS`, because three modules branch on it and a fourth bare ``"en"``
#: literal would be a place for them to disagree.
IDENTITY_LANG = "en"

# The single space that joins translated (English) segments; never applied to native text.
_JOINER = " "

# Columns each final table is read with. Naming them keeps the read pruned and makes it
# structurally impossible to pull a text column the text-free passage table does not have.
_PASSAGE_COLUMNS = (
    "passage_id",
    "document_id",
    "lang",
    "sentence_ids",
    "token_count",
    "is_oversized",
    "paragraph_index",
)
_SENTENCE_COLUMNS = ("sentence_id", "document_id", "start", "end")
_TRANSLATION_COLUMNS = ("sentence_id", "document_id", "text")
_DOCUMENT_COLUMNS = ("document_id", "text")

# `documents.parquet` is the one table whose rows carry corpus text, so it is read in far
# smaller batches than the id and offset tables: a 10k-row batch of documents would hold
# megabytes of text for a value used one row at a time.
_DOCUMENT_BATCH = 256

#: Table names a :class:`FinalRange` carries a row range for. These are the stream's own
#: names rather than filenames, because ``translations/<variant>`` is one table per
#: rendering and which renderings a caller asked for is part of the range's contract.
SENTENCES_TABLE = "sentences"
PASSAGES_TABLE = "passages"


def translations_table(variant: str) -> str:
    """The :class:`FinalRange` table name of one rendering's translation table."""
    return f"translations/{variant}"


class PassageCompositionError(ValueError):
    """A passage's text cannot be composed from the final tables.

    Always a build-coherence failure rather than a recoverable condition: a member sentence
    missing from the inventory, a translation table covering only part of a passage, or
    leftover rows proving the four tables are not from the same corpus build. It is raised
    rather than absorbed because ``render``'s fallback to the original exists for the
    English identity pass-through only, and applying it to a broken build would hide the
    defect.
    """


def _check_rendering(passage_lang: str) -> None:
    if passage_lang not in RENDERINGS:
        raise ValueError(
            f"unknown rendering {passage_lang!r}; expected one of {RENDERINGS}"
        )


# --------------------------------------------------------------------------- #
# The composition rule: native slice against translated join.
# --------------------------------------------------------------------------- #
def compose_passage_text(
    document_text: str,
    spans: Sequence[tuple[int, int]],
    translations: Sequence[str] | None,
    passage_lang: str,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Compose a passage's text in ``passage_lang`` plus its per-sentence spans over it.

    ``spans`` are the member sentences' ``(start, end)`` offsets into ``document_text``, in
    passage order: the contiguous run ``final/passages.sentence_ids`` names.

    - ``passage_lang == "original"``: the text is :func:`schemas.passage_native_text`, the
      single slice ``document_text[spans[0][0] : spans[-1][1]]``, and the returned spans
      are the input spans rebased onto it, so ``text[a:b]`` is that member sentence
      verbatim.
    - ``passage_lang in {"omt", "omt_opus"}``: the text is the translations joined by a
      space, one segment per span in the same order, with ``document_text`` unused, and the
      returned spans are cumulative over the join. ``translations`` is required here and
      must be the same length as ``spans``.

    Returning the spans alongside the text keeps sentence-grain consumers from re-segmenting
    composed text, which does not reproduce the chunker's boundaries. The native pass's spans
    are persisted on each passage record as ``sentence_char_spans``.
    """
    _check_rendering(passage_lang)
    pairs = [(int(a), int(b)) for a, b in spans]
    if passage_lang == "original":
        _check_contiguous(pairs)
        text = passage_native_text(document_text, pairs)
        base = pairs[0][0] if pairs else 0
        return text, tuple((a - base, b - base) for a, b in pairs)
    if translations is None:
        raise PassageCompositionError(
            f"rendering {passage_lang!r} needs the member sentences' translations; got None"
        )
    segments = [str(t) for t in translations]
    if len(segments) != len(pairs):
        raise PassageCompositionError(
            f"rendering {passage_lang!r}: {len(segments)} translated segment(s) for "
            f"{len(pairs)} member sentence(s); the two must correspond 1:1, in order"
        )
    out: list[tuple[int, int]] = []
    offset = 0
    for segment in segments:
        out.append((offset, offset + len(segment)))
        offset += len(segment) + len(_JOINER)
    return _JOINER.join(segments), tuple(out)


def _check_contiguous(spans: Sequence[tuple[int, int]]) -> None:
    """A passage's spans must be an ordered, non-overlapping run of its document.

    The single-slice rule is correct only for such a run, so a violation is a hard error
    rather than a silently wrong slice.
    """
    prev_end: int | None = None
    for start, end in spans:
        if end < start or (prev_end is not None and start < prev_end):
            raise PassageCompositionError(
                f"passage spans are not an ordered contiguous run: {list(spans)}"
            )
        prev_end = end


# --------------------------------------------------------------------------- #
# Streaming the final tables: a forward-only co-walk in document order.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _DocCursor:
    """One document-ordered row stream, taken a ``document_id`` at a time.

    All four final tables are document-ordered subsequences of the one global document
    order, so walking them together behind a single peeked head is a merge join in constant
    memory rather than a hash join over tens of millions of keys. :meth:`take` returns an
    empty list for a document this table has no rows for, which is normal rather than an
    error; :meth:`exhausted` is checked once at the end so leftover rows still fail loudly.

    Private and minimal. ``preprocess.spine.group_by_document`` and
    ``preprocess.reconcile._DocRows`` are the same primitive, but ``preprocess`` sits above
    ``common`` on the dependency spine, so importing either here would invert it. A third
    copy would be the signal to move the primitive down into ``common.io``.

    :class:`FinalRange` needs no new cursor behaviour: a planned sub-range starts at the
    first row of the range's first document and ends where the range's last document's rows
    end, so ``take`` and ``exhausted`` mean the same thing over a range as over the whole
    table.
    """

    _rows: Iterator[Mapping[str, Any]]
    _head: Mapping[str, Any] | None = None

    @classmethod
    def over(cls, rows: Iterable[Mapping[str, Any]]) -> _DocCursor:
        return cls(_rows=iter(rows))

    def take(self, document_id: str) -> list[Mapping[str, Any]]:
        """This table's contiguous rows for ``document_id``, consuming them."""
        out: list[Mapping[str, Any]] = []
        while True:
            if self._head is None:
                self._head = next(self._rows, None)
                if self._head is None:
                    return out
            if self._head["document_id"] != document_id:
                return out
            out.append(self._head)
            self._head = None

    def exhausted(self) -> bool:
        """True when every row has been taken (a leftover means the tables disagree)."""
        if self._head is None:
            self._head = next(self._rows, None)
        return self._head is None


# --------------------------------------------------------------------------- #
# Slicing the co-walk to a document row range: the composition pushdown.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FinalRange:
    """A contiguous slice of ``documents.parquet`` plus every other table's own slice.

    :func:`iter_final_passages` co-walks five tables. Without a range it opens all five at
    row 0 and composes every passage of every document, so a caller wanting one slice of the
    corpus paid for the whole of it, which was the dominant fixed cost of a sharded index
    build and made finer sharding progressively worse. This type is the pushdown: it names
    one document row range and, for each co-ordered table, the row range holding exactly
    that range's documents.

    The sub-ranges cannot be derived from ``doc_row_start`` and ``doc_row_end`` by
    arithmetic: a document has a variable number of sentences and passages, and English
    documents have no rows at all in the translation tables. They are resolved by a forward
    co-walk in :func:`plan_final_ranges`, which is the only sanctioned way to build one,
    because a hand-written range would silently mis-compose a passage.

    ``payload`` and ``from_payload`` round-trip it through a work-queue shard file, as
    ``preprocess.spine.ShardRange`` does: resolve once at seed, read row groups per worker.
    """

    doc_row_start: int
    doc_row_end: int
    table_rows: Mapping[str, tuple[int, int]]

    def rows_for(self, table: str) -> tuple[int, int]:
        """This range's ``[start, end)`` rows of ``table`` (raises if it was not planned)."""
        try:
            return self.table_rows[table]
        except KeyError:
            raise PassageCompositionError(
                f"this document range carries no row range for {table!r} (it names "
                f"{sorted(self.table_rows)}); it was planned for a different set of "
                "renderings than the stream is asking for"
            ) from None

    def payload(self) -> dict[str, Any]:
        """The JSON payload a work-queue shard carries (opaque to the driver)."""
        return {
            "doc_row_start": self.doc_row_start,
            "doc_row_end": self.doc_row_end,
            "table_rows": {k: list(v) for k, v in self.table_rows.items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FinalRange:
        """Rebuild a range from its payload (the worker-side counterpart)."""
        rows = dict(payload.get("table_rows") or {})
        return cls(
            doc_row_start=int(payload["doc_row_start"]),
            doc_row_end=int(payload["doc_row_end"]),
            table_rows={str(k): (int(v[0]), int(v[1])) for k, v in rows.items()},
        )


def _final_table_paths(
    layout: Layout, reconcile_hash: str, pack_hash: str | None, renderings: Sequence[str]
) -> dict[str, Path]:
    """The four co-ordered tables this stream joins onto ``documents.parquet``.

    Resolved in one place so the planner and the reader cannot disagree about which file a
    table name means, which would produce a valid-looking range over the wrong table.
    """
    paths = {
        SENTENCES_TABLE: layout.final_sentences_path(reconcile_hash),
        PASSAGES_TABLE: layout.final_passages_path(reconcile_hash, pack_hash),
    }
    for variant in renderings:
        if variant != "original":
            paths[translations_table(variant)] = layout.final_translations_path(
                reconcile_hash, variant
            )
    return paths


def plan_final_ranges(
    layout: Layout,
    reconcile_hash: str,
    *,
    pack_hash: str | None,
    document_ranges: Sequence[tuple[int, int]],
    renderings: Sequence[str] = RENDERINGS,
) -> list[FinalRange]:
    """Resolve each ``documents.parquet`` row range onto all four co-ordered tables.

    ``document_ranges`` are half-open ``[start, end)`` row ranges over
    ``documents.parquet``, non-decreasing and non-overlapping, which is the shape a
    document-atomic shard plan has. Returns one :class:`FinalRange` per input range, in the
    same order, each carrying the sub-ranges :func:`iter_final_passages` needs.

    This does one forward pass over each table's ``document_id`` column for all ranges
    together, the same plan-once-at-seed and read-row-groups-per-worker split
    ``preprocess.len_max`` and ``preprocess.reconcile`` use, which is why the sub-ranges
    travel in the shard payload rather than being re-derived per worker. Re-deriving them
    per range would put a whole-corpus column scan back in front of every read.

    Empty sub-ranges are normal rather than a failure: an all-English range owns no
    translation rows, and a document with no passages contributes none.
    """
    for name in renderings:
        _check_rendering(name)
    ranges = [(int(a), int(b)) for a, b in document_ranges]
    if not ranges:
        return []
    for start, end in ranges:
        if end < start:
            raise ValueError(f"document range {(start, end)} is inverted")
    for (_, prev_end), (start, _) in pairwise(ranges):
        if start < prev_end:
            raise ValueError(
                f"document ranges must be non-decreasing and non-overlapping; "
                f"{prev_end} then {start}"
            )
    # One co-walk per table over the sorted unique edges, then pair them back up: a range's
    # start and its end ask the same question (the first sub-table row at or after this
    # document ordinal), so both are answered by one forward pass.
    edges = sorted({e for r in ranges for e in r})
    table_edges: dict[str, dict[int, int]] = {}
    for table, path in _final_table_paths(
        layout, reconcile_hash, pack_hash, renderings
    ).items():
        starts = _io.align_document_rows(layout.documents_path(), path, edges)
        table_edges[table] = dict(zip(edges, starts, strict=True))
    return [
        FinalRange(
            doc_row_start=start,
            doc_row_end=end,
            table_rows={t: (e[start], e[end]) for t, e in table_edges.items()},
        )
        for start, end in ranges
    ]


def iter_final_passages(
    layout: Layout,
    reconcile_hash: str,
    *,
    pack_hash: str | None,
    renderings: Sequence[str] = RENDERINGS,
    document_range: FinalRange | None = None,
    document_batch_size: int = _DOCUMENT_BATCH,
    row_batch_size: int = _io.PARQUET_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Stream every passage of ``final/<recon12>/`` with all its renderings composed.

    Yields records in :meth:`PassageStore.from_records` shape (``passage_id`` /
    ``document_id`` / ``lang`` / ``original`` / ``omt`` / ``omt_opus`` / ``sentence_ids`` /
    ``token_count`` / ``is_oversized`` / ``paragraph_index`` / ``sentence_char_spans``), in
    ``passages.parquet`` row order, so one composition stream feeds the in-memory store, the
    LMDB build and the index encode step alike.

    All paths come from ``layout``, which must be family and chunker-hash bound since
    ``documents_path()`` carries neither argument. Every table is read with
    ``io.iter_parquet`` and an explicit column list, and joined by the :class:`_DocCursor`
    co-walk, so memory is proportional to one document rather than to the corpus.

    ``pack_hash`` is required and selects which packing of the inventory to read
    (``final/<recon12>/passages/<pack12>/passages.parquet``). ``recon12`` names the sentences
    and ``pack12`` names their grouping, and both are needed to name a passage, since two
    packings of one inventory legitimately coexist under the same ``final/<recon12>/``.
    ``None`` is legal and resolves the pre-``packing`` layout, in which the file sits directly
    under ``final/<recon12>/``. No production caller passes it and the shipped corpus does not
    carry such a file -- its passages are at ``passages/<pack12>/`` -- so this branch is
    reached only by the tests that build a store without a packing stage.

    ``document_range`` (a :class:`FinalRange` from :func:`plan_final_ranges`) restricts the
    stream to one contiguous slice of ``documents.parquet``, reading only the row groups that
    slice touches in all five tables through the footer index, so the read is proportional to
    the range rather than to the corpus. The composed records are byte-identical to the
    sub-sequence the un-ranged stream yields for those documents: the range changes only
    which rows reach the co-walk. ``None``, the default, is the whole corpus.
    """
    for name in renderings:
        _check_rendering(name)
    variants = [name for name in renderings if name != "original"]

    def rows(table: str, path: Path, columns: Sequence[str]) -> Iterator[Mapping[str, Any]]:
        """This table's rows: the whole table, or the range's, pruned by the footer."""
        if document_range is None:
            return _io.iter_parquet(path, columns=columns, batch_size=row_batch_size)
        start, end = document_range.rows_for(table)
        return _io.iter_parquet_range(
            path, start, end, columns=columns, batch_size=row_batch_size
        )

    paths = _final_table_paths(layout, reconcile_hash, pack_hash, renderings)
    sentences = _DocCursor.over(
        rows(SENTENCES_TABLE, paths[SENTENCES_TABLE], _SENTENCE_COLUMNS)
    )
    passages = _DocCursor.over(rows(PASSAGES_TABLE, paths[PASSAGES_TABLE], _PASSAGE_COLUMNS))
    # One cursor per translated variant. Each table covers only the non-English documents,
    # which the co-walk already handles: `take` returns an empty list for a document a table
    # has no rows for, and `_translated` turns that into native text only for English.
    translations = {
        variant: _DocCursor.over(
            rows(
                translations_table(variant),
                paths[translations_table(variant)],
                _TRANSLATION_COLUMNS,
            )
        )
        for variant in variants
    }

    documents_path = layout.documents_path()
    if document_range is None:
        documents: Iterable[Mapping[str, Any]] = _io.iter_parquet(
            documents_path, columns=_DOCUMENT_COLUMNS, batch_size=document_batch_size
        )
    else:
        documents = _io.iter_parquet_range(
            documents_path,
            document_range.doc_row_start,
            document_range.doc_row_end,
            columns=_DOCUMENT_COLUMNS,
            batch_size=document_batch_size,
        )

    for document in documents:
        document_id = document["document_id"]
        # Every cursor is advanced for every document, even one with no passages: a skipped
        # take would leave a stale head and starve every later document.
        passage_rows = passages.take(document_id)
        spans = {
            row["sentence_id"]: (int(row["start"]), int(row["end"]))
            for row in sentences.take(document_id)
        }
        variant_text = {
            variant: {row["sentence_id"]: row["text"] for row in cursor.take(document_id)}
            for variant, cursor in translations.items()
        }
        for row in passage_rows:
            yield _compose_record(row, document["text"], spans, variant_text)

    # Leftovers fail loudly, and a range does not change the check: a planned sub-range ends
    # exactly where the range's last document's rows end, so "exhausted" means the same thing
    # in both modes. Only the message differs, between a table not from this corpus build and
    # a range that does not describe the documents it was read beside.
    for label, cursor in (
        (PASSAGES_TABLE, passages),
        (SENTENCES_TABLE, sentences),
        *((translations_table(v), c) for v, c in translations.items()),
    ):
        if not cursor.exhausted():
            raise PassageCompositionError(
                f"{label}: rows remain for document(s) absent from documents.parquet (or "
                "out of document order); the tables are not from the same corpus build"
                if document_range is None
                else f"{label}: rows remain after this document range's documents were "
                f"consumed; its row range {document_range.rows_for(label)} does not "
                f"describe documents [{document_range.doc_row_start}, "
                f"{document_range.doc_row_end}) of this build"
            )


def _compose_record(
    row: Mapping[str, Any],
    document_text: str,
    spans: Mapping[str, tuple[int, int]],
    variant_text: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Turn one text-free passage row plus the spine into a fully rendered store record."""
    passage_id = row["passage_id"]
    sentence_ids = tuple(row["sentence_ids"] or ())
    member_spans: list[tuple[int, int]] = []
    for sentence_id in sentence_ids:
        try:
            member_spans.append(spans[sentence_id])
        except KeyError:
            raise PassageCompositionError(
                f"{passage_id}: member sentence {sentence_id!r} is not in the final "
                f"sentence inventory of {row['document_id']!r}; the tables are not from "
                "the same corpus build"
            ) from None
    original, native_spans = compose_passage_text(document_text, member_spans, None, "original")
    record: dict[str, Any] = {
        "passage_id": passage_id,
        "document_id": row["document_id"],
        "lang": row["lang"],
        "original": original,
        "sentence_ids": list(sentence_ids),
        "token_count": int(row["token_count"] or 0),
        "is_oversized": bool(row["is_oversized"]),
        "paragraph_index": [int(i) for i in (row["paragraph_index"] or ())],
        "sentence_char_spans": [list(pair) for pair in native_spans],
    }
    for variant, table in variant_text.items():
        record[variant] = _translated(
            passage_id,
            record["lang"],
            sentence_ids,
            member_spans,
            table,
            variant,
            original,
            document_text,
        )
    return record


def _translated(
    passage_id: str,
    lang: str,
    sentence_ids: Sequence[str],
    member_spans: Sequence[tuple[int, int]],
    table: Mapping[str, str],
    variant: str,
    original: str,
    document_text: str,
) -> str:
    """This passage's ``variant`` text, or ``original`` when and only when it is English.

    The empty-table case is the English identity pass-through, checked against the passage's
    language rather than inferred from the absence of rows. Since English rows stopped being
    stored, absence is the normal state for a large share of the corpus, so an unguarded
    fallback would serve untranslated source text as a translated rendering the moment a
    translation table were truncated or mis-aligned. A partially covered passage raises for
    the same reason.
    """
    if not table:
        if lang != IDENTITY_LANG:
            raise PassageCompositionError(
                f"{passage_id}: translations/{variant} has no rows for this {lang!r} passage. "
                f"Only {IDENTITY_LANG!r} resolves to its native text (English is not "
                "translated, so no row is stored for it); every other language must be "
                "covered, and falling back here would serve untranslated source text as a "
                "translated rendering"
            )
        return original
    missing = [sid for sid in sentence_ids if sid not in table]
    if missing:
        raise PassageCompositionError(
            f"{passage_id}: translations/{variant} covers {len(sentence_ids) - len(missing)} "
            f"of {len(sentence_ids)} member sentence(s) (first missing: {missing[0]!r}); a "
            "partial rendering must never fall back to the original"
        )
    text, _ = compose_passage_text(
        document_text, member_spans, [table[sid] for sid in sentence_ids], variant
    )
    return text


# --------------------------------------------------------------------------- #
# Record to Passage and LMDB blob: one rule, shared by both stores.
# --------------------------------------------------------------------------- #
def _spans(value: Any) -> tuple[tuple[int, int], ...]:
    """Normalize per-sentence char spans to a tuple of ``(start, end)`` int pairs.

    JSON round-trips them as lists, so decode restores the tuple shape
    :class:`Passage` declares (offsets into the passage's own ``text``).
    """
    return tuple((int(s), int(e)) for s, e in value or ())


def _passage_from_record(record: Mapping[str, Any]) -> Passage:
    """The :class:`Passage` a store record describes; ``text`` is its native rendering."""
    return Passage(
        passage_id=record["passage_id"],
        document_id=record["document_id"],
        lang=record["lang"],
        text=record["original"],
        sentence_ids=tuple(record.get("sentence_ids", ())),
        token_count=record.get("token_count", 0),
        is_oversized=record.get("is_oversized", False),
        paragraph_index=tuple(record.get("paragraph_index", ())),
        sentence_char_spans=_spans(record.get("sentence_char_spans")),
    )


def _renderings_from_record(record: Mapping[str, Any]) -> dict[str, str]:
    """The record's per-rendering texts; an absent translation falls back to ``original``."""
    original = record["original"]
    rendered = {"original": original}
    for variant in RENDERINGS:
        if variant != "original":
            rendered[variant] = record.get(variant) or original
    return rendered


def _encode(passage: Passage, renderings: Mapping[str, str]) -> bytes:
    """Serialize a passage + its renderings to a compact JSON blob for LMDB."""
    blob = {
        "document_id": passage.document_id,
        "lang": passage.lang,
        "text": passage.text,
        "sentence_ids": list(passage.sentence_ids),
        "token_count": passage.token_count,
        "is_oversized": passage.is_oversized,
        "paragraph_index": list(passage.paragraph_index),
        # one [start, end] per sentence_id, offsets into "text" (JSON has no tuples)
        "sentence_char_spans": [list(s) for s in passage.sentence_char_spans],
        "omt": renderings.get("omt", passage.text),
        "omt_opus": renderings.get("omt_opus", passage.text),
    }
    return json.dumps(blob, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _decode_passage(passage_id: str, blob: dict[str, Any]) -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id=blob["document_id"],
        lang=blob["lang"],
        text=blob["text"],
        sentence_ids=tuple(blob.get("sentence_ids", ())),
        token_count=blob.get("token_count", 0),
        is_oversized=blob.get("is_oversized", False),
        paragraph_index=tuple(blob.get("paragraph_index", ())),
        sentence_char_spans=_spans(blob.get("sentence_char_spans")),
    )


def _rendering_from_blob(blob: dict[str, Any], passage_lang: str) -> str:
    if passage_lang == "original":
        return blob["text"]
    # omt / omt_opus fall back to the original text (English identity passthrough).
    return blob.get(passage_lang) or blob["text"]


class PassageStore:
    """In-memory by-id passage store, for fixtures, tests and bounded slices.

    Build directly from ``Passage`` objects plus optional per-id rendering maps, or via
    :meth:`from_records`, which :func:`iter_final_passages` feeds directly. It materializes
    everything it is given, so the corpus-scale path is
    :meth:`LmdbPassageStore.build_from_final` instead.
    """

    __slots__ = ("_passages", "_renderings")

    def __init__(
        self,
        passages: Mapping[str, Passage],
        renderings: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self._passages: dict[str, Passage] = dict(passages)
        self._renderings: dict[str, dict[str, str]] = {}
        for pid, passage in self._passages.items():
            rmap = dict((renderings or {}).get(pid, {}))
            rmap.setdefault("original", passage.text)
            self._renderings[pid] = rmap

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
    ) -> PassageStore:
        """Build from simple mappings in :func:`iter_final_passages`' record shape.

        Each record needs ``passage_id``, ``document_id``, ``lang`` and an ``original``
        text; ``omt`` and ``omt_opus`` texts default to ``original``. Optional
        ``sentence_ids`` / ``token_count`` / ``is_oversized`` / ``paragraph_index`` /
        ``sentence_char_spans`` populate the :class:`Passage` (the spans are offsets into
        the ``original`` text, one per ``sentence_ids`` entry).
        """
        passages: dict[str, Passage] = {}
        renderings: dict[str, dict[str, str]] = {}
        for record in records:
            passage = _passage_from_record(record)
            passages[passage.passage_id] = passage
            renderings[passage.passage_id] = _renderings_from_record(record)
        return cls(passages, renderings)

    def passage(self, passage_id: str) -> Passage:
        """Return the :class:`Passage` for ``passage_id``, or raise ``KeyError``."""
        return self._passages[passage_id]

    def render(self, passage_id: str, passage_lang: str) -> str:
        """Return ``passage_id``'s text in ``passage_lang``; total over the three."""
        _check_rendering(passage_lang)
        rmap = self._renderings[passage_id]
        return rmap.get(passage_lang) or rmap["original"]

    def __contains__(self, passage_id: object) -> bool:
        return passage_id in self._passages

    def __len__(self) -> int:
        return len(self._passages)


# --------------------------------------------------------------------------- #
# One read environment per store per process, reference-counted.
# --------------------------------------------------------------------------- #
# LMDB refuses a second `lmdb.open` of one environment in a process, but this store has
# several legitimate simultaneous holders (two retrieval contexts naming the same store,
# the build-time census and validate re-opens, the devkit search tool). The registry below
# gives them one shared environment, released when the last handle closes.
#
# The key is the store's device and inode rather than its path, because the store is
# published by renaming a temp directory onto the final path: a path-keyed entry could
# serve a later caller from the deleted inode.
#
# Reentrant because `LmdbPassageStore.__del__` releases a dropped handle's reference and a
# collection can fire on any allocation, including one made inside this lock.
_ENV_LOCK = threading.RLock()


@dataclass
class _SharedEnv:
    """One open LMDB read environment plus the number of live handles holding it."""

    env: Any
    refs: int


_SHARED_ENVS: dict[tuple[str, int, int], _SharedEnv] = {}


def _env_key(path: Path) -> tuple[str, int, int]:
    """``(resolved path, st_dev, st_ino)`` of the data file: the store's on-disk identity."""
    try:
        st = (path / "data.mdb").stat()
    except OSError:  # absent or unreadable: the open below reports it properly
        return (str(path), -1, -1)
    return (str(path), st.st_dev, st.st_ino)


def _acquire_env(path: str | Path, map_size: int) -> tuple[tuple[str, int, int], Any]:
    """Return ``(key, env)`` for ``path``, opening it only if no live handle holds it."""
    resolved = Path(path).resolve()
    key = _env_key(resolved)
    with _ENV_LOCK:
        entry = _SHARED_ENVS.get(key)
        if entry is not None:
            entry.refs += 1
            return key, entry.env
        # Opened inside the lock so two threads cannot race two environments onto one path.
        # The registry is written only after the open returns, so a failed open leaves
        # nothing behind for the next caller to find.
        env = _io.lmdb_open(resolved, readonly=True, lock=False, map_size=map_size)
        _SHARED_ENVS[key] = _SharedEnv(env=env, refs=1)
        return key, env


def _release_env(key: tuple[str, int, int], env: Any) -> None:
    """Drop one handle's reference; close the environment when the last handle lets go."""
    with _ENV_LOCK:
        entry = _SHARED_ENVS.get(key)
        if entry is None or entry.env is not env:
            # No longer the shared environment for this key, so this handle owns it alone.
            env.close()
            return
        entry.refs -= 1
        if entry.refs > 0:
            return
        del _SHARED_ENVS[key]
    env.close()


class LmdbPassageStore:
    """LMDB-backed by-id passage store: build once, read many.

    :meth:`build_from_final` is the production entrypoint, streaming the final tables and
    never holding more than ``batch_size`` blobs; :meth:`build` takes any record stream, and
    a generator is expected. Constructing with a path opens ``readonly=True, lock=False``
    for queries.

    Opening the same store twice in one process is legal and shares one environment. LMDB
    itself refuses a second ``lmdb.open`` of an environment already open in the process;
    this class absorbs that by reference-counting a single read environment per store, so a
    second holder gets the first holder's environment instead of an error. Each instance is
    still its own handle: :meth:`close` is idempotent, releases only this handle's
    reference, and the environment is closed when the last handle closes.
    """

    __slots__ = ("_closed", "_env", "_env_key", "_map_size")

    def __init__(
        self,
        path: str | Path,
        *,
        map_size: int = _io.DEFAULT_MAP_SIZE,
    ) -> None:
        self._map_size = map_size
        self._env_key, self._env = _acquire_env(path, map_size)
        self._closed = False

    @classmethod
    def build(
        cls,
        path: str | Path,
        records: Iterable[Mapping[str, Any]],
        *,
        map_size: int = _io.DEFAULT_MAP_SIZE,
        batch_size: int = 10_000,
    ) -> LmdbPassageStore:
        """Bulk-load a record stream, in :meth:`PassageStore.from_records` shape, then open.

        ``records`` is consumed lazily and encoded row by row rather than collected into an
        in-memory store first, so a corpus-scale build peaks at ``batch_size`` blobs. Puts
        are batched per transaction. Returns a read-only store over the fresh environment.
        """
        env = _io.lmdb_open(path, readonly=False, map_size=map_size)
        try:
            batch: list[tuple[bytes, bytes]] = []

            def flush() -> None:
                if not batch:
                    return
                with env.begin(write=True) as txn:
                    for key, value in batch:
                        txn.put(key, value)
                batch.clear()

            for record in records:
                passage = _passage_from_record(record)
                value = _encode(passage, _renderings_from_record(record))
                batch.append((passage.passage_id.encode("utf-8"), value))
                if len(batch) >= batch_size:
                    flush()
            flush()
            env.sync(True)
        finally:
            env.close()
        return cls(path, map_size=map_size)

    @classmethod
    def build_from_final(
        cls,
        path: str | Path,
        layout: Layout,
        reconcile_hash: str,
        *,
        pack_hash: str | None,
        renderings: Sequence[str] = RENDERINGS,
        map_size: int = _io.DEFAULT_MAP_SIZE,
        batch_size: int = 10_000,
    ) -> LmdbPassageStore:
        """Build the by-id store from ``final/<recon12>/`` and ``documents.parquet``.

        Streams :func:`iter_final_passages`, the co-walk join with all three renderings
        composed per passage, straight into :meth:`build`. ``layout`` must be bound to the
        family and chunker hash, since ``documents_path()`` carries neither argument. The
        destination ``path`` is resolved through :class:`~ragtime.common.layout.Layout` by
        the caller and keyed by both hashes for the same reason the passages table is: the
        store is a derived copy of one packing, so a store built from one ``pack12`` must
        not be servable at a path another ``pack12`` also names.
        """
        return cls.build(
            path,
            iter_final_passages(
                layout, reconcile_hash, pack_hash=pack_hash, renderings=renderings
            ),
            map_size=map_size,
            batch_size=batch_size,
        )

    def _blob(self, passage_id: str) -> dict[str, Any]:
        with self._env.begin() as txn:
            raw = txn.get(passage_id.encode("utf-8"))
        if raw is None:
            raise KeyError(passage_id)
        return json.loads(raw.decode("utf-8"))

    def passage(self, passage_id: str) -> Passage:
        """Return the :class:`Passage` for ``passage_id``, or raise ``KeyError``."""
        return _decode_passage(passage_id, self._blob(passage_id))

    def render(self, passage_id: str, passage_lang: str) -> str:
        """Return ``passage_id``'s text in ``passage_lang``; total over the three."""
        _check_rendering(passage_lang)
        return _rendering_from_blob(self._blob(passage_id), passage_lang)

    def close(self) -> None:
        """Release this handle; idempotent, and the environment closes with the last one."""
        if self._closed:
            return
        self._closed = True
        _release_env(self._env_key, self._env)

    def __del__(self) -> None:
        """A handle nobody closed still releases its reference when it is collected.

        Before the environment became shared, dropping a store closed it through py-lmdb's
        own finalizer. Keeping that true matters because a leaked read handle would block a
        later write open of the same path, that is a rebuild, with the same already-open
        error this registry exists to remove.
        """
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110 - a finalizer has nowhere to raise or log to
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
