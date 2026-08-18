"""Sentence reconciliation: the stage that settles what the corpus's sentences finally are.

``chunk`` minted a sentence inventory. ``merge`` decided which consecutive sentences MT must
see as one string. ``translate_omt`` spent the GPU hours and stored exactly what came back,
one row per merge unit with the ``§`` markers unscrubbed. None of those three stages could
say what the corpus's sentences finally are, because that depends on something only
observable after translation: did the marker survive?

This stage answers that question per unit, and everything else follows:

1. The marker survives (exactly ``n-1`` markers, every segment non-empty), so the unit is
   split back. Each constituent keeps its own ``sentence_id`` and span and takes its own
   English segment; the inventory is unchanged for that unit.
2. The marker is lost or miscounted, so the unit is fused. The constituents become one
   sentence whose span is ``(first.start, last.end)``. This is not a degraded special case:
   the union of adjacent spans is itself a contiguous span of the normalized document, so a
   fused sentence is a first-class :class:`~ragtime.common.schemas.Sentence`. There is no
   absorbed-id concept, no tombstone row and no nullable text; the constituents stop
   existing at renumbering time and ``remap.parquet`` records what they became.
3. Each document's sentences are renumbered densely, ``0..m-1``, minting fresh
   ``{doc}#s{j}`` ids.
4. ``§`` is scrubbed exactly once, at this boundary.

English is verified here and stored nowhere, under
``reconcile.store_identity_translations: false``. The identity arm's raw rows are read and each
is checked byte-for-byte against ``documents.text[start:end]``; the verified row is simply not
projected into ``translations/<variant>.parquet``. English has no translation, so the table has
no row for it, and ``common.passage_store`` resolves an English passage's ``omt`` and
``omt_opus`` text from the native span. Storing identity copies instead would account for over a
third of each table, and a stored English "translation" that is not its own source span is
unconstructible here rather than merely checked, since there is no row it could live in. The
checks are the structural converse: :func:`verify_document` asserts the projection
emits a row for exactly the non-English sentences, and ``passage_store`` and ``len_max`` both
assert that a sentence with no row is English and one with a row is not.

Grouping those sentences into passages is not this stage's job: that is
``preprocess.packing``, keyed by its own ``pack12`` one path level in. The split is
load-bearing rather than cosmetic, because ``recon12`` keys ``final/<recon12>/``, the node
holding both renderings' translations, so a packing knob that could move ``recon12`` would
orphan all of it by path and the way back would be a checksum-guarded copy, trading the
composite key's structural guarantee for an assertion. Nothing here reads a token budget, an
overlap fraction or a tokenizer.

Fusion is designed for rather than treated as rare: measured on the corpus, about a third of
merged units fail to split (worst in Chinese) and merged units cover about half of the
non-English sentences, so essentially every non-English document is renumbered and almost no
English one is. That is why this is a streaming per-document pass over four co-ordered
tables rather than random access into any of them.

No tokenizer runs here at all. A fused sentence's token count is the merge map's
``merge_unit_token_count``, recorded at merge time for exactly this purpose, and every other
sentence keeps the count ``chunk`` stored.

The translation rows, not the merge map, are the ground truth. ``build_units`` declines to
merge a unit whose source already carries a ``§``, so the map's grouping and the grouping MT
actually saw can legitimately differ. ``translations_raw`` records what was translated, one
row per real unit carrying its own ``merge_constituent_count``, so this stage walks those
rows and requires them to partition each document's sentences exactly (``i == expect``, then
``expect += n``). That single walk proves totality, density, ordering and non-overlap at
once; the map is consulted only for a fused unit's token count.

The output node's key is composite, ``recon12 = H(chunker, merge, translation, reconcile)``
(:meth:`~ragtime.common.Layout.final_dir`), so that pairing a v1 inventory with a v2
translation is unaddressable by path rather than merely asserted against.

There are two keys and the difference matters. ``recon12`` names the node, so it hashes the
whole ``reconcile`` block including the storage policy, whose value changes the node's
contents. :func:`inventory_hash` names only the sentence inventory, so it hashes the fusion
keys alone. An artifact derived from the inventory rather than from this node's contents,
today the low-tier arm's raw OPUS-MT output, is keyed by the latter and lives outside
``final/``, so a storage-policy edit cannot orphan it. Before keying anything new off
``recon12``, check which of the two questions it depends on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import Layout, Statistics, get_logger, sentence_id
from ragtime.common.io import (
    align_document_rows,
    concat_parquet,
    is_done,
    parquet_row_group_sizes,
    write_jsonl,
    write_parquet_stream,
)
from ragtime.common.passage_store import IDENTITY_LANG
from ragtime.common.schemas import (
    sentence_arrow_schema,
    sentence_remap_arrow_schema,
    translation_final_arrow_schema,
)
from ragtime.config import all_hashes, config_hash
from ragtime.orchestration import saturate
from ragtime.orchestration.provenance import sha256_file
from ragtime.orchestration.run_identity import run_family

from . import spine
from .merge_join import MARKER, scrub_markers, split_translation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from ragtime.config import RunConfig

__all__ = [
    "PARTS",
    "VARIANT",
    "FinalSentence",
    "ReconcileAdapter",
    "ReconcileShard",
    "ReconciledDocument",
    "ReconciliationError",
    "align_subtable",
    "build_document",
    "inventory_hash",
    "inventory_options",
    "reconcile_document",
    "reconcile_hash",
    "reconcile_options",
    "reconcile_shards",
    "verify_document",
]

_log = get_logger("preprocess.reconcile")

_DEFAULT_BASE = "runs"  # mirrors cli._LOCAL_ROOT; real runs pass an explicit Layout base
_HASH_DIRLEN = 12  # same truncation as Layout's hash path levels

#: The rendering reconciled by default. The second (low-tier, OPUS-MT) arm does not come
#: through here: it translates the already-reconciled inventory 1:1 and is projected
#: straight to ``translations/omt_opus.parquet``, so no fuse-vs-split decision applies to
#: it. There is no organizer arm: the organizers' shipped MT is not used anywhere, and
#: the axis is translation tier.
VARIANT = "omt"

# `reconcile` block defaults: fallbacks only; every shipped config carries the block
# byte-identically and its hash is one quarter of the composite output key.
_DEFAULT_SPLIT_BACK = True
_DEFAULT_MIN_SEGMENT_CHARS = 1
#: Default true: a config that does not mention the key must hash to the value it has
#: always hashed to and produce the bytes it has always produced. Flipping the code default
#: would make the shipped ``recon12`` (``f8f20fe2cf17``) name two different artifacts
#: depending on which commit wrote it: precisely the "different bytes under the same hash"
#: invariant-7 failure this knob exists to avoid. The shipped configs carry ``false``
#: explicitly.
_DEFAULT_STORE_IDENTITY = True

#: The two ``reconcile`` keys that decide which sentences exist. Everything else in the
#: block decides what is stored. :func:`inventory_hash` hashes this subset; see its
#: docstring for why the distinction has to be expressed in a path.
_FUSION_KEYS: tuple[str, ...] = ("split_back", "min_segment_chars")

# Execution shape (non-shared `execution` block).
_DEFAULT_RECONCILE_SHARDS = 300

# Counter ids (common.stats), sliced by lang only: the canonical vocabulary.
STAT_DOCUMENTS = "reconcile.documents"
STAT_SPLIT = "reconcile.units_split"
STAT_FUSED = "reconcile.units_fused"
STAT_SENTENCES_IN = "reconcile.sentences_in"
STAT_SENTENCES_FINAL = "reconcile.sentences_final"
#: English sentences whose identity row was verified and then not stored. Counted rather
#: than silent, so "37 % of the rows are gone" is a number in the stats bus, not folklore.
STAT_IDENTITY_ROWS_OMITTED = "reconcile.identity_rows_omitted"

#: The three tables one shard emits, as filename parts (passages are
#: ``preprocess.packing``'s, under a different key). ``work`` writes
#: ``<out>/<shard>.<part>.parquet`` for each and ``merge`` concatenates them part by part:
#: shard names sort by row range, so sorting the whole ``out/`` listing and then filtering
#: on a part suffix restores that part's global row order. The driver hands ``merge``
#: Files (a per-shard directory would be invisible to ``saturate._shard_files``), which is
#: why the three are siblings rather than a subdirectory.
PARTS: tuple[str, ...] = ("sentences", "translations", "remap")
_PRIMARY_PART = "sentences"

# Columns projected from each of the four input tables (never a full-row read).
_SENTENCE_COLUMNS = (
    "sentence_id",
    "document_id",
    "sentence_index",
    "lang",
    "start",
    "end",
    "paragraph_index",
    "token_count",
)
_MAP_COLUMNS = (
    "sentence_id",
    "document_id",
    "merge_unit_id",
    "merge_constituent_count",
    "merge_unit_token_count",
)
_RAW_COLUMNS = (
    "sentence_id",
    "document_id",
    "text_raw",
    "source_lang",
    "merge_constituent_count",
)

class ReconciliationError(RuntimeError):
    """The input tables contradict each other, or an output invariant failed.

    Never guessed around. Every condition it reports corresponds to a way the final corpus
    would otherwise be silently wrong, a citation resolving to text that is not the
    sentence's, a passage referencing an id that no longer exists, a sentence with no
    English, or a stored ``§``. Raising fails the shard, which the work-queue retries and
    then poisons; it never degrades into a partial corpus.
    """


# --------------------------------------------------------------------------- #
# Aligning the three sub-tables onto the shard boundaries.
#
# `spine.plan_shards`/`align_documents` cut the work table (`sentences.parquet`) into
# document-atomic ranges and resolve each range's `documents.parquet` range. The other
# three tables are document-ordered too, but each is a subset of the documents: an English
# document has no merge-map and no MT row, a non-English one has no identity row. So a
# boundary document is often simply absent from a sub-table, and `align_documents`: which
# hard-errors on a boundary it cannot find, correctly, for a table that must contain every
# document: cannot be reused verbatim.
#
# The generalization is one CO-walk. Both streams are in document order and the sub-table's
# documents are a subsequence of `documents.parquet`'s, so walking the two together assigns
# every sub-table row its document ordinal with both pointers moving forward only. A
# boundary is then "the first sub-table row whose document ordinal is >= the boundary's",
# which is well-defined whether or not the boundary document itself appears.
# --------------------------------------------------------------------------- #
def align_subtable(
    parent_path: str | os.PathLike[str],
    subtable_path: str | os.PathLike[str],
    boundary_ordinals: Sequence[int],
) -> list[tuple[int, int]]:
    """Row range of ``subtable_path`` for each shard, given the shards' parent-row ordinals.

    ``boundary_ordinals[k]`` is shard ``k``'s first row index in ``parent_path``. Returns one
    half-open ``(start, end)`` per boundary, covering exactly the sub-table rows belonging to
    that shard's documents, empty ranges are normal and correct (an all-English shard
    owns no merge-map rows at all).

    ``parent_path`` is ``documents.parquet`` for reconciliation's three sub-tables, where an
    ordinal is a document index (``ShardRange.doc_row_start``, which ``align_documents``
    already computed). It is ``final/sentences.parquet`` for ``preprocess.len_max``, whose
    shards are sentence row ranges and whose sub-tables (the translation tables, which carry
    no English row) are document-ordered subsets in exactly the same sense. Both work
    unchanged because the walk keys on ``document_id`` and only ever asks "which parent row
    does this sub-table document first appear at?", and shard boundaries are
    document-atomic, so a boundary always falls on a document's first row whichever table is
    the parent.

    Two forward passes, no random access and no id-to-ordinal dictionary (4 M documents' worth
    of strings would be ~400 MB for a value used once). A sub-table document missing from
    ``parent_path`` is a hard error: the tables are not from the same corpus build,
    and shifting a range would hand a worker another document's rows.

    The co-walk itself moved down to :func:`~ragtime.common.io.align_document_rows` (which
    returns the starts): ``common.passage_store`` needs the same generalization to slice the
    five final tables to one document range, and ``common`` may never import ``preprocess``.
    What stays here is this function's own shape, shard ranges, so an end is the next
    shard's start, over the one shared implementation.
    """
    if not boundary_ordinals:
        return []
    total = sum(parquet_row_group_sizes(subtable_path))
    starts = align_document_rows(parent_path, subtable_path, boundary_ordinals)
    ends = [*starts[1:], total]
    return list(zip(starts, ends, strict=True))


# --------------------------------------------------------------------------- #
# Reading co-ordered sub-tables per document.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _DocRows:
    """A document-ordered row stream, taken one document at a time.

    All four tables are document-ordered subsequences of one global document order, so a
    consumer walking documents in that order can pull each table's rows for the current
    document with a single peeked head, a merge join in constant memory, never a hash join
    over tens of millions of keys. :meth:`take` returns ``[]`` for a document this table
    has no rows for (English in the merge map, non-English in the identity part), which is
    the normal case, not an error.
    """

    _groups: Iterator[tuple[str, list[Mapping[str, Any]]]]
    _head: tuple[str, list[Mapping[str, Any]]] | None = None

    @classmethod
    def over(cls, rows: Iterable[Mapping[str, Any]]) -> _DocRows:
        return cls(_groups=spine.group_by_document(rows))

    def take(self, document_id: str) -> list[Mapping[str, Any]]:
        """This table's rows for ``document_id``, consuming them; ``[]`` when it has none."""
        if self._head is None:
            self._head = next(self._groups, None)
        if self._head is None or self._head[0] != document_id:
            return []
        rows = self._head[1]
        self._head = None
        return rows

    def exhausted(self) -> bool:
        """True when every row has been taken (a leftover means the tables disagree)."""
        if self._head is None:
            self._head = next(self._groups, None)
        return self._head is None


# --------------------------------------------------------------------------- #
# The per-document core: pure, no IO, no tokenizer, no model.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FinalSentence:
    """One sentence of the final inventory: a span, its English, and its provenance.

    ``members`` are the pre-reconciliation ``sentence_id``s that became this sentence,
    one for a survivor, ``n`` for a fused unit. ``sentence_index``/``sentence_id`` are the
    renumbered, dense values; ``token_count`` is the stored per-sentence count for a
    survivor and the merge map's ``merge_unit_token_count`` for a fused unit, so no text is
    ever re-tokenized. ``source_lang`` is carried through from the raw row, the FLORES
    code of what MT was actually handed (a merged zh unit decided Hans/Hant on all of its
    constituents' evidence), which is a property of the unit and cannot be re-derived from
    a single constituent afterwards.
    """

    sentence_id: str
    sentence_index: int
    document_id: str
    lang: str
    start: int
    end: int
    paragraph_index: int
    token_count: int
    text_en: str
    source_lang: str
    members: tuple[str, ...]
    fused: bool


@dataclass(frozen=True, slots=True)
class ReconciledDocument:
    """One document's three output projections, already in final order.

    Passages are absent: they are ``preprocess.packing``'s output, keyed by a
    different hash, and a field here would invite this stage to grow a second producer.
    """

    document_id: str
    lang: str
    sentences: tuple[FinalSentence, ...]
    remap: tuple[dict[str, Any], ...]

    def sentence_rows(self) -> Iterator[dict[str, Any]]:
        """``final/sentences.parquet`` rows (``common.schemas.sentence_arrow_schema``)."""
        for s in self.sentences:
            yield {
                "sentence_id": s.sentence_id,
                "document_id": s.document_id,
                "sentence_index": s.sentence_index,
                "lang": s.lang,
                "start": s.start,
                "end": s.end,
                "paragraph_index": s.paragraph_index,
                "token_count": s.token_count,
            }

    def translation_rows(
        self, variant: str, *, store_identity: bool = _DEFAULT_STORE_IDENTITY
    ) -> Iterator[dict[str, Any]]:
        """``final/translations/<variant>.parquet`` rows: one per translated sentence.

        With ``store_identity=False`` (the shipped policy) an English sentence yields no row
        at all: it has no translation, and its ``omt``/``omt_opus`` text is its own native
        span, which every consumer resolves from ``documents.text`` through
        ``common.passage_store``. ``text_en`` is still carried on the :class:`FinalSentence`
        and still verified against that span by :func:`verify_document`, the verification
        happens on the way past, the storage does not.
        """
        for s in self.sentences:
            if not store_identity and s.lang == IDENTITY_LANG:
                continue
            yield {
                "sentence_id": s.sentence_id,
                "document_id": s.document_id,
                "variant": variant,
                "text": s.text_en,
                "source_lang": s.source_lang,
            }

    def n_translation_rows(self, *, store_identity: bool = _DEFAULT_STORE_IDENTITY) -> int:
        """How many rows :meth:`translation_rows` will emit: the count ``validate`` checks."""
        if store_identity or self.lang != IDENTITY_LANG:
            return len(self.sentences)
        return 0


def _unit_token_counts(map_rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    """``{unit_first_sentence_id: (constituent_count, merge_unit_token_count)}``.

    Only the unit-opening rows are kept (``merge_unit_id == sentence_id``): the fused
    sentence's token count is a property of the unit, and this is the one reason this stage
    reads the merge map at all.
    """
    out: dict[str, tuple[int, int]] = {}
    for row in map_rows:
        if row["merge_unit_id"] == row["sentence_id"]:
            out[row["sentence_id"]] = (
                int(row["merge_constituent_count"]),
                int(row["merge_unit_token_count"]),
            )
    return out


def reconcile_document(
    document_id: str,
    lang: str,
    doc_text: str,
    sent_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    map_rows: Sequence[Mapping[str, Any]],
    *,
    split_back: bool = _DEFAULT_SPLIT_BACK,
    min_segment_chars: int = _DEFAULT_MIN_SEGMENT_CHARS,
    marker: str = MARKER,
    stats: Any = None,
) -> list[FinalSentence]:
    """Decide fuse-vs-split for every unit of one document, then renumber densely.

    ``raw_rows``, the raw MT rows, in document order, are the authority on what MT
    actually translated, and they are required to partition ``sent_rows`` exactly: each row
    keyed at ``sentence_id`` with ``merge_constituent_count = n`` claims sentences
    ``[i, i+n)``, and the next row must start at ``i+n``. Walking that single invariant
    proves totality, density, order and non-overlap in one pass; any violation is a
    :class:`ReconciliationError` rather than a repaired guess, because every repair would
    silently attach some sentence's English to a different sentence.

    Scrubbing is unconditional on every non-English path, split, fused and lone alike, so
    the never-store-a-marker guarantee is structural rather than a consequence of the split
    having succeeded. English is identity pass-through: it is never merged, never markered
    by us, and is stored verbatim, because the corpus's natural ``§`` (US legal citations)
    is real text that a scrub would corrupt.
    """
    index_of = {row["sentence_id"]: i for i, row in enumerate(sent_rows)}
    if len(index_of) != len(sent_rows):
        raise ReconciliationError(f"{document_id}: duplicate sentence_id in the spine")
    unit_counts = _unit_token_counts(map_rows)
    is_english = lang == "en"

    finals: list[FinalSentence] = []
    expect = 0
    j = 0
    for raw in raw_rows:
        sid = raw["sentence_id"]
        i = index_of.get(sid)
        if i is None:
            raise ReconciliationError(
                f"{document_id}: translation row keyed {sid!r} is not a sentence of this "
                "document, the raw table and the spine are not from the same build"
            )
        n = int(raw["merge_constituent_count"])
        if i != expect:
            raise ReconciliationError(
                f"{document_id}: translation unit {sid!r} starts at sentence {i} but "
                f"{expect} was expected, the raw rows do not partition the document "
                "(a gap, an overlap, or a wrong order would misattribute English)"
            )
        members = sent_rows[i : i + n]
        if len(members) != n:
            raise ReconciliationError(
                f"{document_id}: unit {sid!r} declares {n} constituents but only "
                f"{len(members)} sentences remain, the raw table over-reaches the spine"
            )
        expect += n
        text_raw = raw["text_raw"]
        src = str(raw.get("source_lang", "") or "")

        segments: list[str] | None = None
        if n > 1 and split_back:
            segments = split_translation(text_raw, n, marker)
            if segments is not None and any(len(s) < min_segment_chars for s in segments):
                segments = None

        if segments is not None:
            # Split-back: every constituent keeps its own id, span and paragraph.
            if stats is not None:
                stats.emit(STAT_SPLIT, lang=lang)
            for member, segment in zip(members, segments, strict=True):
                finals.append(
                    _final(document_id, lang, j, [member], segment, src, fused=False)
                )
                j += 1
        elif n == 1:
            # A lone sentence. English is stored verbatim; everything else is scrubbed.
            text = text_raw if is_english else scrub_markers(text_raw, marker)
            finals.append(
                _final(document_id, lang, j, list(members), text, src, fused=False)
            )
            j += 1
        else:
            # Fuse: the constituents become one sentence spanning their union. Adjacency
            # makes that union a contiguous span, which is checked here rather than assumed.
            if stats is not None:
                stats.emit(STAT_FUSED, lang=lang)
            declared = unit_counts.get(sid)
            if declared is None or declared[0] != n:
                raise ReconciliationError(
                    f"{document_id}: unit {sid!r} (n={n}) has no matching merge-map unit "
                    f"({declared!r}), the fused sentence's token count is only stored "
                    "there, and re-tokenizing here is not an option this stage has"
                )
            _check_contiguous(document_id, sid, members, doc_text)
            finals.append(
                _final(
                    document_id,
                    lang,
                    j,
                    list(members),
                    scrub_markers(text_raw, marker),
                    src,
                    fused=True,
                    token_count=declared[1],
                )
            )
            j += 1

    if expect != len(sent_rows):
        raise ReconciliationError(
            f"{document_id}: the translation rows cover {expect} of {len(sent_rows)} "
            "sentences, every sentence must have exactly one English row, unconditionally"
        )
    if stats is not None:
        stats.emit(STAT_SENTENCES_IN, float(len(sent_rows)), lang=lang)
        stats.emit(STAT_SENTENCES_FINAL, float(len(finals)), lang=lang)
    return finals


def _final(
    document_id: str,
    lang: str,
    index: int,
    members: Sequence[Mapping[str, Any]],
    text_en: str,
    source_lang: str,
    *,
    fused: bool,
    token_count: int | None = None,
) -> FinalSentence:
    """One renumbered final sentence from its constituent spine row(s)."""
    first, last = members[0], members[-1]
    return FinalSentence(
        sentence_id=sentence_id(document_id, index),
        sentence_index=index,
        document_id=document_id,
        lang=lang,
        start=int(first["start"]),
        end=int(last["end"]),
        paragraph_index=int(first["paragraph_index"]),
        token_count=int(first["token_count"]) if token_count is None else int(token_count),
        text_en=text_en,
        source_lang=source_lang,
        members=tuple(m["sentence_id"] for m in members),
        fused=fused,
    )


def _check_contiguous(
    document_id: str,
    unit_id: str,
    members: Sequence[Mapping[str, Any]],
    doc_text: str,
) -> None:
    """A fused unit's span must really be its constituents' union: verified, not assumed.

    MT was handed the constituents' own slices joined by the marker, so what the fused
    sentence may legitimately claim is exactly those slices plus the separators between
    them. The check is therefore per-gap: consecutive constituents must not overlap, and
    the text between them must be whitespace only (the chunker's single-space paragraph and
    sentence joiners). Non-whitespace in a gap means the fused span swallows real text that
    was never translated, a sentence whose English does not cover its own source.

    Not a "starts with the first, ends with the last" check: that is satisfied
    by any gap at all, which is precisely the corruption it would need to catch.
    """
    lo, hi = int(members[0]["start"]), int(members[-1]["end"])
    if not (0 <= lo < hi <= len(doc_text)):
        raise ReconciliationError(
            f"{document_id}: fused unit {unit_id!r} span ({lo}, {hi}) is not inside its "
            f"document (len {len(doc_text)})"
        )
    for prev, nxt in pairwise(members):
        gap_lo, gap_hi = int(prev["end"]), int(nxt["start"])
        if gap_hi < gap_lo or doc_text[gap_lo:gap_hi].strip():
            raise ReconciliationError(
                f"{document_id}: fused unit {unit_id!r} constituents "
                f"{prev['sentence_id']!r} and {nxt['sentence_id']!r} are not adjacent, "
                "the union of their spans would swallow untranslated text"
            )


# --------------------------------------------------------------------------- #
# The always-on output verification.
# --------------------------------------------------------------------------- #
def verify_document(
    doc: ReconciledDocument,
    doc_text: str,
    n_input_sentences: int,
    *,
    marker: str = MARKER,
    store_identity: bool = _DEFAULT_STORE_IDENTITY,
) -> None:
    """Always-on, every document, every build: the stage's own acceptance criteria.

    Cheap (all O(sentences)) and total, in the same spirit as the chunker's per-sentence
    span check: a corpus this size gets exactly one pass, so the pass verifies itself
    rather than deferring to a sampled audit. Each check corresponds to a way a downstream
    stage would silently serve wrong text.
    """
    finals = doc.sentences
    ids = [s.sentence_id for s in finals]
    # 1. Dense ids, 0..m-1, no gaps and no duplicates.
    if ids != [sentence_id(doc.document_id, k) for k in range(len(finals))]:
        raise ReconciliationError(f"{doc.document_id}: final sentence ids are not dense 0..n-1")
    if [s.sentence_index for s in finals] != list(range(len(finals))):
        raise ReconciliationError(f"{doc.document_id}: final sentence_index is not dense")
    # 2. Every span slices to real, in-order, non-overlapping text of this document.
    prev_end = -1
    for s in finals:
        if not (0 <= s.start < s.end <= len(doc_text)):
            raise ReconciliationError(
                f"{s.sentence_id}: span ({s.start}, {s.end}) is not inside its document"
            )
        if s.start < prev_end:
            raise ReconciliationError(f"{s.sentence_id}: span overlaps the previous sentence")
        prev_end = s.end
    # 3. Exactly one English row per final sentence, marker-free; English verbatim.
    #
    # The English half of this check is the one the ~3.2 GB of identity rows used to pay
    # for, and it is kept: it runs here, in memory, over the identity arm's raw row on its
    # way past, whether or not that row is then stored. Dropping it because the row is no
    # longer written would have removed the only proof that the identity arm copied the
    # right span.
    for s in finals:
        if doc.lang == IDENTITY_LANG:
            if s.text_en != doc_text[s.start : s.end]:
                raise ReconciliationError(
                    f"{s.sentence_id}: English identity text is not byte-identical to its "
                    "source span (identity rows are never scrubbed or rewritten)"
                )
        elif marker in s.text_en:
            raise ReconciliationError(
                f"{s.sentence_id}: a join marker reached a stored final translation, "
                "this boundary is the one place it can be removed"
            )
    # 3b. The projection emits a row for exactly the sentences that have a translation.
    # Checked against the real generator rather than re-derived, so the table on disk and
    # this assertion cannot drift apart. With `store_identity=False` this is what makes
    # "no stored English translation" a property of every document, not a global average.
    emitted = [r["sentence_id"] for r in doc.translation_rows("", store_identity=store_identity)]
    expected = [
        s.sentence_id for s in finals if store_identity or s.lang != IDENTITY_LANG
    ]
    if emitted != expected:
        raise ReconciliationError(
            f"{doc.document_id}: the translation projection emitted {len(emitted)} row(s) "
            f"where {len(expected)} were expected "
            f"(store_identity={store_identity}, lang={doc.lang!r}), a row for an "
            f"{IDENTITY_LANG!r} sentence is an identity copy of text that already exists as "
            "a span, and a missing row for any other language is a sentence with no English"
        )
    if doc.n_translation_rows(store_identity=store_identity) != len(emitted):
        raise ReconciliationError(
            f"{doc.document_id}: n_translation_rows disagrees with translation_rows, "
            "validate()'s row-count gate would be checking the wrong number"
        )
    # 4. remap is a total function over the input inventory.
    known = set(ids)
    if len(doc.remap) != n_input_sentences:
        raise ReconciliationError(
            f"{doc.document_id}: remap has {len(doc.remap)} rows for {n_input_sentences} "
            "input sentences, it must be total over the pre-reconciliation inventory"
        )
    if len({r["sentence_id"] for r in doc.remap}) != n_input_sentences:
        raise ReconciliationError(f"{doc.document_id}: remap maps an input id twice")
    unknown = [r for r in doc.remap if r["final_sentence_id"] not in known]
    if unknown:
        raise ReconciliationError(
            f"{doc.document_id}: remap points {len(unknown)} id(s) at a final sentence "
            "that does not exist"
        )


def build_document(
    document_id: str,
    lang: str,
    doc_text: str,
    sent_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    map_rows: Sequence[Mapping[str, Any]],
    *,
    split_back: bool = _DEFAULT_SPLIT_BACK,
    min_segment_chars: int = _DEFAULT_MIN_SEGMENT_CHARS,
    marker: str = MARKER,
    store_identity: bool = _DEFAULT_STORE_IDENTITY,
    stats: Any = None,
) -> ReconciledDocument:
    """Reconcile, renumber and verify one document (the whole per-doc pipeline).

    Passages are not produced here, ``preprocess.packing`` groups this inventory under its
    own ``pack12`` key, reading the very table this stage writes.
    """
    finals = reconcile_document(
        document_id,
        lang,
        doc_text,
        sent_rows,
        raw_rows,
        map_rows,
        split_back=split_back,
        min_segment_chars=min_segment_chars,
        marker=marker,
        stats=stats,
    )
    remap = [
        {
            "sentence_id": old,
            "final_sentence_id": s.sentence_id,
            "document_id": document_id,
            "fused": s.fused,
        }
        for s in finals
        for old in s.members
    ]
    doc = ReconciledDocument(
        document_id=document_id,
        lang=lang,
        sentences=tuple(finals),
        remap=tuple(remap),
    )
    verify_document(
        doc, doc_text, len(sent_rows), marker=marker, store_identity=store_identity
    )
    if stats is not None:
        stats.emit(STAT_DOCUMENTS, lang=lang)
        if not store_identity and lang == IDENTITY_LANG:
            stats.emit(STAT_IDENTITY_ROWS_OMITTED, float(len(finals)), lang=lang)
    return doc


# --------------------------------------------------------------------------- #
# Config reads (the `reconcile` block is hashed on its own; `execution` is not).
# --------------------------------------------------------------------------- #
def reconcile_options(cfg: RunConfig) -> dict[str, Any]:
    """The semantic reconciliation knobs: the fusion policy plus the storage policy.

    Two knobs decide which sentences exist (:data:`_FUSION_KEYS`) and one decides what is
    written down for them (``store_identity_translations``). How those sentences are grouped
    into passages lives in the separate, separately-hashed ``packing`` block, so a packing
    edit cannot move this node (and therefore cannot orphan the translations inside it).
    """
    r = cfg.blocks.get("reconcile", {})
    return {
        "split_back": bool(r.get("split_back", _DEFAULT_SPLIT_BACK)),
        "min_segment_chars": int(r.get("min_segment_chars", _DEFAULT_MIN_SEGMENT_CHARS)),
        "store_identity_translations": bool(
            r.get("store_identity_translations", _DEFAULT_STORE_IDENTITY)
        ),
    }


def inventory_options(cfg: RunConfig) -> dict[str, Any]:
    """Only the ``reconcile`` keys that decide which sentences exist (:data:`_FUSION_KEYS`).

    Read straight off the block, not off :func:`reconcile_options`' defaulted dict: a key the
    config does not carry must be absent from the hashed mapping, or adding a defaulted key
    to this stage would silently re-key every inventory-derived artifact ever built.
    """
    r = cfg.blocks.get("reconcile", {})
    return {k: r[k] for k in _FUSION_KEYS if k in r}


def inventory_hash(cfg: RunConfig) -> str:
    """``H(chunker, merge, translation, the fusion keys)``: the sentence inventory's key.

    ``recon12`` keys ``final/<recon12>/``, the node, i.e. everything stored in it, and
    since ``store_identity_translations`` changes what is stored, it belongs in ``recon12``
    and moves it. But it changes no sentence: the inventory under the new key is
    byte-identical to the inventory under the old one. So an artifact derived from the
    inventory rather than from this node's contents must not be keyed by ``recon12``, or a
    storage-policy edit would orphan it by path for no reason, which is the exact disease
    (an expensive artifact orphaned by a key that named more than it depended on) that the
    ``packing`` split already treated once.

    There is exactly one such artifact today: the low-tier arm's raw OPUS-MT output, whose
    work table is ``final/<recon12>/sentences.parquet`` and which cost GPU time. It lives at
    ``Layout.translations_raw_path("omt_opus", <opus translation hash>, inventory_hash)``,
    outside ``final/``, keyed exactly like its NLLB counterpart is keyed by the merge map it
    consumed, because for this arm the input work table is the inventory. A stale
    ``_SUCCESS`` still cannot be reused across a real fusion change: that moves this hash.

    For the shipped configs this equals the current ``recon12`` both before and after the
    storage knob lands, which is the property that makes the relocation a rename rather than
    a checksum-guarded copy.
    """
    h = all_hashes(cfg)
    return config_hash(
        {
            "chunker": h["chunker"],
            "merge": h["merge"],
            "translation": h["translation"],
            "reconcile": inventory_options(cfg),
        }
    )


def reconcile_hash(cfg: RunConfig) -> str:
    """The composite key of the final corpus: ``H(chunker, merge, translation, reconcile)``.

    Computed with the one canonical hasher (``config.config_hash``) over a small mapping of
    the three parent hashes plus this stage's own block, never a second sha256 call site
    and never a string concatenation whose field order could drift.

    The composite is the point. The final corpus is a join of three independently-versioned
    inputs, and a key naming only one of them would let the other two be substituted at the
    same path: a v1 inventory beside a v2 translation would find the previous run's
    ``_SUCCESS`` and be served as coherent. Making that pairing unaddressable is strictly
    stronger than asserting against it at read time.

    It hashes the whole ``reconcile`` block, including the storage policy, because it names
    a node and the node's contents change with it. :func:`inventory_hash` is the strictly
    coarser sibling that names only the sentence inventory; read its docstring before keying
    anything new off either.
    """
    h = all_hashes(cfg)
    return config_hash(
        {
            "chunker": h["chunker"],
            "merge": h["merge"],
            "translation": h["translation"],
            "reconcile": cfg.blocks.get("reconcile", {}),
        }
    )


def reconcile_shards(cfg: RunConfig) -> int:
    """Number of document-atomic row-range shards to seed (non-shared ``execution``)."""
    return max(
        1,
        int(cfg.blocks.get("execution", {}).get("reconcile_shards", _DEFAULT_RECONCILE_SHARDS)),
    )


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_payload(shard: Path) -> dict[str, Any]:
    return json.loads(shard.read_text(encoding="utf-8").strip())


# --------------------------------------------------------------------------- #
# The saturate.StageAdapter (CPU template: no model, no GPU, no Apptainer).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ReconcileShard:
    """One claimable unit: a document-atomic range of the spine plus its sub-table ranges.

    ``spine.ShardRange`` covers the two tables every post-chunk substage needs (the work
    table and ``documents.parquet``). Reconciliation joins two more tables, whose rows are
    a document subset, so their ranges are resolved at seed time by :func:`align_subtable`
    and carried in the payload. A worker therefore opens exactly its own row groups in all
    four files and never scans one from row 0.
    """

    rows: spine.ShardRange
    map_rows: tuple[int, int] = (0, 0)
    raw_rows: tuple[int, int] = (0, 0)
    identity_rows: tuple[int, int] = (0, 0)

    @property
    def name(self) -> str:
        return self.rows.name

    def payload(self) -> dict[str, Any]:
        p = dict(self.rows.payload())
        p["map_rows"] = list(self.map_rows)
        p["raw_rows"] = list(self.raw_rows)
        p["identity_rows"] = list(self.identity_rows)
        return p

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReconcileShard:
        def _pair(key: str) -> tuple[int, int]:
            a, b = payload.get(key, (0, 0))
            return int(a), int(b)

        return cls(
            rows=spine.ShardRange.from_payload(payload),
            map_rows=_pair("map_rows"),
            raw_rows=_pair("raw_rows"),
            identity_rows=_pair("identity_rows"),
        )


@dataclass(slots=True)
class _ReconcileCtx:
    """The worker's resident context: four input paths and the fusion policy.

    No tokenizer and no budget: this stage produces no passages, so it needs neither.
    """

    documents: Path
    sentences: Path
    merge_map: Path
    raw_data: Path
    raw_identity: Path
    variant: str
    split_back: bool
    min_segment_chars: int
    marker: str
    store_identity: bool = _DEFAULT_STORE_IDENTITY


@dataclass(slots=True)
class _Shard:
    name: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class ReconcileAdapter:
    """Reconciliation as one ``saturate.StageAdapter``: CPU, no model, no new driver.

    ``stage`` embeds the composite hash, so any edit to the chunker, the merge rule, the
    translation config or the reconcile policy resolves to an entirely fresh queue subtree
    and a fresh artifact node: a stale ``done/`` can never be reused across any of the four
    axes (invariant 7).
    """

    recon_hash: str
    variant: str = VARIANT
    base: str | None = None
    template: str = "workqueue_worker_cpu.sbatch"
    stats: Statistics = field(default_factory=Statistics)
    stage: str = field(init=False)
    #: ``work`` -> ``validate`` hand-off: the exact translations row count per shard.
    _expected_translations: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.stage = f"reconcile_{self.recon_hash[:_HASH_DIRLEN]}"

    @classmethod
    def for_config(
        cls, cfg: RunConfig, *, base: str | None = None, variant: str = VARIANT
    ) -> ReconcileAdapter:
        """Build the adapter for ``cfg``: the only sanctioned construction path."""
        return cls(recon_hash=reconcile_hash(cfg), variant=variant, base=base)

    # -- paths ------------------------------------------------------------- #
    def _layout(self, cfg: RunConfig) -> Layout:
        root = self.base if self.base is not None else _DEFAULT_BASE
        return Layout(
            run_dir=root,
            base=root,
            family=run_family(cfg),
            chunker_hash=all_hashes(cfg)["chunker"],
        )

    def documents_path(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).documents_path()

    def sentences_path(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).sentences_path()

    def merge_map_path(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).merge_map_path(all_hashes(cfg)["merge"])

    def raw_path(self, cfg: RunConfig, part: str) -> Path:
        h = all_hashes(cfg)
        return self._layout(cfg).translations_raw_path(
            self.variant, h["translation"], h["merge"], part=part
        )

    def final_paths(self, cfg: RunConfig) -> dict[str, Path]:
        """The three merged artifacts, keyed by :data:`PARTS` name (Layout-resolved)."""
        layout = self._layout(cfg)
        return {
            "sentences": layout.final_sentences_path(self.recon_hash),
            "translations": layout.final_translations_path(self.recon_hash, self.variant),
            "remap": layout.sentence_remap_path(self.recon_hash),
        }

    def manifest_path(self, cfg: RunConfig) -> Path:
        """``<final_dir>/manifest.json``: Layout-resolved, never hand-built."""
        return self._layout(cfg).final_manifest_path(self.recon_hash)

    # -- seed --------------------------------------------------------------- #
    def shards(self, cfg: RunConfig) -> Iterator[_Shard]:
        """Document-atomic spine ranges, each carrying its three sub-table ranges.

        The two alignment passes (:func:`spine.align_documents` for ``documents.parquet``,
        :func:`align_subtable` for the map and the two raw parts) run once per build, here
        in ``seed``, and write nothing.
        """
        docs = self.documents_path(cfg)
        ranges, boundary_docs = spine.plan_shards(
            self.sentences_path(cfg), n_shards=reconcile_shards(cfg)
        )
        aligned = spine.align_documents(docs, ranges, boundary_docs)
        ordinals = [r.doc_row_start for r in aligned]
        map_ranges = align_subtable(docs, self.merge_map_path(cfg), ordinals)
        raw_ranges = align_subtable(docs, self.raw_path(cfg, "data"), ordinals)
        id_ranges = align_subtable(docs, self.raw_path(cfg, "identity"), ordinals)
        _log.info(
            "preprocess.reconcile.shards",
            stage=self.stage,
            shards=len(aligned),
            variant=self.variant,
        )
        for r, m, d, i in zip(aligned, map_ranges, raw_ranges, id_ranges, strict=True):
            yield _Shard(
                name=r.name,
                payload=ReconcileShard(
                    rows=r, map_rows=m, raw_rows=d, identity_rows=i
                ).payload(),
            )

    # -- work --------------------------------------------------------------- #
    def bringup(self, cfg: RunConfig) -> _ReconcileCtx:
        """Resolve the four inputs + the fusion policy. No model and no tokenizer.

        Nothing here is a budget: the content budget belongs to ``preprocess.packing``, the
        stage that actually groups sentences into passages.
        """
        opts = reconcile_options(cfg)
        tcfg = dict(cfg.blocks.get("translation", {}).get("config", {}))
        return _ReconcileCtx(
            documents=self.documents_path(cfg),
            sentences=self.sentences_path(cfg),
            merge_map=self.merge_map_path(cfg),
            raw_data=self.raw_path(cfg, "data"),
            raw_identity=self.raw_path(cfg, "identity"),
            variant=self.variant,
            split_back=opts["split_back"],
            min_segment_chars=opts["min_segment_chars"],
            # The marker is a translation knob (it changed the MT input bytes), so the
            # scrub reads the same value the join used: never a second literal.
            marker=str(tcfg.get("merge_marker", MARKER)),
            store_identity=opts["store_identity_translations"],
        )

    def documents_for_shard(
        self, ctx: _ReconcileCtx, shard: ReconcileShard
    ) -> Iterator[ReconciledDocument]:
        """Stream this shard's documents, reconciled: constant memory, one doc at a time.

        Four co-ordered readers advance together over the shard's own row ranges. The two
        raw parts are separate files (the GPU arm and the CPU English arm never write one
        file), but they are disjoint by document and both document-ordered, so taking from
        each and concatenating yields exactly the document's units in order.
        """
        r = shard.rows
        texts = spine.load_document_texts(ctx.documents, r)
        maps = _DocRows.over(
            spine.iter_row_range(
                ctx.merge_map, *shard.map_rows, columns=list(_MAP_COLUMNS)
            )
        )
        raws = _DocRows.over(
            spine.iter_row_range(ctx.raw_data, *shard.raw_rows, columns=list(_RAW_COLUMNS))
        )
        idents = _DocRows.over(
            spine.iter_row_range(
                ctx.raw_identity, *shard.identity_rows, columns=list(_RAW_COLUMNS)
            )
        )
        sent_rows = spine.iter_row_range(
            ctx.sentences, r.row_start, r.row_end, columns=list(_SENTENCE_COLUMNS)
        )
        for doc_id, rows in spine.group_by_document(sent_rows):
            doc_text = texts.get(doc_id)
            if doc_text is None:
                raise LookupError(
                    f"{doc_id}: no row in documents.parquet for a document in this shard's "
                    "sentence range, the tables are not from the same corpus build"
                )
            raw_rows = [*raws.take(doc_id), *idents.take(doc_id)]
            if not raw_rows:
                raise ReconciliationError(
                    f"{doc_id}: no translation rows at all, every document must be fully "
                    "covered by the raw table before it can be reconciled"
                )
            yield build_document(
                doc_id,
                rows[0]["lang"],
                doc_text,
                rows,
                raw_rows,
                maps.take(doc_id),
                split_back=ctx.split_back,
                min_segment_chars=ctx.min_segment_chars,
                marker=ctx.marker,
                store_identity=ctx.store_identity,
                stats=self.stats,
            )
        # Nothing may be left over. Every sub-table row in this shard's range belongs to a
        # document in this shard's sentence range, so a leftover means the ranges were
        # aligned against a different build: the one corruption a per-document read cannot
        # notice, because it never asks for the rows it fails to consume.
        for name, stream in (("merge_map", maps), ("raw", raws), ("identity", idents)):
            if not stream.exhausted():
                raise ReconciliationError(
                    f"{shard.name}: {name} rows remain after this shard's documents were "
                    "consumed, the sub-table range does not match the sentence range"
                )

    def work(self, ctx: _ReconcileCtx, shard: Path) -> Path:
        """Emit this shard's three tables; return the primary one for ``validate``.

        The shard is reconciled once into a list of :class:`ReconciledDocument` (bounded by
        the shard, not the corpus) and then projected three ways, so the three tables cannot
        disagree about a document, the same guarantee the chunk stage gets from its one
        nested record, applied to four outputs instead of two.
        """
        spec = ReconcileShard.from_payload(_read_payload(shard))
        base = saturate.shard_out_path(shard)
        docs = list(self.documents_for_shard(ctx, spec))
        # The exact row count `validate` will hold the translations part to. Computed from
        # the same objects that produced the rows, not re-derived from the file, so a
        # truncated write is still caught while a legitimately shorter table (no English
        # rows) is not mistaken for one.
        self._expected_translations[str(_part_path(base, _PRIMARY_PART))] = sum(
            d.n_translation_rows(store_identity=ctx.store_identity) for d in docs
        )
        write_parquet_stream(
            _part_path(base, "sentences"),
            (row for d in docs for row in d.sentence_rows()),
            schema=sentence_arrow_schema(),
        )
        write_parquet_stream(
            _part_path(base, "translations"),
            (
                row
                for d in docs
                for row in d.translation_rows(
                    ctx.variant, store_identity=ctx.store_identity
                )
            ),
            schema=translation_final_arrow_schema(),
        )
        write_parquet_stream(
            _part_path(base, "remap"),
            (row for d in docs for row in d.remap),
            schema=sentence_remap_arrow_schema(),
        )
        _log.info(
            "preprocess.reconcile.shard.done",
            shard=shard.name,
            documents=len(docs),
            sentences=sum(len(d.sentences) for d in docs),
            fused=sum(1 for d in docs for s in d.sentences if s.fused),
        )
        return _part_path(base, _PRIMARY_PART)

    def validate(self, path: Path) -> bool:
        """All three parts present and marked done, and their row counts consistent.

        Not a non-zero size: a shard whose documents all reconcile to nothing is impossible
        (every document has sentences), but an empty table is still a legal artifact and the
        marker is the gate, and under ``store_identity_translations: false`` an all-English
        shard legitimately writes zero translation rows. What is checked instead is the
        cross-table arithmetic that a truncated write would break: the translations part
        holds exactly the number of rows ``work`` projected (one per non-English sentence,
        or one per sentence when identity rows are stored), and there is at least one remap
        row per final sentence.
        """
        base = _strip_part(path)
        parts = {p: _part_path(base, p) for p in PARTS}
        for name, p in parts.items():
            if not (p.exists() and is_done(p)):
                _log.warning(
                    "preprocess.reconcile.validate.missing", part=name, path=str(p)
                )
                return False
        counts = {n: sum(parquet_row_group_sizes(p)) for n, p in parts.items()}
        # `work` -> `validate` hand-off (same process, back to back in run_worker). A
        # re-validation with no hand-off falls back to the weaker "never more rows than
        # sentences" bound rather than to the old, now-wrong equality.
        expected = self._expected_translations.pop(str(path), None)
        if expected is None:
            if counts["translations"] > counts["sentences"]:
                _log.warning("preprocess.reconcile.validate.translation_count", **counts)
                return False
        elif counts["translations"] != expected:
            _log.warning(
                "preprocess.reconcile.validate.translation_count",
                expected=expected,
                **counts,
            )
            return False
        if counts["remap"] < counts["sentences"]:
            _log.warning("preprocess.reconcile.validate.remap_count", **counts)
            return False
        return True

    # -- merge -------------------------------------------------------------- #
    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> None:
        """Concat each part into its family artifact, then write ``manifest.json``.

        Shard names are ``rows_<start:012d>_<end:012d>``, so sorting the ``out/`` listing
        and filtering on a part suffix restores that part's global row order, every
        document stays a contiguous row range in all four merged tables, exactly as it is
        in the spine.
        """
        outs = self.final_paths(cfg)
        schemas = {
            "sentences": sentence_arrow_schema(),
            "translations": translation_final_arrow_schema(),
            "remap": sentence_remap_arrow_schema(),
        }
        ordered = sorted(shard_paths)
        for part in PARTS:
            sources = [p for p in ordered if p.name.endswith(f".{part}.parquet")]
            concat_parquet(outs[part], sources, schema=schemas[part])
        self._write_manifest(cfg, outs, n_shards=len(ordered) // max(1, len(PARTS)))
        _log.info(
            "preprocess.reconcile.done",
            stage=self.stage,
            variant=self.variant,
            path=str(outs["sentences"].parent),
        )

    def _write_manifest(
        self, cfg: RunConfig, outs: Mapping[str, Path], *, n_shards: int
    ) -> None:
        """The composite key's audit half: parent hashes, row counts, file checksums.

        The directory name is a 12-hex digest of four inputs; this records those inputs in
        full, so "which translation is this corpus paired with?" is answerable from the
        artifact tree alone, months later, without re-deriving anything. Checksums use
        ``orchestration.provenance.sha256_file`` (a streaming file hasher, never
        ``config.config_hash``, which fingerprints parsed blocks).
        """
        h = all_hashes(cfg)
        manifest = {
            "reconcile_hash": self.recon_hash,
            # The node declares its own inventory key. The low-tier OPUS-MT arm translates
            # this node's `sentences.parquet` but resolves to a different `translation`
            # hash, so it cannot compute this value from its own config (that is the same
            # reason `RAGTIME_INVENTORY_DIR` names the node explicitly): it reads it from
            # here, and its raw output is keyed by it, outside `final/`. Without this field
            # the arm would have to guess, and a guess is exactly the assertion the
            # composite key exists to replace.
            "inventory_hash": inventory_hash(cfg),
            "variant": self.variant,
            "parents": {
                "chunker_hash": h["chunker"],
                "merge_hash": h["merge"],
                "translation_hash": h["translation"],
            },
            "reconcile": json.loads(json.dumps(dict(cfg.blocks.get("reconcile", {})))),
            "shards": n_shards,
            "tables": {
                name: {
                    "path": str(path),
                    "rows": sum(parquet_row_group_sizes(path)),
                    "sha256": sha256_file(path),
                }
                for name, path in sorted(outs.items())
            },
            "created_at": _now(),
        }
        write_jsonl(self.manifest_path(cfg), [manifest])


def _part_path(base: Path, part: str) -> Path:
    """``<out>/<shard>.<part>.parquet``: the per-shard file for one output table."""
    return base.with_name(f"{base.name}.{part}.parquet")


def _strip_part(path: Path) -> Path:
    """Inverse of :func:`_part_path` for the primary part (what ``work`` returns)."""
    suffix = f".{_PRIMARY_PART}.parquet"
    if not path.name.endswith(suffix):
        raise ValueError(f"not a reconcile primary shard output: {path.name!r}")
    return path.with_name(path.name[: -len(suffix)])
