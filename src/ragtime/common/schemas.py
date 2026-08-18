"""The shared record taxonomy: frozen dataclasses, defined once.

Every stage reads and writes these, and each field has a single owning stage.
Validation lives in ``config`` rather than here, so these are plain
``@dataclass(frozen=True, slots=True)`` with no third-party model layer.

Immutability discipline:

- Records are frozen. Growing a Nugget across coverage-audit rounds, by adding
  answers or flipping status, goes through :func:`dataclasses.replace` rather than
  in-place mutation; see :func:`accumulate_answers`.
- List-valued fields are stored as tuples; dict-valued fields such as
  ``references`` use ``default_factory=dict``.

The three track-output rows (``T1ReportRow``, ``T2RunRow``, ``T3NuggetBankRow``)
carry ``to_line`` serializers producing byte-stable output; floats are guarded
against NaN and infinity, and T2 scores use a fixed, non-scientific format.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any

from .io import ensure_finite  # the one non-finite-float serialization guard

__all__ = [
    "Answer",
    "Document",
    "MetricRecord",
    "Nugget",
    "Passage",
    "Retrieved",
    "Sentence",
    "Support",
    "T1ReportRow",
    "T1Response",
    "T2RunRow",
    "T3Answer",
    "T3Nugget",
    "T3NuggetBankRow",
    "Translation",
    "TranslationRaw",
    "accumulate_answers",
    "document_arrow_schema",
    "final_passage_arrow_schema",
    "merge_map_arrow_schema",
    "passage_arrow_schema",
    "passage_native_text",
    "sentence_arrow_schema",
    "sentence_remap_arrow_schema",
    "translation_final_arrow_schema",
    "translation_raw_arrow_schema",
]

# Byte-stable JSON: no ASCII escaping, so native es/ru/zh bytes survive, and no padding.
_JSON_SEP = (",", ":")
# T2 TREC score: fixed decimal, never scientific notation, so the validator accepts it.
# Nine decimals rather than six, because RRF gaps at depth 1000 fall below `.6f`
# resolution and printed ties are re-sorted by doc-id, losing the true ranking.
_T2_SCORE_FMT = "{:.9f}"


def _dumps(obj: Any) -> str:
    return json.dumps(ensure_finite(obj), ensure_ascii=False, separators=_JSON_SEP)


# --------------------------------------------------------------------------- #
# Corpus records, produced by preprocess between chunking and translation.
#
# The corpus data model is a three-artifact spine in which sentence text is stored
# exactly once, as a span:
#
#   documents  document_id -> the normalized document text, the only copy of the corpus
#   sentences  sentence_id -> (document_id, sentence_index, lang, start, end,
#                              paragraph_index, token_count), where start and end are
#                              code-point offsets into `Document.text`
#   passages   a later, derived grouping of sentences, produced by reconciliation
#
# A sentence therefore carries no text field: its text is `document.text[start:end]`,
# which makes "the same sentence in two passages is byte-identical" true by
# construction.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Document:
    """One source document, normalized once: the single home of the corpus text.

    ``text`` is the chunker's structural paragraph split of ``nfc(raw_text)``, with
    boilerplate lines dropped, joined by spaces. It is therefore NFC by construction and
    every :class:`Sentence` span slices out of it verbatim. This is the only place corpus
    text is materialized; every other corpus record addresses it by offsets.
    """

    document_id: str
    lang: str  # en / es / ru / zh (native)
    text: str


@dataclass(frozen=True, slots=True)
class Sentence:
    """One segmented sentence, addressed as a span of its document's text.

    ``document.text[start:end]`` is this sentence, verbatim and NFC, which is checked for
    every sentence at build time. ``sentence_index`` is dense per document, with no gaps
    or duplicates, and matches the ``#s{j}`` suffix of ``sentence_id``;
    ``paragraph_index`` is the structural paragraph the sentence came from.
    """

    sentence_id: str  # <document_id>#s{j}, document-scoped (see common.ids.sentence_id)
    document_id: str
    sentence_index: int
    lang: str
    start: int  # code-point offset into Document.text (inclusive)
    end: int  # code-point offset into Document.text (exclusive)
    paragraph_index: int
    token_count: int


@dataclass(frozen=True, slots=True)
class Passage:
    """A native-language chunk: a derived grouping of :class:`Sentence` records.

    Produced by reconciliation rather than by ``preprocess.chunk``, which emits only
    documents and sentences, so there is exactly one producer of ``passages.parquet``.

    ``text`` is the native, untranslated passage text; the translated renderings live in
    :class:`Translation` records under the same ``passage_id``.

    ``sentence_char_spans`` makes the chunker's sentence grain exactly recoverable: one
    ``(start, end)`` per ``sentence_ids`` entry, in the same order, as offsets into this
    passage's own ``text``, so ``text[start:end]`` is that sentence verbatim. That
    includes an overlap sentence, whose id belongs to its home passage but whose span is
    relative to this one. Re-segmenting ``text`` would not reproduce these boundaries,
    because the chunker segments per source paragraph and boundaries shift with
    segmentation context, so the spans are stored rather than re-derived and
    sentence-grain consumers slice them.
    """

    passage_id: str
    document_id: str
    lang: str  # en / es / ru / zh (native)
    text: str
    sentence_ids: tuple[str, ...] = ()
    token_count: int = 0
    is_oversized: bool = False
    paragraph_index: tuple[int, ...] = ()  # source Step-1 paragraph(s) the passage spans
    # one (start, end) per sentence_ids entry, same order, offsets into ``text``
    sentence_char_spans: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class TranslationRaw:
    """One raw MT output row: a translation unit with its split markers intact.

    The intermediate the translation arm writes and reconciliation consumes; no online
    stage reads it. Three properties are load-bearing:

    - The row is keyed by a real ``sentence_id``, the unit's first constituent's, rather
      than a synthesized unit key, so the key column is a subset of
      ``sentences.parquet``'s ids and every id resolves to its document through
      ``common.doc_id_of``. There is no ``passage_id``, since passages do not exist yet.
    - ``text_raw`` keeps the section split markers. Scrubbing them at MT time would make
      the fuse-versus-split-back policy unchangeable without re-translating the corpus,
      whereas storing the raw string makes a policy change a short CPU re-pass. The
      column name keeps a consumer from mistaking it for display-ready text; the
      never-store-a-marker guarantee belongs to the reconciliation boundary.
    - ``merge_constituent_count`` makes the row self-describing, so
      ``merge_join.split_translation(text_raw, merge_constituent_count)`` is callable
      from the row alone, without re-reading the merge map.
    """

    sentence_id: str  # the unit's first constituent's id (the sentence's own when count is 1)
    document_id: str
    variant: str  # omt (NLLB-200-3.3B) or omt_opus (OPUS-MT)
    text_raw: str  # MT output with split markers unscrubbed
    source_lang: str  # FLORES-200 code of what was translated
    merge_unit_id: str | None = None  # None when the sentence was translated alone
    merge_constituent_count: int = 1
    model_id: str = ""
    model_config_hash: str = ""  # the semantic ``translation`` block hash
    merge_hash: str = ""  # the semantic ``merge`` block hash that grouped the unit
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class Translation:
    """One rendering of a passage, produced by the translate stage.

    ``variant`` is ``original``, ``omt`` or ``omt_opus``, the translation-tier axis. For
    English passages the ``omt`` and ``omt_opus`` renderings are an identity pass-through
    of the native text.
    """

    passage_id: str
    document_id: str
    variant: str  # original / omt / omt_opus
    text: str
    source_lang: str  # native language of the passage
    qe_score: float | None = None  # CometKiwi 0-1; None for identity/English rows
    model_id: str = ""
    model_config_hash: str = ""


# --------------------------------------------------------------------------- #
# Pinned Arrow schemas for the corpus artifacts.
#
# They live beside the dataclasses they mirror because producers and consumers are in
# different modules and none of them should have to import a producer just to learn the
# column order. Field order here is the file's column order; pinning it together with
# `common.io`'s writer options is what makes a merge byte-deterministic.
#
# pyarrow is imported lazily inside each function so `common` stays importable without it.
# --------------------------------------------------------------------------- #
# int32 rather than int64: the longest document is a few tens of thousands of code
# points, and the sentence columns hold the project's largest row count.
_OFFSET_TYPE = "int32"


def document_arrow_schema() -> Any:
    """Pinned Arrow schema of ``documents.parquet``, mirroring :class:`Document`."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("document_id", pa.string()),
            pa.field("lang", pa.string()),
            pa.field("text", pa.string()),
        ]
    )


def sentence_arrow_schema() -> Any:
    """Pinned Arrow schema of ``sentences.parquet``, mirroring :class:`Sentence`.

    Carries no text column by design: the sentence is ``documents.text[start:end]``.
    """
    import pyarrow as pa

    int_t = getattr(pa, _OFFSET_TYPE)()
    return pa.schema(
        [
            pa.field("sentence_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("sentence_index", int_t),
            pa.field("lang", pa.string()),
            pa.field("start", int_t),
            pa.field("end", int_t),
            pa.field("paragraph_index", int_t),
            pa.field("token_count", int_t),
        ]
    )


def passage_arrow_schema() -> Any:
    """Pinned Arrow schema of ``passages/<rendering>.parquet``, mirroring :class:`Passage`.

    ``sentence_char_spans`` is a fixed-size list of two int64s, so a malformed span pair
    fails at write time rather than downstream.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("passage_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("lang", pa.string()),
            pa.field("text", pa.string()),
            pa.field("sentence_ids", pa.list_(pa.string())),
            pa.field("token_count", pa.int64()),
            pa.field("is_oversized", pa.bool_()),
            pa.field("paragraph_index", pa.list_(pa.int64())),
            pa.field("sentence_char_spans", pa.list_(pa.list_(pa.int64(), 2))),
        ]
    )


def merge_map_arrow_schema() -> Any:
    """Pinned Arrow schema of the sentence-merge map, the translation-unit table.

    One row per non-English ``sentence_id``, co-ordered with ``sentences.parquet`` by
    document order then ``sentence_index``, written by ``preprocess.merge`` at
    ``Layout.merge_map_path`` and read by the translation stage and by reconciliation.
    Pinned here for the same reason as the corpus schemas: producer and consumers are
    different stages, and a hand-rolled schema on either side is how a column silently
    changes type between them.

    The contract consumers rely on, and re-check rather than trust:

    - Totality over the merged languages: exactly one row per non-English sentence, and
      a sentence that merged with nobody is still a unit of size 1. Whether a sentence is
      merged is answered by ``merge_constituent_count``, never by absence. English gets
      no rows at all, being identity pass-through, and merging it would cost citation
      granularity for no gain.
    - A unit is a contiguous run of one document's sentences. ``merge_unit_id`` is the
      first constituent's ``sentence_id``, ``merge_constituent_index`` is 0-based within
      the unit and ``merge_constituent_count`` is its size, so a unit is verifiable from
      any one of its rows and a truncated map is detectable.
    - It is a self-sufficient MT work table: ``lang`` plus ``start`` and ``end`` are
      carried so the translate stage reads exactly this file and ``documents.parquet``.
      The unit's source text is ``documents.text[first.start:last.end]``, constituents
      being adjacent by construction, and each constituent's is ``text[start:end]``. Text
      is still stored in exactly one place, since this table carries offsets.
    - ``merge_unit_token_count`` is the content-token count of the fused unit's source
      text, recorded so reconciliation decides fuse-versus-split-back from a stored
      number without loading a tokenizer.
    """
    import pyarrow as pa

    int_t = getattr(pa, _OFFSET_TYPE)()
    return pa.schema(
        [
            pa.field("sentence_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("sentence_index", int_t),
            pa.field("lang", pa.string()),
            pa.field("start", int_t),
            pa.field("end", int_t),
            pa.field("merge_unit_id", pa.string()),
            pa.field("merge_constituent_index", int_t),
            pa.field("merge_constituent_count", int_t),
            pa.field("merge_unit_token_count", int_t),
        ]
    )


def translation_raw_arrow_schema() -> Any:
    """Pinned Arrow schema of ``translations_raw``, mirroring :class:`TranslationRaw`.

    One row per translation unit rather than per constituent sentence. Fanning the merged
    string back onto every constituent would turn a sizeable share of the inventory into
    byte-identical duplicate rows; the honest storage is one row for the one indivisible
    English string MT produced, plus ``merge_constituent_count`` so reconciliation can try
    to split it back.

    ``merge_unit_id`` is null for a sentence translated alone, English identity rows
    included, so a non-null ``merge_unit_id`` means exactly that the row is a merged unit.
    """
    import pyarrow as pa

    int_t = getattr(pa, _OFFSET_TYPE)()
    return pa.schema(
        [
            pa.field("sentence_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("variant", pa.string()),
            pa.field("text_raw", pa.string()),
            pa.field("source_lang", pa.string()),
            pa.field("merge_unit_id", pa.string()),
            pa.field("merge_constituent_count", int_t),
            pa.field("model_id", pa.string()),
            pa.field("model_config_hash", pa.string()),
            pa.field("merge_hash", pa.string()),
            pa.field("created_at", pa.string()),
        ]
    )


# --------------------------------------------------------------------------- #
# Reconciliation outputs: the final corpus every downstream stage reads.
#
# Reconciliation takes the frozen spine, the merge map and the raw marker-bearing MT text,
# decides per unit whether the split-back survived (keep the constituents) or not (fuse them
# into one sentence spanning first.start to last.end), renumbers each document's sentences
# densely, and scrubs the marker exactly once. It settles which sentences exist and stops
# there: the sole producer of a passage is the separate `packing` substage, which groups
# these sentences into `final/<recon12>/passages/<pack12>/passages.parquet`.
#
# `final/sentences.parquet` reuses `sentence_arrow_schema`: a fused sentence is still a
# sentence, with the same columns. The other three tables are schema-only, with no
# mirroring dataclass, because they are streamed straight into Parquet.
# --------------------------------------------------------------------------- #
def passage_native_text(document_text: str, spans: list[tuple[int, int]]) -> str:
    """The native text of a passage: one slice of the document, never a join.

    A passage's sentences are a contiguous run of spans, so the passage text is
    ``document_text[first.start : last.end]``. Slicing preserves whatever separated the
    sentences in the source: a single space in Latin and Cyrillic prose, and nothing
    between CJK sentences.

    Joining the member slices with a space would differ from the slice for most Chinese
    passages, inserting a separator that is not in the source. That separator would land
    in the text the retriever encodes and in the verbatim NFC span-commit target, so a
    claim quoting across a sentence boundary would fail to ground.

    Translated renderings are joined with a single space for every source language,
    because those segments are English.
    """
    if not spans:
        return ""
    return document_text[spans[0][0] : spans[-1][1]]


def final_passage_arrow_schema() -> Any:
    """Pinned Arrow schema of ``final/<recon12>/passages.parquet``, which is text-free.

    Not :func:`passage_arrow_schema`, which mirrors the in-memory
    :class:`Passage` the packer returns and carries ``text`` and
    ``sentence_char_spans``; persisting those would put a second copy of the whole corpus
    on disk. On disk a passage is an ordered list of sentence ids. Its native text is
    :func:`passage_native_text`, a single slice of the document rather than a join over
    the members, and its translated text is the members' translations joined by a space
    for every source language, because those segments are English. Two renderings, two
    rules, and no stored copy that could drift from the sentences it is made of.

    ``token_count`` is the sum of the members' ``token_count``, never re-counted, and
    ``is_oversized`` marks the single-sentence passage that a sentence longer than the
    content budget gets.
    """
    import pyarrow as pa

    int_t = getattr(pa, _OFFSET_TYPE)()
    return pa.schema(
        [
            pa.field("passage_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("lang", pa.string()),
            pa.field("sentence_ids", pa.list_(pa.string())),
            pa.field("token_count", int_t),
            pa.field("is_oversized", pa.bool_()),
            pa.field("paragraph_index", pa.list_(int_t)),
        ]
    )


def translation_final_arrow_schema() -> Any:
    """Pinned Arrow schema of ``final/<recon12>/translations/<variant>.parquet``.

    The marker-free, final-sentence-keyed English every downstream stage reads: the
    counterpart of :func:`translation_raw_arrow_schema`, deliberately under a different
    name in a different directory so no consumer picks up the raw table by accident.

    Two contracts, both asserted by the producer rather than assumed:

    - Exactly one row per final sentence id. A split unit contributes one row per
      constituent; a fused unit contributes one row for the fused sentence. There is no
      absorbed id, because a fused unit's constituents cease to exist as sentences at
      renumbering time.
    - ``text`` carries no join marker. This is the single scrub boundary in the build.
      English is the one exception by construction rather than by exemption: it is
      identity pass-through, never merged and never markered, so an English row is
      byte-identical to ``documents.text[start:end]``, including the natural section
      signs of US legal citations, which must survive verbatim.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("sentence_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("variant", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source_lang", pa.string()),
        ]
    )


def sentence_remap_arrow_schema() -> Any:
    """Pinned Arrow schema of ``final/<recon12>/remap.parquet``, the audit trail.

    A total function from the pre-reconciliation inventory onto the final one: one row for
    every ``sentences.parquet`` id, whatever happened to it. ``final_sentence_id`` is the
    id it became, its own renumbered id when it survived intact or the fused sentence's id
    shared with its fellow constituents, and ``fused`` says which, so whether a unit's
    marker survived is answerable from the artifact tree without re-reading the raw MT
    text.

    It exists because renumbering is otherwise irreversible: after reconciliation the old
    ids appear in no table, so a stored id in an earlier experiment's output would be
    unresolvable without this map.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("sentence_id", pa.string()),
            pa.field("final_sentence_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("fused", pa.bool_()),
        ]
    )


# --------------------------------------------------------------------------- #
# Nugget taxonomy (decompose + RAG loops + citation scoring)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Retrieved:
    """One entry of ``nugget.retrieved[]``: one ranked hit, a passage id and its reranker score.

    The pool accumulates across the loop's rounds, hit by hit, in
    ``decompose.coverage_audit._merge_retrieved``.

    Not what Task 2 ranks. ``select_serialize.task2_claims.claim_support_rows`` ranks the
    documents the committed claims cite, by accumulated ``claim_importance x nugget_importance``;
    this pool reaches Task 2 only through ``select_serialize.claim_evidence.reranker_by_doc``,
    which reads ``score`` back off the loop record for the tie-break.
    """

    passage_id: str
    score: float


@dataclass(frozen=True, slots=True)
class Support:
    """One entry of ``answer.support[]``: a span-grounded passage backing an answer.

    Carries ``passage_id``, which recovers the exact passage that a doc-id cannot, and the
    passage's native ``lang``. The two together are the identity that
    ``select_serialize.dedup`` unions ``support[]`` on when it merges two answers.
    """

    passage_id: str
    lang: str


@dataclass(frozen=True, slots=True)
class Answer:
    """One entry of ``nugget.answers[]``: one distinct answer value over clustered claims.

    ``references`` maps original doc-id to its citation score, the product
    ``nugget_importance x claim_importance``; Task 1 uses the dict directly and Task 3 keeps
    only its keys. The RAG loops set everything except the ``references`` scores, which the
    post-hoc citation scorer fills.
    """

    answer: str  # short value, e.g. "headache" (Task 3)
    sentence: str  # full citeable English report sentence (Task 1)
    #: The verbatim span that grounded this value, copied from the passage the model read.
    #: Dropped by select-and-serialize, so it reaches no submission: it is commit evidence,
    #: written to the per-loop record. Defaults to "" rather than being required, because
    #: it is genuinely absent for an answer with no single claim behind it, and a required
    #: field would force callers to invent one.
    quoted_span: str = ""
    score: float = 0.0  # never written upstream; select-and-serialize derives max(references)
    references: dict[str, float] = field(default_factory=dict)  # {doc_id: citation score}
    support: tuple[Support, ...] = ()


@dataclass(frozen=True, slots=True)
class Nugget:
    """The shared taxonomy entry, one per nugget.

    Accumulates across stages, with each field's owner fixed so no stage re-derives
    another's number. Grow it across rounds with :func:`dataclasses.replace`, for example
    through :func:`accumulate_answers`.
    """

    nugget_id: str  # <topic_id>#n{j}, a join key that is never submitted
    question: str  # single-sentence English question (Task 3, Task 1 skeleton)
    weight: float = 0.0  # importance, the nugget-selection score
    vital: bool = False  # weight >= vital_cutoff, for monitoring only
    aggregator_type: str = "OR"  # AND or OR (Task 3)
    status: str = "unanswered"  # unanswered / answered / pruned
    origin_round: int = 0  # round that minted it; 0 is the seed round
    trigger_passage_id: str | None = None  # audit trajectory
    retrieved: tuple[Retrieved, ...] = ()  # per-nugget ranked retrieval; Task 2 tie-break only
    answers: tuple[Answer, ...] = ()  # appended by the RAG loops


def accumulate_answers(nugget: Nugget, new_answers: tuple[Answer, ...]) -> Nugget:
    """Return a new Nugget with ``new_answers`` appended.

    The coverage-audit growth path replaces the frozen Nugget rather than mutating it,
    and marks it ``answered`` once it carries at least one answer.
    """
    answers = nugget.answers + tuple(new_answers)
    status = "answered" if answers and nugget.status != "pruned" else nugget.status
    return replace(nugget, answers=answers, status=status)


# --------------------------------------------------------------------------- #
# Metrics counter record (stats bus)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One accumulated counter cell: a ``metric_id`` and slice tuple mapped to a value.

    ``slices`` is a sorted, hashable tuple of ``(key, value)`` pairs, so a record is a
    stable key for roll-up. Emitted by :class:`ragtime.common.stats.Statistics`.
    """

    metric_id: str
    value: float
    slices: tuple[tuple[str, Any], ...] = ()


# --------------------------------------------------------------------------- #
# Track-output rows (the three fixed submission formats)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class T1Response:
    """One ``t1.responses[]`` entry: a report sentence + its citation dict."""

    text: str
    citations: dict[str, float] = field(default_factory=dict)  # {doc_id: citation score}


@dataclass(frozen=True, slots=True)
class T1ReportRow:
    """Task-1 report: one JSON line per topic, in ``report_generation/*.jsonl``."""

    topic_id: str
    team_id: str
    run_id: str
    run_desc: str
    responses: tuple[T1Response, ...] = ()
    references: tuple[str, ...] = ()  # union of all cited doc-ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "team_id": self.team_id,
                "topic_id": self.topic_id,
                "run_id": self.run_id,
                "run_desc": self.run_desc,
            },
            "responses": [
                {"text": r.text, "citations": dict(r.citations)} for r in self.responses
            ],
            "references": list(self.references),
        }

    def to_line(self) -> str:
        """Serialize to one byte-stable JSON line, rejecting non-finite floats."""
        return _dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class T2RunRow:
    """Task-2 retrieval: one row of the six-column TREC run, in ``retrieval/*.txt``."""

    request_id: str
    doc_id: str
    rank: int
    score: float
    run_id: str
    q0: str = "Q0"

    def to_line(self) -> str:
        """``request_id Q0 doc_id rank score run_id``, space-separated with a fixed score."""
        if not math.isfinite(self.score):
            raise ValueError(f"non-finite T2 score: {self.score!r}")
        score = _T2_SCORE_FMT.format(self.score)
        return f"{self.request_id} {self.q0} {self.doc_id} {self.rank} {score} {self.run_id}"


@dataclass(frozen=True, slots=True)
class T3Answer:
    """One ``t3.nugget_bank[].answers[]`` entry: a value and a doc-id list, no scores."""

    answer: str
    references: tuple[str, ...] = ()  # keys of the answer's references dict


@dataclass(frozen=True, slots=True)
class T3Nugget:
    """One ``t3.nugget_bank[]`` entry: question, aggregator type and answers."""

    question: str
    aggregator_type: str  # exactly one of AND or OR
    answers: tuple[T3Answer, ...] = ()


@dataclass(frozen=True, slots=True)
class T3NuggetBankRow:
    """Task-3 nuggets: one JSON line per topic, in ``nuggetization/*.jsonl``."""

    topic_id: str
    team_id: str
    run_id: str
    run_desc: str
    nugget_bank: tuple[T3Nugget, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "team_id": self.team_id,
                "topic_id": self.topic_id,
                "run_id": self.run_id,
                "run_desc": self.run_desc,
            },
            "nugget_bank": [
                {
                    "question": n.question,
                    "aggregator_type": n.aggregator_type,
                    "answers": [
                        {"answer": a.answer, "references": list(a.references)}
                        for a in n.answers
                    ],
                }
                for n in self.nugget_bank
            ],
        }

    def to_line(self) -> str:
        """Serialize to one byte-stable JSON line, rejecting non-finite floats."""
        return _dumps(self.to_dict())
