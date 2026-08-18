"""Deterministic structural SaT segmentation into the document and sentence spine.

Runs once on native text (``chunk(cfg)`` takes no ``rendering`` argument): it mints the
sentence ids before any translation exists, so the spine is rendering-independent by
construction and every later stage inherits these ids one for one. Per document it:

1. NFC-normalizes the text once, per document (``common.nfc``), so the online loop's verbatim
   span-commit lands on the exact stored bytes. NFC of a substring of an NFC string is that
   substring, so there is no per-sentence normalization; the property is asserted for every
   sentence rather than assumed.
2. Splits the text structurally into paragraphs on blank-line boundaries, dropping
   pure-boilerplate lines (nav, cookie, share, dateline, copyright and read-more chrome, per
   the versioned ``boilerplate`` rule sets) when ``strip_boilerplate`` is on. A document with
   no surviving structure falls back to one paragraph; one that reduces to zero content is
   dropped rather than emitted as a junk record.
3. Joins the paragraphs with single spaces into the document text, the one copy of the corpus.
4. Segments each paragraph with the SaT segmenter (``serving.segmenter``, reused rather than
   re-instantiated) via ``split_spans`` and ``split_spans_batch``, which return ``(start,
   end)`` offsets rather than copies; the paragraph-local offsets are shifted onto the
   document text, giving each sentence a span of the document.
5. Mints ``<doc>#s{j}`` ids (``common.ids.sentence_id``), dense and 0-based in document
   order, and counts tokens per sentence with the pinned pack tokenizer.

It emits documents and sentences, never passages. ``documents.parquet`` holds
``document_id -> text``, stored exactly once, and ``sentences.parquet`` holds one text-free
row per sentence addressing it as ``documents.text[start:end]``. Passages are a derived
grouping whose sole producer is the ``packing`` substage, so two files named
``passages.parquet`` with different meanings can never coexist. The sentence packer below
(:func:`_pack_sentences` and :func:`chunk_document`) is retained as the packing reference the
derivation is built from and as a byte-identity oracle; it writes nothing.

Verification is always on: for every sentence, ``documents.text[start:end]`` must equal the
segmenter's own segment and must be NFC. A violation increments
``chunk.sentence_span_mismatch`` and fails the shard, with no recovery path, because a
sentence that is not a span of its document is a broken document. Before this data model was
adopted, wtpsplit was checked to reproduce its input verbatim on every paragraph of the real
corpus, so no fallback is needed.

The stage is deterministic: there is no random state anywhere, so two runs over identical
native text are byte-identical. The per-shard work-queue outputs stay byte-stable JSONL (one
nested record per document); only the merged spine is columnar Parquet with zstd, under the
pinned ``common.schemas.document_arrow_schema`` and ``sentence_arrow_schema``.
"""

from __future__ import annotations

import atexit
import math
import multiprocessing
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import (
    Layout,
    Passage,
    Statistics,
    document_arrow_schema,
    get_logger,
    nfc,
    passage_id,
    sentence_arrow_schema,
    sentence_id,
    write_jsonl,
)
from ragtime.common.io import is_done, write_parquet_stream
from ragtime.config import all_hashes
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family
from ragtime.serving.segmenter import Segmenter, SpanMismatchError

from . import corpus
from .boilerplate import boilerplate_rules, is_boilerplate
from .tokenizer import load_pack_tokenizer, load_pack_tokenizer_by_id

# Bounded wait for one pool result. Generous, since a fat doc-batch legitimately takes
# minutes, but finite: the point is to convert an infinite `imap` wedge into a retryable
# shard failure. See `_ChunkPool.chunk_documents`.
_POOL_RESULT_TIMEOUT_S = 1800.0


class PoolStalledError(RuntimeError):
    """A pool child died and wedged `imap`; the shard must fail rather than hang."""


class SentenceSpanMismatchError(RuntimeError):
    """A sentence is not a verbatim NFC span of its document, so the shard must fail.

    Counted as :data:`SPAN_MISMATCH_METRIC` on :data:`STATS` immediately before being
    raised, so the counter and the failure are the same event (a non-zero counter and a
    completed build are not a reachable combination).
    """


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from ragtime.config import RunConfig

__all__ = [
    "ChunkAdapter",
    "SentenceSpanMismatchError",
    "chunk",
    "chunk_document",
    "segment_document",
    "segment_documents",
    "segment_documents_parallel",
]

_log = get_logger("preprocess.chunk")

# The span-verification counter. Process-local by construction (each pool child /
# work-queue worker is its own process and fails its own shard), so a `Statistics`
# instance here is the right granularity: the counter's only job is to name and count the
# event at the point it aborts the build.
SPAN_MISMATCH_METRIC = "chunk.sentence_span_mismatch"
STATS = Statistics()

_DEFAULT_BASE = "runs"  # real runs pass an explicit Layout base
_DEFAULT_CHUNK_BATCH = 64  # docs per batched SaT call; tunable from `execution`

# Chunker defaults, used only as fallbacks when the fairness-shared `chunker.config` block
# omits the key.
_DEFAULT_STRIP_BOILERPLATE = True
_DEFAULT_BOILERPLATE_RULES = "v1"
_DEFAULT_PREFER_PARA_BREAK = True
# "near budget" for prefer_paragraph_break: close at a paragraph boundary once the running
# passage has already filled >= this fraction of the content budget (a soft preference).
_DEFAULT_PREFER_MIN_FILL = 0.6


@dataclass(slots=True)
class _Sent:
    """A sentence being packed: its NFC text, token count, source paragraph, sentence-id.

    ``sid`` is the document-scoped ``<doc>#s{j}`` id minted at segmentation time, so
    it is already fixed before packing ever runs: a sentence carried into the next passage
    as overlap is the same id by construction, not by a carry-the-foreign-id rule. ``para``
    is the Step-1 paragraph index the sentence came from (drives ``prefer_paragraph_break``
    and the passage's ``paragraph_index``).
    """

    text: str
    count: int
    sid: str
    para: int = 0


def _paragraphs(
    text: str,
    *,
    strip_boilerplate: bool,
    rules: tuple[tuple[str, Any], ...],
) -> list[str]:
    """Split NFC ``text`` into paragraphs on blank lines, dropping boilerplate lines.

    Lines are split on ``\\n``; a blank (empty/whitespace) line is a paragraph boundary and
    consecutive content lines join (space-separated) into one paragraph. When
    ``strip_boilerplate`` is on, pure-chrome lines (Step 1's ``boilerplate_rules``) are
    dropped; a dropped line does not itself open a paragraph break, so content around a
    stripped chrome line stays one paragraph. Empty/whitespace paragraphs are dropped
    (order preserved).

    Fallbacks: if nothing survives because boilerplate stripping emptied the document, return
    ``[]``, since an entirely-boilerplate document reduces to zero content and is dropped; a
    genuinely unstructured document (a wall of text, one long line) already forms a single
    paragraph; and a truly empty document returns ``[]``.
    """
    stripped_any = False
    paras: list[str] = []
    cur: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        if strip_boilerplate and is_boilerplate(line, rules):
            stripped_any = True
            continue
        cur.append(line)
    if cur:
        paras.append(" ".join(cur))
    if paras:
        return paras
    # Nothing survived: an entirely-boilerplate doc is dropped (never resurrected via the
    # whole-text fallback); an empty doc is dropped; otherwise fall back to the whole text.
    if strip_boilerplate and stripped_any:
        return []
    whole = text.strip()
    return [whole] if whole else []


# --------------------------------------------------------------------------- #
# The emitted spine: documents and sentences, with sentence text stored once as a span.
# --------------------------------------------------------------------------- #
def _span_mismatch(lang: str, detail: str) -> SentenceSpanMismatchError:
    """Count the violation on :data:`STATS`, log it, and build the fatal error.

    The counter and the abort are one event: `chunk.sentence_span_mismatch`
    can never be non-zero on a build that completed.
    """
    STATS.emit(SPAN_MISMATCH_METRIC, lang=lang)
    _log.error(SPAN_MISMATCH_METRIC, lang=lang, detail=detail)
    return SentenceSpanMismatchError(detail)


def _paragraph_offsets(paras: Sequence[str]) -> tuple[str, list[int]]:
    """``" ".join(paras)`` plus each paragraph's start offset in it.

    Mirrors the joiner exactly, one space between adjacent paragraphs, so
    ``text[off[i]:off[i] + len(paras[i])] == paras[i]`` for every ``i``, which is what
    lets a paragraph-local sentence span be shifted onto the document by pure addition.
    """
    offsets: list[int] = []
    pos = 0
    for i, p in enumerate(paras):
        if i:
            pos += 1  # the single-space joiner between adjacent paragraphs
        offsets.append(pos)
        pos += len(p)
    return " ".join(paras), offsets


def _assemble(
    doc: dict[str, Any],
    paras: Sequence[str],
    para_spans: Sequence[Sequence[tuple[int, int]]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Build one nested ``documents`` and ``sentences`` record from a document's spans.

    ``para_spans[i]`` are the segmenter's spans of ``paras[i]`` (paragraph-local). This
    shifts them onto the joined document text, mints dense ``<doc>#s{j}`` ids in document
    order, counts tokens, and verifies every sentence: the document slice must equal the
    paragraph slice the segmenter produced, and must be NFC. Both checks are always on and
    fatal (see :func:`_span_mismatch`).

    A document whose paragraphs are empty (entirely boilerplate / empty source) yields a
    record with empty text and no sentences; the caller drops it. Measured on the real
    corpus this never happens, but it is a dead branch rather than an impossible one.
    """
    doc_id, lang = doc["id"], doc["lang"]
    text, offsets = _paragraph_offsets(paras)
    sentences: list[dict[str, Any]] = []
    for i, (para, spans) in enumerate(zip(paras, para_spans, strict=True)):
        base = offsets[i]
        for ls, le in spans:
            seg = para[ls:le]
            start, end = base + ls, base + le
            if text[start:end] != seg:
                raise _span_mismatch(
                    lang,
                    f"{doc_id}#s{len(sentences)}: document slice [{start}, {end}) "
                    f"!= segment {seg!r}",
                )
            if nfc(seg) != seg:
                raise _span_mismatch(
                    lang, f"{doc_id}#s{len(sentences)}: span is not NFC: {seg!r}"
                )
            sentences.append(
                {
                    "sentence_id": sentence_id(doc_id, len(sentences)),
                    "document_id": doc_id,
                    "sentence_index": len(sentences),
                    "lang": lang,
                    "start": start,
                    "end": end,
                    "paragraph_index": i,
                    "token_count": tokenizer.count(seg),
                }
            )
    return {"document_id": doc_id, "lang": lang, "text": text, "sentences": sentences}


def segment_document(
    doc: dict[str, Any],
    segmenter: Any,
    tokenizer: Any,
    *,
    strip_boilerplate: bool = _DEFAULT_STRIP_BOILERPLATE,
    boilerplate_rules_version: str = _DEFAULT_BOILERPLATE_RULES,
) -> dict[str, Any]:
    """Segment one native document into its nested ``document`` and ``sentences`` record.

    The single-document reference path: paragraph split, then per-paragraph
    ``segmenter.split_spans``). Byte-identical to the batched path, which only batches the
    segmenter calls. Returns ``{document_id, lang, text, sentences: [...]}``.
    """
    rules = boilerplate_rules(boilerplate_rules_version) if strip_boilerplate else ()
    paras = _paragraphs(nfc(doc["text"]), strip_boilerplate=strip_boilerplate, rules=rules)
    try:
        spans = [segmenter.split_spans(p) for p in paras]
    except SpanMismatchError as exc:
        raise _span_mismatch(doc["lang"], f"{doc['id']}: {exc}") from exc
    return _assemble(doc, paras, spans, tokenizer)


def segment_documents(
    docs: Iterable[dict[str, Any]],
    segmenter: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    strip_boilerplate: bool = _DEFAULT_STRIP_BOILERPLATE,
    boilerplate_rules_version: str = _DEFAULT_BOILERPLATE_RULES,
) -> Iterator[dict[str, Any]]:
    """Segment a stream of native documents in bounded ``batch_size`` doc-batches.

    Collects ``batch_size`` docs, Step-1 splits each into paragraphs, segments the batch's
    paragraphs via ``segmenter.split_spans_batch`` (per-text under the hood, byte-identical
    to per-paragraph ``split_spans``; tensor batching measured slower on the ONNX CPU
    backend), then re-groups the per-paragraph span lists onto their doc. Only
    ``batch_size`` docs are held at once; the same doc-batches are the unit the process pool
    (:func:`segment_documents_parallel`) fans out. Documents that reduce to zero content are
    dropped here (never emitted as an empty record).
    """
    rules = boilerplate_rules(boilerplate_rules_version) if strip_boilerplate else ()
    batch: list[dict[str, Any]] = []

    def _flush_batch() -> Iterator[dict[str, Any]]:
        per_doc_paras = [
            _paragraphs(nfc(d["text"]), strip_boilerplate=strip_boilerplate, rules=rules)
            for d in batch
        ]
        flat = [p for paras in per_doc_paras for p in paras]  # every paragraph in the batch
        try:
            flat_spans = segmenter.split_spans_batch(flat) if flat else []
        except SpanMismatchError as exc:
            raise _span_mismatch(batch[0]["lang"], str(exc)) from exc
        idx = 0
        for d, paras in zip(batch, per_doc_paras, strict=True):
            spans = flat_spans[idx : idx + len(paras)]  # one entry per paragraph, in order
            idx += len(paras)
            rec = _assemble(d, paras, spans, tokenizer)
            if rec["sentences"]:
                yield rec

    for d in docs:
        batch.append(d)
        if len(batch) >= batch_size:
            yield from _flush_batch()
            batch = []
    if batch:
        yield from _flush_batch()


# --------------------------------------------------------------------------- #
# Sentence packing into passages. Retained, but not emitted by this stage: passages are a
# derived grouping whose sole producer is the `packing` substage. This is the packing
# reference that derivation is built from, and a byte-identity oracle for the span-addressed
# rewrite: chunk_document must keep producing what it produced from the older
# representation. Nothing here writes an artifact.
# --------------------------------------------------------------------------- #
def _overlap_tail(
    prev: Sequence[_Sent], overlap_frac: float, budget: int, next_count: int
) -> list[_Sent]:
    """Trailing whole sentences of ``prev`` to carry into the next passage.

    Room outranks the floor, which outranks the cap: the overlap is at least one whole
    sentence that fits within the remaining budget, up to about ``overlap_frac``, and the
    passage-within-budget invariant outranks the floor. The overlap must leave room for the
    sentence that triggered the flush, so it is capped at ``room = budget - next_count``,
    with ``next_count`` the tokens of that next sentence. Within ``room``:

    - Floor: include the last trailing sentence of ``prev`` (one sentence of shared
      context) even if it exceeds the soft cap, but only if it fits in ``room``. If even that
      single sentence exceeds ``room`` the overlap is empty for this seam, since carrying it
      would bust the budget and ``next`` is never split.
    - Cap: keep adding earlier trailing sentences while the running total stays both
      ``<= overlap_frac * budget`` (soft) and ``<= room`` (hard), stopping before either.

    The returned overlap plus ``next_count`` never exceeds ``budget``, so a non-oversized
    passage is always within budget.
    """
    if overlap_frac <= 0 or not prev:
        return []
    room = budget - next_count  # hard: overlap must not crowd out the triggering sentence
    rev = list(reversed(prev))
    if rev[0].count > room:  # even one sentence would bust budget-with-next -> no overlap
        return []
    target = overlap_frac * budget  # soft cap
    tail: list[_Sent] = [rev[0]]  # floor: the last sentence (it fit within room)
    total = rev[0].count
    for se in rev[1:]:
        if total + se.count > target or total + se.count > room:  # soft cap and hard room
            break
        tail.append(se)
        total += se.count
    tail.reverse()
    return tail


def _sentence_spans(cur: Sequence[_Sent]) -> tuple[tuple[int, int], ...]:
    """Per-sentence ``(start, end)`` offsets into ``" ".join(se.text for se in cur)``.

    Mirrors the passage-text assembly exactly: sentence texts are concatenated with a
    single-space joiner, so sentence ``i`` starts one character after the previous one
    ends (the joiner) and spans ``len(se.text)`` characters. Consequently
    ``ptext[start:end] == se.text`` verbatim for every sentence, whatever whitespace or
    script it holds, and an overlap sentence, which keeps its home passage's
    ``sentence_id``) gets the offsets it occupies in this passage's text.

    Offsets are Python string indices, that is code points, into the NFC passage text,
    so slicing is exact for the downstream sentence-grain consumers (sentence-level MT and
    the per-sentence spans the passage store persists) and no stage ever re-segments.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for i, se in enumerate(cur):
        if i:
            pos += 1  # the single-space joiner between adjacent sentences
        spans.append((pos, pos + len(se.text)))
        pos += len(se.text)
    return tuple(spans)


def _sents_of(record: Mapping[str, Any]) -> list[_Sent]:
    """The packer's ``_Sent`` view of a nested ``segment_document`` record.

    Each sentence's text is sliced from the document, never re-derived or re-normalized:
    ``record["text"][start:end]`` is verbatim NFC by the checks in :func:`_assemble`, which
    is why no per-sentence ``nfc()`` call is needed or wanted here.
    """
    text = record["text"]
    return [
        _Sent(
            text=text[s["start"] : s["end"]],
            count=s["token_count"],
            sid=s["sentence_id"],
            para=s["paragraph_index"],
        )
        for s in record["sentences"]
    ]


def _pack_sentences(
    doc_id: str,
    lang: str,
    sents: Sequence[_Sent],
    tokenizer: Any,
    *,
    token_budget: int,
    overlap_frac: float,
    prefer_paragraph_break: bool,
    prefer_min_fill: float,
) -> list[Passage]:
    """Greedily pack one document's already-segmented, identified sentences into passages.

    Each ``_Sent`` arrives with its text, token count, ``<doc>#s{j}`` id and source
    paragraph already fixed by segmentation, so packing is now purely a grouping decision
    (it mints only ``passage_id``). Packing, and the oversized check, target the content
    budget ``token_budget - tokenizer.num_special()``: the retriever's special tokens are
    reserved so a full passage encoded with specials stays within the real window, and the
    overlap tail is capped against the same content budget.

    ``prefer_paragraph_break``: once the running passage has filled ``prefer_min_fill`` of
    the content budget and the next sentence opens a new paragraph, close the passage at that
    paragraph boundary in preference to packing further to the token limit (a soft
    preference, so passages tend to align with the document's block structure).

    The result is not an emitted artifact of this stage; see the section header above.
    """
    content_budget = token_budget - tokenizer.num_special()
    near = prefer_min_fill * content_budget

    passages: list[Passage] = []
    cur: list[_Sent] = []
    cur_tok = 0
    k = 0

    def flush(*, oversized: bool = False) -> None:
        nonlocal cur, cur_tok, k
        if not cur:
            return
        pid = passage_id(doc_id, k)
        ptext = " ".join(se.text for se in cur)
        passages.append(
            Passage(
                passage_id=pid,
                document_id=doc_id,
                lang=lang,
                text=ptext,
                sentence_ids=tuple(se.sid for se in cur),
                token_count=sum(se.count for se in cur),
                is_oversized=oversized,
                paragraph_index=tuple(sorted({se.para for se in cur})),
                sentence_char_spans=_sentence_spans(cur),
            )
        )
        cur = []
        cur_tok = 0
        k += 1

    for se in sents:
        if se.count > content_budget:
            flush()  # close any buffered (non-oversized) passage first
            cur = [se]
            cur_tok = se.count
            flush(oversized=True)  # the over-budget sentence gets its own passage
            continue
        # Break the current passage either because the next sentence overflows the budget, or
        # (under prefer_paragraph_break) because we are near budget and it opens a new
        # paragraph, in which case close at that boundary rather than filling to the limit.
        opens_new_para = bool(cur) and se.para != cur[-1].para
        prefer_break = prefer_paragraph_break and opens_new_para and cur_tok >= near
        if cur and (prefer_break or cur_tok + se.count > content_budget):
            prev = list(cur)
            flush()
            # Overlap must leave room for `se` so the new passage stays <= budget (the
            # passage-within-budget invariant outranks the overlap floor).
            tail = _overlap_tail(prev, overlap_frac, content_budget, se.count)
            cur = list(tail)
            cur_tok = sum(s.count for s in tail)
        cur.append(se)
        cur_tok += se.count
    flush()
    return passages


def chunk_document(
    doc: dict[str, Any],
    segmenter: Any,
    tokenizer: Any,
    *,
    token_budget: int,
    overlap_frac: float,
    strip_boilerplate: bool = _DEFAULT_STRIP_BOILERPLATE,
    boilerplate_rules_version: str = _DEFAULT_BOILERPLATE_RULES,
    prefer_paragraph_break: bool = _DEFAULT_PREFER_PARA_BREAK,
    prefer_min_fill: float = _DEFAULT_PREFER_MIN_FILL,
) -> list[Passage]:
    """Chunk one native document ``{id, text, url, date, lang}`` into ``Passage`` records.

    Segment with :func:`segment_document`, then pack, so the passages are derived from the
    same span-addressed sentences the spine stores, and divergence between the two
    representations is impossible rather than merely unlikely. Writes nothing (see the
    packing section header).
    """
    record = segment_document(
        doc, segmenter, tokenizer,
        strip_boilerplate=strip_boilerplate,
        boilerplate_rules_version=boilerplate_rules_version,
    )
    return _pack_sentences(
        doc["id"], doc["lang"], _sents_of(record), tokenizer,
        token_budget=token_budget, overlap_frac=overlap_frac,
        prefer_paragraph_break=prefer_paragraph_break, prefer_min_fill=prefer_min_fill,
    )


def chunk_documents(
    docs: Iterable[dict[str, Any]],
    segmenter: Any,
    tokenizer: Any,
    *,
    token_budget: int,
    overlap_frac: float,
    batch_size: int,
    strip_boilerplate: bool = _DEFAULT_STRIP_BOILERPLATE,
    boilerplate_rules_version: str = _DEFAULT_BOILERPLATE_RULES,
    prefer_paragraph_break: bool = _DEFAULT_PREFER_PARA_BREAK,
    prefer_min_fill: float = _DEFAULT_PREFER_MIN_FILL,
) -> Iterator[Passage]:
    """Pack a stream of native documents into passages: segmentation plus the packer.

    The batched counterpart of :func:`chunk_document`, sharing its one segmentation core, so
    per-doc and batched packing cannot diverge. Writes nothing.
    """
    for record in segment_documents(
        docs, segmenter, tokenizer, batch_size=batch_size,
        strip_boilerplate=strip_boilerplate,
        boilerplate_rules_version=boilerplate_rules_version,
    ):
        yield from _pack_sentences(
            record["document_id"], record["lang"], _sents_of(record), tokenizer,
            token_budget=token_budget, overlap_frac=overlap_frac,
            prefer_paragraph_break=prefer_paragraph_break, prefer_min_fill=prefer_min_fill,
        )


# --------------------------------------------------------------------------- #
# Multiprocessing across documents (throughput only; byte-identical output).
#
# The pool fans the same ``batch_size`` doc-batches the sequential path forms over N
# worker processes. Each worker loads its own SaT model and pack tokenizer once (they
# are neither picklable nor shareable) with ``intra_op_threads=1``, and every native pool
# (OMP, OpenBLAS, MKL, tokenizers) is pinned to one thread in ``_pool_init``, so the budget
# really is N processes of one thread and the pool never oversubscribes the SLURM cpuset it
# runs inside. ``imap`` preserves submission order, and each batch is segmented by the
# unchanged ``segment_documents`` core, so the concatenated stream is byte-identical to the
# sequential path. Parallelism lives inside one work-queue worker, across the documents of
# its shard, never a SLURM job per batch.
# --------------------------------------------------------------------------- #
_FALLBACK_WORKER_CAP = 8  # dev/login fallback cap when SLURM_CPUS_PER_TASK is unset

_pool_models: dict[str, Any] = {}  # per-worker-process singletons (set by _pool_init)


def _pool_context() -> Any:
    """The pool's start-method context: ``fork`` on Linux, ``spawn`` everywhere else.

    ``fork`` is required on the cluster. With ``spawn``
    the child re-imports the parent's ``__main__`` module, and under the real entrypoint
    ``__main__`` is the console script ``.venv/bin/run``, an extension-less file, so
    ``multiprocessing.spawn`` sends ``init_main_from_path`` and the child re-executes that
    file through ``runpy.run_path``. Two consequences follow, both invisible to a pytest run
    (pytest's ``__main__`` is a real module):

    1. any console script whose ``main()`` call is not behind an ``if __name__ ==
       "__main__"`` guard re-enters ``main()`` in the child and trips
       ``spawn._check_not_importing_main()`` ("An attempt has been made to start a new
       process before the current process has finished its bootstrapping phase"), which the
       Pool answers by respawning replacement workers, an unbounded process storm;
    2. even with a guarded script, every child re-imports the whole ``ragtime`` and site-
       packages graph off shared storage. On a large launch that re-import is what actually
       failed: a child died in ``reduction.pickle.load`` with ``FileNotFoundError`` on
       ``jsonschema_specifications/schemas/draft201909/metaschema.json``, a file that is
       present on disk: a transient metadata failure under the import storm.

    ``fork`` re-imports nothing, neither ``__main__`` nor site-packages, so both failure modes
    are structurally impossible, not merely unlikely.

    Fork is safe here, and the fork-after-threads hazard is addressed by construction:

    - the forking parent has loaded no models. ``ChunkAdapter.bringup`` builds the pool
      itself and constructs a parent-side ``Segmenter`` and tokenizer only on the ``workers <=
      1`` sequential path, so no onnxruntime session, no ORT intra-op thread pool and no
      Rust-tokenizer rayon pool ever exists at fork time; each child creates its own,
      after the fork, in :func:`_pool_init`.
    - the pool is stood up in ``bringup``, before ``saturate.run_worker`` enters its
      claim loop and so before the per-shard heartbeat daemon thread exists: the parent is
      single-threaded at fork. :class:`_ChunkPool` logs
      ``preprocess.chunk.pool.fork_unclean_parent`` if either half ever stops holding.
    - production dispatches ``seed``, ``worker`` and ``drive`` as separate processes
      (``PREPROCESS_ROLE``), so the forking worker process has never run ``download``.
    - inherited ``atexit`` handlers do not run in a forked Pool child (``Process._bootstrap``
      exits via ``os._exit``), so the parent's ``close`` registration cannot fire twice.

    Non-Linux platforms keep ``spawn``: ``fork`` is unsafe there, and both
    entrypoints on that box (``.venv/bin/run``, pytest) have a guarded ``__main__``.
    """
    if sys.platform.startswith("linux") and "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _pool_init(segmenter_model: str, tokenizer_id: str, model_factory: Any = None) -> None:
    """Pool-worker initializer: stand up this process's SaT model and tokenizer once.

    ``model_factory``, a picklable ``() -> (segmenter, tokenizer)``, is a test seam: the
    byte-identity tests inject dependency-free fakes into real worker processes. Production
    always leaves it ``None``, giving the real SaT model and the pinned pack tokenizer.

    Every native thread pool is pinned to one thread first, before the heavy imports, so the
    pool's documented budget of N processes with one thread each is what actually happens.
    Without it each worker reached over a hundred threads and more than a gigabyte of RSS:
    the ``scipy`` and ``sklearn`` import brings up OpenBLAS, which defaults to 128 threads
    on a 256-core node regardless of the ONNX Runtime setting. At six or more workers that
    explosion wedged the pool on a large box (reproduced under both ``fork``
    and ``spawn``, so it is orthogonal to the start method), and inside a SLURM memory cgroup
    it gets workers killed and silently replaced, cascading shard failures onto one worker.
    Pinning to one also removes a machine-dependent
    thread count from the chunk path, which the byte-identical-shared-spine invariant wants.
    """
    # The Rust tokenizer's rayon threads would fight the N sibling processes.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"  # set before scipy/sklearn/onnxruntime are imported below
    if model_factory is not None:
        _pool_models["segmenter"], _pool_models["tokenizer"] = model_factory()
        return
    from ragtime.serving.segmenter import Segmenter as _Segmenter

    _pool_models["segmenter"] = _Segmenter(model=segmenter_model, intra_op_threads=1)
    _pool_models["tokenizer"] = load_pack_tokenizer_by_id(tokenizer_id)


def _pool_segment_batch(
    payload: tuple[list[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Segment one doc-batch in a pool worker via the unchanged sequential core.

    Returns the nested ``{document_id, lang, text, sentences: [...]}`` records as plain
    dicts, so the pool's pickle round-trip carries no dataclass/type coupling.
    """
    docs, kwargs = payload
    return list(
        segment_documents(docs, _pool_models["segmenter"], _pool_models["tokenizer"], **kwargs)
    )


def default_chunk_workers() -> int:
    """Worker-process default: the SLURM cpuset if set, else a capped ``os.cpu_count()``.

    ``SLURM_CPUS_PER_TASK`` is an explicit allocation, so it is used as-is; the bare-metal
    fallback (dev/login nodes) is capped at :data:`_FALLBACK_WORKER_CAP` so an unconfigured
    run never grabs a whole shared machine.
    """
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.isdigit() and int(v) > 0:
        return int(v)
    return max(1, min(os.cpu_count() or 1, _FALLBACK_WORKER_CAP))


def _chunk_workers(cfg: RunConfig) -> int:
    """Chunk pool size: the non-shared ``execution.chunk_workers`` knob, else the default."""
    v = cfg.blocks.get("execution", {}).get("chunk_workers")
    return int(v) if v is not None else default_chunk_workers()


class _ChunkPool:
    """A resident process Pool that segments doc-batches in input order (byte-identical).

    Start method: :func:`_pool_context` (``fork`` on Linux, ``spawn`` elsewhere). Read
    that docstring first, it carries the entrypoint-safety argument (a ``spawn`` child
    re-imports the console-script ``__main__``; ``fork`` re-imports nothing).

    Created once per work-queue worker (in ``ChunkAdapter.bringup``, while the parent is
    still single-threaded and model-free) and reused across its claimed shards, so each pool
    process pays SaT/tokenizer bring-up exactly once. Workers are pool-managed (daemonic);
    ``close`` is registered via ``atexit`` as a tidy fallback.
    """

    def __init__(
        self,
        workers: int,
        segmenter_model: str,
        tokenizer_id: str,
        model_factory: Any = None,
    ) -> None:
        self.workers = workers
        ctx = _pool_context()
        if ctx.get_start_method() == "fork":
            # Fork-safety alarm, not a gate: the parent should be single-threaded and
            # model-free here (see _pool_context). `threading` sees only PYTHON threads, so a
            # loaded onnxruntime, whose Eigen intra-op pool is native and invisible to
            # `active_count()`, is detected by its import instead. Logged, never raised, so
            # a dev/local path still runs; production (bringup, pre-claim-loop) trips neither.
            models = sorted(m for m in ("onnxruntime", "wtpsplit") if m in sys.modules)
            if threading.active_count() > 1 or models:
                _log.warning(
                    "preprocess.chunk.pool.fork_unclean_parent",
                    threads=threading.active_count(),
                    thread_names=[t.name for t in threading.enumerate()],
                    models_imported=models,
                )
        self._pool = ctx.Pool(
            processes=workers,
            initializer=_pool_init,
            initargs=(segmenter_model, tokenizer_id, model_factory),
        )
        atexit.register(self.close)

    def segment_documents(
        self, docs: Iterable[dict[str, Any]], *, batch_size: int, **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        """Segment a doc stream over the pool; output order == input order (imap)."""

        def payloads() -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
            batch: list[dict[str, Any]] = []
            for d in docs:
                batch.append(d)
                if len(batch) >= batch_size:
                    yield batch, {"batch_size": batch_size, **kwargs}
                    batch = []
            if batch:
                yield batch, {"batch_size": batch_size, **kwargs}

        # it.next(timeout=...) instead of plain iteration: if a pool child dies (the
        # spurious-ENOENT import storm is the observed trigger), imap blocks
        # forever waiting on a result that can never arrive. That is not theoretical:
        # it hung a large worker fleet twice, and it is
        # invisible from outside because saturate's heartbeat thread keeps stamping,
        # so the reaper never reclaims the shard and the queue shows zero failed and zero
        # completed simultaneously. A bounded wait turns an unbounded hang into an
        # ordinary shard failure, which the work queue already knows how to retry.
        it = self._pool.imap(_pool_segment_batch, payloads())
        while True:
            try:
                out = it.next(timeout=_POOL_RESULT_TIMEOUT_S)
            except StopIteration:
                return
            except multiprocessing.TimeoutError as exc:
                raise PoolStalledError(
                    f"no pool result for {_POOL_RESULT_TIMEOUT_S}s -- assuming a dead "
                    f"child wedged imap; failing the shard so it can be retried"
                ) from exc
            yield from out

    def close(self) -> None:
        self._pool.close()
        self._pool.join()


def segment_documents_parallel(
    docs: Iterable[dict[str, Any]],
    *,
    segmenter_model: str,
    tokenizer_id: str,
    workers: int,
    batch_size: int,
    **options: Any,
) -> Iterator[dict[str, Any]]:
    """Segment a doc stream over ``workers`` processes, byte-identical to sequential.

    One-shot convenience over :class:`_ChunkPool` (stand up, stream, tear down). With
    ``workers <= 1`` it degrades to the plain sequential path with in-process models.
    """
    if workers <= 1:
        yield from segment_documents(
            docs,
            Segmenter(model=segmenter_model),
            load_pack_tokenizer_by_id(tokenizer_id),
            batch_size=batch_size,
            **options,
        )
        return
    pool = _ChunkPool(workers, segmenter_model, tokenizer_id)
    try:
        yield from pool.segment_documents(docs, batch_size=batch_size, **options)
    finally:
        pool.close()


# --------------------------------------------------------------------------- #
# Shard record <-> the two spine tables.
#
# One nested record per document travels from the pool through the shard JSONL to merge:
# ``{document_id, lang, text, sentences: [...]}``. ``merge`` is then a pure projection of
# that stream (drop the sentences key, or flatten it) rather than a reconstruction, so there
# is no place for the two tables to disagree about a document.
# --------------------------------------------------------------------------- #
def _document_rows(records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """The ``documents.parquet`` projection of the nested shard records."""
    for rec in records:
        yield {"document_id": rec["document_id"], "lang": rec["lang"], "text": rec["text"]}


def _sentence_rows(records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """The ``sentences.parquet`` projection: the nested sentence lists, flattened."""
    for rec in records:
        yield from rec["sentences"]


def _iter_docs(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """All native docs under the family-shared raw store, in a stable (sorted-file) order."""
    for f in sorted(raw_dir.glob(corpus.DOCS_GLOB)):
        yield from corpus.read_native(f)


def _chunk_params(cfg: RunConfig) -> tuple[int, float, str]:
    c = cfg.blocks["chunker"]["config"]
    return int(c["token_budget"]), float(c["overlap_frac"]), c["segmenter_model"]


def _tokenizer_id(cfg: RunConfig) -> str:
    return str(cfg.blocks["chunker"]["config"]["tokenizer_id"])


def _chunk_options(cfg: RunConfig) -> dict[str, Any]:
    """The semantic chunking knobs, read from the fairness-shared ``chunker.config``.

    ``strip_boilerplate``, ``boilerplate_rules``, ``prefer_paragraph_break`` and
    ``prefer_paragraph_break_min_fill`` are chunking semantics: they change the artifact, so
    they fold into ``chunker_hash``. They are read with module
    defaults as fallbacks so an older config still chunks deterministically; a config that
    sets them takes effect and folds into the hash.
    """
    c = cfg.blocks["chunker"]["config"]
    return {
        "strip_boilerplate": bool(c.get("strip_boilerplate", _DEFAULT_STRIP_BOILERPLATE)),
        "boilerplate_rules_version": str(c.get("boilerplate_rules", _DEFAULT_BOILERPLATE_RULES)),
        "prefer_paragraph_break": bool(
            c.get("prefer_paragraph_break", _DEFAULT_PREFER_PARA_BREAK)
        ),
        "prefer_min_fill": float(
            c.get("prefer_paragraph_break_min_fill", _DEFAULT_PREFER_MIN_FILL)
        ),
    }


# The subset of the semantic knobs segmentation consumes. The other two
# (``prefer_paragraph_break``/``prefer_min_fill``) are packing knobs: still fairness-shared
# and still folded into ``chunker_hash``, but read by the passage derivation rather than
# here: segmentation must not silently accept a knob it ignores.
_SEGMENT_OPTION_KEYS = ("strip_boilerplate", "boilerplate_rules_version")


def _segment_options(cfg: RunConfig) -> dict[str, Any]:
    """The Step-1 knobs :func:`segment_documents` takes, from ``chunker.config``."""
    opts = _chunk_options(cfg)
    return {k: opts[k] for k in _SEGMENT_OPTION_KEYS}


def _chunk_batch_size(cfg: RunConfig) -> int:
    """Documents per batched SaT call: the throughput knob, from the non-shared ``execution``
    block (tunable per-config, defaults to :data:`_DEFAULT_CHUNK_BATCH` when unset)."""
    return int(cfg.blocks.get("execution", {}).get("chunk_batch_size", _DEFAULT_CHUNK_BATCH))


def _chunker_hash(cfg: RunConfig) -> str:
    """The semantic chunker hash, which keys the family-shared corpus artifact path."""
    return all_hashes(cfg)["chunker"]


def chunk(
    cfg: RunConfig,
    *,
    base: str | os.PathLike[str] | None = None,
    segmenter: Any = None,
    tokenizer: Any = None,
) -> tuple[Path, Path]:
    """Segment the family's native corpus into the ``(documents, sentences)`` spine.

    Writes exactly ``Layout.documents_path()`` and ``Layout.sentences_path()``, as Parquet
    with zstd under the pinned schemas. No passage
    artifact is written; their sole producer is the ``packing`` substage.
    ``segmenter``/``tokenizer`` may be injected for testing; otherwise the SaT segmenter and
    the pinned ``bge-m3`` pack tokenizer are stood up from config. Resume is a no-op: a
    ``_SUCCESS``-marked pair is returned untouched.
    """
    fam = run_family(cfg)
    ch = _chunker_hash(cfg)
    root = Path(base) if base is not None else Path(_DEFAULT_BASE)
    layout = Layout(run_dir=root, base=root, family=fam, chunker_hash=ch)
    docs_out, sents_out = layout.documents_path(), layout.sentences_path()
    if is_done(docs_out) and is_done(sents_out):
        return docs_out, sents_out

    _, _, seg_model = _chunk_params(cfg)
    workers = _chunk_workers(cfg)
    if segmenter is None and tokenizer is None and workers > 1:
        # Real-model path: fan doc-batches over the process pool (byte-identical output).
        records = list(
            segment_documents_parallel(
                _iter_docs(layout.corpus_raw_dir(fam, ch)),
                segmenter_model=seg_model,
                tokenizer_id=_tokenizer_id(cfg),
                workers=workers,
                batch_size=_chunk_batch_size(cfg),
                **_segment_options(cfg),
            )
        )
    else:
        # Injected-model path (tests) or a single-CPU allocation: the sequential core.
        segmenter = segmenter if segmenter is not None else Segmenter(model=seg_model)
        tokenizer = tokenizer if tokenizer is not None else load_pack_tokenizer(cfg)
        records = list(
            segment_documents(
                _iter_docs(layout.corpus_raw_dir(fam, ch)),
                segmenter,
                tokenizer,
                batch_size=_chunk_batch_size(cfg),
                **_segment_options(cfg),
            )
        )
    write_parquet_stream(docs_out, _document_rows(records), schema=document_arrow_schema())
    write_parquet_stream(sents_out, _sentence_rows(records), schema=sentence_arrow_schema())
    _log.info(
        "preprocess.chunk.done",
        family=fam,
        documents=len(records),
        sentences=sum(len(r["sentences"]) for r in records),
        documents_path=str(docs_out),
        sentences_path=str(sents_out),
    )
    return docs_out, sents_out


# --------------------------------------------------------------------------- #
# The saturate.StageAdapter for the work-queue driver (routed to the CPU template).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _ChunkCtx:
    segmenter: Any
    tokenizer: Any
    raw_dir: Path
    batch_size: int
    # The seed-materialized per-shard slice dir (``Layout.corpus_slices_dir``). ``None``
    # (a hand-built ctx) forces the slow ``read_native_slice`` fallback in ``work``.
    slices_dir: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)  # the Step-1 semantic knobs
    # -- multiprocessing across the shard's docs (workers>1 -> the resident _ChunkPool) --
    workers: int = 1
    segmenter_model: str = ""
    tokenizer_id: str = ""
    pool: Any = None  # _ChunkPool built in bringup, reused across this worker's shards


@dataclass(slots=True)
class _Shard:
    """One claimable chunk shard: a ``(file, doc-offset-range)`` slice. There are many per
    file, and documents are never split across a shard, so reassembly stays shard-local."""

    name: str
    payload: dict = field(default_factory=dict)


def _shard_slices(counts: dict[str, int], corpus_shards: int) -> Iterator[tuple[str, int, int]]:
    """Doc-balanced ``(file, start, end)`` slices ~``corpus_shards`` total, ``docs_per_shard``-capped.

    ``docs_per_shard = ceil(total_docs / corpus_shards)`` bounds each shard's doc count;
    each file is split into contiguous ``[start, end)`` slices of that budget (file-local,
    no cross-file segments) so the union is every doc exactly once, in order.
    """
    total = sum(counts.values())
    per = max(1, math.ceil(total / corpus_shards)) if total else 1
    for fname in sorted(counts):
        c = counts[fname]
        for start in range(0, c, per):
            yield fname, start, min(start + per, c)


@dataclass(slots=True)
class ChunkAdapter:
    """The chunk substage as a ``saturate.StageAdapter``, CPU-only.

    ``shards`` yields many doc-balanced ``(file, start, end)`` slices (~``corpus_shards``
    total, oversubscribing the worker array ``execution.oversubscription``:1) so N
    concurrent workers actually saturate, and, in the same once-per-build ``seed`` call,
    materialises each of those slices as its own small gzip file under
    ``Layout.corpus_slices_dir`` (:meth:`presplit`), so a worker reads a few megabytes
    instead of streaming the whole language file to its offset. ``work`` segments only its
    slice (bounded memory) and writes one nested ``{document, sentences}`` record per
    document as byte-stable JSONL, and ``merge`` streams those shard records in shard order
    into the family's two Parquet tables in constant memory, one row group at a time:
    ``documents.parquet`` and ``sentences.parquet``, never a passage artifact.
    """

    stage: str = "chunk"
    template: str = "workqueue_worker_cpu.sbatch"
    base: str | None = None

    def _layout(self, cfg: RunConfig) -> Layout:
        root = self.base if self.base is not None else _DEFAULT_BASE
        return Layout(run_dir=root, base=root, family=run_family(cfg), chunker_hash=_chunker_hash(cfg))

    def _raw_dir(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).corpus_raw_dir(run_family(cfg), _chunker_hash(cfg))

    def _slices_dir(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).corpus_slices_dir(run_family(cfg), _chunker_hash(cfg))

    def bringup(self, cfg: RunConfig) -> _ChunkCtx:
        """Stand up this worker's resident replica once: the pool, or the models.

        Order matters for fork safety (see :func:`_pool_context`): when ``workers > 1`` the
        parent constructs no ``Segmenter`` or tokenizer at all (the pool processes own them)
        and the pool is forked here: ``saturate.run_worker`` calls ``bringup`` before its claim
        loop, so the parent is still single-threaded (no per-shard heartbeat thread yet) and
        holds no onnxruntime session or Rust-tokenizer thread pool. The parent-side models
        exist only on the ``workers <= 1`` sequential path, which forks nothing.
        """
        _, _, seg_model = _chunk_params(cfg)
        workers = _chunk_workers(cfg)
        parallel = workers > 1
        ctx = _ChunkCtx(
            segmenter=None if parallel else Segmenter(model=seg_model),
            tokenizer=None if parallel else load_pack_tokenizer(cfg),
            raw_dir=self._raw_dir(cfg),
            batch_size=_chunk_batch_size(cfg),
            options=_segment_options(cfg),
            slices_dir=self._slices_dir(cfg),
            workers=workers,
            segmenter_model=seg_model,
            tokenizer_id=_tokenizer_id(cfg),
        )
        if parallel:  # fork the pool from the still-single-threaded, model-free parent
            ctx.pool = _ChunkPool(workers, seg_model, ctx.tokenizer_id)
        return ctx

    def shards(self, cfg: RunConfig) -> Iterator[_Shard]:
        raw = self._raw_dir(cfg)
        corpus_shards = int(cfg.blocks["execution"]["corpus_shards"])
        counts = {f.name: corpus.count_docs(f) for f in sorted(raw.glob(corpus.DOCS_GLOB))}
        # Pool-underutilization guard (warning only, no behaviour change): a shard whose
        # doc count is below ``chunk_workers x chunk_batch_size`` yields fewer ``imap``
        # payloads than pool processes, leaving some idle for that shard (measured live:
        # a 60-document shard used one of the pool's processes). A sizing note, not an error:
        # tiny fixtures/dev corpora trip this legitimately.
        total = sum(counts.values())
        per = max(1, math.ceil(total / corpus_shards)) if total else 1  # == _shard_slices cap
        pool_floor = _chunk_workers(cfg) * _chunk_batch_size(cfg)
        if total and per < pool_floor:
            _log.warning(
                "preprocess.chunk.shards.pool_underutilized",
                docs_per_shard=per,
                chunk_workers=_chunk_workers(cfg),
                chunk_batch_size=_chunk_batch_size(cfg),
                pool_floor=pool_floor,
            )
        specs = list(_shard_slices(counts, corpus_shards))
        self.presplit(cfg, specs)  # materialize each shard's docs once, here in `seed`
        for fname, start, end in specs:
            yield _Shard(
                name=corpus.shard_stem(fname, start, end),
                payload={"docs_file": fname, "start": start, "end": end},
            )

    def presplit(self, cfg: RunConfig, specs: Sequence[tuple[str, int, int]]) -> None:
        """Materialise every shard's documents into its own slice file, one pass per language.

        Called from :meth:`shards`, i.e. from ``saturate.seed``, which runs exactly once on
        one machine before any worker claims a shard. Without this, every worker streamed the
        whole language file from byte 0 to reach its offset, which dominated the round: it
        cost tens of seconds per shard and hundreds of gigabytes of concurrent reads per
        shard-round on a large array, against a few minutes of real chunking.

        Idempotent and resumable: :func:`corpus.write_native_slices` skips any slice that
        already carries its ``_SUCCESS`` marker and regenerates a truncated/missing one, so a
        preempted seed simply continues. It is never fatal: a language whose split fails is
        logged and left to ``work``'s ``read_native_slice`` fallback, so the corpus build
        degrades to today's slow path instead of stopping.
        """
        raw = self._raw_dir(cfg)
        out = self._slices_dir(cfg)
        by_file: dict[str, list[tuple[int, int]]] = {}
        for fname, start, end in specs:
            by_file.setdefault(fname, []).append((start, end))
        for fname, ranges in sorted(by_file.items()):
            try:
                written = corpus.write_native_slices(raw / fname, ranges, out)
            except OSError as exc:
                _log.warning(
                    "preprocess.chunk.slices.split_failed", file=fname, error=str(exc)
                )
                continue
            _log.info(
                "preprocess.chunk.slices.materialized",
                file=fname,
                written=len(written),
                shards=len(ranges),
                dir=str(out),
            )

    def _shard_docs(self, ctx: _ChunkCtx, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """This shard's documents: its own small slice file, else the whole-file slow path.

        The slice written by :meth:`presplit` holds exactly the records
        ``read_native_slice(file, start, end)`` yields, in the same order, so the two paths
        are interchangeable and a shard already completed via the fallback stays valid. The
        fallback is taken only when the slice is missing or unfinalized (a queue seeded before
        slices existed, or a failed split) and is logged, so the slow path is never silent.
        """
        name = corpus.slice_filename(payload["docs_file"], payload["start"], payload["end"])
        if ctx.slices_dir is not None:
            sl = ctx.slices_dir / name
            if is_done(sl):
                return corpus.read_native(sl)
        _log.warning(
            "preprocess.chunk.slice.fallback",
            slice=name,
            slices_dir=str(ctx.slices_dir) if ctx.slices_dir else None,
            docs_file=payload["docs_file"],
            start=payload["start"],
            end=payload["end"],
        )
        return corpus.read_native_slice(
            ctx.raw_dir / payload["docs_file"], payload["start"], payload["end"]
        )

    def work(self, ctx: _ChunkCtx, shard: Path) -> Path:
        payload = _read_shard_payload(shard)
        docs = self._shard_docs(ctx, payload)
        if ctx.workers > 1:
            if ctx.pool is None:
                # Normal production path builds the pool in `bringup` (single-threaded
                # parent, see _pool_context); this is the fallback for a hand-built ctx.
                ctx.pool = _ChunkPool(ctx.workers, ctx.segmenter_model, ctx.tokenizer_id)
            records = ctx.pool.segment_documents(
                docs, batch_size=ctx.batch_size, **ctx.options
            )
        else:
            records = segment_documents(
                docs, ctx.segmenter, ctx.tokenizer, batch_size=ctx.batch_size, **ctx.options
            )
        out = saturate.shard_out_path(shard)
        # One nested record per document (document fields plus its sentences list): the
        # shard is written straight from the streaming segmenter, never materialised.
        write_jsonl(out, records)
        return out

    def validate(self, path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> None:
        """Fold the shard JSONL into the two family tables, documents and sentences.

        Lexicographic sort restores global doc order because shard names are
        ``<stem>_<start:09d>_<end:09d>``: zero-padded offsets order numerically and the
        four fixed ``{eng,spa,rus,zho}-docs`` stems are single-token. A future corpus-naming
        scheme with variable-width offsets or multi-token stems would need a numeric key.

        The JSONL is streamed into Parquet in constant memory: shard records are read one line
        at a time in shard order and handed to ``write_parquet_stream``, which holds only
        one row group (``PARQUET_ROW_GROUP_SIZE`` rows) at a time, so the multi-million
        document corpus is never materialised as Python dicts. The same temp-rename and
        ``_SUCCESS`` skip-if-done semantics apply. Two independent passes over the shard files
        (one per table): each is a pure projection of the same record stream, so the two
        tables cannot disagree, and re-reading transient shard JSONL sequentially is far
        cheaper than buffering either table.

        The container changes at the merge boundary:
        per-shard outputs stay
        append-friendly JSONL, while the long-lived spine every downstream stage scans is
        columnar.
        """
        layout = self._layout(cfg)
        ordered = sorted(shard_paths)
        write_parquet_stream(
            layout.documents_path(),
            _document_rows(_iter_shard_records(ordered)),
            schema=document_arrow_schema(),
        )
        write_parquet_stream(
            layout.sentences_path(),
            _sentence_rows(_iter_shard_records(ordered)),
            schema=sentence_arrow_schema(),
        )


def _iter_shard_records(shard_paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    """Stream every shard output's JSONL records, file by file, in the given order.

    One line is parsed at a time (never a whole shard, let alone the whole corpus), so
    the merge's memory is the row-group buffer downstream of it, nothing more. Blank
    lines are skipped, mirroring ``common.io.read_jsonl``'s tolerance.
    """
    import json

    for src in shard_paths:
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _read_shard_payload(shard: Path) -> dict[str, Any]:
    import json

    return json.loads(shard.read_text(encoding="utf-8").strip())
