"""Post-chunk sentence merge into the translation-unit map; the sentence spine is untouched.

A SaT sentence shorter than ``min_sentence_tokens`` content tokens (a headline, a byline, a
list item, a label) is not a proposition, and on its own it is a poor MT unit: measured on
the native spine, a sixth of it was pathologically short. This stage groups such sentences
with a structural neighbour into a merge unit, the string the translate stage hands to MT,
and writes the grouping out as a merge map with one row per non-English ``sentence_id``.

Three properties make this a separate stage rather than a chunker step:

1. The sentence spine is frozen. Merging inside ``chunk`` would change ``chunker_hash``,
   which is a path level of the family-shared corpus directory, forcing a full re-chunk for
   a rule that changes no sentence boundary. This stage is keyed by its own hash
   (``all_hashes(cfg)["merge"]``) under ``Layout.merge_map_path(merge_hash)``, a path that
   expresses dependence on both the spine and the rule while invalidating neither.
2. ``sentence_id`` remains the citation unit. The merge unit is a translation unit
   only, and the map is additive: a consumer that ignores it sees exactly the spine.
3. English is excluded. It is identity pass-through and never translated, so merging it
   would cost citation granularity for no benefit; no row is emitted for ``lang == "en"``.

Because ``paragraph_index`` is a stored per-sentence column, this stage reads no source
file: two Parquet tables in, one out. Merges are paragraph-local, governed by
``merge_cross_paragraph``; they are not passage-local, because passages do not exist at this
point in the build (their sole producer is the `packing` substage).

The stage runs as one :class:`~ragtime.orchestration.saturate.StageAdapter`
(:class:`MergeAdapter`) on the CPU work-queue template: no model, no GPU, no new driver.
Shards are contiguous, document-atomic row ranges over ``sentences.parquet``, aligned onto
``documents.parquet`` by :mod:`ragtime.preprocess.spine` and resolved through Parquet's own
footer index, so nothing is pre-split onto disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import Layout, Statistics, get_logger
from ragtime.common.io import concat_parquet, is_done, iter_parquet, write_parquet_stream
from ragtime.common.schemas import merge_map_arrow_schema
from ragtime.config import all_hashes
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family

from . import spine
from .tokenizer import load_pack_tokenizer_by_id

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from ragtime.config import RunConfig

__all__ = [
    "STAT_MERGED",
    "STAT_UNMERGEABLE",
    "MergeAdapter",
    "MergeSentence",
    "document_sentences",
    "merge_document",
    "merge_hash",
    "merge_map_violation",
    "merge_options",
    "merge_short_sentences",
]

_log = get_logger("preprocess.merge")

_DEFAULT_BASE = "runs"  # real runs pass an explicit Layout base

# Counter ids (common.stats), sliced by lang only.
STAT_MERGED = "merge.short_sentence.merged"
STAT_UNMERGEABLE = "merge.short_sentence.unmergeable"

# The `merge` config block's defaults, used only as fallbacks: every shipped config
# carries the block byte-identically, and the block's own hash keys the artifact.
_DEFAULT_MERGE_SHORT = True
_DEFAULT_MIN_SENTENCE_TOKENS = 16
_DEFAULT_MERGE_CAP_FRAC = 0.15
_DEFAULT_MERGE_CROSS_PARAGRAPH = "singleton_only"
_DEFAULT_MERGE_LANGUAGES: tuple[str, ...] = ("zh", "ru", "es")
_MERGE_CROSS_PARAGRAPH_VALUES = frozenset({"singleton_only", "never"})

# Execution shape (the non-shared `execution` block): how many row-range shards the queue is
# seeded with. The worker-array width is derived from it by `orchestration.cli`, which owns
# the shards-to-workers oversubscription for every corpus substage.
_DEFAULT_MERGE_SHARDS = 120

_HASH_DIRLEN = 12  # the same truncation Layout uses for hash path levels

# The sentence columns this stage reads. `paragraph_index` is a stored per-sentence label,
# which is why no raw-document re-derivation is needed here.
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


# --------------------------------------------------------------------------- #
# One document's input to the rule.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MergeSentence:
    """One atomic input sentence, as the rule sees it: a span plus its structure.

    There is no text field: the sentence is a span of the document text, and a merged unit's
    source text is ``document_text[first.start:last.end]`` because a unit is a run of
    consecutive sentences, since sentence indices are dense in document order. So the rule
    never fabricates a joiner: whatever separator really sits between two sentences in the
    normalized document (one space inside a Latin paragraph, nothing between CJK sentences,
    the paragraph joiner across the ``singleton_only`` exception) is what gets counted and
    what gets translated.
    """

    sentence_id: str
    sentence_index: int
    start: int
    end: int
    paragraph_index: int
    token_count: int


@dataclass(frozen=True, slots=True)
class MergeUnitRange:
    """A merge-in-progress unit: the inclusive constituent range it covers, plus its count.

    Inclusive indices into the document's sentence list are what make the structural
    predicates (:func:`_first_of_para` / :func:`_last_of_para`) exact for a fused unit as
    well as for a lone sentence.
    """

    start: int
    end: int
    count: int
    para: int


def _first_of_para(sents: Sequence[MergeSentence], start: int) -> bool:
    """Does the unit starting at ``start`` open its paragraph?"""
    return start == 0 or sents[start - 1].paragraph_index != sents[start].paragraph_index


def _last_of_para(sents: Sequence[MergeSentence], end: int) -> bool:
    """Does the unit ending at ``end`` close its paragraph?"""
    return (
        end == len(sents) - 1
        or sents[end + 1].paragraph_index != sents[end].paragraph_index
    )


def merge_short_sentences(
    sents: Sequence[MergeSentence],
    text: str,
    count_tokens: Any,
    *,
    min_tokens: int,
    cap: float,
    cross_paragraph: str,
) -> tuple[list[MergeUnitRange], int, int]:
    """Fuse sub-threshold sentences into a neighbour, defining the translation unit.

    Called once per document over that document's dense sentence list, using the stored
    ``paragraph_index``. Paragraph membership is load-bearing: an ablation showed it changes
    the outcome for most documents.

    The rule (all four knobs live in the fairness-shared, separately hashed ``merge`` block,
    so the behaviour is config-driven and folds into ``merge_hash``, never into
    ``chunker_hash``):

    - Threshold ``min_sentence_tokens``, calibrated on a threshold sweep over the real
      corpus: at 16 the pathological short-sentence share collapses to under a percent,
      nothing above 16 improves it, and 16 preserves more spine resolution than 20 while
      reducing the chance that a unit swallows an unrelated headline or list item.
    - Cap ``merge_cap_frac``, a fraction of the chunker's content budget. It is re-counted
      on the fused source text with the tokenizer rather than summed over the parts, which
      would drift against the cap; the comparison is ``count > cap``, so a unit of exactly
      the cap is allowed. If no permitted direction fits, the sentence is left unmerged and
      counted under :data:`STAT_UNMERGEABLE` rather than force-merged. A lone sentence's
      count is the stored ``token_count`` from the same pinned tokenizer, so only fused
      candidates are ever re-tokenized.
    - Direction, structural and deterministic with no NLP heuristic: a unit that opens its
      paragraph merges with the next one, since its own paragraph continues after it; any
      other unit merges with the previous one. A unit that both opens and closes its
      paragraph (for a lone sentence, it is the whole paragraph: the heading or label case)
      tries next first and falls back to previous when it is document-final or when merging
      forward would bust the cap.
    - Cross-paragraph behaviour, ``merge_cross_paragraph="singleton_only"``: a paragraph
      boundary is crossed only in that paragraph-aligned case, a deliberate narrow exception
      to the paragraph-first structural split. In every other case the permitted direction is
      same-paragraph by construction. ``"never"`` disables the exception entirely.

    One deterministic left-to-right pass with bounded re-absorption: a fused unit still under
    ``min_tokens`` re-enters the rule, re-evaluated at its new boundaries, and may merge
    again. It terminates structurally, because every merge either consumes one more original
    sentence or pops one already-emitted unit, independently of whether the token count
    strictly increases. There is no randomness and no lookahead beyond the immediate
    neighbour, so the output is a pure function of this document's sentence list.

    Returns ``(units, merged, unmergeable)``.
    """
    if cross_paragraph not in _MERGE_CROSS_PARAGRAPH_VALUES:
        raise ValueError(
            f"merge_cross_paragraph={cross_paragraph!r}; "
            f"expected one of {sorted(_MERGE_CROSS_PARAGRAPH_VALUES)}"
        )
    allow_cross = cross_paragraph == "singleton_only"
    n = len(sents)
    out: list[MergeUnitRange] = []
    merged = 0
    unmergeable = 0

    def fused_count(a: int, b: int) -> int:
        """Content tokens of the fused source text of constituents ``a..b``, inclusive."""
        return count_tokens(text[sents[a].start : sents[b].end])

    i = 0
    while i < n:
        u = MergeUnitRange(i, i, sents[i].token_count, sents[i].paragraph_index)
        while u.count < min_tokens:
            # Re-evaluated at the unit's current boundaries, so re-absorption follows the
            # same structural rule the first merge did.
            opens = _first_of_para(sents, u.start)
            closes = _last_of_para(sents, u.end)
            aligned = opens and closes  # a lone sentence is the whole paragraph
            if aligned:
                directions: tuple[str, ...] = ("next", "prev") if allow_cross else ()
            elif opens:
                directions = ("next",)  # same-paragraph by construction (it does not close)
            else:
                directions = ("prev",)  # same-paragraph by construction (it does not open)
            for direction in directions:
                if direction == "next":
                    if u.end + 1 >= n:
                        continue
                    count = fused_count(u.start, u.end + 1)
                    if count > cap:
                        continue
                    u = MergeUnitRange(u.start, u.end + 1, count, u.para)
                else:
                    if not out:
                        continue
                    prev = out[-1]
                    count = fused_count(prev.start, u.end)
                    if count > cap:
                        continue
                    out.pop()
                    u = MergeUnitRange(prev.start, u.end, count, prev.para)
                merged += 1
                break
            else:  # no permitted direction fit: emit the short unit and count it
                unmergeable += 1
                break
        out.append(u)
        i = u.end + 1
    return out, merged, unmergeable


# --------------------------------------------------------------------------- #
# One document to merge-map rows.
# --------------------------------------------------------------------------- #
def merge_document(
    sents: Sequence[MergeSentence],
    text: str,
    document_id: str,
    lang: str,
    count_tokens: Any,
    *,
    min_tokens: int,
    cap: float,
    cross_paragraph: str,
    stats: Statistics | None = None,
) -> list[dict[str, Any]]:
    """Merge-map rows for one document: one row per ``sentence_id``, in spine order.

    Every input sentence produces exactly one row, since a sentence that merged with nobody
    is still a merge unit of size 1, so the map is total over the document's sentences and a
    consumer never has to guess a missing entry. ``merge_unit_token_count`` is stamped on
    every constituent row (the same number for all of a unit's rows), so reconciliation can
    decide fuse-vs-split-back from a stored count and never loads a tokenizer.
    """
    if not sents:
        return []
    units, merged, unmergeable = merge_short_sentences(
        sents,
        text,
        count_tokens,
        min_tokens=min_tokens,
        cap=cap,
        cross_paragraph=cross_paragraph,
    )
    if stats is not None:
        if merged:
            stats.emit(STAT_MERGED, float(merged), lang=lang)
        if unmergeable:
            stats.emit(STAT_UNMERGEABLE, float(unmergeable), lang=lang)
    rows: list[dict[str, Any]] = []
    for u in units:
        n_const = u.end - u.start + 1
        unit_id = sents[u.start].sentence_id  # the first constituent's id
        for j in range(n_const):
            s = sents[u.start + j]
            rows.append(
                {
                    "sentence_id": s.sentence_id,
                    "document_id": document_id,
                    "sentence_index": s.sentence_index,
                    "lang": lang,
                    "start": s.start,
                    "end": s.end,
                    "merge_unit_id": unit_id,
                    "merge_constituent_index": j,
                    "merge_constituent_count": n_const,
                    "merge_unit_token_count": u.count,
                }
            )
    return rows


def document_sentences(rows: Iterable[Mapping[str, Any]]) -> list[MergeSentence]:
    """Convert one document's sentence rows into the rule's input, in stored order."""
    return [
        MergeSentence(
            sentence_id=r["sentence_id"],
            sentence_index=int(r["sentence_index"]),
            start=int(r["start"]),
            end=int(r["end"]),
            paragraph_index=int(r["paragraph_index"]),
            token_count=int(r["token_count"]),
        )
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# Producer-side structural check: the consumer's CorruptMergeMapError conditions,
# asserted at write time so a bad shard fails here rather than at translation time.
# --------------------------------------------------------------------------- #
def merge_map_violation(rows: Iterable[Mapping[str, Any]]) -> str | None:
    """The first structural violation in a merge-map row stream, or ``None`` if clean.

    Checks exactly what ``preprocess.translate_omt`` re-checks before trusting the map (it
    raises ``CorruptMergeMapError`` on each): a declared ``merge_constituent_count`` that
    disagrees with the members present, a unit spanning two documents, constituent indices
    that are not ``0..n-1`` in spine order, a unit not named after its first constituent, a
    duplicate ``sentence_id``, and a ``merge_unit_id`` that reappears after its contiguous
    run ended. All hold by construction in :func:`merge_document`, so this is a cheap guard
    against a truncated or interleaved write, not a second implementation of the rule.

    Streamed in one pass: only the current unit's rows, the seen-sentence set and the
    closed-unit set are held. Rows must arrive in the artifact's own (spine) order, which is
    what makes a unit's rows contiguous and the "reappears later" check meaningful.
    """
    seen_sentences: set[str] = set()
    closed_units: set[str] = set()
    cur_id: str | None = None
    cur: list[Mapping[str, Any]] = []

    def _close(unit: list[Mapping[str, Any]]) -> str | None:
        n = len(unit)
        declared = {int(r["merge_constituent_count"]) for r in unit}
        if declared != {n}:
            return (
                f"unit {unit[0]['merge_unit_id']!r} declares "
                f"merge_constituent_count={sorted(declared)} but has {n} member(s)"
            )
        if [int(r["merge_constituent_index"]) for r in unit] != list(range(n)):
            return (
                f"unit {unit[0]['merge_unit_id']!r} has constituent indices "
                f"{[int(r['merge_constituent_index']) for r in unit]}, expected 0..{n - 1}"
            )
        documents = {r["document_id"] for r in unit}
        if len(documents) > 1:
            return f"unit {unit[0]['merge_unit_id']!r} spans documents {sorted(documents)}"
        if unit[0]["merge_unit_id"] != unit[0]["sentence_id"]:
            return (
                f"unit {unit[0]['merge_unit_id']!r} is not named after its first "
                f"constituent {unit[0]['sentence_id']!r}"
            )
        return None

    for row in rows:
        sid = row["sentence_id"]
        if sid in seen_sentences:
            return f"duplicate sentence_id {sid!r}"
        seen_sentences.add(sid)
        uid = row["merge_unit_id"]
        if uid != cur_id:
            if cur_id is not None:
                err = _close(cur)
                if err:
                    return err
                closed_units.add(cur_id)
            if uid in closed_units:
                return f"merge_unit_id {uid!r} reappears after its contiguous run ended"
            cur_id, cur = uid, []
        cur.append(row)
    if cur_id is not None:
        return _close(cur)
    return None


# --------------------------------------------------------------------------- #
# Config reads (the `merge` block is hashed on its own; `execution` is not hashed).
# --------------------------------------------------------------------------- #
def merge_options(cfg: RunConfig) -> dict[str, Any]:
    """The semantic merge knobs, read from the fairness-shared ``merge`` block."""
    m = cfg.blocks.get("merge", {})
    langs = m.get("merge_languages", _DEFAULT_MERGE_LANGUAGES)
    return {
        "enabled": bool(m.get("merge_short_sentences", _DEFAULT_MERGE_SHORT)),
        "min_tokens": int(m.get("min_sentence_tokens", _DEFAULT_MIN_SENTENCE_TOKENS)),
        "cap_frac": float(m.get("merge_cap_frac", _DEFAULT_MERGE_CAP_FRAC)),
        "cross_paragraph": str(
            m.get("merge_cross_paragraph", _DEFAULT_MERGE_CROSS_PARAGRAPH)
        ),
        "languages": tuple(langs),
    }


def merge_hash(cfg: RunConfig) -> str:
    """The semantic ``merge``-block hash; keys the merge map, never ``chunker_hash``."""
    return all_hashes(cfg)["merge"]


def merge_shards(cfg: RunConfig) -> int:
    """Number of row-range shards to seed (non-shared ``execution`` knob)."""
    return int(cfg.blocks.get("execution", {}).get("merge_shards", _DEFAULT_MERGE_SHARDS))


# --------------------------------------------------------------------------- #
# The saturate.StageAdapter (CPU template, no model, no GPU).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _MergeCtx:
    tokenizer: Any
    documents: Path
    sentences: Path
    min_tokens: int
    cap: float
    cross_paragraph: str
    languages: tuple[str, ...]
    enabled: bool


@dataclass(slots=True)
class _MergeShard:
    name: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class MergeAdapter:
    """The merge substage as a ``saturate.StageAdapter`` (CPU, no Apptainer, no model).

    ``stage`` embeds the ``merge`` block's own hash, so a rule edit gets a fresh work queue
    and a fresh artifact dir without disturbing ``chunker_hash`` or the sentence spine.
    Construct it with :meth:`for_config` so the hash is always the config's, never a literal.
    """

    merge_hash: str
    base: str | None = None
    template: str = "workqueue_worker_cpu.sbatch"
    stage: str = field(init=False)

    def __post_init__(self) -> None:
        self.stage = f"merge_{self.merge_hash[:_HASH_DIRLEN]}"

    @classmethod
    def for_config(cls, cfg: RunConfig, *, base: str | None = None) -> MergeAdapter:
        """Build the adapter for ``cfg``; the sanctioned construction path."""
        return cls(merge_hash=merge_hash(cfg), base=base)

    # -- paths ---------------------------------------------------------------- #
    def _layout(self, cfg: RunConfig) -> Layout:
        root = self.base if self.base is not None else _DEFAULT_BASE
        return Layout(
            run_dir=root,
            base=root,
            family=run_family(cfg),
            chunker_hash=all_hashes(cfg)["chunker"],
        )

    def map_path(self, cfg: RunConfig) -> Path:
        """Return ``<corpus_dir>/merge_map/<merge_hash12>/data.parquet``, the family artifact.

        Resolved through ``Layout.merge_map_path``, the shared path builder the consumer
        (``preprocess.translate_omt``) reads the map with, so producer and consumer cannot
        drift on the filename.
        """
        return self._layout(cfg).merge_map_path(self.merge_hash)

    def documents_path(self, cfg: RunConfig) -> Path:
        """The frozen normalized document text this map's spans point into (read-only)."""
        return self._layout(cfg).documents_path()

    def sentences_path(self, cfg: RunConfig) -> Path:
        """The frozen sentence spine this map is derived from (read-only)."""
        return self._layout(cfg).sentences_path()

    # -- StageAdapter --------------------------------------------------------- #
    def bringup(self, cfg: RunConfig) -> _MergeCtx:
        """Stand up the per-worker resident state once: the pinned pack tokenizer.

        No model and no GPU: the merge rule counts with the same pinned ``PackTokenizer``
        the chunker packed with (``chunker.config.tokenizer_id``), so "16 content tokens"
        means the identical thing here and there.
        """
        c = cfg.blocks["chunker"]["config"]
        opts = merge_options(cfg)
        tokenizer = load_pack_tokenizer_by_id(str(c["tokenizer_id"]))
        content_budget = int(c["token_budget"]) - tokenizer.num_special()
        return _MergeCtx(
            tokenizer=tokenizer,
            documents=self.documents_path(cfg),
            sentences=self.sentences_path(cfg),
            min_tokens=opts["min_tokens"],
            cap=opts["cap_frac"] * content_budget,
            cross_paragraph=opts["cross_paragraph"],
            languages=opts["languages"],
            enabled=opts["enabled"],
        )

    def shards(self, cfg: RunConfig) -> Iterator[_MergeShard]:
        """Contiguous, document-atomic row ranges over ``sentences.parquet``."""
        opts = merge_options(cfg)
        if not opts["enabled"]:
            _log.info("preprocess.merge.shards.disabled", stage=self.stage)
            return
        sents = self.sentences_path(cfg)
        ranges, boundary_docs = spine.plan_shards(sents, n_shards=merge_shards(cfg))
        aligned = spine.align_documents(self.documents_path(cfg), ranges, boundary_docs)
        _log.info(
            "preprocess.merge.shards",
            stage=self.stage,
            shards=len(aligned),
            languages=list(opts["languages"]),
        )
        for r in aligned:
            yield _MergeShard(name=r.name, payload=r.payload())

    def work(self, ctx: _MergeCtx, shard: Path) -> Path:
        """Emit this shard's merge-map rows: two Parquet reads, no source file touched.

        The shard's own documents are loaded once (bounded by the shard, not the corpus)
        and its sentence rows are streamed in document order; each non-English document is
        merged from its stored ``paragraph_index``. An out-of-scope (English) document
        contributes no rows by design: the exclusion is a property of the rule, and the
        stage says so with a counter rather than by silence.
        """
        r = spine.ShardRange.from_payload(_read_shard_payload(shard))
        out = saturate.shard_out_path(shard)
        stats = Statistics()
        texts = spine.load_document_texts(ctx.documents, r)
        rows: list[dict[str, Any]] = []
        n_docs = n_skipped = 0
        langs_seen: set[str] = set()

        sentence_rows = spine.iter_row_range(
            ctx.sentences, r.row_start, r.row_end, columns=list(_SENTENCE_COLUMNS)
        )
        for doc_id, doc_rows in spine.group_by_document(sentence_rows):
            lang = doc_rows[0]["lang"]
            if lang not in ctx.languages:
                n_skipped += 1  # English and any other out-of-scope language emits no rows
                continue
            text = texts.get(doc_id)
            if text is None:
                raise LookupError(
                    f"{doc_id}: no row in documents.parquet for a document whose sentences "
                    "are in this shard; the two tables are not from the same corpus build"
                )
            n_docs += 1
            langs_seen.add(lang)
            rows.extend(
                merge_document(
                    document_sentences(doc_rows),
                    text,
                    doc_id,
                    lang,
                    ctx.tokenizer.count,
                    min_tokens=ctx.min_tokens,
                    cap=ctx.cap,
                    cross_paragraph=ctx.cross_paragraph,
                    stats=stats,
                )
            )
        write_parquet_stream(out, rows, schema=merge_map_arrow_schema())
        _log.info(
            "preprocess.merge.shard.done",
            shard=shard.name,
            langs=sorted(langs_seen),
            documents=n_docs,
            skipped_out_of_scope=n_skipped,
            rows=len(rows),
            merged=stats.total(STAT_MERGED),
            unmergeable=stats.total(STAT_UNMERGEABLE),
        )
        return out

    def validate(self, path: Path) -> bool:
        """Gate ``mark_done`` on the artifact's ``_SUCCESS`` and on its structure.

        It does not check size: a merge shard legitimately emits zero rows (an all-English
        range), so "size > 0" would poison a correct shard. On top of the marker the shard
        is read back and run through :func:`merge_map_violation`, which asserts exactly the
        conditions ``preprocess.translate_omt`` raises ``CorruptMergeMapError`` on. They all
        hold by construction, so a violation means a truncated or interleaved write, and
        catching it here makes it an ordinary retriable shard failure instead of a
        corpus-wide translation abort hours later.
        """
        if not (path.exists() and is_done(path)):
            return False
        violation = merge_map_violation(iter_parquet(path))
        if violation is not None:
            _log.warning(
                "preprocess.merge.shard.corrupt", shard=path.name, violation=violation
            )
            return False
        return True

    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> None:
        """Concatenate the per-shard Parquet outputs into one merge map, Arrow-native.

        Shard names are ``rows_<start:012d>_<end:012d>``, so sorting them lexicographically
        restores the sentence spine's own row order, the co-ordering that lets the
        translate stage read a contiguous map row range per shard instead of building an
        index over tens of millions of keys.
        """
        out = self.map_path(cfg)
        concat_parquet(out, sorted(shard_paths), schema=merge_map_arrow_schema())
        _log.info(
            "preprocess.merge.done", stage=self.stage, shards=len(shard_paths), path=str(out)
        )


def _read_shard_payload(shard: Path) -> dict[str, Any]:
    return json.loads(shard.read_text(encoding="utf-8").strip())
