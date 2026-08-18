"""The rendering-invariant sentence length ``len_max``, and its additive sidecar.

The packer's job is to make "a passage fits the retrieval window" true. Measuring a
sentence in the native text alone leaves a passage that fits in Chinese free to overflow
once it is read in English: corpus-wide, a percent or so of passages exceeded the window in
at least one rendering, concentrated almost entirely in Chinese. That is a per-language
defect introduced by the packing method, the class of asymmetry the uniform-method rule
exists to prevent.

The fix is to change what a sentence's length means::

    len_max(s) = max( len(tok(documents.text[s.start:s.end])),   # native
                      len(tok(translations/omt[s].text)),        # high tier, NLLB
                      len(tok(translations/omt_opus[s].text)) )  # low tier, OPUS-MT

with ``tok`` the retrieval tokenizer. Packing against that makes "every passage fits the
budget in every rendering" a consequence rather than a hope: a passage's length in rendering
*r* is the sum of its members' ``len_r``, which is at most the sum of their ``len_max``,
which the packer keeps within the content budget. This is why rendering-invariant packing
replaces encode-time windowing rather than complementing it.

One tokenizer serves both roles. The late-interaction checkpoint and ``BAAI/bge-m3`` ship a
byte-identical ``sentencepiece.bpe.model``, verified by comparing lengths and token ids over
tens of thousands of real strings, so the chunker's own pinned
:class:`~ragtime.preprocess.tokenizer.PackTokenizer` is the retrieval tokenizer and no
second tokenizer identity enters the build. Which identity was used is still recorded in
``packing.len_max_tokenizer_id`` and keyed into the sidecar's path.

The result is an additive sidecar keyed by the final sentence id. A new column on
``final/<recon12>/sentences.parquet`` would re-key that node and orphan by path
everything hanging off it, including both renderings' translations; a separate table
costs no churn.

Its path level is ``recon12``, the inventory key, never ``pack12``. ``recon12`` is decided by
fusion and by nothing else, which is why the packing knobs live in their own ``packing``
block: a length is a property of a sentence, and no packing knob can change which sentences
exist. So a re-pack finds the sidecar it already paid for under
``<corpus_dir>/sentence_len_max/<recon12>-<lenmax12>/`` with no re-measurement. The pairing
is not merely assumed: the packing stage asserts the sidecar's ids equal the final ids it
reads, document by document.

The translation tables are subsequences rather than row-aligned copies. Under
``reconcile.store_identity_translations: false`` an English sentence has no translation row
at all, since English is not translated and an identity copy of its own span was a third of
each table. So ``translations/<variant>.parquet`` holds exactly the non-English final
sentences, in final order, and this stage joins it to the inventory by a forward co-walk
(``reconcile.align_subtable`` resolves each shard's row range once, at seed) rather than by
reading the same row index in both files. Every step of that walk is checked.

English identity is then structural rather than asserted: with no row stored, all three
lengths are the same integer read from the same ``token_count`` column, so a violation is
unconstructible. Two row-level checks remain:

- an English sentence that does have a translation row gets the full check, the row being
  tokenized and its length required to equal the native one, so the legacy
  ``store_identity_translations: true`` policy loses nothing;
- a non-English sentence with no row raises :class:`LenMaxMismatchError` rather than
  silently taking its native, untranslated length as its maximum.

A small irreducible tail of sentences exceeds the content budget on their own. A packer
cannot split a sentence, so those passages stay over budget whatever the budget is; they are
emitted as ordinary ``is_oversized`` passages plus a counter (:data:`STAT_OVER_BUDGET`).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import Layout, Statistics, get_logger
from ragtime.common.io import concat_parquet, is_done, parquet_row_group_sizes, write_parquet_stream
from ragtime.common.passage_store import IDENTITY_LANG, RENDERINGS
from ragtime.config import all_hashes, config_hash
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family

from . import spine
from .reconcile import align_subtable, reconcile_hash
from .tokenizer import load_pack_tokenizer_by_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Sequence

    from ragtime.config import RunConfig

__all__ = [
    "NATIVE_RENDERING",
    "EnglishIdentityLengthError",
    "LenMaxAdapter",
    "LenMaxMismatchError",
    "LenMaxShard",
    "len_max_arrow_schema",
    "len_max_hash",
    "len_max_options",
    "len_max_rows",
    "len_max_shards",
]

_log = get_logger("preprocess.len_max")

_DEFAULT_BASE = "runs"
_HASH_DIRLEN = 12

#: The rendering that is not a translation table: its length is already stored, exactly, as
#: ``sentences.token_count``: the chunker counted the native span with this same pinned
#: tokenizer, and a fused sentence carries the merge map's re-count of the fused text. So
#: the native leg of the max costs no tokenization at all.
NATIVE_RENDERING = "original"

# Counter ids (common.stats), sliced by lang: this stage emits, monitoring rolls up.
STAT_ROWS = "len_max.rows"
STAT_GREW = "len_max.grew_over_native"
STAT_OVER_BUDGET = "len_max.exceeds_content_budget"
STAT_ENGLISH_CHECKED = "len_max.english_identity_checked"

_DEFAULT_LEN_MAX_SHARDS = 200

#: Rows tokenized per batch. The pinned tokenizer's batch encoder is Rust/rayon-parallel, so
#: the batch wants to be big enough to saturate the worker's cores and small enough that a
#: shard never holds more than a few tens of MB of translated text at once.
_TOKENIZE_BATCH = 20_000

_SENTENCE_COLUMNS = ("sentence_id", "document_id", "lang", "token_count")
_TRANSLATION_COLUMNS = ("sentence_id", "text")


class LenMaxMismatchError(RuntimeError):
    """The final tables this stage reads do not co-walk (or are not the same build).

    ``final/translations/<variant>.parquet`` is a subsequence of ``final/sentences.parquet``,
    the non-English final sentences in final order, so the two are joined by a forward
    co-walk, and every step is checked rather than assumed. Two conditions raise: a
    translation row that does not key to the sentence it is being consumed for (a silent
    misalignment would attach one sentence's translated length to another, and the resulting
    packing would be wrong in a way no downstream check could see), and a non-English
    sentence with no translation row at all (whose native length would then be taken as its
    maximum, quietly reinstating the per-language overflow this stage exists to remove).
    """


class EnglishIdentityLengthError(RuntimeError):
    """A stored English "translation" is not its own source span.

    Kept, and still raised, for the legacy ``store_identity_translations: true`` policy: when
    an identity row exists it is tokenized like any other and its length must equal the
    native one, because reconciliation stores it byte-identically to
    ``documents.text[start:end]``.

    Under the shipped policy no such row exists, and the property it guarded becomes
    unconstructible rather than unchecked: all three of an English sentence's lengths are the
    same ``token_count`` integer, read once. The check that replaces it is the row-level one:
    an English sentence's translation row must be absent or identical, never a third thing.
    """


# --------------------------------------------------------------------------- #
# The measurement-recipe key. There is no second inventory key: `reconcile` holds
# the fusion knobs and nothing else, so recon12 already names which sentences these are,
# which is why the packing knobs live in their own `packing` block.

# --------------------------------------------------------------------------- #
def len_max_options(cfg: RunConfig) -> dict[str, Any]:
    """The measurement recipe: which tokenizer, and which renderings enter the max.

    The tokenizer defaults to ``chunker.config.tokenizer_id``, the pinned
    ``repo@revision`` the chunker already counts with, which is also the same
    SentencePiece model the late-interaction checkpoint uses.
    ``packing.len_max_tokenizer_id`` exists to make that identity an explicit, hashed part
    of the record instead of an implicit inheritance.

    The renderings come from ``common.passage_store.RENDERINGS``, the shipped vocabulary
    ``config.schema.KNOB_VALUES`` is pinned against, never a literal list here: family
    membership is not something a stage gets to hardcode.
    """
    block = cfg.blocks.get("packing", {})
    default = str(cfg.blocks["chunker"]["config"]["tokenizer_id"])
    return {
        "tokenizer_id": str(block.get("len_max_tokenizer_id") or default),
        "renderings": tuple(RENDERINGS),
    }


def len_max_hash(cfg: RunConfig) -> str:
    """Return ``H({tokenizer_id, renderings})``, the measurement recipe's key.

    A separate path level from ``recon12`` because the two answer different questions:
    *which sentences are these?* and *how were they measured?*. Changing the tokenizer
    re-measures the same inventory, so it must resolve to a fresh sidecar beside the same
    one, never overwrite it and never be silently reused across a re-measure. It mirrors
    ``Layout.translations_raw_path``'s two-hash level.

    It is not the whole ``packing`` block: ``pack_length`` and ``pack_budget``
    change how sentences are grouped, which cannot change any sentence's length. Folding
    them in would re-measure the whole corpus every time the budget moved.
    """
    return config_hash(len_max_options(cfg))


# --------------------------------------------------------------------------- #
# The pinned artifact schema.
# --------------------------------------------------------------------------- #
def len_max_arrow_schema(renderings: Sequence[str] = RENDERINGS) -> Any:
    """Pinned Arrow schema of ``len_max.parquet``: one row per final sentence.

    There is a ``len_<rendering>`` column per rendering plus the derived ``len_max``. The
    per-rendering columns are kept rather than collapsed into the max alone because the point
    of this table is that the asymmetry between renderings is measurable, and re-deriving
    them would mean re-tokenizing every translated string in the corpus.


    ``document_id`` is carried so the table can be sharded and co-walked document-atomically
    like every other corpus table (``preprocess.spine``), and ``lang`` so a per-language
    rollup needs no join.
    """
    import pyarrow as pa

    fields = [
        pa.field("sentence_id", pa.string()),
        pa.field("document_id", pa.string()),
        pa.field("lang", pa.string()),
    ]
    fields += [pa.field(f"len_{name}", pa.int32()) for name in renderings]
    fields.append(pa.field("len_max", pa.int32()))
    return pa.schema(fields)


# --------------------------------------------------------------------------- #
# The pure core: rows in, rows out. No IO, no paths, no work queue.
# --------------------------------------------------------------------------- #
def len_max_rows(
    sent_rows: Sequence[Mapping[str, Any]],
    translations: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer: Any,
    *,
    renderings: Sequence[str] = RENDERINGS,
    content_budget: int | None = None,
    stats: Any = None,
) -> list[dict[str, Any]]:
    """Convert one batch of final sentences into their ``len_max`` rows, order preserved.

    ``translations[variant]`` holds that variant's rows for exactly these sentences, in order,
    as a subsequence: one row per non-English sentence and none for an English one (see
    :func:`_align_translations`). The native length is read off ``token_count`` (already
    exact and already stored); every translated length is tokenized in one batched call per
    variant over only the sentences that have a translation, which is the only reason this
    stage is minutes rather than hours.
    """
    variants = [name for name in renderings if name != NATIVE_RENDERING]
    counted: dict[str, list[int | None]] = {}
    for variant in variants:
        texts = _align_translations(sent_rows, translations.get(variant) or (), variant)
        present = [i for i, t in enumerate(texts) if t is not None]
        lengths = tokenizer.count_batch([str(texts[i]) for i in present])
        scattered: list[int | None] = [None] * len(sent_rows)
        for i, n in zip(present, lengths, strict=True):
            scattered[i] = int(n)
        counted[variant] = scattered

    out: list[dict[str, Any]] = []
    for i, row in enumerate(sent_rows):
        native = int(row["token_count"])
        lang = str(row["lang"])
        lengths = {NATIVE_RENDERING: native}
        for variant in variants:
            got = counted[variant][i]
            # No row => this sentence is not translated (English), so its length in that
            # rendering is its native length, the same integer read once. That is what
            # makes "English is identity" a construction here rather than an assertion.
            lengths[variant] = native if got is None else got
        biggest = max(lengths[name] for name in renderings)
        if lang == IDENTITY_LANG and any(v != native for v in lengths.values()):
            # Only reachable when an identity row was stored, under the legacy policy, and
            # its tokenized length disagrees with the span's.
            raise EnglishIdentityLengthError(
                f"{row['sentence_id']}: English is identity pass-through in both MT arms, so "
                f"its three lengths must be equal; got {lengths}. A stored English "
                "'translation' that is not its own source span is a build defect, not a "
                "length anomaly"
            )
        record = {
            "sentence_id": row["sentence_id"],
            "document_id": row["document_id"],
            "lang": lang,
        }
        record.update({f"len_{name}": lengths[name] for name in renderings})
        record["len_max"] = biggest
        out.append(record)
        if stats is not None:
            stats.emit(STAT_ROWS, lang=lang)
            if lang == "en":
                stats.emit(STAT_ENGLISH_CHECKED, lang=lang)
            if biggest > native:
                stats.emit(STAT_GREW, lang=lang)
            if content_budget is not None and biggest > content_budget:
                stats.emit(STAT_OVER_BUDGET, lang=lang)
    return out


def _align_translations(
    sent_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    variant: str,
) -> list[str | None]:
    """Co-walk one variant's subsequence onto the sentences, ``None`` where it has no row.

    Both streams are in final order and ``rows`` is a subsequence of ``sent_rows``, so a
    single forward walk behind one cursor is the whole join. Which sentences may be skipped
    is not left to the data: only :data:`~ragtime.common.passage_store.IDENTITY_LANG` may
    lack a row (it is not translated), and a row that turns up for a sentence it does not key
    to is a misalignment rather than a skip. Leftovers at the end fail too: they are rows
    belonging to sentences this batch does not contain.
    """
    out: list[str | None] = []
    j = 0
    for sent in sent_rows:
        sid = sent["sentence_id"]
        if j < len(rows) and rows[j]["sentence_id"] == sid:
            out.append(str(rows[j]["text"] or ""))
            j += 1
            continue
        if str(sent["lang"]) != IDENTITY_LANG:
            nxt = rows[j]["sentence_id"] if j < len(rows) else None
            raise LenMaxMismatchError(
                f"rendering {variant!r} has no row for {sid!r} (lang "
                f"{sent['lang']!r}; the next row it offers is {nxt!r}). Only "
                f"{IDENTITY_LANG!r} sentences are untranslated, so this is either a "
                "misaligned range or a table shorter than its inventory; taking the native "
                "length as the maximum here would silently restore per-language overflow"
            )
        out.append(None)
    if j != len(rows):
        raise LenMaxMismatchError(
            f"rendering {variant!r} has {len(rows) - j} row(s) left over after "
            f"{len(sent_rows)} sentence(s) were walked (first: "
            f"{rows[j]['sentence_id']!r}); its range does not match the sentence range"
        )
    return out


# --------------------------------------------------------------------------- #
# Config reads: execution shape only, since the semantics live in `reconcile`.
# --------------------------------------------------------------------------- #
def len_max_shards(cfg: RunConfig) -> int:
    """Number of document-atomic row-range shards to seed (non-shared ``execution``)."""
    ex = cfg.blocks.get("execution", {})
    return max(1, int(ex.get("len_max_shards", _DEFAULT_LEN_MAX_SHARDS)))


# --------------------------------------------------------------------------- #
# The saturate.StageAdapter: CPU, a tokenizer, no model, no GPU.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Peek:
    """A one-row-lookahead cursor over a translation stream: ``take(sentence_id)`` or ``None``.

    Minimal and private. ``reconcile._DocRows`` and
    ``common.passage_store._DocCursor`` are the same primitive at document grain; this one is
    at sentence grain and, unlike them, must not consume a non-matching head, because that
    head belongs to a later sentence in the same batch. If a fourth of these appears, that is
    the signal to move a parameterised peek cursor down into ``common.io``.
    """

    _rows: Any
    _head: Mapping[str, Any] | None = None
    _primed: bool = False

    def _peek(self) -> Mapping[str, Any] | None:
        if not self._primed:
            self._head = next(self._rows, None)
            self._primed = True
        return self._head

    def take(self, sentence_id: str) -> Mapping[str, Any] | None:
        """Consume and return the head iff it keys to ``sentence_id``; else ``None``."""
        head = self._peek()
        if head is None or head["sentence_id"] != sentence_id:
            return None
        self._primed = False
        return head

    def exhausted(self) -> bool:
        return self._peek() is None


@dataclass(slots=True)
class _LenMaxCtx:
    """The worker's resident context: the pinned tokenizer and the co-ordered inputs."""

    tokenizer: Any
    sentences: Path
    translations: dict[str, Path]
    renderings: tuple[str, ...]
    content_budget: int | None


@dataclass(frozen=True, slots=True)
class LenMaxShard:
    """One claimable unit: a range of the final sentences plus each variant's own range.

    Before the translation tables became subsequences, one row range addressed all three
    files and a :class:`~ragtime.preprocess.spine.ShardRange` was the whole payload. It no
    longer does: ``translations/<variant>.parquet`` skips every English sentence, so row
    ``i`` of the inventory is not row ``i`` of a translation table. The ranges are resolved
    once at seed by ``reconcile.align_subtable``, the same generalisation of
    ``spine.align_documents`` reconciliation already uses for its three sub-tables, and
    carried here, so a worker still opens exactly its own row groups and never scans a
    table from row 0.
    """

    rows: spine.ShardRange
    translation_rows: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.rows.name

    def payload(self) -> dict[str, Any]:
        p = dict(self.rows.payload())
        p["translation_rows"] = {v: list(r) for v, r in self.translation_rows.items()}
        return p

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LenMaxShard:
        ranges = payload.get("translation_rows") or {}
        return cls(
            rows=spine.ShardRange.from_payload(payload),
            translation_rows={
                str(v): (int(r[0]), int(r[1])) for v, r in dict(ranges).items()
            },
        )


@dataclass(slots=True)
class _Shard:
    name: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class LenMaxAdapter:
    """``len_max`` as one ``saturate.StageAdapter``: CPU, no model, no bespoke driver.

    Sharding is :func:`~ragtime.preprocess.spine.plan_shards` over the final sentence table,
    and because every ``final/translations/<variant>.parquet`` is one row per final sentence
    in the same order, one row range addresses all of them, with no second alignment pass and
    no per-document join. ``documents.parquet`` is never opened: the native length is stored
    as ``token_count``.

    ``stage`` embeds both hashes, so a different inventory or a different tokenizer resolves
    to a fresh queue subtree and a fresh artifact node.
    """

    recon_hash: str
    lm_hash: str
    renderings: tuple[str, ...] = RENDERINGS
    tokenizer_id: str = ""
    base: str | None = None
    template: str = "workqueue_worker_cpu.sbatch"
    stats: Statistics = field(default_factory=Statistics)
    stage: str = field(init=False)
    _expected: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.stage = f"len_max_{self.recon_hash[:_HASH_DIRLEN]}_{self.lm_hash[:_HASH_DIRLEN]}"

    @classmethod
    def for_config(cls, cfg: RunConfig, *, base: str | None = None) -> LenMaxAdapter:
        """Build the adapter for ``cfg``; the sanctioned construction path."""
        opts = len_max_options(cfg)
        return cls(
            recon_hash=reconcile_hash(cfg),
            lm_hash=len_max_hash(cfg),
            renderings=tuple(opts["renderings"]),
            tokenizer_id=str(opts["tokenizer_id"]),
            base=base,
        )

    # -- paths ------------------------------------------------------------- #
    def _layout(self, cfg: RunConfig) -> Layout:
        root = self.base if self.base is not None else _DEFAULT_BASE
        return Layout(
            run_dir=root,
            base=root,
            family=run_family(cfg),
            chunker_hash=all_hashes(cfg)["chunker"],
        )

    def sentences_path(self, cfg: RunConfig) -> Path:
        """The final, post-fusion sentence inventory this table is keyed by."""
        return self._layout(cfg).final_sentences_path(self.recon_hash)

    def translations_path(self, cfg: RunConfig, variant: str) -> Path:
        return self._layout(cfg).final_translations_path(self.recon_hash, variant)

    def out_path(self, cfg: RunConfig) -> Path:
        """Return ``<corpus_dir>/sentence_len_max/<inv12>-<lm12>/len_max.parquet``."""
        return self._layout(cfg).sentence_len_max_path(self.recon_hash, self.lm_hash)

    # -- seed --------------------------------------------------------------- #
    def shards(self, cfg: RunConfig) -> Iterator[_Shard]:
        """Document-atomic ranges over the final sentences, plus each variant's own range.

        ``align_subtable`` is handed the final sentence table as its parent and the shard
        row-starts as ordinals. That is the shape it was written for, a document-ordered
        table whose documents are a subset of the parent's, and the shard
        boundaries are document-atomic, so a boundary always falls on a document's first
        sentence row. One forward pass per variant, once per build, writing nothing.

        This is not free: each call scans the parent's ``document_id`` column and that
        variant's, so seeding does a handful of column scans and takes a few minutes. It buys
        the workers their pruned row-group reads for the whole build; the alternative, every
        worker scanning its table from row 0, is what ``spine.iter_row_range`` exists to
        avoid, and it is roughly thirty times slower per read.
        """
        sentences = self.sentences_path(cfg)
        ranges, _ = spine.plan_shards(sentences, n_shards=len_max_shards(cfg))
        ordinals = [r.row_start for r in ranges]
        variants = [name for name in self.renderings if name != NATIVE_RENDERING]
        aligned = {
            v: align_subtable(sentences, self.translations_path(cfg, v), ordinals)
            for v in variants
        }
        _log.info("preprocess.len_max.shards", stage=self.stage, shards=len(ranges))
        for k, r in enumerate(ranges):
            yield _Shard(
                name=r.name,
                payload=LenMaxShard(
                    rows=r,
                    translation_rows={v: aligned[v][k] for v in variants},
                ).payload(),
            )

    # -- work --------------------------------------------------------------- #
    def bringup(self, cfg: RunConfig) -> _LenMaxCtx:
        """Load the pinned pack tokenizer once per worker. No model, no GPU."""
        opts = len_max_options(cfg)
        variants = [name for name in self.renderings if name != NATIVE_RENDERING]
        tokenizer = load_pack_tokenizer_by_id(self.tokenizer_id or str(opts["tokenizer_id"]))
        budget = _content_budget(cfg, tokenizer)
        translations = {v: self.translations_path(cfg, v) for v in variants}
        _require_renderings(translations)
        return _LenMaxCtx(
            tokenizer=tokenizer,
            sentences=self.sentences_path(cfg),
            translations=translations,
            renderings=tuple(self.renderings),
            content_budget=budget,
        )

    def rows_for_shard(self, ctx: _LenMaxCtx, shard: LenMaxShard) -> Iterator[dict[str, Any]]:
        """Stream one shard's rows in constant memory, one tokenize batch at a time.

        The tokenize batch is cut on sentences, and each variant's rows are pulled onto that
        batch by the co-walk: a row is consumed only when it keys to the sentence being
        buffered, so an English sentence (which has no row) advances the sentence stream and
        not the translation streams. The batch's own leftover check therefore has to be
        deferred to the end of the shard rather than run per batch, hence the peek cursor
        rather than a plain iterator.
        """
        r = shard.rows
        sents = spine.iter_row_range(
            ctx.sentences, r.row_start, r.row_end, columns=list(_SENTENCE_COLUMNS)
        )
        # The fallback range is the sentence range, which is right only for a table that
        # still carries identity rows (the legacy policy, and hand-built test contexts).
        # It is never silently wrong under the shipped policy: the co-walk checks every
        # row against the sentence it is consumed for, so a mis-ranged read fails on the
        # first row rather than producing a plausible number.
        cursors = {
            variant: _Peek(
                spine.iter_row_range(
                    path,
                    *shard.translation_rows.get(variant, (r.row_start, r.row_end)),
                    columns=list(_TRANSLATION_COLUMNS),
                )
            )
            for variant, path in ctx.translations.items()
        }
        buf: list[Mapping[str, Any]] = []
        tbuf: dict[str, list[Mapping[str, Any]]] = {v: [] for v in cursors}

        def flush() -> Iterator[dict[str, Any]]:
            yield from len_max_rows(
                buf, tbuf, ctx.tokenizer,
                renderings=ctx.renderings,
                content_budget=ctx.content_budget,
                stats=self.stats,
            )

        for row in sents:
            buf.append(row)
            for variant, cursor in cursors.items():
                taken = cursor.take(row["sentence_id"])
                if taken is not None:
                    tbuf[variant].append(taken)
            if len(buf) >= _TOKENIZE_BATCH:
                yield from flush()
                buf = []
                tbuf = {v: [] for v in cursors}
        if buf:
            yield from flush()
        for variant, cursor in cursors.items():
            if not cursor.exhausted():
                raise LenMaxMismatchError(
                    f"rendering {variant!r} has rows left over after this shard's sentences "
                    "were consumed; its row range does not match the sentence range"
                )

    def work(self, ctx: _LenMaxCtx, shard: Path) -> Path:
        """Emit this shard's slice of the sidecar, streamed and never materialised."""
        spec = LenMaxShard.from_payload(_read_payload(shard))
        out = saturate.shard_out_path(shard).with_suffix(".parquet")
        write_parquet_stream(
            out,
            self.rows_for_shard(ctx, spec),
            schema=len_max_arrow_schema(ctx.renderings),
        )
        expected = spec.rows.row_end - spec.rows.row_start
        self._expected[str(out)] = expected
        _log.info("preprocess.len_max.shard.done", shard=shard.name, rows=expected)
        return out

    def validate(self, path: Path) -> bool:
        """Check the shard is marked done and holds one row per sentence of its range.

        This checks a row count, not a non-zero size: the table is total over the final
        inventory, so a short shard is the failure mode (a truncated write) and an empty one
        is impossible except for an empty range.
        """
        if not (path.exists() and is_done(path)):
            return False
        expected = self._expected.pop(str(path), None)
        if expected is None:
            return True  # re-validation / hand-built ctx: the marker is the gate
        rows = sum(parquet_row_group_sizes(path))
        if rows != expected:
            _log.warning(
                "preprocess.len_max.validate.row_count",
                path=str(path), rows=rows, expected=expected,
            )
            return False
        return True

    # -- merge -------------------------------------------------------------- #
    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> None:
        """Concatenate the shards in row order into the family sidecar.

        Shard names are ``rows_<start:012d>_<end:012d>``, so a lexicographic sort of the
        output listing is the global row order, so the sidecar stays co-ordered with the final
        sentence table, which is what lets reconciliation read it as a document-atomic range
        instead of a hash join over every id.
        """
        out = self.out_path(cfg)
        concat_parquet(
            out, sorted(shard_paths), schema=len_max_arrow_schema(tuple(self.renderings))
        )
        _log.info(
            "preprocess.len_max.done",
            stage=self.stage,
            path=str(out),
            rows=sum(parquet_row_group_sizes(out)),
        )


def _require_renderings(translations: Mapping[str, Path]) -> None:
    """Require every rendering's final translation table to exist before taking a maximum.

    Named loudly here rather than surfacing as a ``FileNotFoundError`` in a worker, because
    the one way this legitimately happens has a specific cause and fix: the low-tier
    ``omt_opus`` table is produced by the OPUS-MT arm, which runs outside the corpus
    substage chain: its work table is the reconciled inventory rather than the spine, so it
    is launched against an existing final node via ``RAGTIME_INVENTORY_DIR``). On a
    from-scratch build that arm has to run between ``reconcile`` and this stage.

    Silently maxing over the renderings that happen to be present would be the worst
    outcome available: it would produce a table that looks complete, pack a corpus against
    lengths that are not maxima, and reinstate exactly the per-language overflow this stage
    exists to remove, with every downstream check still passing.
    """
    missing = sorted(name for name, path in translations.items() if not path.exists())
    if missing:
        raise FileNotFoundError(
            f"len_max needs every rendering's final translations, missing {missing}: "
            + "; ".join(f"{name} -> {translations[name]}" for name in missing)
            + ". A maximum over a subset is not a maximum: it would pack the corpus against "
            "lengths that are not maxima and silently restore the per-language overflow. "
            "If 'omt_opus' is the missing one, run the OPUS-MT arm against this final node "
            "(RAGTIME_INVENTORY_DIR) first; it is not part of the corpus substage chain."
        )


def _content_budget(cfg: RunConfig, tokenizer: Any) -> int | None:
    """The packing content budget, used for the over-budget counter and never for a decision.

    Read from the same place the packer will read it (``reconcile.pack_budget``, falling back
    to ``chunker.config.token_budget``) so the tail this stage counts is the tail the packer
    will actually be unable to remove.
    """
    r = cfg.blocks.get("reconcile", {})
    budget = int(r.get("pack_budget") or cfg.blocks["chunker"]["config"]["token_budget"])
    return budget - int(tokenizer.num_special())


def _read_payload(shard: Path) -> dict[str, Any]:
    return json.loads(shard.read_text(encoding="utf-8").strip())


def build(cfg: RunConfig, *, base: str | os.PathLike[str] | None = None) -> Path:
    """Single-process build of the whole sidecar (small corpora, tests, bounded shards).

    The production path is the work queue (``saturate.seed``/``run_worker``/``drive`` over
    :class:`LenMaxAdapter`); this is the same adapter driven in one process, for a corpus
    small enough that sharding buys nothing.
    """
    adapter = LenMaxAdapter.for_config(cfg, base=None if base is None else str(base))
    out = adapter.out_path(cfg)
    if is_done(out):
        _log.info("preprocess.len_max.skip", path=str(out))
        return out
    ctx = adapter.bringup(cfg)
    total = sum(parquet_row_group_sizes(adapter.sentences_path(cfg)))
    # One shard covering everything, so each variant's whole table is read and no alignment
    # pass is needed: the co-walk in rows_for_shard does the join and checks every step.
    whole = LenMaxShard(
        rows=spine.ShardRange(row_start=0, row_end=total),
        translation_rows={
            v: (0, sum(parquet_row_group_sizes(p))) for v, p in ctx.translations.items()
        },
    )
    write_parquet_stream(
        out,
        adapter.rows_for_shard(ctx, whole),
        schema=len_max_arrow_schema(tuple(adapter.renderings)),
    )
    return out
