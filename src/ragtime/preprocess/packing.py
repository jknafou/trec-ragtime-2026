"""Passage packing: the sentences-into-passages grouping, and the sole producer of a passage.

Reconciliation answers which sentences exist and stops there. This stage answers the one
remaining question, how those sentences are grouped into retrievable units, and is a
separate stage with a separate key because the two answers have different dependencies and
different costs:

* the inventory depends on fusion (``reconcile.split_back`` and ``min_segment_chars``) and
  is keyed by ``recon12``, which also keys ``final/<recon12>/``, the node holding both
  renderings' translations;
* the grouping depends on the ``packing`` block alone and is keyed by
  ``pack12 = H(packing)``, one level further in.

If a packing knob moved ``recon12``, a re-pack would land in a fresh node and everything
already inside the old one would be orphaned by path; recovering would mean copying the
translations across and guarding the pairing with a checksum, trading the composite key's
structural guarantee (a v1 inventory beside a v2 translation is unaddressable) for an
assertion. So this stage writes exactly one artifact and never re-emits or overwrites the
four tables reconciliation published beside it.

``packing.pack_length`` selects what the packer measures:

``"native"``
    a sentence's length is its native token count. A passage then fits the retrieval window
    in the language it was written in, and may or may not fit once it is read in English.
    On a sample of real passages a small per-language fraction exceeded the content budget
    in some rendering, worst in Chinese: a defect introduced by the packing method.

``"len_max"``
    a sentence's length is the maximum over the three renderings, taken from the
    ``preprocess.len_max`` sidecar. The packer already keeps a passage's summed lengths
    within the content budget, so packing on the maximum makes "the passage fits in every
    rendering" a consequence of that existing invariant rather than a new rule. On the same
    sample, re-packed, nothing is over budget in any language or rendering, measured on the
    composed passage text the encoder actually sees.

The algorithm is unchanged either way: :func:`~ragtime.preprocess.chunk._pack_sentences` is
called verbatim and only the number fed into it differs, which is why this replaces
encode-time windowing outright rather than complementing it.

A small irreducible tail of sentences exceeds the content budget on their own. A packer
cannot split a sentence, so each becomes its own ``is_oversized`` passage and is counted
(:data:`STAT_OVERSIZED`) rather than special-cased.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ragtime.common import Layout, Statistics, get_logger, sentence_id
from ragtime.common.io import concat_parquet, is_done, parquet_row_group_sizes, write_parquet_stream
from ragtime.common.schemas import final_passage_arrow_schema
from ragtime.config import all_hashes, config_hash
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family

from . import len_max as lenmax
from . import spine
from .chunk import _pack_sentences, _Sent
from .reconcile import ReconciliationError, reconcile_hash
from .tokenizer import load_pack_tokenizer_by_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Sequence

    from ragtime.config import RunConfig

__all__ = [
    "IRREDUCIBLE_TAIL_BY_LANG",
    "IRREDUCIBLE_TAIL_TOTAL",
    "PACK_LENGTH_MAX",
    "PACK_LENGTH_NATIVE",
    "PackAdapter",
    "PackedSentence",
    "pack_document",
    "packing_hash",
    "packing_options",
    "packing_shards",
    "verify_passages",
]

_log = get_logger("preprocess.packing")

_DEFAULT_BASE = "runs"
_HASH_DIRLEN = 12

#: ``packing.pack_length`` values. The default is the historical one, so a config predating
#: the block keeps producing byte-identical passages: the new behaviour is a stated run
#: choice rather than a silent code-side upgrade.
PACK_LENGTH_NATIVE = "native"
PACK_LENGTH_MAX = "len_max"
_DEFAULT_PACK_LENGTH = PACK_LENGTH_NATIVE

_DEFAULT_PACK_SHARDS = 200

#: The irreducible tail: sentences whose own ``len_max`` exceeds the content budget. A packer
#: cannot split a sentence, so each becomes an ``is_oversized`` passage that no budget and no
#: packing rule removes. The counts below are read off the built passages table for the
#: shipped ``pack_budget``; they are what a truncation test asserts against, exported so the
#: assertion has one source rather than a retyped literal. Note that the tail is defined by a
#: comparison against the budget, so it has to be re-derived whenever the budget moves.
IRREDUCIBLE_TAIL_BY_LANG: Mapping[str, int] = MappingProxyType(
    {"en": 846, "es": 268, "ru": 338, "zh": 360}
)
IRREDUCIBLE_TAIL_TOTAL = 1812

# Counter ids (common.stats), sliced by lang.
STAT_DOCUMENTS = "packing.documents"
STAT_PASSAGES = "packing.passages"
STAT_OVERSIZED = "packing.oversized_passages"

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
_LEN_MAX_COLUMNS = ("sentence_id", "document_id", "len_max")


@dataclass(frozen=True, slots=True)
class PackedSentence:
    """The packer's view of one final sentence: identity, structure and two lengths.

    ``token_count`` is the stored native length, what ``passages.token_count`` sums and what
    ``sentences.parquet`` records; ``pack_length`` is what the budget is measured on. The two
    are the same number on the native path and differ only when packing is
    rendering-invariant. Keeping both explicit stops the emitted ``token_count`` from
    acquiring a second meaning.
    """

    sentence_id: str
    document_id: str
    lang: str
    paragraph_index: int
    token_count: int
    pack_length: int


# --------------------------------------------------------------------------- #
# The per-document core: pure, no IO, no model.
# --------------------------------------------------------------------------- #
def pack_document(
    sentences: Sequence[PackedSentence],
    lang: str,
    tokenizer: Any,
    *,
    token_budget: int,
    overlap_frac: float,
    prefer_paragraph_break: bool,
    prefer_min_fill: float,
) -> list[dict[str, Any]]:
    """Pack one document's final sentences into passage rows, using the chunker's packer.

    :func:`~ragtime.preprocess.chunk._pack_sentences` is called verbatim, with the same
    overlap tail and paragraph preference, so a passage boundary means what it meant at chunk
    time. Its :class:`~ragtime.common.schemas.Passage` results are projected onto the
    text-free stored row: the packer's ``text`` and ``sentence_char_spans`` are derivable
    joins over the members' spans, and storing them would be a second copy of the corpus.

    No text is read. Packing branches on token counts and paragraph indices, and
    ``_Sent.text`` only reaches the discarded ``Passage.text``, so this stage never opens
    ``documents.parquet``.
    """
    if not sentences:
        return []
    sents = [
        _Sent(text="", count=s.pack_length, sid=s.sentence_id, para=s.paragraph_index)
        for s in sentences
    ]
    packed = _pack_sentences(
        sentences[0].document_id,
        lang,
        sents,
        tokenizer,
        token_budget=token_budget,
        overlap_frac=overlap_frac,
        prefer_paragraph_break=prefer_paragraph_break,
        prefer_min_fill=prefer_min_fill,
    )
    native = {s.sentence_id: s.token_count for s in sentences}
    return [
        {
            "passage_id": p.passage_id,
            "document_id": p.document_id,
            "lang": p.lang,
            "sentence_ids": list(p.sentence_ids),
            # Always the native sum: token_count is a fact about the passage's own-language
            # text and is what sentences.token_count is checked against. The len_max sum is
            # a join away for anyone who needs it.
            "token_count": sum(native[sid] for sid in p.sentence_ids),
            "is_oversized": p.is_oversized,
            "paragraph_index": list(p.paragraph_index),
        }
        for p in packed
    ]


def verify_passages(
    document_id: str,
    sentences: Sequence[PackedSentence],
    passages: Sequence[Mapping[str, Any]],
    *,
    content_budget: int,
) -> None:
    """Check this stage's acceptance criteria; runs on every document of every build.

    All checks are linear in the sentence count, in the same spirit as reconciliation's own
    per-document verification: a corpus this size gets exactly one pass, so the pass verifies
    itself rather than deferring to a sampled audit.
    """
    known = {s.sentence_id for s in sentences}
    tokens_of = {s.sentence_id: s.token_count for s in sentences}
    packs_of = {s.sentence_id: s.pack_length for s in sentences}
    covered: set[str] = set()
    for p in passages:
        members = list(p["sentence_ids"])
        missing = [m for m in members if m not in known]
        if missing:
            raise ReconciliationError(
                f"{p['passage_id']}: references sentence id(s) {missing} absent from the "
                "final inventory, an orphan citation target"
            )
        covered.update(members)
        expected = sum(tokens_of[m] for m in members)
        if int(p["token_count"]) != expected:
            raise ReconciliationError(
                f"{p['passage_id']}: token_count {p['token_count']} != the sum of its "
                f"members' native counts ({expected})"
            )
        # The budget property, checked rather than argued. On the rendering-invariant path a
        # member's pack_length is its maximum over renderings, so a passage within the budget
        # on these numbers is within it in every rendering. An is_oversized passage is the
        # irreducible tail (one sentence that alone exceeds the budget), so it is exempted
        # and counted rather than failing a build that can do nothing about it.
        if not p["is_oversized"]:
            total = sum(packs_of[m] for m in members)
            if total > content_budget:
                raise ReconciliationError(
                    f"{p['passage_id']}: packed to {total} tokens over a {content_budget}-token "
                    "content budget; the packer's own invariant was violated"
                )
    if sentences and covered != known:
        raise ReconciliationError(
            f"{document_id}: {len(known - covered)} final sentence(s) appear in no passage; "
            "every sentence must be retrievable"
        )


def build_document(
    document_id: str,
    lang: str,
    sent_rows: Sequence[Mapping[str, Any]],
    len_rows: Sequence[Mapping[str, Any]] | None,
    tokenizer: Any,
    *,
    token_budget: int,
    overlap_frac: float,
    prefer_paragraph_break: bool,
    prefer_min_fill: float,
    stats: Any = None,
) -> list[dict[str, Any]]:
    """Pair the inventory with its lengths, pack it, and verify the result.

    ``len_rows`` are this document's ``len_max`` sidecar rows (``None`` on the native path).
    They are paired BY ID against the sentence rows, not merely counted: the sidecar was
    measured over the inventory this same ``recon12`` names, and that identity is what makes
    reading it safe. Checking it per document costs one list comparison and converts a hash
    argument into an artifact fact.
    """
    ids = [r["sentence_id"] for r in sent_rows]
    if len_rows is None:
        lengths = [int(r["token_count"]) for r in sent_rows]
    else:
        got = [r["sentence_id"] for r in len_rows]
        if got != ids:
            raise ReconciliationError(
                f"{document_id}: the len_max sidecar names {len(got)} sentence(s) {got[:3]} "
                f"where the inventory has {len(ids)} {ids[:3]}...; the sidecar was measured "
                "over a different sentence inventory, and packing against it would apply "
                "one sentence's length to another"
            )
        lengths = [int(r["len_max"]) for r in len_rows]
    sentences = [
        PackedSentence(
            sentence_id=r["sentence_id"],
            document_id=r["document_id"],
            lang=str(r["lang"]),
            paragraph_index=int(r["paragraph_index"]),
            token_count=int(r["token_count"]),
            pack_length=n,
        )
        for r, n in zip(sent_rows, lengths, strict=True)
    ]
    _check_dense(document_id, sentences)
    passages = pack_document(
        sentences,
        lang,
        tokenizer,
        token_budget=token_budget,
        overlap_frac=overlap_frac,
        prefer_paragraph_break=prefer_paragraph_break,
        prefer_min_fill=prefer_min_fill,
    )
    verify_passages(
        document_id,
        sentences,
        passages,
        content_budget=token_budget - int(tokenizer.num_special()),
    )
    if stats is not None:
        stats.emit(STAT_DOCUMENTS, lang=lang)
        stats.emit(STAT_PASSAGES, float(len(passages)), lang=lang)
        oversized = sum(1 for p in passages if p["is_oversized"])
        if oversized:
            stats.emit(STAT_OVERSIZED, float(oversized), lang=lang)
    return passages


def _check_dense(document_id: str, sentences: Sequence[PackedSentence]) -> None:
    """Check the rows are this document's dense ``0..n-1`` inventory, which the packer assumes.

    ``passage_id``s are minted positionally (``<doc>#p{k}``), so a shard that silently
    received a document's tail would mint ids that collide with the head's.
    """
    want = [sentence_id(document_id, k) for k in range(len(sentences))]
    if [s.sentence_id for s in sentences] != want:
        raise ReconciliationError(
            f"{document_id}: the sentence rows handed to packing are not this document's "
            "dense 0..n-1 inventory; a partial document would mint colliding passage ids"
        )


# --------------------------------------------------------------------------- #
# Config reads. The semantics live in the fairness-shared `packing` block; only the
# execution shape lives in `execution`.
# --------------------------------------------------------------------------- #
def packing_options(cfg: RunConfig) -> dict[str, Any]:
    """The semantic packing knobs, read from the fairness-shared ``packing`` block.

    ``pack_budget`` lives here and not in ``chunker.config.token_budget``: the
    chunker block's hash keys ``corpus_dir``, so moving its budget would re-key the
    documents, the sentence spine, the merge map and 106 GPU-hours of translation BY path.
    Packing happens here, this block keys only ``pack12``, and the same edit therefore costs
    one ~20-minute CPU pass. It falls back to the chunker's value so a config predating the
    block behaves identically.

    ``overlap_frac``/``prefer_paragraph_break``/``prefer_paragraph_break_min_fill`` are still
    read from ``chunker.config``: they describe the same packing rule the chunker's retained
    reference packer implements, and duplicating them would create two editable copies of
    one semantic. Only the two knobs the packing block varies were moved here.
    """
    block = cfg.blocks.get("packing", {})
    chunker = cfg.blocks["chunker"]["config"]
    pack_length = str(block.get("pack_length", _DEFAULT_PACK_LENGTH))
    if pack_length not in (PACK_LENGTH_NATIVE, PACK_LENGTH_MAX):
        raise ValueError(
            f"packing.pack_length must be {PACK_LENGTH_NATIVE!r} or {PACK_LENGTH_MAX!r}, "
            f"got {pack_length!r}"
        )
    return {
        "pack_length": pack_length,
        "pack_budget": int(block.get("pack_budget") or chunker["token_budget"]),
        "tokenizer_id": str(block.get("len_max_tokenizer_id") or chunker["tokenizer_id"]),
        "overlap_frac": float(chunker["overlap_frac"]),
        "prefer_paragraph_break": bool(chunker.get("prefer_paragraph_break", True)),
        "prefer_min_fill": float(chunker.get("prefer_paragraph_break_min_fill", 0.6)),
    }


def packing_hash(cfg: RunConfig) -> str:
    """Return ``pack12 = H(packing)``, the passages artifact's own key and nothing else's.

    Hashed with the canonical hasher over the ``packing`` block as written. It is
    not composite: ``recon12`` is already a path level above, so folding it in
    again would buy nothing and would make the key unreadable.
    """
    return config_hash(cfg.blocks.get("packing", {}))


def packing_shards(cfg: RunConfig) -> int:
    """Number of document-atomic row-range shards to seed (non-shared ``execution``)."""
    ex = cfg.blocks.get("execution", {})
    return max(1, int(ex.get("pack_shards", _DEFAULT_PACK_SHARDS)))


# --------------------------------------------------------------------------- #
# The saturate.StageAdapter: CPU only, one number from a tokenizer, no model, no GPU.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _PackCtx:
    """The worker's resident context: the pinned tokenizer's budget and two co-ordered inputs.

    ``tokenizer`` is here for exactly one method, ``num_special()``, which fixes the content
    budget. ``count()`` is never called: every length this stage needs is already stored
    (``sentences.token_count``, or the sidecar's ``len_max``).
    """

    tokenizer: Any
    sentences: Path
    len_max: Path | None
    token_budget: int
    overlap_frac: float
    prefer_paragraph_break: bool
    prefer_min_fill: float


@dataclass(slots=True)
class _Shard:
    name: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class PackAdapter:
    """Packing as one ``saturate.StageAdapter``: CPU, no model, no bespoke driver.

    Sharding is :func:`~ragtime.preprocess.spine.plan_shards` over the final sentence table,
    and because the ``len_max`` sidecar is one row per final sentence in the same order, one
    row range addresses both, with no alignment pass and no per-document join. The row-by-row
    pairing in :func:`build_document` is what proves that co-ordering rather than trusting it.

    ``stage`` embeds both hashes, so a different inventory or a different packing recipe
    resolves to a fresh queue subtree and a fresh artifact node (invariant 7).
    """

    recon_hash: str
    pack_hash: str
    lm_hash: str = ""
    pack_length: str = _DEFAULT_PACK_LENGTH
    base: str | None = None
    template: str = "workqueue_worker_cpu.sbatch"
    stats: Statistics = field(default_factory=Statistics)
    stage: str = field(init=False)

    def __post_init__(self) -> None:
        self.stage = f"packing_{self.recon_hash[:_HASH_DIRLEN]}_{self.pack_hash[:_HASH_DIRLEN]}"

    @classmethod
    def for_config(cls, cfg: RunConfig, *, base: str | None = None) -> PackAdapter:
        """Build the adapter for ``cfg``; the sanctioned construction path."""
        return cls(
            recon_hash=reconcile_hash(cfg),
            pack_hash=packing_hash(cfg),
            lm_hash=lenmax.len_max_hash(cfg),
            pack_length=str(packing_options(cfg)["pack_length"]),
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
        """The final, post-fusion inventory: reconciliation's output, read-only here."""
        return self._layout(cfg).final_sentences_path(self.recon_hash)

    def len_max_path(self, cfg: RunConfig) -> Path | None:
        """The rendering-invariant length sidecar, or ``None`` on the native path."""
        if self.pack_length != PACK_LENGTH_MAX:
            return None
        return self._layout(cfg).sentence_len_max_path(self.recon_hash, self.lm_hash)

    def out_path(self, cfg: RunConfig) -> Path:
        """Return ``<final_dir>/passages/<pack12>/passages.parquet``, the one artifact written."""
        return self._layout(cfg).final_passages_path(self.recon_hash, self.pack_hash)

    # -- seed --------------------------------------------------------------- #
    def shards(self, cfg: RunConfig) -> Iterator[_Shard]:
        """Document-atomic row ranges over the final sentence table (one pass, writes nothing)."""
        ranges, _ = spine.plan_shards(self.sentences_path(cfg), n_shards=packing_shards(cfg))
        _log.info(
            "preprocess.packing.shards",
            stage=self.stage,
            shards=len(ranges),
            pack_length=self.pack_length,
        )
        for r in ranges:
            yield _Shard(name=r.name, payload=r.payload())

    # -- work --------------------------------------------------------------- #
    def bringup(self, cfg: RunConfig) -> _PackCtx:
        """Resolve the inputs and the packing knobs. No model is loaded.

        The pinned ``PackTokenizer`` is built from the same identity the lengths were
        measured with (``packing.len_max_tokenizer_id``, defaulting to the chunker's), so the
        ``num_special()`` reservation and the lengths come from one tokenizer; otherwise the
        budget would mean two different things at its two ends.
        """
        opts = packing_options(cfg)
        return _PackCtx(
            tokenizer=load_pack_tokenizer_by_id(str(opts["tokenizer_id"])),
            sentences=self.sentences_path(cfg),
            len_max=self.len_max_path(cfg),
            token_budget=int(opts["pack_budget"]),
            overlap_frac=float(opts["overlap_frac"]),
            prefer_paragraph_break=bool(opts["prefer_paragraph_break"]),
            prefer_min_fill=float(opts["prefer_min_fill"]),
        )

    def rows_for_shard(self, ctx: _PackCtx, shard: spine.ShardRange) -> Iterator[dict[str, Any]]:
        """Stream this shard's passage rows in constant memory, one document at a time."""
        sents = spine.iter_row_range(
            ctx.sentences, shard.row_start, shard.row_end, columns=list(_SENTENCE_COLUMNS)
        )
        lens = (
            spine.group_by_document(
                spine.iter_row_range(
                    ctx.len_max, shard.row_start, shard.row_end, columns=list(_LEN_MAX_COLUMNS)
                )
            )
            if ctx.len_max is not None
            else None
        )
        for doc_id, rows in spine.group_by_document(sents):
            len_rows: list[Mapping[str, Any]] | None = None
            if lens is not None:
                got = next(lens, None)
                if got is None or got[0] != doc_id:
                    raise ReconciliationError(
                        f"{doc_id}: the len_max sidecar has no rows for this document "
                        f"(next is {None if got is None else got[0]!r}); it does not cover "
                        "the inventory being packed"
                    )
                len_rows = got[1]
            yield from build_document(
                doc_id,
                rows[0]["lang"],
                rows,
                len_rows,
                ctx.tokenizer,
                token_budget=ctx.token_budget,
                overlap_frac=ctx.overlap_frac,
                prefer_paragraph_break=ctx.prefer_paragraph_break,
                prefer_min_fill=ctx.prefer_min_fill,
                stats=self.stats,
            )
        if lens is not None and next(lens, None) is not None:
            raise ReconciliationError(
                "len_max sidecar rows remain after this shard's documents were consumed; "
                "the sidecar and the inventory are not row-aligned"
            )

    def work(self, ctx: _PackCtx, shard: Path) -> Path:
        """Emit this shard's passages; the other four tables are not touched."""
        spec = spine.ShardRange.from_payload(_read_payload(shard))
        out = saturate.shard_out_path(shard).with_suffix(".parquet")
        write_parquet_stream(
            out, self.rows_for_shard(ctx, spec), schema=final_passage_arrow_schema()
        )
        _log.info(
            "preprocess.packing.shard.done",
            shard=shard.name,
            passages=sum(parquet_row_group_sizes(out)),
        )
        return out

    def validate(self, path: Path) -> bool:
        """Marked done, and non-empty for a non-empty range.

        Every document has at least one sentence and every sentence must appear in some
        passage (asserted per document in :func:`verify_passages`), so a shard that produced
        zero rows over a non-empty range is a truncated write, not a legal artifact.
        """
        if not (path.exists() and is_done(path)):
            return False
        if sum(parquet_row_group_sizes(path)) == 0:
            _log.warning("preprocess.packing.validate.empty", path=str(path))
            return False
        return True

    # -- merge -------------------------------------------------------------- #
    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> None:
        """Concatenate the shards in row order into the pack12-keyed passages artifact."""
        out = self.out_path(cfg)
        concat_parquet(out, sorted(shard_paths), schema=final_passage_arrow_schema())
        _log.info(
            "preprocess.packing.done",
            stage=self.stage,
            path=str(out),
            passages=sum(parquet_row_group_sizes(out)),
            pack_length=self.pack_length,
        )


def _read_payload(shard: Path) -> dict[str, Any]:
    return json.loads(shard.read_text(encoding="utf-8").strip())


def build(cfg: RunConfig, *, base: str | os.PathLike[str] | None = None) -> Path:
    """Single-process build of the whole passages table (small corpora, tests, bounded shards).

    The production path is the work queue (``saturate.seed``/``run_worker``/``drive`` over
    :class:`PackAdapter`); this is the same adapter driven in one process, for a corpus small
    enough that sharding buys nothing.
    """
    adapter = PackAdapter.for_config(cfg, base=None if base is None else str(base))
    out = adapter.out_path(cfg)
    if is_done(out):
        _log.info("preprocess.packing.skip", path=str(out))
        return out
    ctx = adapter.bringup(cfg)
    total = sum(parquet_row_group_sizes(adapter.sentences_path(cfg)))
    whole = spine.ShardRange(row_start=0, row_end=total)
    write_parquet_stream(
        out, adapter.rows_for_shard(ctx, whole), schema=final_passage_arrow_schema()
    )
    return out
