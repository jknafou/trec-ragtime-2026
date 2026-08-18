"""The vectorize half of the index build: passages into vector blocks on disk.

The index build is two stages. This one encodes passages into
:mod:`ragtime.preprocess.vectors`' block format and builds no index at all; assemble reads
those blocks and builds the FAISS, Seismic and PLAID artifacts. The seam between them is
``vectors``, the format, and the reason for the split is resource class: encoding is
GPU-bound, assembling is CPU- and RAM-bound, and an assemble-only recipe change must
re-encode nothing, which :func:`~ragtime.preprocess.index.leg_encode_hash` makes true.

Block membership and within-block order are pinned:

    A block is rows ``[b*B, (b+1)*B)`` of language ``L`` in
    ``final/<recon12>/passages/<pack12>/passages.parquet`` row order (document order), with
    ``B = encode_block_passages``. Within a block, passages are ordered by
    ``(token_count, passage_id)`` and bucketed under the hashed batch bounds.

Both halves of that pin derive from a table with no ``variant`` column, membership from its
row order and the within-block order from two of its columns, so bucket membership is
identical across the three renderings by construction. That is what makes a rank delta
between renderings attributable to translation content rather than to how the corpus was
cut. :class:`~ragtime.preprocess.vectors.BlockPlan` asserts the arithmetic (blocks divide
parts evenly, a block is a fixed ordinal slice) but cannot check the order it was handed;
that is this module's job, and :meth:`VectorizeAdapter.validate` re-derives both from the
real table rather than trusting the artifact.

A block is document-contiguous rather than a slice of the global token order, which is why
the read is proportional to the slice: a block's passages live in a contiguous run of
``documents.parquet`` rows, so ``common.passage_store.plan_final_ranges`` can push the range
down into all five co-ordered tables and
:func:`~ragtime.common.passage_store.iter_final_passages` reads only the row groups that run
touches. Measured, that is a few seconds per block against several minutes for a full
stream. Every assemble part is a whole number of blocks with a normal length distribution,
so even the smallest part carries millions of token vectors, far above fast-plaid's training
floor.

There are three units, and only two of them may move a hash:

* an encode block (``index_build.config.encode_block_passages``, hashed) is the unit of a
  vector file and of a ``_SUCCESS``. It is the total order ``Batcher`` buckets within, so
  its size decides batch membership and therefore the vectors bitwise.
* a vectorize task (``execution.vectorize_blocks_per_task``, not hashed) is a run of
  consecutive blocks one worker claims. It moves no boundary: two runs at different task
  widths produce byte-identical trees and resume each other, because the unit of
  ``_SUCCESS`` is the block (:func:`ragtime.preprocess.vectors.write_block` skips a block
  whose marker exists, leaving bytes and mtime untouched).
* an assemble part (``plaid_part_passages``, hashed) is one engine artifact, and is a whole
  number of blocks by :class:`BlockPlan`'s refusal to exist otherwise.

This stage opens no engine, trains no codec and writes no index manifest: ``preprocess.index``
owns all of that on the assemble side, along with every policy about these vectors
(unit norm, truncation counting, id-set integrity). The encode call sites are
``preprocess.index``'s own :class:`~ragtime.preprocess.index.Leg` implementations, reused
through :meth:`Leg.encode_docs`, so "one method across three renderings" stays a property of
one code path. The three encoders come from the ``serving`` singleton registry via
:meth:`~ragtime.preprocess.index.IndexAdapter.bringup` and are never instantiated here.

For sizing: encoding dominates by a wide margin, with one block costing on the order of a
minute of encode against a second of read. The stage is host-RAM-bound rather than
VRAM-bound, and the late-interaction leg produces by far the largest output per passage.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import Statistics, get_logger
from ragtime.common.io import is_done, iter_parquet, read_jsonl, write_jsonl
from ragtime.common.passage_store import (
    RENDERINGS,
    FinalRange,
    iter_final_passages,
    plan_final_ranges,
)
from ragtime.config import config_hash
from ragtime.orchestration import saturate

from . import vectors
from .index import (
    _TRANSLATED,
    DENSE_LEG,
    LATE_INTERACTION_LEG,
    LEGS,
    SHARED_LANG,
    SPARSE_LEG,
    STAT_EN_DIVERGENCE,
    IndexAdapter,
    IndexIntegrityError,
    IndexShardSpec,
    _assert_encode_device,
    _id_digest,
    _idmap_ids,
    _read_payload,
    _row_order_lang_ids,
    _Shard,
    _ShardItem,
    index_build_options,
    leg_encode_hash,
)
from .packing import packing_hash
from .reconcile import reconcile_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Mapping, Sequence

    from ragtime.common import Layout
    from ragtime.config import RunConfig

    from .index import IndexCtx, Leg

__all__ = [
    "STAT_BLOCKS",
    "STAT_BLOCK_BYTES",
    "STAT_BLOCK_RESUMED",
    "STAT_CELL_PASSAGES",
    "STAT_ENCODED",
    "STAT_LEAF_VECTORS",
    "STAT_PASSAGES",
    "CellBlockPlan",
    "VectorBlockSpec",
    "VectorizeAdapter",
    "VectorizeTaskSpec",
    "blocks_per_task",
    "plan_cell_blocks",
    "store_dtype",
]

_log = get_logger("preprocess.vectorize")

_HASH_DIRLEN = 12  # same truncation as Layout's hash path levels

# --- counters (counter-only invariant: emit here, monitoring rolls up) ------- #
STAT_BLOCKS = "vectorize.blocks"
STAT_PASSAGES = "vectorize.passages"
STAT_ENCODED = "vectorize.encoded_passages"
STAT_BLOCK_BYTES = "vectorize.block_bytes"
STAT_BLOCK_RESUMED = "vectorize.block_skipped_resume"
#: Late-interaction leaf (token) vectors written: the only leg whose passage count and
#: vector count differ, and the number that decides the intermediate's 1.5 TB footprint.
STAT_LEAF_VECTORS = "vectorize.leaf_vectors"
#: One published cell's passage total, cross-checked against the passage table in ``merge``.
STAT_CELL_PASSAGES = "vectorize.cell_passages"

#: How many blocks one worker claims when ``execution.vectorize_blocks_per_task`` is absent.
#: One: the conservative width: a task is the unit of a queue claim, so a narrow default
#: makes the queue self-balance across however many workers show up, and widening it is a
#: pure throughput decision that moves no boundary and no byte.
_DEFAULT_BLOCKS_PER_TASK = 1

#: ``sparse_store_format -> the value-side dtype``. The sparse leg stores a pair (Seismic's
#: ``U30`` component strings + float values), so only the value side has a width; the format
#: name carries both, and this is the one place the width is read out of it.
#: ``seismic_u30_f16`` resolves to ``float16`` rather than being rejected here,
#: :class:`ragtime.preprocess.vectors.SparseVectors` refuses it with the reason (a lossy
#: round-trip into an index that upcasts anyway), and one refusal beats two.
_SPARSE_VALUE_DTYPE: dict[str, str] = {
    "seismic_u30_f32": "float32",
    "seismic_u30_f16": "float16",
}

def store_dtype(opts: Any, leg: str) -> str:
    """The store dtype this run declared for ``leg``: from the hashed recipe, never a
    literal.

    Each leg has its own key because the three store three different shapes
    (``dense_store_dtype`` / ``late_interaction_store_dtype`` / ``sparse_store_format``), and
    the value is closed-set validated at config load
    (``preprocess.index._assert_structural_keys``), so an unknown value never reaches here.
    """
    if leg == DENSE_LEG:
        return str(opts.dense_store_dtype)
    if leg == LATE_INTERACTION_LEG:
        return str(opts.late_interaction_store_dtype)
    if leg == SPARSE_LEG:
        fmt = str(opts.sparse_store_format)
        try:
            return _SPARSE_VALUE_DTYPE[fmt]
        except KeyError:
            raise IndexIntegrityError(
                f"sparse_store_format={fmt!r} names no value width; expected one of "
                f"{sorted(_SPARSE_VALUE_DTYPE)}"
            ) from None
    raise IndexIntegrityError(f"unknown leg {leg!r}; expected one of {LEGS}")


def blocks_per_task(cfg: RunConfig) -> int:
    """``execution.vectorize_blocks_per_task``: a width, not hashed.

    It decides how many already-defined blocks one worker claims and therefore moves no
    boundary and no stored byte: the block's membership is fixed by the hashed
    ``index_build.config.encode_block_passages``. That is exactly why it lives in the
    non-shared ``execution`` block, and why two runs at different widths resume each other.
    """
    value = cfg.blocks.get("execution", {}).get(
        "vectorize_blocks_per_task", _DEFAULT_BLOCKS_PER_TASK
    )
    return max(1, int(value or _DEFAULT_BLOCKS_PER_TASK))


# --------------------------------------------------------------------------- #
# Specs: one block (the _SUCCESS unit) and one task (the claim unit).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class VectorBlockSpec:
    """One encode block: ``(variant, source_lang, block, blocks)``.

    ``variant is None`` names the shared English cell, built once and read by every
    rendering's assemble, exactly as ``IndexShardSpec`` names it.

    ``blocks`` carries the whole cell's plan, not just this unit's index, so a block is
    self-describing: the worker writes it into the census witness and a reader can tell a
    complete cell from a truncated one without a second lookup. Both default to the one-block
    cell so a hand-built spec is legal and obvious, while :meth:`from_payload` requires both,
    a queue payload without the block axis predates it and would silently encode block 0 of a
    355-block cell under that cell's whole id set. Same strictness, same reason, as
    ``IndexShardSpec.from_payload``'s ``part``/``parts``.
    """

    variant: str | None
    source_lang: str
    block: int = 0
    blocks: int = 1

    @property
    def name(self) -> str:
        """``<variant|_shared>_<lang>_b<NNNNN>``: the block's identity in a log/receipt."""
        return f"{self.variant or '_shared'}_{self.source_lang}_b{int(self.block):05d}"

    def payload(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "source_lang": self.source_lang,
            "block": int(self.block),
            "blocks": int(self.blocks),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VectorBlockSpec:
        variant = payload.get("variant")
        missing = [key for key in ("block", "blocks") if payload.get(key) is None]
        if missing:
            raise IndexIntegrityError(
                f"vector block payload {dict(payload)!r} carries no {missing}, the unit of a "
                "vector file and of a _SUCCESS is the block, so a payload without the block "
                "axis predates it and would encode block 0 of a many-block cell under that "
                "cell's whole id set. Re-seed the queue."
            )
        return cls(
            variant=None if variant is None else str(variant),
            source_lang=str(payload["source_lang"]),
            block=int(payload["block"]),
            blocks=int(payload["blocks"]),
        )

    @property
    def text_rendering(self) -> str:
        """Which rendering's text this block encodes (``original`` for the shared cell).

        Read off ``IndexShardSpec`` rather than restated: English is identity pass-through in
        both MT arms, so the shared unit encodes the native composition, the one composition
        that is a document slice rather than a join over re-assembled segments.
        """
        return IndexShardSpec(
            variant=self.variant, source_lang=self.source_lang
        ).text_rendering


@dataclass(frozen=True, slots=True)
class VectorizeTaskSpec:
    """One claimable unit: a run of consecutive blocks of one cell, plus its read plan.

    The task carries what a worker cannot cheaply re-derive: the document ranges its blocks
    live in (resolved once at seed by ``plan_final_ranges``, the only sanctioned
    :class:`FinalRange` constructor) and the language-ordinal the first of those ranges starts
    at. Re-deriving them per worker would put an O(corpus) column scan back in front of every
    read, which is exactly the cost the pushdown removes.

    The ranges are a contiguous run of a global document partition whose edges are every
    cell's block starts, so they are read in order and each is opened once. The run covers one
    document past the task's last block, because a document's passages may straddle a block
    boundary, the stream stops on the ordinal, not on the range, so the extra costs at most
    one row group.
    """

    blocks: tuple[VectorBlockSpec, ...]
    cell_passages: int
    block_passages: int
    start_ordinal: int
    ranges: tuple[FinalRange, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise IndexIntegrityError("a vectorize task covers at least one block")
        cells = {(b.variant, b.source_lang, b.blocks) for b in self.blocks}
        if len(cells) != 1:
            raise IndexIntegrityError(
                f"a vectorize task covers one cell; got {sorted(cells)}, blocks of two cells "
                "in one claim would share a stream that describes neither"
            )
        indices = [int(b.block) for b in self.blocks]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise IndexIntegrityError(
                f"a vectorize task covers consecutive blocks; got {indices}, the task reads "
                "one contiguous document range, which a gap would not describe"
            )

    @property
    def variant(self) -> str | None:
        return self.blocks[0].variant

    @property
    def source_lang(self) -> str:
        return self.blocks[0].source_lang

    @property
    def first_block(self) -> int:
        return int(self.blocks[0].block)

    @property
    def text_rendering(self) -> str:
        return self.blocks[0].text_rendering

    @property
    def name(self) -> str:
        """``vectorize_<variant|_shared>_<lang>_b<NNNNN>``: the queue filename."""
        return f"vectorize_{self.blocks[0].name}"

    def block_bounds(self) -> tuple[int, int]:
        """This task's half-open cell-ordinal window ``[lo, hi)`` over the language."""
        lo = self.first_block * self.block_passages
        hi = min(self.cell_passages, (self.first_block + len(self.blocks)) * self.block_passages)
        return lo, max(lo, hi)

    def payload(self) -> dict[str, Any]:
        return {
            "blocks": [b.payload() for b in self.blocks],
            "cell_passages": int(self.cell_passages),
            "block_passages": int(self.block_passages),
            "start_ordinal": int(self.start_ordinal),
            "ranges": [r.payload() for r in self.ranges],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VectorizeTaskSpec:
        blocks = payload.get("blocks")
        if not blocks:
            raise IndexIntegrityError(
                f"vectorize task payload {dict(payload)!r} names no blocks, the claim unit is "
                "a run of blocks, and an empty run would mark a cell done having encoded none"
            )
        missing = [
            key
            for key in ("cell_passages", "block_passages", "start_ordinal")
            if payload.get(key) is None
        ]
        if missing:
            raise IndexIntegrityError(
                f"vectorize task payload {dict(payload)!r} carries no {missing}, without them "
                "the worker cannot say which ordinals its blocks hold, and would encode "
                "whatever its document range happened to contain. Re-seed the queue."
            )
        return cls(
            blocks=tuple(VectorBlockSpec.from_payload(b) for b in blocks),
            cell_passages=int(payload["cell_passages"]),
            block_passages=int(payload["block_passages"]),
            start_ordinal=int(payload["start_ordinal"]),
            ranges=tuple(
                FinalRange.from_payload(r) for r in (payload.get("ranges") or ())
            ),
        )


# --------------------------------------------------------------------------- #
# The seed-time plan: where every block starts, in document rows.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CellBlockPlan:
    """One language's block plan, in ``documents.parquet`` row coordinates.

    ``block_starts[b] == (document_row, lang_ordinal_of_that_document's_first_passage)``: the
    first coordinate is where the ranged read must begin for block ``b``, the second is the
    ordinal that read's first row of this language carries. They differ whenever a document's
    passages straddle a block boundary, which is the normal case at 2.4 passages/document,
    so the two are recorded separately rather than one derived from the other.
    """

    lang: str
    passages: int
    blocks: int
    block_starts: tuple[tuple[int, int], ...]
    end_document: int


def plan_cell_blocks(
    layout: Layout,
    reconcile_hash: str,
    pack_hash: str | None,
    *,
    block_passages: int,
    langs: Sequence[str],
) -> dict[str, CellBlockPlan]:
    """Where every block of every language begins: One forward co-walk, at seed time.

    Walks ``passages.parquet`` (``document_id``/``lang``, column-pruned) beside
    ``documents.parquet``'s ``document_id`` column, which its documents are a subsequence of
    (a document with no passages contributes no row). Per language it counts passages in table
    row order, the pin's definition of block membership, and records, at every multiple of
    ``block_passages``, the document row the block starts in.

    Cost is the point: ~240 s once for every shard of every language, against re-deriving
    it per worker (an O(corpus) scan in front of every read, i.e. the defect the pushdown
    exists to remove). Memory is O(blocks), never O(corpus): nothing per-passage is retained.
    """
    documents = iter_parquet(layout.documents_path(), columns=["document_id"])
    passages_path = layout.final_passages_path(reconcile_hash, pack_hash)
    wanted = {str(lang) for lang in langs}
    size = max(1, int(block_passages))

    counts: dict[str, int] = {}
    doc_base: dict[str, int] = {}
    starts: dict[str, list[tuple[int, int]]] = {}
    end_document: dict[str, int] = {}
    doc_ordinal = -1
    current_doc: str | None = None

    for row in iter_parquet(passages_path, columns=["document_id", "lang"]):
        document_id = str(row["document_id"])
        if document_id != current_doc:
            while True:
                try:
                    head = next(documents)
                except StopIteration:
                    raise IndexIntegrityError(
                        f"{passages_path}: document {document_id!r} is not in "
                        f"{layout.documents_path()} at or after row {doc_ordinal + 1}, the "
                        "two tables are not from the same corpus build (or not co-ordered), "
                        "so no document range describes this passage"
                    ) from None
                doc_ordinal += 1
                if str(head["document_id"]) == document_id:
                    break
            current_doc = document_id
            # Snapshot before this document's first passage is counted: this is exactly "how
            # many passages of each language precede this document", which is the ordinal a
            # ranged read starting here will hand its first row.
            doc_base = dict(counts)
        lang = str(row["lang"])
        if lang not in wanted:
            continue
        seen = counts.get(lang, 0)
        if seen % size == 0:
            starts.setdefault(lang, []).append((doc_ordinal, doc_base.get(lang, 0)))
        counts[lang] = seen + 1
        end_document[lang] = doc_ordinal + 1

    return {
        lang: CellBlockPlan(
            lang=lang,
            passages=counts.get(lang, 0),
            # A language with no passages still has one (empty) block, exactly as
            # ``BlockPlan.blocks`` does: a cell that contributed no block at all would be
            # invisible to every census and coverage check.
            blocks=max(1, len(starts.get(lang, ()))),
            block_starts=tuple(starts.get(lang, ())),
            end_document=int(end_document.get(lang, 0)),
        )
        for lang in sorted(wanted)
    }


def _partition_ranges(
    layout: Layout,
    reconcile_hash: str,
    pack_hash: str | None,
    plans: Mapping[str, CellBlockPlan],
) -> tuple[list[tuple[int, int]], list[FinalRange]]:
    """The global document partition every task's read is assembled from.

    Its edges are every language's block starts plus every language's end, so a block's
    coverage is always a contiguous run of these cells, and the whole set is resolved by one
    :func:`plan_final_ranges` call (which is the only sanctioned :class:`FinalRange`
    constructor: a hand-written range is a silently mis-composed passage).

    A partition rather than one range per block: adjacent blocks share the
    document that straddles their boundary, and overlapping ranges are (correctly) refused by
    the planner. Chaining two planned cells costs one extra file open; merging them by hand
    would cost the guarantee.
    """
    edges = sorted(
        {start for plan in plans.values() for start, _ in plan.block_starts}
        | {plan.end_document for plan in plans.values() if plan.block_starts}
    )
    if len(edges) < 2:
        return [], []
    cells = list(pairwise(edges))
    planned = plan_final_ranges(
        layout,
        reconcile_hash,
        pack_hash=pack_hash,
        document_ranges=cells,
        # All three renderings, always: a cell's ranges are shared by every variant's task
        # (they describe documents, which have no rendering), so a range planned for a subset
        # would make the shared English task: which composes all three for the divergence
        # canary: unreadable.
        renderings=RENDERINGS,
    )
    return cells, planned


# --------------------------------------------------------------------------- #
# The adapter.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class VectorizeAdapter:
    """Vectorization as one ``saturate.StageAdapter``, claimed a run of blocks at a time.

    It composes an :class:`~ragtime.preprocess.index.IndexAdapter` rather than re-deriving
    anything it already owns: bring-up (the three registry singletons plus the recipe-vs-client
    and window-vs-packing assertions), the ``Batcher`` bucketing, the language counts and the
    paths all come from there. What is new here is only the block: its membership, its read
    plan, and its publication through ``preprocess.vectors``.

    ``legs`` is injectable for hermetic tests, and checked: exactly
    :data:`~ragtime.preprocess.index.LEGS`, in order, so a block carrying two legs' vectors,
    or a leg present on one rendering and absent from another, is unconstructible rather than
    merely tested against.
    """

    recon_hash: str
    #: ``pack12``, or ``None`` for the legacy pre-``packing`` node: the same required-but-
    #: nullable value ``Layout.final_passages_path``/``vectors_cell_dir`` take, threaded
    #: through unchanged so the table the pin is read from and the cell its vectors are
    #: written to are the same packing by construction.
    pack_hash: str | None
    encode_hashes: dict[str, str]
    base: str | None = None
    #: GPU: all three encoders run a forward pass here (the assemble half is the CPU one).
    template: str = "workqueue_worker.sbatch"
    legs: tuple[Leg, ...] = field(default_factory=tuple)
    stats: Statistics = field(default_factory=Statistics)
    stage: str = field(init=False)
    _index: IndexAdapter = field(init=False)

    def __post_init__(self) -> None:
        from .index import default_legs

        if not self.legs:
            self.legs = default_legs()
        names = tuple(leg.name for leg in self.legs)
        if names != LEGS:
            raise ValueError(
                f"a vector block carries exactly these legs, in this order: {LEGS}; got "
                f"{names}. A leg present on one rendering and absent from another is a "
                "failure, not an exception."
            )
        missing = [leg for leg in LEGS if leg not in self.encode_hashes]
        if missing:
            raise ValueError(
                f"no encode hash for leg(s) {missing}: the vector cell is keyed by the per-leg "
                "encode hash (that is what lets an assemble-only recipe change reuse these "
                "vectors), so it cannot be resolved without one"
            )
        self._index = IndexAdapter(
            recon_hash=self.recon_hash,
            pack_hash=self.pack_hash,
            idx_hash="",  # unused here: this stage keys on the per-leg encode hash
            base=self.base,
            legs=self.legs,
            stats=self.stats,
        )
        # The queue re-keys on the encode recipe alone, never on ``index_hash``: an
        # assemble-only change must re-encode nothing and re-fan nothing.
        digest = config_hash({leg: self.encode_hashes[leg] for leg in LEGS})
        self.stage = f"vectorize_{digest[:_HASH_DIRLEN]}_{self.recon_hash[:_HASH_DIRLEN]}"

    @classmethod
    def for_config(
        cls,
        cfg: RunConfig,
        *,
        base: str | os.PathLike[str] | None = None,
        legs: tuple[Leg, ...] | None = None,
        stats: Statistics | None = None,
    ) -> VectorizeAdapter:
        """Build the adapter for ``cfg``: the only sanctioned construction path."""
        from .index import default_legs

        opts = index_build_options(cfg)
        return cls(
            recon_hash=reconcile_hash(cfg),
            pack_hash=packing_hash(cfg),
            encode_hashes={leg: leg_encode_hash(opts, leg) for leg in LEGS},
            base=None if base is None else str(base),
            legs=default_legs() if legs is None else legs,
            stats=stats if stats is not None else Statistics(),
        )

    # -- paths --------------------------------------------------------------- #
    def cell_dir(
        self, layout: Layout, leg: str, variant: str | None, source_lang: str
    ) -> Path:
        """The ``(leg, variant, source_lang)`` cell, keyed by that leg's encode hash.

        Resolved by :meth:`ragtime.common.Layout.vectors_cell_dir` and nowhere else, the
        same method ``preprocess.assemble`` reads these blocks back through, so the two
        halves of the split cannot agree separately and disagree with each other. The cell
        hangs under the packing node (``recon12`` + ``pack12``): a re-pack encodes fresh
        vectors, an assemble-only recipe change reuses these.
        """
        return layout.vectors_cell_dir(
            self.recon_hash,
            self.pack_hash,
            leg,
            self.encode_hashes[leg],
            variant,
            source_lang,
        )

    def passages_path(self, cfg: RunConfig) -> Path:
        """The packed passage table the pin is defined over (``recon12`` and ``pack12``)."""
        return self._index.passages_path(cfg)

    # -- seed ---------------------------------------------------------------- #
    def shards(self, cfg: RunConfig) -> Iterator[_Shard]:
        """Every claimable task: ``renderings x non-English languages x block runs`` + shared
        English.

        The languages come from ``cfg.languages`` (config-driven); only the identity of the
        shared one is structural. The block count per language comes from the corpus itself
        (:func:`plan_cell_blocks`), never from config: a declared count could go stale against
        the table and would then seed a plan that encodes a prefix of a language.
        """
        opts = index_build_options(cfg)
        layout = self._index._layout(cfg)
        size = int(opts.encode_block_passages)
        langs = sorted({str(x) for x in cfg.languages})
        plans = plan_cell_blocks(
            layout,
            self.recon_hash,
            self.pack_hash,
            block_passages=size,
            langs=langs,
        )
        # Constructed for its assertions, not its arithmetic: a block size that does not
        # divide the assemble part size must fail at seed, before any GPU time is spent.
        for plan in plans.values():
            vectors.BlockPlan(
                passages=plan.passages,
                block_passages=size,
                part_passages=int(opts.plaid_part_passages),
            )
        cells, planned = _partition_ranges(layout, self.recon_hash, self.pack_hash, plans)
        cell_at = {start: i for i, (start, _) in enumerate(cells)}
        width = blocks_per_task(cfg)
        for variant, lang in self._cells(cfg):
            plan = plans[lang]
            for first in range(0, plan.blocks, width):
                count = min(width, plan.blocks - first)
                spec = self._task(plan, variant, first, count, size, cell_at, planned)
                yield _Shard(name=spec.name, payload=spec.payload())

    def _cells(self, cfg: RunConfig) -> list[tuple[str | None, str]]:
        """``(variant, source_lang)`` cells: three per non-English language, one for English.

        English is ``variant=None``, the shared cell every rendering's assemble reads. It is
        the same rule ``IndexAdapter.shard_specs`` follows, and it is enforced by the path
        scheme (``Layout``'s ``_shared`` collapse), not by a copy step.
        """
        langs = sorted({str(x) for x in cfg.languages} - {SHARED_LANG})
        out: list[tuple[str | None, str]] = [
            (variant, lang) for variant in RENDERINGS for lang in langs
        ]
        if SHARED_LANG in {str(x) for x in cfg.languages}:
            out.append((None, SHARED_LANG))
        return out

    def _task(
        self,
        plan: CellBlockPlan,
        variant: str | None,
        first: int,
        count: int,
        size: int,
        cell_at: Mapping[int, int],
        planned: Sequence[FinalRange],
    ) -> VectorizeTaskSpec:
        """One task spec: its blocks plus the contiguous run of planned cells they live in.

        The run ends at the cell containing the next task's first document, not before it: a
        document's passages may straddle a block boundary, so the task's last block can have
        rows in that document. Reading it costs at most one row group, because the stream
        stops on the ordinal.
        """
        blocks = tuple(
            VectorBlockSpec(
                variant=variant, source_lang=plan.lang, block=b, blocks=plan.blocks
            )
            for b in range(first, first + count)
        )
        ranges: tuple[FinalRange, ...] = ()
        start_ordinal = 0
        if plan.block_starts:
            lo_doc, start_ordinal = plan.block_starts[first]
            last = first + count
            if last < len(plan.block_starts):
                hi_doc = plan.block_starts[last][0]
                stop = cell_at[hi_doc] + 1  # inclusive: the straddling document lives here
            else:
                stop = cell_at.get(plan.end_document, len(planned))
            ranges = tuple(planned[cell_at[lo_doc] : stop])
        return VectorizeTaskSpec(
            blocks=blocks,
            cell_passages=plan.passages,
            block_passages=size,
            start_ordinal=start_ordinal,
            ranges=ranges,
        )

    # -- bringup ------------------------------------------------------------- #
    def bringup(self, cfg: RunConfig) -> IndexCtx:
        """The three encoders + the recipe, from ``IndexAdapter``: never a second bring-up.

        Reused rather than re-implemented so the checks that guard an encode (the clients must
        match the recorded recipe; the late-interaction window must cover the pack budget) run
        for the stage that actually does the encoding, in exactly the form they were written.

        One check is added here rather than inherited, and it belongs to this stage because
        this is the stage that runs a forward pass: the declared ``encode_device`` must exist
        on this worker (:func:`~ragtime.preprocess.index._assert_encode_device`). All three
        encoders resolve their device as ``"cuda" if torch.cuda.is_available() else "cpu"``,
        so a vectorize worker mis-scheduled onto a GPU-less node would not fail, it would
        encode on CPU and publish the blocks under a ``leg_encode_hash`` that records
        ``cuda``, and no census, no id-set check and no manifest could tell afterwards. It
        runs before the delegation, so the failure costs no client bring-up and no claim.
        """
        _assert_encode_device(index_build_options(cfg))
        return self._index.bringup(cfg)

    # -- work ---------------------------------------------------------------- #
    def load_blocks(
        self, ctx: IndexCtx, spec: VectorizeTaskSpec
    ) -> dict[int, list[_ShardItem]]:
        """This task's passages, composed and grouped into blocks, in the pin's order.

        Streams the task's planned document ranges in order, one pass, each range opened
        once, through the same ``iter_final_passages`` co-walk (and therefore the same
        ``compose_passage_text``) the passage store uses, so the native-slice-vs-translated-join
        rule exists in exactly one place. Rows of this language are numbered from the task's
        ``start_ordinal``, which is what makes a block "rows ``[b*B, (b+1)*B)`` of language L
        in table row order" rather than "whatever this document range contained"; each block's
        list is then sorted by ``(token_count, passage_id)``, both columns of the one
        text-free table, so the order is rendering-invariant.

        For the shared English cell every rendering is composed, not because they can differ
        (English has no stored translation, so all three resolve to the same native slice) but
        so :data:`STAT_EN_DIVERGENCE` stays a live canary for a stray identity row.
        """
        rendering = spec.text_rendering
        shared = spec.variant is None
        renderings = ("original", *_TRANSLATED) if shared else ("original", rendering)
        lo, hi = spec.block_bounds()
        out: dict[int, list[_ShardItem]] = {int(b.block): [] for b in spec.blocks}
        ordinal = spec.start_ordinal
        for document_range in spec.ranges:
            if ordinal >= hi:
                break
            for record in iter_final_passages(
                ctx.layout,
                ctx.recon_hash,
                pack_hash=ctx.pack_hash,
                renderings=tuple(dict.fromkeys(renderings)),
                document_range=document_range,
            ):
                if record["lang"] != spec.source_lang:
                    continue
                position = ordinal
                ordinal += 1
                if position < lo:
                    continue
                if position >= hi:
                    break
                text = record[rendering]
                if shared:
                    for other in _TRANSLATED:
                        if record.get(other) != text:
                            self.stats.emit(
                                STAT_EN_DIVERGENCE, lang=spec.source_lang, variant=other
                            )
                out[position // spec.block_passages].append(
                    _ShardItem(
                        passage_id=record["passage_id"],
                        document_id=record["document_id"],
                        text=text,
                        token_count=int(record["token_count"] or 0),
                        is_oversized=bool(record["is_oversized"]),
                    )
                )
        for block, items in out.items():
            expected = self._expected_size(spec, block)
            if len(items) != expected:
                raise IndexIntegrityError(
                    f"block {block} of ({spec.variant}, {spec.source_lang}) streamed "
                    f"{len(items)} passages, not the {expected} its plan gives it, the "
                    "document range does not describe the block's rows (a stale plan, or a "
                    "table re-written under the same pack12)"
                )
            # The pin's second half. ``Batcher``'s own sort is by length and stable, so the
            # buckets built from this list preserve it: which is why the stored ordinal order
            # Is this order, and why it is identical across the three renderings.
            items.sort(key=lambda it: (it.token_count, it.passage_id))
            self.stats.emit(
                STAT_PASSAGES, float(len(items)), lang=spec.source_lang, variant=rendering
            )
        return out

    @staticmethod
    def _expected_size(spec: VectorizeTaskSpec, block: int) -> int:
        lo = int(block) * spec.block_passages
        return max(0, min(spec.cell_passages, lo + spec.block_passages) - lo)

    def work(self, ctx: IndexCtx, shard: Path) -> Path:
        """Encode every not-yet-published block x leg of one task; return the queue receipt.

        Order of operations is deliberate: blocks whose three legs already carry a ``_SUCCESS``
        are skipped before any passage text is read, so a re-launch (at any task width)
        re-streams nothing for what exists. The remaining blocks share one composed read
        across all three legs.
        """
        spec = VectorizeTaskSpec.from_payload(_read_payload(shard))
        plan = vectors.BlockPlan(
            passages=spec.cell_passages,
            block_passages=spec.block_passages,
            part_passages=int(ctx.opts.plaid_part_passages),
        )
        cells = {
            leg.name: self.cell_dir(ctx.layout, leg.name, spec.variant, spec.source_lang)
            for leg in self.legs
        }
        pending = [
            int(b.block)
            for b in spec.blocks
            if any(
                not is_done(cells[leg.name] / vectors.block_dir_name(int(b.block)))
                for leg in self.legs
            )
        ]
        items = self.load_blocks(ctx, spec) if pending else {}
        report: dict[str, Any] = {
            "variant": spec.variant,
            "source_lang": spec.source_lang,
            "rendering": spec.text_rendering,
            "first_block": spec.first_block,
            "task_blocks": len(spec.blocks),
            "blocks": int(spec.blocks[0].blocks),
            "block_passages": spec.block_passages,
            "cell_passages": spec.cell_passages,
            "encode_hashes": {leg.name: self.encode_hashes[leg.name] for leg in self.legs},
            "cells": {leg.name: str(cells[leg.name]) for leg in self.legs},
            # ``validate`` runs on a bare path in any process, so the receipt names the table
            # it must re-derive the block id sets from: never a same-process hand-off.
            "passages_path": str(
                ctx.layout.final_passages_path(ctx.recon_hash, ctx.pack_hash)
            ),
            "created_at": _now(),
            "worker": saturate.worker_provenance(),
            "built": [],
        }
        for block_spec in spec.blocks:
            block = int(block_spec.block)
            block_items = items.get(block, [])
            entry: dict[str, Any] = {
                "block": block,
                "passages": self._expected_size(spec, block),
                "legs": {},
            }
            for leg in self.legs:
                entry["legs"][leg.name] = str(
                    self._write_leg_block(ctx, spec, leg, plan, cells[leg.name], block, block_items)
                )
            if block in pending:
                entry["id_digest"] = _id_digest({it.passage_id for it in block_items})
            else:
                entry["id_digest"] = _id_digest(
                    set(_idmap_ids(cells[self.legs[0].name] / vectors.block_dir_name(block)))
                )
            report["built"].append(entry)
            self.stats.emit(
                STAT_BLOCKS, lang=spec.source_lang, variant=spec.text_rendering
            )
        out = saturate.shard_out_path(shard)
        write_jsonl(out, [report], skip_if_done=False)
        _log.info(
            "preprocess.vectorize.task.done",
            shard=shard.name,
            variant=spec.variant,
            lang=spec.source_lang,
            first_block=spec.first_block,
            blocks=len(spec.blocks),
            encoded=len(pending),
        )
        return out

    def _write_leg_block(
        self,
        ctx: IndexCtx,
        spec: VectorizeTaskSpec,
        leg: Leg,
        plan: vectors.BlockPlan,
        cell: Path,
        block: int,
        items: Sequence[_ShardItem],
    ) -> Path:
        """Encode + publish one ``(leg, block)`` unit, atomically and independently.

        Per-leg atomicity, one level finer than the shard: three encoders are resident in one
        worker, so an OOM in leg 3 must never discard legs 1-2,
        :func:`ragtime.preprocess.vectors.write_block` publishes each by temp->rename +
        ``_SUCCESS`` and skips one that already exists.
        """
        final = cell / vectors.block_dir_name(block)
        if is_done(final):
            self.stats.emit(
                STAT_BLOCK_RESUMED,
                leg=leg.name,
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )
            return final
        fmt = vectors.format_for(leg.name, dtype=store_dtype(ctx.opts, leg.name))
        reps: list[Any] = []
        ordered: list[_ShardItem] = []
        for bucket in self._index.buckets(ctx, list(items)):
            texts = [it.text for it in bucket]
            self._index._emit_truncation(
                ctx,
                IndexShardSpec(variant=spec.variant, source_lang=spec.source_lang),
                leg.name,
                texts,
            )
            encoded = leg.encode_docs(ctx, texts)
            if len(encoded) != len(bucket):
                raise IndexIntegrityError(
                    f"{leg.name}: encoder returned {len(encoded)} representations for "
                    f"{len(bucket)} passages, a dropped passage would break the id-set "
                    "equality every rendering's vectors are asserted on"
                )
            ordered.extend(bucket)
            reps.extend(encoded)
            self.stats.emit(
                STAT_ENCODED,
                float(len(bucket)),
                leg=leg.name,
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )
        published = vectors.write_block(
            cell,
            block=block,
            plan=plan,
            fmt=fmt,
            encode_hash=self.encode_hashes[leg.name],
            lang=spec.source_lang,
            variant=spec.variant,
            refs=[vectors.PassageRef(it.passage_id, it.document_id) for it in ordered],
            reps=reps,
        )
        witness = _read_witness(published)
        self.stats.emit(
            STAT_BLOCK_BYTES,
            float(witness.get("bytes") or 0),
            leg=leg.name,
            lang=spec.source_lang,
            variant=spec.text_rendering,
        )
        if witness.get("vectors") is not None:
            self.stats.emit(
                STAT_LEAF_VECTORS,
                float(witness["vectors"]),
                leg=leg.name,
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )
        return published

    # -- validate ------------------------------------------------------------ #
    def validate(self, path: Path) -> bool:
        """Every block published on all three legs, with the id set the table gives it.

        Not a non-zero size, and not the digest ``work`` wrote: the block's expected ids are
        re-derived from ``passages.parquet`` in this method's own pass, the same row-order
        slice the pin defines, and compared against each leg's emitted ``idmap.parquet``.
        That is the discipline ``IndexAdapter.expected_ids_at`` already uses, one unit down,
        and it is what catches a block built under a stale plan (which every per-block check
        would otherwise pass). A false return leaves the task reclaimable.
        """
        if not (path.exists() and is_done(path)):
            return False
        report = json.loads(path.read_text(encoding="utf-8").strip())
        passages_path = Path(report["passages_path"])
        lang = str(report["source_lang"])
        size = int(report["block_passages"])
        blocks = [int(entry["block"]) for entry in report["built"]]
        expected, implied = _expected_block_ids(passages_path, lang, size, blocks)
        if implied != int(report["blocks"]):
            _log.warning(
                "preprocess.vectorize.validate.block_count_mismatch",
                lang=lang,
                claimed=int(report["blocks"]),
                implied=implied,
            )
            return False
        for entry in report["built"]:
            block = int(entry["block"])
            for leg, cell in report["cells"].items():
                block_dir = Path(cell) / vectors.block_dir_name(block)
                if not is_done(block_dir):
                    _log.warning(
                        "preprocess.vectorize.validate.block_missing",
                        leg=leg,
                        path=str(block_dir),
                    )
                    return False
                ids = _idmap_ids(block_dir)
                if len(ids) != len(set(ids)):
                    _log.warning("preprocess.vectorize.validate.duplicate_ids", leg=leg)
                    return False
                if frozenset(ids) != expected[block]:
                    _log.warning(
                        "preprocess.vectorize.validate.id_set_mismatch",
                        leg=leg,
                        block=block,
                        got=len(ids),
                        expected=len(expected[block]),
                    )
                    return False
        return True

    # -- merge --------------------------------------------------------------- #
    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> None:
        """Prove every cell is whole and every block holds the same ids in all renderings.

        Three assertions, all aborting before this stage is marked done:

        1. every cell's census (:func:`ragtime.preprocess.vectors.read_cell_census`, read off
           the blocks' own witnesses) agrees with the passage table about the block count, the
           passage total and the leg's encode hash, so a cell that lost a block is a failed
           publication rather than a silently short assemble;
        2. every task of the plan reported, on all three legs;
        3. and the proof the whole Task-2 comparison rests on: block ``b`` of language ``l``
           holds the identical passage ids in ``original``, ``omt`` and ``omt_opus``, compared
           by digest (:func:`ragtime.preprocess.index._id_digest`, so three million-id sets are
           never resident at once). It holds by construction, membership comes from the row
           order of a table with no ``variant`` column, and is asserted anyway, because if it
           failed, a rank difference between renderings would be measuring the cut instead of
           the translation.
        """
        reports = [json.loads(Path(p).read_text(encoding="utf-8").strip()) for p in shard_paths]
        layout = self._index._layout(cfg)
        counts = self._index.lang_counts(cfg)
        size = int(index_build_options(cfg).encode_block_passages)

        digests: dict[tuple[str, int], dict[str, str]] = {}
        seen: dict[tuple[str | None, str], set[int]] = {}
        for report in reports:
            cell = (report["variant"], str(report["source_lang"]))
            for entry in report["built"]:
                seen.setdefault(cell, set()).add(int(entry["block"]))
                digests.setdefault((cell[1], int(entry["block"])), {})[
                    str(report["rendering"])
                ] = str(entry["id_digest"])

        for variant, lang in self._cells(cfg):
            passages = int(counts.get(lang, 0))
            blocks = max(1, -(-passages // max(1, size)))
            got = seen.get((variant, lang), set())
            if got != set(range(blocks)):
                raise IndexIntegrityError(
                    f"({variant}, {lang}): blocks {sorted(got)} reported but the corpus implies "
                    f"{blocks} block(s) 0..{blocks - 1} at {size} passages/block, publishing "
                    "now would leave part of the language unencoded while every per-block "
                    "check still passed"
                )
            for leg in self.legs:
                cell_dir = self.cell_dir(layout, leg.name, variant, lang)
                census = vectors.read_cell_census(cell_dir)
                problems: list[str] = []
                if census.blocks != blocks:
                    problems.append(f"blocks {census.blocks} != {blocks}")
                if census.passages != passages:
                    problems.append(f"passages {census.passages} != {passages}")
                if census.encode_hash != self.encode_hashes[leg.name]:
                    problems.append(
                        f"encode_hash {census.encode_hash!r} != "
                        f"{self.encode_hashes[leg.name]!r}"
                    )
                if problems:
                    raise IndexIntegrityError(
                        f"{cell_dir}: the published cell disagrees with the corpus "
                        f"({'; '.join(problems)}), an assemble over it would build a "
                        "full-looking index that is silently not this corpus"
                    )
                self.stats.emit(
                    STAT_CELL_PASSAGES,
                    float(census.passages),
                    leg=leg.name,
                    lang=lang,
                    variant=variant or "shared",
                )
        _assert_blocks_identical_across_renderings(digests)
        _log.info(
            "preprocess.vectorize.done",
            stage=self.stage,
            cells=len(seen),
            blocks=sum(len(v) for v in seen.values()),
            block_passages=size,
        )


def _assert_blocks_identical_across_renderings(
    digests: Mapping[tuple[str, int], Mapping[str, str]],
) -> None:
    """Block ``b`` of language ``l`` must hold the same ids in every rendering.

    A block is rows ``[b*B, (b+1)*B)`` of one language in the row order of the one text-free
    passage table, which has no ``variant`` column, so this holds by construction, and is
    asserted anyway because it is the property every cross-rendering comparison rests on.
    Cheap (one digest per cell-block) and total (every block of every language).

    Distinct from ``preprocess.index``'s part-level check, and not reusing it:
    that one asserts a cut on the pinned ``(token_count, passage_id)`` ordinal, this one a cut
    on table row order. Same shape, different invariant, and a message naming the wrong one
    would send the next reader to the wrong code.
    """
    disagreeing = {
        cell: dict(by_rendering)
        for cell, by_rendering in digests.items()
        if len(set(by_rendering.values())) > 1
    }
    if disagreeing:
        raise IndexIntegrityError(
            "block membership differs between renderings for "
            f"{sorted(disagreeing)}: a block is a row-order slice of the shared, text-free "
            "passage table, so identical membership is structural, a difference means one "
            "rendering was encoded against a different table or a different block size, and "
            "every cross-rendering comparison over it would measure the cut instead of the "
            "translation"
        )


def _expected_block_ids(
    passages_path: Path, source_lang: str, block_passages: int, blocks: Sequence[int]
) -> tuple[dict[int, frozenset[str]], int]:
    """The id sets ``blocks`` must hold + the block count the table implies.

    The pin's first half, re-derived from the table, and a delegate, not an implementation:
    :func:`~ragtime.preprocess.index._row_order_lang_ids` owns the row-order cut, at both of
    its sizes, because ``preprocess.assemble`` re-derives a part from the very same rule and
    the two ran on different orders once (see that function's docstring). The block unit is
    this stage's own only in its size (``encode_block_passages``); the arithmetic is shared.
    """
    return _row_order_lang_ids(
        Path(passages_path), source_lang, int(block_passages), list(blocks)
    )


def _read_witness(block_dir: Path) -> Mapping[str, Any]:
    """One published block's own census witness (``block.json``), read via ``common.io``."""
    return read_jsonl(block_dir / vectors.BLOCK_FILENAME)[0]


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
