"""Assemble: stored vectors into the FAISS, Seismic and PLAID artifacts; the build's CPU half.

The index build is two stages. Vectorize encodes passages into the block format
:mod:`ragtime.preprocess.vectors` defines and nothing else; this module reads that format
and builds the engines ``preprocess.index``'s three writers already know how to build. The
seam between them is the block format rather than a function call: the two halves never run
in the same process, never share an object, and are scheduled onto different resource
classes.

The reason this is a separate stage is that it wants cores rather than a card. Per assemble
part, the Seismic leg is the longest pole by a wide margin, sustaining high parallelism on
32 cores and touching no GPU; FAISS takes seconds. The late-interaction leg is the only
GPU-capable one, and on the shipped CPU path under heavy contention it takes tens of
minutes, so the stage belongs on a CPU partition where large nodes sit idle rather than
behind the encode queue on a GPU partition.

``index_build.config.assemble_device`` defaults to ``cpu`` for scheduling reasons only.
Determinism does not require it: the non-determinism in the late-interaction k-means came
entirely from the Triton kernel, and with ``use_triton=False`` a CUDA build is identical
across repeats and byte-identical to the CPU build. Seismic is CPU-only regardless, and many
CPU workers beat a handful of GPUs on throughput here, and no GPU is assumed present.

A claimable unit is ``(variant, source_lang, part)`` with ``part`` sized by the hashed
``index_build.config.plaid_part_passages``, the same unit
:func:`~ragtime.preprocess.vectors.stream_part` reads, and the smallest one that can exist:
``fast-plaid`` needs a part's whole token-embedding buffer in one ``add_documents`` call,
because a second call against a created index takes its update path into a device-side
assert and aborts the worker. So a part is exactly one FAISS index, one Seismic index and
one PLAID index over one ordinal space, published together under
``Layout.index_shard_dir(..., part=)``.

When sizing a worker, note that the peak is not our buffer: ``add_documents`` peaks at
:data:`~ragtime.preprocess.index.PLAID_ADD_PEAK_MULTIPLIER` times the buffer handed to it,
because fast-plaid materialises an fp32 tensor whatever the input dtype. Sizing ``--mem``
from ``peak_part_bytes`` under-provisions by a factor of two to three.

The read is sequential, and that is the whole performance argument. One part is a contiguous
run of whole block files, so this module's inner loop is::

    for block in stream_part(cell, part=p, part_passages=N, census=census):
        writer.add(list(block.ordinals), block.reps)

Each file is opened once, in ascending order, with part-local ordinals that ``stream_part``
asserts continue exactly where the previous block's ended. There is no
random-access-by-block-index API: a gather over the same terabyte-scale data costs an order
of magnitude in read amplification, which is the difference between a bounded aggregate
read and a build that does not finish. The part's ``idmap.parquet`` is the concatenation of
its blocks' id maps in that same order, so ``ordinal`` is the row's position in it.

Two things this stage structurally cannot do:

1. It opens no text table: not ``documents.parquet``, not ``sentences.parquet``, not
   ``translations/<variant>.parquet``. The encode half already consumed the text, so a text
   read here is not merely forbidden; there is nothing to read it with, no composer, no
   passage store and no rendering resolution. The only corpus table this module touches is
   the text-free ``final/<recon12>/passages/<pack12>/`` table, and only for its ``lang``,
   ``token_count`` and ``passage_id`` columns, which drive the shard plan and the id-set
   validation.
2. It builds no encoder. ``serving.registry.build_clients`` is never called, so there is no
   analogue of ``index._assert_clients_match_recipe``. Vector provenance is validated by
   hash instead: the ``leg_encode_hash`` in the cell's path is cross-checked against the
   ``encode_hash`` every block recorded in its own witness, and a hash cannot be declared
   but not wired.

What is carried over from ``preprocess.index`` is imported rather than re-implemented. The
three writers (``_FaissWriter``, ``_SeismicWriter``, ``_PlaidWriter``) carry the Seismic
save and load asymmetry, the FAISS internal-id-equals-ordinal assertion, and the PLAID
single-call constraint with its determinism pins (``_pin_torch_rng``, ``use_triton=False``,
``device=assemble_device``). So are the census reader
(:func:`~ragtime.preprocess.index.read_shard_parts`), the manifest shard entry with its
per-part ``worker`` field, the provenance block, the id digest and the cross-rendering part
assertion. :meth:`AssembleAdapter.validate` and :meth:`AssembleAdapter.merge` keep every
check the index adapter has: exactly :data:`~ragtime.preprocess.index.LEGS`, id-set equality
against the part slice re-derived from the table, part disjointness (so a wrong boundary
cannot hide inside a complete-looking union), the on-disk census cross-check, and the
manifest.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import Layout, Statistics, get_logger
from ragtime.common.io import is_done, iter_parquet, iter_parquet_batches, write_jsonl
from ragtime.common.io import write_parquet_stream as _write_parquet_stream
from ragtime.common.passage_store import RENDERINGS
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family

from .index import (
    _PART_FILENAME,
    DENSE_LEG,
    IDMAP_FILENAME,
    LATE_INTERACTION_LEG,
    LEGS,
    SHARED_LANG,
    SPARSE_LEG,
    STAT_ID_INTEGRITY,
    STAT_LEG_BYTES,
    STAT_LEG_RESUMED,
    STAT_PASSAGES,
    STAT_PLAID_PART_BYTES,
    STAT_PLAID_PARTS,
    STAT_SPARSE_EMPTY,
    STAT_SPARSE_PRUNED,
    IndexIntegrityError,
    IndexShardSpec,
    _assert_assemble_device,
    _assert_parts_identical_across_renderings,
    _dir_bytes,
    _FaissWriter,
    _id_digest,
    _idmap_ids,
    _idmap_rows,
    _manifest_shard,
    _now,
    _PlaidWriter,
    _provenance,
    _publish_dir,
    _read_payload,
    _row_order_lang_ids,
    _SeismicWriter,
    idmap_arrow_schema,
    index_build_options,
    index_hash,
    leg_config_hash,
    leg_encode_hash,
    plaid_part_count,
    read_shard_parts,
)
from .packing import packing_hash
from .reconcile import reconcile_hash
from .vectors import BlockPlan, CellCensus, read_cell_census, stream_part

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Mapping, Sequence

    from ragtime.config import RunConfig

    from .index import IndexBuildOptions, LegWriter

__all__ = [
    "ASSEMBLE_TEMPLATE",
    "STAT_ASSEMBLED",
    "AssembleAdapter",
    "AssembleCtx",
]

_log = get_logger("preprocess.assemble")

_DEFAULT_BASE = "runs"  # mirrors index._DEFAULT_BASE / cli._LOCAL_ROOT
_HASH_DIRLEN = 12  # same truncation as Layout's hash path levels

#: The sbatch template this stage routes to: the Apptainer-wrapping CPU one. The SIF is
#: what is needed: the three engines (faiss / pyseismic / pylate+fast-plaid) live only in
#: the container: and the card is not: assembly is CPU-bound at both poles (see the module
#: docstring), so it runs on ``shared-cpu``/``public-cpu`` at 32 cores and 80 GiB, off the
#: encode critical path entirely. Neither older template fits: ``workqueue_worker_cpu.sbatch``
#: carries no Apptainer (so it cannot import faiss) and ``workqueue_worker.sbatch`` demands
#: ``--gres=gpu:1``. Recorded here as the one place the routing is stated.
ASSEMBLE_TEMPLATE = "workqueue_worker_cpu_sif.sbatch"

#: Passages this part folded into its engines. Distinct from ``index.encoded_passages``
#: (which counts a forward pass): this stage runs no model, so a counter named for
#: encoding would misdescribe it, and the two together are what says the split lost nothing.
STAT_ASSEMBLED = "index.assembled_passages"

#: ``leg -> the writer that turns that leg's stored reps into its engine``. Imported from
#: ``preprocess.index``, never re-implemented: each of these carries correctness that was
#: paid for once: Seismic's asymmetric ``save``/``load`` filenames, FAISS's "internal id
#: must equal the ordinal" assertion, and PLAID's exactly-one-``add_documents``-per-part
#: constraint with its three determinism pins. They all take ``(out_dir, opts)`` and expose
#: the ``LegWriter`` protocol (``add`` / ``finish``), which is the whole surface this stage
#: needs: no encoder, no client, no ``IndexCtx``.
_WRITER_OF: dict[str, Any] = {
    DENSE_LEG: _FaissWriter,
    SPARSE_LEG: _SeismicWriter,
    LATE_INTERACTION_LEG: _PlaidWriter,
}


# --------------------------------------------------------------------------- #
# The worker's context: a recipe and paths. No models, by construction.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class AssembleCtx:
    """Everything a worker needs, and nothing it must not have.

    Contrast :class:`~ragtime.preprocess.index.IndexCtx`, which holds three resident
    encoders: there is no client field here at all, so "assemble builds no encoder" is a
    property of the type rather than a discipline at the call sites. Vector provenance comes
    from :attr:`encode_hashes` (checked against every block's own witness), which is the
    replacement for ``index._assert_clients_match_recipe`` and a strictly stronger one, a
    hash cannot be declared-but-not-wired.
    """

    opts: IndexBuildOptions
    layout: Layout
    recon_hash: str
    #: ``pack12``, or ``None`` for the legacy pre-``packing`` node. It keys both the passage
    #: table this part's id set is re-derived from and the vector cell its blocks are read
    #: from (``Layout.vectors_cell_dir``), so a part can never be assembled from one
    #: packing's vectors against another packing's table.
    pack_hash: str | None
    idx_hash: str
    #: ``leg -> leg_encode_hash``: both the vector cell's path key and the value every
    #: block witness must carry. Resolved once per worker so a shard cannot be assembled
    #: against a hash computed differently from the one that named its directory.
    encode_hashes: dict[str, str]
    legs: tuple[str, ...] = LEGS

    @property
    def part_passages(self) -> int:
        """Passages per assemble part: the hashed ``plaid_part_passages`` (see the module
        docstring for why this, and not ``shard_part_passages``, is the irreducible unit)."""
        return max(1, int(self.opts.plaid_part_passages))

    @property
    def block_passages(self) -> int:
        """Passages per encode block: the hashed ``encode_block_passages``."""
        return max(1, int(self.opts.encode_block_passages))


@dataclass(slots=True)
class _Shard:
    """One claimable work unit, in ``saturate``'s ShardSpec shape."""

    name: str
    payload: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The adapter.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class AssembleAdapter:
    """Assembly as one ``saturate.StageAdapter``, sharded per ``(variant, source_lang, part)``.

    One adapter for all three legs, never one per leg, for the same reason the index build
    is one adapter: "the same method on all three renderings" is then a property of one code
    path. ``legs`` is the leg name tuple (this stage needs no leg implementation, only a
    writer) and it is checked to be exactly :data:`~ragtime.preprocess.index.LEGS`, in order,
    so a 2-leg or 4-leg assembly is unconstructible rather than merely untested.
    """

    recon_hash: str
    #: See :attr:`AssembleCtx.pack_hash`: required, nullable, never defaulted.
    pack_hash: str | None
    idx_hash: str
    base: str | None = None
    #: See :data:`ASSEMBLE_TEMPLATE`: the SIF is what is needed, not the GPU.
    template: str = ASSEMBLE_TEMPLATE
    legs: tuple[str, ...] = LEGS
    stats: Statistics = field(default_factory=Statistics)
    stage: str = field(init=False)
    #: ``(source_lang, part, part_size) -> the id set that part must cover``. Keyed by the
    #: whole triple for the reason ``IndexAdapter``'s is: a part's expected set is a slice of
    #: the table's row order, so a cache keyed by language alone would answer a different
    #: question.
    _expected: dict[tuple[str, int, int], frozenset[str]] = field(
        default_factory=dict, init=False
    )
    #: ``passages-table path -> {source_lang: count}``: the input to the shard plan, cached
    #: because ``shards`` and ``merge`` both re-derive the plan and must not disagree.
    _counts: dict[str, dict[str, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        names = tuple(str(leg) for leg in self.legs)
        if names != LEGS:
            raise ValueError(
                f"an assembled index has exactly these legs, in this order: {LEGS}; got "
                f"{names}. A leg present on one rendering and absent from another is a "
                "failure, not an exception."
            )
        self.legs = names
        # Both hashes in the queue name, exactly as the index stage does: an edit to the
        # corpus OR to the build recipe resolves to a fresh queue subtree, so a stale done/
        # can never be reused.
        self.stage = f"assemble_{self.idx_hash[:_HASH_DIRLEN]}_{self.recon_hash[:_HASH_DIRLEN]}"

    @classmethod
    def for_config(
        cls,
        cfg: RunConfig,
        *,
        base: str | os.PathLike[str] | None = None,
        stats: Statistics | None = None,
    ) -> AssembleAdapter:
        """Build the adapter for ``cfg``: the only sanctioned construction path."""
        return cls(
            recon_hash=reconcile_hash(cfg),
            pack_hash=packing_hash(cfg),
            idx_hash=index_hash(cfg),
            base=None if base is None else str(base),
            stats=stats if stats is not None else Statistics(),
        )

    # -- paths -------------------------------------------------------------- #
    def _layout(self, cfg: RunConfig) -> Layout:
        from ragtime.config import all_hashes

        root = self.base if self.base is not None else _DEFAULT_BASE
        return Layout(
            run_dir=root,
            base=root,
            family=run_family(cfg),
            chunker_hash=all_hashes(cfg)["chunker"],
        )

    def index_dir(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).index_dir(self.recon_hash, self.idx_hash)

    def lang_dir(self, cfg: RunConfig, variant: str | None, source_lang: str) -> Path:
        """The ``(variant, source_lang)`` cell: the level a query fans over."""
        return self._layout(cfg).index_lang_dir(
            self.recon_hash, self.idx_hash, variant, source_lang
        )

    def shard_dir(self, cfg: RunConfig, spec: IndexShardSpec) -> Path:
        return self._layout(cfg).index_shard_dir(
            self.recon_hash, self.idx_hash, spec.variant, spec.source_lang, part=spec.part
        )

    def manifest_path(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).index_manifest_path(self.recon_hash, self.idx_hash)

    def passages_path(self, cfg: RunConfig) -> Path:
        """The one text-free passage table: ``recon12`` (which sentences) and ``pack12``
        (how they are grouped). The only corpus table this stage reads, ever."""
        return self._layout(cfg).final_passages_path(self.recon_hash, self.pack_hash)

    # -- seed ---------------------------------------------------------------- #
    def lang_counts(self, cfg: RunConfig) -> dict[str, int]:
        """``source_lang -> passage count``, from the one text-free passage table.

        Column-pruned to ``lang`` alone (measured 82x for a single-column read) and cached;
        it is a count, so nothing is materialized per row. Read from the corpus rather than
        configured for the reason ``IndexAdapter.lang_counts`` is: the plan must be a
        function of the corpus and the part size and nothing else.
        """
        path = str(self.passages_path(cfg))
        if path not in self._counts:
            counts: dict[str, int] = {}
            for batch in iter_parquet_batches(path, columns=["lang"]):
                for row in batch:
                    lang = str(row["lang"])
                    counts[lang] = counts.get(lang, 0) + 1
            self._counts[path] = counts
        return self._counts[path]

    def part_passages(self, cfg: RunConfig) -> int:
        """Passages per assemble part: the hashed ``plaid_part_passages`` (see the module
        docstring for why the PLAID part, not the shard part, is the unit)."""
        return max(1, int(index_build_options(cfg).plaid_part_passages))

    def parts_for(self, cfg: RunConfig, source_lang: str) -> int:
        """How many parts ``source_lang`` is cut into: ``ceil(passages / part size)``.

        Never zero: a language with no passages is one empty part, because an empty index is
        a legal artifact and a language that contributed no unit at all would be invisible to
        ``merge``'s coverage assertion in exactly the way it must not be.
        """
        size = self.part_passages(cfg)
        return max(1, -(-int(self.lang_counts(cfg).get(source_lang, 0)) // size))

    def shard_specs(self, cfg: RunConfig) -> list[IndexShardSpec]:
        """The plan as specs: what ``shards`` seeds and what ``merge`` re-derives.

        Reuses ``IndexShardSpec`` rather than declaring a second ``(variant, source_lang,
        part)`` record: the two stages describe the same cell of the same corpus, and a
        parallel spec type is exactly how the two halves' notions of "part 3 of zh" would
        drift apart. ``en`` is planned once with ``variant=None``, ``Layout`` collapses it
        to ``_shared/en``, so the shared build is the path scheme rather than a copy step.
        """
        langs = sorted(str(x) for x in cfg.languages if str(x) != SHARED_LANG)
        specs = [
            IndexShardSpec(variant=variant, source_lang=lang, part=part, parts=parts)
            for variant in RENDERINGS
            for lang in langs
            for parts in (self.parts_for(cfg, lang),)
            for part in range(parts)
        ]
        shared_parts = self.parts_for(cfg, SHARED_LANG)
        specs.extend(
            IndexShardSpec(variant=None, source_lang=SHARED_LANG, part=part, parts=shared_parts)
            for part in range(shared_parts)
        )
        return specs

    def shards(self, cfg: RunConfig) -> Iterator[_Shard]:
        """The claimable units: ``variants x non-English languages x parts`` + shared English."""
        for spec in self.shard_specs(cfg):
            yield _Shard(name=spec.name, payload=spec.payload())

    # -- bringup ------------------------------------------------------------- #
    def bringup(self, cfg: RunConfig) -> AssembleCtx:
        """Resolve the recipe and the vector keys. No model is built, here or anywhere.

        ``serving.registry.build_clients`` is not called: this stage reads
        vectors, so an encoder would be a resident multi-GB object that changes nothing, and
        the recipe check it would enable is replaced by the stronger hash check in
        :meth:`_assert_cell_matches`. Two things are checked before a shard is claimed, both
        cheap and both fatal later if left:

        * the declared ``assemble_device`` exists (:func:`index._assert_assemble_device`),
          a ``cuda`` declaration silently falling back to CPU would publish bytes that differ
          from their siblings for no reason in the text;
        * blocks divide parts evenly (:class:`~ragtime.preprocess.vectors.BlockPlan`), so a
          geometry in which a part is not a whole run of block files fails at bring-up rather
          than half-way through the first stream.
        """
        opts = index_build_options(cfg)
        _assert_assemble_device(opts)
        # Constructing the plan is the check: BlockPlan refuses a part that is not a whole
        # number of blocks (BlockGeometryError), which is what keeps the read sequential.
        BlockPlan(
            passages=0,
            block_passages=max(1, int(opts.encode_block_passages)),
            part_passages=max(1, int(opts.plaid_part_passages)),
        )
        ctx = AssembleCtx(
            opts=opts,
            layout=self._layout(cfg),
            recon_hash=self.recon_hash,
            pack_hash=self.pack_hash,
            idx_hash=self.idx_hash,
            encode_hashes={leg: leg_encode_hash(opts, leg) for leg in self.legs},
            legs=self.legs,
        )
        _log.info(
            "preprocess.assemble.bringup",
            stage=self.stage,
            part_passages=ctx.part_passages,
            block_passages=ctx.block_passages,
            assemble_device=opts.assemble_device,
            encode_hashes=ctx.encode_hashes,
        )
        return ctx

    def ctx_for(self, cfg: RunConfig, ctx: AssembleCtx | None = None) -> AssembleCtx:
        """``ctx`` or a freshly brought-up one: used by ``validate``/``merge``."""
        return ctx if ctx is not None else self.bringup(cfg)

    # -- work ---------------------------------------------------------------- #
    def work(self, ctx: AssembleCtx, shard: Path) -> Path:
        """Assemble every not-yet-published leg of one part; return the queue receipt.

        Legs already carrying a ``_SUCCESS`` marker are skipped before their vectors are
        streamed, so a retry after a Seismic OOM re-reads nothing for the two legs that
        finished, the same per-leg atomicity ``IndexAdapter.work`` has, one stage later.
        """
        spec = IndexShardSpec.from_payload(_read_payload(shard))
        shard_dir = self.shard_dir_from_ctx(ctx, spec)
        pending = [leg for leg in ctx.legs if not is_done(shard_dir / leg)]
        for leg in ctx.legs:
            if leg not in pending:
                self.stats.emit(
                    STAT_LEG_RESUMED,
                    leg=leg,
                    lang=spec.source_lang,
                    variant=spec.text_rendering,
                )
        report: dict[str, Any] = {
            "variant": spec.variant,
            "source_lang": spec.source_lang,
            # Which rendering's vectors this part assembles: no text is read here; the name
            # is the shard spec's own, so the two stages slice their counters identically.
            "rendering": spec.text_rendering,
            "part": int(spec.part),
            "parts": int(spec.parts),
            "part_passages": ctx.part_passages,
            "shard_dir": str(shard_dir),
            "lang_dir": str(shard_dir.parent),
            "index_hash": self.idx_hash,
            "reconcile_hash": self.recon_hash,
            "packing_hash": self.pack_hash,
            "created_at": _now(),
            # ``validate`` runs on a bare path in any process, so the receipt names the table
            # its id set must be checked against rather than depending on a same-process
            # hand-off from ``work``.
            "passages_path": str(
                ctx.layout.final_passages_path(ctx.recon_hash, ctx.pack_hash)
            ),
            "legs": {},
            "worker": saturate.worker_provenance(),
        }
        refs: list[tuple[str, str]] | None = None
        for leg in pending:
            leg_refs = self._assemble_leg(ctx, spec, leg, shard_dir)
            if refs is None:
                refs = leg_refs
            elif leg_refs != refs:
                # The three legs read three different cells; their id maps are the same
                # concatenation of the same pinned order, so a difference means one cell was
                # written from a different plan (or a different corpus) and the legs would
                # address different passages under one ordinal space.
                raise IndexIntegrityError(
                    f"{shard_dir}: leg {leg!r} assembled a different passage sequence than "
                    f"its siblings ({len(leg_refs)} vs {len(refs)} rows, first difference at "
                    f"{_first_difference(refs, leg_refs)}), the legs of one part must be the "
                    "identical ordinal space, or an ordinal means three different things"
                )
        for leg in ctx.legs:
            leg_dir = shard_dir / leg
            report["legs"][leg] = {
                "path": str(leg_dir),
                "config_hash": leg_config_hash(ctx.opts, leg),
                # The vector key this leg was assembled from: the split's provenance record,
                # and what a re-assemble is checked against. Recorded per leg because an
                # encode change re-keys one leg's vectors, never all three.
                "encode_hash": ctx.encode_hashes[leg],
                "passages": _idmap_rows(leg_dir),
                "bytes": _dir_bytes(leg_dir),
                # Read off the published leg, so a resumed leg reports what it holds rather
                # than what this process happened to write.
                "plaid_parts": plaid_part_count(leg_dir) if is_done(leg_dir) else None,
            }
        # The census witness, written last: every leg above is published (or this line was
        # never reached), so a ``part-*`` directory carrying this file is a complete part and
        # one without it is an abandoned attempt. ``read_shard_parts`` reads exactly these.
        self._write_part_census(spec, shard_dir, ctx.part_passages)
        out = saturate.shard_out_path(shard)
        write_jsonl(out, [report], skip_if_done=False)
        _log.info(
            "preprocess.assemble.shard.done",
            shard=shard.name,
            variant=spec.variant,
            lang=spec.source_lang,
            part=spec.part,
            parts=spec.parts,
            assembled=pending,
            passages=0 if refs is None else len(refs),
        )
        return out

    def shard_dir_from_ctx(self, ctx: AssembleCtx, spec: IndexShardSpec) -> Path:
        """The part's published directory, resolved through ``Layout`` (never hand-built)."""
        return ctx.layout.index_shard_dir(
            ctx.recon_hash, ctx.idx_hash, spec.variant, spec.source_lang, part=spec.part
        )

    def cell_dir(self, ctx: AssembleCtx, spec: IndexShardSpec, leg: str) -> Path:
        """This leg's vector cell for ``(variant, source_lang)``: where the blocks live.

        Resolved through :meth:`ragtime.common.Layout.vectors_cell_dir`, the same method
        ``preprocess.vectorize.VectorizeAdapter.cell_dir`` writes those blocks through, so
        the encode half and the assemble half cannot each be self-consistent and still
        disagree about where a cell is. ``Layout`` owns the path scheme (never hand-built),
        and the cell is keyed by ``leg_encode_hash`` rather than by ``idx_hash``: an
        assemble-only recipe change (``plaid_nbits``, ``ann_factory``, the Seismic
        parameters) leaves the vectors' path, and therefore their ``_SUCCESS``, unmoved,
        which is the point of the encode/assemble key partition.
        """
        return ctx.layout.vectors_cell_dir(
            ctx.recon_hash,
            ctx.pack_hash,
            leg,
            ctx.encode_hashes[leg],
            spec.variant,
            spec.source_lang,
        )

    def _assemble_leg(
        self, ctx: AssembleCtx, spec: IndexShardSpec, leg: str, shard_dir: Path
    ) -> list[tuple[str, str]]:
        """Stream one part of one leg into its engine; publish atomically. Returns its refs.

        The loop is the module's reason for existing and is three lines: the
        blocks arrive in ascending order with gapless part-local ordinals (asserted by
        ``stream_part`` against its own cursor), so the writer sees exactly the ordinal space
        the id map records, and nothing more than one block is ever resident on our side,
        the engine's own buffering (one whole part, for PLAID) is the peak that matters.
        """
        cell = self.cell_dir(ctx, spec, leg)
        census = read_cell_census(cell)
        self._assert_cell_matches(ctx, spec, leg, cell, census)

        final = shard_dir / leg
        tmp = shard_dir / f".{leg}.tmp-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        writer: LegWriter = _WRITER_OF[leg](tmp, ctx.opts)
        refs: list[tuple[str, str]] = []
        for block in stream_part(
            cell, part=spec.part, part_passages=ctx.part_passages, census=census
        ):
            if not block.refs:
                # An empty block (a cell with no passages holds exactly one) is a legal
                # artifact, but it is not an add: ``_FaissWriter`` inspects ``reps[0]`` for
                # the unit-norm assertion, so handing it nothing would raise on a shape
                # rather than on a defect. Every writer publishes an empty index from
                # ``finish`` alone.
                continue
            writer.add(list(block.ordinals), _reps_for(leg, block.reps))
            refs.extend((ref.passage_id, ref.document_id) for ref in block.refs)
        writer.finish()

        empty = int(getattr(writer, "empty", 0) or 0)
        if empty:
            self.stats.emit(STAT_SPARSE_EMPTY, float(empty), leg=leg, lang=spec.source_lang)
        # The top-k cut's own loss, on the substage that applies it: ``seismic_top_k`` drops
        # components here, from vectors that were stored unpruned, so this is the only place
        # the number exists at all.
        pruned = int(getattr(writer, "pruned", 0) or 0)
        if pruned:
            self.stats.emit(
                STAT_SPARSE_PRUNED, float(pruned), leg=leg, lang=spec.source_lang
            )
        parts = int(getattr(writer, "parts", 0) or 0)
        if parts:
            self.stats.emit(
                STAT_PLAID_PARTS,
                float(parts),
                leg=leg,
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )
            self.stats.emit(
                STAT_PLAID_PART_BYTES,
                float(getattr(writer, "buffered_bytes", 0) or 0),
                leg=leg,
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )
        # The id map is written inside the leg's temp dir, so it is published by the same
        # atomic rename as the engine files: a leg can never be marked done with a missing or
        # stale id map. ``ordinal`` is the row's position in the concatenation of this part's
        # blocks' id maps: which is precisely what ``stream_part`` guarantees it read.
        _write_parquet_stream(
            tmp / IDMAP_FILENAME,
            (
                {
                    "ordinal": i,
                    "passage_id": passage,
                    "document_id": document,
                    "source_lang": spec.source_lang,
                }
                for i, (passage, document) in enumerate(refs)
            ),
            schema=idmap_arrow_schema(),
        )
        _publish_dir(tmp, final)
        self.stats.emit(
            STAT_LEG_BYTES,
            float(_dir_bytes(final)),
            leg=leg,
            lang=spec.source_lang,
            variant=spec.text_rendering,
        )
        self.stats.emit(
            STAT_ASSEMBLED,
            float(len(refs)),
            leg=leg,
            lang=spec.source_lang,
            variant=spec.text_rendering,
        )
        if leg == ctx.legs[0]:
            # Once per part, not once per leg: the three legs cover the identical passages
            # (asserted in ``work``), so three emissions would treble the count.
            self.stats.emit(
                STAT_PASSAGES,
                float(len(refs)),
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )
        _log.info(
            "preprocess.assemble.leg.published",
            leg=leg,
            path=str(final),
            lang=spec.source_lang,
            variant=spec.variant,
            part=spec.part,
            passages=len(refs),
            blocks=census.blocks,
            dtype=census.dtype,
        )
        return refs

    def _assert_cell_matches(
        self,
        ctx: AssembleCtx,
        spec: IndexShardSpec,
        leg: str,
        cell: Path,
        census: CellCensus,
    ) -> None:
        """The vectors must be this leg's, this rendering's, and this recipe's. By hash.

        This is the whole of the split's provenance check, and it replaces
        ``index._assert_clients_match_recipe`` rather than paralleling it: the encode
        identity is not compared to a live client (there is none) but to the
        ``leg_encode_hash`` every block wrote into its own witness, which covers the model,
        the revision, the tokenizer, the encode window, the store dtype, the batch
        composition, the device and the packing, because those are exactly the hash's body.
        A cell that disagrees is not assembled: the alternative is an index whose manifest
        describes vectors it does not contain.

        The plan is checked too (parts implied by the census vs the parts this shard was
        seeded under), because a cell vectorized under a different geometry would assemble a
        full-looking part over the wrong ordinals.
        """
        problems: list[str] = []
        if census.leg != leg:
            problems.append(f"leg: cell holds {census.leg!r}")
        if census.encode_hash != ctx.encode_hashes[leg]:
            problems.append(
                f"encode_hash: blocks record {census.encode_hash!r}, this recipe is "
                f"{ctx.encode_hashes[leg]!r}"
            )
        if census.lang != spec.source_lang:
            problems.append(f"lang: cell holds {census.lang!r}, shard is {spec.source_lang!r}")
        if census.variant != spec.variant:
            problems.append(f"variant: cell holds {census.variant!r}, shard is {spec.variant!r}")
        if problems:
            raise IndexIntegrityError(
                f"{cell}: the stored vectors are not the ones this shard assembles ("
                + "; ".join(problems)
                + "). Vector provenance is validated by hash, not by a client, a mismatch "
                "means the path and the bytes disagree, so nothing is published."
            )
        plan = census.plan(ctx.part_passages)
        if plan.parts != int(spec.parts) or not 0 <= int(spec.part) < plan.parts:
            raise IndexIntegrityError(
                f"{cell}: this shard is part {spec.part} of {spec.parts}, but the cell's "
                f"{census.passages} passages at {ctx.part_passages}/part are {plan.parts} "
                "part(s), assembling under a stale plan would index a prefix of the "
                "language while every per-part check still passed"
            )

    def _write_part_census(
        self, spec: IndexShardSpec, shard_dir: Path, part_size: int
    ) -> None:
        """Record this part's whole plan inside its own directory (``part.json``).

        Byte-for-byte the record ``IndexAdapter._write_part_census`` writes, because
        :func:`~ragtime.preprocess.index.read_shard_parts`, the reader every consumer uses
        is the same one: every part is an independent witness to the same plan, which is
        what establishes a cell's part count without a single writer that knows the whole
        cell (there is none) and without a glob (which cannot tell "9 parts" from "10 parts,
        one lost"). No timestamp: ``work`` re-writes this on every re-entry and
        a part assembled twice must be byte-identical (invariant 7).
        """
        write_jsonl(
            shard_dir / _PART_FILENAME,
            [
                {
                    "part": int(spec.part),
                    "parts": int(spec.parts),
                    "part_passages": int(part_size),
                    "passages": _idmap_rows(shard_dir / self.legs[0]),
                    "source_lang": spec.source_lang,
                    "variant": spec.variant,
                }
            ],
            skip_if_done=False,
        )

    # -- validate ------------------------------------------------------------ #
    def expected_ids_at(
        self,
        passages_path: Path,
        source_lang: str,
        *,
        part: int = 0,
        parts: int = 1,
        part_size: int = 0,
    ) -> frozenset[str]:
        """The ``passage_id`` set a part (or a whole language) must cover, from the table.

        ``part_size`` 0 means "the whole language is one part", which is what ``merge``'s
        corpus-coverage assertion means (an id set over the language: order-free, so no cut is
        involved at all).

        A part is rows ``[p*N, (p+1)*N)`` of this language in the passage table's row
        order, the cut ``vectorize`` actually made, taken from the one implementation of it
        (:func:`~ragtime.preprocess.index._row_order_lang_ids`, which
        :func:`~ragtime.preprocess.vectorize._expected_block_ids` also delegates to). This
        method used to slice the ``(token_count, passage_id)`` order instead, which is a
        different set whenever ``token_count`` is not monotone in row order, i.e. always, on
        the real corpus, so every cell with more than one part failed ``validate`` and went
        back to ``pending/`` until it poisoned. A part is a whole number of encode blocks, and
        a block is that same row-order arithmetic one size down
        (:class:`~ragtime.preprocess.vectors.BlockPlan`), so the two axes cannot disagree
        while there is one function.

        Validation therefore re-derives the part from the corpus rather than trusting the
        artifact, and re-derives the part count with it: a shard assembled under a stale plan
        would cover a prefix of its language while every per-part id check still passed.
        """
        size = int(part_size) or 0
        key = (source_lang, int(part), size)
        if key not in self._expected:
            if not size:
                self._expected[key] = frozenset(
                    row["passage_id"]
                    for row in iter_parquet(passages_path, columns=["passage_id", "lang"])
                    if row["lang"] == source_lang
                )
            else:
                found, implied = _row_order_lang_ids(
                    Path(passages_path), source_lang, size, [int(part)]
                )
                if implied != int(parts):
                    raise IndexIntegrityError(
                        f"({source_lang}) part {part}: the shard plan says {parts} part(s) of "
                        f"{size} passages, but {passages_path} implies {implied}, validating "
                        "under a stale plan would leave part of the language unindexed while "
                        "every per-part check still passed"
                    )
                self._expected[key] = found[int(part)]
        return self._expected[key]

    def expected_ids(self, cfg: RunConfig, source_lang: str) -> frozenset[str]:
        """Every ``passage_id`` of ``source_lang``: the whole language, not one part."""
        return self.expected_ids_at(self.passages_path(cfg), source_lang)

    def validate(self, path: Path) -> bool:
        """All three legs published and each leg's id set equal to this part's subset.

        Not a non-zero size: what is checked is that every leg covers exactly the passages
        this ``(variant, source_lang, part)`` unit should hold, nothing dropped by a
        truncated write, nothing duplicated, nothing shifted by a part boundary computed
        differently than the vectorize side cut it. A false return leaves the shard
        reclaimable. Every check ``IndexAdapter.validate`` makes is made here, against the
        same table and the same order.
        """
        if not (path.exists() and is_done(path)):
            return False
        report = json.loads(path.read_text(encoding="utf-8").strip())
        shard_dir = Path(report["shard_dir"])
        if not (shard_dir / _PART_FILENAME).exists():
            _log.warning("preprocess.assemble.validate.part_census_missing", path=str(shard_dir))
            return False
        passages_path = report.get("passages_path")
        expected = (
            self.expected_ids_at(
                Path(passages_path),
                str(report["source_lang"]),
                part=int(report.get("part") or 0),
                parts=int(report.get("parts") or 1),
                part_size=int(report.get("part_passages") or 0),
            )
            if passages_path
            else None
        )
        for leg in self.legs:
            leg_dir = shard_dir / leg
            if not is_done(leg_dir):
                _log.warning(
                    "preprocess.assemble.validate.leg_missing", leg=leg, path=str(leg_dir)
                )
                return False
            ids = _idmap_ids(leg_dir)
            if len(ids) != len(set(ids)):
                _log.warning("preprocess.assemble.validate.duplicate_ids", leg=leg)
                return False
            if expected is not None and frozenset(ids) != expected:
                _log.warning(
                    "preprocess.assemble.validate.id_set_mismatch",
                    leg=leg,
                    got=len(ids),
                    expected=len(expected),
                )
                return False
        return True

    # -- merge ---------------------------------------------------------------- #
    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> Path:
        """Assert cross-variant id integrity, then publish one manifest. Or publish nothing.

        The index build's four assertions, unchanged, this stage is now their owner because
        it is the stage that publishes an index:

        1. every ``(variant, source_lang, part)`` unit of the plan is present and carries all
           three legs, a leg on 2 of 3 renderings is a failure, not an exception;
        2. every cell's on-disk part census agrees with the plan
           (:func:`~ragtime.preprocess.index.read_shard_parts`), so a cell that lost a part
           is a failed publication rather than a short fan;
        3. the parts of one cell are disjoint, their id sets sum to their union, so a part
           built under the wrong boundary (which double-covers some ids and leaves others
           uncovered) cannot hide inside a union that still looks complete;
        4. each variant's id set, unioned over its language cells (its own three plus the
           shared English one), equals the packed passage table's id set exactly.

        And the proof, not an assumption: the three renderings' part ``p`` of language ``l``
        holds the same ids (compared by digest). It holds by construction, both halves cut on
        the row order of the one text-free table, which has no ``variant`` column, and it is
        the property the whole Task-2 comparison rests on, so it is machine-checked.
        """
        ctx = self.bringup(cfg)
        reports = [json.loads(Path(p).read_text(encoding="utf-8").strip()) for p in shard_paths]
        by_unit = {
            (r["variant"], r["source_lang"], int(r.get("part") or 0)): r for r in reports
        }
        specs = self.shard_specs(cfg)
        for spec in specs:
            report = by_unit.get((spec.variant, spec.source_lang, spec.part))
            if report is None:
                raise IndexIntegrityError(
                    f"no assemble output for ({spec.variant}, {spec.source_lang}, part "
                    f"{spec.part} of {spec.parts}), publishing a manifest now would name an "
                    "index that was never built"
                )
            if int(report.get("parts") or 0) != spec.parts:
                raise IndexIntegrityError(
                    f"({spec.variant}, {spec.source_lang}) part {spec.part} was assembled "
                    f"under a {report.get('parts')}-part plan but the corpus implies "
                    f"{spec.parts}, the parts on disk are not one cell"
                )
            missing = [leg for leg in self.legs if leg not in report["legs"]]
            if missing:
                raise IndexIntegrityError(
                    f"({spec.variant}, {spec.source_lang}, part {spec.part}) is missing "
                    f"leg(s) {missing}: every rendering carries exactly {LEGS}, so a leg "
                    "present on one rendering and absent from another is a hard failure"
                )
        digests: dict[tuple[str, int], dict[str, str]] = {}
        variants: dict[str, Any] = {}
        for variant in RENDERINGS:
            langs = sorted({s.source_lang for s in specs if s.variant == variant})
            shards: dict[str, Any] = {}
            covered: set[str] = set()
            for source_lang in (*langs, SHARED_LANG):
                cell_variant = None if source_lang == SHARED_LANG else variant
                cell = [
                    by_unit[(cell_variant, source_lang, s.part)]
                    for s in specs
                    if s.variant == cell_variant and s.source_lang == source_lang
                ]
                entry, ids = self._merge_cell(source_lang, cell)
                for part, _report in enumerate(cell):
                    digests.setdefault((source_lang, part), {})[variant] = entry[
                        "shard_parts"
                    ][part]["id_digest"]
                shards[source_lang] = entry
                covered |= ids
            self._assert_covers_corpus(cfg, variant, covered)
            variants[variant] = {"passages": len(covered), "shards": shards}
            self.stats.emit(STAT_ID_INTEGRITY, float(len(covered)), variant=variant)
        _assert_parts_identical_across_renderings(digests)
        shared = by_unit[(None, SHARED_LANG, 0)]
        manifest = {
            "index_hash": self.idx_hash,
            "reconcile_hash": self.recon_hash,
            "packing_hash": self.pack_hash,
            "legs": list(LEGS),
            "leg_config_hash": {
                leg: shared["legs"][leg]["config_hash"] for leg in self.legs
            },
            # The vector keys the three legs were assembled from: the split's provenance
            # record. It is what makes "these engines are this recipe's vectors" auditable
            # from the manifest alone, and what a re-assemble under a moved encode key is
            # checked against (a moved key resolves to a different vector cell entirely).
            "leg_encode_hash": dict(ctx.encode_hashes),
            "provenance": _provenance(cfg, reports),
            "variants": variants,
            "created_at": _now(),
        }
        out = self.manifest_path(cfg)
        write_jsonl(out, [manifest], skip_if_done=False)
        _log.info(
            "preprocess.assemble.done",
            stage=self.stage,
            path=str(out),
            variants=sorted(variants),
            part_passages=ctx.part_passages,
            passages={v: variants[v]["passages"] for v in sorted(variants)},
        )
        return out

    def _merge_cell(
        self, source_lang: str, cell: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], set[str]]:
        """One ``(variant, source_lang)`` cell's manifest entry + the ids it covers.

        Reads the cell's census off disk instead of trusting the reports it was handed, so a
        part directory that vanished between ``work`` and ``merge``, or was never published
        fails here rather than being published as a complete cell; and asserts the parts
        are disjoint, because a part assembled under a wrong boundary produces a union that
        can still look complete while some ids are covered twice and others not at all.
        """
        lang_dir = Path(cell[0]["lang_dir"])
        census = read_shard_parts(lang_dir)
        if census["parts"] != len(cell):
            raise IndexIntegrityError(
                f"{lang_dir}: {census['parts']} part(s) on disk but {len(cell)} assemble "
                "output(s), the cell is not the one the plan built"
            )
        parts: list[dict[str, Any]] = []
        covered: set[str] = set()
        total = 0
        for report in cell:
            ids = self._part_ids(report)
            entry = _manifest_shard(report)
            entry["id_digest"] = _id_digest(ids)
            # The vector provenance, carried per part beside the per-part worker: the two
            # together answer "which bytes, built where" without a second lookup.
            entry["encode_hash"] = {
                name: leg.get("encode_hash") for name, leg in sorted(report["legs"].items())
            }
            parts.append(entry)
            covered |= ids
            total += len(ids)
        if total != len(covered):
            raise IndexIntegrityError(
                f"{lang_dir}: its {len(cell)} parts cover {total} ids but only "
                f"{len(covered)} distinct ones, parts of one cell are disjoint slices of "
                "the table's row order, so an overlap means at least one was assembled under "
                "a different boundary and some passages are in no part"
            )
        return (
            {
                "path": str(lang_dir),
                "shared": cell[0]["variant"] is None,
                "parts": census["parts"],
                "part_passages": census["part_passages"],
                "part_dirs": census["dirs"],
                "passages": len(covered),
                "shard_parts": parts,
            },
            covered,
        )

    def _part_ids(self, report: Mapping[str, Any]) -> set[str]:
        """The id set a part actually covers: the intersection over its legs.

        Intersection, not union, (the rule
        :func:`~ragtime.preprocess.index._shard_ids` states one stage earlier): an id present
        in two legs and missing from the third is not covered by that rendering's index, and
        a union would hide exactly the partial-leg failure the integrity assertion exists to
        catch.
        """
        shard_dir = Path(report["shard_dir"])
        sets = [frozenset(_idmap_ids(shard_dir / leg)) for leg in self.legs]
        return set(sets[0]).intersection(*sets[1:]) if sets else set()

    def _assert_covers_corpus(self, cfg: RunConfig, variant: str, covered: set[str]) -> None:
        """``covered`` must equal the corpus's whole ``passage_id`` set, exactly."""
        langs = {spec.source_lang for spec in self.shard_specs(cfg)}
        full: set[str] = set()
        for lang in sorted(langs):
            full |= self.expected_ids(cfg, lang)
        if covered != full:
            raise IndexIntegrityError(
                f"variant {variant!r} covers {len(covered)} passage ids but the corpus has "
                f"{len(full)} (missing {len(full - covered)}, extra {len(covered - full)}), "
                "nothing partial is published, so the manifest is not written"
            )


# --------------------------------------------------------------------------- #
# Leg-shaped helpers.
# --------------------------------------------------------------------------- #
def _reps_for(leg: str, reps: Any) -> Any:
    """The stored reps in the form this leg's writer takes.

    Two of the three are already in it: the dense leg's ``(n, dim)`` array and the
    late-interaction leg's list of ``(n_i, dim)`` matrices are exactly what ``_FaissWriter``
    and ``_PlaidWriter`` add.

    The sparse one needs a mapping, because ``_SeismicWriter.add`` calls
    :func:`~ragtime.serving.sparse_milco.seismic_components` itself, while the stored form
    is that function's output (``vectors.parquet`` holds the ``(components, values)`` pair
    verbatim, so the two sides can never order the components differently). Rebuilding a
    mapping from the pair and letting the writer re-derive is exact, not approximate:
    components are the already-stringified dim ids, distinct, and sorted by that same string,
    so the re-sort is the identity and the values round-trip as float32. The alternative
    would be a second Seismic accumulation path in this module, i.e. a second chance to get
    the U30 cap or the component order wrong, which is precisely what importing the writer
    exists to avoid.
    """
    if leg != SPARSE_LEG:
        return reps
    return [dict(zip(components, values, strict=True)) for components, values in reps]


def _first_difference(
    left: Sequence[tuple[str, str]], right: Sequence[tuple[str, str]]
) -> int:
    """The first ordinal at which two id sequences differ (for the error message)."""
    for i, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return i
    return min(len(left), len(right))
