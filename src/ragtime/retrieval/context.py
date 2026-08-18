"""Query-time bring-up: config, clients and layout into one resident :class:`RetrievalContext`.

What this resolves once, so no per-query code path re-derives it:

* which index is searched -- ``retrieval.index`` (Knob 1), and which language cells that variant
  published, read off the index manifest rather than a glob or ``cfg.languages``: the manifest is
  the build's own record of what exists, and English collapses to one ``_shared/en`` cell shared
  by all three variants;
* the open cell handles -- one :class:`LangHandle` per ``(variant, source_lang)`` cell, each built
  from that cell's on-disk part census, so "this handle covers the whole language" is checked
  rather than assumed and a lost part raises here rather than being fanned over short;
* the three query encoders -- taken from the ``serving`` singleton registry's ``.index_dense`` /
  ``.milco`` / ``.mtd_colbert``, the encoders that built the index. Not ``.embedder``: that field
  is the query-time ``retrieval`` block's encoder, which no shipped ``e2e-*`` config carries, and
  using it would put queries and the stored vectors in different embedding spaces -- no crash,
  just silently wrong scores;
* the by-id passage store -- ``common.passage_store.LmdbPassageStore`` at
  ``Layout.passage_store_path(recon_hash, pack_hash)``, used by
  :mod:`~ragtime.retrieval.display` (Knob 2, reading) and by the rerank step (which scores the
  searched rendering, Knob 1). ``execution.passage_store_mirror_root`` may redirect the root this
  path is re-resolved under so a scored run reads a staged copy;
  :mod:`~ragtime.retrieval.store_mirror` owns the resolution, the verification and the fallback;
* the fusion and rerank knobs -- ``retrieval.rrf.k`` and ``retrieval.reranker.depth``, from
  config, never hardcoded here;
* both query-fan widths, as one decision -- ``execution.query_leg_workers`` (the outer
  ``(query, leg, cell)`` fan, kept on the context) and ``execution.query_part_workers`` (the inner
  per-part fan, forwarded onto :class:`IndexCtx`). They nest, so
  :mod:`~ragtime.retrieval.concurrency` resolves the pair.

It stands up no node and instantiates no model: the caller has already done that via
``serving.registry.build_clients`` and hands the
:class:`~ragtime.serving.registry.ClientBundle` in.

Two constructors return the same dataclass. :func:`bring_up` opens the index in this process.
:func:`service_context` does not: it binds a :class:`~ragtime.retrieval.endpoints.ServiceBackend`
and the ranking step travels over a filesystem queue to one resident service that many processes
share. Consequently ``retrieval.service.retrieve`` stays the one entry point, the transport is
dispatched inside it on ``ctx.backend`` and nowhere else, and no stage body branches on which
constructor produced its context. Everything Knob 2 needs is present either way -- the by-id LMDB
store, the configured ``passage_lang``, the stats bus -- so ``display()`` is production's in both;
only Knob 1 moves. What a service-backed context lacks is local index cells, and
:class:`_NoLocalCells` makes that a raise rather than an empty answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import Layout, Statistics, get_logger
from ragtime.common.io import is_done, read_jsonl
from ragtime.common.passage_store import RENDERINGS, LmdbPassageStore
from ragtime.serving.batching import Tier

from ._index_api import (
    IndexCtx,
    IndexIntegrityError,
    LangHandle,
    default_legs,
    index_build_options,
    index_hash,
    open_lang,
)
from .concurrency import QueryConcurrency, plan_query_concurrency
from .endpoints import (
    DEFAULT_MAX_AGE_S,
    DEFAULT_TIMEOUT_S,
    HttpServiceBackend,
    RetrievalServiceError,
    ServiceBackend,
)
from .stats import STAT_STORE_MIRROR_REFUSED, STAT_STORE_MIRRORED
from .store_mirror import StoreLocation, mirror_root, resolve_store_location

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

__all__ = [
    "RetrievalContext",
    "bring_up",
    "retrieval_knobs",
    "service_context",
]

_log = get_logger("retrieval.context")


class RetrievalConfigError(ValueError):
    """A ``retrieval``-block knob the query path needs is missing or unusable."""


@dataclass(frozen=True, slots=True)
class RetrievalKnobs:
    """The query-time knobs, read from the ``retrieval`` block. No defaults invented here.

    ``rrf_k`` is ``None`` when the config does not name one, and
    :func:`ragtime.common.fusion.rrf` is then called without a ``k`` argument so its own default
    applies and this package holds no second copy of that constant. ``rerank_depth`` has no such
    canonical owner, so an absent ``retrieval.reranker.depth`` is a hard error rather than a
    hidden default.

    ``rrf_quality_gated`` is read and never used: no code path, doc or paper section defines what
    it would mean, and inventing behaviour for it would be a run choice made in code instead of in
    the config. It is carried here so the knob is visibly accounted for rather than dropped.
    """

    index: str
    passage_lang: str
    rrf_k: int | None
    rerank_depth: int
    rrf_quality_gated: Any = None


def _block(cfg: Any) -> Mapping[str, Any]:
    blocks = getattr(cfg, "blocks", None) or {}
    return dict(blocks.get("retrieval") or {})


def _knob(cfg: Any, attr: str, key: str) -> str:
    """A knob's value from the resolved :class:`~ragtime.config.RunConfig` field, else the block.

    ``RunConfig`` resolves and enum-checks both knobs at load; the block fallback exists only for
    the hand-built configs the small fixtures use, and it lands on the same default
    (``"original"``) the schema resolves to.
    """
    value = getattr(cfg, attr, None)
    if value:
        return str(value)
    return str(_block(cfg).get(key) or "original")


def retrieval_knobs(cfg: Any) -> RetrievalKnobs:
    """Read the two knobs and the fusion/rerank parameters out of ``cfg``. Pure, no IO."""
    block = _block(cfg)
    index = _knob(cfg, "retrieval_index", "index")
    passage_lang = _knob(cfg, "passage_lang", "passage_lang")
    for name, value in (("retrieval.index", index), ("passage_lang", passage_lang)):
        if value not in RENDERINGS:
            raise RetrievalConfigError(
                f"{name}={value!r} is not one of the three renderings {RENDERINGS}"
            )
    rrf_block = dict(block.get("rrf") or {})
    reranker_block = dict(block.get("reranker") or {})
    raw_k = rrf_block.get("k")
    if "depth" not in reranker_block:
        raise RetrievalConfigError(
            "retrieval.reranker.depth is missing; it bounds the cross-encoder pool and has "
            "no canonical owner elsewhere in the codebase, so this package will not invent a "
            "default for it (all shipped configs declare depth: 100)"
        )
    depth = int(reranker_block["depth"])
    if depth < 0:
        raise RetrievalConfigError(f"retrieval.reranker.depth must be >= 0; got {depth}")
    return RetrievalKnobs(
        index=index,
        passage_lang=passage_lang,
        rrf_k=None if raw_k is None else int(raw_k),
        rerank_depth=depth,
        # Read and ignored; see RetrievalKnobs' docstring.
        rrf_quality_gated=rrf_block.get("quality_gated"),
    )


_LOCAL_SEARCH_MSG = (
    "this RetrievalContext is service-backed: it holds no index cells, because the index lives "
    "in another job. Local ranking would silently return an empty pool, so it raises instead. "
    "Rank through retrieval.service.retrieve (which dispatches on ctx.backend), or build the "
    "context with retrieval.bring_up to open the index in this process."
)


class _NoLocalCells(dict):
    """``cells`` for a service-backed context: touching it is a bug, so it says so.

    ``retrieval.service._pools`` starts with ``for lang in sorted(ctx.cells)`` and ``_tasks`` does
    the same. A plain empty ``dict`` there would fuse zero pools and return an empty ranked list,
    a confident, full-looking answer with no error anywhere. Raising turns that into the failure
    it is.

    Every read accessor refuses, not only ``__iter__``: ``legs(ctx, q, source_lang="es")`` probes
    with ``in`` and ``[]``, which never touch ``__iter__``, so a partial refusal would leave one
    diagnostic path silently answering from nothing. :attr:`RetrievalContext.languages` raises as
    a consequence of the same rule, and that is correct: "which language cells does this process
    hold" has no remote answer.
    """

    def __iter__(self):
        raise RetrievalServiceError(_LOCAL_SEARCH_MSG)

    def __len__(self):
        raise RetrievalServiceError(_LOCAL_SEARCH_MSG)

    def __contains__(self, key: object) -> bool:
        raise RetrievalServiceError(_LOCAL_SEARCH_MSG)

    def __getitem__(self, key: object):
        raise RetrievalServiceError(_LOCAL_SEARCH_MSG)

    def get(self, key: object, default: Any = None):
        raise RetrievalServiceError(_LOCAL_SEARCH_MSG)

    def keys(self):
        raise RetrievalServiceError(_LOCAL_SEARCH_MSG)

    def values(self):
        raise RetrievalServiceError(_LOCAL_SEARCH_MSG)

    def items(self):
        raise RetrievalServiceError(_LOCAL_SEARCH_MSG)


@dataclass(slots=True)
class RetrievalContext:
    """Everything one process needs to answer queries against one published index.

    Immutable in spirit and stateless across calls: nothing here accumulates results, so
    ``retrieve(ctx, A)`` then ``retrieve(ctx, B)`` returns exactly what a fresh context would, and
    owning the per-nugget ``retrieved[]`` trail is the caller's job. The only mutable things it
    holds are the index readers and the LMDB env, which are caches of bytes on disk, not of
    answers.
    """

    knobs: RetrievalKnobs
    cells: dict[str, LangHandle]
    index_ctx: IndexCtx
    leg_names: tuple[str, ...]
    passage_store: Any
    reranker: Any
    layout: Layout
    recon_hash: str
    pack_hash: str | None
    idx_hash: str
    #: Width of the outer ``(query, leg, cell)`` fan, resolved once at bring-up by
    #: :func:`ragtime.retrieval.concurrency.plan_query_concurrency`, which also decided the inner
    #: per-part width sitting on ``index_ctx.query_workers``. Defaults to ``1`` (sequential) so a
    #: hand-built context is never accidentally concurrent.
    leg_workers: int = 1
    #: Where ranking happens, and the only thing that differs between the two constructors.
    #: ``None`` means this process holds the index cells (:func:`bring_up`); otherwise a
    #: :class:`~ragtime.retrieval.endpoints.RankingBackend` answering over a transport
    #: (:func:`service_context`). It is read in exactly one place,
    #: ``retrieval.service.retrieve``.
    backend: Any = None
    manifest: Mapping[str, Any] = field(default_factory=dict)
    stats: Statistics = field(default_factory=Statistics)
    #: Which copy of the by-id passage store this context opened: the canonical artifact or a
    #: verified staged mirror of it (:mod:`~ragtime.retrieval.store_mirror`). ``None`` only when
    #: the store was injected, where the caller owns the path. Like :attr:`transport` it is a
    #: run-record witness, never a branch, since the mirror is declared in ``execution``, which
    #: ``family_guard`` does not compare.
    store_location: StoreLocation | None = None
    #: Set when this context opened the passage store itself, and so owns closing it.
    _owns_store: bool = False

    @property
    def store_witness(self) -> dict[str, Any] | None:
        """The store half of the per-search record, or ``None`` for an injected store."""
        return None if self.store_location is None else self.store_location.witness

    @property
    def transport(self) -> str:
        """``"in_process"`` or ``"service"``, for the run record rather than for a branch.

        ``config.fairness.family_guard`` does not compare the ``execution`` block, so two members
        of one run family could take different transports and the config hash would not notice.
        The artifact is therefore the witness: every search records this plus the answering
        service's ``(rendering, index_hash, job_id)``.
        """
        return "in_process" if self.backend is None else str(self.backend.kind)

    @property
    def index(self) -> str:
        """Knob 1: the rendering that is searched (and the one rerank scores against)."""
        return self.knobs.index

    @property
    def passage_lang(self) -> str:
        """Knob 2: the default rendering :func:`~ragtime.retrieval.display.display` reads."""
        return self.knobs.passage_lang

    @property
    def languages(self) -> tuple[str, ...]:
        """The source languages this variant published, in a stable order."""
        return tuple(sorted(self.cells))

    def close(self) -> None:
        """Close the passage store if this context opened it (idempotent)."""
        if self._owns_store and self.passage_store is not None:
            close = getattr(self.passage_store, "close", None)
            if close is not None:
                close()
            self._owns_store = False


def _read_manifest(layout: Layout, recon_hash: str, idx_hash: str) -> Mapping[str, Any]:
    """The published index manifest, which is required and required to be marked done.

    A query path that fell back to globbing the index tree would happily search a build that was
    never published; ``merge`` refuses to write this file unless every cell of every variant
    covers the corpus exactly, so the manifest is that refusal made readable.
    """
    path = layout.index_manifest_path(recon_hash, idx_hash)
    if not (path.exists() and is_done(path)):
        raise IndexIntegrityError(
            f"{path}: no published index manifest; the index for recon={recon_hash[:12]} "
            f"idx={idx_hash[:12]} was never completed (assemble.merge writes this file only "
            "after every variant's id set covers the corpus exactly), and a query path must "
            "not fall back to whatever cells happen to be on disk"
        )
    return read_jsonl(path)[0]


def _cells_of(manifest: Mapping[str, Any], variant: str) -> tuple[str, ...]:
    variants = dict(manifest.get("variants") or {})
    entry = variants.get(variant)
    if entry is None:
        raise IndexIntegrityError(
            f"the published index carries variants {sorted(variants)} but "
            f"retrieval.index={variant!r} was requested; searching a rendering that was "
            "never built is unconstructible, not a fallback"
        )
    langs = tuple(sorted(dict(entry.get("shards") or {})))
    if not langs:
        raise IndexIntegrityError(f"variant {variant!r} names no language cells in the manifest")
    return langs


def _assert_pack_agrees(manifest: Mapping[str, Any], pack_hash: str | None) -> None:
    """The caller's ``pack_hash`` must be the one the index was built from.

    ``recon_hash`` cannot disagree, since it is a path level and a wrong one finds no manifest at
    all. ``pack_hash`` is not, and a passage store built from one packing served beside an index
    built from another is precisely the stale-artifact pairing the two-hash level exists to make
    unaddressable.
    """
    claimed = manifest.get("packing_hash")
    if claimed is None and pack_hash is None:
        return
    if claimed is None or pack_hash is None or str(claimed) != str(pack_hash):
        raise IndexIntegrityError(
            f"this index was built from packing {claimed!r} but the caller asked to query it "
            f"with pack_hash={pack_hash!r}; the passage ids, ordinals and by-id store all "
            "belong to one packing, so the two must be the same fact"
        )


def bring_up(
    cfg: Any,
    clients: Any,
    layout: Layout,
    *,
    recon_hash: str,
    pack_hash: str | None,
    idx_hash: str | None = None,
    passage_store: Any = None,
    legs: Sequence[Any] | None = None,
    stats: Statistics | None = None,
) -> RetrievalContext:
    """Open one published index and its by-id store for querying. Loads no weights.

    ``recon_hash`` and ``pack_hash`` are required keyword arguments with no defaults: the two
    corpus keys are computed by ``preprocess.reconcile.reconcile_hash`` and
    ``preprocess.packing.packing_hash``, which are outside this package's import allowlist (see
    :mod:`ragtime.retrieval._index_api`), and re-deriving either here would put a second
    implementation of a composite corpus key in the tree. The caller, which already resolved them
    to find the corpus at all, hands them in. ``None`` for ``pack_hash`` is legal and names the
    legacy pre-``packing``-block artifact, exactly as ``Layout``'s own required-but-nullable
    parameter does, and must be spelled out so every legacy read is greppable.

    ``idx_hash`` defaults to ``index_hash(cfg)``, so the common case names no hash at the call
    site.

    ``passage_store`` may be injected (a fixture's in-memory ``PassageStore``, or a store the
    caller already opened and wants shared); otherwise this opens the LMDB store at the
    Layout-resolved path and the returned context owns closing it. "Owns" means one handle, not
    the environment: :class:`~ragtime.common.passage_store.LmdbPassageStore` reference-counts a
    single read env per store per process, so a second context over the same corpus shares that
    env and :meth:`RetrievalContext.close` releases only this context's reference. An injected
    store is never closed here.
    """
    knobs = retrieval_knobs(cfg)
    resolved_idx = str(idx_hash if idx_hash is not None else index_hash(cfg))
    manifest = _read_manifest(layout, recon_hash, resolved_idx)
    _assert_pack_agrees(manifest, pack_hash)

    concurrency = plan_query_concurrency(cfg)
    leg_impls = tuple(legs) if legs is not None else default_legs()
    index_ctx = _index_ctx(
        cfg, layout, leg_impls, recon_hash, pack_hash, resolved_idx, clients, concurrency
    )

    cells: dict[str, LangHandle] = {}
    for source_lang in _cells_of(manifest, knobs.index):
        lang_dir = layout.index_lang_dir(recon_hash, resolved_idx, knobs.index, source_lang)
        cells[source_lang] = open_lang(lang_dir, index_ctx)

    stats_bus = stats if stats is not None else Statistics()
    store, owns, store_location = _open_store(
        layout,
        recon_hash,
        pack_hash,
        passage_store,
        why=(
            "display() needs O(1) by-id text over the whole corpus, and it is built once by "
            "preprocess.passage_store_build.PassageStoreAdapter (never lazily here)"
        ),
        root=mirror_root(cfg),
        stats=stats_bus,
    )

    ctx = RetrievalContext(
        knobs=knobs,
        cells=cells,
        index_ctx=index_ctx,
        leg_names=tuple(leg.name for leg in leg_impls),
        passage_store=store,
        reranker=getattr(clients, "reranker", None),
        layout=layout,
        recon_hash=recon_hash,
        pack_hash=pack_hash,
        idx_hash=resolved_idx,
        leg_workers=concurrency.leg_workers,
        manifest=manifest,
        stats=stats_bus,
        store_location=store_location,
        _owns_store=owns,
    )
    _log.info(
        "retrieval.context.up",
        transport=ctx.transport,
        store=ctx.store_witness,
        index=knobs.index,
        passage_lang=knobs.passage_lang,
        languages=list(ctx.languages),
        parts={lang: len(handle.parts) for lang, handle in sorted(cells.items())},
        legs=list(ctx.leg_names),
        rerank_depth=knobs.rerank_depth,
        # Both fan widths, because the pair is what can oversubscribe a node: `bounded` is false
        # only when an operator explicitly named both.
        leg_workers=concurrency.leg_workers,
        part_workers=concurrency.part_workers,
        concurrency_bounded=concurrency.bounded,
    )
    return ctx


def _open_store(
    layout: Layout,
    recon_hash: str,
    pack_hash: str | None,
    store: Any,
    *,
    why: str,
    root: Path | None = None,
    stats: Statistics | None = None,
) -> tuple[Any, bool, StoreLocation | None]:
    """The by-id passage store, or a refusal, for both constructors so they cannot disagree.

    Returns ``(store, owns, location)``. An injected store is never owned and never closed here,
    and reports ``location=None``, since the caller owns that path and this function resolved
    nothing.

    The published check is applied to the canonical store, and applied first, before any mirror is
    considered: the store is built once by ``preprocess``, a query path that lazily built its own
    would be building a corpus artifact inside a stage, and a staged copy may make a published
    store fast but never stand in for one that was never built.

    ``root`` is ``execution.passage_store_mirror_root``
    (:func:`~ragtime.retrieval.store_mirror.mirror_root`). When it is set, the same
    ``recon12``/``pack12``-keyed path is re-resolved beneath it and verified; an unusable mirror
    falls back to the origin loudly, with one ``retrieval.store_mirror_refused`` counter and one
    warning carrying the reason, because a silent downgrade would leave nothing in the record to
    explain the slowdown.
    """
    if store is not None:
        return store, False, None
    location = resolve_store_location(layout, recon_hash, pack_hash, root=root)
    if not (location.origin.exists() and is_done(location.origin)):
        raise IndexIntegrityError(
            f"{location.origin}: no published by-id passage store; {why}"
        )
    if location.refusal is not None:
        if stats is not None:
            stats.emit(STAT_STORE_MIRROR_REFUSED, 1.0)
        _log.warning(
            "retrieval.store_mirror.refused",
            origin=str(location.origin),
            mirror_root=str(root),
            reason=location.refusal,
        )
    elif location.mirrored:
        if stats is not None:
            stats.emit(STAT_STORE_MIRRORED, 1.0)
        _log.info(
            "retrieval.store_mirror.used",
            origin=str(location.origin),
            path=str(location.path),
        )
    return LmdbPassageStore(location.path), True, location


def service_context(
    cfg: Any,
    layout: Layout,
    *,
    registry: str | Path,
    recon_hash: str,
    pack_hash: str | None,
    idx_hash: str,
    rendering: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    stats: Statistics | None = None,
    passage_store: Any = None,
    http_url: str | None = None,
) -> RetrievalContext:
    """A :class:`RetrievalContext` whose ranking is answered by a shared retrieval service.

    The second constructor of the same dataclass, not a second context type, not a subclass. What
    differs from :func:`bring_up` is exactly three things, each a consequence of the index living
    elsewhere:

    * ``cells`` is :class:`_NoLocalCells`, which raises rather than fusing zero pools;
    * ``index_ctx`` is ``None`` and ``leg_names`` is empty: there are no local handles to fan
      over, and the two ``execution.query_*_workers`` widths are inert here because the service
      has its own;
    * ``reranker`` is ``None``, because the service reranks, on the searched rendering, with the
      config's own model. A client-side reranker would be a second scoring pass over an
      already-reranked list, with a different model from the one the lane was launched with.

    Everything else is identical: the same ``knobs`` read by the same
    :func:`retrieval_knobs`, the same by-id LMDB store so ``display()`` is production's, and the
    same stats bus.

    ``registry``, ``recon_hash``, ``pack_hash`` and ``idx_hash`` are required and have no
    defaults. ``registry`` in particular: a default would let a caller address "whatever fleet
    happens to be on this filesystem", and the registry root is a site fact, not a run choice.

    ``rendering`` is optional and means only "say out loud which lane you expect". It must equal
    the config's ``retrieval.index``, and disagreeing is refused before a byte is written, because
    Knob 1 belongs to the config, which is the run record.
    """
    knobs = retrieval_knobs(cfg)
    wanted = knobs.index if rendering is None else str(rendering)
    if wanted not in RENDERINGS:
        raise RetrievalConfigError(f"rendering {wanted!r} is not one of {RENDERINGS}")
    if wanted != knobs.index:
        raise RetrievalServiceError(
            f"the requested lane {wanted!r} disagrees with this config's retrieval.index "
            f"{knobs.index!r}. Knob 1 belongs to the config, which is the run record."
        )
    stats_bus = stats if stats is not None else Statistics()
    store, owns, store_location = _open_store(
        layout,
        recon_hash,
        pack_hash,
        passage_store,
        why=(
            "the service ranks, but the text the model reads is fetched here by id in "
            "passage_lang; that store is built once by preprocess, never lazily by a client"
        ),
        root=mirror_root(cfg),
        stats=stats_bus,
    )
    ctx = RetrievalContext(
        knobs=knobs,
        cells=_NoLocalCells(),
        index_ctx=None,
        leg_names=(),
        passage_store=store,
        reranker=None,
        layout=layout,
        recon_hash=recon_hash,
        pack_hash=pack_hash,
        idx_hash=str(idx_hash),
        backend=_service_backend(
            registry=registry, wanted=wanted, idx_hash=idx_hash,
            timeout_s=timeout_s, max_age_s=max_age_s, http_url=http_url,
        ),
        stats=stats_bus,
        store_location=store_location,
        _owns_store=owns,
    )
    _log.info(
        "retrieval.context.up",
        transport=ctx.transport,
        store=ctx.store_witness,
        index=knobs.index,
        passage_lang=knobs.passage_lang,
        rerank_depth=knobs.rerank_depth,
        registry=str(registry),
        idx_hash=str(idx_hash),
    )
    return ctx


def _service_backend(
    *,
    registry: str | Path,
    wanted: str,
    idx_hash: str,
    timeout_s: float,
    max_age_s: float,
    http_url: str | None,
) -> Any:
    """Pick the transport to the one lane: same lane, same refusals, different pipe.

    The filesystem queue is the default and the production path: it needs no port, no forward and
    no second process.

    ``http_url`` covers the case the queue structurally cannot serve, namely a worker on a cluster
    that cannot see the queue. Two clusters that share no filesystem give a worker on one no path
    to the other's queue directory, and an ssh forward to a bridge beside it is the only route.

    This is a transport switch and not a lane switch: ``registry``, ``rendering`` and
    ``index_hash`` are carried on both branches, :class:`HttpServiceBackend` overrides only
    ``_ask``, and the reply-lane check still runs client-side. A wrong ``http_url`` can address a
    different service; it cannot make a wrong-lane answer look right.

    The environment fallback lives here rather than in a stage body: which transport reaches the
    service is a site fact, like the registry root, and a stage that can name a transport is a
    second place that has to agree with this one. An explicit argument still wins, so a caller
    that genuinely knows is not overridden by the environment.
    """
    if http_url is None:
        http_url = os.environ.get("RAGTIME_RSVC_HTTP", "").strip() or None
    if http_url:
        return HttpServiceBackend(
            registry=Path(registry), rendering=wanted, index_hash=str(idx_hash),
            timeout_s=float(timeout_s), max_age_s=float(max_age_s), url=str(http_url),
        )
    return ServiceBackend(
        registry=Path(registry), rendering=wanted, index_hash=str(idx_hash),
        timeout_s=float(timeout_s), max_age_s=float(max_age_s),
    )


def _index_ctx(
    cfg: Any,
    layout: Layout,
    leg_impls: Sequence[Any],
    recon_hash: str,
    pack_hash: str | None,
    idx_hash: str,
    clients: Any,
    concurrency: QueryConcurrency,
) -> IndexCtx:
    """The index's own resident context, wired to the encoders that built the index.

    ``dense=clients.index_dense`` / ``milco=clients.milco`` / ``mtd=clients.mtd_colbert``, not
    ``clients.embedder``. Reading the wrong field is a silent correctness bug, since queries and
    stored vectors in different embedding spaces still produce numbers, so the three reads live
    here in one place.

    ``query_workers`` is the inner (per-part) fan's width and is forwarded raw: what a value means
    belongs to ``preprocess.index.resolve_query_workers``, which is not in this package's import
    allowlist, so the single owner still decides. Which raw value gets forwarded is
    :func:`~ragtime.retrieval.concurrency.plan_query_concurrency`'s ``part_workers`` rather than
    the config leaf read straight out of the block; the two differ only when
    ``execution.query_part_workers`` is absent while the outer leg fan is active, where the plan
    substitutes ``1`` to bound the nesting.
    """
    opts = index_build_options(cfg)
    return IndexCtx(
        dense=clients.index_dense,
        milco=clients.milco,
        mtd=clients.mtd_colbert,
        opts=opts,
        legs=tuple(leg_impls),
        layout=layout,
        recon_hash=recon_hash,
        pack_hash=pack_hash or "",
        idx_hash=idx_hash,
        tier=Tier(
            token_budget=opts.encode_batch_token_budget,
            max_items=opts.encode_max_items,
        ),
        query_workers=concurrency.part_workers,
    )
