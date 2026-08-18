"""The fuse-then-rerank query path. One entry point; nothing forks it.

``retrieve(ctx, queries, top_k) -> [(passage_id, score)]`` is the whole public search surface: the
callable behind the RAG loop's ``search`` action. ``legs`` is a per-leg diagnostic probe, never a
ranking path.

Fusion happens at two levels, and getting them backwards produces a silently wrong ranking rather
than a crash:

* within one ``(variant, source_lang)`` cell, across its parts, scores merge raw. Parts of a cell
  are the same encoder over the same leg differing only in which passages they hold, and none of
  the three engines normalizes against anything index-global, so two parts report scores in the
  same space. RRF there would discard the magnitude that makes them comparable and would make the
  answer depend on how the corpus happened to be cut. This level is not implemented here: it is
  :func:`~ragtime.preprocess.index.query_lang_leg`, imported and called verbatim.
* across legs, languages and query variants, fusion is rank-based RRF (``common.fusion.rrf``,
  with ``k`` from ``retrieval.rrf.k``). Those score spaces genuinely differ -- a MaxSim, a SPLADE
  dot product and a cosine are not comparable magnitudes -- so rank is the only common currency.

Mirroring the two fusion levels there are two concurrency levels, and they nest. The outer one is
here (``_pools``: one unit per ``(query, leg, cell)``); the inner one is ``query_lang_leg``'s
per-part fan. Their widths are decided together by :mod:`~ragtime.retrieval.concurrency`. Both
fans collect in submission order, so every width returns the byte-identical ranked list; see
``_pools``' docstring for why that is a correctness property rather than a preference.

Ranking may happen in this process (``context.bring_up`` opened the index) or in a shared
retrieval service (``context.service_context`` bound a ``RankingBackend``). That choice is read in
exactly one statement, inside :func:`retrieve`, off ``ctx.backend``. Everything around it --
``query_strings``, the ``top_k`` truncation, all four counters -- is shared, so
``tool.search_action`` and every stage body are transport-blind.

Which transport a run took decides which fuse-then-rerank actually ran. Every submitted run bound
the service transport, so the ranking behind the reported numbers is the one in
:mod:`ragtime.devkit.rsvc`, not :func:`_rank_local` below. The two agree on the candidate set:
both fuse the same 3 legs by 4 language cells with ``common.fusion.rrf`` and hand the
cross-encoder the top ``retrieval.reranker.depth`` of that one fused ranking. There is no second
candidate-set policy on either path.

What leaves this module is ``(passage_id, score)`` pairs, never text. The only text this module
touches is the rerank pool's, fetched by id from the searched rendering and thrown away with the
pool. The cross-encoder scores ``passage_store.render(passage_id, ctx.index)``, the searched
rendering (Knob 1), never ``passage_lang`` (Knob 2) and never a hardcoded ``original``: reranking
the reading rendering would make what the ``mlir-*`` (Task-2) family retrieves depend on the
Task-1/3 reading knob, and reranking native text would be a no-op for the ``original`` arm and a
two-rendering hybrid for ``omt``/``omt_opus``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ragtime.common import get_logger
from ragtime.common.fusion import rrf

from ._index_api import IndexIntegrityError, query_lang_leg
from .stats import (
    STAT_CANDIDATES_FUSED,
    STAT_LEG_CELL_SEARCHES,
    STAT_LEG_SECONDS,
    STAT_QUERIES,
    STAT_QUERY_SECONDS,
    STAT_QUERY_STRINGS,
    STAT_RERANK_SECONDS,
    STAT_RERANKED,
    STAT_STORE_FETCH_SECONDS,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .context import RetrievalContext

__all__ = ["candidate_depth", "legs", "query_strings", "retrieve"]

_log = get_logger("retrieval.service")

#: The single space that joins several query strings into one cross-encoder query. The model's
#: ``search`` action may emit more than one string per turn; each is fused as its own pool, but
#: the reranker takes one query.
_QUERY_JOINER = " "

#: Named so a stack dump or a sampling profiler says which level a thread belongs to: this prefix
#: is the outer (leg by cell) fan; ``ragtime-query-fan`` is the inner per-part one.
_LEG_THREAD_PREFIX = "ragtime-query-leg-fan"


def query_strings(queries: str | Sequence[str]) -> list[str]:
    """Normalize the caller's query argument to distinct, non-empty strings, in order.

    A bare ``str`` is accepted and means one query, never four characters. Duplicates are dropped
    because fusing a pool with itself would double its RRF weight, which would make a model that
    repeated itself outrank one that did not.
    """
    raw = [queries] if isinstance(queries, str) else [str(q) for q in queries]
    out: list[str] = []
    for q in raw:
        text = q.strip()
        if text and text not in out:
            out.append(text)
    return out


def candidate_depth(ctx: RetrievalContext, top_k: int) -> int:
    """How deep each ``(leg, cell, query)`` pool goes: ``max(top_k, rerank depth)``.

    Deep enough that the reranker sees a full pool and never deeper, since every extra unit here
    is multiplied by the whole fan. Both inputs are caller- or config-supplied; no literal appears
    here.
    """
    return max(1, int(top_k), int(ctx.knobs.rerank_depth))


def _assert_leg_set(ctx: RetrievalContext, source_lang: str) -> None:
    """Every part of this cell must carry exactly the leg set this context queries.

    The build refuses to publish a leg set that is not ``LEGS``, but a reader must not depend on
    the writer's promise: a cell that lost a leg directory between build and query would otherwise
    be fused over silently, two legs deep, and the result list would be full-looking with
    plausible scores. Re-checking costs a dict comparison per cell per call.
    """
    handle = ctx.cells[source_lang]
    expected = set(ctx.leg_names)
    for part in handle.parts:
        present = set(part.legs)
        if present != expected:
            raise IndexIntegrityError(
                f"{part.shard_dir}: this part carries legs {sorted(present)} but the query "
                f"path fuses over {sorted(expected)}; fusing over a subset returns a "
                "full-looking, silently under-retrieved list"
            )


def _tasks(ctx: RetrievalContext, qs: Sequence[str]) -> list[tuple[str, str, str]]:
    """The fan's units, ``(query, leg, source_lang)``, in the canonical order.

    Written as one list rather than three nested loops so that submission order is a value the
    code can hold, compare and hand to a pool.
    """
    return [
        (query, leg, source_lang)
        for query in qs
        for leg in ctx.leg_names
        for source_lang in sorted(ctx.cells)
    ]


def _warm_cells(ctx: RetrievalContext) -> None:
    """Open every part's reader and id map before any pool exists.

    ``query_lang_leg`` warms the parts of the one cell it was given, which is enough when the fan
    below it is the only concurrency. It is not enough here: two units of this fan can address the
    same ``(leg, cell)`` (two query strings in one ``search`` action do exactly that), and
    ``LegHandle.reader`` is a lazy check-then-set. Two threads racing it would both miss and both
    open, which is harmless for the answer but costly in memory, since a second copy of a Seismic
    part is several gigabytes.

    So the concurrent region contains native search and nothing else.
    """
    for source_lang in sorted(ctx.cells):
        for part in ctx.cells[source_lang].parts:
            for leg in ctx.leg_names:
                part.reader(leg)
                part.passage_ids(leg)


def _pools(
    ctx: RetrievalContext, qs: Sequence[str], depth: int
) -> list[list[tuple[str, float]]]:
    """One ranked pool per ``(query string, leg, language cell)``: the RRF layer's input.

    The within-cell part fan happens one level below this loop, inside ``query_lang_leg``, so no
    code here ever sees a part. That is what keeps the raw-score merge in its single owner.

    The fan is concurrent and that changes nothing about what it returns. Width is
    ``ctx.leg_workers`` (``execution.query_leg_workers``, resolved once at bring-up by
    :mod:`~ragtime.retrieval.concurrency`, which also bounds the product against the inner
    per-part fan). Results are collected by ``Executor.map``, which yields in submission order, so
    the returned list is element-for-element the list the sequential loop produced.

    Submission order is a correctness requirement rather than tidiness. The hazard is not inside a
    pool (``query_lang_leg`` returns a total ``(-score, passage_id)`` order) but in what consumes
    the pools: :func:`_fuse` accumulates a float per id across pools (``score += 1/(k + rank)``),
    and float addition is not associative, so the same contributions summed in a different order
    can differ in the last bit and flip an otherwise-exact RRF tie under ``(-score, id)``.
    Reordering the pools could therefore reorder the answer, silently and rarely.

    Timings are collected per unit and emitted after the fan, in submission order, for two
    reasons: ``common.stats.Statistics.emit`` is a read-modify-write on a plain dict and is not
    thread-safe, and a counter bus that recorded work in completion order would make the
    diagnostics themselves nondeterministic.
    """
    for source_lang in sorted(ctx.cells):
        _assert_leg_set(ctx, source_lang)
    tasks = _tasks(ctx, qs)
    workers = min(int(getattr(ctx, "leg_workers", 1) or 1), len(tasks))

    def run(task: tuple[str, str, str]) -> tuple[list[tuple[str, float]], float]:
        query, leg, source_lang = task
        start = time.perf_counter()
        pool = query_lang_leg(ctx.cells[source_lang], leg, query, depth)
        return pool, time.perf_counter() - start

    if workers <= 1:
        outcomes = [run(task) for task in tasks]
    else:
        _warm_cells(ctx)
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=_LEG_THREAD_PREFIX
        ) as pool:
            outcomes = list(pool.map(run, tasks))

    for (_, leg, source_lang), (_, elapsed) in zip(tasks, outcomes, strict=True):
        ctx.stats.emit(STAT_LEG_CELL_SEARCHES, 1.0, leg=leg, lang=source_lang, variant=ctx.index)
        ctx.stats.emit(STAT_LEG_SECONDS, elapsed, leg=leg, lang=source_lang, variant=ctx.index)
    return [hits for hits, _ in outcomes]


def _fuse(ctx: RetrievalContext, pools: Sequence[Sequence[tuple[str, float]]]):
    """RRF across legs, languages and query variants, with ``k`` from config or ``rrf``'s default.

    An absent ``retrieval.rrf.k`` calls ``rrf`` with no ``k`` argument rather than passing a
    literal: the default belongs to ``common.fusion``, and a second copy of it here is what that
    function's docstring exists to prevent.
    """
    if ctx.knobs.rrf_k is None:
        return rrf(pools)
    return rrf(pools, k=ctx.knobs.rrf_k)


def _rerank(
    ctx: RetrievalContext,
    qs: Sequence[str],
    fused: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Re-score the top ``retrieval.reranker.depth`` fused candidates with the cross-encoder.

    The pool's text is ``render(passage_id, ctx.index)``, the searched rendering. The query is the
    space-join of the (possibly several) query strings, in one call.

    Returns the reranked head followed by the untouched RRF tail. In production the tail is never
    reached (``depth`` is at least any ``search_top_k``); it exists so a caller asking for more
    than ``depth`` gets a longer list rather than a truncated one, and its scores are RRF scores,
    so the returned score is always the score of the stage that decided that rank.

    The by-id text fetch and the cross-encoder are timed separately: one is a filesystem read and
    the other a GPU forward pass, and they move for unrelated reasons.
    """
    depth = int(ctx.knobs.rerank_depth)
    if ctx.reranker is None or depth <= 0 or not fused:
        return list(fused)
    head = list(fused[:depth])
    fetch_started = time.perf_counter()
    texts = [ctx.passage_store.render(pid, ctx.index) for pid, _ in head]
    ctx.stats.emit(
        STAT_STORE_FETCH_SECONDS, time.perf_counter() - fetch_started, variant=ctx.index
    )
    start = time.perf_counter()
    scores = ctx.reranker.score(_QUERY_JOINER.join(qs), texts)
    ctx.stats.emit(STAT_RERANK_SECONDS, time.perf_counter() - start, variant=ctx.index)
    ctx.stats.emit(STAT_RERANKED, float(len(head)), variant=ctx.index)
    rescored = [(pid, float(score)) for (pid, _), score in zip(head, scores, strict=True)]
    rescored.sort(key=lambda kv: (-kv[1], kv[0]))
    # The tail is the part of the fused ranking the cross-encoder never saw, in fused order. The
    # reranked set is exactly `fused[:depth]`, so the tail is exactly `fused[depth:]`.
    return rescored + list(fused[depth:])


def _rank_local(
    ctx: RetrievalContext, qs: Sequence[str], *, top_k: int
) -> tuple[list[tuple[str, float]], int]:
    """The in-process ranking: fan, two-level fusion, cross-encoder rerank, truncate.

    Split out of :func:`retrieve` so the transport dispatch above it is three lines and the
    counter emission below it is written once, for both transports.
    """
    depth = candidate_depth(ctx, top_k)
    pools = _pools(ctx, qs, depth)
    fused = _fuse(ctx, pools)
    return _rerank(ctx, qs, fused)[: max(0, int(top_k))], len(fused)


def retrieve(
    ctx: RetrievalContext,
    queries: str | Sequence[str],
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    """Search ``ctx``'s index for ``queries`` and return ``[(passage_id, score)]``, best first.

    ``top_k`` is caller-supplied; this package holds no default for it. The returned list is at
    most ``top_k`` long and carries ids and scores only, since text is a separate by-id fetch via
    :func:`ragtime.retrieval.display.display`.

    Stateless: two calls on one context are independent, and accumulating a nugget's
    ``retrieved[]`` trail across calls is the caller's job.

    This is the one place where "where does ranking happen" is a question. A context built by
    :func:`~ragtime.retrieval.context.service_context` carries a
    :class:`~ragtime.retrieval.endpoints.RankingBackend` and the ids-and-scores step travels over
    a filesystem queue to a shared resident index; a context built by
    :func:`~ragtime.retrieval.context.bring_up` ranks here. Everything around the dispatch is
    shared, so the two transports are indistinguishable to ``tool.search_action`` and to every
    stage body.
    """
    qs = query_strings(queries)
    if not qs:
        return []
    started = time.perf_counter()
    backend = getattr(ctx, "backend", None)
    if backend is None:
        out, fused = _rank_local(ctx, qs, top_k=top_k)
    else:
        out, fused = backend.rank(ctx, qs, top_k=top_k)
    # `fused` is None only when a remote reply carried no `fused_pool`; emitting 0 there would
    # invent a measurement, so the counter is simply not incremented.
    if fused is not None:
        ctx.stats.emit(STAT_CANDIDATES_FUSED, float(fused), variant=ctx.index)
    ctx.stats.emit(STAT_QUERIES, 1.0, variant=ctx.index)
    ctx.stats.emit(STAT_QUERY_STRINGS, float(len(qs)), variant=ctx.index)
    ctx.stats.emit(STAT_QUERY_SECONDS, time.perf_counter() - started, variant=ctx.index)
    _log.debug(
        "retrieval.service.retrieve",
        transport="in_process" if backend is None else str(backend.kind),
        # Beside the transport witness, and for the same reason: the mirror is declared in
        # `execution`, which `family_guard` does not compare, so the artifact is what records
        # which copy of the passage store was read.
        store=getattr(ctx, "store_witness", None),
        index=ctx.index,
        queries=len(qs),
        fused=fused,
        returned=len(out),
        leg_workers=int(getattr(ctx, "leg_workers", 1) or 1),
    )
    return out


def legs(
    ctx: RetrievalContext,
    query: str,
    *,
    top_k: int,
    source_lang: str | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Per-leg raw pools for one query. Diagnostic only, never the ranking path.

    Returns ``{leg: [(passage_id, raw engine score)]}``. It exists so a probe can read the
    engine's own number (the dense leg's self-retrieval score is exactly ``1.0`` under
    L2-normalized exact inner product, which RRF would erase), and for per-leg latency work.

    ``source_lang`` names one cell, which is the honest form: the returned scores are then exactly
    ``query_lang_leg``'s, raw and comparable. With ``source_lang=None`` the per-cell pools are
    concatenated and score-sorted for convenience, and that concatenation is not a fusion rule.
    Fusion across languages is RRF, in :func:`retrieve`, and nothing here may be copied into a
    ranking path. (Self-retrieval at rank 1 is a dense-only guarantee anyway: MILCO's unnormalized
    SPLADE score has no such bound and Seismic stores a pruned document that can under-score
    itself.)

    Sequential while :func:`retrieve` fans: this function reports per-leg cost and
    per-leg raw scores, and a number measured under contention with two other legs is not the
    per-leg number anyone asks it for.
    """
    cells = [source_lang] if source_lang is not None else sorted(ctx.cells)
    for lang in cells:
        if lang not in ctx.cells:
            raise KeyError(f"this index has no {lang!r} cell; it carries {ctx.languages}")
        _assert_leg_set(ctx, lang)
    out: dict[str, list[tuple[str, float]]] = {}
    for leg in ctx.leg_names:
        hits: list[tuple[str, float]] = []
        for lang in cells:
            hits.extend(query_lang_leg(ctx.cells[lang], leg, query, int(top_k)))
            ctx.stats.emit(
                STAT_LEG_CELL_SEARCHES, 1.0, leg=leg, lang=lang, variant=ctx.index
            )
        hits.sort(key=lambda kv: (-kv[1], kv[0]))
        out[leg] = hits[: int(top_k)]
    return out
