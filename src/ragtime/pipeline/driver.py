"""The per-cell entry point: one ``(topic, variant, seed)`` solved on one GPU pair.

``orchestration`` invokes this via ``run --stage pipeline``. It checks ``run.kind`` and is the one
place decompose, the RAG-loop fan and the coverage audit are joined into a single cell.

A GPU pair owns its topic end to end: that topic's decompose, all k of its loops, its audit rounds
and its synthesis are served by that vLLM and no other. Scale out by giving a second pair a second
topic, never by splitting one topic across pairs. Beyond throughput this also protects round-0
parity, since sampled decoding is not batch-invariant and decompose must not share its instance
with concurrent work.

Two entry points are kept. :func:`drive` is the production one: the ``run.kind`` check, the full
coverage loop, the resume marker. :func:`solve_topic` is round 0 plus one fan, with no audit and no
artifact beyond the loop records, which is what fan-concurrency measurements run against. The two
share :func:`~ragtime.pipeline.round_loop.run_round` and nothing else, so neither can drift into
being a slightly different version of the other.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragtime.common import Layout, Statistics, get_logger
from ragtime.common.io import write_jsonl
from ragtime.common.stats import flush as flush_stats
from ragtime.pipeline.decompose import grow_nuggets
from ragtime.pipeline.records import write_round_records
from ragtime.pipeline.round_loop import CoverageLoopResult, run_round, run_rounds

__all__ = [
    "KIND_E2E",
    "CellResult",
    "drive",
    "solve_topic",
    "topic_layout",
]

_log = get_logger("pipeline.driver")

#: Fallback loop ceiling when neither the config nor the node handle supplies one. Deliberately
#: small, so an unconfigured run cannot oversubscribe the KV cache. The intended sources are
#: ``serving.capacity.derive_loop_ceiling`` or an explicit ``rag_loop.fan_out.concurrency``.
_CEILING_FALLBACK = 4


@dataclass(frozen=True)
class CellResult:
    """What one ``(topic, variant, seed)`` produced, and what it cost.

    ``wall_s`` is reported as two phases: ``decompose_s`` is sequential by necessity, while
    ``loops_s`` is the part concurrency compresses. A single total would hide which half to
    optimize.
    """

    topic_id: str
    seed: int
    nuggets: int
    records: tuple[Any, ...]
    decompose_s: float
    loops_s: float
    speedup: float
    search_wall_s: float
    errors: tuple[str, ...] = ()

    @property
    def wall_s(self) -> float:
        return self.decompose_s + self.loops_s

    def perf_line(self, *, job: str = "?", pins: str = "?") -> str:
        """Return a one-line timing summary for this cell, with its job and hardware context.

        Built here because this object is the only thing that knows all four fields, and a timing
        quoted without its context cannot be compared against another run's.
        """
        return (
            f"timing: {self.wall_s:.1f} s/topic "
            f"[{time.strftime('%Y-%m-%d')}, job {job}, {pins}, "
            f"k={self.nuggets} loops, speedup {self.speedup:.2f}x, "
            f"unit=1 topic round-0 + fan]"
        )


def resolve_ceiling(cfg: Any, node: Any = None) -> int:
    """Return the fan's ceiling: explicit config, else the node's KV-derived headroom, else a floor.

    ``rag_loop.fan_out.concurrency`` is fairness-shared config, so a run that pins it is the run
    record and no derivation may override it. Only when it is null do we ask the live instance,
    whose number tracks actual KV headroom.
    """
    block = {}
    if hasattr(cfg, "blocks"):
        block = (cfg.blocks.get("rag_loop", {}) or {}).get("fan_out", {}) or {}
    pinned = block.get("concurrency")
    if pinned is not None:
        return max(1, int(pinned))
    derived = getattr(node, "loop_ceiling", None)
    if derived:
        return max(1, int(derived))
    return _CEILING_FALLBACK


async def solve_topic(
    cfg: Any,
    topic: Any,
    *,
    llm: Any,
    ctx: Any,
    clients: Any,
    layout: Layout,
    seed: int = 0,
    ceiling: int | None = None,
    stats: Statistics | None = None,
    node: Any = None,
) -> CellResult:
    """Run round 0 plus the fan for one topic.

    ``llm``, ``ctx`` and ``clients`` are injected for the same reason ``run_loop`` takes them:
    this function opens no index, loads no model and builds no client, so a test can drive the
    real wiring without a GPU while a production caller keeps the one shared singleton.
    """
    stats = stats if stats is not None else Statistics()
    topic_id = str(_field(topic, "topic_id") or _field(topic, "id") or "")
    problem = str(_field(topic, "problem_statement") or "")
    background = str(_field(topic, "background") or "")
    limit = int(_field(topic, "limit") or 5000)
    # The request's short label, handed to decompose only; `run_round` fans request-blind loops.
    # Read through `_field` so a title-less topic degrades to "" and renders no title section.
    title = str(_field(topic, "title") or "")

    started = time.perf_counter()
    bank = await grow_nuggets(
        problem, background, (), None, 0,
        cfg=cfg, clients=clients, topic_id=topic_id, limit=limit, seed=seed, stats=stats,
        title=title,
    )
    decompose_s = time.perf_counter() - started

    k = resolve_ceiling(cfg, node) if ceiling is None else max(1, int(ceiling))
    started = time.perf_counter()
    round0 = await run_round(
        bank, cfg, llm=llm, ctx=ctx, ceiling=k, round_no=0, seed=seed,
        stats=stats, passage_lang=getattr(cfg, "passage_lang", None),
    )
    loops_s = time.perf_counter() - started

    paths = write_round_records(layout, round0, cfg, topic_id=topic_id, seed=seed)
    stats.emit("pipeline.topic_solved", 1, nugget=topic_id)

    return CellResult(
        topic_id=topic_id,
        seed=seed,
        nuggets=len(bank),
        records=paths,
        decompose_s=decompose_s,
        loops_s=loops_s,
        speedup=round0.speedup,
        search_wall_s=round0.search_wall_s,
        errors=round0.errors,
    )


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


#: The one ``run.kind`` this pipeline implements. ``config.loader`` has already validated the
#: value against its enum; this constant is what :func:`drive` matches it against, so a kind
#: added there without a code path here fails loudly instead of defaulting to `e2e`.
KIND_E2E = "e2e_agentic"


def drive(
    cfg: Any,
    topic: Any,
    variant: str,
    seed: int,
    *,
    ctx: Any,
    clients: Any = None,
    llm: Any = None,
    layout: Layout | None = None,
    stats: Statistics | None = None,
    node: Any = None,
) -> Path:
    """Solve one ``(topic, variant, seed)`` cell end to end and return its run dir.

    ``run.kind`` must be ``e2e_agentic``, the only kind this pipeline runs: round 0 alone, then
    audit rounds until :func:`~ragtime.pipeline.decompose.saturation.saturated` fires or ``R_max``
    is reached. Writes ``decompose/round_{r}.jsonl`` per round and ``rag_loop/{nugget_id}.jsonl``
    per loop. Any other value is refused, never defaulted.

    ``ctx`` is required and has no default. Opening the index needs ``recon_hash``/``pack_hash``,
    computed by ``preprocess``, which ``pipeline`` may not import; the caller that already
    resolved those hashes hands the context in. ``clients``/``llm``/``layout`` do default, to the
    sanctioned singleton registry and the one path builder, so a production caller passes only
    ``ctx``.

    ``Layout.success()`` is written only after every round bank and every loop record is durable,
    and an already-marked run dir returns immediately. It is the topic's marker, not the cell's.
    """
    stats = stats if stats is not None else Statistics()
    topic_id = str(_field(topic, "topic_id") or _field(topic, "id") or "")
    layout = layout if layout is not None else topic_layout(cfg, variant, seed, topic_id)
    # Tested for existence rather than with `common.io.is_done`, which is the companion-marker
    # test and here would ask about `_SUCCESS._SUCCESS`.
    if layout.success().exists():
        _log.info("pipeline.drive.resume_noop", topic=topic_id, seed=seed, run_dir=str(layout.run_dir))
        return layout.run_dir

    # Stamped after the resume return and before any work, so a resumed no-op never rewrites a
    # start it did not perform and the span covers everything the cell does.
    started_at = time.time()

    clients = clients if clients is not None else _build_clients(cfg)
    llm = llm if llm is not None else getattr(clients, "llm", None)
    kind = str(getattr(cfg, "kind", "") or "")
    if kind != KIND_E2E:
        raise ValueError(
            f"run.kind={kind!r} is not a pipeline kind; expected {KIND_E2E!r}, the full "
            "decompose/ragloop coverage loop and the only kind this pipeline runs. "
            "config.loader validates the enum, so reaching this means a new kind was added "
            "there without a code path here."
        )
    result = asyncio.run(
        _drive_e2e(
            cfg, topic, topic_id, llm=llm, ctx=ctx, clients=clients,
            layout=layout, seed=seed, stats=stats, node=node,
        )
    )
    _log.info(
        "pipeline.drive.e2e",
        topic=topic_id, seed=seed, variant=variant, bank=len(result.bank),
        rounds=len(result.rounds), novelty=result.novelty,
        saturated_at=result.saturated_at, wall_s=round(result.wall_s, 1),
    )

    # Citation scoring runs after the loops, before the completion marker: the marker is the
    # resume witness, so a cell that reads as done must already carry its scores.
    # It supplies citation values, the assessment priority signal, not the citations themselves,
    # and never gates the loop, so a failure degrades the cell rather than discarding it.
    from ragtime.pipeline.citation_scoring.scorer import score_citations

    try:
        scores_path = asyncio.run(
            score_citations(cfg, layout=layout, llm=llm, seed=seed, stats=stats)
        )
        _log.info(
            "pipeline.drive.citation_scores",
            topic=topic_id, seed=seed, variant=variant, path=str(scores_path),
        )
    except Exception as exc:  # noqa: BLE001 - a ranking pass may not destroy a finished cell
        _log.error(
            "pipeline.drive.citation_scoring_failed",
            topic=topic_id, seed=seed, variant=variant, error=repr(exc),
            consequence="citations ship unranked (all 0.0); the cell is otherwise complete",
        )

    # Counters are in-memory and instance-scoped, so they are flushed before the completion marker
    # for the same reason as the scores: a resumed cell that reads as done must already have them.
    stats_path = flush_stats(stats, layout)
    _log.info(
        "pipeline.drive.metrics",
        topic=topic_id, seed=seed, variant=variant,
        counters=len(stats), path=str(stats_path),
    )
    _mark_complete(layout, topic_id=topic_id, seed=seed, kind=kind, started_at=started_at)
    return layout.run_dir


def topic_layout(cfg: Any, variant: str, seed: int, topic_id: str) -> Layout:
    """Return the ``Layout`` for one topic of one ``(run_id, variant, seed)`` cell.

    Resolved through ``orchestration``'s own owners, ``artifact_root`` and
    ``run_identity.cell_key``, never by string-building a path here. The topic level hangs under
    the cell because a cell is a topic shard: two topics of one shard must not share a
    ``rag_loop/`` directory, or the second would overwrite the first's records by nugget id.

    The family and chunker hash are required, not optional. A cell's own artifacts hang off
    ``run_dir``, but the corpus it reads hangs off the family-shared
    ``corpus/<family>/<chunker12>/.../final/<recon12>/passages/<pack12>``, and ``Layout`` refuses
    to resolve that subtree unbound. Guessing would address another family's corpus silently,
    since the three renderings share passage ids and every id would resolve.
    """
    from ragtime.config import all_hashes
    from ragtime.orchestration.cli import artifact_root
    from ragtime.orchestration.run_identity import cell_key, run_family

    root = Path(artifact_root(cfg))
    run_dir = root / cell_key(cfg.run_id, variant, seed) / "topics" / topic_id
    return Layout(
        run_dir=run_dir,
        outputs=getattr(cfg, "outputs", None),
        base=root,
        family=run_family(cfg),
        chunker_hash=all_hashes(cfg)["chunker"],
    )


async def _drive_e2e(
    cfg: Any,
    topic: Any,
    topic_id: str,
    *,
    llm: Any,
    ctx: Any,
    clients: Any,
    layout: Layout,
    seed: int,
    stats: Statistics,
    node: Any,
) -> CoverageLoopResult:
    """Run the e2e cell: the coverage loop, with passage text wired in from the run's context."""
    return await run_rounds(
        cfg,
        topic_id,
        str(_field(topic, "problem_statement") or ""),
        str(_field(topic, "background") or ""),
        int(_field(topic, "limit") or 0),
        llm=llm,
        ctx=ctx,
        clients=clients,
        layout=layout,
        seed=seed,
        ceiling=resolve_ceiling(cfg, node),
        stats=stats,
        passage_lang=getattr(cfg, "passage_lang", None),
        render=_renderer(ctx, cfg),
        title=str(_field(topic, "title") or ""),
    )


def _renderer(ctx: Any, cfg: Any):
    """Return a ``[passage_id] -> [(passage_id, text)]`` binding over the run's retrieval context.

    The audit's evidence-driven half needs passage text, and ``LoopResult`` carries only ids and
    scores. ``retrieval.display`` is the one by-id fetch and it reads Knob 2 (``passage_lang``),
    so the auditor reads exactly the rendering the loops read. It returns ``(doc_id, text)``, so
    passage ids are re-attached here from the request order ``display`` preserves.
    """
    from ragtime.retrieval import display

    passage_lang = getattr(cfg, "passage_lang", None)

    def render(passage_ids):
        ids = list(passage_ids)
        rendered = display(ctx, ids, passage_lang)
        return [(pid, text) for pid, (_doc_id, text) in zip(ids, rendered, strict=False)]

    return render


def _build_clients(cfg: Any) -> Any:
    """Build the shared client set. Imported lazily so small tests never pull the heavy stack."""
    from ragtime.serving import build_clients

    return build_clients(cfg)


def _mark_complete(
    layout: Layout, *, topic_id: str, seed: int, kind: str, started_at: float | None = None
) -> None:
    """Write ``<run>/_SUCCESS`` last, through ``common.io``. The topic's completion witness.

    Written after every round bank and loop record is durable, so a half-run topic never reads as
    complete to the resume logic. The write goes through ``common.io`` (temp -> fsync -> replace
    -> fsync(dir)) because a marker whose rename is durable while its directory entry is not can
    vanish, and a vanished marker silently re-runs a finished topic. Carrying the cell's identity
    as content costs nothing and makes an orphaned marker attributable.
    """
    completed_at = time.time()
    write_jsonl(
        layout.success(),
        [
            {
                "topic_id": topic_id,
                "seed": seed,
                "run_kind": kind,
                # `elapsed_s` is derived here rather than by the reader, so the subtraction that
                # defines a cell's duration has one home. Both are null when the caller supplied
                # no start, never 0.0, which would read as an instantaneous cell.
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_s": None if started_at is None else round(completed_at - started_at, 3),
            }
        ],
        skip_if_done=False,
    )
