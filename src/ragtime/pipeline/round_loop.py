"""The round, and the loop of rounds: decompose's bank meets the k RAG loops until saturation.

This module is the join. ``decompose/`` produces a bank of question-nuggets and ``rag_loop/``
answers one nugget. A round is that call, bounded: :func:`run_round`. :func:`run_rounds` is the
loop of rounds: round 0 seeds the bank alone, then each round fans the open nuggets, audits the
answers, grows the bank, and stops on measured novelty.

The two halves are separate functions rather than one with a flag. ``run_round`` is the fan and
knows nothing about banks, audits or stopping; ``run_rounds`` is the coverage loop and issues no
generation of its own. That split lets the expensive property (does the fan actually fan?) be
measured without a bank, and the cheap properties (does saturation fire, does a raising loop
cancel its siblings) be tested without a model.

Rounds are sequential; loops within a round are concurrent. Rounds share the growing bank, so
round r+1's nuggets depend on round r's answers. Only the loops inside one round are independent,
and they fan as asyncio coroutines under one semaphore against the one shared vLLM instance. That
sequencing also keeps the fairness anchor intact without an extra rule: round-0 decompose finishes
before the first loop starts, so it never shares its instance with concurrent work, which matters
because sampled decoding is not batch-invariant.

``orchestration.fanout.map`` calls ``asyncio.gather`` without ``return_exceptions=True``, so a
single raising coroutine would cancel every sibling in flight. The loop already converts the two
failures it knows about into terminal ``LoopResult``s, and :func:`_guarded` converts anything else
into a terminal ``LoopResult(status="error")``, so a round survives one bad nugget and the failure
becomes a record rather than a traceback. Widening ``fanout.map`` itself would change a mechanism
three other call sites share, to fix a problem that belongs to this caller.

``ceiling`` is the vLLM KV headroom and governs how many loops may be thinking. It does not govern
how many may be searching: the retrieval service is a fixed number of replicas, so past its width
the loops queue against each other rather than against the GPU. Both run under the single loop
ceiling (at most k loops in flight implies at most k searches in flight), and what a second
ceiling would need to be is measured through ``RoundResult.search_wall_s`` rather than guessed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ragtime.common import Nugget, Statistics, get_logger
from ragtime.common.io import is_done
from ragtime.orchestration import fanout
from ragtime.pipeline.decompose import bank as bank_ops
from ragtime.pipeline.decompose import coverage_audit, grow_nuggets, saturated
from ragtime.pipeline.rag_loop import run_loop
from ragtime.pipeline.records import write_round_records

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ragtime.common import Layout

__all__ = [
    "COVERAGE_INCOMPLETE",
    "NOVELTY",
    "OPEN_STATUSES",
    "SATURATED_AT",
    "CoverageLoopResult",
    "RoundResult",
    "open_nuggets",
    "run_round",
    "run_rounds",
]

_log = get_logger("pipeline.round_loop")

#: This cell read a round-0 bank another arm had already computed. Together with
#: :data:`STAT_SEED_BANK_COMPUTED` this says whether sharing is working: `computed` climbing
#: towards the total cell count means the key is partitioning by something it should not.
STAT_SEED_BANK_REUSED = "decompose.seed_bank_reused"
#: This cell computed the round-0 bank, being the first arm to reach it.
STAT_SEED_BANK_COMPUTED = "decompose.seed_bank_computed"

#: The per-round novelty curve: net-new nuggets after dedup, the saturation evidence. Round 1 is
#: the first observation, since round 0 mints the whole bank and is not a novelty measurement.
NOVELTY = "decompose.novelty"
#: The round where the low-novelty streak fired. Absent means saturation never fired.
SATURATED_AT = "decompose.saturated_at"
#: ``R_max`` was hit instead of saturation, the coverage-incomplete flag. Emitted so the novelty
#: curve shows it; the bank at ``R_max`` is still emitted, never silently truncated.
COVERAGE_INCOMPLETE = "decompose.coverage_incomplete"

#: A loop that raised. Distinct from the loop's own terminals (``timeout``,
#: ``malformed_action``), which are the loop choosing to stop rather than breaking.
CLOSED_ERROR = "error"


@dataclass(frozen=True)
class RoundResult:
    """One round's outcome: every loop's record, plus what the round cost.

    ``wall_s`` and ``search_wall_s`` denominate the downstream planning questions (how many GPU
    pairs, how many seeds are affordable, what the retrieval admission ceiling should be), so a
    round reports its own cost rather than leaving the next estimate to a guess.
    """

    round_no: int
    results: tuple[Any, ...]
    wall_s: float
    #: Summed wall time of the loops, as if they had run one after another. Divided by ``wall_s``
    #: this is the achieved concurrency, distinct from the ceiling, which is only what was
    #: permitted.
    sequential_wall_s: float
    search_wall_s: float = 0.0
    errors: tuple[str, ...] = ()

    @property
    def speedup(self) -> float:
        """Achieved concurrency: sequential cost / actual cost. 1.0 means the fan bought nothing."""
        return (self.sequential_wall_s / self.wall_s) if self.wall_s > 0 else 0.0


#: The statuses that mean "this nugget still has to be answered".
#:
#: ``bank.STATUS_UNANSWERED`` is the value a real ``common.Nugget`` carries; ``"open"`` and
#: ``None`` are the shapes a plain mapping uses. The bank's own vocabulary is read rather than a
#: literal, and the mapping forms are kept so a test double stays cheap.
OPEN_STATUSES = frozenset({None, "open", bank_ops.STATUS_UNANSWERED})


def open_nuggets(bank: Sequence[Any]) -> tuple[Any, ...]:
    """Return the nuggets this round must answer: the open ones, in bank order.

    Bank order is load-bearing for reproducibility: ``fanout.map`` is order-preserving, so the
    results line up with the bank without a re-sort, and two runs at one seed fan the same
    nuggets in the same order. An answered or pruned nugget is never re-fanned, which is what
    makes the round loop converge: the audit is the only thing that moves a nugget out of the
    open set.
    """
    out = []
    for n in bank:
        status = getattr(n, "status", None)
        if status is None and isinstance(n, dict):
            status = n.get("status")
        if status in OPEN_STATUSES:
            out.append(n)
    return tuple(out)


def _as_mapping(nugget: Any) -> dict[str, Any]:
    """Reduce a nugget to the two fields ``run_loop`` needs: ``nugget_id`` and ``question``.

    This is the request-blindness boundary: the loop's behaviour is a function of its nugget
    question alone. The topic ``title``, ``problem_statement``, ``background``, ``limit`` and the
    nugget's ``weight``/``vital`` all stop here and reach decompose only.
    """
    if isinstance(nugget, dict):
        return {"nugget_id": nugget.get("nugget_id"), "question": nugget.get("question")}
    return {
        "nugget_id": getattr(nugget, "nugget_id", None),
        "question": getattr(nugget, "question", ""),
    }


async def run_round(
    bank: Sequence[Any],
    cfg: Any,
    *,
    llm: Any,
    ctx: Any,
    ceiling: int,
    round_no: int = 0,
    seed: int = 0,
    stats: Statistics | None = None,
    passage_lang: str | None = None,
    only: Sequence[str] | None = None,
) -> RoundResult:
    """Fan this round's open nuggets to ``run_loop``, at most ``ceiling`` in flight.

    Returns a :class:`RoundResult` whose ``results`` line up one-to-one with the fanned nuggets:
    a loop that raised appears as a terminal record, never as a hole, so a caller can zip the two
    without checking for gaps.

    ``only`` restricts the fan to those nugget ids, for the sweep round: the coverage audit runs
    after a round's loops, so a nugget it mints on the last round would otherwise never be fanned.
    Default ``None`` fans every open nugget, which is every ordinary round.
    """
    targets = open_nuggets(bank)
    if only is not None:
        wanted = set(only)
        targets = tuple(n for n in targets if _as_mapping(n).get("nugget_id") in wanted)
    if not targets:
        return RoundResult(round_no=round_no, results=(), wall_s=0.0, sequential_wall_s=0.0)

    stats = stats if stats is not None else Statistics()
    per_loop_wall: list[float] = []
    errors: list[str] = []

    async def _guarded(nugget: Any) -> Any:
        """Run one loop and time it, turning any escaping exception into a record."""
        started = time.perf_counter()
        try:
            return await run_loop(
                _as_mapping(nugget),
                cfg,
                llm=llm,
                ctx=ctx,
                seed=seed,
                stats=stats,
                passage_lang=passage_lang,
            )
        except Exception as exc:  # noqa: BLE001 - a round must survive one bad loop
            nid = _as_mapping(nugget)["nugget_id"]
            errors.append(f"{nid}: {type(exc).__name__}: {exc}")
            stats.emit("ragloop.loop_raised", 1, nugget=str(nid))
            return _error_result(nugget, exc)
        finally:
            per_loop_wall.append(time.perf_counter() - started)

    started = time.perf_counter()
    results = await fanout.map(_guarded, targets, ceiling=max(1, int(ceiling)))
    wall = time.perf_counter() - started

    return RoundResult(
        round_no=round_no,
        results=tuple(results),
        wall_s=wall,
        sequential_wall_s=sum(per_loop_wall),
        search_wall_s=_search_wall(results),
        errors=tuple(errors),
    )


def _error_result(nugget: Any, exc: BaseException) -> Any:
    """Build a terminal ``LoopResult`` for a loop that raised. Imported lazily to avoid a cycle."""
    from ragtime.pipeline.rag_loop.loop import LoopResult

    m = _as_mapping(nugget)
    return LoopResult(
        nugget_id=str(m["nugget_id"]),
        question=str(m["question"] or ""),
        status="error",
        closed_by=CLOSED_ERROR,
        rationale_trail=[f"{type(exc).__name__}: {exc}"],
    )


@dataclass(frozen=True)
class CoverageLoopResult:
    """What the whole coverage loop produced for one topic, and how it ended.

    ``novelty`` is the per-round net-new-nuggets curve in round order, round 1 first, and is the
    series :func:`~ragtime.pipeline.decompose.saturation.saturated` consumes. ``saturated_at`` is
    the round the low-novelty streak fired, or ``None`` when ``R_max`` was hit instead, which is
    the coverage-incomplete flag: the bank is still emitted, and the fact that it may be
    incomplete is a value rather than a silence.
    """

    topic_id: str
    seed: int
    bank: tuple[Nugget, ...]
    rounds: tuple[RoundResult, ...]
    novelty: tuple[int, ...]
    saturated_at: int | None
    decompose_s: float
    loops_s: float
    audit_s: float

    @property
    def coverage_incomplete(self) -> bool:
        """True when the loop stopped on ``R_max`` rather than on measured novelty."""
        return self.saturated_at is None

    @property
    def wall_s(self) -> float:
        return self.decompose_s + self.loops_s + self.audit_s


async def run_rounds(
    cfg: Any,
    topic_id: str,
    problem_statement: str,
    background: str,
    limit: int,
    *,
    llm: Any,
    ctx: Any,
    clients: Any,
    layout: Layout,
    seed: int = 0,
    ceiling: int = 1,
    stats: Statistics | None = None,
    passage_lang: str | None = None,
    render: Callable[[Sequence[str]], Sequence[tuple[str, str]]] | None = None,
    title: str = "",
) -> CoverageLoopResult:
    """Run the dynamic coverage loop for one topic: round 0, then audit rounds until saturation.

    The algorithm::

        bank <- grow_nuggets(..., [], None, 0)              # round 0, alone on the instance
        for r in 1..R_max:
            passages <- run_rag_loops(unanswered(bank))     # the fan, k concurrent
            bank     <- grow_nuggets(..., bank, passages, r)
            net_new  <- |{n in bank : n.origin_round == r}| # net-new after dedup
            if saturated(novelty): break

    ``title`` is the topic's short request label and goes to ``grow_nuggets`` at round 0 and at
    every audit round, and to nothing else. It is absent from :func:`run_round`'s
    signature, which fans the request-blind loops.

    The stop condition is read, never re-derived: ``R_max``/``min_new``/``low_streak`` come from
    the fairness-shared ``decomposition`` block and the predicate is
    :func:`~ragtime.pipeline.decompose.saturation.saturated`. A round whose open set is empty
    still counts as a novelty-0 observation rather than breaking immediately, because
    ``low_streak`` is the anti-premature floor and skipping it would end the loop one round early.

    Every round writes ``Layout.decompose_round(r)`` only after that round's loops and audit are
    complete, so a crash mid-round leaves round r-1 as the last complete round and the resume
    re-runs r. A resumed round is read back rather than recomputed, and the novelty curve is
    recovered from the round banks themselves.
    """
    stats = stats if stats is not None else Statistics()
    dcfg = cfg.blocks["decomposition"]
    r_max = int(dcfg["R_max"])
    min_new = int(dcfg["min_new"])
    low_streak = int(dcfg["low_streak"])

    started = time.perf_counter()
    bank = await _round_zero(
        cfg,
        topic_id,
        problem_statement,
        background,
        limit,
        clients=clients,
        layout=layout,
        seed=seed,
        stats=stats,
        title=title,
    )
    decompose_s = time.perf_counter() - started

    rounds: list[RoundResult] = []
    novelty: list[int] = []
    saturated_at: int | None = None
    loops_s = 0.0
    audit_s = 0.0

    for r in range(1, r_max + 1):
        resumed = _resume_round(layout, r)
        if resumed is not None:
            bank = resumed
            novelty.append(_net_new(bank, r))
        else:
            started = time.perf_counter()
            round_result = await run_round(
                bank, cfg, llm=llm, ctx=ctx, ceiling=ceiling, round_no=r,
                seed=seed, stats=stats, passage_lang=passage_lang,
            )
            loops_s += time.perf_counter() - started
            rounds.append(round_result)
            write_round_records(layout, round_result, cfg, topic_id=topic_id, seed=seed)

            started = time.perf_counter()
            evidence = coverage_audit.evidence_from_results(round_result.results, render=render)
            bank = await grow_nuggets(
                problem_statement, background, bank, evidence, r,
                cfg=cfg, clients=clients, topic_id=topic_id, limit=limit,
                seed=seed, stats=stats, title=title,
            )
            audit_s += time.perf_counter() - started
            novelty.append(_net_new(bank, r))
            bank_ops.write_bank(layout.decompose_round(r), bank)

        stats.emit(NOVELTY, float(novelty[-1]), round=r, seed=seed)
        _log.info(
            "pipeline.round_loop.round",
            topic=topic_id, seed=seed, round=r, bank=len(bank),
            novelty=novelty[-1], resumed=resumed is not None,
        )
        if saturated(novelty, min_new=min_new, low_streak=low_streak):
            saturated_at = r
            stats.emit(SATURATED_AT, float(r), seed=seed)
            break

    # The sweep round. Within a round the order is fan, then audit (which adds nuggets), then the
    # saturation break, so a nugget the audit mints on the final round would never be fanned: it
    # would reach Task 3 as a question nothing ever tried to answer. The sweep fans only the
    # never-attempted ids (typically one or two) and does not audit afterwards, since another
    # audit would mint more nuggets and the loop would not terminate.
    #
    # "Attempted" is read off the artifact tree rather than off `rounds`, because a resumed round
    # restores the bank without appending to `rounds`; an in-memory count would report every
    # nugget of a resumed cell as never fanned and sweep the whole bank.
    unattempted = [
        nid
        for n in open_nuggets(bank)
        if (nid := str(_as_mapping(n).get("nugget_id") or ""))
        and not layout.rag_loop(nid).exists()
    ]
    if unattempted:
        _log.info(
            "pipeline.round_loop.sweep",
            topic=topic_id, seed=seed, nuggets=tuple(unattempted),
            detail="nuggets the final audit added after the last fan, swept so no nugget "
                   "reaches Task 3 unattempted",
        )
        started = time.perf_counter()
        sweep = await run_round(
            bank, cfg, llm=llm, ctx=ctx, ceiling=ceiling, round_no=len(rounds) + 1,
            seed=seed, stats=stats, passage_lang=passage_lang, only=unattempted,
        )
        loops_s += time.perf_counter() - started
        rounds.append(sweep)
        write_round_records(layout, sweep, cfg, topic_id=topic_id, seed=seed)
        # Attach the evidence without auditing: `grow_nuggets` is attach + judge + apply_delta,
        # and its gap-detect half is what mints new nuggets. `attach_evidence` is the attach half.
        # The status comes from the loop's own verdict, not from a second judge.
        bank = coverage_audit.attach_evidence(
            bank, coverage_audit.evidence_from_results(sweep.results, render=render)
        )
        answered_now = {
            str(res.nugget_id)
            for res in sweep.results
            if getattr(res, "status", None) == bank_ops.STATUS_ANSWERED
        }
        if answered_now:
            # `bank_ops.close` is the existing mark-answered primitive and also leaves a pruned
            # nugget alone.
            bank = bank_ops.close(bank, answered_now)
        bank_ops.write_bank(layout.decompose_round(len(rounds) - 1), bank)

    if saturated_at is None:
        # `R_max` hit without saturation: emit the bank and say so, because a coverage-incomplete
        # bank and a saturated one are different claims about the topic.
        stats.emit(COVERAGE_INCOMPLETE, 1.0, seed=seed)
        _log.warning(
            "pipeline.round_loop.coverage_incomplete",
            topic=topic_id, seed=seed, r_max=r_max, novelty=tuple(novelty),
            detail="R_max was reached before the novelty floor fired; the bank may be incomplete",
        )
    return CoverageLoopResult(
        topic_id=topic_id,
        seed=seed,
        bank=bank,
        rounds=tuple(rounds),
        novelty=tuple(novelty),
        saturated_at=saturated_at,
        decompose_s=decompose_s,
        loops_s=loops_s,
        audit_s=audit_s,
    )


async def _round_zero(
    cfg: Any,
    topic_id: str,
    problem_statement: str,
    background: str,
    limit: int,
    *,
    clients: Any,
    layout: Layout,
    seed: int,
    stats: Statistics,
    title: str = "",
) -> tuple[Nugget, ...]:
    """Return the seed bank: read back if round 0 already completed, else decomposed and written.

    Resuming round 0 from its artifact is not merely a saving. Re-decomposing would issue a fresh
    sampled generation, and at ``temperature=0.7`` the second bank need not equal the first even
    at one seed once the instance's batching differs. The persisted round-0 bank is the fairness
    anchor's subject, so a resume reads it rather than re-deriving it.

    The seed bank is shared across the family. Round 0 is retrieval-free by construction and reads
    only the request plus three fairness-shared blocks (``decomposition``, ``llm``, ``topics``),
    so neither translation knob can reach it and two arms cannot legitimately hold different
    round-0 banks for one ``(topic, seed)``. Computing it once and reading it from every arm makes
    that invariant true by construction, and collapses the family's decompositions from one per
    arm to one per cell.
    """
    path = layout.decompose_round(0)
    if is_done(path):
        return bank_ops.read_bank(path)

    shared = layout.seed_bank(bank_ops.seed_bank_hash(cfg), topic_id, seed)
    if is_done(shared):
        bank = bank_ops.read_bank(shared)
        bank_ops.write_bank(path, bank)  # the per-cell copy every later stage reads
        stats.emit(STAT_SEED_BANK_REUSED, 1, nugget=topic_id)
        _log.info("round_loop.seed_bank.reused", topic=topic_id, seed=seed, path=str(shared))
        return bank

    bank = await grow_nuggets(
        problem_statement, background, (), None, 0,
        cfg=cfg, clients=clients, topic_id=topic_id, limit=limit, seed=seed, stats=stats,
        title=title,
    )
    # First writer wins, and the returned bank is the one we keep: a concurrent arm may have
    # published between the `is_done` check and here, and the canonical artifact must not depend
    # on who finished last. `publish_seed_bank` returns the winner, so both arms proceed on the
    # same nuggets.
    bank = bank_ops.publish_seed_bank(shared, bank)
    stats.emit(STAT_SEED_BANK_COMPUTED, 1, nugget=topic_id)
    bank_ops.write_bank(path, bank)
    return bank


def _net_new(bank: Sequence[Nugget], r: int) -> int:
    """Return round ``r``'s novelty: nuggets minted at round ``r`` that survived dedup.

    Not ``len(bank) - len(bank_before)``, which measures ``minted - merged_away``.
    Under the size-delta reading, a round that mints three gaps while dedup merges three
    pre-existing nuggets scores novelty 0; twice in a row and saturation fires with six freshly
    minted, never-fanned facets in the bank. The delta can also go negative when a round merges
    more than it mints, which reads as more saturated than saturated.

    Counting surviving ``origin_round == r`` nuggets is non-negative by construction, and the
    resume branch computes it the same way from the persisted bank alone, so a resumed curve is
    identical to an uninterrupted one.
    """
    return sum(1 for n in bank if n.origin_round == r)


def _resume_round(layout: Layout, r: int) -> tuple[Nugget, ...] | None:
    """Return round ``r``'s persisted post-audit bank, or ``None`` if that round never completed.

    Keyed on the ``_SUCCESS`` marker rather than the file's existence: an atomic temp-then-rename
    leaves no half file, but a round whose bank was written while its loop records were not is
    exactly the state the marker distinguishes, since ``write_bank`` writes the marker last.
    """
    path = layout.decompose_round(r)
    return bank_ops.read_bank(path) if is_done(path) else None


def _search_wall(results: Sequence[Any]) -> float:
    """Return the total time the round spent inside retrieval, from each loop's search trail.

    This is the input to sizing a separate retrieval admission ceiling. If it approaches
    ``wall_s`` the loops are queueing on the index rather than the GPU and a second semaphore is
    worth building; if it is a small fraction, one ceiling is enough.
    """
    total = 0.0
    for r in results:
        for hop in getattr(r, "search_trail", ()) or ():
            if isinstance(hop, dict):
                for key in ("client_wall_s", "wall_s", "service_wall_s"):
                    if isinstance(hop.get(key), (int, float)):
                        total += float(hop[key])
                        break
    return total
