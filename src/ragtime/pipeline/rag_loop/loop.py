"""``run_loop``: one agentic gather-then-close conversation answering one nugget.

Each turn is a single grammar-constrained generation on the shared vLLM; the model reads what it
has gathered, reasons, and picks its own next action. Code owns exactly three things, the phase
gate, the verbatim span check and fan-in, and decides nothing else.

One session, one growing context, no reset. ``messages`` is created once, from the system prompt
and the nugget question, and is only ever appended to: search results, the model's own actions,
and feedback. A failed answer attempt does not reset anything; it appends the failure plus a
reason to the same conversation, so the next turn runs with more information than the last. That
is what makes progress likely, and the budget is what makes termination certain.

Termination is guaranteed twice over, and both halves are necessary. The counters ``n_iters``,
``n_search`` and ``tokens`` only increase, so the backstop fires and forces a single terminal; and
each individual generation is bounded in wall-clock by ``llm.call_timeout_s``, without which the
first half is vacuous, because a turn that never returns cannot reach the next backstop check.

What this function never does: instantiate a model, since every call is the injected ``llm``
singleton; open an index, since every search goes through ``retrieval.tool`` via ``ctx``; learn
which index it is searching or which rendering it is reading, both of which are the config's
business; name a doc-id itself, since that is derived from the passage id; or manufacture an
answer with no committed claim, which is unreachable because the forced-terminal menu offers only
abstain when nothing is committed.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragtime.common import Statistics, get_logger
from ragtime.common.schemas import Answer, Retrieved
from ragtime.serving.llm import GenerationTimeout, GuidedJsonError, decoding_kwargs
from ragtime.serving.schemas import action_schema_for

from . import stats as S
from .actions import do_search, do_submit_claim
from .commit import CommittedClaim, Rejection
from .fanin import fan_in, is_contested
from .phase_gate import (
    ABSTAIN,
    SEARCH,
    SUBMIT_ANSWER,
    SUBMIT_CLAIM,
    assert_allowed,
    forced_terminal_menu,
    phase_menu,
)
from .prompts import SYSTEM_PROMPT, question_prompt
from .state import LoopState, Trail, budget_from_config

__all__ = ["LoopResult", "run_loop"]

_log = get_logger("pipeline.rag_loop")

#: Terminal statuses, matching the loop's output contract.
ANSWERED = "answered"
UNANSWERED = "unanswered"

#: How the loop ended, for the "closed itself against was forced" statistic.
CLOSED_MODEL = "model"
CLOSED_BUDGET = "budget"
#: A turn overran `llm.call_timeout_s`. Recorded distinctly from `budget` because it is a serving
#: fault rather than the loop spending its allowance; conflating them would hide a
#: degenerate-generation regression inside a normal-looking backstop rate.
CLOSED_TIMEOUT = "timeout"
#: A schema-valid but unusable action, such as `search` with no query. Distinct from a timeout
#: because it is a model fault rather than a serving one.
CLOSED_MALFORMED = "malformed_action"

#: reason -> the ``closed_by`` breakdown counter it increments, or ``None`` for already counted
#: elsewhere. Exhaustive over the ``CLOSED_*`` constants above, and pinned as such by a test.
#: A ternary over four reasons would record every serving timeout and every malformed action as
#: "the model closed itself", inflating the one rate that should mean the loop decided it was
#: done. The two faults map to ``None`` deliberately: counting them a second time under another
#: name would be a worse error.
CLOSED_STAT: dict[str, str | None] = {
    CLOSED_MODEL: S.STAT_CLOSED_BY_MODEL,
    CLOSED_BUDGET: S.STAT_CLOSED_BY_BUDGET,
    CLOSED_TIMEOUT: None,
    CLOSED_MALFORMED: None,
}

#: Rough characters-per-token, used only to advance the ``token_budget`` backstop. It is an
#: approximation and is never reported as a measurement: `guided_json` returns the validated
#: object rather than the usage block, and plumbing usage through for a backstop that `max_iters`
#: already bounds would be a worse trade. Under-counting cannot hang the loop, since ``max_iters``
#: and ``max_searches`` still bind, and over-counting only ends it earlier.
CHARS_PER_TOKEN = 4


@dataclass(slots=True)
class LoopResult:
    """The full loop record, richer than any single task keeps.

    ``retrieved`` and ``search_trail`` are the Task-2 provenance and are retained on abstain: a
    nugget the corpus could not answer still produced real evidence about what is in it.
    """

    nugget_id: str
    question: str
    status: str
    answers: tuple[Answer, ...] = ()
    retrieved: tuple[Retrieved, ...] = ()
    search_trail: list[dict[str, Any]] = field(default_factory=list)
    action_trail: list[str] = field(default_factory=list)
    rationale_trail: list[str] = field(default_factory=list)
    contested: bool = False
    closed_by: str = CLOSED_MODEL
    turns: int = 0
    searches: int = 0
    claims_committed: int = 0
    claims_rejected: int = 0
    #: reason -> count for every dropped claim. The aggregate `claims_rejected` cannot
    #: distinguish a grammar hole (`empty_span`) from the commit gate working as designed
    #: (`not_verbatim`), and those want opposite responses. Persisted on the record so the
    #: histogram survives the process rather than having to be reconstructed from the model's
    #: own prose.
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    claim_retries: int = 0
    answer_failures: int = 0
    #: Which exception ended the loop and where, as ``"<ExcType> at <file>:<line>: <message>"``,
    #: empty when the loop ended normally.
    #:
    #: ``closed_by: malformed_action`` is the non-timeout arm of a handler catching three unrelated
    #: failure classes, which read identically on the record without this field. The file and line
    #: are the load-bearing half; the message is truncated because a message is not a log.
    closed_detail: str = ""

    @property
    def answered(self) -> bool:
        return self.status == ANSWERED


#: How much of an exception message survives onto the loop record. The full text and the
#: traceback go to `rag_loop.loop_aborted` at error level; this is the part that must still be
#: readable when hundreds of records are being counted at once.
MAX_FAULT_CHARS = 300


def _fault_detail(exc: BaseException) -> str:
    """Return ``"<ExcType> at <file>:<line>: <message>"``: which exception ended the loop, and where.

    The site is the load-bearing half. `run_loop`'s handler catches ``(GenerationTimeout,
    GuidedJsonError, ValueError)`` around the whole turn body, so a `ValueError` there can come
    from the search tool, from a parse, or from any bookkeeping statement in the body. The
    exception type cannot tell those apart; the file and line can.

    Takes the last frame of the traceback, which is the raise site rather than the catch site, and
    degrades to the type alone rather than raising: a diagnostic that can fail is worse than no
    diagnostic, and this one runs inside a handler whose job is to keep `run_loop`'s guarantee of
    always returning a ``LoopResult``.
    """
    where = ""
    try:
        frames = traceback.extract_tb(exc.__traceback__)
        if frames:
            last = frames[-1]
            where = f" at {Path(last.filename).name}:{last.lineno}"
    except Exception:  # noqa: BLE001  # pragma: no cover - a diagnostic must never raise
        where = ""
    return f"{type(exc).__name__}{where}: {str(exc)[:MAX_FAULT_CHARS]}"


def _retrieved(trail: Trail) -> tuple[Retrieved, ...]:
    """Return the Task-2 artifact: the union of every search, best score first, ties by id.

    Sorted deterministically so two runs over the same trail serialize identically. Float scores
    tie often enough at this scale that relying on dict order would be a real reproducibility hole
    rather than a theoretical one.
    """
    items = sorted(trail.scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(Retrieved(passage_id=pid, score=score) for pid, score in items)


async def run_loop(
    nugget: Mapping[str, Any],
    cfg: Any,
    *,
    llm: Any,
    ctx: Any,
    seed: int = 0,
    stats: Statistics | None = None,
    passage_lang: str | None = None,
    char_budget: int | None = None,
) -> LoopResult:
    """Answer one nugget from the corpus. See the module docstring for the guarantees.

    ``nugget`` needs only ``nugget_id`` and ``question``: the loop is request-blind, and taking a
    Mapping rather than a ``common.Nugget`` keeps that visible, since there is no field here it
    could accidentally read. ``llm`` and ``ctx`` are injected, being the shared vLLM singleton and
    the run's ``RetrievalContext``, and are never built here.
    """
    budget = budget_from_config(cfg)
    # Config-driven like every other knob: `claim_commit` is fairness-shared, so a family cannot
    # differ on how many re-decodes a claim gets.
    max_retries = int(
        (getattr(cfg, "blocks", {}).get("claim_commit", {}) or {}).get("max_retries", 2)
    )
    # The run record declares `llm.decoding` and provenance writes those numbers down, so they are
    # applied explicitly here; without this every loop would run at `GenCtx`'s zero-temperature
    # default and the record would not describe the run.
    gen_kw: dict[str, Any] = {"system": SYSTEM_PROMPT}
    gen_kw.update(decoding_kwargs((getattr(cfg, "blocks", {}).get("llm", {}) or {}).get("decoding")))
    # The read rendering, resolved once. Every claim counter is sliced by it and by the cited
    # passage's native language, because the central language-fairness diagnostic is the verbatim
    # commit rate by language, and a counter carrying only `nugget=` cannot be rolled up into it.
    read_variant = str(passage_lang or getattr(cfg, "passage_lang", "") or "")
    state = LoopState()
    trail = Trail()
    committed: list[CommittedClaim] = []
    n_rejected = 0
    warned = False
    rejection_reasons: dict[str, int] = {}
    n_retries = 0
    stats = stats if stats is not None else Statistics()
    nugget_id = str(nugget.get("nugget_id") or "")
    question = str(nugget.get("question") or "")
    started = time.perf_counter()

    messages: list[str] = [question_prompt(question)]
    closed_by = CLOSED_MODEL
    closed_detail = ""
    status = UNANSWERED

    # One wall-clock guard for the whole turn rather than one per `generate()` call site: the
    # bounded claim re-decode below is a second `llm.generate()`, so a per-call-site guard would
    # leave it uncovered and depend on every future call site re-wrapping itself.
    #
    # It ends in a terminal rather than a raise, because `run_loop` returning a LoopResult
    # unconditionally is the contract the k-way fan-out depends on: one degenerate nugget must not
    # cancel every sibling in the same `asyncio.gather`.
    try:
        while True:
            # Budget backstop: force a single terminal, never deadlock.
            forced = state.budget_exhausted(budget)
            menu = forced_terminal_menu(bool(committed)) if forced else phase_menu(state, budget)

            # The closing warning, sent once. `phase_menu` offers `submit_answer` on every turn, so
            # the model is always free to stop, but without a signal it runs its full allowance and
            # stops only once the backstop has narrowed the menu to that one action. This adds no
            # terminal and no action, only a message, which is the difference between
            # `closed_by: model` and `closed_by: budget`. It fires only when something is
            # committed; warning an empty loop would push it to abstain and waste its remaining
            # searches.
            if (
                not forced
                and not warned
                and committed
                and budget.warn_at_frac is not None
                and state.n_iters >= max(1, int(budget.max_iters * budget.warn_at_frac))
            ):
                warned = True
                stats.emit(S.STAT_CLOSE_WARNED, 1.0, nugget=nugget_id)
                messages.append(
                    f"NOTE: you have used {state.n_iters} of {budget.max_iters} turns and have "
                    f"{len(committed)} committed claim(s). If those already answer the question, "
                    "submit_answer NOW — the claims you have committed ARE the answer, and "
                    "nothing further is needed from you. Only keep searching if a specific, "
                    "identifiable gap remains."
                )
            if not forced and state.effort_floor(budget):
                state.abstain_ever_legal = True

            state.n_iters += 1
            prompt = "\n\n".join(messages)
            # The session size, not the running sum of every turn's prompt: `+=` would make the
            # counter quadratic in turns, so a session-level token ceiling would trip after two or
            # three turns instead of bounding the full allowance. It stays monotone either way,
            # since the context only grows, so termination is not at risk.
            state.tokens = max(1, len(prompt) // CHARS_PER_TOKEN)

            obj = await llm.generate(
                action_schema_for(menu),
                prompt,
                seed,
                **gen_kw,
            )
            action = str(obj.get("action") or "")
            rationale = str(obj.get("rationale") or "")
            # Stream the turn: the trails are written only when the whole round closes, so
            # without this a fan of a dozen loops is opaque for its entire duration and "working
            # slowly" is indistinguishable from "stuck". One line per generation, carrying what
            # the model decided and why, truncated because the full text still lands in
            # `rationale_trail`. `menu` is included because it is the grammar the turn ran under,
            # which makes an unexpected action interpretable.
            _log.info(
                "rag_loop.turn",
                nugget=nugget_id, turn=state.n_iters + 1, action=action,
                menu="|".join(menu), rationale=rationale,
                span=str(obj.get("span") or "")[:120],
                query=str(obj.get("query") or "")[:120],
            )
            assert_allowed(action, menu)
            trail.actions.append(action)
            trail.rationales.append(rationale)
            # Append the whole action, not just its type: appending only `{"action": "search"}`
            # leaves the model unable to see which query it already ran or which value it already
            # committed, which defeats the multi-hop contract and makes fan-in's exact-string
            # clustering a coin flip. `rationale` is excluded, being telemetry that
            # would double the context for no decision the model has not already made.
            shown = {k: v for k, v in obj.items() if k != "rationale"}
            messages.append(
                "YOUR ACTION: " + json.dumps(shown, ensure_ascii=False, sort_keys=True)
            )

            if action == SEARCH:
                rendered = do_search(
                    ctx, obj, state, budget, trail,
                    passage_lang=passage_lang, char_budget=char_budget,
                )
                messages.append(rendered)
                # Log what came back, not just that a search happened: `rendered` is the exact
                # text appended to the conversation, so without it the stream shows a query and
                # then, several turns later, a span, with no way to see what the model was
                # reacting to. Truncated to keep the stream readable; the full text is
                # reconstructible from `search_trail`, which carries the passage ids.
                _log.info(
                    "rag_loop.search_result",
                    nugget=nugget_id, turn=state.n_iters,
                    hits=len(trail.scores), preview=str(rendered)[:900],
                )

            elif action == SUBMIT_CLAIM:
                # Bounded same-turn re-decode: a span-formatting slip is not a failure of search
                # effort, so it must not consume a turn of `max_iters`, or two near-misses would
                # cost a quarter of the loop's budget and push an answerable nugget toward the
                # backstop. Termination is unaffected and still bounded by the counters alone:
                # this inner loop runs at most `claim_commit.max_retries` times, so generations
                # per loop stay under `max_iters * (1 + max_retries)`, each wall-clock bounded.
                claim_lang = trail.native_lang.get(str(obj.get("passage_id") or ""), "")
                _cs = {"nugget": nugget_id, "lang": claim_lang, "variant": read_variant}
                stats.emit(S.STAT_CLAIMS_SUBMITTED, 1.0, **_cs)
                outcome, feedback = do_submit_claim(obj, trail)
                retries = 0
                while isinstance(outcome, Rejection) and retries < max_retries:
                    retries += 1
                    n_retries += 1
                    stats.emit(S.STAT_CLAIM_RETRIES, 1.0, **_cs)
                    messages.append(feedback)
                    obj = await llm.generate(
                        action_schema_for((SUBMIT_CLAIM,)),
                        "\n\n".join(messages),
                        seed,
                        **gen_kw,
                    )
                    assert_allowed(str(obj.get("action") or ""), (SUBMIT_CLAIM,))
                    # The retry is streamed too: it can account for a large share of a round's
                    # GPU time and is otherwise invisible.
                    _log.info(
                        "rag_loop.claim_retry", nugget=nugget_id, attempt=n_retries,
                        reason=getattr(outcome, "reason", "?"),
                        span=str(obj.get("span") or "")[:120],
                        passage_id=str(obj.get("passage_id") or "")[:60],
                    )
                    outcome, feedback = do_submit_claim(obj, trail)
                if isinstance(outcome, CommittedClaim):
                    committed.append(outcome)
                    stats.emit(
                        S.STAT_CLAIMS_COMMITTED, 1.0, **{**_cs, "lang": outcome.lang or claim_lang}
                    )
                    if outcome.value_fallback:
                        stats.emit(S.STAT_CLAIM_VALUE_FALLBACK, 1.0, nugget=nugget_id)
                else:
                    # Dropped after bounded retry, never committed ungrounded.
                    n_rejected += 1
                    # The reason is the slice. Without it the counter says only that claims were
                    # dropped, and the reasons want opposite fixes: an `empty_span` is a grammar
                    # hole, an `unseen_passage` is the model citing a doc-id instead of a passage
                    # handle, and a `not_verbatim` is the commit gate doing its job.
                    reason = getattr(outcome, "reason", "unknown")
                    stats.emit(S.STAT_CLAIMS_REJECTED, 1.0, **{**_cs, "reason": reason})
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                messages.append(feedback)

            elif action == SUBMIT_ANSWER:
                if committed:
                    # `submit_answer` carries no value: the short answer and the report sentence
                    # were emitted with each claim, and there is no second generation for them.
                    status = ANSWERED
                    closed_by = CLOSED_BUDGET if forced else CLOSED_MODEL
                    break
                # No committed claims: a failed attempt, not a reset and not an answer.
                state.n_answer_fail += 1
                stats.emit(S.STAT_ANSWER_FAILED, 1.0, nugget=nugget_id)
                messages.append(
                    "answer NOT accepted: you have committed no claims yet, so there is nothing to "
                    "ground it in. Keep searching and commit a claim with an exact span."
                )

            elif action == ABSTAIN:
                status = UNANSWERED
                closed_by = CLOSED_BUDGET if forced else CLOSED_MODEL
                break
    except (GenerationTimeout, GuidedJsonError, ValueError) as exc:
        # All three arms are caught for the same reason: the action union requires only `rationale`
        # and `action`, so a search with no query is schema-valid and raises `ValueError` downstream,
        # and an exhausted schema retry raises `GuidedJsonError`. Either escaping `run_loop` would
        # cancel every sibling loop in the round's `asyncio.gather`.
        #
        # Timeouts and malformed actions are counted separately, since a timeout is a serving fault
        # and a malformed action is a model fault. The loop still returns everything it committed
        # before the failure.
        if isinstance(exc, GenerationTimeout):
            stats.emit(S.STAT_GENERATION_TIMEOUT, 1.0, nugget=nugget_id)
            closed_by = CLOSED_TIMEOUT
        else:
            stats.emit(S.STAT_MALFORMED_ACTION, 1.0, nugget=nugget_id)
            closed_by = CLOSED_MALFORMED
        # Record which exception and from where: this handler merges three unrelated failure
        # classes into one `closed_by` value. Logged at error level and persisted on the record,
        # because the two survive different things: the log is the only place with the full
        # message and dies with the node's log file, while the record is the only thing a later
        # analysis can read at scale.
        closed_detail = _fault_detail(exc)
        _log.error(
            "rag_loop.loop_aborted",
            nugget=nugget_id, closed_by=closed_by, turns=state.n_iters,
            committed=len(committed), rejected=n_rejected, detail=closed_detail,
        )
        status = ANSWERED if committed else UNANSWERED

    answers = fan_in(committed) if committed else ()
    contested = is_contested(answers)

    stats.emit(S.STAT_LOOPS, 1.0, nugget=nugget_id, variant=read_variant)
    stats.emit(S.STAT_LOOP_SECONDS, time.perf_counter() - started, nugget=nugget_id)
    stats.emit(S.STAT_TURNS, float(state.n_iters), nugget=nugget_id)
    stats.emit(S.STAT_SEARCHES, float(state.n_search), nugget=nugget_id)
    if state.abstain_ever_legal:
        stats.emit(S.STAT_ABSTAIN_EVER_LEGAL, 1.0, nugget=nugget_id)
    stats.emit(
        S.STAT_OUTCOME_ACCEPT if status == ANSWERED else S.STAT_OUTCOME_ABSTAIN,
        1.0,
        nugget=nugget_id,
        variant=read_variant,
    )
    # An exhaustive table rather than a ternary over four reasons, so a loop ended by a serving
    # timeout or a malformed action is not recorded as "the model closed itself". An unmapped
    # reason emits nothing rather than falling back, so a new terminal reason shows up as a
    # missing breakdown rather than a quietly wrong one, and exhaustiveness is checked by a test
    # over the module's own CLOSED_* constants rather than by a runtime warning at the end of an
    # otherwise successful loop.
    stat = CLOSED_STAT.get(closed_by)
    if stat is not None:
        stats.emit(stat, 1.0, nugget=nugget_id)
    if contested:
        stats.emit(S.STAT_CONTESTED, 1.0, nugget=nugget_id)

    return LoopResult(
        nugget_id=nugget_id,
        question=question,
        status=status,
        answers=answers,
        retrieved=_retrieved(trail),
        search_trail=trail.searches,
        action_trail=trail.actions,
        rationale_trail=trail.rationales,
        contested=contested,
        closed_by=closed_by,
        closed_detail=closed_detail,
        turns=state.n_iters,
        searches=state.n_search,
        claims_committed=len(committed),
        claims_rejected=n_rejected,
        rejection_reasons=dict(rejection_reasons),
        claim_retries=n_retries,
        answer_failures=state.n_answer_fail,
    )
