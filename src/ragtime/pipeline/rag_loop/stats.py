"""The counter ids this package emits onto ``common.stats``. Counters only, never a metric.

Declared here once, in the same shape as ``retrieval.stats``, so the monitoring rollup can read
the whole loop vocabulary in one file. Slice keys are the canonical ones; this package slices by
``nugget`` (which loop) and ``variant`` (the read rendering), never by anything else.

The set answers the questions the loop's design asks of it, namely outcome, agency, effort floor,
conflict, closing and cost, and nothing more.
"""

from __future__ import annotations

__all__ = [
    "STAT_ABSTAIN_EVER_LEGAL",
    "STAT_ANSWER_FAILED",
    "STAT_CLAIMS_COMMITTED",
    "STAT_CLAIMS_REJECTED",
    "STAT_CLAIMS_SUBMITTED",
    "STAT_CLAIM_RETRIES",
    "STAT_CLAIM_VALUE_FALLBACK",
    "STAT_CLOSED_BY_BUDGET",
    "STAT_CLOSED_BY_MODEL",
    "STAT_CLOSE_WARNED",
    "STAT_CONTESTED",
    "STAT_GENERATION_TIMEOUT",
    "STAT_LOOPS",
    "STAT_LOOP_SECONDS",
    "STAT_MALFORMED_ACTION",
    "STAT_OUTCOME_ABSTAIN",
    "STAT_OUTCOME_ACCEPT",
    "STAT_SEARCHES",
    "STAT_TURNS",
]

#: One per ``run_loop`` call.
STAT_LOOPS = "rag_loop.loops"
#: Wall seconds per loop (summed; the bus is the only sink a stage has).
STAT_LOOP_SECONDS = "rag_loop.loop_seconds"
#: Generation calls actually issued (turns). 1..``max_iters``.
STAT_TURNS = "rag_loop.turns"
#: ``search`` actions executed.
STAT_SEARCHES = "rag_loop.searches"

#: Claims the model emitted, whether or not they grounded.
STAT_CLAIMS_SUBMITTED = "rag_loop.claims_submitted"
#: Claims whose span matched verbatim and were committed.
STAT_CLAIMS_COMMITTED = "rag_loop.claims_committed"
#: Same-turn re-decodes spent on a non-verbatim span, the retries-per-commit statistic. Bounded
#: by `claim_commit.max_retries`.
STAT_CLAIM_RETRIES = "rag_loop.claim_retries"

#: Committed claims where the model omitted `answer` or `sentence` and the claim text stood in.
#: Both are optional in the flat, grammar-friendly union, where only `rationale` and `action` are
#: required, so a schema-valid turn can omit them and `Answer.answer` then holds a whole sentence
#: rather than a short value. Counted rather than prevented, because tightening `required` per
#: action would mean the per-action field surgery this union avoids; a high rate here is the
#: signal to revisit that trade.
STAT_CLAIM_VALUE_FALLBACK = "rag_loop.claim_value_fallback"

#: Claims rejected for a non-verbatim span, the grounding rate's denominator partner. A rejection
#: is never a repair.
STAT_CLAIMS_REJECTED = "rag_loop.claims_rejected"

#: ``submit_answer`` attempts that carried no committed claims (``n_answer_fail``).
STAT_ANSWER_FAILED = "rag_loop.answer_failed"
#: Loops in which the effort floor F ever held, meaning ``abstain`` entered the menu.
STAT_ABSTAIN_EVER_LEGAL = "rag_loop.abstain_ever_legal"

#: Terminal outcomes, mutually exclusive.
STAT_OUTCOME_ACCEPT = "rag_loop.outcome_accept"
STAT_OUTCOME_ABSTAIN = "rag_loop.outcome_abstain"

#: How the loop ended: the model chose a terminal, or the budget backstop forced one.
STAT_CLOSED_BY_MODEL = "rag_loop.closed_by_model"
STAT_CLOSED_BY_BUDGET = "rag_loop.closed_by_budget"

#: Loops told they were running out of turns while holding at least one committed claim. Paired
#: with `closed_by_model` this is the observable for whether the warning works: warned then
#: closed-by-model is the intended path, while warned and still closed-by-budget means the model
#: ignored it and the turns were spent anyway.
STAT_CLOSE_WARNED = "rag_loop.close_warned"

#: Loops ended by a generation exceeding its wall-clock bound. A serving fault: if this is
#: non-zero at scale, `llm.call_timeout_s` or the model is what to look at.
STAT_GENERATION_TIMEOUT = "rag_loop.generation_timeout"

#: Loops ended by a schema-valid but unusable action. A model fault, kept distinct from a serving
#: timeout so one cannot hide inside the other.
STAT_MALFORMED_ACTION = "rag_loop.malformed_action"

#: Accepts carrying two or more distinct answer values: corpus disagreement surfaced, not dropped.
STAT_CONTESTED = "rag_loop.contested"
