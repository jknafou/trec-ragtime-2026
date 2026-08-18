"""Execute one action the model chose: the half of a turn that is not a generation call.

Each function here takes the emitted action object and the loop's state, does the one thing that
action means, and returns the text to append to the growing context. None of them decides what to
do next: the model does that, and the phase gate decides what it may choose from.

The ``search`` branch is where the retrieval stack and the three display renderings meet the
agent. Both translation knobs apply here and neither is this module's choice: ``ctx`` was built
against the run's ``retrieval.index`` (what is searched) and ``passage_lang`` (what is read) comes
from the same config. The loop never learns which index it is on.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragtime.retrieval.tool import format_passages, search_action

from .commit import Rejection, commit_span
from .state import Budget, LoopState, Trail

__all__ = ["do_search", "do_submit_claim"]


def _native_lang(ctx: Any, passage_id: str) -> str:
    """Return the passage's native source language, via the route ``retrieval.display`` documents.

    Best-effort: a service-backed context with no local passage store cannot answer, and the
    honest value is then ``""`` rather than a guess. The field feeds the per-language support
    analysis, which can resolve it from the passage id itself, and inventing a language here would
    corrupt that analysis silently.
    """
    store = getattr(ctx, "passage_store", None)
    if store is None:
        return ""
    try:
        return str(store.passage(passage_id).lang)
    except (KeyError, AttributeError):
        return ""


def do_search(
    ctx: Any,
    action: Mapping[str, Any],
    state: LoopState,
    budget: Budget,
    trail: Trail,
    *,
    passage_lang: str | None = None,
    char_budget: int | None = None,
) -> str:
    """Run the model's own query through the shared fuse-then-rerank stack and return what it reads.

    Everything retrieval-shaped is delegated to ``retrieval.tool.search_action``: this function
    adds no ranking, no fusion and no second display path. What it owns is the trail, meaning the
    union of hits across searches, which is the Task-2 artifact, and the passage text the model
    has been shown, which is the only thing a later ``submit_claim`` may quote against.
    """
    result = search_action(
        ctx,
        action,
        top_k=budget.search_top_k,
        char_budget=char_budget,
        passage_lang=passage_lang,
    )
    state.n_search += 1

    for pid, score in result.hits:
        # Keep the best score seen for a passage across searches: `retrieved[]` is a union, so a
        # passage found twice is one entry, at its strongest evidence.
        prev = trail.scores.get(pid)
        if prev is None or score > prev:
            trail.scores[pid] = score
    for pid, _doc_id, text in result.passages:
        trail.passages[pid] = text
        if pid not in trail.native_lang:
            trail.native_lang[pid] = _native_lang(ctx, pid)

    trail.searches.append(
        {
            "turn": state.n_iters,
            "queries": list(result.queries),
            "returned": list(result.passage_ids),
        }
    )
    return format_passages(result)


def do_submit_claim(
    action: Mapping[str, Any],
    trail: Trail,
) -> tuple[Any, str]:
    """Ground one claim. Returns ``(CommittedClaim | Rejection, feedback_for_the_model)``.

    The feedback matters as much as the verdict: a rejected claim appends a specific reason to the
    same conversation, so the model's next turn is better informed than its last. That is what
    makes a bounded retry productive rather than a re-roll of identical state.
    """
    passage_id = str(action.get("passage_id") or "")
    outcome = commit_span(
        claim=str(action.get("claim") or ""),
        span=str(action.get("span") or ""),
        passage_id=passage_id,
        shown_text=trail.passages.get(passage_id),
        shown_ids=tuple(trail.passages),
        native_lang=trail.native_lang.get(passage_id, ""),
        answer=str(action.get("answer") or ""),
        sentence=str(action.get("sentence") or ""),
    )
    if isinstance(outcome, Rejection):
        return outcome, outcome.feedback()
    return outcome, (
        f"claim COMMITTED against doc_id={outcome.doc_id} "
        f"(passage {outcome.passage_id}). Submit another claim, search again, or answer."
    )
