"""The phase gate: which of the four actions is legal on this turn.

A pure function of the counters, with no model judgement, no I/O and no clock, so the half of the
loop that code owns stays decidable and testable without a GPU::

    menu <- { submit_claim, submit_answer }                     # gather core + close-when-ready
    if n_search < max_searches:                        menu += search
    if n_search >= min_searches and n_answer_fail >= 1: menu += abstain     # effort floor F

Gather and close are not two sessions, two contexts or two schemas: they only label which action
the model picked over the one growing conversation. ``submit_claim`` and ``submit_answer`` are in
the menu every turn, which is why there is no phase argument here, only counters.

The menu is enforced at the grammar level, by narrowing the compiled schema's ``action`` enum to
exactly these names, so the model cannot emit anything else and cannot emit nothing. It is
re-checked in code anyway, in :func:`assert_allowed`, because a structured-output request can be
ignored silently on this serving stack.
"""

from __future__ import annotations

from .state import Budget, LoopState

__all__ = [
    "ABSTAIN",
    "ACTIONS",
    "SEARCH",
    "SUBMIT_ANSWER",
    "SUBMIT_CLAIM",
    "assert_allowed",
    "forced_terminal_menu",
    "phase_menu",
]

SEARCH = "search"
SUBMIT_CLAIM = "submit_claim"
SUBMIT_ANSWER = "submit_answer"
ABSTAIN = "abstain"

#: Every action the compiled schema knows. The menu is always a subset of this; a name outside it
#: is a programming error rather than a model error.
ACTIONS = (SEARCH, SUBMIT_CLAIM, SUBMIT_ANSWER, ABSTAIN)


def phase_menu(state: LoopState, budget: Budget) -> tuple[str, ...]:
    """Return the actions legal this turn, in a stable order.

    Stable because the tuple is hashed to pick which precompiled schema to pass, and because a
    menu that reordered itself between turns would change the grammar, and therefore the KV
    prefix, for no reason.
    """
    menu = [SUBMIT_CLAIM, SUBMIT_ANSWER]
    if state.n_search < budget.max_searches:
        menu.append(SEARCH)
    if state.effort_floor(budget):
        menu.append(ABSTAIN)
    return tuple(a for a in ACTIONS if a in menu)


def forced_terminal_menu(has_committed_claims: bool) -> tuple[str, ...]:
    """Return the budget backstop's menu: exactly one action, so the loop cannot fail to terminate.

    ``submit_answer`` if any claim is committed, else ``abstain``. The effort floor is waived
    here, because further search is impossible by construction once a cap has been reached, and
    letting the model pick from a normal menu after the budget is spent is how a loop deadlocks.

    Note what this cannot do: manufacture a closed-book answer. With zero committed
    claims the only terminal is abstain, so an ungrounded guess is unreachable rather than merely
    discouraged.
    """
    return (SUBMIT_ANSWER,) if has_committed_claims else (ABSTAIN,)


def assert_allowed(action: str, menu: tuple[str, ...]) -> None:
    """Raise if the model emitted an action outside the menu it was given.

    A runtime assertion against a silent grammar bypass rather than defensive programming: on
    this stack a structured-output request can be ignored without raising, so "the enum made it
    impossible" is an assumption with a known counter-example. If this fires, the emission channel
    is broken and the run must stop: an out-of-menu ``abstain`` before the effort floor would
    convert an answerable nugget into an unanswered one, invisibly.
    """
    if action not in menu:
        raise PhaseGateViolation(
            f"model emitted {action!r}, which is not in the menu {menu!r} it was constrained to. "
            "The grammar was bypassed; this is not a model error and must not be retried."
        )


class PhaseGateViolation(RuntimeError):
    """The emitted action was outside the grammar-enforced menu: a serving-layer failure."""
