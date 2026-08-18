"""The loop's counters and budget: the whole of its control state, in one place.

Termination is guaranteed by these counters alone. ``n_iters``, ``n_search`` and ``tokens`` only
ever increase, and the budget backstop fires on the next turn once any cap is hit. Keeping them in
one object rather than as locals is what lets :func:`phase_gate.phase_menu` be a pure function of
the state, which is in turn what makes the phase gate testable without a model.

``LoopState`` is mutable, since the loop advances it in place, while ``Budget`` is frozen: the
caps come from the fairness-shared config and must not be adjustable mid-loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Budget", "InertKnobError", "LoopState", "Trail", "budget_from_config"]


@dataclass(frozen=True, slots=True)
class Budget:
    """The four caps and the effort floor, from ``cfg.blocks['rag_loop']['budget']``.

    No default is invented here: ``budget_from_config`` reads the config, and every config
    declares all five, since ``rag_loop`` is a shared block and a config lacking it does not load.
    The dataclass defaults exist only so a unit test can build one without a config, and they
    match the shipped values.
    """

    min_searches: int = 2
    max_iters: int = 8
    max_searches: int = 5
    token_budget: int = 20000
    search_top_k: int = 20
    #: Fraction of ``max_iters`` at which the loop is told it is running out of turns, so it can
    #: close on its own terms instead of being handed a one-action menu at the wall. ``None``
    #: disables the warning.
    #:
    #: It exists because `submit_answer` is offered on every turn but is rarely elected early:
    #: a loop otherwise runs its full turn allowance and submits once the backstop has forced a
    #: single-action menu. The model does not stop because nothing tells it that stopping is due.
    #: `forced_terminal_menu` already narrows the grammar to one action; this leaf supplies the
    #: warning that precedes it.
    warn_at_frac: float | None = None


class InertKnobError(ValueError):
    """A declared, hashed config knob whose value this code does not implement."""


#: Knobs the config declares that this code implements by construction rather than by branching:
#: `emission: action`, one grammar-constrained `{rationale, action}` per turn and never native
#: tool-calls, and `abstain: gated`, the effort floor. Because nothing reads them, a config could
#: set `emission: tool_calls`, hash it into the fairness-shared record and change nothing, so the
#: values are checked rather than ignored: the run record must not claim a behaviour the code does
#: not have.
_PINNED_KNOBS: dict[str, str] = {"emission": "action", "abstain": "gated"}


def budget_from_config(cfg: Any) -> Budget:
    """Read the shipped ``rag_loop`` block. Nothing here is hardcoded."""
    block: Mapping[str, Any] = dict(getattr(cfg, "blocks", {}).get("rag_loop", {}) or {})
    for knob, only in _PINNED_KNOBS.items():
        got = block.get(knob)
        if got is not None and str(got) != only:
            raise InertKnobError(
                f"rag_loop.{knob}={got!r} is declared and hashed into the fairness-shared record, "
                f"but this code implements only {only!r} and would silently ignore it. Either "
                f"set it to {only!r} or implement the alternative, rather than leaving the run "
                "record claiming a behaviour the run does not have."
            )
    caps: Mapping[str, Any] = dict(block.get("budget", {}) or {})
    return Budget(
        min_searches=int(caps.get("min_searches", 2)),
        max_iters=int(caps.get("max_iters", 8)),
        max_searches=int(caps.get("max_searches", 5)),
        token_budget=int(caps.get("token_budget", 20000)),
        warn_at_frac=(None if caps.get("warn_at_frac") is None else float(caps["warn_at_frac"])),
        search_top_k=int(block.get("search_top_k", 20)),
    )


@dataclass(slots=True)
class LoopState:
    """Monotonic counters for one loop. Every field only ever increases."""

    n_iters: int = 0
    n_search: int = 0
    n_answer_fail: int = 0
    tokens: int = 0
    #: Latched the first turn the effort floor F held, for the "fraction of loops where abstain
    #: ever became legal" statistic. Latched rather than recomputed at the end because F can only
    #: hold while zero claims are committed, so a late commit would erase the evidence.
    abstain_ever_legal: bool = False

    def effort_floor(self, budget: Budget) -> bool:
        """F := (n_search >= min_searches) AND (n_answer_fail >= 1).

        Because a ``submit_answer`` with at least one committed claim terminates,
        ``n_answer_fail >= 1`` is reachable only when no claim has been committed, which is what
        reserves ``abstain`` for genuine evidence absence rather than a free way out.
        """
        return self.n_search >= budget.min_searches and self.n_answer_fail >= 1

    def budget_exhausted(self, budget: Budget) -> bool:
        """Return whether any backstop cap has been reached.

        Checked before each turn, so it can force a terminal rather than merely stopping: the
        loop must never end without an outcome.
        """
        return (
            self.n_iters >= budget.max_iters
            or self.n_search >= budget.max_searches
            or self.tokens >= budget.token_budget
        )


@dataclass(slots=True)
class Trail:
    """What the loop accumulated: the Task-2 artifact plus what the model is allowed to cite.

    ``passages`` is the only thing ``commit_span`` checks a quoted span against, and it holds
    exactly the text the model was shown, in the run's ``passage_lang``. A claim citing a passage
    the model never saw therefore cannot ground, with no separate "did you see this?" check, and
    a claim quoting the original rendering when the model read a translation cannot ground either,
    which is correct for a verbatim commit against what was actually read.
    """

    #: passage_id -> the exact text shown, in the read rendering.
    passages: dict[str, str] = field(default_factory=dict)
    #: passage_id -> native source language, for Support and the per-language support cut.
    native_lang: dict[str, str] = field(default_factory=dict)
    #: Best score seen per passage across all searches: the Task-2 ``retrieved[]``.
    scores: dict[str, float] = field(default_factory=dict)
    #: One entry per search action: {turn, queries, returned}.
    searches: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    rationales: list[str] = field(default_factory=list)
