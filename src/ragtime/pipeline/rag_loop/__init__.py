"""The request-blind agentic RAG loop: one gather-then-close conversation per open nugget.

Public surface, in the order a reader meets it:

* :func:`run_loop` answers one nugget from the corpus. ``round_loop`` calls it once per open
  nugget, k at a time, under ``orchestration.fanout``.
* :func:`phase_menu` and :func:`forced_terminal_menu` say which actions are legal this turn. Both
  are pure functions of the counters, so the gate is testable without a model.
* :func:`commit_span` is the verbatim NFC grounding check.
* :func:`fan_in` clusters committed claims into ``answers[]``, surfacing disagreement.

Everything expensive is injected: the shared vLLM singleton (``llm``) and the run's
``RetrievalContext`` (``ctx``). This package instantiates no model and opens no index.
"""

from __future__ import annotations

from .commit import CommittedClaim, Rejection, commit_span
from .fanin import cluster_key, fan_in, is_contested
from .loop import ANSWERED, UNANSWERED, LoopResult, run_loop
from .phase_gate import (
    ABSTAIN,
    ACTIONS,
    SEARCH,
    SUBMIT_ANSWER,
    SUBMIT_CLAIM,
    PhaseGateViolation,
    assert_allowed,
    forced_terminal_menu,
    phase_menu,
)
from .state import Budget, LoopState, Trail, budget_from_config

__all__ = [
    "ABSTAIN",
    "ACTIONS",
    "ANSWERED",
    "SEARCH",
    "SUBMIT_ANSWER",
    "SUBMIT_CLAIM",
    "UNANSWERED",
    "Budget",
    "CommittedClaim",
    "LoopResult",
    "LoopState",
    "PhaseGateViolation",
    "Rejection",
    "Trail",
    "assert_allowed",
    "budget_from_config",
    "cluster_key",
    "commit_span",
    "fan_in",
    "forced_terminal_menu",
    "is_contested",
    "phase_menu",
    "run_loop",
]
