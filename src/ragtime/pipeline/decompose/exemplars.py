"""Frozen, versioned few-shot exemplar sets for the decompose prompts.

The design pins two to four few-shot exemplars at the target granularity, in the AutoNuggetizer
house style: atomic, single-fact, single-sentence questions. The config carries
``decomposition.few_shot = {n, exemplar_set}``, where ``exemplar_set`` is a versioned id whose text
is a frozen constant here.

A v2 is a new constant; it never edits v1 in place. The ``decomposition`` block's hash is the
prompt hash and is shared byte-identically across a run family, so an in-place exemplar edit would
change the round-0 seed anchor of every already-recorded run without moving a single config byte.
Re-keying happens by naming a new set in the config.

This module is a data table, not a mechanism: no state, no IO, no model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

__all__ = [
    "EXEMPLAR_SETS",
    "MAX_EXEMPLARS",
    "MIN_EXEMPLARS",
    "Exemplar",
    "exemplar_set_fingerprint",
    "select_exemplars",
]

#: The pinned few-shot band. ``select_exemplars`` refuses an ``n`` outside it rather than
#: clamping: a config asking for seven exemplars is a config error, and silently serving four
#: would make the run record untrue.
MIN_EXEMPLARS = 2
MAX_EXEMPLARS = 4


@dataclass(frozen=True, slots=True)
class Exemplar:
    """One few-shot demonstration: a request, and the nuggets it should decompose to.

    ``nuggets`` are the house-style target: atomic, single-fact, single-sentence English
    questions. They demonstrate granularity, count and format.
    """

    problem_statement: str
    background: str
    nuggets: tuple[str, ...]


# autonuggetizer_v1 is frozen: add a v2 constant rather than editing it.
# Four exemplars so the whole 2-4 band is servable. The requests are synthetic rather than sliced
# from the released topics, so no topic's own facet structure is handed to the model.
_AUTONUGGETIZER_V1: tuple[Exemplar, ...] = (
    Exemplar(
        problem_statement=(
            "Report on the 2023 recall of a blood-pressure medication: which products "
            "were recalled, why, and what regulators told patients to do."
        ),
        background=(
            "You are briefing a hospital pharmacy committee that must decide whether to "
            "pull stock from its shelves this week."
        ),
        nuggets=(
            "Which blood-pressure medication products were recalled in 2023?",
            "Which manufacturer produced the recalled products?",
            "What contaminant triggered the recall?",
            "Which regulatory agency issued the recall notice?",
            "What did regulators instruct patients already taking the medication to do?",
            "How many lots were affected by the recall?",
        ),
    ),
    Exemplar(
        problem_statement=(
            "Report on a cross-border river dispute: what each state claims, what "
            "agreements exist, and what has changed most recently."
        ),
        background=(
            "You are preparing a background note for a diplomat with no prior "
            "knowledge of the basin."
        ),
        nuggets=(
            "Which states are parties to the river dispute?",
            "What does each state claim about its water entitlement?",
            "Which treaty currently governs the river?",
            "In what year was that agreement signed?",
            "What infrastructure project triggered the most recent escalation?",
            "What dispute-resolution mechanism has been invoked?",
        ),
    ),
    Exemplar(
        problem_statement=(
            "Report on an industrial fire at a chemical plant: the cause, the human and "
            "environmental impact, and the official response."
        ),
        background=(
            "You are writing for an environmental-health newsletter read by local "
            "residents."
        ),
        nuggets=(
            "Where did the chemical plant fire occur?",
            "On what date did the fire start?",
            "What was identified as the cause of the fire?",
            "How many people were killed by the fire?",
            "Which chemicals were released by the fire?",
            "What evacuation order was issued to residents?",
        ),
    ),
    Exemplar(
        problem_statement=(
            "Report on a national election result: the outcome, the turnout, the "
            "disputes raised, and the transition that followed."
        ),
        background=(
            "You are briefing an analyst who tracks political stability indicators."
        ),
        nuggets=(
            "Which party won the election?",
            "What share of the vote did the winner receive?",
            "What was the voter turnout?",
            "Who disputed the election result?",
            "On what grounds was the result disputed?",
            "When did the new government take office?",
        ),
    ),
)

#: Name -> frozen exemplar tuple. A new recipe is a new key, never an edit.
EXEMPLAR_SETS: dict[str, tuple[Exemplar, ...]] = {
    "autonuggetizer_v1": _AUTONUGGETIZER_V1,
}


def exemplar_set_fingerprint(name: str) -> str:
    """Return the sha256 of the canonical JSON of exemplar set ``name``, the freeze witness.

    Byte-stable (sorted keys, no ASCII escaping) so a test can pin the digest of a shipped set and
    go red the moment the set is edited in place instead of versioned.
    """
    exemplars = _lookup(name)
    payload = json.dumps(
        [asdict(e) for e in exemplars],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_exemplars(name: str, n: int) -> tuple[Exemplar, ...]:
    """Return the first ``n`` exemplars of set ``name``, per ``decomposition.few_shot``.

    The slice is a deterministic, order-stable prefix: the same ``(name, n)`` always yields the
    same demonstrations, which is half of why round 0 is reproducible. Raises ``KeyError`` for an
    unknown set and ``ValueError`` for an ``n`` outside the 2-4 band or beyond the set's size,
    never a silent clamp.
    """
    exemplars = _lookup(name)
    upper = min(MAX_EXEMPLARS, len(exemplars))
    if not MIN_EXEMPLARS <= n <= upper:
        raise ValueError(
            f"decomposition.few_shot.n={n} is outside the pinned {MIN_EXEMPLARS}-{upper} "
            f"band for exemplar_set {name!r}; a silent clamp would make the config record untrue"
        )
    return exemplars[:n]


def _lookup(name: str) -> tuple[Exemplar, ...]:
    try:
        return EXEMPLAR_SETS[name]
    except KeyError:
        raise KeyError(
            f"unknown decomposition.few_shot.exemplar_set {name!r}; "
            f"known sets: {sorted(EXEMPLAR_SETS)}"
        ) from None
