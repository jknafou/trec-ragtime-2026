"""Task 3: the top ``k_t3`` nuggets, with ``references`` as keys only.

After dedup, the bank is sorted descending by ``weight``, ties broken on the lower ``nugget_id``,
and the top ``k_t3`` are kept. Both answered and unanswered nuggets are kept: an unanswered
on-topic question is still valid Task-3 output, and Task 3 is scored on question coverage only.

Two things this module does not do:

* It does not rank the emitted bank. The nugget bank and the answers array are treated as sets by
  the evaluation, so their order does not matter.
* It does not tune ``answers``. The track does not score them; they are emitted because they are
  free and they hedge. Recorded so no later work optimizes an unscored field.

``aggregator_type`` is the one field a typo gets the whole run rejected on, since the schema's
enum admits only ``AND`` and ``OR``. It is therefore asserted here, before anything is written,
rather than discovered by the validator after a file exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ragtime.common import Nugget, T3Answer, T3Nugget, T3NuggetBankRow, doc_id_of

if TYPE_CHECKING:
    from ragtime.common import Statistics, Topic

    from .knobs import SerializeKnobs

__all__ = [
    "AGGREGATOR_TYPES",
    "T3_ANSWERS_DROPPED",
    "T3_DROPPED_TAIL_MEAN_WEIGHT",
    "T3_NUGGETS_DROPPED",
    "T3_NUGGETS_SUBMITTED",
    "BadAggregatorType",
    "apply_caps",
    "task3_bank",
]

#: The validator's enum, verbatim.
AGGREGATOR_TYPES = ("AND", "OR")

T3_NUGGETS_SUBMITTED = "serialize.t3_nuggets_submitted"
T3_NUGGETS_DROPPED = "serialize.t3_nuggets_dropped_by_cap"
T3_DROPPED_TAIL_MEAN_WEIGHT = "serialize.t3_dropped_tail_mean_weight"
T3_ANSWERS_DROPPED = "serialize.t3_answers_dropped_by_cap"


class BadAggregatorType(ValueError):
    """``aggregator_type`` is not exactly ``AND`` or ``OR``, so the run would be rejected."""


def apply_caps(
    bank: Sequence[Nugget],
    knobs: SerializeKnobs,
    *,
    stats: Statistics | None = None,
    variant: str | None = None,
    seed: int | None = None,
) -> tuple[Nugget, ...]:
    """Keep the top ``k_t3`` nuggets by weight, then the top ``answers_cap`` answers by score.

    Shared by Tasks 1 and 3, since the caps are applied once before both projections and only
    Task 2 works from the uncapped bank. A bank smaller than ``k_t3`` is not truncated, because
    ``k`` is a cap rather than a target. Every cut is by an owned score with a deterministic tie,
    so the stage stays a pure reproducible function.
    """
    ranked = sorted(bank, key=lambda n: (-n.weight, n.nugget_id))
    kept, dropped = ranked[: knobs.k_t3], ranked[knobs.k_t3 :]
    answers_dropped = 0
    capped: list[Nugget] = []
    for nugget in kept:
        answers = sorted(nugget.answers, key=lambda a: -a.score)[: knobs.answers_cap]
        answers_dropped += len(nugget.answers) - len(answers)
        capped.append(replace(nugget, answers=tuple(answers)))

    if stats is not None:
        slices = _slices(variant, seed)
        stats.emit(T3_NUGGETS_DROPPED, float(len(dropped)), **slices)
        if dropped:
            stats.emit(
                T3_DROPPED_TAIL_MEAN_WEIGHT,
                sum(n.weight for n in dropped) / len(dropped),
                **slices,
            )
        stats.emit(T3_ANSWERS_DROPPED, float(answers_dropped), **slices)
    return tuple(capped)


def task3_bank(
    kept: Sequence[Nugget],
    topic: Topic,
    knobs: SerializeKnobs,
    *,
    team_id: str,
    run_id: str,
    run_desc: str,
    stats: Statistics | None = None,
    variant: str | None = None,
    seed: int | None = None,
) -> T3NuggetBankRow:
    """Project the already-capped bank from :func:`apply_caps` into the Task-3 row."""
    nuggets: list[T3Nugget] = []
    for nugget in kept:
        if nugget.aggregator_type not in AGGREGATOR_TYPES:
            raise BadAggregatorType(
                f"nugget {nugget.nugget_id!r} has aggregator_type "
                f"{nugget.aggregator_type!r}; the validator's enum is {list(AGGREGATOR_TYPES)} "
                "and a mismatch gets the whole run rejected"
            )
        nuggets.append(
            T3Nugget(
                question=nugget.question,
                aggregator_type=nugget.aggregator_type,
                answers=tuple(
                    T3Answer(
                        answer=a.answer,
                        # Keys only: the Task-3 schema carries no numeric score, so the citation
                        # scores are dropped here and survive only in Task 1's `citations`.
                        # Sorted for byte-determinism, which is our choice since the track treats
                        # them as a set. `doc_id_of` is applied again on the way out so a passage
                        # id cannot reach a submission even if one ever entered `references`.
                        references=tuple(sorted({doc_id_of(d) for d in a.references})),
                    )
                    for a in nugget.answers
                ),
            )
        )
    if stats is not None:
        stats.emit(T3_NUGGETS_SUBMITTED, float(len(nuggets)), **_slices(variant, seed))
    return T3NuggetBankRow(
        topic_id=str(topic.topic_id),
        team_id=team_id,
        run_id=run_id,
        run_desc=run_desc,
        nugget_bank=tuple(nuggets),
    )


def _slices(variant: str | None, seed: int | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if variant is not None:
        out["variant"] = variant
    if seed is not None:
        out["seed"] = seed
    return out
