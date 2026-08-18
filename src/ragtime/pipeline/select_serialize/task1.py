"""Task 1: the coverage-first budgeted greedy with an in-loop grounding gate.

Task 1 is scored on nugget recall, where a nugget is covered once any admitted sentence supports
it, so two sentences on the same nugget add almost no recall while both spend the character
budget. A pure ``weight x score`` greedy is blind to that, hence a coverage-first budgeted greedy,
a specialization of budgeted submodular-coverage maximization, rather than plain importance order:

* Tier 1 takes one sentence per distinct answered nugget, nuggets by descending ``weight`` with
  ties broken on the lower ``nugget_id``, and each nugget's best answer by ``score`` with ties
  broken on the lower answer index.
* Tier 2 fills the residual budget by descending ``weight x score``; a redundant same-nugget
  sentence may enter only here, after coverage is saturated.

Both tiers share one admission gate, and the order inside it is load-bearing: grounding is checked
before the sentence spends budget, so a rejected sentence frees its budget for the next well-cited
candidate. The emitted order is the admission order, which is the report order.

Everything emitted is a verbatim copy: the text is ``answer.sentence`` and the citations are
``answer.references``. The substantive-against-framing distinction is resolved by construction,
since the pipeline generates no free-form narrative and every sentence here is an extracted,
span-grounded report sentence. The single exception is the fallback's neutral framing template,
reached only when nothing at all clears the gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ragtime.common import Nugget, T1ReportRow, T1Response, nfc

from .budget import fits, joined_len, trim_to_limit

if TYPE_CHECKING:
    from ragtime.common import Answer, Statistics, Topic

    from .knobs import SerializeKnobs

__all__ = [
    "FALLBACK_SENTENCE",
    "T1_BUDGET_UTILISATION",
    "T1_FALLBACK",
    "T1_NUGGETS_COVERED",
    "T1_SENTENCES_ADMITTED",
    "T1_SENTENCES_DROPPED_BUDGET",
    "T1_SENTENCES_DROPPED_DUPLICATE",
    "T1_SENTENCES_DROPPED_GROUNDING",
    "task1_report",
]

STATUS_ANSWERED = "answered"

#: Counters only; `monitoring` owns every rollup.
T1_SENTENCES_ADMITTED = "serialize.t1_sentences_admitted"
T1_SENTENCES_DROPPED_GROUNDING = "serialize.t1_sentences_dropped_grounding"
T1_SENTENCES_DROPPED_BUDGET = "serialize.t1_sentences_dropped_budget"
#: Repeats collapsed into an earlier identical sentence, with their citations merged into it.
#: Counted separately from `dropped_budget` because the two are opposites: a budget drop means the
#: report is full, a duplicate drop means it is redundant, and reading the second as the first
#: would hide a dedup regression behind a healthy-looking budget.
T1_SENTENCES_DROPPED_DUPLICATE = "serialize.t1_sentences_dropped_duplicate"
T1_BUDGET_UTILISATION = "serialize.t1_budget_utilisation"
T1_FALLBACK = "serialize.t1_fallback_topics"
T1_NUGGETS_COVERED = "serialize.t1_nuggets_covered"

#: The last-resort framing sentence, emitted with empty citations only when not even the single
#: best candidate is admissible. A no-citation-needed sentence scores as neutral, whereas an empty
#: ``responses[]`` scores zero, which is what this avoids. It is a closed-class template rather
#: than generated text.
FALLBACK_SENTENCE = "No sufficiently supported answer was found in the collection for this request."


def _grounding(answer: Answer) -> float:
    """Return the answer's top citation priority: ``max(references.values())``, 0.0 when uncited.

    Ordering only; it is not a gate. ``references`` is a ranking signal, so nothing is discarded on
    its value. The gate is instead a presence check, ``if not answer.references``, and the
    distinction matters: ``max(..., default=0.0)`` makes an uncited answer score 0.0, so a
    threshold of 0.0 would admit every uncited sentence, while a presence check drops exactly the
    uncited ones and nothing else.
    """
    return max(answer.references.values(), default=0.0)


def task1_report(
    bank: Sequence[Nugget],
    topic: Topic,
    knobs: SerializeKnobs,
    *,
    team_id: str,
    run_id: str,
    run_desc: str,
    stats: Statistics | None = None,
    variant: str | None = None,
    seed: int | None = None,
) -> T1ReportRow:
    """Project one topic's bank into the Task-1 report row.

    ``topic.limit`` is read per topic, never as a constant. This budget-fit, not the validator, is
    what enforces it: the validator only warns by default and never truncates.
    """
    limit = int(topic.limit)
    slices = _slices(variant, seed)
    answered = sorted(
        [n for n in bank if n.status == STATUS_ANSWERED and n.answers],
        key=lambda n: (-n.weight, n.nugget_id),
    )

    texts: list[str] = []
    citations: list[dict[str, float]] = []
    used = 0
    admitted: set[tuple[str, int]] = set()
    considered: set[tuple[str, int]] = set()
    dropped_grounding = 0
    dropped_budget = 0
    #: sentence-key -> index in `texts`, so a repeat merges its citations into the first copy.
    by_text: dict[str, int] = {}
    dropped_duplicate = 0

    def admit(nugget: Nugget, index: int) -> bool:
        nonlocal used, dropped_grounding, dropped_budget, dropped_duplicate
        considered.add((nugget.nugget_id, index))
        answer = nugget.answers[index]
        if not answer.references:
            dropped_grounding += 1
            return False

        # The same sentence is admitted once and its evidence merged rather than discarded. The
        # terminal `merge_answers` dedup runs per nugget, so two nuggets answered from the same
        # passage each keep their own copy and both reach here. Merging rather than skipping keeps
        # the later copy's `references`, which can cite different documents, costs no characters
        # since the sentence is already paid for, and preserves the rule that `references` covers
        # every cited document.
        #
        # Keyed on NFC, collapsed whitespace and casefold, the same canonicalisation `fanin` uses
        # to cluster answer values. Exact by design: collapsing paraphrases needs an embedding and
        # a threshold, which is `merge_answers`' job rather than a budget-fill loop's.
        key = " ".join(nfc(answer.sentence).split()).casefold()
        if key in by_text:
            citations[by_text[key]].update(answer.references)
            dropped_duplicate += 1
            return False

        cost = fits(used, answer.sentence, len(texts), limit)
        if cost is None:
            dropped_budget += 1
            return False
        by_text[key] = len(texts)
        texts.append(answer.sentence)
        citations.append(dict(answer.references))
        used += cost
        admitted.add((nugget.nugget_id, index))
        return True

    # Tier 1: one sentence per distinct nugget, best answer first.
    for nugget in answered:
        admit(nugget, _best_answer_index(nugget.answers))

    # Tier 2 fills the residual budget by weight times score. The excluded set is everything Tier 1
    # considered, keyed by `(nugget_id, answer index)` rather than by identity against index 0:
    # Tier 1 picks the best-scoring answer, which is not necessarily the first, so an index test
    # would re-offer an already-admitted sentence. A candidate Tier 1 rejected cannot pass in
    # Tier 2 either, since the budget only shrinks, so re-offering it would waste work and
    # double-count the drop.
    tier2 = sorted(
        (
            (nugget, i)
            for nugget in answered
            for i in range(len(nugget.answers))
            if (nugget.nugget_id, i) not in considered
        ),
        key=lambda pair: (-(pair[0].weight * pair[0].answers[pair[1]].score), pair[0].nugget_id, pair[1]),
    )
    for nugget, index in tier2:
        admit(nugget, index)

    if not texts and knobs.empty_fallback:
        texts, citations = _fallback(answered, knobs, limit)
        if stats is not None:
            stats.emit(T1_FALLBACK, 1.0, **slices)

    # The number that decides acceptance is measured over the final list; see
    # `budget.trim_to_limit` for why the incremental accounting is not the last word.
    trimmed = trim_to_limit(texts, limit)
    if len(trimmed) != len(texts):
        dropped_budget += len(texts) - len(trimmed)
        citations = citations[: len(trimmed)]
        texts = trimmed

    responses = tuple(T1Response(text=t, citations=c) for t, c in zip(texts, citations, strict=True))
    references = _reference_union(responses)

    if stats is not None:
        stats.emit(T1_SENTENCES_ADMITTED, float(len(responses)), **slices)
        stats.emit(T1_SENTENCES_DROPPED_GROUNDING, float(dropped_grounding), **slices)
        stats.emit(T1_SENTENCES_DROPPED_BUDGET, float(dropped_budget), **slices)
        stats.emit(T1_SENTENCES_DROPPED_DUPLICATE, float(dropped_duplicate), **slices)
        stats.emit(T1_BUDGET_UTILISATION, joined_len(texts) / limit if limit else 0.0, **slices)
        stats.emit(T1_NUGGETS_COVERED, float(len({nid for nid, _ in admitted})), **slices)

    return T1ReportRow(
        topic_id=str(topic.topic_id),
        team_id=team_id,
        run_id=run_id,
        run_desc=run_desc,
        responses=responses,
        references=references,
    )


def _best_answer_index(answers: Sequence[Answer]) -> int:
    """Return the highest-scoring answer index, with ties broken on the lower index."""
    return min(range(len(answers)), key=lambda i: (-answers[i].score, i))


def _reference_union(responses: Sequence[T1Response]) -> tuple[str, ...]:
    """Return the sorted set of every cited doc-id.

    The union is a real requirement that the validator does not check: an empty ``references``
    beside citing sentences passes. The sort is our own determinism choice, also unchecked and
    free, and it makes the artifact byte-deterministic across runs.
    """
    return tuple(sorted({doc for r in responses for doc in r.citations}))


def _fallback(
    answered: Sequence[Nugget], knobs: SerializeKnobs, limit: int
) -> tuple[list[str], list[dict[str, float]]]:
    """Return the empty-output fallback, so ``responses[]`` is never empty.

    The single highest ``weight x score`` sentence with its citations; only if even that is
    uncited, or there is no answered nugget at all, one neutral framing sentence with no
    citations. An empty ``responses[]`` passes the validator but scores zero.
    """
    # Sorted rather than `max`, so the tie-break has the same shape as Tier 2's: descending
    # `weight x score`, then ascending `nugget_id`, then ascending answer index. An all-zero
    # weight or score bank is a real case, so the tie branch is exercised rather than decorative.
    candidates = sorted(
        (
            (nugget, index)
            for nugget in answered
            for index in range(len(nugget.answers))
        ),
        key=lambda pair: (
            -(pair[0].weight * pair[0].answers[pair[1]].score),
            pair[0].nugget_id,
            pair[1],
        ),
    )
    if candidates:
        nugget, index = candidates[0]
        answer = nugget.answers[index]
        if answer.references:
            text = answer.sentence
            # Admit the single highest-scoring sentence even if it alone approaches the budget,
            # then stop, but never past the hard limit, which is what the scorer measures. A
            # single over-limit sentence falls through to the framing template.
            if text and joined_len([text]) <= limit:
                return [text], [dict(answer.references)]
    return [FALLBACK_SENTENCE], [{}]


def _slices(variant: str | None, seed: int | None) -> dict[str, Any]:
    """Return the canonical stats slice keys this stage emits on."""
    out: dict[str, Any] = {}
    if variant is not None:
        out["variant"] = variant
    if seed is not None:
        out["seed"] = seed
    return out
