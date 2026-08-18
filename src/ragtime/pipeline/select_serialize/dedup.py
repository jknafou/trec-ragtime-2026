"""The terminal, deterministic, English-only dedup: a third policy, not a reuse of two others.

Dedup exists in three policies by design, and none substitutes for another:

* ``decompose.dedup.dedup_nuggets`` is a paraphrase merge over the growing bank, greedy and
  incremental with the older id winning, using cosine as a recall pre-filter behind an LLM gate.
  It is a coverage-loop policy: it changes what the next round searches for.
* ``rag_loop.fanin.fan_in`` is string-identity clustering of committed claims into answer values,
  never embeddings.
* This module is a bank-level guard run once, immediately before the three projections. Questions
  merge on embeddings only, because an LLM merge would break the determinism the projection
  promises; answer values need the LLM as a confirmatory gate because short phrases make cosine
  noisy.

What is shared is the arithmetic: :func:`~ragtime.pipeline.decompose.dedup.cosine` is imported
rather than re-spelled, so the repository has one similarity implementation, a plain dot product
over the already-normalized ``serving.embed`` vectors.

Similarity is computed on English text only, ``nugget.question`` and ``answer.answer``, so the
cutoff is language-invariant and the multilingual doc-ids never enter it. That is what keeps dedup
from confounding the translation comparison.

No model is constructed here. ``embed`` and ``confirm`` arrive as injected callables sliced from
the one ``serving`` ``ClientBundle``, so this module structurally cannot stand up a second
embedder or a second vLLM.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ragtime.common import Answer, Nugget

# The arithmetic, not the policy: one dot product in this repository.
from ragtime.pipeline.decompose.dedup import cosine

if TYPE_CHECKING:
    from ragtime.common import Statistics

__all__ = [
    "SERIALIZE_ANSWERS_MERGED",
    "SERIALIZE_QUESTIONS_MERGED",
    "SERIALIZE_QUESTION_CLUSTER_MEAN",
    "merge_answers",
    "merge_questions_embedding",
]

#: Near-duplicate questions merged at serialize, which should be near zero if the saturation dedup
#: held. A high value here is a bug signal about decompose, not a success.
SERIALIZE_QUESTIONS_MERGED = "serialize.questions_merged"
#: Mean size of the merged connected components; 1.0 when nothing merged.
SERIALIZE_QUESTION_CLUSTER_MEAN = "serialize.question_cluster_mean"
#: Answer values merged inside OR nuggets.
SERIALIZE_ANSWERS_MERGED = "serialize.answers_merged"

#: A merge rate above this fraction of the bank is logged as a warning. It changes no behaviour:
#: an automatic response to a bug signal would hide the bug.
BUG_SIGNAL_MERGE_RATE = 0.2

#: The one aggregator value this module branches on: an ``AND`` nugget is exempt from the answer
#: merge. ``OR`` needs no constant here, because it is the case that is not skipped.
AGGREGATOR_AND = "AND"


def _rows(vectors: Any, n: int, what: str) -> list[tuple[float, ...]]:
    """Return whatever ``embed`` produced as plain float tuples, with no numpy or sentence-transformers.

    Same constraint as ``decompose.dedup._rows`` and for the same reason: neither package is in
    the base environment, so importing one would make this module, and every small test of it,
    unimportable on the dev box.
    """
    rows = [tuple(float(x) for x in row) for row in vectors]
    if len(rows) != n:
        raise ValueError(f"embed() returned {len(rows)} vectors for {n} {what}")
    widths = {len(r) for r in rows}
    if len(widths) > 1:
        raise ValueError(f"embed() returned ragged vectors (widths {sorted(widths)})")
    return rows


class _Components:
    """Union-find over indices, for the connected-component merge."""

    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            # Lower index wins, so the representative is the earlier bank entry and, bank order
            # being nugget-id order, the lower nugget_id. That is the tie rule, structurally.
            self.parent[max(a, b)] = min(a, b)


def merge_questions_embedding(
    bank: Sequence[Nugget],
    cutoff: float,
    *,
    embed: Callable[[list[str]], Any],
    stats: Statistics | None = None,
    log: Any = None,
) -> tuple[Nugget, ...]:
    """Connect question pairs at ``cos >= cutoff`` and merge the connected components.

    Per component: the maximum ``weight``, the union of ``answers[]`` and ``retrieved[]``, and the
    lower ``nugget_id`` as representative. Deterministic, because connected components do not
    depend on iteration order, unlike the greedy incremental merge in decompose, which is why the
    terminal projection uses this policy rather than that one.

    ``embed`` is called once for the whole bank, never per pair.
    """
    bank = tuple(bank)
    if len(bank) < 2:
        return bank
    # Bank order is nugget-id order for a bank minted by `common.ids.nugget_id`; sorting here
    # makes the representative rule hold even for a hand-assembled bank.
    order = sorted(range(len(bank)), key=lambda i: bank[i].nugget_id)
    ordered = [bank[i] for i in order]
    vectors = _rows(embed([n.question for n in ordered]), len(ordered), "questions")

    comps = _Components(len(ordered))
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if cosine(vectors[i], vectors[j]) >= cutoff:
                comps.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(ordered)):
        groups.setdefault(comps.find(i), []).append(i)

    merged_out: list[Nugget] = []
    for root in sorted(groups):
        members = [ordered[i] for i in groups[root]]
        merged_out.append(_merge_component(members))

    merged_away = len(ordered) - len(merged_out)
    if stats is not None:
        stats.emit(SERIALIZE_QUESTIONS_MERGED, float(merged_away))
        stats.emit(SERIALIZE_QUESTION_CLUSTER_MEAN, len(ordered) / len(merged_out))
    if log is not None and merged_away > BUG_SIGNAL_MERGE_RATE * len(ordered):
        log.warning(
            "serialize.dedup.high_question_merge_rate",
            merged=merged_away,
            bank=len(ordered),
            note="a high merge rate here is a bug signal about decompose, not a success",
        )
    return tuple(merged_out)


def _merge_component(members: Sequence[Nugget]) -> Nugget:
    """Collapse one connected component to one nugget: max weight, union of answers and retrieved."""
    keep = members[0]  # members arrive in nugget-id order; the lower id represents
    if len(members) == 1:
        return keep
    answers: list[Answer] = []
    seen: set[tuple[str, str]] = set()
    retrieved: list[Any] = []
    seen_passages: set[str] = set()
    for member in members:
        for answer in member.answers:
            key = (answer.answer, answer.quoted_span)
            if key not in seen:
                seen.add(key)
                answers.append(answer)
        for hit in member.retrieved:
            if hit.passage_id not in seen_passages:
                seen_passages.add(hit.passage_id)
                retrieved.append(hit)
    return replace(
        keep,
        weight=max(m.weight for m in members),
        answers=tuple(answers),
        retrieved=tuple(retrieved),
    )


async def merge_answers(
    nugget: Nugget,
    cutoff: float,
    *,
    embed: Callable[[list[str]], Any],
    confirm: Callable[[str, str], Awaitable[bool]],
    stats: Statistics | None = None,
) -> Nugget:
    """Merge duplicate answer values inside one ``OR`` nugget.

    An embedding candidate at ``cos >= cutoff`` merges only if the greedy, order-canonicalized
    ``confirm`` gate also agrees, so either gate may veto, which is the recall-safe direction. On
    merge, references are unioned keeping the maximum score per doc-id, and ``support[]`` is
    unioned.

    ``AND`` nuggets are exempt and return untouched: their answers are required components rather
    than paraphrases, and each scores as a separate nugget.

    The pair handed to ``confirm`` is sorted by the answer value, so the judge sees a canonical
    order and its position bias cannot make the merge depend on bank order.
    """
    if nugget.aggregator_type == AGGREGATOR_AND or len(nugget.answers) < 2:
        return nugget
    answers = list(nugget.answers)
    vectors = _rows(embed([a.answer for a in answers]), len(answers), "answer values")

    survivors: list[int] = []
    merged = 0
    for i, answer in enumerate(answers):
        target: int | None = None
        for pos in survivors:
            if cosine(vectors[pos], vectors[i]) < cutoff:
                continue
            left, right = sorted((answers[pos].answer, answer.answer))
            if await confirm(left, right):
                target = pos
                break
        if target is None:
            survivors.append(i)
            continue
        answers[target] = _absorb_answer(answers[target], answer)
        merged += 1

    if stats is not None and merged:
        stats.emit(SERIALIZE_ANSWERS_MERGED, float(merged), nugget=nugget.nugget_id)
    return replace(nugget, answers=tuple(answers[i] for i in survivors))


def _absorb_answer(kept: Answer, merged_away: Answer) -> Answer:
    """Union the doc-ids, keeping the max score each, and the ``support[]`` of two merged answers."""
    references = dict(kept.references)
    for doc, score in merged_away.references.items():
        if doc not in references or score > references[doc]:
            references[doc] = score
    support = list(kept.support)
    seen = {(s.passage_id, s.lang) for s in support}
    for s in merged_away.support:
        if (s.passage_id, s.lang) not in seen:
            seen.add((s.passage_id, s.lang))
            support.append(s)
    return replace(
        kept,
        references=references,
        support=tuple(support),
        score=max(kept.score, merged_away.score),
    )
