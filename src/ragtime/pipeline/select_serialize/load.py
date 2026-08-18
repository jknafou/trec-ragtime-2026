"""Assemble the bank view from a finished run's own artifacts.

Three jobs, each of which exists because reading the obvious file alone would be wrong:

1. Deciding which bank is authoritative. ``decompose/round_{r}.jsonl`` is written once per round
   and ``rag_loop/{nugget_id}.jsonl`` gains a line per round, so a cell interrupted between a fan
   and the audit that consumes it has loop records for a round with no bank. The authoritative
   bank is the last round carrying ``_SUCCESS``, and loop-record rounds beyond it are ignored;
   otherwise serialize would ship evidence the coverage audit never saw, and two re-runs over the
   same tree could disagree.

2. Finding where the citations come from. ``decompose.coverage_audit._merge_answers`` rebuilds
   every bank answer from an ``AnswerView`` that carries only ``answer``, ``sentence``,
   ``quoted_span`` and ``passage_ids``, so the persisted bank's answers have empty ``references``
   and a zero ``score``. Reading the bank alone would emit an empty citation dict on every Task-1
   sentence and an empty ``references`` list on every Task-3 answer: a file that validates and
   carries nothing. :func:`assemble_bank` therefore enriches each bank answer from the loop
   records, joined on the same idempotence key the loop side uses. A bank answer with no
   loop-record twin is a hard error, never a silent empty citation dict.

3. Owning the single point of coupling to the citation scorer. :func:`citation_scores` is the only
   function here that knows the scorer's on-disk shape, so if the scorer moves its output, exactly
   this function changes and nothing else does.

Nothing here invents a number: ``answer.score`` is derived under a config-declared policy, never
fabricated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ragtime.common import Answer, Nugget, doc_id_of
from ragtime.common.io import read_jsonl
from ragtime.common.layout import Layout
from ragtime.pipeline.decompose.bank import last_complete_round, read_bank
from ragtime.pipeline.records import answer_key, loop_evidence

__all__ = [
    "MissingLoopEvidence",
    "answer_key",
    "assemble_bank",
    "citation_scores",
    "last_complete_round",
    "topic_dirs",
]

#: The citation-scorer seam: ``<run>/citation_scores/scores.jsonl``, one row per scored citation
#: carrying ``nugget_id``, ``answer``, ``quoted_span``, ``doc_id`` and ``score``. The producer
#: spells the same name in ``citation_scoring.scorer.SCORES_FILE`` and the two are kept equal by
#: hand: the scorer is upstream and must not import this package, so the alternative would be a
#: shared constant in ``common`` for one filename. Nothing checks that they still agree.
SCORES_FILE = "scores.jsonl"


class MissingLoopEvidence(RuntimeError):
    """A bank answer has no matching loop record; refuse rather than emit an empty citation."""


def topic_dirs(cell_dir: str | Path) -> tuple[Path, ...]:
    """Return the completed topic directories of one cell, in topic-id order.

    A cell is a topic shard, and a topic is complete when ``Layout.success()`` exists, which is
    the witness the driver writes last. An incomplete topic is skipped, not half-serialized.

    A sorted listing and nothing more: this is not a work queue. There is no claim,
    no heartbeat and no poison count, because ``orchestration.slurm.workqueue`` exists for
    crash-safe shard claiming across array tasks and this is one single-cell job reading
    directories that are already finished.
    """
    base = Path(cell_dir) / "topics"
    if not base.is_dir():
        return ()
    return tuple(
        d for d in sorted(base.iterdir(), key=lambda p: p.name)
        if d.is_dir() and (d / "_SUCCESS").exists()
    )




def citation_scores(layout: Layout) -> dict[tuple[str, str, str], float] | None:
    """Return the citation scores as ``(nugget_id, answer_key, doc_id) -> score``, or ``None``.

    ``None`` means the citation scorer has not run for this cell, deliberately distinct from an
    empty mapping, which means it ran and scored nothing. Neither outcome is refused: an absent
    scorer simply falls back to weight-only ordering. But the manifest records which happened, and
    a reader must be able to tell a score-ordered run from a weight-ordered one, so collapsing
    ``None`` into an empty mapping would erase that.

    This is the only place in this package that knows the scorer's on-disk shape; the writing side
    of the same shape is ``citation_scoring.scorer``.
    """
    directory = layout.citation_scores()
    path = directory / SCORES_FILE
    if not path.exists():
        return None
    scores: dict[tuple[str, str, str], float] = {}
    for row in read_jsonl(path):
        key = (
            str(row["nugget_id"]),
            answer_key(row),
            doc_id_of(str(row["doc_id"])),
        )
        scores[key] = float(row["score"])
    return scores


def assemble_bank(
    layout: Layout,
    *,
    topic_id: str,
    answer_score: str = "max_reference",
    scores: Mapping[tuple[str, str, str], float] | None = None,
    drop_orphan_answers: bool = False,
    dropped: list[tuple[str, str]] | None = None,
) -> tuple[Nugget, ...]:
    """Return the authoritative, enriched bank for one topic.

    ``scores`` is the citation scorer's map; ``None`` leaves whatever reference scores the loop
    records carry, where ``fan_in`` wrote the keys and the scorer supplies the values.
    ``answer_score`` names the derivation policy for ``answer.score``.

    ``drop_orphan_answers`` drops a bank answer with no loop twin instead of raising
    :class:`MissingLoopEvidence`. It defaults to false, because an orphan is a real inconsistency
    and silence about it is how a class of defect hides.

    The option exists because the raise protects one thing: never ship a report sentence whose
    citation dict is empty. Dropping the uncitable answer achieves exactly that, and does it
    locally, whereas raising discards a whole arm for one bad row. Every drop is appended to
    ``dropped`` as ``(nugget_id, answer_key)`` for the caller to report, so a caller that passes
    the flag and ignores ``dropped`` has turned a loud failure into a quiet one.
    """
    through = last_complete_round(layout)
    bank = read_bank(layout.decompose_round(through))
    enriched: list[Nugget] = []
    for nugget in bank:
        if not nugget.answers:
            # Recover a swept answer the bank never recorded. The final sweep writes its loop
            # records at round `through + 1` but targets a bank round that already carries
            # `_SUCCESS`, so `write_bank` is a no-op and the answer never reaches the bank; without
            # this the nugget reaches Task 3 as a question with zero answers.
            #
            # This recovers a lost write rather than inventing an answer, and a nugget whose loop
            # produced nothing is left alone. It is safe only because the cell is complete, so the
            # only records past the last `_SUCCESS` bank are the sweep's; on an in-flight cell a
            # higher round could be half-written, which is why `loop_evidence` drops later rounds
            # in general and why this exception is scoped to unanswered nuggets.
            swept = _swept_answers(layout, nugget.nugget_id, after_round=through)
            if swept:
                enriched.append(replace(nugget, answers=swept, status="answered"))
                continue
        evidence = loop_evidence(layout, nugget.nugget_id, through_round=through)
        answers: list[Answer] = []
        for answer in nugget.answers:
            key = answer_key(answer)
            found = evidence.get(key)
            if found is None:
                if drop_orphan_answers:
                    # The answer has no citations to carry, so it cannot be emitted either way.
                    # Dropping it keeps the rest of the topic, and it is recorded so the caller
                    # can report the omission.
                    if dropped is not None:
                        dropped.append((nugget.nugget_id, key))
                    continue
                raise MissingLoopEvidence(
                    f"bank answer {key!r} of nugget {nugget.nugget_id!r} (topic {topic_id}) has "
                    f"no twin in {layout.rag_loop(nugget.nugget_id)} through round {through}. "
                    "Emitting it would ship a sentence with an empty citation dict, which "
                    "validates and cites nothing."
                )
            references = dict(found["references"])
            if scores is not None:
                for doc in list(references):
                    scored = scores.get((nugget.nugget_id, key, doc))
                    if scored is not None:
                        references[doc] = float(scored)
            # `support` is not enriched: it reaches no submission and the bank's own copy already
            # round-trips it, so touching it here would be work with no consumer.
            answers.append(
                replace(
                    answer,
                    references=references,
                    score=_derive_score(answer, references, answer_score),
                )
            )
        enriched.append(replace(nugget, answers=tuple(answers)))
    return tuple(enriched)


def _swept_answers(layout: Layout, nugget_id: str, *, after_round: int) -> tuple[Answer, ...]:
    """Return answers a loop recorded after the authoritative bank round: the sweep's lost write.

    Returns an empty tuple when the loop record is absent, has no rounds past ``after_round``, or
    those rounds carry no answers. It reads the same fields the normal path reads and applies the
    same always-original citation rule, so a recovered answer is indistinguishable from one the
    bank had recorded itself. It never fabricates: no loop answer, no recovery.
    """
    path = layout.rag_loop(nugget_id)
    if not path.exists():
        return ()
    out: list[Answer] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        if int(row.get("round", 0)) <= after_round:
            continue
        for raw in row.get("answers", ()) or ():
            key = answer_key(raw)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Answer(
                    answer=str(raw.get("answer", "") or ""),
                    sentence=str(raw.get("sentence", "") or ""),
                    quoted_span=str(raw.get("quoted_span", "") or ""),
                    references={
                        doc_id_of(str(d)): float(s)
                        for d, s in (raw.get("references", {}) or {}).items()
                    },
                    support=(),
                )
            )
    return tuple(out)


def _derive_score(answer: Answer, references: Mapping[str, float], policy: str) -> float:
    """Return ``answer.score`` under the declared policy: derived, never invented."""
    if policy == "as_written":
        return float(answer.score)
    return max(references.values(), default=0.0)
