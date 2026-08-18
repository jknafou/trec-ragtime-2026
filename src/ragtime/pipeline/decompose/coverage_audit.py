"""The coverage audit: mark-answered, gap-detect, prune, over the loops' own evidence.

This is the data half of ``grow_nuggets``' branch for rounds 1 and later. It reads what the k RAG
loops already produced this round, judges which nuggets are now covered, proposes nuggets for the
facets that are not, and flags the off-topic or redundant ones. The auditor issues no queries of
its own, and the whole thing is one constrained-decoding call per round on the shared vLLM.

The judge is not trusted with the closing decision, because a coverage judge can
over-claim and premature stopping is the dominant failure mode of these systems. Three
consequences are enforced in code rather than left to the prompt:

* a nugget flips to ``answered`` only on ``full``, and only if it actually has a committed answer.
  A label with nothing behind it is counted as :data:`AUDIT_OVERCLAIM_BLOCKED` and ignored,
  because a closed nugget stops driving retrieval permanently;
* a nugget that has committed answers is never pruned. Answered is not deleted: an answered nugget
  is part of the coverage taxonomy and carries Task-1 sentences and Task-3 answers, so dropping it
  would delete deliverable content rather than a question;
* the loop's stop condition is nowhere in this module. Saturation is measured on net-new nuggets
  by :func:`~ragtime.pipeline.decompose.saturation.saturated`, and the audit schema carries no
  ``saturated`` field, so ending the loop on the model's own say-so is unrepresentable rather than
  merely discouraged.

This module imports nothing from ``ragtime.retrieval`` or ``ragtime.pipeline.rag_loop``, the same
structural rule the rest of ``decompose/`` lives under. :func:`evidence_from_results` therefore
reads a loop result by attribute, and passage text arrives through an injected ``render`` callable
the driver builds from ``retrieval.display``. That keeps "the audit sees only what the loops saw" a
property of the data flow, and lets every test here run with no index and no model. It imports
nothing from ``serving.registry`` either: the LLM arrives as the injected ``ClientBundle`` and the
compiled schema is read off ``clients.schemas``, never re-declared here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ragtime.common import Nugget, Statistics, get_logger

from . import bank as bank_ops
from .prompts import AUDIT_SYSTEM, decoding_kwargs, gap_audit_prompt

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "AUDIT_COVERAGE_LABEL",
    "AUDIT_GAPS_PROPOSED",
    "AUDIT_MARKED_ANSWERED",
    "AUDIT_OVERCLAIM_BLOCKED",
    "AUDIT_PRUNED",
    "AUDIT_PRUNE_REFUSED",
    "AUDIT_TRIGGER_UNKNOWN",
    "COVERAGE_FULL",
    "COVERAGE_LABELS",
    "COVERAGE_NONE",
    "COVERAGE_PARTIAL",
    "MAX_EVIDENCE_PASSAGES",
    "AnswerView",
    "AuditDelta",
    "Evidence",
    "Gap",
    "apply_delta",
    "attach_evidence",
    "audit",
    "evidence_from_results",
]

_log = get_logger("pipeline.decompose.coverage_audit")

#: The AutoNuggetizer three-way coverage scale, the same scale the track's evaluator uses, so the
#: audit is aligned to what Task 3 is scored against.
COVERAGE_FULL = "full"
COVERAGE_PARTIAL = "partial"
COVERAGE_NONE = "none"
COVERAGE_LABELS = (COVERAGE_FULL, COVERAGE_PARTIAL, COVERAGE_NONE)

#: Per-round counters. The novelty curve itself is emitted by ``round_loop``, the only thing that
#: can see two consecutive banks; these are the audit's own per-round rates.
AUDIT_COVERAGE_LABEL = "decompose.audit_coverage_label"
AUDIT_MARKED_ANSWERED = "decompose.audit_marked_answered"
#: A ``full`` label the code refused because the nugget has no committed answer behind it. The
#: over-claim rate is the diagnostic that says whether the judge can be trusted at all.
AUDIT_OVERCLAIM_BLOCKED = "decompose.audit_overclaim_blocked"
AUDIT_GAPS_PROPOSED = "decompose.audit_gaps_proposed"
AUDIT_PRUNED = "decompose.audit_pruned"
#: A prune the code refused: an unknown id, or a nugget carrying committed answers.
AUDIT_PRUNE_REFUSED = "decompose.audit_prune_refused"
#: A ``trigger_passage_id`` naming a passage this round never retrieved. Dropped to ``None``
#: rather than stored: the trajectory log is provenance, and a fabricated id in it is worse than
#: an absent one because it reads as evidence.
AUDIT_TRIGGER_UNKNOWN = "decompose.audit_trigger_unknown"

#: How many retrieved passages the audit prompt may carry. A prompt-size safety bound rather than
#: a run choice or a quality knob: the audit reads whatever the loops returned, and this only caps
#: how much of it is rendered into one request so a wide round cannot walk the context wall.
#: ``max_tokens`` is never sent, so the caller-side bound is the only one. It is set well above
#: the number of distinct passages a handful of loops surface, and passages are dropped
#: lowest-score-first.
MAX_EVIDENCE_PASSAGES = 24


@dataclass(frozen=True, slots=True)
class AnswerView:
    """One committed answer as the auditor sees it, plus the ids that back it.

    A projection of ``common.schemas.Answer`` rather than the record itself, because the audit
    judges over the span-grounded committed claims and nothing else; carrying the whole record
    would hand the prompt builder fields such as ``score`` and ``references`` that the coverage
    decision may not depend on.
    """

    answer: str
    sentence: str
    quoted_span: str
    passage_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Evidence:
    """One round's reusable evidence: answers per nugget, retrieval per nugget, passage text.

    ``passages`` is ``[(passage_id, text)]`` and is empty unless the caller injected a ``render``,
    because ``LoopResult`` carries ids and scores but no passage text. Without a renderer the audit
    judges over the committed spans alone, which are verbatim corpus text and are exactly what
    coverage is judged over; evidence-driven gap detection is the part that degrades, and the
    audit's own counters say so.
    """

    answers: Mapping[str, tuple[AnswerView, ...]]
    retrieved: Mapping[str, tuple[tuple[str, float], ...]]
    passages: tuple[tuple[str, str], ...] = ()

    def passage_ids(self) -> frozenset[str]:
        """Return every passage id this round retrieved: the ``trigger_passage_id`` domain."""
        ids: set[str] = set()
        for hits in self.retrieved.values():
            ids.update(pid for pid, _ in hits)
        ids.update(pid for pid, _ in self.passages)
        for views in self.answers.values():
            for view in views:
                ids.update(view.passage_ids)
        return frozenset(ids)

    def answered_ids(self) -> frozenset[str]:
        """Return the nuggets with at least one committed answer: the only ones that may close."""
        return frozenset(nid for nid, views in self.answers.items() if views)


@dataclass(frozen=True, slots=True)
class Gap:
    """One proposed nugget, before the on-topic admission gate has seen it."""

    question: str
    trigger_passage_id: str | None = None
    weight: float = 0.0

    @property
    def evidence_driven(self) -> bool:
        """True when a retrieved passage surfaced this facet, rather than the request spine.

        New nuggets are reported split by origin, evidence-driven or request-driven, and this is
        the only place the distinction is knowable.
        """
        return bool(self.trigger_passage_id)


@dataclass(frozen=True, slots=True)
class AuditDelta:
    """One audit call's output: the three-way labels, the proposed gaps, the prune list."""

    rationale: str
    coverage: Mapping[str, str]
    gaps: tuple[Gap, ...]
    prune: tuple[str, ...]


def evidence_from_results(
    results: Sequence[Any],
    *,
    render: Callable[[Sequence[str]], Sequence[tuple[str, str]]] | None = None,
    max_passages: int = MAX_EVIDENCE_PASSAGES,
) -> Evidence:
    """Normalize one round's loop results into :class:`Evidence`. Duck-typed.

    Reads ``nugget_id``, ``answers`` and ``retrieved`` by attribute, so this module imports nothing
    from ``pipeline.rag_loop`` and a test can drive it with a plain namespace. ``render`` maps
    passage ids to ``[(passage_id, text)]`` and is the driver's ``retrieval.display`` binding;
    without it the audit runs on committed spans alone.

    Passages are ranked by their best score across the round's loops and cut at ``max_passages``,
    lowest score first, so what survives is what the round found most relevant. The cut is
    deterministic, with ties broken on the id.
    """
    answers: dict[str, tuple[AnswerView, ...]] = {}
    retrieved: dict[str, tuple[tuple[str, float], ...]] = {}
    best: dict[str, float] = {}
    for result in results:
        nugget_id = str(getattr(result, "nugget_id", "") or "")
        if not nugget_id:
            continue
        views = tuple(_answer_view(a) for a in (getattr(result, "answers", ()) or ()))
        answers[nugget_id] = tuple(v for v in views if v.answer or v.quoted_span)
        hits = tuple(
            (str(getattr(r, "passage_id", "")), float(getattr(r, "score", 0.0)))
            for r in (getattr(result, "retrieved", ()) or ())
        )
        retrieved[nugget_id] = hits
        for pid, score in hits:
            if pid and score > best.get(pid, float("-inf")):
                best[pid] = score

    passages: tuple[tuple[str, str], ...] = ()
    if render is not None and best:
        ordered = [pid for pid, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]
        wanted = ordered[: max(0, int(max_passages))]
        if wanted:
            passages = tuple((str(pid), str(text)) for pid, text in render(wanted))
    return Evidence(answers=answers, retrieved=retrieved, passages=passages)


def _answer_view(answer: Any) -> AnswerView:
    def field(name: str, default: str = "") -> str:
        if isinstance(answer, dict):
            return str(answer.get(name, default) or default)
        return str(getattr(answer, name, default) or default)

    support = answer.get("support", ()) if isinstance(answer, dict) else getattr(answer, "support", ())
    passage_ids = tuple(
        str(s.get("passage_id", "") if isinstance(s, dict) else getattr(s, "passage_id", ""))
        for s in (support or ())
    )
    return AnswerView(
        answer=field("answer"),
        sentence=field("sentence"),
        quoted_span=field("quoted_span"),
        passage_ids=tuple(pid for pid in passage_ids if pid),
    )


def attach_evidence(bank: tuple[Nugget, ...], evidence: Evidence) -> tuple[Nugget, ...]:
    """Fold this round's ``answers`` and ``retrieved`` onto the bank's nuggets. No LLM call.

    This is the data half of the audit's mark-answered step: the loops own ``answers[]`` and
    ``retrieved[]``, and the bank is where they accumulate. The status half is :func:`apply_delta`,
    which runs after the judge has spoken.

    Append, never replace: a nugget still open after round 1 is re-fanned in round 2 and its
    round-1 evidence must survive, or the audit would re-judge it against a shrinking record.
    Idempotent by ``(answer, quoted_span)`` and by ``passage_id``, so a resumed round cannot
    double-count.
    """
    out: list[Nugget] = []
    for n in bank:
        views = evidence.answers.get(n.nugget_id, ())
        hits = evidence.retrieved.get(n.nugget_id, ())
        if not views and not hits:
            out.append(n)
            continue
        out.append(
            replace(
                n,
                answers=_merge_answers(n.answers, views),
                retrieved=_merge_retrieved(n.retrieved, hits),
            )
        )
    return tuple(out)


def _merge_answers(existing: tuple[Any, ...], views: Sequence[AnswerView]) -> tuple[Any, ...]:
    from ragtime.common import Answer, Support

    seen = {(a.answer, a.quoted_span) for a in existing}
    merged = list(existing)
    for view in views:
        key = (view.answer, view.quoted_span)
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            Answer(
                answer=view.answer,
                sentence=view.sentence,
                quoted_span=view.quoted_span,
                support=tuple(Support(passage_id=pid, lang="") for pid in view.passage_ids),
            )
        )
    return tuple(merged)


def _merge_retrieved(existing: tuple[Any, ...], hits: Sequence[tuple[str, float]]) -> tuple[Any, ...]:
    from ragtime.common import Retrieved

    seen = {r.passage_id for r in existing}
    merged = list(existing)
    for pid, score in hits:
        if not pid or pid in seen:
            continue
        seen.add(pid)
        merged.append(Retrieved(passage_id=pid, score=score))
    return tuple(merged)


async def audit(
    bank: tuple[Nugget, ...],
    answers: Evidence,
    *,
    problem_statement: str,
    background: str,
    cfg: Any,
    clients: Any,
    seed: int,
    round: int,
    stats: Statistics | None = None,
    title: str = "",
) -> AuditDelta:
    """Judge coverage, propose gaps and flag prunes in one constrained-decoding call.

    ``title`` is the topic's short request label, passed through to the prompt for the
    request-driven half of gap detection: this pass proposes a facet the problem statement demands
    that nothing has surfaced yet, and the seed enumerated against a request that included the
    title, so an auditor shown a smaller request would be measuring a different one. It enters
    decompose only and never reaches the RAG loop, which is request-blind by construction.

    ``bank`` should already carry this round's evidence via :func:`attach_evidence`, because the
    coverage judgement is made over the answers shown in the prompt and nothing else.

    Only open nuggets are offered for a coverage label. An already-answered or pruned nugget is
    shown for context, which is what makes a duplicate proposal visible to the judge, but its
    label is discarded, so the audit cannot re-open or re-close a settled nugget.

    Everything the model returns is treated as a proposal: labels outside the three-way enum are
    dropped, ids that are not in the bank are dropped, and malformed questions are dropped by the
    same ``question_of`` rule round 0 uses. The schema already forbids most of this; the code
    re-checks because a structured-output request can be silently ignored on this stack.
    """
    stats = stats if stats is not None else Statistics()
    open_ids = {n.nugget_id for n in bank if n.status == bank_ops.STATUS_UNANSWERED}
    known_ids = {n.nugget_id for n in bank}

    obj = await clients.llm.generate(
        clients.schemas.coverage_audit,
        gap_audit_prompt(
            problem_statement,
            background,
            [_tagged(n, answers) for n in bank],
            answers.passages,
            round=round,
            title=title,
        ),
        seed,
        system=AUDIT_SYSTEM,
        **decoding_kwargs(cfg),
    )

    coverage: dict[str, str] = {}
    for entry in obj.get("coverage", ()) or ():
        nugget_id = str((entry or {}).get("nugget_id", "") or "")
        label = str((entry or {}).get("coverage", "") or "")
        if nugget_id not in open_ids or label not in COVERAGE_LABELS:
            continue
        coverage[nugget_id] = label
        stats.emit(AUDIT_COVERAGE_LABEL, 1.0, round=round, nugget=nugget_id)

    gaps: list[Gap] = []
    for entry in obj.get("add", ()) or ():
        gap = _gap(entry or {}, answers, stats=stats, round=round)
        if gap is not None:
            gaps.append(gap)
    if gaps:
        stats.emit(AUDIT_GAPS_PROPOSED, float(len(gaps)), round=round)

    prune = tuple(
        dict.fromkeys(
            str(pid) for pid in (obj.get("prune", ()) or ()) if str(pid) in known_ids
        )
    )
    _log.info(
        "decompose.audit.delta",
        round=round,
        bank=len(bank),
        open=len(open_ids),
        labelled=len(coverage),
        full=sum(1 for v in coverage.values() if v == COVERAGE_FULL),
        proposed=len(gaps),
        pruned=len(prune),
        evidence_passages=len(answers.passages),
    )
    return AuditDelta(
        rationale=str(obj.get("rationale", "") or ""),
        coverage=coverage,
        gaps=tuple(gaps),
        prune=prune,
    )


def _tagged(nugget: Nugget, evidence: Evidence) -> dict[str, Any]:
    """Render one bank entry in the shape :func:`gap_audit_prompt` expects.

    ``answers`` is the short value plus its grounding span, which is the committed claim the
    coverage judgement is made over. The report sentence is left out: it is model
    prose, and judging coverage against it would be judging free text.
    """
    views = evidence.answers.get(nugget.nugget_id, ())
    shown = [
        f"{v.answer}" + (f'  (quoted: "{v.quoted_span}")' if v.quoted_span else "")
        for v in views
    ] or [
        f"{a.answer}" + (f'  (quoted: "{a.quoted_span}")' if a.quoted_span else "")
        for a in nugget.answers
    ]
    return {
        "nugget_id": nugget.nugget_id,
        "question": nugget.question,
        "aggregator_type": nugget.aggregator_type,
        "status": nugget.status,
        "answers": shown,
    }


def _gap(entry: Mapping[str, Any], evidence: Evidence, *, stats: Statistics, round: int) -> Gap | None:
    """Validate one ``add[]`` entry. Returns ``None`` when it is not a usable proposal."""
    # Imported lazily because `grow_nuggets` imports this module for its later-round branch. The
    # rule is round 0's, unchanged: an emission whose `question` carries the model's own
    # deliberation is discarded rather than repaired.
    from .grow_nuggets import question_of

    question = question_of(str(entry.get("question", "") or "").strip())
    if not question:
        return None
    trigger = entry.get("trigger_passage_id")
    trigger = str(trigger) if trigger else None
    if trigger is not None and trigger not in evidence.passage_ids():
        # A fabricated provenance id is worse than an absent one: `trigger_passage_id` is the
        # trajectory log's evidence link, so an invented id reads downstream as a passage that
        # surfaced this facet. Drop to request-driven, and count it.
        stats.emit(AUDIT_TRIGGER_UNKNOWN, 1.0, round=round)
        trigger = None
    weight = entry.get("weight")
    return Gap(
        question=question,
        trigger_passage_id=trigger,
        weight=0.0 if weight is None else float(weight),
    )


# Applying the delta is where the judge stops being trusted.
def apply_delta(
    bank: tuple[Nugget, ...],
    delta: AuditDelta,
    evidence: Evidence,
    *,
    stats: Statistics | None = None,
    round: int = 0,
) -> tuple[Nugget, ...]:
    """Close the nuggets labelled ``full`` that have answers, prune the flagged ones.

    Returns a new bank. Two refusals, both deliberate and both counted:

    * a ``full`` label on a nugget with no committed answer does not close it, because closing
      stops the nugget driving retrieval for the rest of the run and over-claimed coverage is the
      dominant risk;
    * a prune of a nugget that has committed answers is refused, because an answered nugget
      carries Task-1 sentences and Task-3 answers, so pruning it deletes deliverable content
      rather than a redundant question.
    """
    stats = stats if stats is not None else Statistics()
    answered = evidence.answered_ids()
    have_answers = {n.nugget_id for n in bank if n.answers} | answered

    to_close: list[str] = []
    for nugget_id, label in delta.coverage.items():
        if label != COVERAGE_FULL:
            continue
        if nugget_id not in have_answers:
            stats.emit(AUDIT_OVERCLAIM_BLOCKED, 1.0, round=round, nugget=nugget_id)
            continue
        to_close.append(nugget_id)

    to_prune: list[str] = []
    for nugget_id in delta.prune:
        if nugget_id in have_answers or nugget_id in to_close:
            stats.emit(AUDIT_PRUNE_REFUSED, 1.0, round=round, nugget=nugget_id)
            continue
        to_prune.append(nugget_id)

    out = bank_ops.close(bank, to_close) if to_close else bank
    out = bank_ops.prune(out, to_prune) if to_prune else out
    if to_close:
        stats.emit(AUDIT_MARKED_ANSWERED, float(len(to_close)), round=round)
    if to_prune:
        stats.emit(AUDIT_PRUNED, float(len(to_prune)), round=round)
    return out
