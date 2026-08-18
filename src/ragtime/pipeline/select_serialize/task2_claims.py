"""Task 2 from the committed claims of an end-to-end run: the claim-support projection.

This is the Task-2 projection: :func:`claim_support_rows` is what :func:`.project.project` calls
for every ``task == 2`` deliverable, and every submitted Task-2 run file came out of it.

The unit here is the committed claim rather than the candidate pool, under four rules:

1. every document supporting a claim is a retrieved document;
2. a retrieved document's score is ``claim_importance x nugget_importance``;
3. a document appearing more than once accumulates, its scores added;
4. ties are broken by the reranker score of the tied documents.

Every quantity comes off disk and nothing here is invented.

* ``claim_importance x nugget_importance`` is exactly what the citation scorer persists as
  ``citation_scores/scores.jsonl``'s ``score``, since ``citation_scoring/aggregate.py`` computes
  that same product of the two factors. Rule 2 is therefore a lookup rather than a re-derivation,
  and the score here cannot drift from the scorer's.
* ``nugget_importance`` is ``nugget.weight`` from the bank, reused verbatim, the same factor the
  scorer uses.
* the reranker score is ``Retrieved.score`` from the loop record's ``retrieved[]``, taken for the
  passages the claim cited: one passage seen again in a later round keeps its best score, and a
  document's several cited passages reduce to the mean of their log-probabilities, in
  :func:`.claim_evidence.reranker_by_doc`.

One factor can be absent. ``citation_scores/scores.jsonl`` is written by a separate post-hoc pass,
so on a live tree some completed topics do not have it yet. When a claim has no scores row,
``claim_importance`` is not invented: the factor is dropped and the contribution is
``nugget_importance`` alone, multiplying by the identity rather than by a guessed number. The
affected topics are recorded in :class:`ClaimT2Diagnostics`, so a topic ranked on
``nugget_importance`` alone is legible as one.

No qrels exist, so correctness here is a format claim plus a faithful reading of the spec, and
never evidence about retrieval quality.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ragtime.common import T2RunRow, doc_id_of

if TYPE_CHECKING:
    from ragtime.common import Nugget, Statistics, Topic

    from .knobs import SerializeKnobs

__all__ = [
    "T2C_CLAIM_CITATIONS",
    "T2C_DOCS",
    "T2C_UNSCORED_CITATIONS",
    "ClaimT2Diagnostics",
    "claim_support_rows",
]

#: The (claim, document) citation pairs seen for this topic: the real unit of the run.
T2C_CLAIM_CITATIONS = "serialize.t2c_claim_citations"
#: Distinct documents in this topic's ranking, before the ``top_docs`` cap.
T2C_DOCS = "serialize.t2c_docs"
#: Citations with no scores row, meaning they were scored on ``nugget_importance`` alone. A
#: non-zero value says the topic's ranking used one factor fewer than the design asks for.
T2C_UNSCORED_CITATIONS = "serialize.t2c_unscored_citations"


@dataclass
class ClaimT2Diagnostics:
    """What a caller must be told about how a topic's rows were actually derived."""

    claim_citations: int = 0
    unscored_citations: int = 0
    docs: int = 0
    #: doc-ids for which no reranker score was recoverable, so rule 4 could not act on them.
    docs_without_reranker: int = 0
    #: rows whose score was perturbed downward to carry the rule-4 tie order into the score column.
    tie_separated_rows: int = 0
    topics_missing_scores: list[str] = field(default_factory=list)

    def merge(self, other: ClaimT2Diagnostics) -> None:
        self.claim_citations += other.claim_citations
        self.unscored_citations += other.unscored_citations
        self.docs += other.docs
        self.docs_without_reranker += other.docs_without_reranker
        self.tie_separated_rows += other.tie_separated_rows
        self.topics_missing_scores.extend(other.topics_missing_scores)


def claim_support_rows(
    bank: Sequence[Nugget],
    topic: Topic,
    knobs: SerializeKnobs,
    *,
    run_id: str,
    scores: Mapping[tuple[str, str, str], float] | None,
    reranker: Mapping[str, float],
    answer_keys: Mapping[str, str] | None = None,
    stats: Statistics | None = None,
    variant: str | None = None,
    seed: int | None = None,
    diagnostics: ClaimT2Diagnostics | None = None,
) -> tuple[T2RunRow, ...]:
    """Return one topic's claim-derived TREC rows, best first.

    ``bank`` is the authoritative enriched bank from :func:`.load.assemble_bank`; ``scores`` is the
    citation scorer's ``(nugget_id, answer_key, doc_id) -> claim_importance x nugget_importance``
    map, or ``None`` when the scorer never ran for this topic; ``reranker`` is
    ``doc_id -> reranker score`` for the rule-4 tie-break.

    ``answer_keys`` is unused by the arithmetic and exists only so a caller can pass a precomputed
    key map; the keys are recomputed here from the bank so this function has no hidden dependency.
    """
    from ragtime.pipeline.records import answer_key as _answer_key

    diag = diagnostics if diagnostics is not None else ClaimT2Diagnostics()

    # Rules 1, 2 and 3 in one pass: the unit is the (claim, document) citation, its value is the
    # two-factor product, and a document seen again adds.
    total: dict[str, float] = defaultdict(float)
    for nugget in bank:
        weight = float(nugget.weight)
        for answer in nugget.answers:
            key = _answer_key(answer)
            for ref_id in answer.references:
                doc = doc_id_of(str(ref_id))
                diag.claim_citations += 1
                scored = None if scores is None else scores.get((nugget.nugget_id, key, doc))
                if scored is None:
                    # The factor is dropped rather than guessed: multiplying by the identity
                    # leaves `nugget_importance` alone and is recorded, whereas substituting a
                    # plausible claim_importance would fabricate a judgement the model never made.
                    diag.unscored_citations += 1
                    total[doc] += weight
                else:
                    total[doc] += float(scored)

    if scores is None:
        diag.topics_missing_scores.append(str(topic.topic_id))
    diag.docs += len(total)

    # Rule 4: ties broken by the reranker score of the tied documents. Negated because the sort is
    # ascending on the key and a higher log P(yes) must come first. A document with no recoverable
    # reranker score sorts last within its tie group rather than being dropped, since rule 1
    # already made it a retrieved document.
    missing_rerank = 0
    for doc in total:
        if doc not in reranker:
            missing_rerank += 1
    diag.docs_without_reranker += missing_rerank

    def _tie_key(doc: str) -> tuple[float, ...]:
        rr = reranker.get(doc)
        return (float("inf"),) if rr is None else (-float(rr),)

    ranked = sorted(total.items(), key=lambda ds: (-ds[1], _tie_key(ds[0]), ds[0]))
    kept = ranked[: knobs.top_docs]

    # The tie order has to live in the score column or it does not exist, because the evaluator
    # re-sorts equal scores orthographically by doc-id and discards our line order.
    # `_separate_tied_groups`, defined below, spreads only the groups the tie key separated, inside
    # the gap to the next distinct score, and leaves documents it cannot separate byte-identical.
    before = list(kept)
    # `span_full_gap` consumes the whole gap to the next distinct score instead of leaving headroom
    # unused, which roughly doubles the separation and makes the emitted order robust to the
    # rounding the retrieval service applies upstream. `strict` forbids ties outright: rows an
    # equal tie key cannot separate are still ordered, by the doc-id already in the sort key. The
    # alternative is not "no order" but TREC's orthographic doc-id sort, which is not neutral in a
    # multilingual corpus. It raises `UnseparableTie` rather than emitting a tie silently.
    kept = _separate_tied_groups(kept, _tie_key, span_full_gap=True, strict=True)
    diag.tie_separated_rows += sum(1 for a, b in zip(before, kept) if a[1] != b[1])

    slices: dict[str, Any] = {}
    if variant is not None:
        slices["variant"] = variant
    if seed is not None:
        slices["seed"] = seed
    if stats is not None:
        stats.emit(T2C_CLAIM_CITATIONS, float(diag.claim_citations), **slices)
        stats.emit(T2C_DOCS, float(len(total)), **slices)
        stats.emit(T2C_UNSCORED_CITATIONS, float(diag.unscored_citations), **slices)

    rows: list[T2RunRow] = []
    for rank, (doc, score) in enumerate(kept, start=1):
        rows.append(
            T2RunRow(
                request_id=str(topic.topic_id),
                doc_id=doc,
                rank=rank,
                score=float(score),
                run_id=run_id,
            )
        )
    return tuple(rows)


#: Resolution of the emitted score column. The emitted format keeps nine decimals, so two scores
#: closer than this print identically and any separation below it is fiction. Defined here rather
#: than imported because it is a property of the emitted text, and a serializer reasoning about
#: float distances while the file reasons about nine decimals is the mismatch this closes.
_T2_MIN_STEP = 1e-9


class UnseparableTie(ValueError):
    """Strict separation could not give every tied row a distinct score at the emitted precision.

    Raised rather than emitting a tie: a validator that forbids ties paired with a serializer that
    quietly produces them is a red gate with no diagnosis attached.
    """


def _separate_tied_groups(
    kept: Sequence[tuple[str, float]],
    tie_key: Callable[[str], tuple[float, ...]],
    *,
    span_full_gap: bool = False,
    strict: bool = False,
    eps: float = 1e-6,
) -> list[tuple[str, float]]:
    """Push the tie-break order into the score, so the evaluator honours it.

    ``kept`` is already sorted by ``(-score, tie_key, doc_id)``. A tie-break expressed as row order
    does not exist in a TREC run file, because the evaluator re-sorts equal scores orthographically
    by doc-id and discards our line order, so the order has to live in the score column or not at
    all.

    :func:`claim_support_rows`, the only caller, uses ``span_full_gap=True, strict=True``; that is
    the combination every submitted Task-2 file was produced with. The other settings are retained
    because the unit tests pin the arithmetic of each knob separately, not because any run used
    them.

    ``span_full_gap=True`` takes ``step = (gap - eps) / (groups - 1)``, so the last group lands just
    above the next score and the group spans the whole gap. With ``gap=1.0``, ``eps=1e-6`` and three
    groups::

        A 6.0 -> 6.000000000   (best group, untouched)
        B 6.0 -> 5.500000500   (midpoint of the usable gap)
        C 6.0 -> 5.000001000   (just above the next score, never below it)
        D 5.0 -> 5.000000000   (untouched)

    ``span_full_gap=False`` instead takes ``step = gap / (groups + 1)`` and never consumes the gap:
    three groups in a gap of 1.0 land at 6.0, 5.75 and 5.50, leaving 0.5 unused.

    Wider separation is worth having because the retrieval service rounds every reranker
    log-probability before it reaches serialize, which is the measured cause of the tie rate.

    ``strict=True`` allows no ties at all: every row becomes its own group, so documents an equal
    ``tie_key`` could not separate are still ordered, by the doc-id already in the sort key. The
    argument is that the fallback we would otherwise cede to is TREC's orthographic doc-id sort,
    which is not neutral in a multilingual corpus, since doc-id prefixes track source and therefore
    language. Choosing the order ourselves is not less honest, only differently arbitrary, and it
    is recorded. It raises :class:`UnseparableTie` when the gap cannot fit the run at the emitted
    precision.

    The invariant throughout: a separated row may never cross a document with a genuinely lower
    score, because the spread stays strictly inside the gap and the best group keeps its original
    value. The reranker is a tie-break, never an override of the accumulated importance.

    Returns a new list; the input is not mutated, and ordering is preserved exactly.
    """
    # A tie is an equal emitted score, not an equal float. Values one ULP apart print identically
    # at nine decimals, so they are the same emitted score and belong in the same run; grouping by
    # float equality would split them, leave the separator a vanishing gap to spread into, and
    # raise `UnseparableTie` on a topic with no real tie problem. Accumulated importance sums land
    # on ULP-apart values routinely. The validator already compares the value parsed back out of
    # the text, so grouping here on the emitted string keeps the two halves of the rule agreeing.
    def _emitted(x: float) -> str:
        return f"{x:.9f}"

    out: list[tuple[str, float]] = list(kept)
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and _emitted(out[j + 1][1]) == _emitted(out[i][1]):
            j += 1
        if j > i:  # a run of >=2 rows sharing one score
            if strict:
                # Every row is its own group; the list is already in final order within the run.
                starts = list(range(i, j + 1))
            else:
                # Group boundaries inside the run. Equal keys are adjacent because the list is
                # sorted, so rows the tie key cannot separate stay byte-identical.
                starts = [i]
                for k in range(i + 1, j + 1):
                    if tie_key(out[k][0]) != tie_key(out[k - 1][0]):
                        starts.append(k)
            groups = len(starts)
            if groups > 1:
                score = out[i][1]
                # The next strictly smaller score, or a full-width gap for the final run.
                nxt = out[j + 1][1] if j + 1 < len(out) else score - 1.0
                gap = score - nxt
                if span_full_gap:
                    usable = gap - eps
                    if usable <= 0.0:
                        if strict:
                            raise UnseparableTie(
                                f"{groups} rows tied at {score!r} with only {gap!r} to the next "
                                f"distinct score {nxt!r}: cannot separate at eps={eps!r}. Emitting "
                                "a tie would violate the no-ties rule silently."
                            )
                        i = j + 1
                        continue
                    step = usable / (groups - 1)
                else:
                    step = gap / (groups + 1)
                # A separation smaller than the emitted precision is not a separation: two rows a
                # step apart must differ by at least the emitted resolution or they print
                # identically and the tie survives into the file, invisible to every check that
                # reads floats rather than the emitted text.
                if strict and step < _T2_MIN_STEP:
                    raise UnseparableTie(
                        f"{groups} rows tied at {score!r}: step {step!r} is below the emitted "
                        f"precision {_T2_MIN_STEP!r}, so the rows would print identically. "
                        f"gap={gap!r} to next score {nxt!r}."
                    )
                for g, start in enumerate(starts):
                    if g == 0:
                        continue  # the best group keeps the original score untouched
                    end = starts[g + 1] if g + 1 < groups else j + 1
                    for k in range(start, end):
                        out[k] = (out[k][0], score - g * step)
        i = j + 1
    return out
