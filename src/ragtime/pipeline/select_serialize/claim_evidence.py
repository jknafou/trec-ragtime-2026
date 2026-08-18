"""The reranker score a cited document carries: rule 4's tie-break signal, read off disk.

:mod:`.load` assembles the bank and ``records.loop_evidence`` joins the answers, but neither
carries ``support[]`` or ``retrieved[]``, so neither can say what the cross-encoder scored the
passage this claim quoted. That number is ``rag_loop/{nugget_id}.jsonl``'s ``retrieved[].score``,
and this module is the one place that reads it.

Cited passages rather than the whole pool, because rule 1 makes the unit the document that supports
a claim: its tie-break should be the cross-encoder's opinion of the passage the claim was grounded
on, not of the best passage the same document happened to place anywhere in a candidate pool. The
whole-pool score is used only as a fallback, for a cited passage absent from that round's pool.

Rounds strictly greater than ``through_round`` are dropped, matching ``records.loop_evidence``: a
cell interrupted between a fan and the audit that consumes it has loop records for a round with no
bank, and Task 2 must not rank on evidence the coverage audit never saw.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragtime.common import doc_id_of
from ragtime.common.io import read_jsonl
from ragtime.common.layout import Layout

__all__ = ["reranker_by_doc"]


def reranker_by_doc(
    layout: Layout,
    nugget_ids: Iterable[str],
    *,
    through_round: int,
) -> dict[str, float]:
    """Return ``doc_id -> reranker score`` over the passages the loops actually cited.

    Two reductions happen here and conflating them is the trap. Within one nugget the same passage
    can be retrieved in several rounds; that is one passage seen twice, so its score is the
    maximum. The second, semantic one is how a document's several distinct cited passages reduce to
    the single number rule 4 sorts on, and there is exactly one rule for it: the arithmetic mean of
    the log-probabilities.

    That is equivalently the geometric mean of ``P(yes)``, because ``Retrieved.score`` is the
    reranker's ``log_softmax`` output. It is direct arithmetic on the quantity the reranker emits,
    with no exponential round trip and no underflow on a strongly negative passage, and it is the
    conservative reading for a citation-support signal, since the worst cited passage drags the
    document down. A document cited at -0.1 and -3.0 collapses to -1.55 and therefore loses to one
    cited once at -0.5, where under a maximum it would have won. This is the rule every submitted
    Task-2 file was produced with, and :func:`.project.project` is the only caller.

    Not the arithmetic mean of the probabilities, which is dominated by the best
    passage; that is a defensible reading of average relevance that we do not take.

    A document with no cited passage carrying a score is absent from the mapping, and
    :func:`.task2_claims.claim_support_rows` sorts such a document last inside its tie group rather
    than dropping it, because rule 1 already declared it retrieved.
    """
    # Every value is kept, rather than a running best, because a mean needs the whole multiset.
    # Both maps accumulate across nuggets: a document cited by three claims contributes three
    # scores.
    cited_vals: dict[str, list[float]] = {}
    pool_vals: dict[str, list[float]] = {}
    for nugget_id in nugget_ids:
        path = layout.rag_loop(nugget_id)
        if not path.exists():
            continue
        pool: dict[str, float] = {}
        support_pids: set[str] = set()
        for row in read_jsonl(path):
            if int(row.get("round", 0)) > through_round:
                continue
            for hit in row.get("retrieved", ()) or ():
                pid = str(hit.get("passage_id", ""))
                if not pid:
                    continue
                score = float(hit.get("score", 0.0) or 0.0)
                # The same passage seen again in a later round is one passage, best observation.
                if pid not in pool or score > pool[pid]:
                    pool[pid] = score
            for answer in row.get("answers", ()) or ():
                for sup in answer.get("support", ()) or ():
                    pid = str(sup.get("passage_id", ""))
                    if pid:
                        support_pids.add(pid)
        for pid, score in pool.items():
            pool_vals.setdefault(doc_id_of(pid), []).append(score)
        for pid in support_pids:
            score = pool.get(pid)
            if score is None:
                continue
            cited_vals.setdefault(doc_id_of(pid), []).append(score)

    def _reduce(values: list[float]) -> float:
        return sum(values) / float(len(values))

    # The cited-passage score wins; the pool score only fills a document whose cited passage was
    # not in that round's pool (so rule 4 still has something real to act on).
    out = {doc: _reduce(vals) for doc, vals in pool_vals.items()}
    out.update({doc: _reduce(vals) for doc, vals in cited_vals.items()})
    return out
