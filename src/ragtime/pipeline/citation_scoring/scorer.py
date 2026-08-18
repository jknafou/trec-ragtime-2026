"""The per-topic driver: fill in every citation's score, and write them where the serializer looks.

Runs after the loops, over one topic's already-committed ``answers[]``, and fills values only: for
every existing key of ``answer.references``, which the RAG loop's fan-in wrote as
``{doc_id: 0.0}``, it computes a score. It never adds a citation and never removes one, because
the set of cited documents is the loop's decision and evidence of what the model read. A scorer
that could add a key would be inventing a citation, and one that could drop a key would be
re-running the commit gate under another name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragtime.common import doc_id_of, write_jsonl
from ragtime.pipeline.decompose.bank import last_complete_round, read_bank
# `records` owns loop records, since it writes them, so reading them back lives there too. This
# package must not import `select_serialize`, which is downstream and consumes this one's output.
from ragtime.pipeline.records import answer_key, loop_evidence
from ragtime.serving.schemas import compile_schemas

from .aggregate import citation_score
from .importance import claim_importance, nugget_importance

if TYPE_CHECKING:
    from pathlib import Path

    from ragtime.common import Layout, Statistics
    from ragtime.serving.llm import LlmClient

__all__ = ["SCORES_FILE", "score_citations"]

#: The one file `select_serialize.load.citation_scores` opens. The name is duplicated there as
#: `load.SCORES_FILE` and the two are kept equal by hand, because a scorer writing `scores.jsonl`
#: while the reader opens `score.jsonl` fails as "the scorer never ran", which is a silent
#: downgrade to weight-only ordering rather than an error.
SCORES_FILE = "scores.jsonl"

#: Counters only; `monitoring` owns every rollup.
CITATIONS_SCORED = "citation_scoring.citations_scored"
CLAIM_JUDGE_CALLS = "citation_scoring.claim_importance_calls"
#: Always 0.0: the citation score has two factors, and no probability-of-support term feeds it.
#: The counter is emitted all the same because it is a row in every `metrics/counters.jsonl` the
#: scorer writes, so a re-scored topic's counters stay byte-identical to the ones already filed;
#: dropping it would silently change an artifact of record to make the code look tidier.
SUPPORT_PRESENT = "citation_scoring.probability_support_present"


async def score_citations(
    cfg: Any,
    *,
    layout: Layout,
    llm: LlmClient,
    seed: int,
    stats: Statistics | None = None,
) -> Path:
    """Score one topic's citations, write ``citation_scores/scores.jsonl``, and return its path.

    The grain is per topic, matching ``driver.solve_topic``: a cell holds many topics but
    ``Layout.citation_scores()`` is a per-topic directory, so a single call cannot own both
    granularities without silently writing one topic's scores. Fanning over a cell stays the
    driver's job. ``seed`` is explicit because every constrained-decoding call in this project is
    seeded and a scoring pass must be reproducible.

    One ``claim_importance`` call per answer, broadcast to that answer's citations: the factor
    judges the claim against the nugget's question and is shown no passage at all, so which
    document a citation points at cannot enter it and calling per citation would multiply cost for
    an identical value.

    Every row is the product ``nugget_importance x claim_importance`` and names both in its
    ``factors_used``.
    """
    schema = compile_schemas().claim_importance

    through = last_complete_round(layout)
    bank = read_bank(layout.decompose_round(through))
    rows: list[dict[str, Any]] = []
    scored = judged = 0

    for nugget in bank:
        weight = nugget_importance(nugget)
        # The citations are not in the bank: `coverage_audit._merge_answers` rebuilds every bank
        # answer from an `AnswerView` that carries no `references`, so `nugget.answers[i]
        # .references` is always empty here. The real citations live in the loop records, and
        # `loop_evidence` is the one join that reads them, with the same key, the same
        # max-per-doc merge rule and the same doc-id normalization the serializer uses.
        evidence = loop_evidence(layout, nugget.nugget_id, through_round=through)
        for answer in nugget.answers:
            cited_docs = (evidence.get(answer_key(answer)) or {}).get("references") or {}
            if not cited_docs:
                continue  # nothing to score; an uncited answer has no citation to rank
            importance = await claim_importance(
                nugget.question, answer.sentence, llm=llm, schema=schema, seed=seed
            )
            judged += 1
            for cited in cited_docs:
                score, factors = citation_score(weight, importance)
                rows.append(
                    {
                        "nugget_id": nugget.nugget_id,
                        # `answer` and `quoted_span` are the join key `load.answer_key` rebuilds.
                        # Writing the two fields rather than the joined string keeps the file
                        # readable and lets the reader own the key's spelling; two spellings of
                        # that key would silently score nothing.
                        "answer": answer.answer,
                        "quoted_span": answer.quoted_span,
                        # Always the original doc-id, whatever was read or searched.
                        "doc_id": doc_id_of(str(cited)),
                        "score": score,
                        "factors_used": list(factors),
                    }
                )
                scored += 1

    if stats is not None:
        stats.emit(CITATIONS_SCORED, float(scored))
        stats.emit(CLAIM_JUDGE_CALLS, float(judged))
        stats.emit(SUPPORT_PRESENT, 0.0)

    directory = layout.citation_scores()
    directory.mkdir(parents=True, exist_ok=True)
    return write_jsonl(directory / SCORES_FILE, rows)
