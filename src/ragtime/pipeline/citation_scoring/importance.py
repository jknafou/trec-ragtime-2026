"""The two importance factors: one pure reuse, one new per-answer judge.

``nugget_importance`` is `nugget.weight`, verbatim. ``claim_importance`` asks the model how much of
the nugget's question this ONE claim actually answers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragtime.common import Nugget
    from ragtime.serving.llm import LlmClient
    from ragtime.serving.schemas import CompiledSchema

__all__ = ["CLAIM_IMPORTANCE_PROMPT", "COVERAGE_TO_IMPORTANCE", "claim_importance", "nugget_importance"]

#: {full, partial, none} -> {1.0, 0.5, 0.0}. The same vocabulary the coverage audit uses, so the
#: project has one notion of how much of a nugget is answered rather than two that drift. The 0.5
#: for `partial` is a stated convention rather than a measurement, and because `citation_score` is
#: a product, mapping `none` to 0.0 makes an irrelevant claim a zero-scored citation.
COVERAGE_TO_IMPORTANCE: dict[str, float] = {"full": 1.0, "partial": 0.5, "none": 0.0}

#: One (question, claim) pair per call. The claim is quoted verbatim from what the model committed;
#: nothing is re-retrieved and no passage is shown, because this factor judges the claim against
#: the question, not the claim against its evidence. Whether the cited passage supports the claim
#: is a different question, and it is not one this score asks.
CLAIM_IMPORTANCE_PROMPT = (
    "You are judging how much of a specific question a single statement answers.\n\n"
    "QUESTION:\n{question}\n\n"
    "STATEMENT:\n{sentence}\n\n"
    "Answer with one word:\n"
    '- "full" if the statement answers the question completely.\n'
    '- "partial" if it answers part of the question, or answers it with less specificity '
    "than the question asks for.\n"
    '- "none" if it does not answer the question at all, even if it is on a related topic.\n\n'
    "Judge ONLY the statement as written. Do not infer facts it does not state."
)


def nugget_importance(nugget: Nugget) -> float:
    """Return ``nugget.weight`` verbatim; no second nugget-importance number is computed.

    A one-line function rather than an inlined attribute read: it is the named seam
    where "how important is this nugget" is answered, so if the source of ``weight`` ever changes
    upstream this file needs no edit and no second opinion appears anywhere in the tree. The
    contract is ``weight: float`` in [0, 1] either way.
    """
    return float(nugget.weight)


async def claim_importance(
    question: str,
    sentence: str,
    *,
    llm: LlmClient,
    schema: CompiledSchema,
    seed: int,
) -> float:
    """Issue one constrained-decoding call per (question, claim) pair, never per citation.

    An answer's several citations share this number: ``claim_importance`` is a property of the
    claim against the question, and the document a citation points at does not enter it. Calling
    per citation would multiply cost by the citation count and return the same value each time.

    Greedy decode and the run's own seed, the same convention as every other call site: this is a
    scoring pass, so two runs of the same cell must agree.
    """
    reply: dict[str, Any] = await llm.generate(
        schema,
        CLAIM_IMPORTANCE_PROMPT.format(question=question, sentence=sentence),
        seed,
    )
    coverage = str(reply["coverage"])
    if coverage not in COVERAGE_TO_IMPORTANCE:
        # The schema's enum makes this unreachable through a compliant server; if it ever fires,
        # the grammar was not applied and a silent default would invent a score.
        raise ValueError(
            f"claim_importance got coverage={coverage!r}, outside the schema enum "
            f"{sorted(COVERAGE_TO_IMPORTANCE)}: the guided decode did not constrain the reply"
        )
    return COVERAGE_TO_IMPORTANCE[coverage]
