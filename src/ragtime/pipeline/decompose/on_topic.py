"""The on-topic grounding gate: nugget admission, never an in-loop evidence drop.

Under ``grounding_gate: on_topic``, the config's only supported value, a newly proposed nugget is
admitted only if it is a genuine facet of ``problem_statement``. It judges subject, not support:
a support-based drop would reject a nugget the corpus has not answered yet, whereas this gate
rejects it for being about something else. Accordingly this module imports nothing from the
citation scorer, and a test asserts that structurally, on the import graph.

The gate is invoked only from the ``round >= 1`` branch, where it gates the audit's ``delta.add``;
round 0 never calls it and passes ``passages=None`` by construction.

The LLM comes in as an injected ``ClientBundle``, the one shared vLLM; this module builds no
client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .prompts import SEED_SYSTEM, decoding_kwargs, on_topic_prompt

if TYPE_CHECKING:
    from ragtime.common import Statistics

__all__ = ["ON_TOPIC_ADMITTED", "ON_TOPIC_REJECTED", "on_topic"]

ON_TOPIC_ADMITTED = "decompose.on_topic_admitted"
#: The drift-guard rate's numerator: proposed nuggets the gate discarded.
ON_TOPIC_REJECTED = "decompose.on_topic_rejected"


async def on_topic(
    question: str,
    problem_statement: str,
    passages: Any | None = None,
    *,
    cfg: Any,
    clients: Any,
    seed: int,
    stats: Statistics | None = None,
    round: int | None = None,
    title: str = "",
) -> bool:
    """Return whether ``question`` is a genuine facet of ``problem_statement``.

    There is no ``background`` parameter: ``background`` biases which facets matter,
    it does not define what is on topic, and admitting on a persona would let the gate reject a
    required facet the persona happens not to emphasise.

    ``title`` is the one request field that is threaded, because a persona is orthogonal to the
    subject while a title names it, and drifting to an adjacent subject is exactly the rejection
    this gate exists to make.

    ``passages`` is the round-1-and-later evidence context and is ``None`` at round 0. When
    present it is offered as context for judging topicality only, never as evidence the nugget
    must be supported by.

    Issues one constrained-decoding call against the ``on_topic`` schema
    (``{rationale, on_topic}``) through the injected ``clients.llm`` singleton.
    """
    obj = await clients.llm.generate(
        clients.schemas.on_topic,
        on_topic_prompt(question, problem_statement, passages, title=title),
        seed,
        system=SEED_SYSTEM,
        **decoding_kwargs(cfg),
    )
    verdict = bool(obj["on_topic"])
    if stats is not None:
        slices = {} if round is None else {"round": round}
        stats.emit(ON_TOPIC_ADMITTED if verdict else ON_TOPIC_REJECTED, 1.0, **slices)
    return verdict
