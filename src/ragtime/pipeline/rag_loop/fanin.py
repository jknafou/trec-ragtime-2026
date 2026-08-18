"""Fan-in: cluster the committed claims into ``answers[]``, one entry per distinct value.

Two claims that say the same thing become one answer citing both documents; two claims that say
different things become two answers under the same nugget. That asymmetry is the design: corpus
disagreement is surfaced as a multi-valued answer rather than averaged, dropped or resolved by us.
The knowledge-conflict literature is explicit that a RAG system should expose conflicting evidence
instead of silently picking one, and a loop holding contradictory committed claims cannot abstain,
because abstain is unreachable once any claim is committed.

Clustering is deterministic string equality rather than embeddings: NFC, casefold and whitespace
collapse, and nothing else. Three reasons. A similarity threshold would be a hidden run parameter
that had to be fairness-shared, hashed and tuned, and at the wrong setting it would merge two
genuinely different values. The values being clustered are short atomic claims, where paraphrase
is rare and exact repetition across documents is the common case. And the terminal English-only
dedup at select-and-serialize is a separate policy that owns the harder job, so making this step
clever would duplicate that policy in a second place with a different answer.
"""

from __future__ import annotations

import unicodedata

from ragtime.common import nfc
from ragtime.common.schemas import Answer, Support

from .commit import CommittedClaim

__all__ = ["cluster_key", "fan_in", "is_contested"]


def cluster_key(text: str) -> str:
    """Return the identity two claims must share to be the same answer value.

    ``nfc`` first, as the project's one normalizer for text identity, then casefold, then collapse
    all Unicode whitespace to single spaces. Not ``nfkc``, which is the Task-1 budget
    normalizer: folding compatibility forms here would merge values a reader can tell apart.
    """
    folded = nfc(text).casefold()
    parts = [p for p in folded.split() if p]
    # `split()` already handles Unicode whitespace, but strip zero-width marks that survive it
    # and would otherwise make two visually identical claims cluster apart.
    cleaned = [
        "".join(ch for ch in p if unicodedata.category(ch) != "Cf")
        for p in parts
    ]
    return " ".join(p for p in cleaned if p)


def fan_in(
    claims: list[CommittedClaim] | tuple[CommittedClaim, ...],
) -> tuple[Answer, ...]:
    """Cluster committed claims into ``answers[]``, preserving first-seen order.

    Order is first-seen rather than sorted, so the output is deterministic given the action trail
    while still reflecting the order the model established its evidence in.

    ``references`` is filled with the supporting original doc-ids mapped to ``0.0``: the keys are
    this stage's output, since Task 3 keeps only the keys, and the scores belong to the post-hoc
    citation scorer, which ranks citations and never gates. A placeholder rather than an empty
    dict keeps the shape downstream expects, and 0.0 is the honest value for "not scored yet"
    rather than a confidence we computed.

    Clustering is on the claim's own ``answer``, the short value emitted with the claim. That is
    what makes a multi-valued answer work: each distinct value keeps its own short form for Task 3
    and its own report ``sentence`` for Task 1. Taking a single short value from ``submit_answer``
    instead would fill ``Answer.answer`` with a whole sentence exactly when the nugget carries
    more than one value.
    """
    order: list[str] = []
    grouped: dict[str, list[CommittedClaim]] = {}
    for c in claims:
        key = cluster_key(c.answer)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(c)

    out: list[Answer] = []
    for key in order:
        members = grouped[key]
        # De-duplicate references and support while keeping first-seen order: one document may
        # back the same value through several passages, and it must be cited once.
        refs: dict[str, float] = {}
        support: list[Support] = []
        seen_pids: set[str] = set()
        for m in members:
            refs.setdefault(m.doc_id, 0.0)
            if m.passage_id not in seen_pids:
                seen_pids.add(m.passage_id)
                support.append(Support(passage_id=m.passage_id, lang=m.lang))
        out.append(
            Answer(
                answer=members[0].answer,
                sentence=members[0].sentence,
                # The first member's span is the right one: `references` and `support` already
                # record every passage that backed this value, so the span identifies which text
                # was quoted to establish it.
                quoted_span=members[0].span,
                score=0.0,
                references=refs,
                support=tuple(support),
            )
        )
    return tuple(out)


def is_contested(answers: tuple[Answer, ...]) -> bool:
    """Return whether there are two or more distinct answer values. Telemetry only.

    It has no grammar or control effect, and is named as a predicate rather than folded into
    ``fan_in``'s return so that no caller can accidentally branch the loop on it: the moment
    contestedness changed behaviour, the system would be resolving conflicts instead of reporting
    them.
    """
    return len(answers) >= 2
