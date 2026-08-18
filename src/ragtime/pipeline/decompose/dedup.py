"""The paraphrase and embedding merge over the growing nugget bank.

``dedup = {llm_paraphrase_merge + embedding_merge@cos_cutoff}`` is a two-stage cascade, and the
division of labour between the stages is the design:

* Cosine is a generous recall pre-filter, not the arbiter. Published optimal paraphrase cutoffs
  span roughly 0.33 to 0.87 across studies, so no single threshold is defensible as a final
  decision without labelled data, which this project does not have. A lower cutoff only costs
  extra confirm calls, while a higher one silently misses paraphrases with no second chance. The
  threshold is not written down in this module at all: it arrives as an argument from
  ``decomposition.dedup.cosine_cutoff``.
* The LLM gate is the precision decision, using the existing ``dedup`` schema, where ``duplicate``
  holds only if ``paraphrase_match`` and ``entity_match`` both do.

The merge is greedy, incremental and append-only, and never un-merges. Each candidate is compared
only against the survivors accumulated so far, in bank order, so the older nugget always wins and
an id, once merged away, never resurfaces on a different question. That monotonicity matters
because committed answers reference ``nugget_id``. It also keeps gate volume in the tens per round
rather than quadratic in bank size, and those calls compete for the one shared vLLM against the k
RAG loops and the scorer.

Three import constraints, all load-bearing:

* No ``sentence_transformers``, not even ``util.cos_sim``. ``serving.embed`` already returns
  normalized vectors, so cosine here is a plain dot product, and ``sentence-transformers`` lives in
  the Linux-only ``heavy`` extra, so a direct import would make this module unimportable on the
  macOS dev box. Its ``paraphrase_mining`` and ``community_detection`` helpers were also rejected
  on their own terms: they target very large collections and re-cluster globally on every call, so
  a nugget's cluster membership could flip between rounds.
* No ``numpy``. It is pulled only by the ``chunk`` and ``heavy`` extras, so a bare ``uv sync``
  environment has none, and importing it here would reproduce the same platform split one layer
  down. The dot product below is stdlib, and affordable by construction: the merge is greedy over
  a bank of tens to a few hundred, not over the corpus.
* No ``serving.registry``. The embedder and the confirm gate arrive as injected callables sliced
  from the one ``ClientBundle``, so this module structurally cannot construct a second client.

This is a policy over two existing interfaces, and is distinct from the terminal
``select_serialize.dedup_english``, which is deterministic, English-only, and runs over answers
rather than over the growing question bank.
"""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ragtime.common import Nugget

if TYPE_CHECKING:
    from ragtime.common import Statistics

__all__ = ["DEDUP_CANDIDATES", "DEDUP_MERGED", "cosine", "dedup_nuggets"]

#: Pairs the cosine pre-filter proposed, which is the gate's call volume when it is on.
DEDUP_CANDIDATES = "decompose.dedup_candidates"
#: Nuggets actually merged away: the "removed as redundant" statistic.
DEDUP_MERGED = "decompose.dedup_merged"


def _rows(vectors: Any) -> list[tuple[float, ...]]:
    """Normalize whatever ``embed`` returned into a list of plain float tuples.

    Accepts a list of sequences or anything array-like that iterates rows, without importing
    numpy, so this module stays importable in the base environment on both platforms.

    ``serving.embed(texts, mode="dense")`` already normalizes, so no re-normalization is applied
    here: doing it would paper over an embedder that stopped normalizing. A zero row is left as-is
    and simply scores 0 against everything.
    """
    rows: list[tuple[float, ...]] = []
    for row in vectors:
        try:
            rows.append(tuple(float(x) for x in row))
        except TypeError:  # a scalar row means embed() did not return one vector per text
            raise ValueError(
                "embed() must return one vector per text (a 2-D result); got a scalar row"
            ) from None
    widths = {len(r) for r in rows}
    if len(widths) > 1:
        raise ValueError(f"embed() returned ragged vectors (widths {sorted(widths)})")
    return rows


async def dedup_nuggets(
    bank: tuple[Nugget, ...],
    *,
    cosine_cutoff: float,
    llm_paraphrase_merge: bool,
    embed: Callable[[list[str]], Any],
    llm_confirm: Callable[[str, str], Awaitable[bool]] | None = None,
    stats: Statistics | None = None,
) -> tuple[Nugget, ...]:
    """Merge duplicate nuggets out of ``bank``, keeping the earliest of each cluster.

    ``embed`` is the injected embedding callable, bound at the real call site to
    ``clients.index_dense.embed`` so dedup shares retrieval's embedding space, and called once for
    the whole bank rather than per pair. ``llm_confirm(kept_question, candidate_question)`` is the
    injected gate; it is awaited only for pairs the cosine pre-filter proposed, and only when
    ``llm_paraphrase_merge`` is on. With the gate off the cascade degenerates to cosine-only, which
    is what ``decomposition.dedup.llm_paraphrase_merge: false`` means, rather than no dedup.

    When a candidate merges into a survivor, the survivor keeps its identity and its scores and
    inherits the candidate's ``answers`` and ``retrieved``: the merged nugget's id disappears, so
    any evidence attached to it would otherwise be lost.

    A coroutine because the confirm gate goes through the async ``serving.llm.guided_json``; dedup
    defines no concurrency of its own.
    """
    if len(bank) < 2:
        return tuple(bank)

    vectors = _rows(embed([n.question for n in bank]))
    if len(vectors) != len(bank):
        raise ValueError(
            f"embed() returned {len(vectors)} vectors for {len(bank)} nuggets; "
            "the dedup pre-filter cannot align a partial embedding batch"
        )

    survivors: list[Nugget] = [bank[0]]
    survivor_rows: list[int] = [0]
    merged = 0
    candidates = 0

    for i in range(1, len(bank)):
        candidate = bank[i]
        # Most similar survivor first, so the closest paraphrase is offered to the gate before a
        # merely adjacent one. Ties break on the earlier survivor, keeping the older id, because
        # the sort is stable over the append-ordered survivor list.
        scored = sorted(
            ((cosine(vectors[r], vectors[i]), pos) for pos, r in enumerate(survivor_rows)),
            key=lambda t: -t[0],
        )
        target: int | None = None
        for sim, pos in scored:
            if sim < cosine_cutoff:
                break
            candidates += 1
            kept = survivors[pos]
            if llm_paraphrase_merge:
                if llm_confirm is None:
                    raise ValueError(
                        "decomposition.dedup.llm_paraphrase_merge is true but no llm_confirm "
                        "gate was injected; the cosine pre-filter is not an arbiter and must "
                        "not decide a merge on its own"
                    )
                if not await llm_confirm(kept.question, candidate.question):
                    continue
            target = pos
            break
        if target is None:
            survivors.append(candidate)
            survivor_rows.append(i)
            continue
        survivors[target] = _absorb(survivors[target], candidate)
        merged += 1

    if stats is not None:
        if candidates:
            stats.emit(DEDUP_CANDIDATES, float(candidates))
        if merged:
            stats.emit(DEDUP_MERGED, float(merged))
    return tuple(survivors)


def _absorb(kept: Nugget, merged_away: Nugget) -> Nugget:
    """Fold ``merged_away``'s evidence into ``kept`` without touching its identity."""
    if not merged_away.answers and not merged_away.retrieved:
        return kept
    return replace(
        kept,
        answers=kept.answers + tuple(merged_away.answers),
        retrieved=kept.retrieved + tuple(merged_away.retrieved),
    )


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity of two already-normalized vectors, a plain dot product.

    Stdlib only, and the one place the pre-filter's arithmetic lives, so the no-numpy and
    no-sentence-transformers constraints stay checkable by reading a few lines.
    """
    return float(sum(map(operator.mul, a, b)))
