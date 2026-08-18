"""Rank fusion: reciprocal-rank fusion and max-passage collapse.

``rrf`` is used for index-time leg fusion, loop-search leg fusion and Task-2
nugget fusion. ``maxp_collapse`` projects ranked passages onto documents for the
Task-2 doc-level run. The default ``k=60`` lives here so no call site repeats it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = ["maxp_collapse", "rrf"]


def _as_id(entry: Any) -> str:
    """Extract the id from a pool entry (plain id, ``(id, score)``, or a record)."""
    if isinstance(entry, str):
        return entry
    pid = getattr(entry, "passage_id", None)
    if pid is not None:
        return pid
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        return entry[0]
    raise TypeError(f"cannot extract an id from pool entry: {entry!r}")


def rrf(pools: Iterable[Sequence[Any]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion over ranked ``pools``.

    Each pool is a ranked sequence (best first); an entry may be a plain id, an
    ``(id, score)`` pair, or a record exposing ``.passage_id``. The fused score of
    an id is ``sum(1 / (k + rank))`` over the pools, with a 1-based rank. Returns
    ``[(id, score)]`` sorted by descending score, tie-broken on ``(-score, id)``.
    """
    scores: dict[str, float] = {}
    for pool in pools:
        for rank, entry in enumerate(pool, start=1):
            pid = _as_id(entry)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def maxp_collapse(
    ranked: Iterable[tuple[str, float]],
    doc_id_of: Callable[[str], str],
) -> list[tuple[str, float]]:
    """Collapse ranked passages to docs, keeping the best score per doc-id.

    Maps each ``passage_id`` through ``doc_id_of`` and keeps the maximum score
    seen for each doc. Returns ``[(doc_id, score)]`` sorted by descending score,
    tie-broken on ``(-score, doc_id)``.
    """
    best: dict[str, float] = {}
    for passage_id, score in ranked:
        doc = doc_id_of(passage_id)
        if doc not in best or score > best[doc]:
            best[doc] = score
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
