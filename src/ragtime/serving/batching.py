"""Length-bucket, token-budget batcher (pure, stdlib only).

``batch(items, tier)`` greedily buckets ``items`` so each batch stays under the
tier's per-batch token budget and item-count cap. Shared by the MT replicas and the
index-build encoders; the tier comes from the capacity profile rather than being
hardcoded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Batcher", "Tier"]


@dataclass(frozen=True, slots=True)
class Tier:
    """One batching tier: ``token_budget`` per batch and ``max_items`` count cap."""

    token_budget: int = 8192
    max_items: int = 64


class Batcher:
    """Length-bucket token-budget batcher. ``length_of`` maps an item -> token count."""

    __slots__ = ("length_of",)

    def __init__(self, length_of: Callable[[Any], int] | None = None) -> None:
        self.length_of = length_of or (lambda x: len(x) if hasattr(x, "__len__") else 1)

    def batch(self, items: Sequence[Any], tier: Tier | None = None) -> list[list[Any]]:
        """Bucket ``items`` into batches under ``tier``'s token + count budgets.

        Items are sorted by length so a batch's longest item bounds its padding;
        an item longer than the whole token budget still gets its own batch (never
        dropped).
        """
        t = tier or Tier()
        ordered = sorted(items, key=self.length_of)
        batches: list[list[Any]] = []
        cur: list[Any] = []
        cur_tokens = 0
        for it in ordered:
            n = self.length_of(it)
            if cur and (cur_tokens + n > t.token_budget or len(cur) >= t.max_items):
                batches.append(cur)
                cur, cur_tokens = [], 0
            cur.append(it)
            cur_tokens += n
        if cur:
            batches.append(cur)
        return batches
