"""The fanout.map concurrency ceiling.

Invariant 3: the k-loop concurrency never exceeds the ceiling, the ceiling is read from
the caller as-is (never cached or re-derived), and results come back in input order
despite out-of-order completion.
"""

from __future__ import annotations

import asyncio

import pytest

from ragtime.orchestration import fanout

pytestmark = pytest.mark.small


class _Tracker:
    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def coro(self, item: int) -> int:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)  # hold the slot so real overlap occurs
        self.in_flight -= 1
        return item * 10


def test_never_exceeds_and_reaches_ceiling(stub_node) -> None:
    """max in-flight <= C AND == C is reached (real concurrency, not accidental serial)."""
    node = stub_node(3)
    tracker = _Tracker()
    items = list(range(20))  # N=20 > C=3

    results = asyncio.run(fanout.map(tracker.coro, items, ceiling=node.loop_ceiling))

    assert tracker.max_in_flight <= 3
    assert tracker.max_in_flight == 3  # the ceiling is actually saturated
    assert results == [i * 10 for i in items]  # order-preserving


def test_ceiling_is_read_not_cached_between_calls(stub_node) -> None:
    """Mutating loop_ceiling between two map calls changes the observed cap (no caching)."""
    node = stub_node(2)
    items = list(range(12))

    t1 = _Tracker()
    asyncio.run(fanout.map(t1.coro, items, ceiling=node.loop_ceiling))
    assert t1.max_in_flight == 2

    node.loop_ceiling = 5  # serving may hand over a different runtime headroom
    t2 = _Tracker()
    asyncio.run(fanout.map(t2.coro, items, ceiling=node.loop_ceiling))
    assert t2.max_in_flight == 5


def test_results_ordered_despite_out_of_order_completion(stub_node) -> None:
    """Coroutines that finish in reverse order still return aligned to input order."""
    node = stub_node(10)

    async def _coro(item: int) -> int:
        # earlier items sleep longer -> complete later than later items.
        await asyncio.sleep(0.01 * (10 - item))
        return item * 10

    items = list(range(10))
    results = asyncio.run(fanout.map(_coro, items, ceiling=node.loop_ceiling))
    assert results == [i * 10 for i in items]


def test_ceiling_below_one_is_rejected(stub_node) -> None:
    async def _coro(item: int) -> int:
        return item

    with pytest.raises(ValueError, match="ceiling"):
        asyncio.run(fanout.map(_coro, [1, 2], ceiling=0))
