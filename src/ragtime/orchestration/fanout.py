"""Bounded async ``map`` that spawns the k RAG loops as coroutines.

SLURM is the coarse grain (one GPU cell) and asyncio the fine grain (the k loops
within a cell), never a SLURM job per loop. ``map`` runs the coroutines under an
``asyncio.Semaphore`` whose ceiling comes from ``pipeline.driver.resolve_ceiling``:
the configured ``rag_loop.fan_out.concurrency``, else the single vLLM instance's
KV headroom.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable

__all__ = ["map"]


async def map[T, R](
    coro: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    *,
    ceiling: int,
) -> list[R]:
    """Apply ``coro`` to every item, at most ``ceiling`` in flight, order-preserving.

    ``ceiling`` is passed in by the caller (see ``pipeline.driver.resolve_ceiling``) and is
    never cached here, so a changed ceiling takes effect on the next call.
    """
    if ceiling < 1:
        raise ValueError(f"ceiling must be >= 1; got {ceiling}")
    sem = asyncio.Semaphore(ceiling)

    async def _one(item: T) -> R:
        async with sem:
            return await coro(item)

    # gather preserves input order regardless of completion order.
    return list(await asyncio.gather(*[_one(it) for it in items]))
