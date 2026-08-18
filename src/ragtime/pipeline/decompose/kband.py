"""The deterministic seed-only breadth lookup: ``limit -> (k_min, k_max)``.

The topic's ``limit``, its Task-1 character budget, enters decompose at the seed only, as a coarse
breadth hint, via a deterministic table walk rather than LLM free choice. The table lives in the
config (``decomposition.k_band``, a list of ``[threshold, k_min, k_max]`` triples), so this module
encodes no band values of its own.

The band is a hint, not a clamp. It is rendered into the seed prompt and nothing downstream slices
or pads the model's output against it, because a hard clamp would silently drop real facets or
fabricate filler ones. The drafted bank is kept as-is, and only self-critique and dedup change its
size.

A pure function over a list of triples: no state, no cache, no IO.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragtime.common import Statistics

__all__ = ["KBAND_OUT_OF_RANGE", "limit_to_k"]

#: Counter emitted when ``limit`` exceeds every configured threshold and the widest
#: band is used instead (a topic outside the table the config anticipated).
KBAND_OUT_OF_RANGE = "decompose.kband_out_of_range"


def _triples(k_band: Sequence[Any]) -> list[tuple[int, int, int]]:
    """Validate and normalize the config table to ascending ``(threshold, k_min, k_max)``."""
    if not k_band:
        raise ValueError("decomposition.k_band is empty: the seed breadth hint has no table")
    out: list[tuple[int, int, int]] = []
    for row in k_band:
        if len(row) != 3:
            raise ValueError(
                f"decomposition.k_band row {list(row)!r} is not a "
                "[limit_threshold, k_min, k_max] triple"
            )
        threshold, k_min, k_max = (int(v) for v in row)
        if k_min > k_max:
            raise ValueError(f"decomposition.k_band row {list(row)!r} has k_min > k_max")
        out.append((threshold, k_min, k_max))
    thresholds = [t for t, _, _ in out]
    if thresholds != sorted(thresholds):
        raise ValueError(
            f"decomposition.k_band thresholds must ascend; got {thresholds}. "
            "First-match lookup over an unsorted table would pick a different band "
            "depending on authoring order."
        )
    return out


def limit_to_k(
    limit: int,
    k_band: Sequence[Any],
    *,
    stats: Statistics | None = None,
) -> tuple[int, int]:
    """Return ``(k_min, k_max)`` for ``limit``: the first ascending threshold with ``limit <= t``.

    Boundary semantics are inclusive on the threshold: with the shipped table
    ``[[2000,4,8],[5000,8,14],[10000,14,24]]``, ``limit=2000`` gives ``(4,8)``, ``limit=5000``
    gives ``(8,14)`` and ``limit=10000`` gives ``(14,24)``. A ``limit`` above every threshold
    clamps to the last and widest band and emits :data:`KBAND_OUT_OF_RANGE`, so a too-long topic
    still gets a sane breadth rather than a crash while the out-of-table case is never silent.

    ``stats`` is injected rather than a module-level singleton, so a caller or test observes an
    isolated counter set per call.
    """
    triples = _triples(k_band)
    for threshold, k_min, k_max in triples:
        if limit <= threshold:
            return k_min, k_max
    _, k_min, k_max = triples[-1]
    if stats is not None:
        stats.emit(KBAND_OUT_OF_RANGE, 1.0, round=0)
    return k_min, k_max
