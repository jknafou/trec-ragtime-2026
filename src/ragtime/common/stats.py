"""Counter bus: stages emit raw counters, monitoring rolls them up.

Stages call :meth:`Statistics.emit` with a ``metric_id`` and canonical slice keys
only; they never compute a reported metric themselves. The slice vocabulary is
fixed here and an unknown key raises ``ValueError``. The bus is instance-scoped,
so each run or test keeps its own counters and reads back an explicit snapshot.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .schemas import MetricRecord

__all__ = ["CANONICAL_SLICE_KEYS", "Statistics", "flush"]

# The only slice keys a stage may emit on. Adding a key is additive and cannot
# invalidate an existing emit, but a stage-specific key belongs here only when
# monitoring needs to roll it up.
CANONICAL_SLICE_KEYS: frozenset[str] = frozenset(
    {"variant", "seed", "lang", "round", "nugget", "leg", "reason"}
)

_SliceKey = tuple[tuple[str, Any], ...]


def _slice_key(slices: dict[str, Any]) -> _SliceKey:
    """Canonicalize slices to a sorted, hashable ``(key, value)`` tuple."""
    return tuple(sorted(slices.items()))


class Statistics:
    """An instance-scoped counter bus.

    ``emit`` accumulates ``value`` (default 1.0) into the cell keyed by
    ``(metric_id, sorted slices)``. Only :data:`CANONICAL_SLICE_KEYS` are allowed
    as slice names.
    """

    __slots__ = ("_counters",)

    def __init__(self) -> None:
        self._counters: dict[tuple[str, _SliceKey], float] = {}

    def emit(self, metric_id: str, value: float = 1.0, **slices: Any) -> None:
        """Add ``value`` to the counter cell ``(metric_id, slices)``.

        Raises ``ValueError`` if any slice key is outside the canonical vocabulary.
        """
        unknown = set(slices) - CANONICAL_SLICE_KEYS
        if unknown:
            raise ValueError(
                f"unknown slice key(s) {sorted(unknown)}; "
                f"allowed: {sorted(CANONICAL_SLICE_KEYS)}"
            )
        key = (metric_id, _slice_key(slices))
        self._counters[key] = self._counters.get(key, 0.0) + value

    def value(self, metric_id: str, **slices: Any) -> float:
        """Read back the exact-slice counter (0.0 if never emitted)."""
        return self._counters.get((metric_id, _slice_key(slices)), 0.0)

    def total(self, metric_id: str | None = None) -> float:
        """Sum counters for ``metric_id`` (or across all metrics if ``None``)."""
        return sum(
            v for (mid, _), v in self._counters.items() if metric_id is None or mid == metric_id
        )

    def records(self) -> Iterator[MetricRecord]:
        """Yield the current counter cells as :class:`MetricRecord`s (read-back)."""
        for (metric_id, slice_key), value in self._counters.items():
            yield MetricRecord(metric_id=metric_id, value=value, slices=slice_key)

    def __len__(self) -> int:
        return len(self._counters)


def flush(stats: Statistics, layout: Any) -> Any:
    """Write ``stats``' counters to ``layout.metrics()`` and return the path.

    A module function rather than a ``Statistics`` method, so the counter bus never
    has to import :class:`~ragtime.common.layout.Layout`. An empty file is written
    when a stage emitted nothing, because "ran and counted zero" and "never wrote"
    are different facts. ``skip_if_done=False``: a cell may legitimately be re-run
    in place, and the counters of the run that finished are the ones that belong on
    disk.
    """
    from .io import write_jsonl

    # `slices` is a tuple of (key, value) pairs in memory because it keys a dict;
    # on disk it becomes a mapping, which is the shape a roll-up groups on. Sorted
    # by construction, so identical counters serialize byte-identically.
    rows = [
        {"metric_id": r.metric_id, "value": float(r.value), "slices": dict(r.slices)}
        for r in stats.records()
    ]
    rows.sort(key=lambda r: (r["metric_id"], sorted(r["slices"].items())))
    path = layout.metrics()
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(path, rows, skip_if_done=False)
