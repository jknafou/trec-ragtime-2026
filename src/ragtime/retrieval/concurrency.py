"""How wide the query path fans, at both levels.

A single ``retrieve`` call has two parallel axes, and they nest:

* the outer axis, spent in :func:`~ragtime.retrieval.service._pools`, one unit per
  ``(query string, leg, language cell)`` -- three legs by four cells for one query string on the
  shipped geometry;
* the inner axis, owned by ``preprocess.index`` and spent inside ``query_lang_leg``, one unit per
  index part of one cell, width ``execution.query_part_workers``.

Composed naively the two multiply, on top of each engine's own thread pool (FAISS OpenMP,
Seismic Rayon, PyLate/PLAID torch intra-op), on a node that is also hosting the shared vLLM. So
both widths are decided together, here, once per bring-up.

The policy: when the outer fan is active and the config names no inner width, the inner width is
set to 1, spending the whole thread budget on the outer axis. The two axes are close substitutes
and running both measured about 4% faster for four times the threads. An operator who names
``execution.query_part_workers`` is obeyed exactly, and the resulting product is logged
(``concurrency_bounded`` in ``retrieval.context.up``) so an oversubscribed pairing is visible.

Per-leg cost on the built index is very unbalanced: the late-interaction leg dominates a query's
search time, so this fan is effectively a fan over that leg's language cells and wall time
plateaus at the number of cells rather than at legs times cells.

The leaf lives in ``execution`` beside ``execution.query_part_workers``, because it decides how
the same work is fanned and never what the work produces: the fan collects in submission order,
so hits and scores are identical at any width. It is not in ``index_build.config`` (which is
hashed into the index path, where a latency retune would orphan a built index) and not in
``retrieval`` (whose leaves are search semantics).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ragtime.config import ConfigError

__all__ = [
    "LEG_WORKERS_CAP",
    "QUERY_LEG_WORKERS_KEY",
    "QUERY_PART_WORKERS_KEY",
    "QueryConcurrency",
    "default_leg_workers",
    "plan_query_concurrency",
    "resolve_leg_workers",
]

#: The ``execution`` key naming the width of the per-(leg, cell) query fan, the outer axis.
QUERY_LEG_WORKERS_KEY = "query_leg_workers"

#: The inner axis's key. Read here only to answer "did the operator name one?"; what a value
#: means belongs to ``preprocess.index.resolve_query_workers``.
QUERY_PART_WORKERS_KEY = "query_part_workers"

#: Ceiling on the auto-resolved outer width: three legs by four language cells is the most units
#: this axis can have for one query string. Several query strings lengthen the task list rather
#: than the useful concurrency. An explicit config value is honored above this cap. Wall time in
#: practice plateaus at the number of language cells, so this is a ceiling, not a target.
LEG_WORKERS_CAP = 12


@dataclass(frozen=True, slots=True)
class QueryConcurrency:
    """The two fan widths, decided together.

    ``leg_workers`` is resolved (an int >= 1, where 1 means no pool at all). ``part_workers`` is
    the raw value to forward to ``IndexCtx.query_workers``: ``None`` means "absent, let
    ``preprocess.index`` derive it", which is a different statement from ``1`` and stays
    distinguishable down to its single owner.
    """

    leg_workers: int
    part_workers: Any

    @property
    def bounded(self) -> bool:
        """True when this pairing cannot oversubscribe: at most one axis is fanned."""
        return self.leg_workers <= 1 or self.part_workers == 1


def default_leg_workers() -> int:
    """Outer width when the config names none: the task's CPUs, capped at :data:`LEG_WORKERS_CAP`.

    ``SLURM_CPUS_PER_TASK`` is preferred over ``os.cpu_count`` because it reports the cores this
    task was allocated rather than the whole node. Unlike ``index.default_query_workers`` the
    result is not halved: that function halves to leave room for the fan below it, and under this
    module's policy the inner width is 1 whenever this one is greater than 1.
    """
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        cpus = int(allocated) if allocated else (os.cpu_count() or 1)
    except ValueError:  # pragma: no cover - a non-numeric SLURM env is not worth a raise
        cpus = os.cpu_count() or 1
    return max(1, min(LEG_WORKERS_CAP, cpus))


def resolve_leg_workers(value: Any = None) -> int:
    """Turn the raw ``execution.query_leg_workers`` value into a usable width.

    ``None`` (absent from the config) means :func:`default_leg_workers`; anything else is the
    operator's explicit choice, floored at 1 so a ``0`` degrades to the sequential path rather
    than to a pool that can never run. Same three-case contract as
    ``preprocess.index.resolve_query_workers``, so the two leaves behave identically.
    """
    if value is None:
        return default_leg_workers()
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        raise ConfigError(
            f"execution.{QUERY_LEG_WORKERS_KEY}={value!r}: the per-(leg, cell) query fan's "
            "width must be an integer (absent = derive it from the task's CPU allocation)"
        ) from None


def _execution(cfg: Any) -> dict[str, Any]:
    return dict((getattr(cfg, "blocks", None) or {}).get("execution") or {})


def plan_query_concurrency(cfg: Any) -> QueryConcurrency:
    """Both fan widths for one run config, which is where the nesting is bounded.

    An absent inner width becomes ``1`` rather than its own default whenever the outer fan is
    active; that substitution is the bound described in the module docstring.
    """
    block = _execution(cfg)
    leg = resolve_leg_workers(block.get(QUERY_LEG_WORKERS_KEY))
    part = block.get(QUERY_PART_WORKERS_KEY)
    if part is None and leg > 1:
        part = 1
    return QueryConcurrency(leg_workers=leg, part_workers=part)
