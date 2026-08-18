"""The novelty-saturation stop predicate.

``saturated(novelty_history, min_new=..., low_streak=...)`` is the coverage loop's
termination test: *fewer than* ``min_new`` net-new nuggets for ``low_streak``
consecutive rounds means the bank has stopped growing and the loop stops before
``R_max`` (which is the safety net, not the plan).

A pure function over a plain tuple of ints: no bank, no round object, no config object. That shape
lets ``round_loop`` call it every round without importing any decompose internals beyond this name,
and lets a test drive every boundary case from hand-built sequences with no fixtures.

``min_new`` and ``low_streak`` are keyword arguments read from ``decomposition.min_new`` and
``decomposition.low_streak`` by the caller, never module constants, so the config stays the record.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["saturated"]


def saturated(novelty_history: Sequence[int], *, min_new: int, low_streak: int) -> bool:
    """True iff the last ``low_streak`` entries of ``novelty_history`` are all ``<= min_new``.

    ``novelty_history`` is the net-new-nuggets-per-round series in round order (round 1
    first; round 0 mints the whole bank and is not a novelty observation).

    Boundary semantics:

    * ``<=`` rather than ``<``, so a round that produced exactly ``min_new`` counts as low. At the
      shipped ``min_new=1``, one lonely new nugget is not growth.
    * A history shorter than ``low_streak`` can never be saturated: the loop has not yet observed
      enough consecutive low rounds to conclude anything.
    * ``low_streak <= 0`` raises rather than degenerating into "always saturated", which would
      stop the loop after round 0.
    """
    if low_streak <= 0:
        raise ValueError(
            f"decomposition.low_streak must be >= 1; got {low_streak} "
            "(a zero streak would saturate before any round ran)"
        )
    if len(novelty_history) < low_streak:
        return False
    return all(int(n) <= min_new for n in novelty_history[-low_streak:])
