"""Derive ``vital`` from ``weight``, then dedup. No LLM call of its own.

The closing step of every round is ``weight_and_dedup(bank + new)``. It does two things, neither of
them a generation:

* ``vital = weight >= decomposition.vital_cutoff``, a pure derivation. Decompose owns both numbers:
  ``weight`` is the continuous nugget-selection score select-and-serialize budget-fits on, while
  ``vital`` is the AutoNuggetizer vital/okay flag, monitoring-only and never a submission field.
* ``dedup_nuggets``, the paraphrase-plus-embedding merge, injected as ``dedup_fn`` so this module
  imports no client and no schema.

``weight`` itself is not scored here. It arrives already attached, emitted by
``grow_nuggets._self_critique`` in the same ``{rationale, nuggets}`` response that carries the
dedup and coverage revisions, because weight is scored strictly after self-critique: scoring the
draft first would orphan the score on nuggets self-critique still merges or drops. Folding it in
costs no extra constrained-decoding calls.

Out-of-range scores are clamped rather than propagated. The schema types ``weight`` as a plain
number, since a JSON-Schema ``minimum``/``maximum`` is not enforced by grammar-constrained
decoding, so a 1.05 would burn the whole retry budget and then raise instead of degrading. The
clamp is counted, so a model that systematically over-scores is visible rather than silently
normalized.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ragtime.common import Nugget

if TYPE_CHECKING:
    from ragtime.common import Statistics

__all__ = ["WEIGHT_OUT_OF_RANGE", "clamp_weight", "weight_and_dedup"]

#: A ``weight`` the model emitted outside [0, 1] (or non-finite) and we clamped.
WEIGHT_OUT_OF_RANGE = "decompose.weight_out_of_range"

WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0


def clamp_weight(weight: float) -> tuple[float, bool]:
    """Return ``(clamped, was_out_of_range)`` for a model-emitted ``weight``.

    A non-finite score, which ``common.io.ensure_finite`` would later refuse to serialize anyway,
    is treated as out of range and clamped to the floor, so a single bad token cannot poison the
    whole bank artifact.
    """
    value = float(weight)
    if not math.isfinite(value):
        return WEIGHT_MIN, True
    if value < WEIGHT_MIN:
        return WEIGHT_MIN, True
    if value > WEIGHT_MAX:
        return WEIGHT_MAX, True
    return value, False


async def weight_and_dedup(
    bank: tuple[Nugget, ...],
    *,
    cfg: Any,
    dedup_fn: Callable[[tuple[Nugget, ...]], Awaitable[tuple[Nugget, ...]]],
    stats: Statistics | None = None,
) -> tuple[Nugget, ...]:
    """Clamp ``weight``, derive ``vital``, then run ``dedup_fn`` over the bank.

    ``vital_cutoff`` is read from ``decomposition.vital_cutoff`` and has no module default, so the
    shipped 0.5 midpoint is a recorded config choice rather than a code constant.

    A coroutine only because ``dedup_fn`` is, its confirm gate going through the shared vLLM; the
    weighting half issues no call at all.
    """
    cutoff = float(cfg.blocks["decomposition"]["vital_cutoff"])
    scored: list[Nugget] = []
    out_of_range = 0
    for n in bank:
        weight, bad = clamp_weight(n.weight)
        out_of_range += int(bad)
        scored.append(replace(n, weight=weight, vital=weight >= cutoff))
    if stats is not None and out_of_range:
        stats.emit(WEIGHT_OUT_OF_RANGE, float(out_of_range))
    return await dedup_fn(tuple(scored))
