"""The citation score itself: a product of independent factors, and nothing else.

``citation_score = nugget_importance x claim_importance``

A product rather than a weighted sum, because the two factors answer different questions: how much
the nugget matters to the request, and how much of the nugget this claim answers. A sum lets a high
value on one factor compensate for a near-zero on another, which is wrong here, since a claim that
fully answers an irrelevant nugget is not a good citation. A product makes either factor a veto,
which is what the hierarchy implies.

The factor names travel with the score rather than being left implicit, so a row on disk states
what its number is a product of instead of relying on the reader to know.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["FactorName", "citation_score"]

FactorName = Literal["nugget_importance", "claim_importance"]


def citation_score(
    nugget_importance: float,
    claim_importance: float,
) -> tuple[float, tuple[FactorName, ...]]:
    """Return the product, plus the names of the factors it is a product of.

    Both factors must lie in ``[0, 1]``. An out-of-range value raises rather than being clamped,
    because a clamp turns a caller's bug into a plausible number that then ranks a submission.
    """
    present: list[tuple[FactorName, float]] = [
        ("nugget_importance", nugget_importance),
        ("claim_importance", claim_importance),
    ]

    for name, value in present:
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(
                f"{name}={value!r} is outside [0, 1]; citation_score never clamps, because a "
                "clamped factor silently converts a caller's bug into a plausible score"
            )

    score = 1.0
    for _, value in present:
        score *= float(value)
    return score, tuple(name for name, _ in present)
