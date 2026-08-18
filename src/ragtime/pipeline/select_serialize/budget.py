"""The Task-1 character budget: one place, and it is NFKC, never NFC.

The rule is the validator's own,
``len(unicodedata.normalize('NFKC', " ".join(s['text'] for s in r['responses'])))``, with four
load-bearing consequences:

* NFKC, not NFC. Three thousand instances of the ``fi`` ligature are three thousand characters
  under NFC and six thousand under NFKC, so an NFC implementation passes a report the validator
  rejects. ``common.nfc`` is the commit path, the verbatim span gate, and conflating the two is a
  standing invariant violation, which is why this module imports only ``nfkc_len``.
* Per whole report, not per sentence: every ``responses[].text`` is joined first.
* The join costs exactly one character per gap, so summing bare ``nfkc_len(sentence)``
  under-counts by the number of gaps and can push a report past ``limit``.
* The comparison is inclusive.

:func:`trim_to_limit` exists on top of :func:`fits` because NFKC normalization is not additive
across a join: a sentence ending in a base character and one starting with a combining mark can
normalize differently joined than apart. The incremental accounting in :func:`fits` is what lets
the grounding gate free a rejected sentence's budget in-loop, but the number that decides
acceptance is :func:`joined_len` over the final list. So the selector admits incrementally and
then re-measures the whole report once, dropping from the tail while it does not fit. On ordinary
text the trim is a no-op.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragtime.common import nfkc_len

__all__ = ["JOINER", "fits", "joined_len", "trim_to_limit"]

#: The scorer joins report sentences with exactly this string.
JOINER = " "


def joined_len(texts: Sequence[str]) -> int:
    """Return ``nfkc_len(" ".join(texts))``, the measure the validator and scorer both apply."""
    return nfkc_len(JOINER.join(texts))


def fits(used: int, sentence: str, n_admitted: int, limit: int) -> int | None:
    """Return the cost of admitting ``sentence``, or ``None`` if it does not fit.

    ``used`` is the running joined length of the ``n_admitted`` sentences already admitted, and
    the joiner is charged from the second sentence on. Returns the increment to add to ``used``,
    so a caller cannot admit without accounting, rather than a bare bool.
    """
    add = nfkc_len(sentence) + (len(JOINER) if n_admitted else 0)
    if used + add > limit:
        return None
    return add


def trim_to_limit(texts: list[str], limit: int) -> list[str]:
    """Drop sentences from the tail until ``joined_len(texts) <= limit``.

    Trims from the end because ``responses`` is in report order and the coverage-first selector
    puts the highest-value sentences first, so the last admitted sentence is by construction the
    cheapest one to lose.
    """
    out = list(texts)
    while out and joined_len(out) > limit:
        out.pop()
    return out
