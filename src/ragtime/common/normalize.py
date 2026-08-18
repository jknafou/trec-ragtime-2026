"""Unicode normalization: two distinct primitives, not merged.

``nfc`` is the verbatim-commit normalizer, applied symmetrically to a claim span
and its cited passage at the span-grounding gate. ``nfkc_len`` is the Task-1
report character-budget counter. Conflating the two would silently change either
grounding or the budget, so they are kept apart.
"""

from __future__ import annotations

import unicodedata

__all__ = ["nfc", "nfkc_len"]


def nfc(s: str) -> str:
    """Return the NFC (canonical composition) normalization of ``s``.

    The only normalizer used for the verbatim span-commit gate, applied identically
    to the claim span and to the passage text, so a span differing from the passage
    only by canonical (de)composition still commits. NFC is canonical-equivalence
    only: it never folds compatibility variants such as ligatures or full-width
    forms.
    """
    return unicodedata.normalize("NFC", s)


def nfkc_len(s: str) -> int:
    """Return the NFKC character length of ``s``, the Task-1 budget count.

    NFKC applies compatibility folding (a full-width digit or a ligature collapses
    to its ASCII form) before counting, so the budget measures the rendered length.
    Used for the report character budget only, never for grounding.
    """
    return len(unicodedata.normalize("NFKC", s))
