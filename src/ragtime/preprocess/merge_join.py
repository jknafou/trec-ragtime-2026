"""The ``§`` marker join and split-back: how a merged unit is fed to MT and taken apart.

A pure text mechanism with no model and no heavy import. :mod:`ragtime.preprocess.translate_omt`
owns which sentences form a unit; this module owns how a unit becomes one MT input and how
that input's English comes back apart into per-sentence segments.

Very short sentences translate poorly on their own, because NLLB fills the context vacuum
with template output. Merging them with their neighbours fixes that, but NLLB is a
sentence-level model with no document mode: handed several propositions in one input it
compresses and silently drops one. Joining the constituents with a distinctive marker and
splitting the English back on it buys the context while keeping most of the granularity.
``§`` was chosen over ``¶`` because it survives NLLB's SentencePiece as a single atomic
piece and is re-emitted by the decoder far more reliably.

A marker must never reach a stored passage, but the enforcement point is reconciliation,
not this stage: the translate stage stores its MT output raw with markers intact, and
:func:`scrub_markers` runs exactly once at reconciliation's boundary, where the
fuse-versus-split-back decision is also made. Storing raw keeps a change of fusion policy
to a CPU re-pass over stored text instead of a re-translation of the corpus.

Only the marker in use is scrubbed. The interpunct ``·`` is left alone: it is
a legitimate character in transliterated names, and this pipeline never inserts it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "MARKER",
    "contains_marker",
    "join_constituents",
    "scrub_markers",
    "split_translation",
]

#: The join marker. ``serving.mt.MtClient.assert_marker_atomic`` re-checks at every
#: bring-up that the tokenizer still treats it as one piece.
MARKER = "§"

#: The marker is surrounded by spaces so SentencePiece sees it as a word-initial piece
#: (``▁§``); a bare ``a§b`` would fuse it into its neighbours.
_JOIN_FMT = " {} "


def join_constituents(texts: Sequence[str], marker: str = MARKER) -> str:
    """Build one MT input from a merged unit's constituents: ``a § b § c``.

    The marker sits inside the space a plain join would use, so the only difference from an
    unmarked merge is the boundary signal. Constituent text is otherwise unaltered: the
    verbatim document span goes in.
    """
    return _JOIN_FMT.format(marker).join(texts)


def contains_marker(text: str, marker: str = MARKER) -> bool:
    """Return True when the source text already carries the marker.

    A constituent containing the marker natively would make the returned marker count
    ambiguous, so :mod:`ragtime.preprocess.translate_omt` declines to merge that unit rather
    than escaping it: translating the constituents alone is exactly the pre-merge behaviour,
    whereas an escape sequence's own translation is unproven. Roughly 0.002% of non-English
    passages are affected.
    """
    return marker in text


def scrub_markers(text: str, marker: str = MARKER) -> str:
    """Remove every marker glyph and normalize the whitespace it leaves behind.

    It takes no "did the split work" argument: reconciliation runs it on the final text of
    every path (split, fused, declined and single-constituent alike), so the guarantee that
    no marker is stored is structural rather than a consequence of the split succeeding. It
    is not called by the translate stage, whose output is raw by contract.
    """
    if not text:
        return text
    return " ".join(text.replace(marker, " ").split())


def split_translation(
    out_text: str, n_constituents: int, marker: str = MARKER
) -> list[str] | None:
    """Split a merged unit's English back into per-constituent segments, or return ``None``.

    Segments are returned only when the marker count is exactly ``n_constituents - 1``, that
    is when the model emitted every boundary it was given and no extra. Any other count means
    the segment-to-constituent correspondence is unknown, so the caller keeps the unit fused.
    A wrong count is therefore positive evidence that the model compressed, and the fallback
    is safe by construction.

    An empty segment is also a refusal: the boundary is present but one constituent came back
    with nothing, and attributing it would store an empty translation for a real sentence.

    Segments are scrubbed, so this function can serve as reconciliation's split path directly.
    """
    if n_constituents < 2 or marker not in out_text:
        return None
    parts = out_text.split(marker)
    if len(parts) != n_constituents:
        return None
    segments = [scrub_markers(p, marker) for p in parts]
    return segments if all(segments) else None
