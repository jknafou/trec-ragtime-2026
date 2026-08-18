"""Knob 2: fetch passage text by id, in a rendering chosen independently of what was searched.

:func:`~ragtime.retrieval.service.retrieve` returns ids and scores and never text; this module
turns ids into text in ``passage_lang``, which is a different knob from ``retrieval.index``.
"Search OMT, display Original" is therefore first-class: nothing here depends on which index the
ids came from, and ``retrieve``'s signature carries no rendering parameter.

The doc-id is always resolved through ``common.doc_id_of``, so every citation points at the
original document whatever was read or searched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragtime.common.ids import doc_id_of
from ragtime.common.passage_store import RENDERINGS

from .stats import STAT_DISPLAY_LOOKUPS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .context import RetrievalContext

__all__ = ["display"]


def display(
    ctx: RetrievalContext,
    passage_ids: Sequence[str],
    passage_lang: str | None = None,
) -> list[tuple[str, str]]:
    """``[(doc_id, text)]`` for ``passage_ids``, in ``passage_lang``, in the given order.

    Order is preserved and duplicates are kept, so a caller can zip the result back onto the ids
    it asked for and recover the ``passage_id`` (and, via ``ctx.passage_store.passage(pid).lang``,
    the native language) that the ``(doc_id, text)`` shape does not carry.

    ``passage_lang=None`` means the run's own ``passage_lang`` (Knob 2); passing it explicitly
    makes a per-call rendering choice greppable, which the cross-rendering diagnostics need.
    """
    lang = _rendering(ctx, passage_lang)
    out = [
        (doc_id_of(passage_id), ctx.passage_store.render(passage_id, lang))
        for passage_id in passage_ids
    ]
    if out:
        ctx.stats.emit(STAT_DISPLAY_LOOKUPS, float(len(out)), variant=lang)
    return out


def _rendering(ctx: RetrievalContext, passage_lang: str | None) -> str:
    lang = passage_lang if passage_lang is not None else ctx.passage_lang
    if lang not in RENDERINGS:
        raise ValueError(f"unknown rendering {lang!r}; expected one of {RENDERINGS}")
    return lang
