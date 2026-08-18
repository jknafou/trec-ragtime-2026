"""The LLM-facing search tool: the binding from a model ``search`` action to readable passages.

It composes the two halves of the stack rather than adding a third: :func:`retrieve` (ids and
scores) and :func:`display` (text by id) are called unchanged, so the single-code-path invariant
survives, and text still meets ranking only after ranking is final.

The char budget here is a context budget -- how much a model may read in one turn. It is not the
Task-1 submission budget and must never be computed with ``common.nfkc_len``, which is that other
budget's unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .display import display
from .service import retrieve
from .stats import (
    STAT_TOOL_BUDGET_DROPPED,
    STAT_TOOL_CALLS,
    STAT_TOOL_PASSAGES_SHOWN,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .context import RetrievalContext

__all__ = ["SearchResult", "format_passages", "is_search", "search_action"]

SEARCH_ACTION = "search"


@dataclass(frozen=True)
class SearchResult:
    """What one tool call produced, carrying both ``passage_id`` and ``doc_id``.

    ``display`` returns ``(doc_id, text)``, which is right for citation but loses the id the
    claim-commit path needs to re-fetch a passage in another rendering. Keeping both means no
    caller re-derives one from the other, and ``doc_id_of`` stays the single ID-algebra owner.
    """

    queries: tuple[str, ...]
    hits: tuple[tuple[str, float], ...]
    passages: tuple[tuple[str, str, str], ...]  # (passage_id, doc_id, text)
    passage_lang: str
    dropped_for_budget: int

    @property
    def shown(self) -> int:
        return len(self.passages)

    @property
    def passage_ids(self) -> tuple[str, ...]:
        return tuple(pid for pid, _, _ in self.passages)


def is_search(action: Mapping[str, Any] | None) -> bool:
    """True when this decision is a search.

    A malformed action makes the loop take the not-a-search branch rather than crashing it.
    """
    return bool(action) and action.get("action") == SEARCH_ACTION


def _queries_of(action: Mapping[str, Any]) -> tuple[str, ...]:
    """The schema allows one ``query``; ``retrieve`` accepts several. Normalize, invent nothing."""
    raw = action.get("query")
    if raw is None:
        return ()
    items: Sequence[Any] = raw if isinstance(raw, (list, tuple)) else [raw]
    return tuple(q.strip() for q in items if isinstance(q, str) and q.strip())


def search_action(
    ctx: RetrievalContext,
    action: Mapping[str, Any],
    *,
    top_k: int,
    char_budget: int | None = None,
    passage_lang: str | None = None,
) -> SearchResult:
    """Execute the model's ``search`` decision and return passages it can read.

    ``top_k`` is caller-supplied; this package holds no default for it. ``char_budget`` truncates
    by whole passages, never mid-passage: a clipped passage would break the verbatim NFC
    span-commit downstream, since the model could quote a span that does not exist in the stored
    text. Dropped passages are counted, never silently discarded.
    """
    if not is_search(action):
        raise ValueError(f"not a search action: {action.get('action')!r}")
    queries = _queries_of(action)
    if not queries:
        raise ValueError("search action carried no usable query")

    ctx.stats.emit(STAT_TOOL_CALLS, 1.0)
    hits = retrieve(ctx, list(queries), top_k=top_k)
    ids = [pid for pid, _ in hits]
    rendered = display(ctx, ids, passage_lang)

    lang = passage_lang if passage_lang is not None else ctx.passage_lang
    kept: list[tuple[str, str, str]] = []
    used, dropped = 0, 0
    for pid, (doc_id, text) in zip(ids, rendered, strict=True):
        if char_budget is not None and used + len(text) > char_budget:
            dropped += 1
            continue
        kept.append((pid, doc_id, text))
        used += len(text)

    if kept:
        ctx.stats.emit(STAT_TOOL_PASSAGES_SHOWN, float(len(kept)), variant=lang)
    if dropped:
        ctx.stats.emit(STAT_TOOL_BUDGET_DROPPED, float(dropped), variant=lang)

    return SearchResult(
        queries=queries,
        hits=tuple(hits),
        passages=tuple(kept),
        passage_lang=lang,
        dropped_for_budget=dropped,
    )


def format_passages(result: SearchResult, *, header: str = "RETRIEVED PASSAGES") -> str:
    """Render the tool's output as the block the model reads back.

    Each passage is labeled with its ``passage_id``, which is the handle ``submit_claim``
    commits against; the three renderings share passage ids, so it always resolves. The
    ``doc_id`` is not shown: it is derived from the passage id by ``common.doc_id_of``, and
    printing it would invite a citation in the one form the commit path cannot accept.
    """
    if not result.passages:
        return f"{header}: none."
    lines = [f"{header} ({result.shown}, rendering={result.passage_lang}):"]
    for i, (pid, _doc_id, text) in enumerate(result.passages, 1):
        lines.append(f"[{i}] passage_id={pid}\n{text}")
    if result.dropped_for_budget:
        lines.append(
            f"({result.dropped_for_budget} further passage(s) omitted — context budget reached)"
        )
    return "\n\n".join(lines)
