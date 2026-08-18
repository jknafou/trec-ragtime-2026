"""Counter ids this package emits onto ``common.stats``.

Every id is declared here so the vocabulary a monitoring rollup consumes is readable in one
file. Slice keys are the canonical ones ``common.stats.CANONICAL_SLICE_KEYS`` allows
(``variant``/``seed``/``lang``/``round``/``nugget``/``leg``); ``variant`` carries the searched
rendering on the search counters and the read rendering on the display counters.

Wall time is emitted as an ordinary counter value (seconds, summed), so a run's logs can answer
"how long did the fan take, per leg" without a profiler.
"""

from __future__ import annotations

__all__ = [
    "STAT_CANDIDATES_FUSED",
    "STAT_DISPLAY_LOOKUPS",
    "STAT_LEG_CELL_SEARCHES",
    "STAT_LEG_SECONDS",
    "STAT_QUERIES",
    "STAT_QUERY_SECONDS",
    "STAT_QUERY_STRINGS",
    "STAT_RERANKED",
    "STAT_RERANK_SECONDS",
    "STAT_STORE_FETCH_SECONDS",
    "STAT_STORE_MIRRORED",
    "STAT_STORE_MIRROR_REFUSED",
]

#: One per ``retrieve()`` call, sliced by the searched rendering.
STAT_QUERIES = "retrieval.queries"
#: Distinct query strings encoded in that call (the model may emit more than one per search).
STAT_QUERY_STRINGS = "retrieval.query_strings"
#: One per ``query_lang_leg`` call, i.e. per (leg, language cell, query string). The part fan
#: lives one level below this and belongs to the index.
STAT_LEG_CELL_SEARCHES = "retrieval.leg_cell_searches"
#: Distinct passage ids the RRF layer fused (pre-rerank pool size).
STAT_CANDIDATES_FUSED = "retrieval.candidates_fused"
#: Candidates actually handed to the cross-encoder (``<= retrieval.reranker.depth``).
STAT_RERANKED = "retrieval.reranked_candidates"
#: By-id text fetches, sliced by the read rendering (Knob 2), never the searched one.
STAT_DISPLAY_LOOKUPS = "retrieval.display_lookups"

# Which copy of the by-id store this context opened (see retrieval.store_mirror).
#: One per context that opened a verified staged mirror of the passage store.
STAT_STORE_MIRRORED = "retrieval.store_mirrored"
#: One per context that was offered a mirror and refused it, falling back to the canonical
#: store. The reason is not a canonical slice key, so it is carried by the accompanying
#: ``retrieval.store_mirror.refused`` warning.
STAT_STORE_MIRROR_REFUSED = "retrieval.store_mirror_refused"

# The LLM-facing search tool (retrieval.tool): how often the model chose to search, how much it
# was shown, and what the context budget cost it.
STAT_TOOL_CALLS = "retrieval.tool_calls"
STAT_TOOL_PASSAGES_SHOWN = "retrieval.tool_passages_shown"
STAT_TOOL_BUDGET_DROPPED = "retrieval.tool_budget_dropped"

# Wall time, in seconds, summed.
STAT_QUERY_SECONDS = "retrieval.query_seconds"
STAT_LEG_SECONDS = "retrieval.leg_seconds"
#: The cross-encoder alone, i.e. ``reranker.score`` and nothing else.
STAT_RERANK_SECONDS = "retrieval.rerank_seconds"
#: The by-id text fetch the reranker's pool needs, timed separately from the model: a
#: filesystem cost and a GPU cost move for unrelated reasons and get their own counters.
STAT_STORE_FETCH_SECONDS = "retrieval.store_fetch_seconds"
