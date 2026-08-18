"""Query-time fuse-then-rerank retrieval stack.

Used by the RAG loop's ``search`` action: encode the English query per leg, search every
language cell of the index named by ``retrieval.index`` across every part, merge within a cell
on raw score, fuse across legs/languages/query variants with ``common.fusion.rrf``, rerank the
top ``retrieval.reranker.depth`` with the cross-encoder, and return ``[(passage_id, score)]``.

The same ranking runs behind two transports, and binding one is the only choice a caller
makes. :func:`bring_up` opens the index in this process, which is what a single-machine run
or a test does. :func:`service_context` binds a client of the long-lived service in
:mod:`ragtime.devkit.rsvc`, which holds one whole rendering resident and answers over a
directory of request and reply files, or over an HTTP forward when the two ends share no
filesystem. The choice is read in one statement inside :func:`retrieve`, so ``tool`` and every
stage body are transport-blind. Every run behind this repository's submission bound the service
transport, which makes the in-process ranker the reference implementation rather than the one
that produced the results; the two fuse the same legs and cells and rerank the same fused top
``retrieval.reranker.depth``.

No text leaves the search path. Passage text is a separate by-id fetch (:func:`display`) in
``passage_lang`` (Knob 2, independent of the searched index, Knob 1), and every emitted
reference resolves through ``common.doc_id_of`` to the original doc-id. That separation is
also what lets one service cover all three readings: changing what is read costs a function
argument, changing what is searched costs a reload.

This package defines no models and stands up no node: every client comes from the ``serving``
singleton registry via the caller's ``ClientBundle``.
"""

from __future__ import annotations

from .concurrency import QueryConcurrency, plan_query_concurrency
from .context import (
    RetrievalContext,
    RetrievalKnobs,
    bring_up,
    retrieval_knobs,
    service_context,
)
from .display import display
from .endpoints import (
    RetrievalServiceError,
    ServiceBackend,
    StaleServiceError,
)
from .service import candidate_depth, legs, query_strings, retrieve
from .store_mirror import (
    PASSAGE_STORE_MIRROR_ROOT_KEY,
    StoreLocation,
    mirror_root,
    resolve_store_location,
)
from .tool import (
    SearchResult,
    format_passages,
    is_search,
    search_action,
)

__all__ = [
    "PASSAGE_STORE_MIRROR_ROOT_KEY",
    "QueryConcurrency",
    "RetrievalContext",
    "RetrievalKnobs",
    "RetrievalServiceError",
    "SearchResult",
    "ServiceBackend",
    "StaleServiceError",
    "StoreLocation",
    "bring_up",
    "candidate_depth",
    "display",
    "format_passages",
    "is_search",
    "legs",
    "mirror_root",
    "plan_query_concurrency",
    "query_strings",
    "resolve_store_location",
    "retrieval_knobs",
    "retrieve",
    "search_action",
    "service_context",
]
