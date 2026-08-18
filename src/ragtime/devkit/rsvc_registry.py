"""The service directory's writer half, plus the context-free ``search`` seam.

The reader half lives elsewhere. ``read_descriptors``, ``is_stale``, ``live_services``,
``service_stamp``, ``ask``, ``ask_service`` and the two error classes are defined in
:mod:`ragtime.retrieval.endpoints` and re-exported here unchanged. They belong there because the
shared-retriever shape is production (one resident index, N topic-owning RAG workers, each a
client) and a production caller may not import ``devkit``, which is a leaf. Re-exporting rather
than re-implementing is the point: there is still exactly one resolution, one liveness check and
one wait, so a client looking in the wrong place or forgetting the rendering filter is
unconstructible rather than merely discouraged.

What stays here is what only the service side needs:

* The writer half, :func:`publish`, :func:`unpublish`, :class:`Heartbeat` and
  :func:`descriptor_path`. ``devkit.rsvc`` is the only thing in the tree that publishes a
  descriptor, so one publisher, one home.
* The default registry path, :data:`DEFAULT_REGISTRY` and :func:`default_registry`. Production
  does not inherit it: ``retrieval.context.service_context`` requires an explicit registry,
  because the registry root is a site fact (which filesystem, which fleet) and a defaulted one is
  how a client reports "no live lane" for a fleet that is up.
* The context-free :func:`search` and :func:`search_batch` seam: a registry, a rendering and a
  string, for smoking a lane or driving a probe. The tool path, an LLM's ``search`` action inside
  a run, reading passages in ``passage_lang``, is ``retrieval.tool.search_action`` over a
  service-backed context. Do not grow either into the other.

A RAG worker must never queue into a black hole. The three facts a descriptor has to carry, and
why presence is not liveness, are documented once, in :mod:`ragtime.retrieval.endpoints`.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from ragtime.retrieval.endpoints import (
    DEFAULT_MAX_AGE_S,
    RetrievalServiceError,
    StaleServiceError,
    ask,
    ask_service,
    is_stale,
    live_services,
    read_descriptors,
    service_stamp,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "REGISTRY_ENV",
    "RENDERINGS",
    "Heartbeat",
    "Hit",
    "RetrievalServiceError",
    "StaleServiceError",
    "ask",
    "ask_service",
    "default_registry",
    "descriptor_path",
    "is_stale",
    "live_services",
    "publish",
    "read_descriptors",
    "search",
    "search_batch",
    "service_stamp",
    "unpublish",
]

#: The closed set of renderings a service may serve: Knob 1 (``retrieval.index``), the index that
#: is searched. Mirrors ``common.passage_store.RENDERINGS``; :func:`search` re-asserts against it.
RENDERINGS: tuple[str, ...] = ("original", "omt", "omt_opus")

#: Where a client looks for descriptors. An env var first, because the registry directory is a
#: site fact (which filesystem, which fleet) and not a run choice, and the same precedence chain
#: `orchestration.cli.artifact_root` uses for the artifact root.
REGISTRY_ENV = "RAGTIME_RSVC_REGISTRY"
DEFAULT_REGISTRY = "logs/rsvc/registry"


def default_registry() -> Path:
    """The registry a client uses when it was not told one: ``$RAGTIME_RSVC_REGISTRY``, else the
    devkit default. Callers that know better pass ``registry=`` explicitly."""
    return Path(os.environ.get(REGISTRY_ENV) or DEFAULT_REGISTRY)


def descriptor_path(registry: Path, name: str) -> Path:
    registry.mkdir(parents=True, exist_ok=True)
    return registry / f"{name}.json"


def publish(path: Path, payload: dict[str, Any]) -> None:
    """Write the descriptor atomically (temp -> rename), stamping the heartbeat.

    Atomic because a worker scanning the registry mid-write would otherwise read half a JSON
    object and conclude the service is broken.
    """
    doc = {
        "name": path.stem,
        "pid": os.getpid(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.uname().nodename,
        "heartbeat": time.time(),
        **payload,
    }
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def unpublish(path: Path) -> None:
    """Delete the descriptor. Called from the service's ``finally`` and by the reaper."""
    Path(path).unlink(missing_ok=True)


class Heartbeat:
    """Republish the descriptor on a timer from a background thread, not from the serve loop.

    A busy service must not look like a dead one. Beating from the idle polling loop instead would
    mean a service processing a long batch stops beating and every client concludes it is a corpse:
    the job is running and visibly answering while its descriptor goes stale and
    :func:`live_services` returns nothing. That is the black-hole failure inverted, and it is
    worse, because it happens exactly when the service is under load.

    A thread rather than a forked child, and the difference from the GIL-blocked-heartbeat lesson
    elsewhere in the tree is measured rather than assumed: the work this thread must interleave
    with is fast-plaid's Rust search, which does release the GIL, proven by the cell fan
    overlapping 10.97 s of per-cell work into a 3.10 s wall. A pure-Python hot loop would need the
    forked-child treatment; this one does not.
    """

    def __init__(self, path: Path, payload: dict[str, Any], interval_s: float = 30.0) -> None:
        import threading

        self.path = Path(path)
        self.payload = dict(payload)
        self.interval = float(interval_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rsvc-heartbeat", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                publish(self.path, self.payload)
            except OSError:  # pragma: no cover - a transient FS error must not kill the beat
                pass
            self._stop.wait(self.interval)

    def start(self) -> Heartbeat:
        publish(self.path, self.payload)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


class Hit(NamedTuple):
    """One retrieved passage, carrying both ids.

    ``passage_id`` is what the retriever ranks and what the reader fetches text by; ``doc_id`` is
    :func:`ragtime.common.doc_id_of` applied to it, that is, the original document id, the one a
    citation must resolve to whichever rendering was searched or read. It is resolved here, at the
    seam, rather than left to each caller, because "resolve every emitted reference through
    ``doc_id_of``" is an invariant, and an invariant with N call sites has N chances to be
    forgotten.
    """

    passage_id: str
    score: float
    doc_id: str
    rank: int


def _hits(pairs: Sequence[Sequence[Any]]) -> list[Hit]:
    from ragtime.common import doc_id_of

    return [
        Hit(passage_id=str(pid), score=float(score), doc_id=doc_id_of(str(pid)), rank=i)
        for i, (pid, score, *_) in enumerate(pairs)
    ]


def search(
    query: str,
    *,
    rendering: str,
    registry: Path | str | None = None,
    top_k: int = 20,
    rerank_depth: int | None = None,
    index_hash: str | None = None,
    timeout_s: float = 900.0,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> list[Hit]:
    """The call a client makes: discover a live service for ``rendering``, ask it, get hits back.

    This is the whole client contract, and it exists so that no caller has to know a queue path,
    remember to filter by rendering, or remember to check that anything is alive:

    * Discovery, not a path. :func:`live_services` is filtered on ``rendering``, and on
      ``index_hash`` when given, and rejects any descriptor whose heartbeat has stopped. Presence
      is not liveness: a stale descriptor will happily advertise a dead job, and the error it
      eventually produces (a missing queue directory) reads like a build failure rather than a
      dead service.
    * The rendering is required and validated twice: once here, against the closed set and the
      registry filter, and once at the service, which refuses a request naming a rendering it does
      not serve. The reply's own ``rendering`` is then checked against what was asked. This is
      Knob 1, ``retrieval.index``, the index that is searched, and a wrong one is the worst failure
      available here, because nothing downstream errors: the Knob-1 comparison is silently
      corrupted and it looks like a result. There is no default.
    * Every hit carries the original doc-id, resolved through ``common.doc_id_of``.

    Returns :class:`Hit` in rank order. Raises :class:`StaleServiceError` if no live service serves
    this rendering, or if the one being waited on dies mid-request, and
    :class:`RetrievalServiceError` if the service reports an error or answers for the wrong index.

    The shape mirrors ``retrieval.service.retrieve(ctx, queries, top_k)``, the one production
    fuse-then-rerank surface, so the in-process and out-of-process spellings stay swappable: this
    module adds an RPC seam, never a second ranking path.

    There are two client entry points, and this is the context-free one: it needs a registry, a
    rendering and a string, and it is what a caller uses to ask the fleet a question, such as
    smoking a lane or driving a probe. The tool path, an LLM's ``search`` action inside a
    run, reading passages in ``passage_lang``, is ``retrieval.tool.search_action`` over a context
    from :func:`ragtime.retrieval.context.service_context`, so query normalisation, the
    whole-passage character budget, ``display()``, ``doc_id_of`` and ``format_passages`` are
    production's and only the ids-and-scores step travels over the queue. Do not grow either one
    into the other: this function must never acquire a ``RetrievalContext``, a rerank policy or a
    fusion step, and the tool path must never stop going through ``ask_service``. Both call
    ``ask_service``, so there is exactly one place where the rendering filter and the liveness
    check live, and that is the property worth protecting.
    """
    return search_batch(
        [query],
        rendering=rendering,
        registry=registry,
        top_k=top_k,
        rerank_depth=rerank_depth,
        index_hash=index_hash,
        timeout_s=timeout_s,
        max_age_s=max_age_s,
    )[0]


def search_batch(
    queries: Sequence[str],
    *,
    rendering: str,
    registry: Path | str | None = None,
    top_k: int = 20,
    rerank_depth: int | None = None,
    index_hash: str | None = None,
    timeout_s: float = 900.0,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> list[list[Hit]]:
    """:func:`search` for many queries in one request: one hit list per query, in input order.

    Batching is the reason a service has replicas. The service dispatches the list across its
    resident stacks and returns results at the index they were submitted with, so a batch of N
    costs about 1/replicas of the serial wall while the i-th list still belongs to the i-th query.
    Order is a correctness property rather than a nicety: the caller pairs positionally and the
    scores are floats, so completion order would silently re-pair queries with answers.
    """
    if rendering not in RENDERINGS:
        raise ValueError(
            f"rendering={rendering!r} is not one of {RENDERINGS}. It is Knob 1 (retrieval.index, "
            "the index that is searched, never passage_lang) and it is required: a default is "
            "exactly how a caller ends up comparing two runs that searched different indexes."
        )
    qs = [str(q) for q in queries]
    if not qs:
        return []
    reg = Path(registry) if registry is not None else default_registry()
    # `rerank` and `rerank_depth` are not forced here. The service was launched from
    # a run config, so whether it reranks and how deep is that config's `retrieval` block talking:
    # a client that overrode it would be moving a shared, fairness-relevant knob from the outside.
    request: dict[str, Any] = {"query": qs, "top_k": int(top_k)}
    if rerank_depth is not None:
        request["rerank_depth"] = int(rerank_depth)
    reply = ask_service(
        reg,
        request,
        rendering=rendering,
        index_hash=index_hash,
        max_age_s=max_age_s,
        timeout_s=timeout_s,
    )
    if not reply.get("ok"):
        raise RetrievalServiceError(
            f"retrieval service ({reply.get('shape', {}).get('node')}, job "
            f"{reply.get('shape', {}).get('job_id')}) failed request {reply.get('id')}: "
            f"{reply.get('error')}"
        )
    # Belt and braces over the service's own refusal: the answer must say which index produced it.
    got = str(reply.get("rendering") or "")
    if got != rendering:
        raise RetrievalServiceError(
            f"asked {rendering!r} but the reply came from {got!r} (index_hash "
            f"{str(reply.get('index_hash'))[:12]}), refusing an answer from the wrong index "
            "rather than silently corrupting the Knob-1 comparison"
        )
    runs = reply.get("runs") or []
    if len(runs) != len(qs):
        raise RetrievalServiceError(
            f"asked {len(qs)} queries and got {len(runs)} result lists back; a positional pairing "
            "over a short list would re-pair queries with answers"
        )
    return [_hits(run.get("hits") or []) for run in runs]
