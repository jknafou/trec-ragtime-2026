"""Reach one resident index from many processes: the service directory and the queue transport.

``bring_up`` opens the index in the calling process, which is a large working set and a long cold
load. That is affordable once, so the deployment shape is one retrieval service and N
topic-owning RAG workers, each worker a cheap client. This module is the client half.

The transport is a filesystem queue rather than HTTP because compute nodes on the target cluster
export ``http_proxy`` and cross-node HTTP times out: a queue on shared storage has no port to be
proxied and no readiness endpoint to poll, which is what lets a one-core CPU client drive a
two-GPU service.

A worker must never queue into a black hole, and that decomposes into three facts a descriptor
carries:

* where -- the queue directory, so a client can drop a request;
* what -- the ``rendering`` and the ``index_hash``, so a client picks the service that searches
  the index its run is about. "Any live service" is wrong: the three renderings share passage
  ids, so a result from the wrong lane is not an error anywhere downstream, it is silently the
  wrong experiment;
* whether it is still alive -- a ``heartbeat`` the service refreshes on a timer. A descriptor
  file outlives its job, so presence is not liveness. :func:`is_stale` treats a missed beat as
  dead and :func:`live_services` refuses to return one.

:class:`ServiceBackend` is the ranking client, and it carries four refusals, because the
alternative to each is a plausible-looking wrong answer rather than an error:

1. the run config's own ``retrieval.index`` (Knob 1) must equal the lane being addressed, checked
   before a byte is written (:meth:`ServiceBackend._assert_lane_matches_config`);
2. the registry filter resolves only services matching ``(rendering, index_hash)`` and beating
   (:func:`live_services`, via :func:`ask_service`);
3. the reply's own ``rendering``/``index_hash`` are re-checked, and before ``ok``
   (:meth:`ServiceBackend._hits_of`);
4. the server refuses a request naming a rendering it does not serve. This is a real second
   witness only because :func:`ask_service` preserves a caller-supplied ``index`` instead of
   defaulting it from the descriptor it just picked; overwriting it would make the server compare
   its own rendering to a value derived from its own descriptor.

This module is not a second ranking path. It carries ids and scores over a wire and nothing else:
``retrieval.service.retrieve`` remains the one entry point and ``retrieval.tool.search_action``
is not widened. Which transport answered is a property of the
:class:`~ragtime.retrieval.context.RetrievalContext` the caller was handed, never a branch in a
stage body.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .context import RetrievalContext

__all__ = [
    "DEFAULT_MAX_AGE_S",
    "DEFAULT_TIMEOUT_S",
    "RankingBackend",
    "RetrievalServiceError",
    "HttpServiceBackend",
    "ServiceBackend",
    "StaleServiceError",
    "ask",
    "ask_service",
    "is_stale",
    "live_services",
    "read_descriptors",
    "service_stamp",
]

#: A descriptor whose heartbeat is older than this is treated as dead. Three times the service's
#: default 30 s beat, so a busy service answering a long query is never mistaken for a corpse.
DEFAULT_MAX_AGE_S = 120.0

#: A replica with several callers queued in front of it can legitimately take minutes, so this is
#: long enough never to abandon a live service; the heartbeat check, not this number, is what
#: catches a dead one.
DEFAULT_TIMEOUT_S = 99999.0


class StaleServiceError(RuntimeError):
    """Nothing answered: no live lane, or the one being waited on stopped beating mid-flight.

    A sibling of :class:`RetrievalServiceError` rather than a subclass. The two mean
    different things to a supervisor: "the fleet is down" (retryable infrastructure) against "the
    fleet answered and the answer was wrong".
    """


class RetrievalServiceError(RuntimeError):
    """The service answered and the answer was a failure, or was not the lane that was asked for."""


# --------------------------------------------------------------------------- #
# The directory: which services exist, over which index, and are they alive.
# --------------------------------------------------------------------------- #
def read_descriptors(registry: Path | str) -> list[dict[str, Any]]:
    """Every descriptor in the registry, with torn reads skipped rather than raised.

    A descriptor is published temp-then-rename, so a scan that catches one mid-replace sees either
    the old file or the new one, but a directory listing can still race the unlink. A JSON error
    here means "that file was being replaced", so it is skipped and picked up on the next scan.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(Path(registry).glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        doc["_path"] = str(path)
        out.append(doc)
    return out


def is_stale(doc: dict[str, Any], *, max_age_s: float = DEFAULT_MAX_AGE_S) -> bool:
    """A descriptor is stale when its heartbeat has stopped. Presence is never liveness."""
    beat = doc.get("heartbeat")
    if beat is None:
        return True
    return (time.time() - float(beat)) > max_age_s


def live_services(
    registry: Path | str,
    *,
    rendering: str | None = None,
    index_hash: str | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> list[dict[str, Any]]:
    """Every service that is beating and searches the index the caller means.

    Filtering on ``index_hash`` is not optional hygiene: the three renderings share passage ids,
    so a result from the wrong index errors nowhere downstream. Two services over two builds of
    the same rendering are the same hazard, which is why the pair is filtered and not just the
    name.
    """
    out = []
    for doc in read_descriptors(registry):
        if is_stale(doc, max_age_s=max_age_s):
            continue
        if rendering is not None and doc.get("rendering") != rendering:
            continue
        if index_hash is not None and doc.get("index_hash") != index_hash:
            continue
        out.append(doc)
    return sorted(out, key=lambda d: str(d.get("name")))


def service_stamp(doc: dict[str, Any]) -> dict[str, Any]:
    """The JSON-safe identity of one service for a run record: which lane, over which index.

    A projection of the descriptor rather than the descriptor itself, so a record carries the
    fields provenance is stated in rather than a whole file that grows keys. It is simultaneously
    the operational record (which service restart answered), the fairness witness, and the
    debugging record.
    """
    keys = ("name", "slot", "tier", "rendering", "index_hash", "node", "job_id", "queue")
    out = {k: doc.get(k) for k in keys if doc.get(k) is not None}
    if doc.get("_path"):
        out["descriptor"] = str(doc["_path"])
    return out


# --------------------------------------------------------------------------- #
# The transport: one request file in, one reply file out, abortable.
# --------------------------------------------------------------------------- #
def ask_service(
    registry: Path | str,
    request: dict[str, Any],
    *,
    rendering: str | None = None,
    index_hash: str | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    **kwargs: Any,
) -> dict[str, Any]:
    """Resolve a live service for the wanted lane and ask it. The only safe client entrypoint.

    Prefer this over :func:`ask` with a hand-carried queue path: it is the one form in which the
    rendering filter and the liveness check cannot be forgotten, and forgetting either is how a
    client either reads the wrong index or blocks on a corpse.

    The reply is stamped with ``_service`` (:func:`service_stamp`), naming which lane answered,
    because the resolver is the only code that knows which descriptor it picked.
    """
    services = live_services(
        registry, rendering=rendering, index_hash=index_hash, max_age_s=max_age_s
    )
    if not services:
        raise StaleServiceError(
            f"no live service in {registry} for rendering={rendering!r} "
            f"index_hash={index_hash!r} (descriptors present: "
            f"{[d.get('name') for d in read_descriptors(registry)]})"
        )
    svc = services[0]
    # `request` is spread last, so a caller-supplied `index` is preserved and the default below
    # applies only when the caller named nothing. Overwriting it would make the server's own
    # refusal a tautology: it would compare its rendering to a value derived from its own
    # descriptor, leaving the client-side check as the single witness that a query hit the lane
    # the caller meant.
    payload = {"index": rendering if rendering is not None else svc["rendering"], **request}
    reply = ask(
        svc["queue"],
        payload,
        descriptor=Path(svc["_path"]),
        max_age_s=max_age_s,
        **kwargs,
    )
    if isinstance(reply, dict):
        reply["_service"] = service_stamp(svc)
    return reply


def ask(
    queue: Path | str,
    request: dict[str, Any],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_s: float = 0.05,
    descriptor: Path | str | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> dict[str, Any]:
    """Drop one request in ``<queue>/in`` and wait for ``<queue>/out/<id>.json``.

    ``descriptor`` makes the wait abortable. A queue directory belonging to a cancelled service
    still accepts a request file and simply never answers, so a client without this check blocks
    for its full timeout; and when the service staged its blobs in tmpfs the directory can vanish
    on job exit, so the caller sees ``FileNotFoundError``, which reads like a build problem rather
    than a dead service. When a descriptor is given, its heartbeat is re-checked while waiting and
    a stopped beat raises :class:`StaleServiceError` immediately.

    The request file is written temp-then-rename for the same reason the reply is: the service
    polls the directory and a partially written request would be read as malformed JSON. The id is
    a uuid unless the caller supplied one, so two clients cannot collide on a reply file, which is
    what makes one queue safe for N concurrent clients.
    """
    queue = Path(queue)
    qin, qout = queue / "in", queue / "out"
    qin.mkdir(parents=True, exist_ok=True)
    qout.mkdir(parents=True, exist_ok=True)
    rid = str(request.get("id") or uuid.uuid4().hex[:12])
    payload = {**request, "id": rid}
    tmp = qin / f".{rid}.json.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(qin / f"{rid}.json")

    reply = qout / f"{rid}.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if reply.exists():
            try:
                got = json.loads(reply.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(poll_s)
                continue
            reply.unlink(missing_ok=True)
            return got
        if descriptor is not None and not _still_beating(Path(descriptor), max_age_s):
            raise StaleServiceError(
                f"the service owning {queue} stopped beating while request {rid} was in flight "
                f"(descriptor {descriptor}); it is dead or hung, and waiting out the remaining "
                f"{deadline - time.time():.0f}s would only turn a clear failure into a slow one"
            )
        time.sleep(poll_s)
    raise TimeoutError(f"no reply for {rid} within {timeout_s}s (queue={queue})")


def _still_beating(descriptor: Path, max_age_s: float) -> bool:
    """Cheap liveness re-check: the descriptor exists and its heartbeat is fresh."""
    try:
        doc = json.loads(descriptor.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, json.JSONDecodeError):
        return True  # a torn read during the atomic replace is not evidence of death
    return not is_stale(doc, max_age_s=max_age_s)


# --------------------------------------------------------------------------- #
# The ranking backend a service-backed RetrievalContext carries.
# --------------------------------------------------------------------------- #
class RankingBackend(Protocol):
    """The seam ``retrieval.service.retrieve`` dispatches on. One method.

    ``rank`` is handed the already-normalized query strings (``service.query_strings`` ran) and
    returns exactly what the in-process path returns plus the fused-pool size for the counter bus:
    ``([(passage_id, score)], fused_pool | None)``, best first, at most ``top_k`` long, no text.
    Anything wider would be a second ranking path wearing a Protocol.
    """

    def rank(
        self, ctx: RetrievalContext, queries: Sequence[str], *, top_k: int
    ) -> tuple[list[tuple[str, float]], int | None]: ...

    @property
    def kind(self) -> str:
        """The transport name recorded in the artifact (``"service"``)."""
        ...


@dataclass(frozen=True)
class ServiceBackend:
    """One retrieval lane: a registry, the rendering it must serve, and the build it must be.

    ``rendering`` and ``index_hash`` have no defaults. A default would let a caller
    address "whatever service happens to be up", which is exactly the Knob-1 substitution the
    fairness family guard exists to prevent, and which nothing downstream would report.
    """

    registry: Path
    #: Knob 1, ``retrieval.index``: the rendering that is searched. Never ``passage_lang``.
    rendering: str
    #: The published index this lane must be serving, so a service over a different build of the
    #: same rendering cannot answer for it.
    index_hash: str
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_age_s: float = DEFAULT_MAX_AGE_S
    #: Every exchange, for the run record: which service answered, how long it took, how long it
    #: queued. ``family_guard`` cannot see ``execution``, so the artifact is what proves two
    #: members of a family searched the same lane.
    calls: list[dict[str, Any]] = field(default_factory=list, compare=False)

    @property
    def kind(self) -> str:
        return "service"

    def rank(
        self, ctx: RetrievalContext, queries: Sequence[str], *, top_k: int
    ) -> tuple[list[tuple[str, float]], int | None]:
        """Ask the lane for ``[(passage_id, score)]``. Signature-compatible with the local path.

        ``rerank_depth`` travels from ``ctx.knobs``, this run's config rather than a literal here,
        so the service reranks to the depth the run record declares.
        """
        qs = list(queries)
        if not qs:
            return [], None
        if len(qs) > 1:
            raise RetrievalServiceError(
                f"the retrieval service answers one query per request, but this action carried "
                f"{len(qs)}. Production fuses several query strings as one rrf over "
                "(query x leg x cell) pools; fusing the service's per-query answers here would "
                "be a second, different ranking policy. Run this config against an in-process "
                "context instead."
            )
        self._assert_lane_matches_config(ctx)

        started = time.perf_counter()
        reply = self._ask(
            {
                "query": qs[0],
                "top_k": int(top_k),
                "rerank_depth": int(ctx.knobs.rerank_depth),
                # Named, and `ask_service` preserves it: the client's half of the server-side
                # refusal (witness 4 in the module docstring).
                "index": self.rendering,
            }
        )
        elapsed = time.perf_counter() - started
        hits, fused = self._hits_of(reply)
        summary = reply.get("summary") or {}
        self.calls.append(
            {
                "query": qs[0],
                "top_k": int(top_k),
                "rerank_depth": int(ctx.knobs.rerank_depth),
                "rendering": self.rendering,
                "index_hash": self.index_hash,
                # Which service answered, stamped by the resolver, the only code that knows.
                "service": reply.get("_service"),
                "replicas": reply.get("replicas"),
                # Separates "the query is slow" from "the query queued", which a wall alone
                # cannot, and is what sizes an admission ceiling.
                "queued_s": summary.get("queued_s"),
                "client_wall_s": round(elapsed, 3),
                "service_wall_s": (reply.get("runs") or [{}])[0].get("wall_s"),
                "hits": len(hits),
            }
        )
        return hits[: max(0, int(top_k))], fused

    def _ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Fetch one reply. The only transport-specific step in this class.

        Split out so a second transport (see :class:`HttpServiceBackend`) reuses ``rank``
        verbatim: the lane assertion, the ``_hits_of`` validation that checks the reply's
        rendering before its ``ok``, and the per-call run record are then the same code on both
        transports.
        """
        return ask_service(
            self.registry,
            payload,
            rendering=self.rendering,
            index_hash=self.index_hash,
            max_age_s=self.max_age_s,
            timeout_s=self.timeout_s,
        )

    def _assert_lane_matches_config(self, ctx: RetrievalContext) -> None:
        if ctx.index != self.rendering:
            raise RetrievalServiceError(
                f"this lane serves {self.rendering!r} but the run's retrieval.index (Knob 1) is "
                f"{ctx.index!r}. The config is the run record, so the substitution is refused "
                "rather than silently honored: launch a service for the rendering this config "
                "searches, or run the config whose Knob 1 is the one you meant."
            )

    def _hits_of(self, reply: dict[str, Any]) -> tuple[list[tuple[str, float]], int | None]:
        """Validate the reply's lane first, then its contents.

        A wrong-lane answer is checked before ``ok``, because a service that answered happily from
        the wrong index produces a plausible result, which is the most expensive failure available
        here. An ``ok: false`` from the right lane is merely an error.
        """
        got_rendering = reply.get("rendering")
        got_hash = reply.get("index_hash")
        if got_rendering != self.rendering or got_hash != self.index_hash:
            raise RetrievalServiceError(
                f"asked the {self.rendering!r} lane (index_hash {self.index_hash[:12]}) and the "
                f"reply came back as {got_rendering!r} (index_hash {str(got_hash)[:12]}); the "
                "three renderings share passage ids, so this answer would be silently the wrong "
                "experiment rather than an error. Refusing it."
            )
        if not reply.get("ok"):
            raise RetrievalServiceError(f"the retrieval service refused: {reply.get('error')!r}")
        runs = reply.get("runs") or []
        if len(runs) != 1:
            raise RetrievalServiceError(
                f"expected exactly one ranked list for one query; the service returned {len(runs)}"
            )
        raw = runs[0].get("hits") or []
        hits = [(str(pid), float(score)) for pid, score in raw]
        fused = runs[0].get("fused_pool")
        return hits, (int(fused) if fused is not None else None)


@dataclass(frozen=True)  # ServiceBackend is frozen; a non-frozen child is a TypeError
class HttpServiceBackend(ServiceBackend):
    """The same lane, reached over HTTP instead of the filesystem queue, for cross-cluster work.

    Two clusters that share no filesystem cannot share a queue directory, and they may reuse the
    same private address space, so a worker on one cannot address the other's queue or node
    directly. What does work is an ssh port-forward to a small HTTP bridge sitting beside the
    queue on the service's cluster, itself an ordinary :func:`ask_service` client::

        worker --ssh -L--> login node : bridge --queue--> retrieval service

    The transport overhead measures in the low tens of milliseconds against a multi-second query.

    It reuses ``rank`` verbatim and overrides only :meth:`_ask`, so the lane assertion, the
    reply-lane check that runs before ``ok``, and the per-call run record are the same code on
    both transports.

    One thing does change: refusals 1 to 3 (config-against-lane, the registry's
    ``(rendering, index_hash)`` filter, the heartbeat check) execute inside the bridge, because
    that is where ``ask_service`` runs. What survives client-side is the reply check in
    :meth:`_hits_of` and the server's own refusal, so a wrong-lane answer is still caught here;
    what is delegated is the selection of the lane.
    """

    #: The local end of the ssh forward, e.g. ``http://127.0.0.1:18799``. Never a remote address:
    #: the tunnel terminates on this host, so nothing has to be exposed on a real interface.
    url: str = ""

    @property
    def kind(self) -> str:
        return "service_http"

    def _ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        import json as _json
        import urllib.error
        import urllib.request

        if not self.url:
            raise RetrievalServiceError(
                "HttpServiceBackend needs `url` (the local end of the ssh -L forward). There is "
                "no default: guessing a port would silently address whatever else is listening."
            )
        body = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310 - a localhost tunnel endpoint, not user input
            self.url.rstrip("/") + "/retrieve",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                return _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 503 from the bridge means the service is unreachable or stale: infrastructure, not
            # a format verdict, so raise the stale error and let the caller retry.
            detail = exc.read().decode("utf-8", "replace")[:200] if exc.fp else ""
            if exc.code == 503:
                raise StaleServiceError(
                    f"bridge reports the service unavailable: {detail}"
                ) from exc
            raise RetrievalServiceError(f"bridge HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # The tunnel is down. Infrastructure again, never a ranking result.
            raise StaleServiceError(
                f"cannot reach the retrieval bridge at {self.url} ({type(exc).__name__}: {exc}). "
                "Is the `ssh -L` forward up? Note a multiplexed ssh session silently ignores -L, "
                "so the forward must be made with ControlPath=none."
            ) from exc
