"""The per-pair vLLM registry: N instances, each leased to exactly one worker.

A single shared endpoint file is fine with one instance, but with a fan of instances it makes
the fan pointless: every worker reads the same file, so N workers pile onto one GPU pair. That
is not merely a throughput loss, because sampled decoding is not batch-invariant. The same seed
produces a different round-0 nugget bank when a second client shares the instance, and the same
bank twice when it does not, so co-tenancy also destroys the reproducibility the fairness
invariant depends on.

The contract this module enforces is therefore that a vLLM pair is exclusive to one worker for
the duration of a topic. That is the opposite of ``retrieval.endpoints``, where one resident
index is shared by many clients: retrieval returns ids and scores and does no sampling, so a
co-tenant there costs latency and nothing else.

The exclusion mechanism is ``orchestration.slurm.workqueue.claim``, an atomic rename where the
first rename wins, the same protocol the translation fan and the topic fan already run on. An
endpoint descriptor sits in ``free/`` when unleased and is moved to ``leased/`` by whichever
worker gets there first; releasing moves it back. A worker that dies without releasing is
reclaimed by ``reap_stale``, exactly as an abandoned shard is, because the lease carries the
same heartbeat. "Who owns this pair" and "who owns this topic" are the same question asked
twice, and answering it twice in two different ways is how the answers drift apart.

A descriptor carries:

* ``url``: where to send requests. The worker exports it as ``RAGTIME_VLLM_URL``.
* ``job_id`` and ``pair``: which allocation and which cards, so a run record can say where a
  cell was generated.
* ``model``: lets a worker refuse a lane serving a different checkpoint, the same class of
  refusal as retrieval's rendering filter, where the answer would look fine and come from the
  wrong system.
* ``gpu_model``: one GPU architecture per run family. Different architectures run different FP8
  kernels, and seeded generation is not reproducible across them, so a family that silently
  mixed architectures would put a kernel term inside the translation delta.
* ``heartbeat``: presence is not liveness. A descriptor outlives its job through walltime
  expiry, preemption or a node crash, and a stale one advertising a dead endpoint is how a
  worker blocks forever.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragtime.common import get_logger

__all__ = [
    "DEFAULT_MAX_AGE_S",
    "EndpointLease",
    "beat",
    "claim_endpoint",
    "endpoint_registry",
    "live_endpoints",
    "publish_endpoint",
    "release_endpoint",
    "unpublish_endpoint",
]

_log = get_logger("serving.vllm_registry")

#: A descriptor whose heartbeat is older than this is treated as dead when free, or abandoned
#: when leased.
#:
#: This is a liveness bound rather than a work bound, and the difference matters. A topic takes
#: tens of minutes, which is nowhere near this value, because what must beat is not the topic but
#: the holder; see :func:`beat`. Without a beat a healthy worker's lease would be reclaimed
#: mid-topic and its pair handed to a second worker, which is exactly the co-tenancy this module
#: exists to prevent. Both sides therefore beat: the vLLM job while its endpoint is free, and the
#: worker while it holds the lease.
#:
#: It is sized against the slower of the two beats. The service's supervise loop sleeps 60 s plus
#: its cycle work, so a 120 s bound would put one slow cycle on the threshold and start evicting
#: healthy pairs for no visible reason. 240 s is four times that cycle and eight times the
#: worker's beat, and the cost of the wider bound is cheap: a genuinely dead endpoint stays
#: visible for up to four minutes, and a worker that leases it fails against a dead socket rather
#: than corrupting anything.
DEFAULT_MAX_AGE_S = 240.0


@dataclass(frozen=True)
class EndpointLease:
    """One exclusively held vLLM endpoint. ``path`` is its file in ``leased/``."""

    url: str
    name: str
    job_id: str
    pair: str
    model: str
    gpu_model: str
    path: Path

    @property
    def env(self) -> dict[str, str]:
        """What a worker must export so its clients address this pair and no other."""
        return {"RAGTIME_VLLM_URL": self.url}


def endpoint_registry(base: str | os.PathLike[str]) -> Any:
    """Return the free/leased queue for endpoints, reusing ``workqueue.init_queue``.

    ``pending`` is ``free`` and ``running`` is ``leased``. The names differ in this module's
    prose because "a pending endpoint" reads wrongly, but they are the same directories the
    claim and reap protocol already operates on, so there is no second scheme.
    """
    from ragtime.orchestration.slurm.workqueue import init_queue

    return init_queue(base)


def publish_endpoint(
    base: str | os.PathLike[str],
    *,
    name: str,
    url: str,
    job_id: str,
    pair: str,
    model: str,
    gpu_model: str,
) -> Path:
    """Advertise one live instance as claimable, called by the vLLM job once it answers.

    Publish only after the endpoint has served a real request: a launcher that publishes while
    the engine has crashed leaves a descriptor that is a promise a worker will block on.

    The call is idempotent and must not resurrect a leased descriptor into ``free/``. A vLLM job
    that re-publishes on a loop, which is the obvious way to write a keep-alive, would otherwise
    put a second free copy of an endpoint a worker is actively using back on the shelf, a
    double-booking manufactured by the liveness mechanism itself. A name already present in
    ``leased/`` is therefore beaten rather than re-published: :func:`beat` is the keep-alive and
    this is the announcement.
    """
    wq = endpoint_registry(base)
    leased = Path(wq.running) / f"{name}.json"
    if leased.exists():
        beat(base, name)
        return leased
    doc = {
        "name": name, "url": url, "job_id": job_id, "pair": pair,
        "model": model, "gpu_model": gpu_model,
        "heartbeat": time.time(), "published": time.time(),
    }
    path = Path(wq.pending) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_doc(path, doc)
    _log.info("serving.vllm_registry.published", name=name, url=url, pair=pair, gpu=gpu_model)
    return path


def beat(base: str | os.PathLike[str], name: str) -> Path | None:
    """Re-stamp one descriptor's liveness wherever it sits, returning ``None`` if it is gone.

    It is called by the vLLM job while its endpoint sits free, so ``live_endpoints`` keeps
    offering it, and by the worker while it holds the lease, so ``reap_stale`` does not reclaim
    a live topic. Two callers share one function, because "is this thing alive" must not have
    two answers.

    It stamps two places as one mechanism: the descriptor's own ``heartbeat`` field, which
    :func:`live_endpoints` reads for a free endpoint, and the work queue's heartbeat sidecar,
    which ``reap_stale`` reads for a leased one. The sidecar lives in the queue's shared
    ``meta/`` directory and is addressed identically from either side, so a descriptor keeps its
    liveness history across a claim.
    """
    from ragtime.orchestration.slurm.workqueue import heartbeat

    wq = endpoint_registry(base)
    for d in (wq.pending, wq.running):
        path = Path(d) / f"{name}.json"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        doc["heartbeat"] = time.time()
        _atomic_write_doc(path, doc)
        heartbeat(path)
        return path
    return None


def _atomic_write_doc(path: Path, doc: dict[str, Any]) -> None:
    """Write via temp file and rename, so a reader sees the old or new file, never a torn one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def live_endpoints(
    base: str | os.PathLike[str],
    *,
    model: str | None = None,
    gpu_model: str | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> list[dict[str, Any]]:
    """Return every free endpoint that is beating and matches the wanted lane.

    ``gpu_model`` filtering makes the one-architecture-per-family rule mechanical rather than
    remembered: pass the family's architecture and a worker cannot be handed a pair that would
    generate with a different FP8 kernel.
    """
    wq = endpoint_registry(base)
    out = []
    for path in sorted(Path(wq.pending).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # being replaced; next scan
        if (time.time() - float(doc.get("heartbeat", 0))) > max_age_s:
            continue
        if model is not None and doc.get("model") != model:
            continue
        if gpu_model is not None and doc.get("gpu_model") != gpu_model:
            continue
        doc["_path"] = str(path)
        out.append(doc)
    return out


def claim_endpoint(
    base: str | os.PathLike[str],
    *,
    model: str | None = None,
    gpu_model: str | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> EndpointLease | None:
    """Take exclusive ownership of one endpoint, or return ``None`` if none is free.

    It first reclaims abandoned leases from workers that died mid-topic, then claims by atomic
    rename. When two workers race the same descriptor exactly one rename succeeds, so the pair
    cannot be double-booked, which is why this is a claim rather than a read.
    """
    from ragtime.orchestration.slurm.workqueue import claim, reap_stale, requeue

    wq = endpoint_registry(base)
    reclaimed = reap_stale(wq.running, wq.pending, max_age=max_age_s)
    if reclaimed:
        _log.info("serving.vllm_registry.reclaimed", count=len(reclaimed))
    # `claim` takes the first claimable file and offers no skip list, and the lane filter can
    # only be applied after reading the descriptor, so a scan has to get a rejected file out of
    # `claim`'s way without giving it away.
    #
    # A wrong-lane descriptor is therefore held, staying in `leased/` for the rest of the scan
    # and released in the `finally`. Requeueing it immediately does not terminate, because it
    # goes back exactly where `claim` looks first and the loop re-claims the same file forever.
    # Remembering the name instead does not work either: seeing a name twice means "this one
    # again" rather than "we have been all the way round", and short-circuiting there hides an
    # endpoint behind an alphabetically earlier one from another lane.
    #
    # The cost is that a concurrent worker in the other lane cannot see the held descriptors for
    # the duration of the scan. That window is a few renames, it always closes, and a worker
    # that misses it simply gets `None` and retries.
    held: list[Path] = []
    try:
        while True:
            got = claim(wq.pending, wq.running)
            if got is None:
                return None
            try:
                doc = json.loads(Path(got).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # Not ours to keep and not ours to delete: hold it and let the scan go on.
                held.append(got)
                continue
            if (model is not None and doc.get("model") != model) or (
                gpu_model is not None and doc.get("gpu_model") != gpu_model
            ):
                held.append(got)
                _log.info("serving.vllm_registry.wrong_lane_held", name=doc.get("name"))
                continue
            _log.info("serving.vllm_registry.leased", name=doc.get("name"), url=doc.get("url"))
            return EndpointLease(
                url=str(doc.get("url") or ""), name=str(doc.get("name") or ""),
                job_id=str(doc.get("job_id") or ""), pair=str(doc.get("pair") or ""),
                model=str(doc.get("model") or ""), gpu_model=str(doc.get("gpu_model") or ""),
                path=Path(got),
            )
    finally:
        # Unconditional: on the `None` return, on a successful lease, and on an exception. A held
        # descriptor never released is a pair permanently removed from the fleet by a filter.
        for path in held:
            requeue(path, wq.pending)


def release_endpoint(base: str | os.PathLike[str], lease: EndpointLease) -> Path:
    """Give the pair back so another worker can take it. The counterpart of `claim_endpoint`."""
    from ragtime.orchestration.slurm.workqueue import requeue

    wq = endpoint_registry(base)
    out = requeue(lease.path, wq.pending)
    _log.info("serving.vllm_registry.released", name=lease.name)
    return out


def unpublish_endpoint(base: str | os.PathLike[str], name: str) -> None:
    """Withdraw an instance entirely, because the vLLM job is going away.

    It is removed from both directories: an instance that dies while leased must not be left
    advertised, since a stale descriptor is what makes a worker block on a dead endpoint.
    """
    wq = endpoint_registry(base)
    for d in (wq.pending, wq.running):
        (Path(d) / f"{name}.json").unlink(missing_ok=True)
    _log.info("serving.vllm_registry.unpublished", name=name)


def normalize_gpu_model(raw: str) -> str:
    """Normalize a GPU name, turning ``NVIDIA RTX PRO 6000`` into ``nvidia_rtx_pro_6000``.

    The lane is a string equality test, so what matters is that the publisher and the claimer
    spell an architecture the same way, not that the spelling matches SLURM's GRES name. One
    normalizer applied at publish time makes that true without anyone having to remember it.
    """
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw.strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# A small CLI, so a launcher script publishes through this code rather than a shell copy of the
# descriptor schema.
def main(argv: list[str] | None = None) -> int:
    """``python -m ragtime.serving.vllm_registry <publish|beat|unpublish|list> ...``

    The service is a shell script and the descriptor is a contract. Writing the JSON with a
    heredoc would put a second, silently drifting definition of that contract beside this one,
    so the script shells out here instead.
    """
    import argparse

    p = argparse.ArgumentParser(prog="vllm_registry")
    p.add_argument("action", choices=("publish", "beat", "unpublish", "list"))
    p.add_argument("--base", required=True, help="the registry dir (Layout.vllm_registry_dir())")
    p.add_argument("--name", default="", help="descriptor name; defaults to the job id")
    p.add_argument("--url", default="")
    p.add_argument("--job-id", default="")
    p.add_argument("--pair", default="")
    p.add_argument("--model", default="")
    p.add_argument("--gpu-model", default="", help="raw GPU name; normalized before it is stored")
    a = p.parse_args(argv)

    name = a.name or a.job_id
    if a.action == "publish":
        if not (name and a.url and a.model and a.gpu_model):
            p.error("publish needs --name/--job-id, --url, --model and --gpu-model")
        gpu = normalize_gpu_model(a.gpu_model)
        publish_endpoint(
            a.base, name=name, url=a.url, job_id=a.job_id, pair=a.pair,
            model=a.model, gpu_model=gpu,
        )
        # Printed because it is the value an operator types into RAGTIME_VLLM_GPU_MODEL to pin a
        # lane, and a normalized string nobody can see is a string nobody can match.
        print(f"published {name} url={a.url} gpu_model={gpu}")
    elif a.action == "beat":
        print("beat" if beat(a.base, name) else "GONE", name)
    elif a.action == "unpublish":
        unpublish_endpoint(a.base, name)
        print("unpublished", name)
    else:
        for d in live_endpoints(a.base):
            print(f"{d.get('name')}\t{d.get('gpu_model')}\t{d.get('url')}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin argv shim over the tested functions
    raise SystemExit(main())
