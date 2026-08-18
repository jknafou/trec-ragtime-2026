"""Send one query to a live retrieval service and print what it reports about the work it did.

    .venv/bin/python slurm/retrieval_probe.py "some query" \
        --registry logs/rsvc/registry --rendering omt_opus

Written for ``verify_service.sh``, which parses the three labelled lines below, and usable on its
own to read the per-leg cost of a single query.

Three ways a reply can be misread:

A reply that is not ``ok`` comes back in about a third of a second, which reads as a very fast
query to anything that only looks at the wall clock. So ``ok`` is asserted and a failure exits
non-zero.

The hits are nested one level down, under ``runs[0]``, because the service answers a batch. Reading
``rep["hits"]`` finds nothing and looks like a search that matched nothing.

The client wall includes time queued behind other clients. Under a busy pipeline fleet, a service
answering in 2.5 seconds has been measured at 21, 62 and 119 seconds of client wall. The ``TIMING``
legs are the service's own clocks; ``client_wall`` is printed beside them, never instead of them.

A request may carry its own ``rerank_depth``, and the service honours it. A probe that always sent
one would read its own argument back out of the reply, so ``--rerank-depth`` has no default here and
the reply reports the depth the service was configured with.

The first query after a boot pays lazy initialization and has been measured at 201 to 228 seconds.
Send one and discard it.
"""

from __future__ import annotations

import argparse
import json
import time

from ragtime.retrieval.endpoints import ask_service


def main() -> int:
    ap = argparse.ArgumentParser(description="query a live retrieval service once")
    ap.add_argument("query")
    ap.add_argument("--registry", default="logs/rsvc/registry",
                    help="directory of service descriptors")
    ap.add_argument("--rendering", default=None,
                    help="which searched rendering to resolve; unset takes whichever is live")
    ap.add_argument("--top-k", type=int, default=20)
    # No default. A request that carries a depth is answered at that depth, so a probe that always
    # sent one would read its own argument back out of the reply and report it as the service's
    # setting. Left unset, the service answers at its config's `retrieval.reranker.depth`, which is
    # what the pipeline client does and the only reading worth verifying.
    ap.add_argument("--rerank-depth", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--show", type=int, default=10, help="how many hits to print")
    a = ap.parse_args()

    request: dict[str, object] = {"query": a.query, "top_k": a.top_k}
    if a.rerank_depth is not None:
        request["rerank_depth"] = int(a.rerank_depth)

    print("QUERY:", a.query, flush=True)
    for i in range(1, a.repeat + 1):
        t = time.perf_counter()
        rep = ask_service(
            a.registry,
            dict(request),
            rendering=a.rendering,
            timeout_s=a.timeout,
        )
        wall = time.perf_counter() - t
        if not rep.get("ok"):
            print(f"NOT OK after {wall:.3f}s: {rep.get('error')}", flush=True)
            print(rep.get("traceback", ""), flush=True)
            return 3
        run = (rep.get("runs") or [{}])[0]
        svc = rep.get("_service") or {}
        print(f"=== reply {i}/{a.repeat} client_wall={wall:.3f}s "
              f"service={svc.get('name')} rendering={rep.get('rendering')} "
              f"fused_pool={run.get('fused_pool')}", flush=True)
        print("TIMING:", json.dumps(run.get("timing") or {}), flush=True)
        print("CPUCELL:", json.dumps(run.get("cpu_per_cell") or {}), flush=True)
        for rank, hit in enumerate((run.get("hits") or [])[: a.show], 1):
            pid, score = (hit["passage_id"], hit["score"]) if isinstance(hit, dict) else hit
            print(f"[{rank:2d}] {score:12.5f}  {pid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
