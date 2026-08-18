"""How the retrieval service assigns work, and the two things that assignment may never change.

Within one request the dispatch was already least-busy: :meth:`ReplicaPool.run_batch` hands the
next job to whichever connection is idle. One level up it was not. The serve loop ran ``for
req_path in pending: self._handle(...)``, one whole request at a time, so N-1 replicas idled
through every single-query search, and a single-query search is the production shape, because the
k RAG loops each issue their own. The ``served_by`` counts never showed it: each request was
perfectly balanced across the replicas it used, and it used one.

Two invariants are checked harder than the speed-up is, because a scheduling change that breaks
either of them is worse than the blocking it removes:

* Reply order equals submission order. Callers zip scores onto candidate ids positionally and RRF
  accumulates floats, so a reordered reply raises nothing; it silently re-pairs queries with
  answers. Checked against the sequential one-replica path, element for element, with per-query
  costs shuffled so completion order is guaranteed to differ from submission order.
* A dead replica fails the request and names itself. Never a short list that looks complete.

The replicas are forked stub processes that sleep for a scripted number of seconds instead of
searching, driven through the real :class:`ReplicaPool` and the real serve loop over a real
filesystem queue with the real client (:func:`rsvc_registry.ask`). Only the search itself is
faked, which is the part that would otherwise need a GPU and a resident index.
"""

from __future__ import annotations

import os
import threading
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ragtime.devkit import rsvc
from ragtime.devkit.rsvc_registry import ask

pytestmark = pytest.mark.small

#: Scripted costs are in seconds and every test's wall is a multiple of them, so keep them small
#: enough that the file stays a smoke test and large enough that a 10 ms poll cannot blur them.
UNIT = 0.20


def _cost_of(job: dict[str, Any]) -> float:
    """A stub replica's "search time": carried on the job, or encoded in the query text.

    The second spelling exists because a request that goes through the real queue can only carry
    strings in ``query``: the service builds the job dict itself, which is the point.
    """
    if "cost" in job:
        return float(job["cost"])
    return float(str(job.get("query", "q|0")).rsplit("|", 1)[-1])


def _tag_of(job: dict[str, Any]) -> str:
    return str(job.get("tag") if "tag" in job else str(job.get("query", "")).rsplit("|", 1)[0])


class _StubFleet:
    """Stands in for ``Service`` at the one seam :class:`ReplicaPool` uses: ``_replica_body``.

    A replica here is a forked process that sleeps and answers, so the pool under test is the real
    one: real pipes, real ``multiprocessing.connection.wait``, real death when a child exits.
    """

    def __init__(self, die_on: str | None = None, err_on: str | None = None) -> None:
        self.die_on = die_on
        self.err_on = err_on

    def _replica_body(self, index: int, device: str, conn: Any) -> None:
        conn.send({"pid": os.getpid()})
        while True:
            try:
                job = conn.recv()
            except (EOFError, KeyboardInterrupt):
                return
            if job is None:
                return
            tag = _tag_of(job)
            if self.die_on is not None and tag == self.die_on:
                os._exit(7)  # a replica that dies, not one that raises: no reply ever comes back
            if self.err_on is not None and tag == self.err_on:
                conn.send({"error": "ValueError: scripted query failure", "traceback": "",
                           "replica": index, "replica_device": device})
                continue
            cost = _cost_of(job)
            time.sleep(cost)
            conn.send({
                "wall_s": cost, "replica": index, "replica_device": device,
                "tag": tag, "query": job.get("query"), "timing": {}, "vram": {},
            })


#: Every pool this module opens, so teardown can reap them. See :func:`_reap_pools`.
_OPEN: list[rsvc.ReplicaPool] = []


def _pool(n: int, **kw) -> rsvc.ReplicaPool:
    pool = rsvc.ReplicaPool(
        _StubFleet(**kw), [f"cuda:{i}" for i in range(n)], lambda *a, **k: None
    )
    _OPEN.append(pool)
    return pool


@pytest.fixture(autouse=True)
def _reap_pools():
    """Close every pool a test opened, whatever the test did.

    Replica processes are non-daemonic, because a replica forks its own cell workers and a daemonic
    process may not have children, so ``multiprocessing``'s exit handler joins them. A pool left
    open therefore does not leak; it wedges the interpreter forever on a child blocked reading a
    pipe nobody will write to again, which presents as a suite that prints every dot and then
    hangs. ``close`` is idempotent, so a test may still close its own.
    """
    yield
    while _OPEN:
        _OPEN.pop().close()


class _StubService(rsvc.Service):
    """The real serve loop and the real request handling, over a stub fleet.

    ``__init__`` does not call ``super().__init__``: everything it would build is a
    GPU stack. What the methods under test actually touch is exactly the six attributes set here,
    so a stub that sets them exercises the production code path rather than a copy of it.
    """

    def __init__(self, pool: rsvc.ReplicaPool, *, idle_exit: float = 1.5) -> None:
        self.args = types.SimpleNamespace(poll=0.01, idle_exit=idle_exit)
        self.ctx = types.SimpleNamespace(
            index="original", knobs=types.SimpleNamespace(rerank_depth=100)
        )
        self.index_hash = "a" * 64
        self.cold_s = 1.0
        self.replicas = pool
        self.log_path = None

    def _log(self, event: str, **kw: Any) -> None:
        pass


def _serving(pool: rsvc.ReplicaPool, queue: Path, **kw) -> tuple[threading.Thread, _StubService]:
    qin, qout = queue / "in", queue / "out"
    qin.mkdir(parents=True, exist_ok=True)
    qout.mkdir(parents=True, exist_ok=True)
    svc = _StubService(pool, **kw)
    thread = threading.Thread(target=svc._serve_multiplexed, args=(qin, qout), daemon=True)
    thread.start()
    return thread, svc


def _q(tag: str, cost: float) -> str:
    return f"{tag}|{cost}"


# --------------------------------------------------------------------------- #
# 1. The invariant: which replica runs a query may change; where its answer lands may not.
# --------------------------------------------------------------------------- #
def test_reply_order_equals_submission_order_when_costs_are_shuffled() -> None:
    """Costs chosen so completion order cannot equal submission order, on 3 replicas.

    The first job is the slowest, so it finishes last; if results were appended as they completed,
    element 0 would be someone else's answer and nothing downstream would notice: the client zips
    scores onto candidate ids positionally.
    """
    costs = [6, 1, 1, 5, 1, 1, 4, 1, 1][:9]
    jobs = [{"tag": f"j{i}", "cost": c * UNIT / 2} for i, c in enumerate(costs)]
    pool = _pool(3)
    try:
        got = pool.run_batch(jobs)
    finally:
        pool.close()
    assert [r["tag"] for r in got] == [j["tag"] for j in jobs]
    assert [r["wall_s"] for r in got] == [j["cost"] for j in jobs]
    # And more than one replica really did answer, so the ordering above is not trivially true.
    assert len({r["replica"] for r in got}) > 1


def test_the_parallel_reply_is_identical_to_the_sequential_one() -> None:
    """Same jobs, fleet width 1 vs 3: the two replies must agree element for element.

    This is the strongest form of the ordering check available without a GPU: it compares the
    dispatch policy against the path that has no dispatch policy at all.
    """
    jobs = [{"tag": f"j{i}", "cost": c * UNIT / 4}
            for i, c in enumerate([7, 1, 2, 1, 6, 1, 1, 3, 1, 1, 5, 1])]
    pool = _pool(3)
    try:
        one = pool.run_batch(jobs, 1)
        many = pool.run_batch(jobs)
    finally:
        pool.close()
    assert [r["tag"] for r in one] == [r["tag"] for r in many] == [j["tag"] for j in jobs]
    assert all(r["replica"] == 0 for r in one)  # `use_replicas=1` really is one replica


# --------------------------------------------------------------------------- #
# 2. The policy inside one request: least-busy.
# --------------------------------------------------------------------------- #
def test_within_a_request_a_free_replica_takes_the_next_job_rather_than_waiting_its_turn() -> None:
    """One 10x job followed by nine 1x jobs, on 2 replicas.

    Round-robin would give replica 0 jobs 0,2,4,6,8, the slow one plus four fast ones, for a
    makespan of 10+4 = 14 units while replica 1 finished in 5 and idled. Least-busy gives replica 0
    the slow job alone and replica 1 all nine fast ones: 10 vs 9, makespan 10. So the two policies
    are told apart by the ``served_by`` counts, which is the discriminator a balanced count on
    uniform queries cannot provide.
    """
    jobs = [{"tag": "slow", "cost": 10 * UNIT}] + [
        {"tag": f"f{i}", "cost": UNIT} for i in range(9)
    ]
    pool = _pool(2)
    started = time.perf_counter()
    try:
        got = pool.run_batch(jobs)
    finally:
        pool.close()
    wall = time.perf_counter() - started
    counts = Counter(r["replica"] for r in got)
    assert counts[got[0]["replica"]] == 1, f"the slow replica took more than the slow job: {counts}"
    assert max(counts.values()) == 9, f"work was not stolen by the free replica: {counts}"
    # Round-robin would be >= 14 units; least-busy is ~10. Generous margin: this is a sleep-based
    # test on a shared node, and the two policies differ by 40 %, far outside the noise.
    assert wall < 12 * UNIT, f"wall {wall:.2f}s looks like a static split, not least-busy"


# --------------------------------------------------------------------------- #
# 3. The policy across requests: head-of-line blocking.
# --------------------------------------------------------------------------- #
def test_concurrent_single_query_requests_use_the_whole_fleet(tmp_path: Path) -> None:
    """Four clients, one query each, 2 replicas: the k-RAG-loop shape.

    Serialized -- one pending request at a time -- this is 4 costs end to end with one card idle
    throughout. Multiplexed it is 2. The assertion is on the wall and on both replicas appearing,
    because either alone can be satisfied by accident.
    """
    queue = tmp_path / "rsvc"
    thread, _ = _serving(_pool(2), queue)
    replies: list[dict[str, Any]] = []
    lock = threading.Lock()

    def client(i: int) -> None:
        got = ask(queue, {"query": _q(f"c{i}", UNIT * 4), "top_k": 5}, timeout_s=60, poll_s=0.01)
        with lock:
            replies.append(got)

    started = time.perf_counter()
    threads = [threading.Thread(target=client, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    wall = time.perf_counter() - started
    thread.join(timeout=20)

    assert len(replies) == 4 and all(r["ok"] for r in replies), replies
    served = [r["summary"]["served_by"][0] for r in replies]
    assert set(served) == {0, 1}, f"only one replica answered four concurrent clients: {served}"
    # Serialized would be >= 16 UNIT; multiplexed is ~8 UNIT plus poll overhead.
    assert wall < 13 * UNIT, f"wall {wall:.2f}s, the fleet is still serving one request at a time"


def test_a_one_query_request_does_not_wait_out_a_long_batch(tmp_path: Path) -> None:
    """Fair-queueing across requests: a search arriving behind a 10-query batch is not stuck.

    FIFO over requests would move the blocking up a level rather than removing it: the k loops
    would still queue behind whichever loop happened to submit a big batch first.

    One replica, which makes the discriminator exact rather than lucky. With two, the
    replicas tend to free up together and even a head-of-list dispatcher hands the second slot to
    the newcomer; with one, only a genuine round-robin over requests lets the single query through
    after ~1 job instead of after all 10.
    """
    queue = tmp_path / "rsvc"
    thread, _ = _serving(_pool(1), queue)
    out: dict[str, Any] = {}

    def big() -> None:
        out["big"] = ask(
            queue,
            {"query": [_q(f"b{i}", UNIT) for i in range(10)], "top_k": 5},
            timeout_s=60, poll_s=0.01,
        )

    def small() -> None:
        t0 = time.perf_counter()
        out["small"] = ask(queue, {"query": _q("s", UNIT), "top_k": 5}, timeout_s=60, poll_s=0.01)
        out["small_latency"] = time.perf_counter() - t0

    tb = threading.Thread(target=big)
    tb.start()
    time.sleep(UNIT / 2)  # arrive after the batch has the fleet
    ts = threading.Thread(target=small)
    ts.start()
    tb.join(timeout=60)
    ts.join(timeout=60)
    thread.join(timeout=20)

    assert out["big"]["ok"] and out["small"]["ok"]
    assert len(out["big"]["runs"]) == 10
    # Behind a 10-job batch on one replica, FIFO would make this wait ~10 UNIT. Fair-queued it
    # waits for the one job in flight, then takes the replica next.
    assert out["small_latency"] < 4 * UNIT, (
        f"the single query waited {out['small_latency']:.2f}s behind the batch, "
        f"queued_s={out['small']['summary']['queued_s']}"
    )


def test_multiplexed_replies_keep_their_own_submission_order(tmp_path: Path) -> None:
    """A shuffled-cost batch answered while other requests share the fleet still comes back in
    submission order. This is the invariant under exactly the condition that stresses it: results
    from several requests interleave on the same connections."""
    queue = tmp_path / "rsvc"
    thread, _ = _serving(_pool(3), queue)
    out: dict[str, Any] = {}
    costs = [6, 1, 1, 4, 1, 1, 3, 1]

    def big() -> None:
        out["big"] = ask(
            queue,
            {"query": [_q(f"b{i}", c * UNIT / 3) for i, c in enumerate(costs)], "top_k": 5},
            timeout_s=60, poll_s=0.01,
        )

    def noise(i: int) -> None:
        ask(queue, {"query": _q(f"n{i}", UNIT / 2), "top_k": 5}, timeout_s=60, poll_s=0.01)

    threads = [threading.Thread(target=big)] + [
        threading.Thread(target=noise, args=(i,)) for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    thread.join(timeout=20)

    runs = out["big"]["runs"]
    assert [r["tag"] for r in runs] == [f"b{i}" for i in range(len(costs))]
    assert [r["wall_s"] for r in runs] == [c * UNIT / 3 for c in costs]


def test_a_request_naming_use_replicas_owns_the_fleet(tmp_path: Path) -> None:
    """``use_replicas`` is a measurement of fleet width, so it must not be measured while sharing.

    Without the exclusivity rule, "1 replica" would mean "1 replica, plus whatever else happened
    to be in flight on the other card", and the 1-vs-N comparison the flag exists for would be
    quietly worthless.
    """
    queue = tmp_path / "rsvc"
    thread, _ = _serving(_pool(2), queue)
    out: dict[str, Any] = {}

    def measured() -> None:
        out["m"] = ask(
            queue,
            {"query": [_q(f"m{i}", UNIT) for i in range(4)], "top_k": 5, "use_replicas": 1},
            timeout_s=60, poll_s=0.01,
        )

    def other(i: int) -> None:
        out[f"o{i}"] = ask(queue, {"query": _q(f"o{i}", UNIT), "top_k": 5},
                           timeout_s=60, poll_s=0.01)

    threads = [threading.Thread(target=measured)] + [
        threading.Thread(target=other, args=(i,)) for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    thread.join(timeout=20)

    m = out["m"]
    assert m["ok"], m.get("error")
    assert m["summary"]["served_by"] == [0, 0, 0, 0], m["summary"]
    assert m["summary"]["used_replicas"] == 1
    # Its wall is its own four queries end to end, not four plus a share of the neighbours'.
    assert m["summary"]["batch_wall_s"] < 6 * UNIT, m["summary"]
    assert all(out[f"o{i}"]["ok"] for i in range(3))


# --------------------------------------------------------------------------- #
# 4. Failures stay loud and attributable.
# --------------------------------------------------------------------------- #
def test_a_replica_that_dies_mid_batch_raises_naming_it() -> None:
    """Never a short list. The message must name the replica and its card."""
    jobs = [{"tag": f"j{i}", "cost": UNIT / 4} for i in range(6)]
    jobs[3]["tag"] = "poison"
    pool = _pool(2, die_on="poison")
    try:
        with pytest.raises(rsvc.ReplicaDied) as caught:
            pool.run_batch(jobs)
        msg = str(caught.value)
        assert "cuda:" in msg and "died" in msg
        assert caught.value.index in (0, 1)
        # And the pool is now poisoned: a service that kept answering from the survivors would be
        # advertising a fleet it no longer has.
        with pytest.raises(RuntimeError, match="refusing to dispatch"):
            pool.dispatch("after", {"tag": "after", "cost": 0.0})
    finally:
        pool.close()


def test_a_query_that_raises_fails_its_own_request_and_names_the_replica(tmp_path: Path) -> None:
    """A query error leaves the replica alive, so it must fail one request: not the fleet, and
    not silently. The neighbouring request sharing the fleet must still be answered."""
    queue = tmp_path / "rsvc"
    thread, _ = _serving(_pool(2, err_on="bad"), queue)
    out: dict[str, Any] = {}

    def bad() -> None:
        out["bad"] = ask(queue, {"query": _q("bad", UNIT), "top_k": 5}, timeout_s=60, poll_s=0.01)

    def good() -> None:
        out["good"] = ask(queue, {"query": _q("good", UNIT), "top_k": 5},
                          timeout_s=60, poll_s=0.01)

    threads = [threading.Thread(target=bad), threading.Thread(target=good)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    thread.join(timeout=20)

    assert out["bad"]["ok"] is False
    assert "replica" in out["bad"]["error"] and "scripted query failure" in out["bad"]["error"]
    assert out["good"]["ok"] is True, "one bad query took its neighbour down with it"


def test_a_request_with_no_queries_is_refused_rather_than_answered_empty(tmp_path: Path) -> None:
    """An empty ``runs`` list reads downstream as a successful search that found nothing."""
    queue = tmp_path / "rsvc"
    thread, _ = _serving(_pool(2), queue)
    got = ask(queue, {"query": [], "top_k": 5}, timeout_s=60, poll_s=0.01)
    thread.join(timeout=20)
    assert got["ok"] is False and "no queries" in got["error"]


def test_a_request_for_another_rendering_is_still_refused(tmp_path: Path) -> None:
    """A request naming another rendering is refused wherever the request is parsed.

    A result from the wrong index raises nothing downstream; it is silently the wrong experiment.
    So the refusal is asserted against the accepting path, not against one function.
    """
    queue = tmp_path / "rsvc"
    thread, _ = _serving(_pool(2), queue)
    got = ask(queue, {"query": _q("x", UNIT), "index": "omt"}, timeout_s=60, poll_s=0.01)
    thread.join(timeout=20)
    assert got["ok"] is False and "refusing to answer from a different rendering" in got["error"]
