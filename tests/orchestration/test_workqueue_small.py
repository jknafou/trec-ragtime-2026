"""Work-queue lifecycle, including the heartbeat reaper and the poison counter.

The lifecycle is an atomic first-mv-wins claim, validate before done, and a USR1 requeue,
plus two further primitives: a heartbeat with a staleness reaper, and a persisted poison
counter that routes a repeatedly failing shard to ``failed/``. Each has its own non-vacuous
assertion. The ``frozen_now`` fixture freezes ``workqueue._now`` so staleness is
deterministic without real sleeps, and every path is asserted to stay under the
caller-supplied ``wq_root`` base rather than a hardcoded ``/tmp``.
"""

from __future__ import annotations

import ast
import inspect
import threading
from pathlib import Path

import pytest

from ragtime.common import read_jsonl
from ragtime.common.io import success_marker
from ragtime.orchestration.slurm import workqueue

pytestmark = pytest.mark.small


def _under(path: Path, base: Path) -> bool:
    return base.resolve() in path.resolve().parents or path.resolve() == base.resolve()


# --------------------------------------------------------------------------- #
# a. Atomic claim: the first mv wins under a real thread race.
# --------------------------------------------------------------------------- #
def test_claim_first_mv_wins_under_race(wq_root) -> None:
    d = wq_root.dirs
    wq_root.drop_shards(1)  # exactly one shard both threads fight for

    results: list[Path | None] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def _grab() -> None:
        barrier.wait()
        got = workqueue.claim(d.pending, d.running)
        with lock:
            results.append(got)

    threads = [threading.Thread(target=_grab) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [r for r in results if r is not None]
    assert len(winners) == 1  # exactly one claimer owns the shard
    assert results.count(None) == 1  # the loser's rename raised -> None
    assert len(list(d.running.glob("shard_*"))) == 1  # in running exactly once
    assert list(d.pending.glob("shard_*")) == []  # not left in pending
    assert _under(winners[0], d.base)


def test_claim_returns_none_when_empty(wq_root) -> None:
    d = wq_root.dirs
    assert workqueue.claim(d.pending, d.running) is None


# --------------------------------------------------------------------------- #
# b. Heartbeat: the sidecar's recorded time advances.
# --------------------------------------------------------------------------- #
def test_heartbeat_advances_recorded_time(wq_root, frozen_now) -> None:
    d = wq_root.dirs
    wq_root.drop_shards(1)
    claimed = workqueue.claim(d.pending, d.running)
    assert claimed is not None

    hb = workqueue.heartbeat(claimed)
    t0 = float(hb.read_text().strip())
    frozen_now.tick(37.0)
    workqueue.heartbeat(claimed)
    t1 = float(hb.read_text().strip())
    assert t1 > t0
    assert _under(hb, d.base)


# --------------------------------------------------------------------------- #
# c. The reaper reclaims a stale running/ shard.
# --------------------------------------------------------------------------- #
def test_reaper_reclaims_stale_running_shard(wq_root, frozen_now) -> None:
    d = wq_root.dirs
    wq_root.drop_shards(1)
    claimed = workqueue.claim(d.pending, d.running)  # stamps heartbeat at now
    assert claimed is not None

    max_age = 100.0
    frozen_now.tick(max_age + 1.0)  # heartbeat is now older than max_age
    reclaimed = workqueue.reap_stale(d.running, d.pending, max_age)

    assert len(reclaimed) == 1
    assert list(d.running.glob("shard_*")) == []  # left running/
    assert len(list(d.pending.glob("shard_*"))) == 1  # back in pending/, NOT failed/
    assert list(d.failed.glob("shard_*")) == []


def test_reaper_leaves_fresh_shard_alone(wq_root, frozen_now) -> None:
    d = wq_root.dirs
    wq_root.drop_shards(1)
    workqueue.claim(d.pending, d.running)
    frozen_now.tick(10.0)  # well within max_age
    assert workqueue.reap_stale(d.running, d.pending, max_age=100.0) == []
    assert len(list(d.running.glob("shard_*"))) == 1


# --------------------------------------------------------------------------- #
# d. Validate-before-done -> _SUCCESS(key).
# --------------------------------------------------------------------------- #
def test_validate_gate_blocks_and_then_completes(wq_root) -> None:
    d = wq_root.dirs
    wq_root.drop_shards(1)
    claimed = workqueue.claim(d.pending, d.running)

    # Failing validation must not mark done; shard stays in running/.
    assert workqueue.mark_done(claimed, d.done, key="k-1", validate=lambda _p: False) is None
    assert len(list(d.running.glob("shard_*"))) == 1
    assert list(d.done.glob("shard_*")) == []

    # Passing validation completes: moved to done/ + a durable, key-bearing result.
    dst = workqueue.mark_done(claimed, d.done, key="k-1", validate=lambda _p: True)
    assert dst is not None and dst.parent == d.done
    result = d.done / f"{dst.name}.result"
    assert success_marker(result).exists()  # common.io durability marker
    assert read_jsonl(result)[0]["key"] == "k-1"  # keyed by the shard's key
    assert _under(dst, d.base)


def test_mark_done_records_the_observed_hardware_provenance(wq_root) -> None:
    """A heterogeneous GPU OR-set is not byte-homogeneous: 464 of 512 sentences were
    identical between a Blackwell and a 5090 at identical bucket composition, so the result
    row must be able to name the silicon. This is additive only: shard and key are untouched,
    and empty values are dropped rather than written as blanks."""
    d = wq_root.dirs
    wq_root.drop_shards(1)
    claimed = workqueue.claim(d.pending, d.running)
    dst = workqueue.mark_done(
        claimed,
        d.done,
        key="k-1",
        provenance={"gpu": "NVIDIA H100 NVL", "node": "gpu004", "array_task": ""},
    )
    assert dst is not None
    row = read_jsonl(d.done / f"{dst.name}.result")[0]
    assert row["shard"] == dst.name and row["key"] == "k-1"
    assert row["gpu"] == "NVIDIA H100 NVL" and row["node"] == "gpu004"
    assert "array_task" not in row  # an empty observation is a non-observation


def test_worker_provenance_reads_only_observed_env_facts() -> None:
    """Machine facts read off the running job, never a run choice; those live in config."""
    from ragtime.orchestration import saturate

    got = saturate.worker_provenance(
        {
            "RAGTIME_GPU_MODEL": " NVIDIA GeForce RTX 5090 ",
            "SLURMD_NODENAME": "gpu009",
            "SLURM_JOB_ID": "4176640",
            "SLURM_ARRAY_TASK_ID": "",
        }
    )
    assert got == {
        "gpu": "NVIDIA GeForce RTX 5090",
        "node": "gpu009",
        "job_id": "4176640",
    }
    assert saturate.worker_provenance({}) == {}  # off SLURM: silent, not a fabricated row


# --------------------------------------------------------------------------- #
# e. A poison shard reaches failed/ after K attempts, with the counter persisted on disk.
# --------------------------------------------------------------------------- #
def test_poison_shard_moves_to_failed_after_k_attempts(wq_root) -> None:
    d = wq_root.dirs
    wq_root.drop_shards(1)
    k_max = 3

    for attempt in range(1, k_max):  # attempts 1..K-1 stay retriable
        claimed = workqueue.claim(d.pending, d.running)
        assert claimed is not None
        dst = workqueue.fail(claimed, d.pending, d.failed, k_max=k_max)
        assert dst.parent == d.pending  # requeued for another try
        # Counter is read fresh from the meta/ sidecar (survives a process restart).
        assert workqueue.attempts(dst) == attempt

    # The K-th failure is the hard stop -> failed/.
    claimed = workqueue.claim(d.pending, d.running)
    dst = workqueue.fail(claimed, d.pending, d.failed, k_max=k_max)
    assert dst.parent == d.failed
    assert workqueue.attempts(dst) == k_max
    assert list(d.failed.glob("shard_*"))  # non-empty: not an infinite pending<->running loop
    assert list(d.pending.glob("shard_*")) == []


# --------------------------------------------------------------------------- #
# f. USR1 requeue helper returns an in-flight shard to pending.
# --------------------------------------------------------------------------- #
def test_usr1_requeue_returns_claimed_shard_to_pending(wq_root) -> None:
    d = wq_root.dirs
    wq_root.drop_shards(1)
    claimed = workqueue.claim(d.pending, d.running)
    dst = workqueue.requeue(claimed, d.pending)
    assert dst.parent == d.pending
    assert list(d.running.glob("shard_*")) == []  # not left claimed / lost


# --------------------------------------------------------------------------- #
# g. The module hardcodes no /tmp path.
# --------------------------------------------------------------------------- #
def test_workqueue_never_hardcodes_a_tmp_path() -> None:
    """No code path builds a ``/tmp`` string or reaches for ``tempfile`` (the docstring may
    still mention /tmp in prose; this scans literals, not prose)."""
    tree = ast.parse(inspect.getsource(workqueue))
    module_doc = ast.get_docstring(tree, clean=False)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == module_doc:
                continue  # the module docstring explains the /tmp gotcha in prose
            assert "/tmp" not in node.value
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, "names", [])]
            assert "tempfile" not in names
