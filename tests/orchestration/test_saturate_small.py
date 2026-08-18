"""The saturate driver lifecycle, over a trivial adapter.

``seed``, ``run_worker`` and ``drive`` compose ``workqueue`` end to end for any
``StageAdapter``, which is the surface every preprocess stage reuses verbatim; they stay off
``/tmp``, and they import no stage code. The adapter here has no ML dependencies. This also
pins the ``StageAdapter`` protocol surface and the agreement between the Layout corpus helper
and ``plan.cell_artifact``.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from ragtime.common import Layout
from ragtime.common.layout import _CORPUS_NODE
from ragtime.orchestration import plan, saturate

pytestmark = pytest.mark.small


def _under(path: Path, base: Path) -> bool:
    return base.resolve() in path.resolve().parents or path.resolve() == base.resolve()


def _shards_in(directory: Path) -> list[Path]:
    """Exactly the ``shard_NNNN`` files (excludes ``.result`` / ``._SUCCESS`` sidecars)."""
    return [p for p in directory.glob("shard_*") if "." not in p.name]


# --------------------------------------------------------------------------- #
# Lifecycle: seed -> run_worker -> drive.
# --------------------------------------------------------------------------- #
def test_seed_writes_one_pending_file_per_shard(wq_root, trivial_stage_adapter) -> None:
    n = saturate.seed(None, trivial_stage_adapter, wq_root.dirs)
    assert n == trivial_stage_adapter.n_shards
    pending = [p for p in wq_root.dirs.pending.glob("shard_*") if not p.name.endswith("._SUCCESS")]
    assert len(pending) == trivial_stage_adapter.n_shards


def test_run_worker_brings_up_once_and_drains(wq_root, trivial_stage_adapter) -> None:
    saturate.seed(None, trivial_stage_adapter, wq_root.dirs)
    done = saturate.run_worker(None, trivial_stage_adapter, wq_root.dirs, max_iters=50)
    assert done == trivial_stage_adapter.n_shards
    assert trivial_stage_adapter.bringups == 1  # once despite claiming N shards
    assert trivial_stage_adapter.works == trivial_stage_adapter.n_shards
    assert not list(wq_root.dirs.pending.glob("shard_*"))
    # every completed shard lives in done/ exactly once (claimed exactly once).
    assert len(_shards_in(wq_root.dirs.done)) == trivial_stage_adapter.n_shards
    out = wq_root.dirs.base / "out"
    assert len(_shards_in(out)) == trivial_stage_adapter.n_shards  # per-shard outputs


def test_two_concurrent_workers_claim_each_shard_exactly_once(
    wq_root, trivial_stage_adapter
) -> None:
    """The load-bearing property: under a real thread race no shard is double-processed.

    ``heartbeat``'s temporary name carries a uuid, so two threads in one process cannot
    collide on it and ``done == n`` holds every run. Conservation, ``done + failed == n``, is
    asserted beside it.
    """
    import threading

    trivial_stage_adapter.n_shards = 8
    saturate.seed(None, trivial_stage_adapter, wq_root.dirs)

    barrier = threading.Barrier(2)

    def _worker() -> None:
        barrier.wait()  # maximize overlap
        saturate.run_worker(
            None, trivial_stage_adapter, wq_root.dirs, backoff_s=0.0, max_iters=200
        )

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    done = _shards_in(wq_root.dirs.done)
    failed = _shards_in(wq_root.dirs.failed)
    assert len(done) == 8  # all shards completed (no heartbeat-race flakiness)
    assert len({p.name for p in done}) == 8  # each exactly once (no duplicate in done/)
    assert len(done) + len(failed) == 8  # conservation: no shard lost or duplicated
    outs = _shards_in(wq_root.dirs.base / "out")
    assert len(outs) == 8  # exactly one output per shard (no double-processing)
    assert not list(wq_root.dirs.pending.glob("shard_*"))
    assert not list(wq_root.dirs.running.glob("shard_*"))


def test_drive_merges_once_and_sets_done(wq_root, trivial_stage_adapter) -> None:
    saturate.seed(None, trivial_stage_adapter, wq_root.dirs)
    saturate.run_worker(None, trivial_stage_adapter, wq_root.dirs, max_iters=50)
    saturate.drive(None, trivial_stage_adapter, wq_root.dirs, max_polls=5)
    assert trivial_stage_adapter.merges == 1
    assert (wq_root.dirs.base / "_DONE").exists()  # workers self-terminate off this


def test_resume_is_a_no_op(wq_root, trivial_stage_adapter) -> None:
    saturate.seed(None, trivial_stage_adapter, wq_root.dirs)
    saturate.run_worker(None, trivial_stage_adapter, wq_root.dirs, max_iters=50)
    saturate.drive(None, trivial_stage_adapter, wq_root.dirs, max_polls=5)
    works, merges = trivial_stage_adapter.works, trivial_stage_adapter.merges
    # second full lifecycle over the finalized queue re-invokes work/merge zero times.
    assert saturate.seed(None, trivial_stage_adapter, wq_root.dirs) == 0
    saturate.run_worker(None, trivial_stage_adapter, wq_root.dirs, max_iters=50)
    saturate.drive(None, trivial_stage_adapter, wq_root.dirs, max_polls=5)
    assert trivial_stage_adapter.works == works
    assert trivial_stage_adapter.merges == merges


def test_reaper_reclaims_a_stale_shard_and_completes(
    wq_root, frozen_now, trivial_stage_adapter
) -> None:
    from ragtime.orchestration.slurm import workqueue

    saturate.seed(None, trivial_stage_adapter, wq_root.dirs)
    # simulate a dead worker: claim one shard, then let its heartbeat go stale.
    dead = workqueue.claim(wq_root.dirs.pending, wq_root.dirs.running)
    assert dead is not None
    frozen_now.tick(saturate.MAX_AGE_S + 1.0)
    # run_worker reaps the stale shard back to pending and completes all shards.
    saturate.run_worker(None, trivial_stage_adapter, wq_root.dirs, max_iters=50)
    assert not list(wq_root.dirs.running.glob("shard_*"))  # nothing stranded
    assert len(_shards_in(wq_root.dirs.done)) == trivial_stage_adapter.n_shards


def test_heartbeat_refreshes_during_work_and_reaper_leaves_in_flight_shard_alone(
    wq_root,
) -> None:
    """While ``work`` runs, the forked refresher re-stamps the heartbeat every
    ``heartbeat_s``, so ``reap_stale`` with ``heartbeat_s < max_age < work duration`` never
    requeues the in-flight shard. A genuinely dead worker's child dies with it, so its shard
    still goes stale and is reaped by ``test_reaper_reclaims_a_stale_shard_and_completes``.

    This uses the real clock rather than ``frozen_now``: the adapter's ``work`` sleeps for
    more than twice ``heartbeat_s`` while sampling the heartbeat sidecar and running reaper
    passes. That the stamps survive a call holding the GIL, which is the property a thread
    could not provide and the reason the refresher is a process, is pinned separately in
    ``test_heartbeat_liveness_small.py``.
    """
    import time
    from types import SimpleNamespace

    from ragtime.orchestration.slurm import workqueue

    hb_s, max_age, work_s = 0.05, 0.5, 1.2  # heartbeat_s < max_age < work duration
    dirs = wq_root.dirs

    class _SlowAdapter:
        stage = "slow"
        template = "workqueue_worker_cpu.sbatch"

        def __init__(self) -> None:
            self.hb_stamps: list[float] = []
            self.reclaimed: list[Path] = []

        def bringup(self, cfg):
            return object()

        def shards(self, cfg):
            yield SimpleNamespace(name="shard_0000", payload={"i": 0})

        def work(self, ctx, shard: Path) -> Path:
            hb = workqueue._hb_path(shard)
            deadline = time.monotonic() + work_s
            while time.monotonic() < deadline:
                time.sleep(hb_s)
                self.hb_stamps.append(float(hb.read_text(encoding="utf-8").strip()))
                # a concurrent reaper pass mid-work must see a FRESH heartbeat.
                self.reclaimed += workqueue.reap_stale(dirs.running, dirs.pending, max_age)
            out = saturate.shard_out_path(shard)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"ok\n")
            return out

        def validate(self, path: Path) -> bool:
            return path.exists() and path.stat().st_size > 0

        def merge(self, cfg, shard_paths) -> None:
            pass

    adapter = _SlowAdapter()
    saturate.seed(None, adapter, dirs)
    done = saturate.run_worker(
        None, adapter, dirs, heartbeat_s=hb_s, max_age=max_age, max_iters=5
    )
    assert done == 1
    # the heartbeat advanced during work, so heartbeat_s is genuinely used.
    assert len(set(adapter.hb_stamps)) > 1
    assert max(adapter.hb_stamps) > min(adapter.hb_stamps)
    # no mid-work reaper pass ever reclaimed the in-flight shard.
    assert adapter.reclaimed == []
    assert len(_shards_in(dirs.done)) == 1
    assert not list(dirs.running.glob("shard_*"))


def test_poison_shards_land_in_failed_and_drive_hard_stops(
    wq_root, poison_stage_adapter, caplog
) -> None:
    with caplog.at_level("WARNING", logger="ragtime.orchestration.saturate"):
        saturate.seed(None, poison_stage_adapter, wq_root.dirs)
        saturate.run_worker(None, poison_stage_adapter, wq_root.dirs, k_max=2, max_iters=80)
    assert list(wq_root.dirs.failed.glob("shard_*"))  # poison -> failed/ (not an infinite loop)
    # The reason is in the log. A detail-free `saturate.work.failed` line makes a large
    # parallel failure undiagnosable, so the event carries the exception type, its repr and
    # the traceback.
    events = [json.loads(r.message) for r in caplog.records if "work.failed" in r.message]
    assert events
    for ev in events:
        assert ev["error_type"] == "RuntimeError"
        assert "poison shard" in ev["error"]
        assert "Traceback" in ev["traceback"] and "work" in ev["traceback"]
    with pytest.raises(RuntimeError, match="failed"):
        saturate.drive(None, poison_stage_adapter, wq_root.dirs, max_polls=5)


def test_every_path_stays_under_wq_base(wq_root, trivial_stage_adapter) -> None:
    saturate.seed(None, trivial_stage_adapter, wq_root.dirs)
    saturate.run_worker(None, trivial_stage_adapter, wq_root.dirs, max_iters=50)
    saturate.drive(None, trivial_stage_adapter, wq_root.dirs, max_polls=5)
    for sub in ("pending", "running", "done", "failed", "out"):
        d = wq_root.dirs.base / sub
        for p in d.glob("*") if d.exists() else []:
            assert _under(p, wq_root.dirs.base)


# --------------------------------------------------------------------------- #
# Spine: saturate imports no stage code.
# --------------------------------------------------------------------------- #
def test_saturate_imports_no_stage_code() -> None:
    tree = ast.parse(inspect.getsource(saturate))
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        for m in mods:
            for stage in ("preprocess", "retrieval", "pipeline", "monitoring"):
                assert f"ragtime.{stage}" not in m, f"saturate must not import {m}"


# --------------------------------------------------------------------------- #
# Cross-stage pins: StageAdapter surface + Layout==cell_artifact anchor.
# --------------------------------------------------------------------------- #
def test_stage_adapter_protocol_names_and_arity_stable() -> None:
    expected = {
        "bringup": ("cfg",),
        "shards": ("cfg",),
        "work": ("ctx", "shard"),
        "validate": ("path",),
        "merge": ("cfg", "shard_paths"),
    }
    for name, params in expected.items():
        fn = getattr(saturate.StageAdapter, name)
        sig = inspect.signature(fn)
        got = [p for p in sig.parameters if p != "self"]
        assert got == list(params), f"{name}{tuple(got)} != {name}{params}"


def test_layout_corpus_dir_equals_plan_cell_artifact(real_e2e_paths) -> None:
    from ragtime.config import all_hashes, load

    root = "/beegfs/root"
    cfg = load(real_e2e_paths[0])
    ch = all_hashes(cfg)["chunker"]
    dag = plan.build_plan(cfg)
    corpus_node = dag.node(plan.CORPUS)
    expected = plan.cell_artifact(root, dag, corpus_node, 0)
    layout = Layout(run_dir=root, base=root, family=dag.family, chunker_hash=ch)
    # The anchor, the hashed corpus dir, is byte-identical to what plan.cell_artifact derives,
    # so the planner and the driver cannot diverge.
    assert layout.corpus_dir(dag.family, ch) == expected
    assert ch[:12] in expected.parts  # the semantic chunker hash is a path level
    # the queue, raw and passages helpers all hang under that same anchor.
    assert _under(layout.wq_dir(dag.family, ch, "chunk"), expected)
    assert _under(layout.corpus_raw_dir(dag.family, ch), expected)
    assert _under(layout.passages_path("native"), expected)
    assert _CORPUS_NODE == plan.CORPUS  # the constant can't drift from the planner


class _FakeCfg:
    """Minimal cfg exposing ``.blocks`` for ``all_hashes`` (avoids re-loading a file)."""

    def __init__(self, blocks: dict) -> None:
        self.blocks = blocks


def _blocks(token_budget: int, corpus_shards: int) -> dict:
    return {
        "chunker": {
            "config": {
                "token_budget": token_budget,
                "overlap_frac": 0.15,
                "segmenter_model": "sat-3l-sm",
                "tokenizer_id": "BAAI/bge-m3@abc",
            }
        },
        "execution": {"corpus_shards": corpus_shards, "oversubscription": 5},
    }


def test_execution_tweak_does_not_change_the_chunker_hash_but_a_semantic_edit_does() -> None:
    """Execution knobs are not corpus identity; the chunker's own fields are."""
    from ragtime.config import all_hashes

    base = all_hashes(_FakeCfg(_blocks(512, 100)))["chunker"]
    assert all_hashes(_FakeCfg(_blocks(512, 999)))["chunker"] == base  # execution tweak: same
    assert all_hashes(_FakeCfg(_blocks(256, 100)))["chunker"] != base  # semantic edit: different


def test_corpus_already_done_is_keyed_off_the_chunker_hash(tmp_path) -> None:
    """The artifact tree is the checkpoint. Re-running the same config finds the marker and
    does nothing; a changed semantic chunker field resolves to a fresh corpus path, so
    ``already_done`` is False and the corpus is rebuilt rather than reused stale."""
    from ragtime.common.io import success_marker
    from ragtime.orchestration.plan import JobDAG, JobNode, already_done, cell_artifact

    def _dag(ch: str) -> JobDAG:
        node = JobNode(plan.CORPUS, "preprocess", key=f"e2e:{ch}", after=(), family_shared=True)
        return JobDAG(run_id="e2e-original", family="e2e", variant="original", nodes=(node,))

    dag_a, dag_b = _dag("a" * 64), _dag("b" * 64)  # two distinct chunker hashes
    node_a, node_b = dag_a.node(plan.CORPUS), dag_b.node(plan.CORPUS)
    pa = cell_artifact(tmp_path, dag_a, node_a, 0)
    pb = cell_artifact(tmp_path, dag_b, node_b, 0)
    assert pa != pb  # different chunker hash -> different corpus dir

    pa.mkdir(parents=True, exist_ok=True)
    success_marker(pa).write_bytes(b"")
    assert already_done(node_a, tmp_path, dag_a)  # same config re-run -> no-op
    assert not already_done(node_b, tmp_path, dag_b)  # changed chunker -> rebuild
