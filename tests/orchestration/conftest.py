"""Shared fixtures for the orchestration tests.

No test imports torch/vllm/transformers; ``sbatch``/``scancel``/``squeue`` are never
actually invoked: the ``sbatch_spy`` records the call and returns a fake job id, so the gate is
cluster-independent. The tracked ``config/*.yml`` are opened read-only, and mutated
fixtures are in-memory text written to a ``tmp_path`` copy (the ``tests/config``
precedent), never edits to a tracked file.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from ragtime.common import write_jsonl
from ragtime.orchestration import plan
from ragtime.orchestration.plan import JobDAG, JobNode, node_artifact
from ragtime.orchestration.slurm import launcher, workqueue

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG_DIR = _REPO_ROOT / "config"


# --------------------------------------------------------------------------- #
# Real config paths (mirrors tests/config/conftest.py; read-only test data).
# --------------------------------------------------------------------------- #
def _cfg(name: str) -> Path:
    return _CFG_DIR / name


@pytest.fixture
def real_e2e_paths() -> list[Path]:
    return [
        _cfg("e2e-omt.yml"),
        _cfg("e2e-omt-weak.yml"),
        _cfg("e2e-original.yml"),
    ]


@pytest.fixture
def real_mlir_paths() -> list[Path]:
    return [
        _cfg("mlir-omt.yml"),
        _cfg("mlir-omt-weak.yml"),
        _cfg("mlir-original.yml"),
    ]


@pytest.fixture
def real_config_paths(real_e2e_paths: list[Path], real_mlir_paths: list[Path]) -> list[Path]:
    return real_e2e_paths + real_mlir_paths


@pytest.fixture
def uv_lock_path() -> Path:
    return _REPO_ROOT / "uv.lock"


# --------------------------------------------------------------------------- #
# sbatch spy: records every subprocess call the launcher would make, and never shells
# out. Tests that must not submit assert `.calls == []`.
# --------------------------------------------------------------------------- #
@dataclass
class _SbatchSpy:
    calls: list[list[str]]
    ids: list[int] | None = None  # the distinct jobid returned per call (call-order aligned)
    _next_id: int = 12345

    def run(self, argv, *args, **kwargs):
        if self.ids is None:
            self.ids = []
        self.calls.append(list(argv))
        # Emulate `sbatch --parsable` -> "<jobid>;<cluster>" with distinct increasing ids
        # (first call is 12345, so single-submit tests still see 12345) so afterok wiring
        # between specific jobs is assertable, not masked by a constant id.
        jobid = self._next_id
        self._next_id += 1
        self.ids.append(jobid)
        return subprocess.CompletedProcess(argv, 0, stdout=f"{jobid};cluster", stderr="")


@pytest.fixture
def sbatch_spy(monkeypatch: pytest.MonkeyPatch) -> _SbatchSpy:
    spy = _SbatchSpy(calls=[], ids=[])
    monkeypatch.setattr(launcher.subprocess, "run", spy.run)
    return spy


# --------------------------------------------------------------------------- #
# Work-queue fixtures: a stand-in queue directory and a frozen clock.
# --------------------------------------------------------------------------- #
@dataclass
class _WqRoot:
    dirs: workqueue.QueueDirs

    def drop_shards(self, n: int) -> list[Path]:
        """Create ``n`` fake shard files in ``pending/``; return their paths."""
        out = []
        for i in range(n):
            p = self.dirs.pending / f"shard_{i:04d}"
            p.write_text("payload\n", encoding="utf-8")
            out.append(p)
        return out


@pytest.fixture
def wq_root(tmp_path: Path) -> _WqRoot:
    return _WqRoot(dirs=workqueue.init_queue(tmp_path / "wq"))


@dataclass
class _Clock:
    t: float

    def tick(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock(t=1_000_000.0)
    monkeypatch.setattr(workqueue, "_now", lambda: clock.t)
    return clock


# --------------------------------------------------------------------------- #
# Stub node handle carrying the one consumed attribute (`.loop_ceiling: int`).
# --------------------------------------------------------------------------- #
@dataclass
class _StubNode:
    loop_ceiling: int


@pytest.fixture
def stub_node() -> Callable[[int], _StubNode]:
    def _make(loop_ceiling: int) -> _StubNode:
        return _StubNode(loop_ceiling=loop_ceiling)

    return _make


# --------------------------------------------------------------------------- #
# Two-node stub DAG: two `run --stage` stand-ins with call counters, writing
# artifact + _SUCCESS under a tmp_path-rooted, Layout-shaped run dir.
# --------------------------------------------------------------------------- #
@dataclass
class _StubDag:
    dag: JobDAG
    root: Path
    counters: dict[str, int]
    runner: Callable[[JobNode], None]


@pytest.fixture
def two_stage_stub_dag(tmp_path: Path) -> _StubDag:
    root = tmp_path / "runs"
    dag = JobDAG(
        run_id="e2e-stub",
        family="e2e",
        variant="original",
        nodes=(
            JobNode("stage_a", "pipeline", key="k-a", after=()),
            JobNode("stage_b", "pipeline", key="k-b", after=("stage_a",)),
        ),
    )
    counters = {"stage_a": 0, "stage_b": 0}

    def runner(node: JobNode) -> None:
        counters[node.name] += 1
        art = node_artifact(root, dag, node)
        write_jsonl(art, [{"node": node.name, "key": node.key}])

    return _StubDag(dag=dag, root=root, counters=counters, runner=runner)


# expose plan for tests that patch build_plan
@pytest.fixture
def plan_module():
    return plan


# --------------------------------------------------------------------------- #
# trivial_stage_adapter: a saturate.StageAdapter with no ML dependencies, instrumented with
# call counters, to exercise the full driver lifecycle (seed/run_worker/drive).
# --------------------------------------------------------------------------- #
@dataclass
class _TrivialShard:
    name: str
    payload: dict


@dataclass
class _TrivialAdapter:
    """bringup = sentinel (counted once); shards = 3 tiny specs; work = copy input->out
    (counted); validate = non-empty; merge = concat (counted). ``poison`` makes work raise."""

    stage: str = "trivial"
    template: str = "workqueue_worker_cpu.sbatch"
    poison: bool = False
    bringups: int = 0
    works: int = 0
    merges: int = 0
    n_shards: int = 3

    def bringup(self, cfg):
        self.bringups += 1
        return object()  # a sentinel "service"

    def shards(self, cfg):
        for i in range(self.n_shards):
            yield _TrivialShard(name=f"shard_{i:04d}", payload={"i": i})

    def work(self, ctx, shard: Path) -> Path:
        self.works += 1
        if self.poison:
            raise RuntimeError("poison shard")
        from ragtime.orchestration import saturate

        out = saturate.shard_out_path(shard)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(shard.read_bytes())  # deterministic copy input -> output
        return out

    def validate(self, path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    def merge(self, cfg, shard_paths) -> None:
        self.merges += 1
        from ragtime.orchestration.saturate import out_dir  # via drive-supplied paths

        _ = out_dir  # documentation: paths come from drive's out/ scan
        merged = shard_paths[0].parent.parent / "merged.jsonl" if shard_paths else None
        if merged is not None:
            merged.write_bytes(b"".join(sorted(p.read_bytes() for p in shard_paths)))


@pytest.fixture
def trivial_stage_adapter() -> _TrivialAdapter:
    return _TrivialAdapter()


@pytest.fixture
def poison_stage_adapter() -> _TrivialAdapter:
    return _TrivialAdapter(poison=True)
