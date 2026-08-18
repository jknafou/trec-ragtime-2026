"""The generic saturation driver, shared by the corpus build and the online half.

A stage-parameterized pull work-queue driver: stand up an in-process model replica per
worker, self-claim shards off one shared queue, saturate. The stage supplies a small
:class:`StageAdapter` (``bringup`` / ``shards`` / ``work`` / ``validate`` / ``merge``),
so chunking, translation and the index build are adapters over this same driver -- and so
is the online pipeline stage, whose shard is one ``(topic, seed)`` cell
(``pipeline.topic_fleet.TopicCellAdapter``). That reuse is why the fleet needs no second
fan: ``run --stage pipeline`` reaches :func:`run_worker` here, and that is how every topic
behind the submission was claimed. Built on ``slurm.workqueue`` (atomic claim, heartbeat,
reaper, poison shards, mark_done), ``common.Layout`` for paths, and ``config_hash``-keyed
``_SUCCESS`` idempotence.

Three entrypoints:

- ``seed(cfg, adapter, wq)``: CPU, once. Write one ``pending/`` file per work unit.
- ``run_worker(cfg, adapter, wq)``: the claim loop. ``bringup`` once, then
  claim -> work -> validate -> mark_done, with a heartbeat and poison handling.
- ``drive(cfg, adapter, wq)``: CPU, once. Drain, hard-stop on ``failed/``, merge, then
  write a DONE sentinel so workers self-terminate.

This module owns the queue lifecycle only and imports no stage code: stages depend on
orchestration, never the reverse.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ragtime.common import Layout, get_logger

from .run_identity import run_family
from .slurm import workqueue
from .slurm.workqueue import QueueDirs

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "HEARTBEAT_S",
    "MAX_AGE_S",
    "StageAdapter",
    "drive",
    "out_dir",
    "queue_for",
    "run_worker",
    "seed",
    "shard_out_path",
    "worker_provenance",
]

_log = get_logger("orchestration.saturate")

# Defaults for the worker loop, all overridable per call so nothing becomes a hardcoded
# run choice. The two liveness constants are public because `cli` resolves a
# per-substage max_age against them from config.
HEARTBEAT_S = 30.0
#: The reaper window when a substage names none: wide, since a shard is
#: then reclaimed only after many consecutive missed stamps, which a live worker cannot
#: produce now that the refresher is a forked child. It stays a window rather than an
#: eternity, because the reaper's job is to return a genuinely dead worker's shard to
#: the queue. A substage whose per-unit work is long sets its own
#: ``execution.<substage>_max_age_s`` (see ``cli._substage_max_age``).
MAX_AGE_S = 99999.0
_K_MAX = 3
# Bounded backoff so a worker or the sequencer never hot-spins the shared filesystem
# once concurrency exists; tests pass 0.0 to keep the small suite fast.
_CLAIM_BACKOFF_S = 0.5
_DRIVE_POLL_S = 1.0


def _sleep(seconds: float) -> None:
    """Indirected sleep; skips the syscall for a non-positive interval."""
    if seconds > 0:
        time.sleep(seconds)


@runtime_checkable
class StageAdapter(Protocol):
    """The stage-supplied contract the generic driver is parameterized by.

    A stage supplies ``bringup`` (load the resident model once per worker), ``shards``
    (the seed unit), ``work`` (claim, process and write one shard), ``validate`` (guard
    against a zero-output shard) and ``merge`` (fold shard outputs into the family
    artifact). ``stage`` names the substage, which selects the queue subdirectory, and
    ``template`` selects the sbatch template.
    """

    stage: str
    template: str

    def bringup(self, cfg: object) -> object:
        """Stand up the resident in-process model or tokenizer once per worker process."""
        ...

    def shards(self, cfg: object) -> Iterable[ShardSpec]:
        """Yield the claimable work units; the seed granularity."""
        ...

    def work(self, ctx: object, shard: Path) -> Path:
        """Process one claimed shard and return the output artifact path."""
        ...

    def validate(self, path: Path) -> bool:
        """Predicate gating ``mark_done``; falsey leaves the shard reclaimable."""
        ...

    def merge(self, cfg: object, shard_paths: Sequence[Path]) -> None:
        """Fold the per-shard outputs into the family artifact."""
        ...


class ShardSpec(Protocol):
    """One claimable unit of work.

    ``name`` is the queue filename; ``payload`` is a JSON-serializable description the
    adapter's ``work`` reads back, opaque to the driver.
    """

    name: str
    payload: dict


# --------------------------------------------------------------------------- #
# Path conventions: one place for the per-shard output directory, so adapters and
# `drive` agree without the driver importing stage code.
# --------------------------------------------------------------------------- #
def out_dir(wq: QueueDirs) -> Path:
    """``<wq.base>/out``, where per-shard outputs land for ``merge`` to collect."""
    return wq.base / "out"


def shard_out_path(shard: Path) -> Path:
    """The output path for a claimed shard: ``<base>/out/<name>``.

    A shard lives at ``<base>/{running,done}/<name>``, so the base is
    ``shard.parent.parent``.
    """
    return shard.parent.parent / "out" / shard.name


def _done_sentinel(wq: QueueDirs) -> Path:
    """``<wq.base>/_DONE``, set by ``drive`` when the stage is finalized."""
    return wq.base / "_DONE"


def queue_for(cfg: object, adapter: StageAdapter, *, base: str | os.PathLike[str]) -> QueueDirs:
    """Resolve and create this stage's work queue under ``Layout.wq_dir``.

    ``base`` is the artifact root. The queue lives under the family-shared corpus
    directory keyed off the semantic chunker hash, per substage, so a chunker edit gets
    a fresh queue too.
    """
    from ragtime.config import all_hashes  # function-local: orchestration -> config

    fam = run_family(cfg)  # type: ignore[arg-type]
    ch = all_hashes(cfg)["chunker"]  # type: ignore[arg-type]
    layout = Layout(run_dir=base, base=base)
    return workqueue.init_queue(layout.wq_dir(fam, ch, adapter.stage))


def _shard_files(directory: Path) -> list[Path]:
    """The real shard and result files in a queue subdir, excluding ``_SUCCESS`` sidecars."""
    return [
        p
        for p in sorted(directory.glob("*"))
        if p.is_file() and not p.name.endswith("._SUCCESS")
    ]


def _has_work(wq: QueueDirs) -> bool:
    return bool(_shard_files(wq.pending) or _shard_files(wq.running))


# --------------------------------------------------------------------------- #
# seed: shard the corpus into pending/.
# --------------------------------------------------------------------------- #
def seed(cfg: object, adapter: StageAdapter, wq: QueueDirs) -> int:
    """Write one ``pending/`` file per ``adapter.shards(cfg)``; return the shard count.

    Idempotent: over an already-finalized queue (the DONE sentinel is present) it
    writes nothing. Each shard file carries the ShardSpec payload as one JSON line,
    written temp-then-rename so ``claim``'s glob never races a partial file, and
    without a ``_SUCCESS`` sidecar, which would pollute that glob.
    """
    if _done_sentinel(wq).exists():
        return 0
    n = 0
    for spec in adapter.shards(cfg):
        dst = wq.pending / spec.name
        if dst.exists():
            n += 1
            continue
        data = (json.dumps(spec.payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        n += 1
    _log.info("saturate.seed", stage=adapter.stage, shards=n)
    return n


# --------------------------------------------------------------------------- #
# run_worker: the claim loop. bringup once.
# --------------------------------------------------------------------------- #
#: Environment names carrying the observed hardware a worker landed on, mapped to the
#: key each becomes in the shard's ``done/<shard>.result`` row. These are facts read off
#: the running job, never run choices, and they exist because a heterogeneous
#: OR-constraint array is not byte-homogeneous: the same shard on two GPU
#: architectures can produce different output, so the artifact must name its silicon.
_PROVENANCE_ENV = {
    "RAGTIME_GPU_MODEL": "gpu",
    "SLURMD_NODENAME": "node",
    "SLURM_JOB_ID": "job_id",
    "SLURM_ARRAY_TASK_ID": "array_task",
}


def worker_provenance(env: dict[str, str] | None = None) -> dict[str, str]:
    """The observed hardware and job facts for this worker; empty off SLURM."""
    src = os.environ if env is None else env
    return {key: v for name, key in _PROVENANCE_ENV.items() if (v := src.get(name, "").strip())}


def _refresh_heartbeat_until(shard: Path, stop: threading.Event, interval: float) -> None:
    """Degraded fallback: re-stamp ``shard``'s heartbeat from a daemon thread.

    Reachable only when ``os.fork`` itself fails, because a thread cannot be relied on
    to stamp: ``stop.wait()`` needs the GIL to return and a native extension that holds
    it for the whole call starves the thread for minutes. It is kept because it is
    still better than no heartbeat for stages whose native work releases the GIL, and
    its use is always logged as a degradation. Heartbeat IO errors are tolerated, the
    same policy as ``claim``'s post-claim stamp.
    """
    while not stop.wait(interval):
        try:
            workqueue.heartbeat(shard)
        except OSError as exc:
            _log.warning(
                "saturate.heartbeat.refresh_failed", shard=shard.name, error=str(exc)
            )


@dataclass(slots=True)
class _ThreadHeartbeat:
    """The fallback refresher, wearing :class:`workqueue.HeartbeatProcess`'s ``stop()``."""

    stop_event: threading.Event
    thread: threading.Thread

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()


def _start_heartbeat(
    shard: Path, interval: float
) -> workqueue.HeartbeatProcess | _ThreadHeartbeat | None:
    """The forked-child refresher for ``shard``, or the degraded thread if ``fork`` fails.

    One call site decides the mechanism, so ``run_worker``'s ``finally`` has exactly
    one ``stop()`` to call. ``interval <= 0`` disables refreshing, which the
    clock-freezing tests and the reaper test rely on.
    """
    if interval <= 0:
        return None
    try:
        return workqueue.start_heartbeat(shard, interval)
    except OSError as exc:
        _log.error(
            "saturate.heartbeat.fork_failed",
            shard=shard.name,
            error=str(exc),
            fallback="thread (GIL-dependent, see _refresh_heartbeat_until)",
        )
        stop = threading.Event()
        thread = threading.Thread(
            target=_refresh_heartbeat_until,
            args=(shard, stop, interval),
            name=f"hb-{shard.name}",
            daemon=True,
        )
        thread.start()
        return _ThreadHeartbeat(stop_event=stop, thread=thread)


def run_worker(
    cfg: object,
    adapter: StageAdapter,
    wq: QueueDirs,
    *,
    key: str = "corpus",
    heartbeat_s: float = HEARTBEAT_S,
    max_age: float = MAX_AGE_S,
    k_max: int = _K_MAX,
    backoff_s: float = _CLAIM_BACKOFF_S,
    max_iters: int | None = None,
) -> int:
    """Bring the model up once, then drain the queue: claim, work, validate, done.

    Returns the number of shards this worker completed. The model is stood up outside
    the loop. Each iteration reaps a heartbeat-stale ``running/`` shard back to
    ``pending/`` before claiming, and while ``work`` runs a forked child refreshes the
    claimed shard's heartbeat every ``heartbeat_s`` seconds, so a long-running shard is
    never requeued mid-flight by another worker's reaper. A shard whose ``work`` raises
    or whose output fails ``validate`` is failed: requeued while retriable, else moved
    to ``failed/`` after ``k_max`` attempts. On a lost claim race the worker sleeps
    ``backoff_s``. ``max_iters`` bounds the loop for tests; a live worker runs until the
    queue drains or ``drive`` sets the DONE sentinel.

    On walltime preemption, the sbatch templates' USR1 trap does not requeue the
    in-flight shard, because a bash trap cannot reach the Python worker's claimed
    shard. Recovery is via the heartbeat reaper: the killed worker stops refreshing, so
    the next launch's ``reap_stale`` returns the shard to ``pending/`` after
    ``max_age``.
    """
    sentinel = _done_sentinel(wq)
    if sentinel.exists() and not _has_work(wq):
        return 0  # already finalized, do not even load the model
    ctx = adapter.bringup(cfg)  # once, outside the loop
    provenance = worker_provenance()
    done = 0
    it = 0
    while True:
        if max_iters is not None and it >= max_iters:
            break
        it += 1
        if sentinel.exists():
            break
        workqueue.reap_stale(wq.running, wq.pending, max_age)
        shard = workqueue.claim(wq.pending, wq.running)
        if shard is None:
            if not _has_work(wq):
                break  # queue drained
            _sleep(backoff_s)  # lost every race this pass, back off
            continue
        try:
            workqueue.heartbeat(shard)
            # The fork happens here, between the claim and `work`, when the parent is
            # at its quietest rather than inside a native call. A crashed worker's child
            # dies with it via pipe EOF, so its shard still goes stale and is reclaimed.
            hb = _start_heartbeat(shard, heartbeat_s)
            try:
                out = adapter.work(ctx, shard)
            finally:
                if hb is not None:
                    hb.stop()
            if adapter.validate(out):
                workqueue.mark_done(shard, wq.done, key, provenance=provenance)
                done += 1
            else:
                workqueue.fail(shard, wq.pending, wq.failed, k_max)
        except Exception as exc:  # noqa: BLE001 - a poison shard must not kill the worker
            # The exception detail is part of the event: type, repr and the formatted
            # traceback go into the JSON line so a poisoned shard is explicable from
            # `failed/` plus the worker log alone, without a re-run.
            _log.warning(
                "saturate.work.failed",
                stage=adapter.stage,
                shard=shard.name,
                error_type=type(exc).__name__,
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
            workqueue.fail(shard, wq.pending, wq.failed, k_max)
    return done


# --------------------------------------------------------------------------- #
# drive: drain, merge, DONE.
# --------------------------------------------------------------------------- #
def drive(
    cfg: object,
    adapter: StageAdapter,
    wq: QueueDirs,
    *,
    poll_s: float = _DRIVE_POLL_S,
    max_polls: int | None = None,
) -> None:
    """Wait for the queue to drain, hard-stop on ``failed/``, then merge once and mark DONE.

    Idempotent: over an already-finalized queue it returns immediately, so ``merge`` is
    never re-invoked. A non-empty ``failed/`` raises rather than publishing a partial
    family artifact. On drain it collects the per-shard outputs from ``out/`` and hands
    them to ``adapter.merge``, then writes the DONE sentinel so workers self-terminate.
    """
    sentinel = _done_sentinel(wq)
    if sentinel.exists():
        return  # already finalized, no re-merge
    polls = 0
    while _has_work(wq):
        if _shard_files(wq.failed):
            break
        polls += 1
        if max_polls is not None and polls >= max_polls:
            break
        _sleep(poll_s)
    if _shard_files(wq.failed):
        raise RuntimeError(
            f"saturate.drive: {len(_shard_files(wq.failed))} shard(s) in failed/ "
            f"(stage={adapter.stage}); refusing to merge a partial corpus"
        )
    if _has_work(wq):
        raise RuntimeError(
            f"saturate.drive: queue not drained (stage={adapter.stage}) and no failed/ "
            f"shards; a worker is still running or max_polls was reached"
        )
    shard_paths = _shard_files(out_dir(wq))
    adapter.merge(cfg, shard_paths)
    tmp = sentinel.with_name(f".{sentinel.name}.tmp-{os.getpid()}")
    tmp.write_bytes(b"")
    os.replace(tmp, sentinel)
    _log.info("saturate.drive.done", stage=adapter.stage, shards=len(shard_paths))
