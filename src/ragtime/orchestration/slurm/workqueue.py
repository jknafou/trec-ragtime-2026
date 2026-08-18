"""A dynamic, self-claiming, file-based work queue.

The queue has ``pending/ running/ done/ failed/ meta/`` subdirectories under a
caller-supplied base, which must live on the shared filesystem rather than node-local
``/tmp``: an empty per-node ``/tmp`` would let every worker win its atomic rename
independently. This module hardcodes no path.

The primitives:

- Atomic claim: first-rename-wins ``os.rename(pending -> running)``; a loser's rename
  raises and it moves on.
- Heartbeat and staleness reaper: a worker stamps a sidecar timestamp from a forked
  child, and the reaper returns a ``running/`` shard whose heartbeat is older than
  ``max_age`` to ``pending/``.
- Validate before done: a supplied predicate must pass before ``running -> done`` and
  before the ``_SUCCESS(key)`` marker is written.
- Poison shards: a persisted per-shard attempt counter sends a shard that has failed
  ``k_max`` times to ``failed/``, which is the orchestrator's hard stop.
- USR1 requeue: the walltime-preemption helper that returns an in-flight shard to
  ``pending/``.
"""

from __future__ import annotations

import json
import os
import select
import signal
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragtime.common import get_logger, write_jsonl

_log = get_logger("orchestration.workqueue")

__all__ = [
    "HeartbeatProcess",
    "QueueDirs",
    "attempts",
    "claim",
    "fail",
    "heartbeat",
    "init_queue",
    "mark_done",
    "reap_stale",
    "requeue",
    "start_heartbeat",
]


def _now() -> float:
    """Wall-clock seconds, indirected so tests can freeze the clock."""
    return time.time()


def _tmp_name(target: Path) -> Path:
    """Return a collision-proof temp sibling of ``target`` for temp-then-rename writes.

    The suffix carries a fresh uuid so no two concurrent writers, threads in one
    process or separate processes, ever compute the same temp path.
    """
    return target.with_name(f"{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")


def _move_tolerant(src: Path, dst: Path) -> Path:
    """Rename ``src`` to ``dst``, tolerating ``src`` already being gone; return ``dst``.

    A transient filesystem error or a concurrent reaper may have already moved the
    shard, and the failure and requeue paths must not crash the worker loop over that.
    """
    try:
        os.rename(src, dst)
    except FileNotFoundError:
        _log.warning("workqueue.move.already_gone", src=Path(src).name, dst=str(dst.parent.name))
    return dst


@dataclass(frozen=True, slots=True)
class QueueDirs:
    """The five queue subdirectories under one caller-supplied base."""

    base: Path
    pending: Path
    running: Path
    done: Path
    failed: Path
    meta: Path


def init_queue(base: str | os.PathLike[str]) -> QueueDirs:
    """Create ``pending/ running/ done/ failed/ meta/`` under ``base`` and return them.

    ``base`` is always a parameter: the caller resolves it on the shared filesystem so
    the atomic claim is globally consistent across nodes.
    """
    b = Path(base)
    dirs = QueueDirs(
        base=b,
        pending=b / "pending",
        running=b / "running",
        done=b / "done",
        failed=b / "failed",
        meta=b / "meta",
    )
    for d in (dirs.pending, dirs.running, dirs.done, dirs.failed, dirs.meta):
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _meta_dir(shard_or_dir: Path) -> Path:
    """The ``meta/`` directory for a shard living in one of the queue subdirs."""
    return shard_or_dir.parent.parent / "meta"


def _hb_path(shard: Path) -> Path:
    return _meta_dir(shard) / f"{shard.name}.hb"


def _attempts_path(shard: Path) -> Path:
    return _meta_dir(shard) / f"{shard.name}.attempts.json"


# --------------------------------------------------------------------------- #
# Atomic claim and USR1 requeue.
# --------------------------------------------------------------------------- #
def claim(pending: str | os.PathLike[str], running: str | os.PathLike[str]) -> Path | None:
    """Claim one pending shard by atomic rename; return its ``running/`` path or ``None``.

    Iterates ``pending/`` and tries ``os.rename`` into ``running/``; the first rename
    to succeed owns the shard, and a claimer that lost the race skips to the next
    candidate. Returns ``None`` when nothing is claimable. Also stamps an initial
    heartbeat so the reaper sees the fresh claim immediately.
    """
    pdir, rdir = Path(pending), Path(running)
    rdir.mkdir(parents=True, exist_ok=True)
    for src in sorted(pdir.glob("*")):
        if not src.is_file():
            continue
        dst = rdir / src.name
        try:
            os.rename(src, dst)
        except OSError:
            continue  # lost the race, another worker claimed it first
        # The shard is ours. A failed heartbeat here must not propagate and crash the
        # caller: the reaper reclaims a shard that never gets one.
        try:
            heartbeat(dst)
        except OSError as exc:
            _log.warning("workqueue.claim.heartbeat_failed", shard=dst.name, error=str(exc))
        return dst
    return None


def requeue(shard: str | os.PathLike[str], pending: str | os.PathLike[str]) -> Path:
    """Return an in-flight shard to ``pending``, the USR1-trap requeue path.

    A worker caught by ``--signal=B:USR1@300`` calls this so the shard is reclaimable
    by a resumed worker rather than left claimed.
    """
    s, pdir = Path(shard), Path(pending)
    pdir.mkdir(parents=True, exist_ok=True)
    # Idempotent: a concurrent reaper may already have requeued this shard.
    return _move_tolerant(s, pdir / s.name)


# --------------------------------------------------------------------------- #
# Heartbeat and staleness reaper.
# --------------------------------------------------------------------------- #
def heartbeat(shard: str | os.PathLike[str]) -> Path:
    """Stamp ``shard``'s liveness sidecar with the current time; return its path.

    A live worker calls this periodically, and the reaper reads the recorded time to
    tell a still-working shard from one stranded by a killed worker.
    """
    s = Path(shard)
    hb = _hb_path(s)
    hb.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(hb)
    tmp.write_text(f"{_now()}\n", encoding="utf-8")
    os.replace(tmp, hb)
    return hb


#: How long :meth:`HeartbeatProcess.stop` waits for the child to exit on its own before
#: sending SIGKILL. The child wakes on pipe EOF, so this is milliseconds in practice.
_HB_STOP_GRACE_S = 10.0
_HB_STOP_POLL_S = 0.01


@dataclass(slots=True)
class HeartbeatProcess:
    """A forked child re-stamping one claimed shard's heartbeat until stopped or orphaned.

    See :func:`start_heartbeat` for why this is a process rather than a thread.
    """

    pid: int
    shard: Path
    #: The parent's end of the stop pipe. Closing it is what the child sees as EOF, and
    #: the kernel closes it if the parent dies without calling :meth:`stop`.
    stop_fd: int
    stopped: bool = field(default=False)

    def stop(self) -> None:
        """Close the stop pipe and reap the child, killing it if it overstays. Idempotent."""
        if self.stopped:
            return
        self.stopped = True
        try:
            os.close(self.stop_fd)  # EOF in the child, which returns from select at once
        except OSError:
            pass
        deadline = time.monotonic() + _HB_STOP_GRACE_S
        while True:
            try:
                reaped, _status = os.waitpid(self.pid, os.WNOHANG)
            except OSError:  # includes ChildProcessError (already reaped)
                return
            if reaped == self.pid:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(_HB_STOP_POLL_S)
        _log.warning("workqueue.heartbeat.child_kill", shard=self.shard.name, pid=self.pid)
        try:
            os.kill(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
        except OSError:
            pass


def _heartbeat_child(shard: Path, stop_fd: int, interval: float) -> None:
    """The child's whole life: select with timeout, stamp, repeat.

    Minimal, because this runs in a process forked from a parent that may
    hold a CUDA context and native thread pools: it touches no model, no logger, no
    stdio and no third-party library, and its caller leaves via ``os._exit``.
    """
    parent = os.getppid()
    while True:
        try:
            ready, _, _ = select.select([stop_fd], [], [], interval)
        except OSError:
            return  # the pipe went away, treat as stop
        if ready:
            return  # write end closed: the parent stopped us, exited or was killed
        if os.getppid() != parent:
            # Reparented, so the parent is gone even though something still holds a dup
            # of the write end. Bounded by one tick.
            return
        try:
            heartbeat(shard)
        except OSError:
            # Liveness is best-effort: the reaper's window is many intervals wide so a
            # few dropped stamps are survivable.
            continue


def start_heartbeat(
    shard: str | os.PathLike[str], interval: float
) -> HeartbeatProcess | None:
    """Fork a child that re-stamps ``shard``'s heartbeat every ``interval`` seconds.

    Returns ``None`` when ``interval <= 0``, which disables the refresher entirely; the
    caller then has only :func:`claim`'s single post-claim stamp, which is what the
    clock-freezing tests want. An ``OSError`` from ``fork`` (a process or thread
    rlimit) propagates, because running with no heartbeat at all is a decision for the
    caller to take.

    This is a process rather than a thread because ``threading.Event.wait`` needs the
    GIL to return, and a native extension that holds the GIL for its whole call (the
    Seismic index build, PLAID) starves the stamping thread for minutes at a time.
    Heartbeats then land only at phase boundaries, every other worker's reaper reclaims
    the in-flight shard, and because :func:`reap_stale` requeues without incrementing
    ``attempts`` nothing ever poisons, so the queue churns instead of failing. A forked
    child has its own interpreter and its own GIL. The fork happens at the quietest
    moment in the worker loop, right after :func:`claim` and before the stage's work
    begins, never during a native call; the child is a single process, re-imports
    nothing, and its pages are copy-on-write.

    A heartbeat that outlives its worker is worse than none, because it makes a dead
    shard look alive forever. Four mechanisms kill the child, and the first covers
    every exit path on its own:

    1. Stop-pipe EOF. The parent holds the write end and the kernel closes it when the
       parent dies for any reason, including SIGKILL, an OOM kill and the SLURM
       walltime kill; the child is in ``select`` on the read end, so it exits at once.
    2. A ``getppid`` check each tick, for the one case (1) cannot see: another process
       holding a dup of the write end.
    3. :meth:`HeartbeatProcess.stop`'s explicit reap, escalating to SIGKILL after
       :data:`_HB_STOP_GRACE_S`.
    4. The SLURM cgroup, which kills the whole job step.

    A fresh heartbeat proves the worker process is alive, never that its work is
    progressing: a parent deadlocked on a dead pool child would keep being stamped.
    Progress is the monitor's job.
    """
    s = Path(shard)
    if interval <= 0:
        return None
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
    except OSError:
        os.close(read_fd)
        os.close(write_fd)
        raise
    if pid == 0:  # child: must never return past this block
        try:
            os.close(write_fd)
            _heartbeat_child(s, read_fd, interval)
        except BaseException:  # noqa: BLE001,S110
            # Silent by design: this is a forked child of a worker holding models and
            # native thread pools, so reaching for the logger risks a lock the parent
            # held at fork time. Nothing may escape into the parent's interpreter state
            # either. The parent notices a dead child the usual way, a stale heartbeat.
            pass
        finally:
            os._exit(0)  # bypass atexit, finalization and buffered-stdio flush
    os.close(read_fd)
    return HeartbeatProcess(pid=pid, shard=s, stop_fd=write_fd)


def _hb_time(shard: Path) -> float | None:
    try:
        return float(_hb_path(shard).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def reap_stale(
    running: str | os.PathLike[str],
    pending: str | os.PathLike[str],
    max_age: float,
) -> list[Path]:
    """Return every ``running/`` shard whose heartbeat is older than ``max_age`` to ``pending``.

    Without this a killed worker strands its shard in ``running/`` forever. ``max_age``
    is the effective threshold the caller computes, typically a multiple of the
    heartbeat interval. Returns the reclaimed shard paths.
    """
    rdir = Path(running)
    now = _now()
    reclaimed: list[Path] = []
    for shard in sorted(rdir.glob("*")):
        if not shard.is_file():
            continue
        hb = _hb_time(shard)
        if hb is None:
            # A missing sidecar is not proof of a dead worker: `claim` renames the
            # shard into running/ and stamps the heartbeat immediately afterwards, and
            # on a distributed filesystem that sidecar takes a moment to become visible
            # from another node. Treating it as stale lets a concurrent worker yank a
            # live claim back to pending/, whose owner then fails on the vanished path.
            # Fall back to the shard's own ctime, which the rename bumps to the claim
            # instant; mtime is preserved by rename and would still read the seed time.
            try:
                hb = shard.stat().st_ctime
            except OSError:
                hb = 0.0  # genuinely unreadable, let the age test reclaim it
        if (now - hb) > max_age:
            reclaimed.append(requeue(shard, pending))
    return reclaimed


# --------------------------------------------------------------------------- #
# Poison-shard attempt counter.
# --------------------------------------------------------------------------- #
def attempts(shard: str | os.PathLike[str]) -> int:
    """Read a shard's persisted failure count, 0 if it has never failed."""
    try:
        return int(json.loads(_attempts_path(Path(shard)).read_text(encoding="utf-8"))["attempts"])
    except (OSError, ValueError, KeyError):
        return 0


def _write_attempts(shard: Path, n: int) -> None:
    ap = _attempts_path(shard)
    ap.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(ap)
    tmp.write_text(json.dumps({"shard": shard.name, "attempts": n}), encoding="utf-8")
    os.replace(tmp, ap)


def fail(
    shard: str | os.PathLike[str],
    pending: str | os.PathLike[str],
    failed: str | os.PathLike[str],
    k_max: int,
) -> Path:
    """Record one failure; requeue to ``pending`` while retriable, else move to ``failed/``.

    The attempt counter is persisted in ``meta/`` keyed by shard name, so it survives a
    worker restart and a poison shard cannot loop forever. On the ``k_max``-th failure
    the shard moves to ``failed/``, which is the orchestrator's hard stop.
    """
    s = Path(shard)
    n = attempts(s) + 1
    _write_attempts(s, n)
    if n >= k_max:
        fdir = Path(failed)
        fdir.mkdir(parents=True, exist_ok=True)
        # Tolerant move: this runs from the worker's exception handler and must not
        # itself raise on an already-moved shard.
        return _move_tolerant(s, fdir / s.name)
    return requeue(s, pending)


# --------------------------------------------------------------------------- #
# Validate before done.
# --------------------------------------------------------------------------- #
def mark_done(
    shard: str | os.PathLike[str],
    done: str | os.PathLike[str],
    key: str,
    *,
    validate: Callable[[Path], bool] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path | None:
    """Validate, then move a shard from ``running`` to ``done`` and write ``_SUCCESS(key)``.

    If ``validate`` is supplied and returns falsey the shard is not completed: it stays
    in ``running/`` for the caller to requeue and ``None`` is returned, which is the
    guard against a zero-output shard being marked done. On success the shard moves to
    ``done/`` and a key-bearing result row is written through ``common.io``, composing
    the rename claim with atomic, durable artifact IO.

    ``provenance`` adds observed machine facts (GPU model, node, job) to that row. It
    matters because a shard's output bytes can depend on the silicon that produced it,
    and an artifact assembled from a heterogeneous OR-constraint array must be able to
    say which card produced each part. It is always an observed fact and always
    additive: ``shard`` and ``key`` are unchanged.
    """
    s = Path(shard)
    if validate is not None and not validate(s):
        return None
    ddir = Path(done)
    ddir.mkdir(parents=True, exist_ok=True)
    dst = ddir / s.name
    os.rename(s, dst)
    result = ddir / f"{s.name}.result"
    row: dict[str, Any] = {"shard": s.name, "key": key}
    row.update({k: v for k, v in (provenance or {}).items() if v not in (None, "")})
    write_jsonl(result, [row])
    return dst
