"""The heartbeat must not be startable by the GIL, and the reaper window must fit the stage.

The defect these tests pin is not theoretical. Refreshing a claimed shard's heartbeat from
a daemon thread does not work here: ``threading.Event.wait`` needs the GIL to return, and a
native extension that holds the GIL for its whole call starves it for minutes. On a
corpus-index build the ``.hb`` files went hundreds of seconds stale while the workers burned
CPU at high parallelism, so every other worker's ``reap_stale`` pass reclaimed in-flight
shards: a handful of parts finished in an hour while ``running/`` and ``pending/`` churned,
and several parts were built by two array tasks at once. And it could not even fail:
``reap_stale`` uses ``requeue``, which does not increment ``attempts``, so nothing ever
reached ``k_max`` and the churn was unbounded.

Four properties, each a direct answer to one link of that chain:

1. It is GIL-proof: the refresher stamps while the parent holds the GIL in a way that
   provably starves a Python thread. This is the load-bearing one, and it is written so it
   would fail against a thread implementation rather than pass vacuously: the parent runs a
   pure-Python busy loop under a raised ``sys.setswitchinterval``, which is exactly the
   condition under which a thread gets no timeslice.
2. It dies with its parent, including when the parent is killed outright. A heartbeat that
   outlives its worker is worse than none, because a dead shard then looks alive forever and
   is never reclaimed. The test kills a real subprocess and watches the grandchild go.
3. It degrades loudly rather than silently: if ``fork`` itself fails, the thread fallback runs
   and says so; it never proceeds with no heartbeat at all.
4. The reaper window is per substage and is sane: ``execution.<substage>_max_age_s`` is read
   for each link of the corpus chain, every substage's key is a declared schema key (so a new
   link cannot typo its way into the default), and a window at or below the heartbeat interval
   is rejected at load, since it would reclaim every live shard on the first pass.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ragtime import config
from ragtime.orchestration import cli, saturate
from ragtime.orchestration.slurm import workqueue

pytestmark = pytest.mark.small

_REPO_ROOT = Path(__file__).resolve().parents[2]
_E2E = _REPO_ROOT / "config" / "e2e-omt.yml"


def _hb_value(shard: Path) -> float | None:
    try:
        return float(workqueue._hb_path(shard).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _hold_the_gil(seconds: float) -> None:
    """Burn CPU in pure Python without ever releasing the GIL.

    No syscall, no I/O and no ``sleep``, since every one of those is a GIL release point.
    Combined with a raised switch interval, which is the caller's job, this reproduces the
    production failure: the interpreter never hands another Python thread a timeslice.
    """
    deadline = time.monotonic() + seconds
    x = 0
    while True:
        for _ in range(200_000):
            x += 1
        if time.monotonic() >= deadline:  # the only call, once per 200k iterations
            return


# --------------------------------------------------------------------------- #
# 1. GIL-proof.
# --------------------------------------------------------------------------- #
def test_heartbeat_advances_while_the_parent_holds_the_gil(wq_root) -> None:
    """The property the thread could not have: stamps land during a GIL-holding call."""
    wq_root.drop_shards(1)
    shard = workqueue.claim(wq_root.dirs.pending, wq_root.dirs.running)
    assert shard is not None
    before = _hb_value(shard)
    assert before is not None  # claim() stamps once

    hb = workqueue.start_heartbeat(shard, 0.05)
    assert hb is not None
    assert hb.pid != os.getpid()  # a process, not a thread
    previous_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(30.0)  # a Python thread gets no timeslice for 30 s
        _hold_the_gil(1.0)  # >> 20 heartbeat intervals
        during = _hb_value(shard)
    finally:
        sys.setswitchinterval(previous_interval)
        hb.stop()

    assert during is not None
    assert during > before, (
        "the heartbeat did not advance while the parent held the GIL; this is the "
        "assemble livelock, and it is what a threaded refresher does"
    )


def test_stop_reaps_the_child_and_is_idempotent(wq_root) -> None:
    wq_root.drop_shards(1)
    shard = workqueue.claim(wq_root.dirs.pending, wq_root.dirs.running)
    assert shard is not None
    hb = workqueue.start_heartbeat(shard, 0.05)
    assert hb is not None
    pid = hb.pid
    hb.stop()
    hb.stop()  # idempotent: run_worker's `finally` may double-fire on an exception path
    with pytest.raises(OSError):  # already reaped: no zombie left behind
        os.waitpid(pid, os.WNOHANG)


def test_closing_the_stop_pipe_alone_ends_the_child(wq_root) -> None:
    """EOF is the mechanism: the same event the kernel delivers when a parent is killed."""
    wq_root.drop_shards(1)
    shard = workqueue.claim(wq_root.dirs.pending, wq_root.dirs.running)
    assert shard is not None
    hb = workqueue.start_heartbeat(shard, 30.0)  # a LONG interval: only EOF can wake it
    assert hb is not None
    os.close(hb.stop_fd)  # simulate the parent dying without calling stop()
    hb.stopped = True  # don't double-close in the assertion path below
    deadline = time.monotonic() + 10.0
    reaped = 0
    while time.monotonic() < deadline:
        reaped, _status = os.waitpid(hb.pid, os.WNOHANG)
        if reaped == hb.pid:
            break
        time.sleep(0.02)
    assert reaped == hb.pid, "the child outlived its stop pipe and would stamp forever"


def test_disabled_when_the_interval_is_not_positive(wq_root) -> None:
    wq_root.drop_shards(1)
    shard = workqueue.claim(wq_root.dirs.pending, wq_root.dirs.running)
    assert shard is not None
    assert workqueue.start_heartbeat(shard, 0.0) is None
    assert saturate._start_heartbeat(shard, 0.0) is None


# --------------------------------------------------------------------------- #
# 2. Dies with its parent, including on SIGKILL.
# --------------------------------------------------------------------------- #
_SIGKILL_PARENT = """
import os, signal, sys, time
sys.path.insert(0, {code!r})
from ragtime.orchestration.slurm import workqueue
d = workqueue.init_queue({base!r})
shard = d.running / "shard_0000"
shard.write_bytes(b"x")
hb = workqueue.start_heartbeat(shard, 0.05)
print(hb.pid, flush=True)
time.sleep(0.3)
os.kill(os.getpid(), signal.SIGKILL)   # no finally, no atexit, no cleanup runs
"""


def test_child_dies_when_its_parent_is_sigkilled(tmp_path: Path) -> None:
    """The exit path that matters most: walltime / scancel / OOM all arrive as SIGKILL.

    A heartbeat outliving its worker is strictly worse than no heartbeat: the shard looks
    alive forever and nothing reclaims it. Nothing in the parent runs on this path: no
    ``finally`` and no ``atexit``, so only the kernel closing the stop pipe can end the child.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _SIGKILL_PARENT.format(
                code=str(_REPO_ROOT / "src"), base=str(tmp_path / "wq")
            ),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,  # it kills itself on purpose; the return code is the assertion
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    child_pid = int(proc.stdout.strip().splitlines()[-1])

    deadline = time.monotonic() + 30.0
    alive = True
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)  # the grandchild is reparented, so only a probe works
        except OSError:
            alive = False
            break
        time.sleep(0.05)
    assert not alive, (
        f"heartbeat child {child_pid} survived its killed parent, so its shard would "
        "look alive forever and never be reclaimed"
    )


# --------------------------------------------------------------------------- #
# 3. Degrades loudly.
# --------------------------------------------------------------------------- #
def test_fork_failure_falls_back_to_the_thread_and_says_so(
    wq_root, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    wq_root.drop_shards(1)
    shard = workqueue.claim(wq_root.dirs.pending, wq_root.dirs.running)
    assert shard is not None

    def _no_fork(*_args, **_kwargs):
        raise OSError(11, "Resource temporarily unavailable")

    monkeypatch.setattr(workqueue, "start_heartbeat", _no_fork)
    before = _hb_value(shard)
    with caplog.at_level("ERROR", logger="ragtime.orchestration.saturate"):
        hb = saturate._start_heartbeat(shard, 0.02)
    assert isinstance(hb, saturate._ThreadHeartbeat)  # degraded, NOT "no heartbeat at all"
    try:
        time.sleep(0.2)  # sleep RELEASES the GIL, so the fallback thread does run here
        during = _hb_value(shard)
    finally:
        hb.stop()
    assert during is not None and before is not None and during > before
    assert "fork_failed" in caplog.text  # the degradation is visible in the log


# --------------------------------------------------------------------------- #
# 4. The reaper window: per substage, declared, and sane.
# --------------------------------------------------------------------------- #
def _cfg_with(execution: dict) -> config.RunConfig:
    """The real ``e2e-omt.yml`` with ``execution`` overridden in memory, never on disk."""
    import dataclasses

    cfg = config.load(_E2E)
    blocks = dict(cfg.blocks)
    # Strip every shipped `*_max_age_s` first, so `_cfg_with({})` genuinely means "this
    # substage declares no window" and the default is what gets exercised. The shipped
    # configs carry `assemble_max_age_s`, so without the strip this
    # helper would silently test the shipped value while claiming to test the fallback,
    # the same trap as a config-silent test run against a config that sets the key.
    base = {
        k: v
        for k, v in dict(blocks.get("execution", {})).items()
        if not k.endswith("_max_age_s")
    }
    blocks["execution"] = {**base, **execution}
    return dataclasses.replace(cfg, blocks=blocks)


def test_every_substage_declares_a_max_age_key_that_the_schema_allows() -> None:
    """A new corpus link cannot silently fall back to the default by typo.

    ``execution`` is a closed schema, so an undeclared key would make every config fail to
    load, and a declared-but-unread key would be an inert knob.
    Both directions are checked here, derived from the substage tuple rather than restated.
    """
    from ragtime.config.schema import _ALLOWED

    allowed = _ALLOWED["execution"]
    keys = [sub.max_age_key for sub in cli._CORPUS_SUBSTAGES]
    assert len(set(keys)) == len(keys)  # no two substages share a window knob
    for key in keys:
        assert key in allowed, f"execution.{key} is not a declared schema key"


def test_max_age_defaults_then_reads_the_substage_key() -> None:
    sub = next(s for s in cli._CORPUS_SUBSTAGES if s.name == "assemble")
    assert cli._substage_max_age(_cfg_with({}), sub) == saturate.MAX_AGE_S
    assert cli._substage_max_age(_cfg_with({sub.max_age_key: 3600}), sub) == 3600.0
    # and it is genuinely per substage: one stage's window does not move another's.
    other = next(s for s in cli._CORPUS_SUBSTAGES if s.name == "merge")
    assert cli._substage_max_age(_cfg_with({sub.max_age_key: 3600}), other) == saturate.MAX_AGE_S


def test_a_window_at_or_below_the_heartbeat_is_rejected() -> None:
    sub = next(s for s in cli._CORPUS_SUBSTAGES if s.name == "assemble")
    for bad in (saturate.HEARTBEAT_S, saturate.HEARTBEAT_S / 2, 0, -1):
        with pytest.raises(config.ConfigError):
            cli._substage_max_age(_cfg_with({sub.max_age_key: bad}), sub)
    with pytest.raises(config.ConfigError):
        cli._substage_max_age(_cfg_with({sub.max_age_key: "soon"}), sub)


def test_default_window_is_many_heartbeats_wide() -> None:
    """Defence in depth: the default must survive a long run of dropped stamps, and must
    still be a window, since a reaper that never fires is a different bug: a crashed worker's
    shard would then be stranded for hours."""
    assert saturate.MAX_AGE_S >= 30 * saturate.HEARTBEAT_S
    assert saturate.MAX_AGE_S <= 3600.0
