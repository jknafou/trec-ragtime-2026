"""The vLLM bring-up failure path, with no GPU, no vLLM and no network.

The full tier can only observe a bring-up failure by spending a GPU-hour, so the behaviour
that made an earlier failure undiagnosable is pinned here instead: a server process that dies
must abort the readiness poll at once, and the raised error must carry vLLM's own root-cause
line rather than a bare "did not become ready".
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from ragtime.serving.node import vllm_server

pytestmark = pytest.mark.small

# Verbatim shape of the real failure: the API-server traceback says only
# "See root cause above", so the informative line is the EngineCore ValueError.
_REAL_LOG = """(APIServer pid=2346728) INFO 08-09 06:16:07 [utils.py:302] version 0.17.1
(EngineCore_DP0 pid=2347063) ERROR 08-09 06:16:46 [core.py:1100] EngineCore failed to start.
(EngineCore_DP0 pid=2347063) ValueError: Free memory on device cuda:0 (99.04/140.06 GiB) on \
startup is less than desired GPU memory utilization (0.92, 128.86 GiB). Decrease GPU memory \
utilization or reduce GPU memory used by other processes.
(APIServer pid=2346728) RuntimeError: Engine core initialization failed. See root cause above.
"""


class _FakeProc:
    """A subprocess stand-in: ``exit_code=None`` stays alive, otherwise it is dead."""

    pid = None  # not a real pid -> teardown falls back to terminate(), never killpg

    def __init__(self, exit_code: int | None = None, *, stdout=None, log_text: str = ""):
        self._exit_code = exit_code
        self.terminated = False
        if stdout is not None and log_text:
            stdout.write(log_text)
            stdout.flush()

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = -15

    def kill(self):
        self._exit_code = -9

    def wait(self, timeout=None):
        return self._exit_code


def test_vs1_poll_aborts_immediately_when_the_server_process_exited():
    """A dead child must not be polled until the deadline: the poll returns as soon as it exits."""
    started = time.monotonic()
    reason = _poll(_FakeProc(exit_code=1), timeout_s=60.0)
    assert time.monotonic() - started < 15.0, "poll waited on a process that had already exited"
    assert reason is not None
    assert "exited with code 1" in reason


def test_vs2_poll_reports_a_timeout_only_while_the_process_is_alive():
    reason = _poll(_FakeProc(exit_code=None), timeout_s=1.0)
    assert reason is not None
    assert "timed out" in reason and "still alive" in reason


def _poll(proc, *, timeout_s):
    # Port 1 is never listening; the poll's connection attempts all fail fast.
    return vllm_server._poll_ready(
        "http://localhost:1/v1", proc=proc, timeout_s=timeout_s, interval_s=0.2
    )


def test_vs3_log_diagnosis_surfaces_vllms_own_root_cause_line(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text(_REAL_LOG, encoding="utf-8")
    root, tail = vllm_server._log_diagnosis(log)
    assert root.startswith("ValueError: Free memory on device cuda:0")
    assert "128.86 GiB" in root
    assert "EngineCore failed to start" in tail


def test_vs4_bringup_raises_a_diagnosable_error_and_closes_the_log(tmp_path, monkeypatch):
    """The exception the caller sees must name the exit code and the root cause."""
    log_path = tmp_path / "vllm.log"
    captured: dict[str, object] = {}

    real_popen = vllm_server.subprocess.Popen

    def _fake_popen(cmd, **kwargs):
        # bringup's own launch is the one asking for its own session; anything else
        # (nvidia-smi via subprocess.run, which goes through Popen too) is passed on.
        if "start_new_session" not in kwargs:
            return real_popen(cmd, **kwargs)
        captured["cmd"] = cmd
        captured["start_new_session"] = kwargs["start_new_session"]
        captured["stdout"] = kwargs.get("stdout")
        return _FakeProc(exit_code=1, stdout=kwargs.get("stdout"), log_text=_REAL_LOG)

    monkeypatch.setattr(vllm_server.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        vllm_server, "_poll_ready", lambda *a, **k: "server process exited with code 1 after 40s"
    )

    config = SimpleNamespace(blocks={"llm": {"model": "Qwen/Qwen3.5-122B-A10B-FP8"}})
    knobs = SimpleNamespace(
        tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.92
    )
    with pytest.raises(vllm_server.VllmBringupError) as exc:
        vllm_server.bringup(config, knobs, log_path=log_path)

    msg = str(exc.value)
    assert "exited with code 1" in msg
    assert "Free memory on device cuda:0" in msg, "the real root cause is not in the exception"
    assert "Qwen/Qwen3.5-122B-A10B-FP8" in msg and "tensor_parallel_size=1" in msg
    assert str(log_path) in msg
    # the process group flag is what lets teardown reap EngineCore children
    assert captured["start_new_session"] is True
    # teardown ran on the failure path: the captured log file handle is closed, not leaked
    assert captured["stdout"].closed
    assert vllm_server._ACTIVE is None


def test_vs5_gpu_memory_summary_never_raises_without_nvidia_smi(monkeypatch):
    def _boom(*a, **k):
        raise OSError("nvidia-smi not found")

    monkeypatch.setattr(vllm_server.subprocess, "run", _boom)
    assert vllm_server.gpu_memory_summary() == "unavailable"


def test_vs6_port_is_derived_from_the_job_id_not_hardcoded(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4294211")
    assert vllm_server.port_from_jobid() == 20000 + (4294211 % 20000)
    assert vllm_server.port_from_jobid() == 34211
    monkeypatch.delenv("SLURM_JOB_ID")
    assert vllm_server.port_from_jobid(default=8000) == 8000
