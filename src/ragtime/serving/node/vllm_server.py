"""vLLM server bring-up and teardown, the only place a server starts.

``bringup(config, knobs) -> Handle`` launches ``vllm serve`` with the capacity-derived
``knobs`` (tensor parallelism, the xgrammar structured-output backend, a
``gpu_memory_utilization`` that leaves headroom for the co-resident clients), a port
derived from ``$SLURM_JOB_ID``, a ``/v1/models`` readiness poll and a signal-safe
teardown trap.
``teardown()`` stops the whole process tree.

The readiness poll fails fast and diagnoses itself: it aborts the moment the server
process exits, dials ``localhost`` with the proxy disabled, since compute nodes export
``http_proxy`` and the proxy returns 504, and raises ``VllmBringupError`` carrying the
child's exit code together with the root-cause line and the tail of the captured server
log. Without that, a server that died forty seconds in is reported ten minutes later as
an undifferentiated "did not become ready".

The call shape is ``python -m vllm.entrypoints.openai.api_server`` with
``--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
--enforce-eager --disable-custom-all-reduce --max-num-batched-tokens 4096`` plus NCCL
tuning by GPU type. Every parser name is read from config rather than hardcoded.
Structured output is enforced via ``response_format`` or ``structured_outputs`` at
request time; the legacy ``guided_json`` field is never used.

The knobs arrive as arguments rather than from a file: ``slurm/vllm_service.sbatch`` passes
``--tensor-parallel-size`` and ``--max-model-len`` on the ``--print-argv`` call, and nothing
here reads ``config/serving/<cluster>.yml``. Heavy and subprocess imports are lazy. This
module is GPU-bound and is exercised by the full gate rather than by small tests.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragtime.common import get_logger

__all__ = [
    "Handle",
    "VllmBringupError",
    "bringup",
    "gpu_memory_summary",
    "parse_startup_kv",
    "port_from_jobid",
    "server_argv",
    "teardown",
]

_log = get_logger("serving.vllm")


class VllmBringupError(RuntimeError):
    """vLLM failed to come up.

    Carries the child's exit code together with the root-cause line and the tail of the
    captured server log, so the failure is diagnosable from the traceback itself rather
    than only from a file the reader has to find.
    """

# Module-level handle so the signal trap can tear down the in-flight server.
_ACTIVE: Handle | None = None

# vLLM startup log lines carrying the actual KV budget and concurrency, for example
# "GPU KV cache size: 209,536 tokens" and "Maximum concurrency for 8,192 tokens per
# request: 25.58x".
_KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_CONCURRENCY_RE = re.compile(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x")

# The first exception line in vLLM's own log is the root cause; the API-server traceback
# only says "See root cause above". Surfacing it is the difference between "did not
# become ready" and a message naming the actual failure.
_ROOT_CAUSE_RE = re.compile(
    r"^.*?((?:torch\.)?(?:OutOfMemoryError|ValueError|RuntimeError|ImportError|"
    r"AssertionError|OSError|NotImplementedError):\s.*)$",
    re.MULTILINE,
)
_LOG_TAIL_LINES = 40

# Readiness budget. Cold bring-ups of the served 122B at TP=2 with a warm model cache
# take between two and five minutes from process start to "Application startup
# complete", so 1200 s is roughly four times the slowest observed. It is only ever spent
# on a server that is still alive, since a dead child aborts the poll immediately.
_READY_TIMEOUT_S = 1200.0
_READY_HEARTBEAT_S = 60.0


@dataclass(slots=True)
class Handle:
    """A live vLLM server handle.

    ``log_path`` captures the server's stdout so the runtime k-ceiling can read vLLM's
    own KV budget and concurrency.
    """

    proc: Any
    port: int
    base_url: str
    model: str
    log_path: str | None = None
    # The open file object the child writes into, closed by `teardown`; an unclosed
    # handle leaks a file descriptor per bring-up attempt.
    log_fh: Any = None


def parse_startup_kv(text: str) -> tuple[int, float] | None:
    """Parse vLLM's startup log for ``(kv_cache_tokens, max_concurrency)``.

    Returns ``None`` if the lines are not present yet. This is what the runtime k-ceiling
    reads, since the capacity calculator's numbers are only a conservative pre-flight
    estimate.
    """
    m_tok = _KV_TOKENS_RE.search(text)
    m_conc = _CONCURRENCY_RE.search(text)
    if not (m_tok and m_conc):
        return None
    return int(m_tok.group(1).replace(",", "")), float(m_conc.group(1))


def port_from_jobid(default: int = 8000) -> int:
    """Derive a stable per-job port from ``$SLURM_JOB_ID`` (falls back to ``default``)."""
    jid = os.environ.get("SLURM_JOB_ID")
    if jid and jid.isdigit():
        return 20000 + (int(jid) % 20000)
    return default


def _poll_ready(
    base_url: str,
    *,
    proc: Any = None,
    timeout_s: float = _READY_TIMEOUT_S,
    interval_s: float = 3.0,
) -> str | None:
    """Poll ``/v1/models`` until the server serves.

    Returns ``None`` on success, and otherwise a one-line reason. Two properties matter:

    * A dead child aborts the poll immediately. If ``proc`` has exited the server is
      never coming up, and waiting out the full budget only hides the real error while
      burning GPU time.
    * The poll never goes through a proxy. Compute nodes export ``http_proxy`` and the
      proxy returns 504 on cross-node HTTP, so ``localhost`` is dialled directly and the
      opener is built with an empty ``ProxyHandler`` regardless of the environment.
    """
    import time
    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.time()
    deadline = started + timeout_s
    url = f"{base_url}/models"
    next_beat = started + _READY_HEARTBEAT_S
    while time.time() < deadline:
        try:
            with opener.open(url, timeout=5) as resp:
                if resp.status == 200:
                    return None
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        rc = proc.poll() if proc is not None else None
        if rc is not None:
            return (
                f"server process exited with code {rc} after "
                f"{time.time() - started:.0f}s without serving {url}"
            )
        now = time.time()
        if now >= next_beat:
            _log.info("vllm.waiting", url=url, elapsed_s=round(now - started), timeout_s=timeout_s)
            next_beat = now + _READY_HEARTBEAT_S
        time.sleep(interval_s)
    return f"timed out after {timeout_s:.0f}s while the server process was still alive"


def gpu_memory_summary() -> str:
    """``free/total`` VRAM per visible GPU, e.g. ``gpu0=99.0/140.1GiB``.

    vLLM requires ``gpu_memory_utilization`` times the card's total VRAM to be free at
    startup, so a card that merely has room is not enough: anything already resident, such
    as a co-located encoder or an orphaned server, causes a bring-up failure. Recording
    this at launch makes that visible without a second job.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    parts = []
    for i, line in enumerate(out.strip().splitlines()):
        try:
            free_mib, total_mib = (int(x) for x in line.split(","))
        except ValueError:
            continue
        parts.append(f"gpu{i}={free_mib / 1024:.1f}/{total_mib / 1024:.1f}GiB")
    return " ".join(parts) or "unavailable"


def _log_diagnosis(log_path: str | Path | None) -> tuple[str, str]:
    """``(root_cause_line, tail)`` from the captured server log (best effort)."""
    if log_path is None:
        return ("no server log was captured (log_path=None)", "")
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return (f"server log unreadable ({type(exc).__name__})", "")
    m = _ROOT_CAUSE_RE.search(text)
    root = m.group(1).strip() if m else "no exception line found in the server log"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return (root, "\n".join(lines[-_LOG_TAIL_LINES:]))


def _install_teardown_trap() -> None:
    import signal

    def _handler(signum: int, frame: Any) -> None:
        _log.warning("vllm.signal_teardown")
        teardown()

    try:
        signal.signal(signal.SIGUSR1, _handler)
    except (ValueError, OSError):
        # Not in the main thread, for example under a test harness, so skip the trap.
        pass


def _nccl_env() -> dict[str, str]:
    """NCCL and GLOO tuning by GPU type.

    NVLink cards keep peer-to-peer enabled; non-NVLink cards disable shared memory and
    peer-to-peer. Both set GDR level 0 and run gloo over loopback.
    """
    env = {"NCCL_NET_GDR_LEVEL": "0", "GLOO_SOCKET_IFNAME": "lo"}
    try:
        name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        name = ""
    if not re.search(r"H100|H200|B200|Blackwell", name, re.IGNORECASE):
        env["NCCL_SHM_DISABLE"] = "1"
        env["NCCL_P2P_DISABLE"] = "1"
    return env


def server_argv(
    *,
    model: str,
    tensor_parallel_size: int | str,
    max_model_len: int | str,
    gpu_memory_utilization: float | str,
    reasoning_parser: str = "qwen3",
    tool_parser: str = "qwen3_coder",
    port: int | str,
    python: str | None = None,
) -> list[str]:
    """Build the argv for a vLLM server: one definition, used by production and by every gate.

    A second, hand-maintained copy of this flag list previously lived in the launcher
    script and drifted from this one, so a gate measured a configuration production never
    ran. The launcher now executes this list, via ``--print-argv``, rather than restating
    it, and there is only ever one shape of running vLLM.

    The four values that legitimately vary per node (tensor parallelism, max model length,
    GPU memory utilization and port) are arguments; the rest are the invariant deployment,
    so changing a flag here changes production and every gate together.
    """
    return [
        python or sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--enable-prefix-caching",
        "--reasoning-parser",
        str(reasoning_parser),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        str(tool_parser),
        "--enforce-eager",
        "--disable-custom-all-reduce",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]


def bringup(config: Any, knobs: Any, *, log_path: str | Path | None = None) -> Handle:
    """Launch the vLLM server for this node using ``knobs`` and ``config``.

    Tensor parallelism, max model length and GPU memory utilization come from ``knobs``,
    which are config- and capacity-driven rather than hardcoded; tensor parallelism falls
    back to ``$SLURM_GPUS_ON_NODE``. The call blocks on the ``/v1/models`` readiness poll,
    installs the teardown signal trap and returns a handle. ``log_path`` must be on shared
    storage rather than node-local ``/tmp``, and captures the server's stdout and stderr
    for the capacity-versus-log cross-check.
    """
    global _ACTIVE

    llm_cfg = dict(config.blocks.get("llm", {})) if config is not None else {}
    model = str(llm_cfg.get("model", ""))
    # Parser names for the served instruct deployment, overridable from config.
    tool_parser = str(llm_cfg.get("tool_parser", "qwen3_coder"))
    reasoning_parser = str(llm_cfg.get("reasoning_parser", "qwen3"))

    tp = getattr(knobs, "tensor_parallel_size", None)
    if tp is None:
        tp = knobs["tensor_parallel_size"] if isinstance(knobs, dict) else None
    tp = tp or int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
    mml = getattr(knobs, "max_model_len", None) or (
        knobs["max_model_len"] if isinstance(knobs, dict) else 8192
    )
    gmu = getattr(knobs, "gpu_memory_utilization", None) or (
        knobs["gpu_memory_utilization"] if isinstance(knobs, dict) else 0.92
    )

    port = port_from_jobid()
    base_url = f"http://localhost:{port}/v1"
    cmd = server_argv(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=mml,
        gpu_memory_utilization=gmu,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
        port=port,
    )
    env = {**os.environ, **_nccl_env(), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}
    vram = gpu_memory_summary()
    _log.info(
        "vllm.bringup",
        model=model,
        port=port,
        tensor_parallel_size=tp,
        max_model_len=mml,
        gpu_memory_utilization=gmu,
        vram=vram,
    )

    log_fh = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fh = Path(log_path).open("w", encoding="utf-8")  # noqa: SIM115 (held by the proc)
    # `start_new_session` puts the API server and its engine children in their own process
    # group, so teardown can reap the whole tree; an orphaned engine keeps holding VRAM and
    # makes the next bring-up fail for a reason that looks unrelated.
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True
    )
    handle = Handle(
        proc=proc,
        port=port,
        base_url=base_url,
        model=model,
        log_path=str(log_path) if log_path is not None else None,
        log_fh=log_fh,
    )
    _ACTIVE = handle
    _install_teardown_trap()

    reason = _poll_ready(base_url, proc=proc)
    if reason is not None:
        root, tail = _log_diagnosis(log_path)
        _log.error("vllm.bringup_failed", model=model, port=port, reason=reason, root_cause=root)
        teardown()
        raise VllmBringupError(
            f"vLLM did not become ready on {base_url}: {reason}.\n"
            f"  request: model={model} tensor_parallel_size={tp} max_model_len={mml} "
            f"gpu_memory_utilization={gmu}\n"
            f"  VRAM at launch (free/total): {vram}\n"
            f"  root cause from the server log: {root}\n"
            f"  server log: {log_path}\n"
            f"--- last {_LOG_TAIL_LINES} log lines ---\n{tail}"
        )
    _log.info("vllm.ready", model=model, port=port)
    return handle


def _signal_group(proc: Any, sig: int) -> bool:
    """Signal the child's whole process group (see ``start_new_session`` in bringup)."""
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int):
        return False
    try:
        os.killpg(os.getpgid(pid), sig)
        return True
    except (OSError, AttributeError, ProcessLookupError):
        return False


def teardown(handle: Handle | None = None) -> None:
    """Stop the vLLM server process tree. Signal-safe and idempotent.

    The API server forks engine workers, and terminating only the parent can leave a
    worker resident on the GPU, which then fails the next bring-up with a memory error
    that looks like a capacity bug. The whole process group is therefore signalled, and
    then the captured log file handle is closed.
    """
    import signal

    global _ACTIVE
    h = handle or _ACTIVE
    if h is None or h.proc is None:
        return
    try:
        if not _signal_group(h.proc, signal.SIGTERM):
            h.proc.terminate()
        h.proc.wait(timeout=30)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        _log.warning("vllm.teardown_terminate_failed", error=type(exc).__name__)
        with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
            if not _signal_group(h.proc, signal.SIGKILL):
                h.proc.kill()
    finally:
        with contextlib.suppress(OSError, ValueError):
            if h.log_fh is not None:
                h.log_fh.close()
                h.log_fh = None
        if h is _ACTIVE:
            _ACTIVE = None


# `--print-argv` is how a shell script gets production's argv instead of keeping its own
# copy. The launcher script asks for the flag list here, so the service a gate measures and
# the server a scored run launches are the same command by construction. It prints and exits
# without starting anything, so a shell can safely read it with `mapfile`.
def _main(argv: list[str] | None = None) -> int:
    import argparse
    import shlex

    p = argparse.ArgumentParser(prog="serving.node.vllm_server", description=__doc__)
    p.add_argument("--print-argv", action="store_true", required=True,
                   help="print the production vLLM argv (one token per line) and exit")
    p.add_argument("--model", required=True)
    p.add_argument("--tensor-parallel-size", required=True)
    p.add_argument("--max-model-len", required=True)
    p.add_argument("--gpu-memory-utilization", default="0.92")
    p.add_argument("--reasoning-parser", default="qwen3")
    p.add_argument("--tool-parser", default="qwen3_coder")
    p.add_argument("--port", required=True)
    p.add_argument(
        "--python", default=None, help="interpreter to put in argv[0] (default: this one)"
    )
    p.add_argument(
        "--shell",
        action="store_true",
        help="print one shell-quoted line instead of one token per line",
    )
    a = p.parse_args(argv)

    cmd = server_argv(
        model=a.model,
        tensor_parallel_size=a.tensor_parallel_size,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        reasoning_parser=a.reasoning_parser,
        tool_parser=a.tool_parser,
        port=a.port,
        python=a.python,
    )
    print(shlex.join(cmd) if a.shell else "\n".join(cmd))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(_main())
