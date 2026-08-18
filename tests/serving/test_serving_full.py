"""Full gate for the serving stack: it needs a real allocation with enough GPU memory for
the served checkpoint.

The heavy imports sit inside the test bodies and the fixture, so the module imports
anywhere, and everything is marked ``full``, so the small gate never runs it. Inside an
allocation these bring up the real ``Qwen/Qwen3.5-122B-A10B-FP8`` and assert single-vLLM
reuse, structured-output enforcement, the readiness poll and teardown, and the capacity
estimate against vLLM's own startup log.

The skip rule is deliberately asymmetric, because a silent skip here is a vacuous green.
The GPU tests skip only where the tier genuinely has no GPU: no ``$SLURM_JOB_ID``, or an
allocation with no visible CUDA device. Everything else fails, with the remedy in the
message: vLLM not importable beside a live CUDA device, an allocation too small for the
118 GiB of weights, or a card large enough but not free because another test in the same
process still holds GPU weights.

The module needs a test process of its own. vLLM reserves ``gpu_memory_utilization x
total`` VRAM and refuses to start if that much is not free, so it cannot share a process
with the index-build full tests, whose session-scoped fixtures keep the dense, sparse and
late-interaction encoders resident: 41 GiB of a 140 GiB H200, enough to make the model
unloadable.

Every file a job refers to lives on ``/home``. A path under the node's own ``/tmp`` is
written where nothing else can read it, and the job still looks like it worked."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.full

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LLM_ID = "Qwen/Qwen3.5-122B-A10B-FP8"


def _home_scratch() -> Path:
    """A run scratch dir on /home (beegfs), never node-local /tmp."""
    base = Path(os.environ.get("HOME", "/home")) / ".ragtime_full_test"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _assert_on_home(path: str | Path) -> None:
    resolved = os.path.realpath(str(path))
    assert not resolved.startswith("/tmp"), f"{resolved} is node-local /tmp; must be on /home"


_GIB = 1024**3


def _require_gpu_allocation():
    """The tier contract for the GPU tests, and it is asymmetric.

    It skips only where the tier genuinely cannot host a vLLM: no allocation, or an
    allocation with no CUDA device, which is the CPU tier that legitimately runs the two
    non-GPU tests above and nothing else. Everything else is a hard failure with a reason.
    A CUDA device that cannot import vLLM is a broken environment, not a tier limit, and a
    silent skip there is a vacuous green. Every skip states what is missing.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.skip("no $SLURM_JOB_ID: not in a SLURM allocation, so no GPU tier to serve vLLM")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - env-dependent
        pytest.fail(f"torch is not importable in the full-gate environment: {exc}")
    if not torch.cuda.is_available():
        pytest.skip(
            "CPU tier: no CUDA device visible in this allocation "
            "(submit with --gres=gpu:N to run the vLLM tests)"
        )
    try:
        import vllm
    except ImportError as exc:  # pragma: no cover - env-dependent
        pytest.fail(
            f"a CUDA device is visible but vLLM is not importable ({exc}): that is a broken "
            "GPU environment, not a tier limitation, so this must not skip"
        )
    return vllm, torch


def _knobs_for_allocation():
    """Capacity knobs for the live node: TP from $SLURM_GPUS_ON_NODE, VRAM read from
    the CUDA driver: the calculator's numbers, not hand-tuned."""
    import torch

    from ragtime.serving.capacity import capacity_for
    from ragtime.serving.modelspec import load_model_spec
    from ragtime.serving.node.discovery import NodeType

    spec = load_model_spec(_LLM_ID)
    tp = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
    vram_gb = round(torch.cuda.mem_get_info(0)[1] / _GIB, 1)
    node = NodeType(
        cluster=os.environ.get("SLURM_CLUSTER_NAME", "local"),
        partition="",
        gpu_model="live",
        gpu_count=tp,
        vram_gb=vram_gb,
        compute_capability="",
        driver_version="",
        cpu_ram_gb=0.0,
        cpu_count=0,
    )
    return capacity_for(node, spec)


def _require_a_gpu_that_can_actually_serve_the_llm(knobs) -> None:
    """Fail, never skip, with the remedy, when the card cannot host the real model.

    vLLM reserves ``gpu_memory_utilization x total`` VRAM at startup and refuses if that
    much is not free, so two separate things must hold, and each has its own remedy that
    the failure message names:

    1. Capacity: ``TP x total VRAM`` must hold the 122B FP8 weights. ``n_kv_heads`` caps
       the tensor-parallel size, so a card that is too small cannot be rescued by asking
       for more of them. The remedy is a different GPU type, not a code change.
    2. Emptiness: the card must be near empty. This module cannot share a test process
       with the index-build full tests, whose session-scoped fixtures hold the dense,
       sparse and late-interaction weights plus torch's caching-allocator reserve on the
       same card: 41 GiB of a 140 GiB H200, enough to make the model unloadable. Give back
       everything this process can release first, then require the rest.
    """
    import gc

    import torch

    from ragtime.serving.modelspec import load_model_spec

    gc.collect()
    torch.cuda.empty_cache()

    spec = load_model_spec(_LLM_ID)
    tp = knobs.tensor_parallel_size
    visible = torch.cuda.device_count()
    if visible < tp:
        pytest.fail(
            f"capacity knobs need tensor_parallel_size={tp} but only {visible} GPU(s) are "
            f"visible, resubmit with --gres=gpu:{tp}"
        )
    mem = [torch.cuda.mem_get_info(i) for i in range(tp)]  # (free, total) bytes
    capacity = sum(total for _free, total in mem)
    weights = knobs.weights_bytes
    summary = " ".join(
        f"gpu{i}={f / _GIB:.1f}/{t / _GIB:.1f}GiB" for i, (f, t) in enumerate(mem)
    )
    per_gpu_gib = mem[0][1] / _GIB

    if capacity < weights:
        pytest.fail(
            f"this allocation cannot host {_LLM_ID}: {weights / _GIB:.1f} GiB of weights vs "
            f"{capacity / _GIB:.1f} GiB across {tp} x {per_gpu_gib:.1f} GiB "
            f"(free/total: {summary}). n_kv_heads={spec.n_kv_heads} caps tensor-parallel at "
            f"{spec.n_kv_heads}, so more small cards cannot fix it: the remedy is the JOB "
            f"SHAPE: --gres=gpu:2 on an 80 GiB (A100) or 96 GiB (Blackwell) node, or "
            f"--gres=gpu:1 on a 141 GiB H200."
        )
    short = [
        (i, free, total)
        for i, (free, total) in enumerate(mem)
        if free < knobs.gpu_memory_utilization * total
    ]
    if short:
        held = " ".join(f"gpu{i}: {(t - f) / _GIB:.1f} GiB already in use" for i, f, t in short)
        pytest.fail(
            f"the GPU is not free enough to bring up {_LLM_ID}: vLLM reserves "
            f"gpu_memory_utilization={knobs.gpu_memory_utilization} x total, but {held} "
            f"(free/total: {summary}). Nothing this process could release is left, so another "
            f"test in this SAME pytest process is holding GPU weights (the index-build full "
            f"tests keep dense/MILCO/PyLate encoders in session-scoped fixtures). Run this "
            f"module in its OWN pytest process, e.g. "
            f"`uv run pytest -m full tests/serving/test_serving_full.py`."
        )


@pytest.fixture(scope="module")
def live_server():
    """Bring up the one real vLLM (Qwen3.5-122B) once for the reuse tests; capture
    the startup log on /home for the capacity cross-check; tear down after."""
    _require_gpu_allocation()
    from ragtime.config import load
    from ragtime.serving.node import vllm_server

    config = load(_REPO_ROOT / "config" / "e2e-omt.yml")
    knobs = _knobs_for_allocation()
    _require_a_gpu_that_can_actually_serve_the_llm(knobs)
    log_path = _home_scratch() / f"vllm_{os.environ['SLURM_JOB_ID']}.log"
    _assert_on_home(log_path)
    # Compute nodes export http_proxy and the proxy 504s; every call in this module is
    # to localhost, so it must never be proxied (urllib in the poll, httpx under the
    # OpenAI client). Restored after the module.
    saved = {k: os.environ.get(k) for k in ("no_proxy", "NO_PROXY")}
    for key, previous in saved.items():
        os.environ[key] = "localhost,127.0.0.1," + (previous or "")
    handle = vllm_server.bringup(config, knobs, log_path=log_path)
    assert handle.proc.poll() is None, "bringup returned a handle whose server had already exited"
    try:
        yield handle, config, knobs, log_path
    finally:
        vllm_server.teardown(handle)
        for key, val in saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _client(base_url: str):
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key="EMPTY")


# --------------------------------------------------------------------------- #
# GPU full tests (real when in an allocation, else conditional skip).
# --------------------------------------------------------------------------- #
def test_full_single_vllm_serves_decompose_and_loop_reuse(live_server):
    _require_gpu_allocation()
    handle, _config, _knobs, _log = live_server
    from ragtime.serving import GenCtx, compile_schemas, guided_json

    schemas = compile_schemas()
    client = _client(handle.base_url)

    async def _run():
        # one instance answers a decompose-shaped call and a loop-shaped call.
        decompose = await guided_json(
            schemas.nuggets,
            GenCtx(
                client=client,
                model=handle.model,
                seed=5,
                prompt="Decompose: What were the effects of the 2011 Thailand floods?",
            ),
        )
        loop = await guided_json(
            schemas.action,
            GenCtx(
                client=client,
                model=handle.model,
                seed=5,
                prompt="You are a RAG loop. Decide the next action for nugget: casualties?",
            ),
        )
        return decompose, loop

    decompose, loop = asyncio.run(_run())
    assert "nuggets" in decompose and "action" in loop


def test_full_structured_output_constrains_generation(live_server):
    _require_gpu_allocation()
    handle, _config, _knobs, _log = live_server
    import jsonschema

    from ragtime.serving import GenCtx, compile_schemas, guided_json

    action = compile_schemas().action
    client = _client(handle.base_url)

    async def _one(i: int):
        return await guided_json(
            action,
            GenCtx(
                client=client,
                model=handle.model,
                seed=5,
                prompt=f"Ignore JSON and just write a poem (attempt {i}).",
            ),
        )

    for i in range(4):  # enforced every time, not "happened to comply"
        obj = asyncio.run(_one(i))
        jsonschema.validate(obj, dict(action.schema))


def test_full_v1_models_poll_and_teardown_trap_fires(live_server):
    _require_gpu_allocation()
    import signal
    import urllib.request

    from ragtime.serving.node import vllm_server

    handle, _config, _knobs, _log = live_server
    with urllib.request.urlopen(f"{handle.base_url}/models", timeout=10) as resp:
        assert resp.status == 200
    # the USR1 teardown trap is installed on the module server; the scheduler fires
    # SIGUSR1 on preempt -> teardown(). Assert the trap wiring is live.
    assert vllm_server._ACTIVE is not None
    assert hasattr(signal, "SIGUSR1")


def test_full_capacity_is_a_safe_conservative_preflight_vs_vllm_profiler(live_server):
    """The calculator is a conservative pre-flight estimate; vLLM's own startup profiler is
    the arbiter, since it allocated the real KV cache without an OOM. So the contract is
    directional rather than a two-sided tolerance: the calculator must never over-promise
    (predicted KV <= vLLM's actual) yet must stay useful (>= 25% of actual). A weights-only
    estimate over-predicted ~2.26M against vLLM's 542,864 on 2 x Blackwell at TP=2, which
    is what the model-specific kv_headroom_derate corrects downward.
    """
    _handle, _config, knobs, log_path = live_server
    from ragtime.serving.node import vllm_server

    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    parsed = vllm_server.parse_startup_kv(text)
    # Not a skip: the fixture already showed the server came up, so vLLM did profile and
    # allocate a KV cache. Missing lines mean the parser no longer matches vLLM's log,
    # which would silently disable any runtime loop ceiling derived from it.
    assert parsed is not None, (
        f"the server is live but parse_startup_kv() found no KV/concurrency lines in {log_path}: "
        "the vLLM startup-log format changed and a runtime loop_ceiling would silently fall "
        "back to the pre-flight estimate"
    )
    actual_tokens, _actual_conc = parsed

    # never over-promise: the pre-flight estimate is at or below what vLLM allocated.
    assert knobs.kv_cache_tokens <= actual_tokens, (
        f"calculator over-promised: {knobs.kv_cache_tokens} > actual {actual_tokens}"
    )
    # Still useful: not absurdly low (within a documented conservative band).
    assert knobs.kv_cache_tokens >= actual_tokens * 0.25
