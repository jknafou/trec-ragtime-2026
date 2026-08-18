"""Deterministic serving-capacity calculator: pure, stdlib only, unit-testable.

Maps ``(NodeType, ModelSpec) -> ServingKnobs`` for the LLM and
``(NodeType, MtSpec) -> MtKnobs`` for the MT model. No library models vLLM's
tensor-parallel and KV budget in a form that can be imported and pinned, and vLLM's
own profiler needs a live GPU, so this is a first-principles pre-flight estimate.
vLLM's startup log (``Available KV cache memory`` and ``Maximum concurrency``) stays
the runtime arbiter.

Formulae:

* ``weights_bytes = params_total x bytes(weight dtype)``, using ``params_total`` with
  all MoE experts resident rather than ``params_active``, which would under-predict.
* ``kv_bytes/token = 2 x n_full_attn_layers x n_kv_heads x head_dim x bytes(kv
  dtype)``, summed over attention layers only, since hybrid DeltaNet layers carry no
  KV proportional to sequence length.
* ``tensor_parallel_size`` is the smallest divisor of ``gpu_count`` that divides both
  ``n_heads`` and ``n_kv_heads`` and still fits the weights with activation headroom;
  it falls back to the largest head-divisible divisor if none fits.
* KV budget = ``(tp x vram x util x (1 - activation_headroom) - weights) x
  kv_headroom_derate``, where ``kv_headroom_derate`` is the per-model fraction of the
  post-weights budget that becomes usable KV. The remainder is non-weight overhead
  (activations, attention-kernel workspace, a multimodal tower, NCCL buffers). The
  k-loop ceiling is derived from this conservative budget.

What this calculator produced, and what it did not. Its output is the committed
``config/serving/<cluster>.yml``: every ``llm`` and ``mt`` sub-block there is a
:class:`ServingKnobs` / :class:`MtKnobs` written out by ``node.discovery.profile_entry``, and
``tests/serving/test_discovery_small.py``'s ``test_committed_cluster_profile_matches_calculator``
re-derives that whole committed file from this module, so a hand-edited knob fails. The
profile is a survey of the cluster, not a launch input.

No runtime path calls this. :func:`derive_loop_ceiling` in particular is not what the runs
used: ``pipeline.driver.resolve_ceiling`` prefers the fairness-shared
``rag_loop.fan_out.concurrency``, and every shipped config pins it by hand, saying why --
"set explicitly rather than derived from KV headroom, because the derivation divides the
cache by what a request could consume rather than by what ``budget.token_budget`` allows, and
over-provisions by ~6.5x". The derivation is kept as the thing that number was judged
against, not as the thing that decided it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .modelspec import ModelSpec, MtSpec
from .node.discovery import NodeType

__all__ = [
    "MtKnobs",
    "ServingKnobs",
    "capacity_for",
    "derive_loop_ceiling",
    "dtype_bytes",
    "mt_capacity",
]

_GIB = 1024**3

_DTYPE_BYTES: dict[str, int] = {
    "fp8": 1,
    "int8": 1,
    "e4m3": 1,
    "e5m2": 1,
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
    "float": 4,
}

# Activation and other non-KV overhead as a fraction of usable VRAM, calibrated so the
# pre-flight KV estimate lands within a few percent of what vLLM reports at startup.
_ACTIVATION_HEADROOM = 0.06
# Matches the GPU memory utilization the served instances are launched with, so the KV
# prediction is comparable to vLLM's startup log.
_DEFAULT_UTIL = 0.92
_MAX_MODEL_LEN_CAP = 8192
_MAX_NUM_SEQS_CAP = 256


def dtype_bytes(dtype: str) -> int:
    """Bytes per element for a weight/KV dtype (fp8/int8=1, bf16/fp16=2, fp32=4)."""
    try:
        return _DTYPE_BYTES[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {dtype!r}") from exc


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


@dataclass(frozen=True, slots=True)
class ServingKnobs:
    """The vLLM serving knobs a node and model fit into."""

    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    kv_cache_tokens: int
    max_concurrency: float
    weights_bytes: int
    kv_bytes_per_token: int
    loop_ceiling: int


@dataclass(frozen=True, slots=True)
class MtKnobs:
    """The MT serving knobs: weights plus a beam x seq_len batch budget, no KV term."""

    batch_size: int
    beam_size: int
    max_seq_len: int
    weights_bytes: int


def derive_loop_ceiling(max_num_seqs: int, kv_cache_tokens: int, tokens_per_loop: int) -> int:
    """Return the k-loop concurrency ceiling as a pure function.

    This is the number of concurrent RAG-loop coroutines a single vLLM instance can
    serve, bounded by both ``max_num_seqs`` and the KV budget at ``tokens_per_loop``
    tokens per loop. It is always at least 1, and it is the value ``fanout.map`` reads
    when no explicit ``rag_loop.fan_out.concurrency`` is configured.
    """
    if tokens_per_loop <= 0:
        return max(1, max_num_seqs)
    kv_ceiling = kv_cache_tokens // tokens_per_loop
    return max(1, min(max_num_seqs, kv_ceiling))


def _pick_tp(node: NodeType, weights_bytes: int, spec: ModelSpec, util: float) -> int:
    head_ok = [
        d for d in _divisors(node.gpu_count) if spec.n_heads % d == 0 and spec.n_kv_heads % d == 0
    ]
    for tp in head_ok:  # ascending, so the first hit is the smallest that fits
        usable = tp * node.vram_gb * _GIB * util
        if weights_bytes <= usable * (1 - _ACTIVATION_HEADROOM):
            return tp
    # Nothing head-divisible fits the weights; the estimate will show a tight or zero
    # KV budget, which is the signal the caller needs.
    return head_ok[-1] if head_ok else node.gpu_count


def capacity_for(
    node: NodeType,
    model: ModelSpec,
    *,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = _DEFAULT_UTIL,
) -> ServingKnobs:
    """Size the vLLM serving knobs for ``model`` on ``node``."""
    weights_bytes = int(model.params_total * dtype_bytes(model.dtype))
    kv_layers = model.n_full_attn_layers or model.n_layers
    kv_bytes_per_token = 2 * kv_layers * model.n_kv_heads * model.head_dim * dtype_bytes(
        model.kv_dtype
    )
    mml = (
        max_model_len
        if max_model_len is not None
        else min(model.context_window, _MAX_MODEL_LEN_CAP)
    )

    # TP selection considers weights only: the parameters must physically fit, and
    # runtime overhead is not part of whether the model loads.
    tp = _pick_tp(node, weights_bytes, model, gpu_memory_utilization)
    usable = tp * node.vram_gb * _GIB * gpu_memory_utilization
    post_weights = max(0.0, usable * (1 - _ACTIVATION_HEADROOM) - weights_bytes)
    # Only a model-specific fraction of the post-weights budget becomes usable KV; the
    # rest is activations, kernel workspace and communication buffers.
    available_kv = post_weights * model.kv_headroom_derate
    kv_tokens = int(available_kv // kv_bytes_per_token)
    max_concurrency = (kv_tokens / mml) if mml else 0.0
    max_num_seqs = max(1, min(_MAX_NUM_SEQS_CAP, int(max_concurrency))) if kv_tokens else 1
    loop_ceiling = derive_loop_ceiling(max_num_seqs, kv_tokens, mml)

    return ServingKnobs(
        tensor_parallel_size=tp,
        max_model_len=mml,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_tokens=kv_tokens,
        max_concurrency=max_concurrency,
        weights_bytes=weights_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
        loop_ceiling=loop_ceiling,
    )


def mt_capacity(
    node: NodeType,
    mt: MtSpec,
    *,
    gpu_memory_utilization: float = _DEFAULT_UTIL,
) -> MtKnobs:
    """Size the MT serving knobs on a simpler path with no KV term.

    Only weights and a beam x seq_len batch budget matter here: how many source
    sentences fit per batch once the weights are resident, given a per-slot activation
    estimate. This is structurally distinct from ``capacity_for``, which owns the
    continuous-batching KV budget.
    """
    weights_bytes = int(mt.params_total * dtype_bytes(mt.dtype))
    usable = node.vram_gb * _GIB * gpu_memory_utilization
    free_for_batch = max(0.0, usable * (1 - _ACTIVATION_HEADROOM) - weights_bytes)
    # Coarse per-sequence activation slot: beam x seq_len x dtype bytes x a constant.
    per_seq = max(1, mt.beam_size * mt.max_seq_len * dtype_bytes(mt.dtype) * 64)
    batch_size = max(1, int(free_for_batch // per_seq))
    return MtKnobs(
        batch_size=batch_size,
        beam_size=mt.beam_size,
        max_seq_len=mt.max_seq_len,
        weights_bytes=weights_bytes,
    )
