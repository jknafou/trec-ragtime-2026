"""The capacity calculator, as pure functions: MoE params_total, hybrid attention-only KV,
GQA tensor-parallel divisibility, the 24 GB card that forces TP>=2 against the 80 GB one that
does not, the regression against numbers measured in an earlier bring-up experiment, the
separate MT path, and the loop-ceiling contract (J2)."""

from __future__ import annotations

import pytest

from ragtime.serving.capacity import (
    MtKnobs,
    ServingKnobs,
    capacity_for,
    derive_loop_ceiling,
    dtype_bytes,
    mt_capacity,
)

pytestmark = pytest.mark.small


# D. Capacity calculator -------------------------------------------------------
def test_capacity_moe_uses_params_total_not_active(qwen35_122b_a10b_fp8_spec, node_96gb_blackwell):
    knobs = capacity_for(node_96gb_blackwell, qwen35_122b_a10b_fp8_spec)
    expected = qwen35_122b_a10b_fp8_spec.params_total * dtype_bytes("fp8")
    assert knobs.weights_bytes == expected  # 122e9 bytes, not the ~10e9 active figure
    assert knobs.weights_bytes != qwen35_122b_a10b_fp8_spec.params_active * dtype_bytes("fp8")


def test_capacity_hybrid_kv_sums_only_attention_layers(
    qwen35_122b_a10b_fp8_spec, node_96gb_blackwell
):
    spec = qwen35_122b_a10b_fp8_spec
    knobs = capacity_for(node_96gb_blackwell, spec)
    attn_only = 2 * spec.n_full_attn_layers * spec.n_kv_heads * spec.head_dim * dtype_bytes(
        spec.kv_dtype
    )
    naive_all = 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * dtype_bytes(spec.kv_dtype)
    assert knobs.kv_bytes_per_token == attn_only
    assert knobs.kv_bytes_per_token < naive_all


def test_capacity_gqa_tp_divides_heads_and_kv_heads(qwen3_32b_fp8_spec, node_24gb_4gpu):
    knobs = capacity_for(node_24gb_4gpu, qwen3_32b_fp8_spec)
    tp = knobs.tensor_parallel_size
    assert qwen3_32b_fp8_spec.n_heads % tp == 0
    assert qwen3_32b_fp8_spec.n_kv_heads % tp == 0  # never a heads-only divisor


def test_capacity_24gb_card_forces_tp_ge_2(qwen3_32b_fp8_spec, node_24gb_4gpu):
    knobs = capacity_for(node_24gb_4gpu, qwen3_32b_fp8_spec)
    assert knobs.tensor_parallel_size >= 2  # FP8 32B weights exceed one 24 GB card


def test_capacity_80gb_card_single_gpu(qwen3_32b_fp8_spec, node_80gb_8gpu):
    knobs = capacity_for(node_80gb_8gpu, qwen3_32b_fp8_spec)
    assert knobs.tensor_parallel_size == 1  # weights + KV + headroom fit one card


def test_capacity_regression_matches_proven_spike_numbers(qwen3_32b_fp8_spec, node_96gb_blackwell):
    knobs = capacity_for(
        node_96gb_blackwell, qwen3_32b_fp8_spec, max_model_len=8192, gpu_memory_utilization=0.90
    )
    # Measured in an earlier bring-up experiment: 209,536 KV tokens, 25.58x concurrency.
    assert abs(knobs.kv_cache_tokens - 209_536) / 209_536 < 0.15
    assert abs(knobs.max_concurrency - 25.58) / 25.58 < 0.15


def test_mt_capacity_separate_fn_no_kv_term(nllb_200_3_3b_mt_spec, node_96gb_blackwell):
    knobs = mt_capacity(node_96gb_blackwell, nllb_200_3_3b_mt_spec)
    assert isinstance(knobs, MtKnobs)
    assert knobs.weights_bytes == nllb_200_3_3b_mt_spec.params_total * dtype_bytes("bf16")
    assert knobs.batch_size >= 1 and knobs.beam_size == 4 and knobs.max_seq_len == 512
    # structurally distinct from the LLM path: no continuous-batching KV term.
    assert not hasattr(knobs, "kv_cache_tokens")


def test_capacity_for_pure_deterministic(qwen3_32b_fp8_spec, node_96gb_blackwell):
    a = capacity_for(node_96gb_blackwell, qwen3_32b_fp8_spec)
    b = capacity_for(node_96gb_blackwell, qwen3_32b_fp8_spec)
    assert a == b and isinstance(a, ServingKnobs)  # frozen dataclass value-equality


# J2. loop-ceiling contract ----------------------------------------------------
def test_loop_ceiling_is_positive_int():
    val = derive_loop_ceiling(max_num_seqs=25, kv_cache_tokens=209_536, tokens_per_loop=8192)
    assert isinstance(val, int) and val >= 1


def test_loop_ceiling_bounded_by_kv_headroom():
    # kv budget (2 loops) tighter than max_num_seqs (10) -> ceiling is 2.
    assert derive_loop_ceiling(10, 16_384, 8192) == 2
    # always >= 1 even when the kv budget rounds to 0.
    assert derive_loop_ceiling(10, 100, 8192) == 1
