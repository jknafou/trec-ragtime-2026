"""Shared fixtures for the serving-registry tests.

Everything here is CPU-only: a fake ``openai.AsyncOpenAI``-shaped client that
captures kwargs and returns a configurable canned sequence, synthetic ``ModelSpec``
/ ``NodeType`` records with hand-computed expected knobs, captured ``scontrol
--json`` / ``sinfo`` text (no live SLURM), and known-good/bad payloads per schema.
No fixture imports ``torch``/``vllm``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragtime.serving.capacity import ServingKnobs  # noqa: F401  (re-export sanity)
from ragtime.serving.modelspec import ModelSpec, MtSpec
from ragtime.serving.node.discovery import NodeType

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Fake OpenAI client (captures kwargs; returns a canned content sequence).
# --------------------------------------------------------------------------- #
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, contents):
        self._contents = list(contents)
        self._i = 0
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._contents[min(self._i, len(self._contents) - 1)]
        self._i += 1
        return _Resp(content)


class _Chat:
    def __init__(self, contents):
        self.completions = _Completions(contents)


class StubOpenAI:
    """Minimal async fake: ``client.chat.completions.create(**kwargs)``; every
    kwargs dict is recorded in ``client.chat.completions.calls``."""

    def __init__(self, contents):
        self.chat = _Chat(contents)

    @property
    def calls(self):
        return self.chat.completions.calls


@pytest.fixture
def make_stub_client():
    """Factory: ``make_stub_client([content, ...]) -> StubOpenAI``."""

    def _make(contents):
        return StubOpenAI(contents)

    return _make


# --------------------------------------------------------------------------- #
# Synthetic model specs (dims from the real published config.json where proven).
# --------------------------------------------------------------------------- #
@pytest.fixture
def qwen3_32b_fp8_spec() -> ModelSpec:
    """The spec served in an earlier bring-up experiment: dense GQA, FP8, dimensions from
    Qwen3-32B's published config.json."""
    return ModelSpec(
        name="Qwen/Qwen3-32B-FP8",
        dtype="fp8",
        kv_dtype="bf16",
        params_total=32_800_000_000,
        params_active=32_800_000_000,
        n_layers=64,
        n_full_attn_layers=64,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=5120,
        context_window=32768,
    )


@pytest.fixture
def qwen35_122b_a10b_fp8_spec() -> ModelSpec:
    """The real production MoE/hybrid spec (published config.json, qwen3_5_moe):
    params_total (122B) >> params_active (~10B); 12 of 48 layers are full-attention
    (full_attention_interval:4), the rest linear-attention (no KV); GQA 32/2."""
    return ModelSpec(
        name="Qwen/Qwen3.5-122B-A10B-FP8",
        dtype="fp8",
        kv_dtype="bf16",
        params_total=122_000_000_000,
        params_active=10_000_000_000,
        n_layers=48,
        n_full_attn_layers=12,
        n_heads=32,
        n_kv_heads=2,
        head_dim=256,
        hidden_size=3072,
        context_window=262144,
        kv_headroom_derate=0.20,  # conservative MoE overhead reserve (real datapoint)
    )


@pytest.fixture
def nllb_200_3_3b_mt_spec() -> MtSpec:
    return MtSpec(
        name="facebook/nllb-200-3.3B",
        dtype="bf16",
        params_total=3_300_000_000,
        beam_size=4,
        max_seq_len=512,
    )


# --------------------------------------------------------------------------- #
# Synthetic node types.
# --------------------------------------------------------------------------- #
@pytest.fixture
def node_96gb_blackwell() -> NodeType:
    """Mirrors a real RTX PRO 6000 Blackwell node."""
    return NodeType(
        cluster="bamboo",
        partition="shared-gpu",
        gpu_model="rtx_pro_6000",
        gpu_count=1,
        vram_gb=95.6,  # 97887 MiB / 1024
        compute_capability="12.0",
        driver_version="610.43.02",
        cpu_ram_gb=503.0,
        cpu_count=64,
        node_name="gpu008",
    )


@pytest.fixture
def node_24gb_4gpu() -> NodeType:
    """A 24 GB-class card (forces tensor_parallel_size >= 2 for FP8 32B weights)."""
    return NodeType(
        cluster="baobab",
        partition="shared-gpu",
        gpu_model="titan",
        gpu_count=4,
        vram_gb=24.0,
        compute_capability="7.5",
        driver_version="610.43.02",
        cpu_ram_gb=256.0,
        cpu_count=32,
        node_name="gpu100",
    )


@pytest.fixture
def node_80gb_8gpu() -> NodeType:
    """An 80 GB-class card (fits FP8 32B single-GPU with headroom)."""
    return NodeType(
        cluster="baobab",
        partition="shared-gpu",
        gpu_model="a100",
        gpu_count=8,
        vram_gb=80.0,
        compute_capability="8.0",
        driver_version="610.43.02",
        cpu_ram_gb=1024.0,
        cpu_count=128,
        node_name="gpu200",
    )


# --------------------------------------------------------------------------- #
# Captured scontrol --json / sinfo text (no live SLURM).
# --------------------------------------------------------------------------- #
def _node_json(name, gres, comment, real_memory=515000, cpus=64, partitions="shared-gpu"):
    return {
        "name": name,
        "partitions": partitions,
        "gres": gres,
        "comment": comment,
        "real_memory": real_memory,
        "cpus": cpus,
        "cluster_name": "",  # empty in the payload, stamped from the loop var
    }


@pytest.fixture
def scontrol_multi_gres() -> str:
    """A node whose GRES has repeated/segmented gpu segments (summed to 4)."""
    return json.dumps(
        {
            "nodes": [
                _node_json(
                    "gpu042",
                    "gpu:h100:2(S:0-1),gpu:h100:2(S:2-3)",
                    "VramPerGpu:no_consume:81920M,DriverVersion:610.43.02,ComputeCapability:9.0",
                )
            ]
        }
    )


@pytest.fixture
def scontrol_missing_vram() -> str:
    """A node with no VramPerGpu (forces the model->VRAM fallback table)."""
    return json.dumps(
        {
            "nodes": [
                _node_json(
                    "gpu050",
                    "gpu:a100:1(S:0)",
                    "DriverVersion:610.43.02,ComputeCapability:8.0",
                )
            ]
        }
    )


@pytest.fixture
def scontrol_blackwell() -> str:
    """A Blackwell node carrying full VRAM, driver and compute-capability metadata."""
    return json.dumps(
        {
            "nodes": [
                _node_json(
                    "gpu008",
                    "gpu:rtx_pro_6000:1(S:0)",
                    "VramPerGpu:no_consume:97887M,DriverVersion:610.43.02,ComputeCapability:12.0",
                )
            ]
        }
    )


@pytest.fixture
def sinfo_plus_aggregated() -> str:
    """A `+`-aggregated sinfo row (collapses several node shapes -> lossy count)."""
    return "gpu[001-008] gpu:rtx_pro_6000:1+ 515000 64\n"


# --------------------------------------------------------------------------- #
# Known-good / known-bad payloads per compiled schema.
# --------------------------------------------------------------------------- #
@pytest.fixture
def good_payloads() -> dict[str, dict]:
    return {
        "action": {"rationale": "need evidence", "action": "search", "query": "floods", "lang": "ru"},
        # `weight` is required here. It was absent from `_NUGGETS_SCHEMA` while `prompts.py`
        # asked the model for it, so under `additionalProperties: false` the key was
        # unemittable and every round-0 nugget shipped `weight: 0.0`, which sorted the report
        # request's own primary facets dead last in the Task 3 budget fit.
        "nuggets": {
            "rationale": "facets",
            "nuggets": [{"nugget_id": None, "question": "What happened?", "weight": 0.9}],
        },
        # The round loop's audit delta: the three-way coverage mark, gap detection and the
        # prune list in one object. There is no `saturated` key, because the loop stops on
        # measured novelty rather than on the judge's own say-so.
        "coverage_audit": {
            "rationale": "gaps remain",
            "coverage": [{"nugget_id": "2061#n0", "coverage": "partial"}],
            "add": [{"question": "x?", "trigger_passage_id": None, "weight": 0.5}],
            "prune": [],
        },
        "aggregator": {"rationale": "picked", "selected": ["c1", "c2"]},
        "dedup": {
            "duplicate": True,
            "paraphrase_match": True,
            "entity_match": True,
            "reason": "same fact",
        },
        # The admission gate decompose applies to a candidate nugget; it lives in `serving`
        # so one compiled schema serves both callers.
        "on_topic": {"rationale": "it asks about the request's subject", "on_topic": True},
        # The citation scorer's claim-importance judge -- question and claim sentence only,
        # no passage. Its enum is the {full, partial, none}
        # vocabulary the coverage audit already uses, so there is one notion of how much of a
        # nugget is answered; `importance.COVERAGE_TO_IMPORTANCE` maps it to {1.0, 0.5, 0.0}.
        "claim_importance": {"rationale": "it states the figure the question asks for",
                             "coverage": "full"},
    }


@pytest.fixture
def bad_payloads() -> dict[str, dict]:
    return {
        # missing required `action`
        "action": {"rationale": "no action key"},
        # nuggets not an array
        "nuggets": {"rationale": "x", "nuggets": "not-a-list"},
        # A coverage label outside the three-way enum. "mostly" must be rejected by the
        # grammar rather than coerced by the caller, or an over-claiming judge closes a nugget.
        "coverage_audit": {
            "rationale": "x",
            "coverage": [{"nugget_id": "2061#n0", "coverage": "mostly"}],
            "add": [],
            "prune": [],
        },
        # missing required `selected`
        "aggregator": {"rationale": "x"},
        # missing required `reason` (nullable-not-absent -> omission is invalid)
        "dedup": {"duplicate": True, "paraphrase_match": True, "entity_match": False},
        # `on_topic` with the wrong type. The gate is binary, so a model that answers "maybe"
        # must be rejected by the grammar rather than coerced by the caller.
        "on_topic": {"rationale": "x", "on_topic": "maybe"},
        # A coverage value outside the enum. `importance.claim_importance` raises rather than
        # defaulting here, so an unapplied grammar surfaces instead of inventing a score; this
        # pins the layer that rejects it first.
        "claim_importance": {"rationale": "x", "coverage": "mostly"},
    }
