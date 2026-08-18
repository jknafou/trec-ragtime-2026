"""Serving layer: the single ``config -> client`` registry and the node lifecycle.

Every model access in the system flows through the interfaces re-exported here; no
stage loads weights or opens a port on its own. Callers write
``from ragtime.serving import build_clients, guided_json, compile_schemas,
discover_nodes, capacity_for``.

One client per model identity, and the three retrieval legs are three of them: ``encoders``
serves the dense leg (``BAAI/bge-m3``), ``sparse_milco`` the learned-sparse leg
(``omai-research/milco-650m``) and ``late_interaction`` the late-interaction leg
(``hltcoe/plaidx-large-...``). They are three separate checkpoints rather than three outputs of
one model, and ``reranker`` is a fourth, the cross-encoder that rescores the fused head.

Heavy libraries (torch, vllm, transformers, sentence-transformers, wtpsplit, ctranslate2) are
imported inside the backend that needs them rather than at module load, so ``import
ragtime.serving`` together with ``build_stub_clients``, ``compile_schemas``, ``capacity_for``
and ``discover_nodes`` all work on a machine that has only ``jsonschema``, ``openai`` and
``ruamel`` installed.
"""

from __future__ import annotations

from .batching import Batcher, Tier
from .capacity import MtKnobs, ServingKnobs, capacity_for, derive_loop_ceiling, mt_capacity
from .late_interaction import MtdEncoder
from .llm import GenCtx, GuidedJsonError, LlmClient, guided_json
from .modelspec import ModelSpec, MtSpec, load_model_spec, load_mt_spec, load_specs
from .node.discovery import NodeType, discover_nodes, read_profile, write_profile
from .registry import ClientBundle, build_clients, build_stub_clients
from .schemas import CompiledSchema, SchemaSet, compile_schemas
from .sparse_milco import MilcoEncoder, seismic_components

__all__ = [  # noqa: RUF022 - grouped by concern rather than sorted
    # registry (the sole factory) and emission discipline
    "build_clients",
    "build_stub_clients",
    "ClientBundle",
    "guided_json",
    "GenCtx",
    "GuidedJsonError",
    "LlmClient",
    "compile_schemas",
    "SchemaSet",
    "CompiledSchema",
    # capacity and model specs
    "capacity_for",
    "mt_capacity",
    "derive_loop_ceiling",
    "ServingKnobs",
    "MtKnobs",
    "ModelSpec",
    "MtSpec",
    "load_specs",
    "load_model_spec",
    "load_mt_spec",
    # discovery
    "discover_nodes",
    "NodeType",
    "read_profile",
    "write_profile",
    # batching
    "Batcher",
    "Tier",
    # index-leg encoders: the sparse and late-interaction clients
    "MilcoEncoder",
    "MtdEncoder",
    "seismic_components",
]
