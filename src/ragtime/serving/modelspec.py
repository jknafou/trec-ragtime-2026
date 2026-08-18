"""Frozen model-spec records loaded from ``config/serving/models.yml``.

The capacity calculator needs a handful of integer dimensions per model (parameter,
layer and head counts, dtype). Reading them from a committed spec table means the
calculator never imports ``transformers`` just to read a ``config.json``.

The table is read with a ruamel.yaml safe loader, the one YAML library used across
the project. Serving instantiates its own ``YAML(typ="safe")`` rather than importing
the private instance inside ``ragtime.config.loader``, which exposes no public loader
object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

__all__ = [
    "ModelSpec",
    "MtSpec",
    "default_models_path",
    "load_model_spec",
    "load_mt_spec",
    "load_specs",
]

_YAML = YAML(typ="safe")  # YAML 1.2, raises on duplicate keys


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Serving dimensions for one LLM.

    ``params_total`` (all experts resident) drives the weight footprint,
    ``params_active`` affects throughput only, and the KV term sums over
    ``n_full_attn_layers`` (equal to ``n_layers`` for a dense model).

    ``kv_headroom_derate`` is the empirical, model-specific fraction of the
    post-weights VRAM budget that becomes usable KV cache; the remainder is
    activations, attention-kernel workspace, a multimodal tower and CUDA/NCCL
    buffers. It is around 0.20 for a large hybrid MoE and around 1.0 for a lean
    dense model, and it keeps the pre-flight estimate conservative. vLLM's startup
    profiler remains the arbiter.
    """

    name: str
    dtype: str
    params_total: int
    params_active: int
    n_layers: int
    n_full_attn_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    context_window: int
    kv_dtype: str = "bf16"
    kv_headroom_derate: float = 1.0


@dataclass(frozen=True, slots=True)
class MtSpec:
    """Serving dimensions for one MT model: weights plus a beam x seq_len batch
    budget, with no continuous-batching KV term."""

    name: str
    dtype: str
    params_total: int
    beam_size: int
    max_seq_len: int


def default_models_path() -> Path:
    """Path to the repository's ``config/serving/models.yml``."""
    return Path(__file__).resolve().parents[3] / "config" / "serving" / "models.yml"


def _load_raw(path: str | Path | None) -> dict[str, Any]:
    p = Path(path) if path is not None else default_models_path()
    return _YAML.load(p.read_text(encoding="utf-8")) or {}


def load_specs(
    path: str | Path | None = None,
) -> tuple[dict[str, ModelSpec], dict[str, MtSpec]]:
    """Parse the spec table into ``({llm name: ModelSpec}, {mt name: MtSpec})``."""
    raw = _load_raw(path)
    llm: dict[str, ModelSpec] = {}
    for name, d in (raw.get("llm") or {}).items():
        llm[name] = ModelSpec(
            name=name,
            dtype=d["dtype"],
            params_total=int(d["params_total"]),
            params_active=int(d["params_active"]),
            n_layers=int(d["n_layers"]),
            n_full_attn_layers=int(d["n_full_attn_layers"]),
            n_heads=int(d["n_heads"]),
            n_kv_heads=int(d["n_kv_heads"]),
            head_dim=int(d["head_dim"]),
            hidden_size=int(d["hidden_size"]),
            context_window=int(d["context_window"]),
            kv_dtype=str(d.get("kv_dtype", "bf16")),
            kv_headroom_derate=float(d.get("kv_headroom_derate", 1.0)),
        )
    mt: dict[str, MtSpec] = {}
    for name, d in (raw.get("mt") or {}).items():
        mt[name] = MtSpec(
            name=name,
            dtype=d["dtype"],
            params_total=int(d["params_total"]),
            beam_size=int(d["beam_size"]),
            max_seq_len=int(d["max_seq_len"]),
        )
    return llm, mt


def load_model_spec(name: str, path: str | Path | None = None) -> ModelSpec:
    """Look up one LLM spec by model id."""
    llm, _ = load_specs(path)
    if name not in llm:
        raise KeyError(f"no LLM spec for {name!r} in {path or default_models_path()}")
    return llm[name]


def load_mt_spec(name: str, path: str | Path | None = None) -> MtSpec:
    """Look up one MT spec by model id."""
    _, mt = load_specs(path)
    if name not in mt:
        raise KeyError(f"no MT spec for {name!r} in {path or default_models_path()}")
    return mt[name]
