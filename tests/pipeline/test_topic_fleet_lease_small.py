"""The fleet worker's side of the per-pair lease: which pair, which lane, and refusing to share.

`serving.vllm_registry` proves the exclusion mechanism (`tests/serving/test_vllm_registry_small.py`).
This module proves the policy the fleet applies on top of it: the two decisions a worker makes
before it will start, either of which corrupts a run silently if it goes wrong.

  * Which checkpoint. A pair serving a different model answers perfectly and is the wrong system.
  * Which GPU architecture. The A100 runs the Marlin FP8 kernel and Blackwell runs Triton, and
    seeded generation is not reproducible across them: four of four reproducibility tests failed
    on 2xA100 and passed on 2xBlackwell at one code fingerprint. Half a run family generated on
    the wrong kernel puts a kernel term inside the translation delta, where it cannot be told
    apart from the effect being measured.

Not covered here:
  - `bringup` end to end. It builds real clients and a retrieval context, and the fleet's
    production-scale unit covers that; what is covered here is the lease decision it delegates.
"""

from __future__ import annotations

import pytest

from ragtime.pipeline.topic_fleet import ENDPOINT_ENV, LEASE_HEARTBEAT_S, TopicCellAdapter
from ragtime.serving.vllm_registry import DEFAULT_MAX_AGE_S, publish_endpoint

_BW = "nvidia_rtx_pro_6000_blackwell"
_A100 = "nvidia_a100_80gb_pcie"


class _Cfg:
    def __init__(self, model: str | None) -> None:
        self.blocks = {"llm": {"model": model}} if model is not None else {}


def _publish(base, name, gpu):
    publish_endpoint(
        base, name=name, url=f"http://{name}:8000/v1", job_id=f"j-{name}",
        pair=f"{name}:0,1", model="Qwen/Qwen3.5-122B-A10B-FP8", gpu_model=gpu,
    )


# --------------------------------------------------------------------------- #
# The lane: one GPU architecture per run family
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_a_HOMOGENEOUS_registry_needs_no_declaration(tmp_path, monkeypatch) -> None:
    """If every live pair is Blackwell, that is the lane and there is nothing to choose."""
    monkeypatch.delenv("RAGTIME_VLLM_GPU_MODEL", raising=False)
    _publish(tmp_path, "b0", _BW)
    _publish(tmp_path, "b1", _BW)
    assert TopicCellAdapter._lane(tmp_path) == _BW


@pytest.mark.small
def test_a_MIXED_registry_with_no_declared_lane_REFUSES(tmp_path, monkeypatch) -> None:
    """The case that must never be resolved silently.

    Picking either architecture here is a coin flip that lands inside the experiment: some cells
    on Marlin, some on Triton, one run family, no record of which. Raising makes it the
    operator's decision, which is what it is.
    """
    monkeypatch.delenv("RAGTIME_VLLM_GPU_MODEL", raising=False)
    _publish(tmp_path, "a0", _A100)
    _publish(tmp_path, "b0", _BW)
    with pytest.raises(RuntimeError, match="more than one GPU architecture"):
        TopicCellAdapter._lane(tmp_path)


@pytest.mark.small
def test_an_EXPLICIT_lane_wins_over_a_mixed_registry(tmp_path, monkeypatch) -> None:
    """Declaring the lane resolves the refusal above, and holds with both architectures live."""
    monkeypatch.setenv("RAGTIME_VLLM_GPU_MODEL", _BW)
    _publish(tmp_path, "a0", _A100)
    _publish(tmp_path, "b0", _BW)
    assert TopicCellAdapter._lane(tmp_path) == _BW


@pytest.mark.small
def test_an_EMPTY_registry_declares_no_lane_rather_than_inventing_one(tmp_path, monkeypatch) -> None:
    """`None` means no constraint, which is right when there is nothing to constrain. The lease
    that follows returns None and `bringup` refuses on that, reporting an empty registry rather
    than a fabricated architecture mismatch."""
    monkeypatch.delenv("RAGTIME_VLLM_GPU_MODEL", raising=False)
    assert TopicCellAdapter._lane(tmp_path) is None


# --------------------------------------------------------------------------- #
# The checkpoint
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_the_model_filter_comes_from_the_CONFIG_not_a_constant() -> None:
    """The config is the run record. A hardcoded model id here would let a run whose config
    names one checkpoint be served by another, with nothing recording it."""
    assert TopicCellAdapter._llm_model(_Cfg("Qwen/Qwen3.5-122B-A10B-FP8")) == (
        "Qwen/Qwen3.5-122B-A10B-FP8"
    )
    assert TopicCellAdapter._llm_model(_Cfg(None)) is None


# --------------------------------------------------------------------------- #
# Refusal
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_a_worker_with_NO_free_pair_RAISES_instead_of_sharing(tmp_path, monkeypatch) -> None:
    """A worker with no free pair refuses rather than sharing one.

    Falling back to a single shared endpoint file is the obvious shortcut and is the defect: one
    file with many readers puts every worker on one pair instead of fanning them out, and that
    co-tenancy breaks seeded reproducibility.
    """
    monkeypatch.delenv(ENDPOINT_ENV, raising=False)
    monkeypatch.delenv("RAGTIME_VLLM_GPU_MODEL", raising=False)
    monkeypatch.setenv("RAGTIME_ARTIFACT_ROOT", str(tmp_path))
    adapter = TopicCellAdapter(registry="unused", variant="original", seeds=(0,))
    with pytest.raises(RuntimeError, match="no free vLLM pair"):
        adapter._lease_a_pair(_Cfg("Qwen/Qwen3.5-122B-A10B-FP8"))


# --------------------------------------------------------------------------- #
# The beat must be faster than the reaper
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_the_lease_beat_is_comfortably_INSIDE_the_staleness_bound() -> None:
    """An arithmetic tripwire, not a style check.

    If the beat interval drifts above the staleness bound, every healthy worker's lease is
    reclaimed mid-topic and its pair handed to a second worker, so the registry manufactures the
    co-tenancy it exists to prevent, and it looks like a scheduling hiccup rather than a
    reproducibility failure. The 3x margin tolerates one missed beat on a loaded node.
    """
    assert LEASE_HEARTBEAT_S * 3 <= DEFAULT_MAX_AGE_S, (
        f"beat {LEASE_HEARTBEAT_S}s vs staleness {DEFAULT_MAX_AGE_S}s leaves no room for a "
        "missed heartbeat"
    )
