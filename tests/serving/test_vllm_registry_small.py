"""The per-pair vLLM registry: a pair is leased to exactly one worker, or the fan is fake.

The property under test is exclusion. A single shared endpoint file is one file with many
readers, so every worker addresses one instance, and sampled decoding is not batch-invariant:
the same seed produced a different round-0 nugget bank when a second client shared the
instance, and the same bank twice when it did not. A registry that hands one endpoint to two
workers therefore does not merely under-use the fleet, it destroys the reproducibility the
fairness invariant rests on.

The mechanism is filesystem renames under a `tmp_path`, so a real vLLM would add nothing but
its bring-up time.

Not covered here:
  - that a published URL actually serves, which needs a live vLLM and is proven by the fleet
    scale unit.
  - beegfs rename semantics under real cross-node contention, proven by the OMT translation
    fan, which runs this same `workqueue.claim` at 400 workers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ragtime.serving.vllm_registry import (
    DEFAULT_MAX_AGE_S,
    beat,
    claim_endpoint,
    endpoint_registry,
    live_endpoints,
    publish_endpoint,
    release_endpoint,
    unpublish_endpoint,
)

_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
_BW = "nvidia_rtx_pro_6000_blackwell"
_A100 = "nvidia_a100_80gb_pcie"


def _publish(base: Path, name: str, *, gpu: str = _BW, model: str = _MODEL) -> Path:
    return publish_endpoint(
        base, name=name, url=f"http://node-{name}:8000/v1", job_id=f"job-{name}",
        pair=f"{name}:0,1", model=model, gpu_model=gpu,
    )


# --------------------------------------------------------------------------- #
# Publish / discover
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_a_published_endpoint_is_discoverable_and_carries_its_provenance(tmp_path: Path) -> None:
    """`job_id`/`pair` are not decoration: the fleet provenance records where a cell generated."""
    _publish(tmp_path, "bw0")
    live = live_endpoints(tmp_path)
    assert len(live) == 1
    assert live[0]["url"] == "http://node-bw0:8000/v1"
    assert live[0]["job_id"] == "job-bw0"
    assert live[0]["pair"] == "bw0:0,1"


@pytest.mark.small
def test_an_unpublished_endpoint_is_gone_from_both_directories(tmp_path: Path) -> None:
    """A descriptor that outlives its job is how a worker blocks forever on a corpse."""
    _publish(tmp_path, "bw0")
    lease = claim_endpoint(tmp_path)
    assert lease is not None
    unpublish_endpoint(tmp_path, "bw0")
    wq = endpoint_registry(tmp_path)
    assert list(Path(wq.pending).glob("*.json")) == []
    assert list(Path(wq.running).glob("*.json")) == []
    assert live_endpoints(tmp_path) == []


# --------------------------------------------------------------------------- #
# the property: exclusion
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_ONE_endpoint_TWO_claimers_yields_exactly_ONE_lease(tmp_path: Path) -> None:
    """The reason this is a claim and not a read.

    Two workers, one free pair: exactly one gets it, the other is told there is nothing. If
    both got it they would share one instance, and their seeds would stop reproducing.
    """
    _publish(tmp_path, "bw0")
    first = claim_endpoint(tmp_path)
    second = claim_endpoint(tmp_path)
    assert first is not None, "the first claimer must get the free pair"
    assert second is None, "the second claimer must NOT be handed a pair already in use"


@pytest.mark.small
def test_a_leased_endpoint_is_not_offered_as_live(tmp_path: Path) -> None:
    """`live_endpoints` means free. A leased pair listed as available is the same bug one layer up."""
    _publish(tmp_path, "bw0")
    claim_endpoint(tmp_path)
    assert live_endpoints(tmp_path) == []


@pytest.mark.small
def test_release_makes_the_pair_claimable_again(tmp_path: Path) -> None:
    """A topic finishes and the pair goes back on the shelf: that is what makes a fan a fan."""
    _publish(tmp_path, "bw0")
    lease = claim_endpoint(tmp_path)
    assert lease is not None
    release_endpoint(tmp_path, lease)
    again = claim_endpoint(tmp_path)
    assert again is not None
    assert again.url == lease.url


@pytest.mark.small
def test_N_claimers_over_M_endpoints_partition_them_with_no_duplicates(tmp_path: Path) -> None:
    """The fan property stated over more than two: 3 pairs, 5 workers, no URL served twice."""
    for i in range(3):
        _publish(tmp_path, f"bw{i}")
    leases = [claim_endpoint(tmp_path) for _ in range(5)]
    got = [x for x in leases if x is not None]
    assert len(got) == 3, "every free pair must be handed out"
    assert len({x.url for x in got}) == 3, "and no pair handed out twice"
    assert leases.count(None) == 2, "the two extra workers must be refused, not doubled up"


# --------------------------------------------------------------------------- #
# Lane filtering: one GPU architecture per run family
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_a_wrong_ARCHITECTURE_endpoint_is_refused_and_LEFT_FREE(tmp_path: Path) -> None:
    """The A100 runs a Marlin FP8 kernel and Blackwell runs a Triton one, so seeded
    generation does not carry across: 4/4 reproducibility tests failed on two A100s and
    passed on two Blackwells at one code fingerprint. A family that mixed them would put a
    kernel term inside the translation delta.

    Refusing must also not consume the endpoint: the A100 pair stays available to an A100
    worker.
    """
    _publish(tmp_path, "a100", gpu=_A100)
    assert claim_endpoint(tmp_path, gpu_model=_BW) is None
    assert claim_endpoint(tmp_path, gpu_model=_A100) is not None


@pytest.mark.small
def test_the_wrong_lane_scan_TERMINATES_instead_of_spinning(tmp_path: Path) -> None:
    """Requeueing a rejected descriptor puts it back where `claim` looks first.

    Without a memory of what it has already rejected, the claim loop re-claims the same file
    forever: a Blackwell-only registry would hang an A100 worker in a tight rename loop rather
    than returning None. The bound is what this asserts; the timeout is the tripwire.
    """
    for i in range(4):
        _publish(tmp_path, f"bw{i}", gpu=_BW)
    started = time.perf_counter()
    assert claim_endpoint(tmp_path, gpu_model=_A100) is None
    assert time.perf_counter() - started < 5.0, "the scan must terminate, not spin"


@pytest.mark.small
def test_a_wrong_MODEL_endpoint_is_refused(tmp_path: Path) -> None:
    """Same class of refusal as retrieval's (rendering, index_hash) filter: the answer would look
    entirely fine and come from a different system."""
    _publish(tmp_path, "other", model="some/other-model")
    assert claim_endpoint(tmp_path, model=_MODEL) is None


@pytest.mark.small
def test_the_RIGHT_lane_is_picked_out_of_a_MIXED_registry(tmp_path: Path) -> None:
    """A mixed fleet is the realistic case: A100 nodes beside Blackwell ones."""
    _publish(tmp_path, "a0", gpu=_A100)
    _publish(tmp_path, "b0", gpu=_BW)
    lease = claim_endpoint(tmp_path, gpu_model=_BW)
    assert lease is not None
    assert lease.gpu_model == _BW and lease.name == "b0"
    # and the A100 one was left alone for whoever wants it
    assert [d["name"] for d in live_endpoints(tmp_path)] == ["a0"]


# --------------------------------------------------------------------------- #
# Liveness: presence is not liveness
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_a_STALE_free_descriptor_is_not_offered(tmp_path: Path) -> None:
    """A descriptor outlives its job (walltime, preemption, SIGKILL, node crash). Offering one is
    how a worker blocks forever on an endpoint that stopped answering."""
    path = _publish(tmp_path, "bw0")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["heartbeat"] = time.time() - (DEFAULT_MAX_AGE_S + 60)
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert live_endpoints(tmp_path) == []


@pytest.mark.small
def test_beat_REVIVES_a_free_descriptor(tmp_path: Path) -> None:
    """The vLLM job's keep-alive. Without it every endpoint ages out of the fleet in 2 minutes."""
    path = _publish(tmp_path, "bw0")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["heartbeat"] = time.time() - (DEFAULT_MAX_AGE_S + 60)
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert live_endpoints(tmp_path) == []
    assert beat(tmp_path, "bw0") is not None
    assert len(live_endpoints(tmp_path)) == 1


@pytest.mark.small
def test_publishing_over_a_LEASED_name_does_NOT_create_a_free_copy(tmp_path: Path) -> None:
    """The subtle double-booking: a vLLM job that re-publishes on a loop as its keep-alive would
    put a second, FREE copy of an endpoint a worker is actively using back on the shelf. The
    liveness mechanism would then manufacture exactly the co-tenancy the registry exists to
    prevent, which is why `publish` beats an already-leased name instead of re-announcing it."""
    _publish(tmp_path, "bw0")
    lease = claim_endpoint(tmp_path)
    assert lease is not None
    _publish(tmp_path, "bw0")  # the job's keep-alive fires mid-topic
    assert live_endpoints(tmp_path) == [], "a leased pair must not reappear as free"
    assert claim_endpoint(tmp_path) is None


@pytest.mark.small
def test_an_ABANDONED_lease_is_reclaimed_so_the_pair_is_not_lost(tmp_path: Path) -> None:
    """A worker killed mid-topic must not strand an expensive pair forever. This is `reap_stale`,
    reused verbatim: the same reaper that reclaims an OOM-killed translation shard."""
    _publish(tmp_path, "bw0")
    lease = claim_endpoint(tmp_path)
    assert lease is not None
    _age_workqueue_heartbeat(lease.path, DEFAULT_MAX_AGE_S + 60)
    assert claim_endpoint(tmp_path) is not None, "the abandoned pair must come back"


@pytest.mark.small
def test_a_BEATING_lease_is_NOT_reclaimed_mid_topic(tmp_path: Path) -> None:
    """The other half, and the one that matters more: a topic takes tens of minutes against a
    120 s bound, so if the holder's beat did not count, every healthy worker would have its pair
    yanked and handed to a second worker: co-tenancy caused by the liveness check itself."""
    _publish(tmp_path, "bw0")
    lease = claim_endpoint(tmp_path)
    assert lease is not None
    _age_workqueue_heartbeat(lease.path, DEFAULT_MAX_AGE_S + 60)
    beat(tmp_path, "bw0")  # the worker says: still here, still working
    assert claim_endpoint(tmp_path) is None, "a live holder must keep its pair"


def _age_workqueue_heartbeat(shard: Path, seconds: float) -> None:
    """Backdate both liveness signals `reap_stale` consults for a claimed shard.

    It reads the `meta/` sidecar and falls back to the shard's own ctime when that is missing, so
    backdating only the sidecar would leave a fresh ctime and the reaper would (correctly) keep the
    lease: the test would then pass for the wrong reason.
    """
    meta = shard.parent.parent / "meta"
    hb = meta / f"{shard.name}.hb"
    old = time.time() - seconds
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.write_text(f"{old}\n", encoding="utf-8")
    os.utime(hb, (old, old))
    os.utime(shard, (old, old))


# --------------------------------------------------------------------------- #
# What the worker is actually handed
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_the_lease_yields_exactly_the_env_the_worker_refuses_to_start_without(
    tmp_path: Path,
) -> None:
    """`pipeline.topic_fleet.TopicCellAdapter.bringup` refuses without `RAGTIME_VLLM_URL`: that
    refusal is what stops a fan from silently collapsing onto one pair, and this is what satisfies
    it. Asserted as the exact contract, because a renamed key would turn the refusal back on."""
    _publish(tmp_path, "bw0")
    lease = claim_endpoint(tmp_path)
    assert lease is not None
    assert lease.env == {"RAGTIME_VLLM_URL": "http://node-bw0:8000/v1"}
