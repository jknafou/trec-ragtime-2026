"""The contracts orchestration holds with config, serving and the preprocess stages.

Pins the concrete producer and consumer shapes the other packages rely on: config's
``all_hashes`` on the consumer side, and the ``slurm.workqueue`` claim API the stages
import verbatim, plus the plan's "keep the two SLURM shapes cleanly separate" guard.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ragtime.config import all_hashes, load
from ragtime.orchestration.plan import build_plan
from ragtime.orchestration.slurm import launcher, workqueue

pytestmark = pytest.mark.small

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# all_hashes is consumed as-is: the plan keys its nodes off config's own digests and
# never re-derives a fingerprint of its own.
# --------------------------------------------------------------------------- #
def test_plan_node_keys_are_all_hashes_values_consumed_as_is() -> None:
    cfg = load(_REPO_ROOT / "config" / "e2e-omt.yml")
    hashes = all_hashes(cfg)
    assert isinstance(hashes, dict)
    for node in build_plan(cfg).nodes:
        # Keys are `<prefix>:<digest>` (the tail nodes add a third `:<node>` segment).
        digest = node.key.split(":")[1]
        assert digest in hashes.values(), node.name
        assert len(digest) == 64 and digest == digest.lower()  # not re-wrapped


# --------------------------------------------------------------------------- #
# the workqueue claim API the stages import verbatim stays named and arity-stable.
# --------------------------------------------------------------------------- #
def test_workqueue_public_api_names_and_arity_stable() -> None:
    expected = {
        "claim": ("pending", "running"),
        "heartbeat": ("shard",),
        "reap_stale": ("running", "pending", "max_age"),
        "mark_done": ("shard", "done", "key"),
        "fail": ("shard", "pending", "failed", "k_max"),
    }
    for name, params in expected.items():
        fn = getattr(workqueue, name)
        sig = inspect.signature(fn)
        for p in params:
            assert p in sig.parameters, f"{name} lost parameter {p}"


def test_two_slurm_shapes_stay_separate() -> None:
    """launcher (static array) and workqueue (dynamic claim) never share a call surface."""
    for name in ("claim", "heartbeat", "reap_stale"):
        assert not hasattr(launcher, name)
    for name in ("submit_array", "afterok"):
        assert not hasattr(workqueue, name)
