"""Full gate: a --local two-node DAG executes in order, and resume is a no-op.

Invariant 7 (the artifact tree is the checkpoint) end-to-end with no SLURM/GPU/122B
dependency: ``plan.run_local`` runs a trivial two-node afterok DAG in this process, and a
second identical run recomputes nothing (counters frozen, artifact mtimes unchanged). The
full-data gate is genuinely cluster-independent.
"""

from __future__ import annotations

import pytest

from ragtime.common.io import success_marker
from ragtime.orchestration import plan
from ragtime.orchestration.plan import node_artifact

pytestmark = pytest.mark.full


def test_local_dag_executes_in_order_then_resumes_as_noop(two_stage_stub_dag, sbatch_spy) -> None:
    stub = two_stage_stub_dag

    # --- first run: both stages execute, in afterok order ---
    executed = plan.run_local(stub.dag, stub.runner, root=stub.root)
    assert executed == ["stage_a", "stage_b"]  # stage_a BEFORE stage_b (afterok chain)
    assert stub.counters == {"stage_a": 1, "stage_b": 1}

    artifacts = {n.name: node_artifact(stub.root, stub.dag, n) for n in stub.dag.nodes}
    for name, art in artifacts.items():
        assert art.exists(), name
        assert success_marker(art).exists(), name  # _SUCCESS checkpoint present

    mtimes = {name: art.stat().st_mtime_ns for name, art in artifacts.items()}

    # --- second identical run: a full no-op (resume) ---
    executed_again = plan.run_local(stub.dag, stub.runner, root=stub.root)
    assert executed_again == []  # nothing recomputed
    assert stub.counters == {"stage_a": 1, "stage_b": 1}  # runner NOT re-invoked
    for name, art in artifacts.items():
        assert art.stat().st_mtime_ns == mtimes[name], name  # bytes/mtime untouched

    # --local never shells out to SLURM.
    assert sbatch_spy.calls == []


def test_every_node_reads_as_done_after_a_full_run(two_stage_stub_dag) -> None:
    """A re-launch finds no work: ``already_done`` short-circuits every node."""
    stub = two_stage_stub_dag
    plan.run_local(stub.dag, stub.runner, root=stub.root)

    # build_plan re-derives the DAG from a real config, so drive already_done directly
    # over the stub DAG to prove the checkpoint short-circuits every node.
    missing = [n for n in stub.dag.nodes if not plan.already_done(n, stub.root, stub.dag)]
    assert missing == []
