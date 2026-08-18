"""Launcher argv shape: the flags an sbatch submission must carry.

The static-array launcher builds the ``sbatch`` argv the cluster would run, wiring
``--dependency=afterok:<id>`` and the GPU or-constraint, captured through the ``sbatch_spy``
so that no real cluster is needed.
"""

from __future__ import annotations

import pytest

from ragtime.orchestration.slurm import launcher

pytestmark = pytest.mark.small


def test_submit_argv_wires_afterok_and_gpu_constraint(sbatch_spy) -> None:
    jobid = launcher.submit("worker.sbatch", after=[123], gpu=True)
    assert jobid == 12345  # parsed from the spy's `<id>;cluster` stdout
    assert len(sbatch_spy.calls) == 1
    argv = sbatch_spy.calls[0]
    assert argv[0] == "sbatch"
    assert "--dependency=afterok:123" in argv
    assert any(a.startswith("--constraint=") for a in argv)
    assert "--gres=gpu:1" in argv
    assert argv[-1] == "worker.sbatch"


def test_submit_array_sets_array_range(sbatch_spy) -> None:
    launcher.submit_array("worker.sbatch", 5, gpu=False)
    argv = sbatch_spy.calls[0]
    assert "--array=0-4" in argv
    assert not any(a.startswith("--constraint=") for a in argv)  # gpu=False -> no constraint


def test_multi_parent_afterok_chain(sbatch_spy) -> None:
    launcher.submit("w.sbatch", after=[10, 20], gpu=True)
    argv = sbatch_spy.calls[0]
    assert "--dependency=afterok:10:20" in argv


def test_after_any_wires_afterany_for_a_drain_then_merge_node(sbatch_spy) -> None:
    """``afterany`` is the edge for a node whose parent is an N-task ARRAY it only needs
    to have FINISHED: one non-zero task would make ``afterok`` unsatisfiable forever."""
    launcher.submit("drive.sbatch", after_any=[99], gpu=False)
    argv = sbatch_spy.calls[0]
    assert "--dependency=afterany:99" in argv
    assert not any(a.startswith("--dependency=afterok") for a in argv)

    launcher.submit_array("w.sbatch", 3, after_any=[7, 8], gpu=False)
    assert "--dependency=afterany:7:8" in sbatch_spy.calls[1]


def test_afterok_and_afterany_cannot_both_be_asked_for(sbatch_spy) -> None:
    with pytest.raises(ValueError, match="not both"):
        launcher.submit("w.sbatch", after=[1], after_any=[2])
    assert sbatch_spy.calls == []  # rejected before anything was submitted


def test_the_gpu_or_set_carries_every_calibrated_card_and_excludes_the_3090() -> None:
    """The OR-set is a hardware-acceptance statement: a card is in it only if its measured
    ``max_batch_tokens`` clears the semantic ``bucket_token_budget`` of 16 384. The 3090's
    9 216 does not, so CT2 would re-split every bucket there and the same shard would stop
    composing identically across cards."""
    constraint = next(
        a for a in launcher.gpu_constraint_args() if a.startswith("--constraint=")
    ).split("=", 1)[1]
    cards = set(constraint.split("|"))
    assert cards == {
        "nvidia_a100_80gb_pcie",
        "nvidia_h100_nvl",
        "nvidia_h200_nvl",
        "nvidia_rtx_pro_6000_blackwell",
        "nvidia_geforce_rtx_5090",
    }
    assert "nvidia_geforce_rtx_3090" not in cards
