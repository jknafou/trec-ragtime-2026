"""build_plan JobDAG shape over the six real configs.

Proves the documented ``afterok`` chain (corpus-preprocess -> pipeline ->
citation_scoring -> select_serialize -> monitoring), exactly one family-shared corpus node
per family (never one-per-config), ``run_id <= 25``, deterministic re-planning, and that
``--dry-run`` never shells out to ``sbatch``.

Shape only: the plan is checked as a data structure, without submitting anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragtime.config import load
from ragtime.orchestration import build_plan, cli, run_id
from ragtime.orchestration.plan import (
    CITATION_SCORING,
    CORPUS,
    MONITORING,
    PIPELINE,
    SELECT_SERIALIZE,
)

pytestmark = pytest.mark.small

_EXPECTED_EDGES = (
    (CORPUS, PIPELINE),
    (PIPELINE, CITATION_SCORING),
    (CITATION_SCORING, SELECT_SERIALIZE),
    (SELECT_SERIALIZE, MONITORING),
)


def test_afterok_chain_is_the_documented_shape(real_config_paths: list[Path]) -> None:
    """Every run's DAG wires the exact corpus -> pipeline -> citation_scoring -> select ->
    monitor afterok chain."""
    for p in real_config_paths:
        dag = build_plan(load(p))
        assert dag.edges == _EXPECTED_EDGES
        # afterok direction: each child names its parent.
        assert dag.node(PIPELINE).after == (CORPUS,)
        assert dag.node(CITATION_SCORING).after == (PIPELINE,)
        assert dag.node(SELECT_SERIALIZE).after == (CITATION_SCORING,)
        assert dag.node(MONITORING).after == (SELECT_SERIALIZE,)


def test_one_shared_corpus_node_per_family_not_per_config(real_e2e_paths: list[Path]) -> None:
    """The 3 controlled e2e members resolve to the same corpus node key (never three)."""
    # Every e2e config in this repository is a controlled member, so there is nothing to
    # filter out before comparing corpus node keys.
    corpus_keys = {build_plan(load(p)).node(CORPUS).key for p in real_e2e_paths}
    assert len(corpus_keys) == 1  # one byte-identical corpus for the whole family
    # And it is flagged family-shared so node_artifact routes it under the family root.
    for p in real_e2e_paths:
        assert build_plan(load(p)).node(CORPUS).family_shared is True


def test_corpus_node_array_size_is_config_derived_worker_count(real_config_paths: list[Path]) -> None:
    """The CORPUS worker-array width tracks the seeded shard count via config, not hardcoded 1."""
    from ragtime.orchestration import plan

    for p in real_config_paths:
        cfg = load(p)
        c = cfg.blocks["execution"]  # execution knobs live outside the shared chunker block
        import math

        expected = max(1, math.ceil(int(c["corpus_shards"]) / int(c["oversubscription"])))
        node = build_plan(cfg).node(CORPUS)
        assert node.array_size == expected == plan.corpus_workers(cfg)
        assert node.array_size > 1  # a real parallel array, not a serial job


def test_run_id_within_submission_ceiling(real_config_paths: list[Path]) -> None:
    """run_id fits the <=25-char TREC submission-filename ceiling for all 6 configs."""
    for p in real_config_paths:
        assert len(run_id(load(p))) <= 25


def test_build_plan_is_deterministic(real_config_paths: list[Path]) -> None:
    """Re-planning the same config yields a value-equal DAG and byte-equal render (no set leak)."""
    for p in real_config_paths:
        cfg = load(p)
        a, b = build_plan(cfg), build_plan(cfg)
        assert a == b
        assert a.render() == b.render()


def test_dry_run_over_six_configs_never_submits(real_config_paths: list[Path], sbatch_spy) -> None:
    """--dry-run prints the DAG and exits 0 for each config; no sbatch is ever fired."""
    for p in real_config_paths:
        assert cli.main(["--config", str(p), "--dry-run"]) == 0
    assert sbatch_spy.calls == []
