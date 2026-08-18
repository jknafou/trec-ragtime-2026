"""Determinism and run identity over the shipped configs.

``expand_seeds`` and ``run_id`` read the config rather than a constant and the
``(run_id, variant, seed)`` key is a stable pure function. This is the reproducibility
half of the fairness invariant.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ragtime.config import load
from ragtime.orchestration import cell_key, expand_seeds, run_id

pytestmark = pytest.mark.small


def test_expand_seeds_counts_are_config_driven(
    real_e2e_paths: list[Path], real_mlir_paths: list[Path]
) -> None:
    """`expand_seeds` yields exactly `cfg.seeds` cells for every shipped config.

    Asserted against the field rather than a literal. A hardcoded count goes red on a
    legitimate config change and stays green if `expand_seeds` starts ignoring the field
    while the constant happens to match. The sibling test pins that the field is read at
    all; this one pins agreement with the shipped configs.
    """
    for p in (*real_e2e_paths, *real_mlir_paths):
        cfg = load(p)
        assert len(expand_seeds(cfg)) == cfg.seeds


def test_expand_seeds_reads_the_field_not_a_hardcoded_constant() -> None:
    """A cfg-shaped object with seeds=3 yields three seeds, so the field is read."""
    assert expand_seeds(SimpleNamespace(seeds=3)) == [0, 1, 2]


def test_run_ids_are_bounded_and_pairwise_distinct(real_config_paths: list[Path]) -> None:
    rids = [run_id(load(p)) for p in real_config_paths]
    assert all(len(r) <= 25 for r in rids)
    assert len(set(rids)) == len(rids)  # no two configs collide to one run_id


def test_cell_key_is_a_pure_function() -> None:
    assert cell_key("e2e-omt", "omt", 0) == cell_key("e2e-omt", "omt", 0)
    assert cell_key("e2e-omt", "omt", 0) != cell_key("e2e-omt", "omt", 1)  # seed matters
