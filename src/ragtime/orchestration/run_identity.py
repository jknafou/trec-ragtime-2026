"""Run identity: the ``run_id``, its family grouping and the per-cell artifact key.

How a config becomes an identity: the ``run_id`` (at most 25 characters, the length
TREC submission filenames must respect), the family an ``e2e-*`` or ``mlir-*`` run
belongs to, the translation ``variant`` it moves, and the ``(run_id, variant, seed)``
cell key that coordinates an array task with its artifact directory. All pure
functions of the validated ``RunConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragtime.config import RunConfig

__all__ = ["cell_key", "run_family", "run_id", "variant"]

# The submission-filename ceiling. `config.schema` also does a cheap advisory check,
# but the canonical derivation lives here.
MAX_RUN_ID_LEN = 25


def run_id(cfg: RunConfig) -> str:
    """Return the run id, asserting it fits the submission-filename ceiling."""
    rid = cfg.run_id
    if len(rid) > MAX_RUN_ID_LEN:
        raise ValueError(
            f"run_id {rid!r} is {len(rid)} chars; must be <= {MAX_RUN_ID_LEN} "
            f"(TREC submission-filename ceiling)."
        )
    return rid


def run_family(cfg: RunConfig) -> str:
    """Return the family an ``e2e-*`` or ``mlir-*`` run belongs to, by run-id prefix.

    Delegates to ``config.fairness``'s canonical grouping, so the plan, the roster
    discovery and the fairness gate cannot drift on which runs share a corpus.
    """
    from ragtime.config.fairness import _family_of

    return _family_of(cfg.run_id)


def variant(cfg: RunConfig) -> str:
    """Return the run's ``passage_lang``, the translation value the cell key carries.

    Every shipped run, both families, keys its cell on the reading knob: the deployed
    directories are ``e2e-omt__omt__seed0`` but ``mlir-omt__original__seed0``, because an
    ``mlir-*`` run holds ``passage_lang`` at ``original`` and moves ``retrieval.index``
    instead. Which index an mlir arm searched is recorded by its config and its retrieval
    service descriptor, not by this name; ``run_id`` already disambiguates the arms.
    """
    return cfg.passage_lang


def cell_key(run_id: str, variant: str, seed: int) -> str:
    """Return the stable ``(run_id, variant, seed)`` key.

    This is both the array-task coordinate and the per-cell artifact directory name.
    """
    return f"{run_id}__{variant}__seed{seed}"

