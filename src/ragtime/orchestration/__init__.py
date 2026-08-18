"""SLURM-first backbone behind the ``run --config`` flow.

Turns one self-contained ``config/<run>.yml`` into launched, resumable SLURM work. Owns
the config-to-JobDAG planner, run identity, seed expansion, the async k-loop fan-out map,
``saturate`` (the seed/worker/drive protocol every work-queued stage is driven by) and the
SLURM layer: the ``sbatch`` submitter, the dynamic self-claiming work queue and the sbatch
templates.

At module level it imports only ``common``, ``config`` and ``serving``, so the import graph
flows one way and nothing here depends on a stage. ``cli`` is the exception that proves it:
dispatching ``run --stage`` means reaching forward into ``preprocess`` and ``pipeline``, and
it does so with the import deferred into the dispatching function, so a stage is loaded only
when it is about to run.
"""

from __future__ import annotations

from .cli import main
from .determinism import expand_seeds
from .fanout import map
from .plan import JobDAG, JobNode, already_done, build_plan, run_local
from .provenance import sha256_file
from .run_identity import cell_key, run_family, run_id, variant

__all__ = [  # noqa: RUF022 - grouped by module rather than sorted
    # cli entrypoint (the `run` console script)
    "main",
    # plan
    "build_plan",
    "JobDAG",
    "JobNode",
    "already_done",
    "run_local",
    # run identity
    "run_id",
    "run_family",
    "variant",
    "cell_key",
    # determinism
    "expand_seeds",
    # provenance
    "sha256_file",
    # fanout
    "map",
]
