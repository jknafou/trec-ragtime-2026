"""Load, validate, hash and fairness-gate the launch config.

Owns the semantics of the self-contained ``config/<run>.yml`` that is the whole
reproducible record; there is no external registry. It loads and validates every
block, computes the per-block ``config_hash`` that keys every artifact, and
machine-checks the fairness invariant across a run family, which is the hard gate
that runs before any SLURM fan-out or GPU allocation. This package imports only
``ragtime.common``, and no stage imports its logic.
"""

from __future__ import annotations

from .fairness import FairnessError, check, family_guard, shared_block_hash, top_level_blocks
from .hashing import all_hashes, config_hash
from .loader import load
from .schema import ConfigError, RunConfig, validate

__all__ = [
    "ConfigError",
    "FairnessError",
    "RunConfig",
    "all_hashes",
    "check",
    "config_hash",
    "family_guard",
    "load",
    "shared_block_hash",
    "top_level_blocks",
    "validate",
]
