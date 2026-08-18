"""Seed expansion: the concrete seed values a run fans over.

``expand_seeds`` turns the required ``cfg.seeds`` count into the seed values the plan and
the pipeline stage iterate. A pure function of the validated ``RunConfig``.

Seeds are the only determinism input this module owns. The rest of a run's identity is carried
by the run config file, which is itself the complete record, by ``config.all_hashes``, the
canonical block fingerprint, and by ``saturate.worker_provenance``, which stamps the observed
hardware onto each shard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragtime.config import RunConfig

__all__ = ["expand_seeds"]


def expand_seeds(cfg: RunConfig) -> list[int]:
    """Return the concrete seed values for the run, read from ``cfg.seeds``.

    ``cfg.seeds`` is the validated count; the values are the stable range
    ``0..count-1``. The count is always read from the config, never defaulted here,
    so a config with a different count yields a different-length list.
    """
    count = int(cfg.seeds)
    return list(range(count))
