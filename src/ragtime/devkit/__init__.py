"""The retrieval service, and the client side of its registry.

:mod:`~ragtime.devkit.rsvc` is the long-lived process that holds one whole index rendering
resident and answers ranking requests off a filesystem queue. It is what served every query
behind this repository's submission, and ``slurm/retrieval_service.sbatch`` launches it.
:mod:`~ragtime.devkit.rsvc_registry` is the descriptor protocol a client resolves it through:
the service publishes its rendering, its index hash and a heartbeat, and
:func:`ragtime.retrieval.service_context` binds a client to a live one.

The package is a leaf of the dependency spine. It imports the layers to its left -- ``retrieval``
for the ranking path, ``preprocess.index`` for the engines, ``serving`` for the encoders and the
reranker -- and nothing imports it back, so the service may hold code no stage sees and no stage
acquires a dependency on a running service. It reads the real corpus and index while writing only
into a service-scoped tree; see :mod:`ragtime.devkit.devrun` for how one ``Layout`` does both.
"""

from __future__ import annotations

from .devrun import DevRun, dev_root, resolve_dev_run

__all__ = ["DevRun", "dev_root", "resolve_dev_run"]
