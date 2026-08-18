"""A streaming file hasher, used where an artifact records the sha256 of an input file.

``sha256_file`` reads in 1 MiB chunks, so a multi-gigabyte file is never loaded whole. Its
one production consumer is ``preprocess.reconcile``, which stamps the hash of each input
table into the reconciled inventory. It is distinct from ``config.config_hash``, which
fingerprints parsed config blocks rather than bytes on disk.

What a run records is written down elsewhere: the config file is the complete run record,
``config.all_hashes`` is the canonical block fingerprint, and ``saturate.worker_provenance``
stamps node, job id and GPU model onto every finished shard. ``devkit.devrun`` keeps its own
``_git_sha`` for the dev-harness marker, which is the only place a commit sha is written beside
artifacts.
"""

from __future__ import annotations

import hashlib
import os

__all__ = ["sha256_file"]

_CHUNK = 1 << 20  # streaming read size, so a multi-GB file is never loaded whole


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the sha256 of a file as 64 lowercase hex characters, read in chunks.

    This is a file hasher, distinct from ``config.config_hash``, which fingerprints
    parsed config blocks.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()
