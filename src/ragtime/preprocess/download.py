"""Byte-exact, idempotent corpus download via ``huggingface_hub.snapshot_download``.

Fetches the RAGTIME2 dataset repo's raw LFS files with per-file sha verification
rather than ``datasets.load_dataset``, which re-encodes to Arrow and so no longer
carries the source bytes. The store is shared by a whole run family and resolved
through ``common.Layout``.

The download lands in a temp directory, every file's sha256 is checked against the
repo's LFS blob hash, the temp directory is atomically renamed into place and a
``_SUCCESS`` marker is written last, so a partial file is never visible at the final
path and a re-run over a complete store is a no-op. A sha mismatch raises before
``_SUCCESS`` is written.

``snapshot_download`` is imported at module level as the seam tests patch; the network
is touched only inside :func:`download`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download
from ragtime.common import Layout, get_logger
from ragtime.common.io import is_done, success_marker
from ragtime.config import all_hashes
from ragtime.orchestration.run_identity import run_family

if TYPE_CHECKING:
    from ragtime.config import RunConfig

__all__ = ["download"]

_log = get_logger("preprocess.download")

_REPO_ID = "trec-ragtime/ragtime2"
# Raw files only: native docs, doc-level organizer MT and the README.
_ALLOW_PATTERNS = ["*-docs.jsonl.gz", "*-trans.jsonl.gz", "README*"]
# Fallback artifact root; real runs pass an explicit ``base`` resolved from Layout.
_DEFAULT_BASE = "runs"
_CHUNK = 1 << 20  # streamed sha256 block size


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_expected_shas(repo_id: str) -> dict[str, str]:
    """Return the repo's per-file LFS blob sha256s, the byte-exactness ground truth.

    A network read of the dataset repo metadata; tests monkeypatch this seam so the sha
    check runs offline against fixture bytes.
    """
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, repo_type="dataset", files_metadata=True)
    return {s.rfilename: s.lfs.sha256 for s in (info.siblings or ()) if getattr(s, "lfs", None)}


def _snapshot(repo_id: str, local_dir: Path) -> None:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=_ALLOW_PATTERNS,
        local_dir=str(local_dir),
    )


def _verify_shas(root: Path, expected: dict[str, str]) -> None:
    """Check each downloaded file's sha256 against its recorded LFS blob hash."""
    for rel, want in expected.items():
        f = root / rel
        if not f.is_file():
            continue  # allow_patterns may not have requested this repo file
        got = _sha256_file(f)
        if got != want:
            raise ValueError(
                f"corrupted download: {rel} sha256={got} != expected {want} "
                f"(refusing to publish; delete the partial store and re-run)"
            )


def download(cfg: RunConfig, *, base: str | os.PathLike[str] | None = None) -> Path:
    """Fetch the raw corpus into ``Layout.corpus_raw_dir(family)``.

    Returns the family-shared raw-corpus directory. A complete, ``_SUCCESS``-marked store
    is a no-op. To force a re-fetch after a suspected partial download, delete the store's
    ``_SUCCESS`` marker.
    """
    fam = run_family(cfg)
    ch = all_hashes(cfg)["chunker"]  # the corpus store is keyed off the semantic chunker hash
    root = Path(base) if base is not None else Path(_DEFAULT_BASE)
    raw = Layout(run_dir=root, base=root).corpus_raw_dir(fam, ch)
    if is_done(raw):
        return raw

    expected = _fetch_expected_shas(_REPO_ID)
    raw.parent.mkdir(parents=True, exist_ok=True)
    tmp = raw.parent / f".{raw.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        _snapshot(_REPO_ID, tmp)
        _verify_shas(tmp, expected)
        _promote(tmp, raw)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    marker = success_marker(raw)
    marker.write_bytes(b"")
    _log.info("preprocess.download.done", family=fam, path=str(raw))
    return raw


def _promote(tmp: Path, raw: Path) -> None:
    """Atomically replace ``raw`` with the fully verified ``tmp`` store.

    Kept as a separate seam so a test can force a mid-finalize failure and assert the
    canonical path is never left partial.
    """
    if raw.exists():
        shutil.rmtree(raw)
    os.replace(tmp, raw)
