"""``download`` idempotence and byte-exactness, against a mocked hub.

The corpus fetch has to keep the source bytes exactly as published and never edit them in
place, and a re-run over a complete store has to be a no-op. Both are checked without the
network: ``snapshot_download`` and the source of the LFS shas are mocked seams inside
``download.py``. A corrupted download raises before ``_SUCCESS`` is written, and deleting
``_SUCCESS`` is what forces a re-fetch.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

from ragtime.common import Layout
from ragtime.common.io import is_done, success_marker
from ragtime.config import all_hashes

pytestmark = pytest.mark.small

dl = importlib.import_module("ragtime.preprocess.download")


def _raw(tmp_path: Path, cfg) -> Path:
    ch = all_hashes(cfg)["chunker"]
    return Layout(run_dir=tmp_path, base=tmp_path).corpus_raw_dir("e2e", ch)


def test_first_download_lands_files_with_success_and_correct_shas(
    tmp_path, chunk_cfg, mocked_snapshot_download, lfs_blob_shas
) -> None:
    raw = dl.download(chunk_cfg, base=tmp_path)
    assert raw == _raw(tmp_path, chunk_cfg)
    assert is_done(raw)  # _SUCCESS written last
    for name, sha in lfs_blob_shas.items():
        f = raw / name
        assert f.is_file()
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha
    # no partial temp dir left behind (atomic rename).
    assert not any(p.name.startswith(f".{raw.name}.tmp") for p in raw.parent.iterdir())


def test_idempotent_no_op_does_not_reinvoke_snapshot(
    tmp_path, chunk_cfg, mocked_snapshot_download
) -> None:
    dl.download(chunk_cfg, base=tmp_path)
    assert len(mocked_snapshot_download.calls) == 1
    dl.download(chunk_cfg, base=tmp_path)  # complete store -> no-op
    assert len(mocked_snapshot_download.calls) == 1  # not re-invoked


def test_deleting_success_forces_refetch(
    tmp_path, chunk_cfg, mocked_snapshot_download
) -> None:
    raw = dl.download(chunk_cfg, base=tmp_path)
    assert len(mocked_snapshot_download.calls) == 1
    success_marker(raw).unlink()  # simulate a suspected-partial store
    dl.download(chunk_cfg, base=tmp_path)
    assert len(mocked_snapshot_download.calls) == 2  # re-invoked
    assert is_done(raw)


def test_corrupted_download_raises_before_success(
    tmp_path, chunk_cfg, monkeypatch
) -> None:
    """A sha mismatch raises, and no _SUCCESS is published."""

    def _bad_snapshot(repo_id, repo_type, allow_patterns, local_dir):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "eng-docs.jsonl.gz").write_bytes(b"tampered\n")

    monkeypatch.setattr(dl, "snapshot_download", _bad_snapshot, raising=False)
    monkeypatch.setattr(
        dl,
        "_fetch_expected_shas",
        lambda repo_id: {"eng-docs.jsonl.gz": hashlib.sha256(b"expected\n").hexdigest()},
    )
    with pytest.raises(ValueError, match="corrupted download"):
        dl.download(chunk_cfg, base=tmp_path)
    raw = _raw(tmp_path, chunk_cfg)
    assert not is_done(raw)  # never marked complete


def test_finalize_failure_is_all_or_nothing(
    tmp_path, chunk_cfg, mocked_snapshot_download, monkeypatch
) -> None:
    """A crash during the promote leaves the canonical path absent rather than partial."""

    def _boom(tmp, raw):
        raise OSError("disk full mid-promote")

    monkeypatch.setattr(dl, "_promote", _boom)
    raw = _raw(tmp_path, chunk_cfg)
    with pytest.raises(OSError, match="mid-promote"):
        dl.download(chunk_cfg, base=tmp_path)
    assert not raw.exists()  # all-or-nothing: the final store is never partially visible
    assert not is_done(raw)
    # no orphaned temp dir left behind either.
    assert not any(p.name.startswith(f".{raw.name}.tmp") for p in raw.parent.iterdir())


def test_download_is_byte_exact_and_never_edits_in_place(
    tmp_path, chunk_cfg, mocked_snapshot_download, lfs_blob_shas
) -> None:
    """Stored bytes are the LFS blob bytes with no re-encode, and a resume no-op leaves
    both the content and the mtime alone."""
    raw = dl.download(chunk_cfg, base=tmp_path)
    for name, sha in lfs_blob_shas.items():
        assert hashlib.sha256((raw / name).read_bytes()).hexdigest() == sha  # byte-exact
    snap = {name: (raw / name).read_bytes() for name in lfs_blob_shas}
    mtimes = {name: (raw / name).stat().st_mtime_ns for name in lfs_blob_shas}

    dl.download(chunk_cfg, base=tmp_path)  # resume no-op
    for name in lfs_blob_shas:
        assert (raw / name).read_bytes() == snap[name]  # bytes unchanged
        assert (raw / name).stat().st_mtime_ns == mtimes[name]  # never touched in place
