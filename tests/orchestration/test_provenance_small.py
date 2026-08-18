"""``sha256_file``, the streaming file hasher.

The one function ``orchestration.provenance`` still holds: it hashes a file in 1 MiB
chunks, so a multi-gigabyte input is never loaded whole, and its production consumer is
``preprocess.reconcile``, which stamps each input table's hash into the reconciled
inventory. The streamed digest must equal a naive whole-file digest, which is what makes
the chunking an optimization rather than a different hash.

The module's ``pins`` / ``write`` / ``git_sha`` -- an eight-key manifest and the
``provenance.json`` writer -- were removed together with their tests: nothing outside this
file ever called them, so no run ever wrote that file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ragtime.orchestration import sha256_file

pytestmark = pytest.mark.small


def test_streamed_digest_equals_a_whole_file_digest(uv_lock_path: Path) -> None:
    """Chunked and whole-file sha256 agree on a real, committed multi-hundred-KB file."""
    streamed = sha256_file(uv_lock_path)
    naive = hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
    assert streamed == naive
    assert len(streamed) == 64
    assert streamed == streamed.lower()  # 64 lowercase hex, never re-wrapped


def test_hashes_a_file_larger_than_one_chunk(tmp_path: Path) -> None:
    """A file spanning several 1 MiB reads hashes correctly, so the loop is not off by one."""
    payload = (b"ragtime" * 1024) * 512  # ~3.5 MiB, not a whole multiple of the chunk size
    p = tmp_path / "big.bin"
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_empty_file_hashes_to_the_empty_digest(tmp_path: Path) -> None:
    """A zero-byte file terminates the read loop immediately rather than hanging."""
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()
