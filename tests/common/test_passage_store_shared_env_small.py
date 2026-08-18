"""One LMDB read env per store per process, reference-counted, and never a stale one.

LMDB refuses a second ``lmdb.open`` of an environment already open in the process. That
refusal is correct, since two envs are two mmaps that can disagree, but this store has
several legitimate simultaneous holders in one process: ``retrieval.context.bring_up``
(the store path is keyed by ``(recon, pack)`` alone, so contexts over two renderings name
the same store), ``preprocess.passage_store_build``'s census and validate re-opens, and
the search tool. Without a shared env the second corpus-scale caller of ``bring_up``
dies with::

    lmdb.Error: The environment '.../passages.lmdb' is already open in this process.

Four properties, the last of which is why the fix is not simply a cache keyed by path:

1. two opens of one store succeed and share one env;
2. closing one handle leaves the other fully readable, and the env closes exactly when
   the last handle closes, which a write open of the same path then demonstrates;
3. different stores never share an env;
4. a republished store at the same path is never served from the previous env.
   Publication is build-into-``tmp``-then-rename (``preprocess.index._publish_dir``), so
   a path-keyed cache would hand a later reader the deleted inode. The key therefore
   carries the env's on-disk identity (device and inode) rather than just its path, and a
   caller that asks for a republished store while the pre-swap handle is alive gets
   py-lmdb's own refusal instead of the previous build's bytes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import io as common_io
from ragtime.common.passage_store import _SHARED_ENVS, LmdbPassageStore

pytestmark = pytest.mark.small


def _records(marker: str) -> list[dict[str, Any]]:
    """Two passages whose ``original`` text carries ``marker``, an identity fingerprint."""
    return [
        {
            "passage_id": f"doc-{i}#p0",
            "document_id": f"doc-{i}",
            "lang": "en",
            "sentence_ids": [f"doc-{i}#s0"],
            "sentence_char_spans": [[0, len(f"{marker} passage {i}.")]],
            "paragraph_index": [0],
            "token_count": 4,
            "is_oversized": False,
            "original": f"{marker} passage {i}.",
            "omt": f"{marker} passage {i}.",
            "omt_opus": f"{marker} passage {i}.",
        }
        for i in range(2)
    ]


def _build(path: Path, marker: str) -> None:
    """Build a store at ``path`` and leave no handle open on it."""
    LmdbPassageStore.build(path, _records(marker)).close()


def _keys_for(path: Path) -> list[tuple[str, int, int]]:
    resolved = str(Path(path).resolve())
    return [k for k in _SHARED_ENVS if k[0] == resolved]


def test_two_opens_of_one_store_share_a_single_env(tmp_path: Path) -> None:
    """The failure at its smallest: open the same store twice, in one process."""
    db = tmp_path / "passages.lmdb"
    _build(db, "alpha")

    first = LmdbPassageStore(db)
    try:
        # Without a shared env this raises lmdb.Error("... already open in this process").
        second = LmdbPassageStore(db)
        try:
            assert first._env is second._env, "two envs over one store, not one shared env"
            assert len(_keys_for(db)) == 1
            assert first.render("doc-0#p0", "original") == "alpha passage 0."
            assert second.render("doc-0#p0", "original") == "alpha passage 0."
        finally:
            second.close()
        # The first handle is untouched by the second's close: a holder may never close
        # another holder's reader.
        assert first.render("doc-1#p0", "original") == "alpha passage 1."
    finally:
        first.close()


def test_the_env_closes_with_the_last_handle_not_the_first(tmp_path: Path) -> None:
    """A write open is LMDB's own witness that no reader env is left over."""
    db = tmp_path / "passages.lmdb"
    _build(db, "beta")

    a, b = LmdbPassageStore(db), LmdbPassageStore(db)
    a.close()
    assert _keys_for(db), "the env was dropped while a handle still held it"
    assert b.render("doc-0#p0", "original") == "beta passage 0."
    b.close()
    assert not _keys_for(db), "the last close did not release the env"

    # LMDB refuses any second open of a live environment, so this write open succeeding is
    # proof the read env is really closed rather than merely dereferenced.
    env = common_io.lmdb_open(db, readonly=False)
    env.close()

    # close() is idempotent per handle: a double close must not release a reference twice.
    a.close()
    b.close()
    assert not _keys_for(db)


def test_two_different_stores_never_share_an_env(tmp_path: Path) -> None:
    left, right = tmp_path / "left.lmdb", tmp_path / "right.lmdb"
    _build(left, "left")
    _build(right, "right")

    with LmdbPassageStore(left) as a, LmdbPassageStore(right) as b:
        assert a._env is not b._env
        assert a.render("doc-0#p0", "original") == "left passage 0."
        assert b.render("doc-0#p0", "original") == "right passage 0."


def test_a_republished_store_is_never_served_from_the_previous_env(tmp_path: Path) -> None:
    """temp->rename publication swaps the inode; a path-keyed cache would read the old one.

    Sharing is for the case where the store on disk is the same store. A republished one
    is not, and the key says so by carrying ``data.mdb``'s device and inode. While the
    pre-swap handle is alive the caller gets py-lmdb's own refusal, since its
    duplicate-env guard is path-level, rather than the previous build's bytes. Once the
    stale handle lets go, the fresh store opens and reads the new content.
    """
    import lmdb

    final = tmp_path / "passages.lmdb"
    staging = tmp_path / "passages.lmdb.tmp"
    _build(final, "v1")

    stale = LmdbPassageStore(final)  # a live handle on the old inode
    try:
        assert stale.render("doc-0#p0", "original") == "v1 passage 0."
        # Republish exactly as ``preprocess.index._publish_dir`` does: build elsewhere,
        # remove the old directory, rename onto the final path.
        _build(staging, "v2")
        shutil.rmtree(final)
        os.replace(staging, final)

        with pytest.raises(lmdb.Error):
            LmdbPassageStore(final)  # never silently served the dead inode's env
    finally:
        stale.close()

    fresh = LmdbPassageStore(final)
    try:
        assert fresh._env is not stale._env
        assert fresh.render("doc-0#p0", "original") == "v2 passage 0."
    finally:
        fresh.close()
