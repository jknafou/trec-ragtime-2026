"""``bring_up`` twice in one process over one corpus: the LMDB ownership contract.

A corpus-scale gate once failed here rather than in the retrieval logic. Five tests each stood a
context up over the real index, and the second one to reach ``context.py``'s
``store = LmdbPassageStore(store_path)`` died with

    lmdb.Error: The environment '.../passages.lmdb' is already open in this process.

That is not a test-only shape. The store path is keyed by ``(recon_hash, pack_hash)`` alone, with
no ``index`` or ``passage_lang`` level, so any process holding one context per rendering (a
multilingual sweep over Knob 1, a driver comparing Knob 2) opens the same environment twice by
design. The rest of the small suite injects an in-memory store into ``bring_up`` and never
exercises the branch that opens one, which is why the failure surfaced only at corpus scale.

Both halves of the ownership contract are pinned:

* a context that opened the store owns one handle, not the environment: a second bring-up
  shares the env, and closing one context leaves the other's store readable;
* a context handed an injected store never closes it (``_owns_store`` is false), because the
  caller that opened it may still be reading through it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Statistics
from ragtime.common.io import success_marker
from ragtime.common.passage_store import LmdbPassageStore
from ragtime.retrieval import bring_up
from tests.retrieval.conftest import Built, retrieval_cfg

pytestmark = pytest.mark.small


def _publish_store(built: Built) -> Path:
    """Build + publish the by-id store where ``bring_up`` looks for it, then let go of it.

    Built through the production entrypoint (``build_from_final`` over the fixture's own
    final tables), so the bytes ``bring_up`` opens are the bytes
    ``preprocess.passage_store_build`` would have written.
    """
    path = built.layout.passage_store_path(built.recon_hash, None)
    LmdbPassageStore.build_from_final(
        path, built.layout, built.recon_hash, pack_hash=None
    ).close()
    success_marker(path).write_bytes(b"")
    return path


def _up(built: Built, cfg: Any, **kwargs: Any):
    return bring_up(
        cfg,
        built.clients,
        built.layout,
        recon_hash=built.recon_hash,
        pack_hash=None,
        idx_hash=built.idx_hash,
        legs=built.legs,
        stats=Statistics(),
        **kwargs,
    )


def test_two_bring_ups_over_one_corpus_share_the_store_env(built: Built) -> None:
    """Two renderings, one store path, one process, and no lmdb.Error."""
    _publish_store(built)
    probe = next(iter(built.records))

    first = _up(built, retrieval_cfg(built.root, passage_lang="original"))
    try:
        # Without one shared env this second bring-up raises lmdb.Error at context.py's open.
        second = _up(built, retrieval_cfg(built.root, passage_lang="omt"))
        try:
            assert first._owns_store and second._owns_store
            assert first.passage_store is not second.passage_store  # two handles
            assert first.passage_store._env is second.passage_store._env  # one env
            assert first.passage_lang == "original"
            assert second.passage_lang == "omt"
            assert first.passage_store.render(probe, "original") == (
                built.records[probe]["original"]
            )
        finally:
            second.close()
        # Closing one context may not disturb the other's reader.
        assert first.passage_store.render(probe, "original") == (
            built.records[probe]["original"]
        )
    finally:
        first.close()


def test_close_is_idempotent_and_releases_only_this_contexts_handle(built: Built) -> None:
    path = _publish_store(built)
    probe = next(iter(built.records))

    holder = LmdbPassageStore(path)  # a caller that opened the store for itself
    try:
        ctx = _up(built, retrieval_cfg(built.root))
        assert ctx._owns_store
        ctx.close()
        ctx.close()  # idempotent: a second close must not release a second reference
        assert holder.render(probe, "original") == built.records[probe]["original"]
    finally:
        holder.close()


def test_an_injected_store_is_never_closed_by_the_context(built: Built) -> None:
    """The caller kept a reference to it; a context that did not open it may not close it."""
    path = _publish_store(built)
    probe = next(iter(built.records))

    injected = LmdbPassageStore(path)
    try:
        ctx = _up(built, retrieval_cfg(built.root), passage_store=injected)
        assert not ctx._owns_store
        ctx.close()
        assert injected.render(probe, "original") == built.records[probe]["original"]
    finally:
        injected.close()
