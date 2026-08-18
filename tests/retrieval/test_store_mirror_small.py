"""``execution.passage_store_mirror_root``: a scored run can open a staged passage store.

These are identity tests, not latency ones. The mirror is the same LMDB opened from a different
inode, so the only thing worth proving is that it changes no returned byte: origin and mirror,
element for element, the same ids in the same order, in every rendering. Latency is what the mirror
is for, but it is not what these tests measure, because a tmpfs is not reachable from a small
fixture and a timing assertion in a unit test measures the machine.

The other half is the fallback. A mirror that quietly disabled itself would be orders of magnitude
slower with nothing in the record, so every refusal is pinned: missing, unpublished, wrong size,
addressed off-key, and an origin that was never built at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Statistics
from ragtime.common.io import success_marker
from ragtime.common.passage_store import RENDERINGS, LmdbPassageStore
from ragtime.retrieval import bring_up, display, resolve_store_location, retrieve
from ragtime.retrieval._index_api import IndexIntegrityError
from ragtime.retrieval.stats import (
    STAT_RERANK_SECONDS,
    STAT_STORE_FETCH_SECONDS,
    STAT_STORE_MIRROR_REFUSED,
    STAT_STORE_MIRRORED,
)
from ragtime.retrieval.store_mirror import (
    PASSAGE_STORE_MIRROR_ROOT_KEY,
    STORE_DATA_FILE,
)
from tests.retrieval.conftest import Built, retrieval_cfg

pytestmark = pytest.mark.small


def _publish_store(built: Built) -> Path:
    """Build + publish the canonical by-id store through the PRODUCTION entrypoint."""
    path = built.layout.passage_store_path(built.recon_hash, None)
    LmdbPassageStore.build_from_final(
        path, built.layout, built.recon_hash, pack_hash=None
    ).close()
    success_marker(path).write_bytes(b"")
    return path


def _stage(built: Built, root: Path, origin: Path) -> Path:
    """A STAGER's output: the store copied to the same relative path under ``root``, published.

    The whole contract a real stager must satisfy is the relative path and the ``_SUCCESS`` marker,
    so a test that skipped either would be testing a shape no run produces. A staging script moves
    the bytes at about 1 GiB/s with 32 byte-range workers; a fixture store is a few KiB, so a plain
    copy produces the same artifact.
    """
    relative = origin.relative_to(Path(built.layout.base))
    mirrored = root / relative
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(origin, mirrored)
    success_marker(mirrored).write_bytes(b"")
    return mirrored


def _cfg_with_mirror(built: Built, root: Path | str | None, **kwargs: Any):
    cfg = retrieval_cfg(built.root, **kwargs)
    if root is not None:
        cfg.blocks.setdefault("execution", {})["passage_store_mirror_root"] = str(root)
    return cfg


def _up(built: Built, cfg: Any, stats: Statistics | None = None):
    """``bring_up`` with no injected store: the constructor a scored run actually uses."""
    return bring_up(
        cfg,
        built.clients,
        built.layout,
        recon_hash=built.recon_hash,
        pack_hash=None,
        idx_hash=built.idx_hash,
        legs=built.legs,
        stats=stats if stats is not None else Statistics(),
    )


# --------------------------------------------------------------------------- #
# The gate: the mirror changes no returned byte.
# --------------------------------------------------------------------------- #
def test_mirror_and_origin_return_element_for_element_identical_passages(
    built: Built, tmp_path: Path
) -> None:
    """Every id, every rendering, same order: the whole list, not a spot check.

    This is the property the whole change rests on: a mirror is a COPY, so if any pair
    disagreed the optimization would be a silent results change on the one path where being
    wrong is invisible (the three renderings share passage ids, so a wrong store answers
    every id and errors nowhere).
    """
    origin = _publish_store(built)
    root = tmp_path / "shm"
    _stage(built, root, origin)
    ids = sorted(built.records)
    assert len(ids) >= 24  # the fixture corpus, all four language cells

    plain = _up(built, _cfg_with_mirror(built, None))
    staged = _up(built, _cfg_with_mirror(built, root))
    try:
        assert plain.store_location is not None
        assert not plain.store_location.mirrored
        assert plain.store_location.path == origin
        assert staged.store_location is not None
        assert staged.store_location.mirrored, staged.store_location.refusal
        assert staged.store_location.path != origin
        assert staged.store_location.refusal is None

        for rendering in RENDERINGS:
            assert display(plain, ids, rendering) == display(staged, ids, rendering)
        # ...and the rendering-agnostic raw accessor, id by id.
        for pid in ids:
            for rendering in RENDERINGS:
                assert plain.passage_store.render(pid, rendering) == (
                    staged.passage_store.render(pid, rendering)
                )
        # Order is part of the contract, so a reversed request must agree too.
        assert display(plain, ids[::-1], "original") == display(staged, ids[::-1], "original")
    finally:
        staged.close()
        plain.close()


def test_the_mirrored_context_really_opened_the_other_inode(
    built: Built, tmp_path: Path
) -> None:
    """Not a tautology: prove the two contexts read two different files on disk.

    Without this the identity test above could pass by opening the origin twice, which is
    exactly the way a "fast path" quietly reports success while doing nothing.
    """
    origin = _publish_store(built)
    root = tmp_path / "shm"
    mirrored = _stage(built, root, origin)

    stats = Statistics()
    ctx = _up(built, _cfg_with_mirror(built, root), stats=stats)
    try:
        assert ctx.store_location is not None
        assert ctx.store_location.path == mirrored
        assert (mirrored / STORE_DATA_FILE).stat().st_ino != (
            origin / STORE_DATA_FILE
        ).stat().st_ino
        assert stats.value(STAT_STORE_MIRRORED) == 1.0
        assert stats.value(STAT_STORE_MIRROR_REFUSED) == 0.0
        assert ctx.store_witness == {"path": str(mirrored), "mirrored": True}
    finally:
        ctx.close()


# --------------------------------------------------------------------------- #
# The fallback: loud, counted, and never a wrong answer.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "break_it",
    [
        pytest.param(lambda m: shutil.rmtree(m), id="missing"),
        pytest.param(lambda m: success_marker(m).unlink(), id="unpublished"),
        pytest.param(
            lambda m: (m / STORE_DATA_FILE).write_bytes(b"truncated"), id="wrong_size"
        ),
    ],
)
def test_an_unusable_mirror_falls_back_to_the_origin_loudly(
    built: Built, tmp_path: Path, break_it: Any
) -> None:
    """Three ways a staged copy is not servable; one behaviour: origin + counter + reason."""
    origin = _publish_store(built)
    root = tmp_path / "shm"
    mirrored = _stage(built, root, origin)
    break_it(mirrored)

    stats = Statistics()
    ctx = _up(built, _cfg_with_mirror(built, root), stats=stats)
    try:
        assert ctx.store_location is not None
        assert not ctx.store_location.mirrored
        assert ctx.store_location.path == origin
        assert ctx.store_location.refusal  # a reason, not just a flag
        assert stats.value(STAT_STORE_MIRROR_REFUSED) == 1.0
        assert stats.value(STAT_STORE_MIRRORED) == 0.0
        assert "refusal" in (ctx.store_witness or {})
        # The run is still CORRECT: a refusal costs latency, never an answer.
        probe = next(iter(built.records))
        assert ctx.passage_store.render(probe, "original") == (
            built.records[probe]["original"]
        )
    finally:
        ctx.close()


def test_a_store_placed_off_key_under_the_mirror_root_is_not_found(
    built: Built, tmp_path: Path
) -> None:
    """The ROOT is redirected and the recon12/pack12 path is RE-RESOLVED: property 1.

    A free-form path argument would let a caller address a store built from a different
    packing, which answers every passage id and errors nowhere. So a copy sitting at the root
    rather than at its keyed relative path is simply not the store this run asked for, and the
    run must fall back rather than serve it.
    """
    origin = _publish_store(built)
    root = tmp_path / "shm"
    root.mkdir()
    shutil.copytree(origin, root / "passages.lmdb")  # right bytes, wrong address
    success_marker(root / "passages.lmdb").write_bytes(b"")

    ctx = _up(built, _cfg_with_mirror(built, root))
    try:
        assert ctx.store_location is not None
        assert ctx.store_location.path == origin
        assert "does not exist" in (ctx.store_location.refusal or "")
    finally:
        ctx.close()


def test_a_perfect_mirror_cannot_substitute_for_an_unpublished_origin(
    built: Built, tmp_path: Path
) -> None:
    """The origin's published check runs first and is unchanged.

    A mirror may make a published store fast; it may never make an unbuilt one exist. This is
    the ordering that keeps ``_open_store``'s refusal meaningful.
    """
    origin = _publish_store(built)
    root = tmp_path / "shm"
    _stage(built, root, origin)
    success_marker(origin).unlink()  # the canonical store was never published

    with pytest.raises(IndexIntegrityError, match="no published by-id passage store"):
        _up(built, _cfg_with_mirror(built, root))


def test_no_mirror_root_is_the_shipped_default(built: Built) -> None:
    """Absent (and empty-string) means the canonical store, with nothing counted either way."""
    origin = _publish_store(built)
    for root in (None, ""):
        stats = Statistics()
        ctx = _up(built, _cfg_with_mirror(built, root), stats=stats)
        try:
            assert ctx.store_location is not None
            assert ctx.store_location.path == origin
            assert not ctx.store_location.mirrored
            assert ctx.store_location.refusal is None
            assert stats.total(STAT_STORE_MIRRORED) == 0.0
            assert stats.total(STAT_STORE_MIRROR_REFUSED) == 0.0
        finally:
            ctx.close()


def test_an_injected_store_reports_no_location(built: Built, tmp_path: Path) -> None:
    """The caller owns that path; this context resolved nothing and must not claim it did."""
    _publish_store(built)
    ctx = bring_up(
        _cfg_with_mirror(built, tmp_path / "shm"),
        built.clients,
        built.layout,
        recon_hash=built.recon_hash,
        pack_hash=None,
        idx_hash=built.idx_hash,
        passage_store=built.store(),
        legs=built.legs,
        stats=Statistics(),
    )
    try:
        assert ctx.store_location is None
        assert ctx.store_witness is None
    finally:
        ctx.close()


def test_the_leaf_is_declared_in_the_closed_execution_schema() -> None:
    """``execution`` is a CLOSED schema, so an undeclared key makes every config fail to load.

    And it must be in ``execution`` rather than a hashed block: the mirror changes no returned
    byte, so hashing it would re-key every artifact the day a node's scratch path moves.
    """
    from ragtime.config.schema import _ALLOWED

    assert PASSAGE_STORE_MIRROR_ROOT_KEY in _ALLOWED["execution"]
    for hashed in ("retrieval", "index_build", "packing", "chunker"):
        assert PASSAGE_STORE_MIRROR_ROOT_KEY not in _ALLOWED.get(hashed, {})


def test_resolve_store_location_is_pure_and_never_raises(built: Built, tmp_path: Path) -> None:
    """The resolver reports; the CALLER counts and logs. Pinned so the two never merge."""
    origin = built.layout.passage_store_path(built.recon_hash, None)
    absent = resolve_store_location(
        built.layout, built.recon_hash, None, root=tmp_path / "nope"
    )
    assert absent.origin == origin and absent.path == origin
    assert not absent.mirrored and absent.refusal

    unset = resolve_store_location(built.layout, built.recon_hash, None, root=None)
    assert unset.path == origin and unset.refusal is None

    # A mirror root that resolves BACK to the origin is a no-op, not a "mirror".
    same = resolve_store_location(
        built.layout, built.recon_hash, None, root=Path(built.layout.base)
    )
    assert same.path == origin and not same.mirrored and same.refusal is None


# --------------------------------------------------------------------------- #
# The attribution fix: the fetch is timed apart from the model.
# --------------------------------------------------------------------------- #
def test_the_store_fetch_is_timed_separately_from_the_cross_encoder(built: Built) -> None:
    """``service._rerank`` used to start its timer after the fetch, so the fetch had no counter.

    At corpus scale that untimed step is the largest single cost on the query path: 24.5339 s
    median off shared storage, against a 3.93 s cross-encoder. Two costs, two counters.
    """
    _publish_store(built)
    stats = Statistics()
    ctx = _up(built, _cfg_with_mirror(built, None), stats=stats)
    try:
        assert retrieve(ctx, "berth0", top_k=5)
        # presence, not magnitude: a fixture fetch is microseconds and a wall-clock threshold
        # in a unit test measures the machine. The defect was that one of these two ids was
        # emitted by nothing at all.
        emitted = {record.metric_id for record in stats.records()}
        assert STAT_STORE_FETCH_SECONDS in emitted
        assert STAT_RERANK_SECONDS in emitted
        assert stats.value(STAT_STORE_FETCH_SECONDS, variant=ctx.index) >= 0.0
        assert stats.value(STAT_RERANK_SECONDS, variant=ctx.index) >= 0.0
    finally:
        ctx.close()
