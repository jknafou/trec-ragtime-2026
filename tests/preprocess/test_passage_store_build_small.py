"""The passage-store build driver, run without touching the cluster.

``LmdbPassageStore.build_from_final`` was written well before anything drove it at corpus
scale, so ``retrieval.display``'s by-id backend had no artefact to read. The driver's
contract is pinned here: it composes what the in-memory store composes, it publishes
atomically under both corpus hashes, a second run is a no-op, and what it writes reads back
through the same call ``retrieval.display`` makes.

The fixture is the reconciled final-table generator from ``test_index_small.py``. It
includes the multi-sentence Chinese document with no separator between its sentences, where
``original`` is a single slice of the document and the two MT renderings are joins. A store
that round-tripped only single-sentence Latin passages would say nothing about that case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import PassageStore
from ragtime.common.io import is_done
from ragtime.common.passage_store import (
    RENDERINGS,
    LmdbPassageStore,
    iter_final_passages,
)
from ragtime.orchestration import saturate
from ragtime.preprocess.passage_store_build import (
    PassageStoreAdapter,
    PassageStoreError,
)
from tests.preprocess.test_index_small import _RECON, _base_docs, _cfg, _tables, _write

pytestmark = pytest.mark.small

_ZH_MULTI_SENTENCE = "zho-docs/2288104#p0"  # two CJK sentences, nothing between them


def _adapter(root: Path, cfg: Any) -> PassageStoreAdapter:
    """The adapter under test, keyed exactly as the fixture tables are written (pack=None)."""
    return PassageStoreAdapter(recon_hash=_RECON, pack_hash=None, base=str(root))


def _drive(root: Path, cfg: Any, adapter: PassageStoreAdapter) -> list[Path]:
    """Seed the real work-queue and run its one shard through claim -> work -> validate."""
    wq = saturate.queue_for(cfg, adapter, base=root)
    assert saturate.seed(cfg, adapter, wq) == 1, "the store build is exactly one shard"
    ctx = adapter.bringup(cfg)
    receipts: list[Path] = []
    while True:
        shard = saturate.workqueue.claim(wq.pending, wq.running)
        if shard is None:
            break
        out = adapter.work(ctx, shard)
        assert adapter.validate(out), "validate rejected a store it had just built"
        saturate.workqueue.mark_done(shard, wq.done, "corpus")
        receipts.append(out)
    return receipts


@pytest.fixture
def corpus(tmp_path: Path):
    cfg = _cfg(tmp_path)
    layout = _write(tmp_path, _tables(_base_docs()), cfg)
    return cfg, layout


def test_smr21_the_adapter_store_round_trips_into_display_byte_identically(
    tmp_path: Path, corpus
) -> None:
    """The built LMDB store answers what the in-memory store answers.

    All three renderings, every passage, including the multi-sentence Chinese one. The read
    goes through :func:`ragtime.retrieval.display.display`, which is the real consumer,
    rather than through the store's own API.
    """
    from ragtime.retrieval.display import display

    cfg, layout = corpus
    adapter = _adapter(tmp_path, cfg)
    _drive(tmp_path, cfg, adapter)

    expected = PassageStore.from_records(
        iter_final_passages(layout, _RECON, pack_hash=None)
    )
    store = LmdbPassageStore(adapter.out_path(cfg))
    try:
        ctx = _display_ctx(store)
        ids = sorted(r["passage_id"] for r in iter_final_passages(layout, _RECON, pack_hash=None))
        assert _ZH_MULTI_SENTENCE in ids
        for rendering in RENDERINGS:
            got = display(ctx, ids, rendering)
            assert [text for _, text in got] == [expected.render(pid, rendering) for pid in ids]
        # The CJK case: the native rendering is one slice with no fabricated separator,
        # and the two MT renderings join independently translated English sentences.
        native = expected.render(_ZH_MULTI_SENTENCE, "original")
        assert " " not in native
        assert " " in expected.render(_ZH_MULTI_SENTENCE, "omt")
        assert store.render(_ZH_MULTI_SENTENCE, "original") == native
    finally:
        store.close()


def test_smr21_the_adapter_stores_what_build_from_final_stores(tmp_path: Path, corpus) -> None:
    """The driver counts records on their way past; it does not compose them differently."""
    cfg, layout = corpus
    adapter = _adapter(tmp_path, cfg)
    _drive(tmp_path, cfg, adapter)

    reference = tmp_path / "reference.lmdb"
    ref = LmdbPassageStore.build_from_final(reference, layout, _RECON, pack_hash=None)
    mine = LmdbPassageStore(adapter.out_path(cfg))
    try:
        for record in iter_final_passages(layout, _RECON, pack_hash=None):
            pid = record["passage_id"]
            assert mine.passage(pid) == ref.passage(pid)
            for rendering in RENDERINGS:
                assert mine.render(pid, rendering) == ref.render(pid, rendering)
    finally:
        ref.close()
        mine.close()


def test_smr21_publication_is_atomic_keyed_by_both_hashes_and_resumable(
    tmp_path: Path, corpus
) -> None:
    """Publication is temp then rename then ``_SUCCESS``, and a second run recomputes nothing."""
    cfg, layout = corpus
    adapter = _adapter(tmp_path, cfg)
    receipts = _drive(tmp_path, cfg, adapter)

    final = adapter.out_path(cfg)
    assert final == layout.passage_store_path(_RECON, None)
    assert final.is_dir() and is_done(final)
    assert not list(final.parent.glob("*.tmp-*")), "a temp env survived publication"
    first = json.loads(receipts[0].read_text(encoding="utf-8").strip())
    assert first["passages"] == len(list(iter_final_passages(layout, _RECON, pack_hash=None)))
    assert first["resumed"] is False
    assert adapter.merge(cfg, receipts) == final

    # Re-claiming the same shard is a no-op: the artefact tree is the checkpoint.
    mtimes = {p: p.stat().st_mtime_ns for p in sorted(final.rglob("*"))}
    again = PassageStoreAdapter(recon_hash=_RECON, pack_hash=None, base=str(tmp_path))
    second = _drive(tmp_path, cfg, again)
    assert json.loads(second[0].read_text(encoding="utf-8").strip())["resumed"] is True
    assert {p: p.stat().st_mtime_ns for p in sorted(final.rglob("*"))} == mtimes

    # A different packing addresses a different store, not this one.
    other = PassageStoreAdapter(recon_hash=_RECON, pack_hash="deadbeefcafe", base=str(tmp_path))
    assert other.out_path(cfg) != final
    assert other.stage != adapter.stage


def test_smr21_the_adapter_matches_the_stage_adapter_protocol(tmp_path: Path, corpus) -> None:
    """It is a ``saturate.StageAdapter``, not a bespoke driver: the same five methods."""
    cfg, _ = corpus
    adapter = _adapter(tmp_path, cfg)
    for name in ("bringup", "shards", "work", "validate", "merge"):
        assert callable(getattr(adapter, name))
    assert isinstance(adapter.stage, str) and adapter.stage.startswith("passage_store_")
    assert adapter.template == "workqueue_worker_cpu.sbatch"
    shards = list(adapter.shards(cfg))
    assert len(shards) == 1 and shards[0].payload["reconcile_hash"] == _RECON


def test_smr21_a_worker_that_published_elsewhere_fails_the_merge(tmp_path: Path, corpus) -> None:
    """Two Layout roots inside one family would diverge the corpus silently, so it raises."""
    cfg, _ = corpus
    adapter = _adapter(tmp_path, cfg)
    receipts = _drive(tmp_path, cfg, adapter)
    doctored = tmp_path / "doctored.jsonl"
    record = json.loads(receipts[0].read_text(encoding="utf-8").strip())
    record["store_path"] = str(tmp_path / "somewhere-else" / "passages.lmdb")
    doctored.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(PassageStoreError, match="Layout roots disagree"):
        adapter.merge(cfg, [doctored])


def _display_ctx(store: Any):
    """The minimum a :func:`display` call reads: a store, a default rendering, a counter bus."""
    import types

    from ragtime.common import Statistics

    return types.SimpleNamespace(
        passage_store=store, passage_lang="original", stats=Statistics()
    )
