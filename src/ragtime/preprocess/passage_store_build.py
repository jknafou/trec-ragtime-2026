"""Corpus-scale build driver for the by-id rendering store.

:meth:`common.passage_store.LmdbPassageStore.build_from_final` is the primitive: it streams
``documents.parquet``, ``final/<recon12>/{sentences,translations/*}`` and one packing's
``passages.parquet``, composes all three renderings per passage and bulk-loads them into an
LMDB environment. This module is the driver that calls it once, corpus-wide, under the same
resume and atomicity contract as every other corpus substage, so that ``retrieval.display``
has an artifact to read. It adds no storage mechanism, no second composition rule and no
second path scheme.

There is exactly one shard because an LMDB environment has one writer: sharding would mean N
environments and a merge that either concatenates them (LMDB has no such operation) or
rewrites every record. The stage is still a ``saturate.StageAdapter`` rather than a script,
which is what makes it claimable, heartbeated, poison-tracked and resumable like its siblings.

It is keyed by both corpus hashes, at ``Layout.passage_store_path(recon_hash, pack_hash)``:
the store is a derived copy of exactly the tables beside it, so a store built from one packing
must not be servable at a path another packing also names. Publication applies ``common.io``'s
temp-rename-``_SUCCESS`` contract to a directory, reusing
:func:`~ragtime.preprocess.index._publish_dir`.

``orchestration.cli._CORPUS_SUBSTAGES`` does not name this stage, so it is driven explicitly;
everything else about it (the queue, the claim, the marker) is the shared machinery.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import Layout, Statistics, get_logger
from ragtime.common.io import DEFAULT_MAP_SIZE, is_done, read_jsonl, write_jsonl
from ragtime.common.passage_store import (
    RENDERINGS,
    LmdbPassageStore,
    iter_final_passages,
)
from ragtime.config import all_hashes
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family

from .index import _publish_dir
from .packing import packing_hash
from .reconcile import reconcile_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Mapping, Sequence

    from ragtime.config import RunConfig

__all__ = [
    "STAT_PASSAGES",
    "STAT_RENDERING_BYTES",
    "PassageStoreAdapter",
    "PassageStoreError",
    "passage_store_shard_name",
]

_log = get_logger("preprocess.passage_store")

_DEFAULT_BASE = "runs"
_HASH_DIRLEN = 12

#: Passages written into the store; the store is corpus-wide, so this is not sliced.
STAT_PASSAGES = "passage_store.passages"
#: Composed text bytes per rendering (the store's own size driver), sliced by ``variant``.
STAT_RENDERING_BYTES = "passage_store.rendering_bytes"

#: Records per LMDB write transaction, named here so the one knob this stage passes
#: down is visible rather than implicit.
_PUT_BATCH = 10_000


class PassageStoreError(RuntimeError):
    """The store could not be built, or the built store does not read back."""


def passage_store_shard_name(recon_hash: str, pack_hash: str | None) -> str:
    """Return the single shard's queue filename, stable across re-seeds."""
    return (
        f"passage_store_{recon_hash[:_HASH_DIRLEN]}_"
        f"{(pack_hash or 'legacy')[:_HASH_DIRLEN]}"
    )


@dataclass(slots=True)
class _Shard:
    """One claimable work unit, in ``saturate``'s ShardSpec shape."""

    name: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class _StoreCtx:
    """The worker's resident context: paths and knobs only, no model and no GPU."""

    layout: Layout
    recon_hash: str
    pack_hash: str | None
    renderings: tuple[str, ...]
    store_path: Path
    map_size: int


@dataclass(slots=True)
class PassageStoreAdapter:
    """The by-id passage store as one ``saturate.StageAdapter``, CPU-only.

    ``stage`` embeds both corpus hashes, so a different inventory or a different packing
    resolves to a fresh queue subtree and a fresh artifact node instead of finding the
    previous build's ``_SUCCESS``.
    """

    recon_hash: str
    pack_hash: str | None
    renderings: tuple[str, ...] = RENDERINGS
    base: str | None = None
    map_size: int = DEFAULT_MAP_SIZE
    template: str = "workqueue_worker_cpu.sbatch"
    stats: Statistics = field(default_factory=Statistics)
    stage: str = field(init=False)

    def __post_init__(self) -> None:
        self.stage = (
            f"passage_store_{self.recon_hash[:_HASH_DIRLEN]}_"
            f"{(self.pack_hash or 'legacy')[:_HASH_DIRLEN]}"
        )

    @classmethod
    def for_config(
        cls, cfg: RunConfig, *, base: str | None = None
    ) -> PassageStoreAdapter:
        """Build the adapter for ``cfg``; the sanctioned construction path.

        It resolves both corpus keys through their owners (``reconcile.reconcile_hash`` and
        ``packing.packing_hash``), which is why this driver lives in ``preprocess``: the
        ``retrieval`` package that consumes the store may not import either of them.
        """
        return cls(
            recon_hash=reconcile_hash(cfg),
            pack_hash=packing_hash(cfg),
            base=base,
        )

    # -- paths -------------------------------------------------------------- #
    def _layout(self, cfg: RunConfig) -> Layout:
        root = self.base if self.base is not None else _DEFAULT_BASE
        return Layout(
            run_dir=root,
            base=root,
            family=run_family(cfg),
            chunker_hash=all_hashes(cfg)["chunker"],
        )

    def out_path(self, cfg: RunConfig) -> Path:
        """Return ``<final_dir>/passages/<pack12>/passages.lmdb``, resolved through Layout."""
        return self._layout(cfg).passage_store_path(self.recon_hash, self.pack_hash)

    # -- seed --------------------------------------------------------------- #
    def shards(self, cfg: RunConfig) -> Iterator[_Shard]:
        """Yield the single shard: the whole corpus."""
        yield _Shard(
            name=passage_store_shard_name(self.recon_hash, self.pack_hash),
            payload={
                "reconcile_hash": self.recon_hash,
                "packing_hash": self.pack_hash,
                "renderings": list(self.renderings),
            },
        )

    # -- work --------------------------------------------------------------- #
    def bringup(self, cfg: RunConfig) -> _StoreCtx:
        """Resolve the paths this build reads and writes. Loads nothing."""
        layout = self._layout(cfg)
        return _StoreCtx(
            layout=layout,
            recon_hash=self.recon_hash,
            pack_hash=self.pack_hash,
            renderings=tuple(self.renderings),
            store_path=self.out_path(cfg),
            map_size=self.map_size,
        )

    def _records(self, ctx: _StoreCtx) -> Iterator[Mapping[str, Any]]:
        """Yield the composition stream, counting records on their way past.

        ``build_from_final`` is ``build(path, iter_final_passages(...))``; calling the two
        halves separately records the counts without a second pass over the corpus. The
        composition itself is unchanged, which a test pins by comparing this stage's store
        against ``build_from_final``'s over one fixture.
        """
        for record in iter_final_passages(
            ctx.layout,
            ctx.recon_hash,
            pack_hash=ctx.pack_hash,
            renderings=ctx.renderings,
        ):
            self.stats.emit(STAT_PASSAGES, 1.0)
            for variant in ctx.renderings:
                text = record.get(variant) or ""
                self.stats.emit(
                    STAT_RENDERING_BYTES, float(len(text.encode("utf-8"))), variant=variant
                )
            yield record

    def work(self, ctx: _StoreCtx, shard: Path) -> Path:
        """Build (or skip) the store, then write this shard's receipt.

        Skip-if-done is checked against the store's own ``_SUCCESS`` rather than the receipt's:
        the store is the artifact, the receipt is bookkeeping, and a re-claimed shard whose
        store is already published must not re-derive it.
        """
        final = ctx.store_path
        if is_done(final):
            _log.info("preprocess.passage_store.skip", path=str(final))
            return self._receipt(shard, final, self._census(final), resumed=True)

        tmp = final.with_name(f"{final.name}.tmp-{os.getpid()}")
        if tmp.exists():  # a crashed attempt: incomplete by definition
            import shutil

            shutil.rmtree(tmp)
        store = LmdbPassageStore.build(
            tmp, self._records(ctx), map_size=ctx.map_size, batch_size=_PUT_BATCH
        )
        store.close()
        _publish_dir(tmp, final)
        census = self._census(final)
        if census["passages"] <= 0:
            raise PassageStoreError(
                f"{final}: the built store holds no passages; the final tables under "
                f"recon={ctx.recon_hash[:12]} pack={str(ctx.pack_hash)[:12]} are empty or "
                "were never published"
            )
        _log.info(
            "preprocess.passage_store.done",
            path=str(final),
            passages=census["passages"],
            bytes=census["bytes"],
        )
        return self._receipt(shard, final, census, resumed=False)

    def _census(self, final: Path) -> dict[str, Any]:
        """Read the built environment's own entry count and size."""
        env = LmdbPassageStore(final)
        try:
            entries = int(env._env.stat()["entries"])
        finally:
            env.close()
        return {
            "passages": entries,
            "bytes": sum(p.stat().st_size for p in final.rglob("*") if p.is_file()),
        }

    def _receipt(
        self, shard: Path, final: Path, census: Mapping[str, Any], *, resumed: bool
    ) -> Path:
        out = saturate.shard_out_path(shard)
        write_jsonl(
            out,
            [
                {
                    "store_path": str(final),
                    "reconcile_hash": self.recon_hash,
                    "packing_hash": self.pack_hash,
                    "renderings": list(self.renderings),
                    "passages": int(census["passages"]),
                    "bytes": int(census["bytes"]),
                    "resumed": bool(resumed),
                }
            ],
            skip_if_done=False,
        )
        return out

    # -- validate ----------------------------------------------------------- #
    def validate(self, path: Path) -> bool:
        """Check the store is marked done, published, non-empty and readable back.

        A row count alone would pass for an environment whose blobs cannot be decoded, so this
        also re-opens the published store and renders one passage in every rendering, the same
        call ``retrieval.display`` makes.
        """
        if not (path.exists() and is_done(path)):
            return False
        record = read_jsonl(path)[0]
        final = Path(record["store_path"])
        if not (final.exists() and is_done(final) and int(record.get("passages") or 0) > 0):
            _log.warning("preprocess.passage_store.validate.unpublished", path=str(final))
            return False
        try:
            store = LmdbPassageStore(final)
        except Exception:  # noqa: BLE001 - any open failure means 'not a store'
            _log.warning("preprocess.passage_store.validate.unopenable", path=str(final))
            return False
        try:
            with store._env.begin() as txn:
                cursor = txn.cursor()
                if not cursor.first():
                    return False
                passage_id = cursor.key().decode("utf-8")
            for variant in record.get("renderings") or RENDERINGS:
                if not isinstance(store.render(passage_id, variant), str):
                    return False
        except Exception:  # noqa: BLE001 - any read failure means 'not a store'
            _log.warning("preprocess.passage_store.validate.unreadable", path=str(final))
            return False
        finally:
            store.close()
        return True

    # -- merge -------------------------------------------------------------- #
    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> Path:
        """Assert the one shard published the store this config names.

        The store is the family artifact and there are no per-shard pieces to concatenate, so
        this stage's ``merge`` only checks that the artifact on disk is the one the config
        addresses. A receipt naming a different path would mean the worker and the submitter
        resolved two different ``Layout`` roots.
        """
        expected = self.out_path(cfg)
        receipts = [json.loads(Path(p).read_text(encoding="utf-8").strip()) for p in shard_paths]
        if not receipts:
            raise PassageStoreError("no passage-store shard output to merge")
        for record in receipts:
            if Path(record["store_path"]) != expected:
                raise PassageStoreError(
                    f"a worker published the store at {record['store_path']} but this config "
                    f"resolves {expected}; the two Layout roots disagree"
                )
        if not is_done(expected):
            raise PassageStoreError(f"{expected}: not published (no _SUCCESS marker)")
        _log.info(
            "preprocess.passage_store.merge",
            path=str(expected),
            passages=int(receipts[0].get("passages") or 0),
        )
        return expected
