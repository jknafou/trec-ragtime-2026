"""Three index legs over three renderings, on a bounded shard of the real corpus.

There are no qrels, no dev set and no training data. Every test here is tagged
``[liveness]`` or ``[consistency]`` in its docstring and asks whether the built index
returns something structurally sound. None is a quality measurement, nothing here computes
nDCG or recall against an assumed relevance judgement, and nothing here should be turned
into one later.

The shard is bounded: :data:`_DOCS_PER_LANG` documents per language are carved out of the
real, already-built ``final/`` tables, so what is exercised is real reconciled data, with
real fused and split-back sentences, both translation tiers, and real Chinese passages with
no separator, rather than a synthetic subset. The corpus-scale build, 9,405,925 passages in
three renderings, is a separate operation: FL09 measures the per-leg cost on the slice and
extrapolates rather than attempting it.

What only this file can show is the three engines going through their own
publish-then-reopen path with the real checkpoints. The sparse leg's save and load are
asymmetric, since ``save`` takes a stem and appends ``.index.seismic`` while ``load`` needs
the full filename, and passing one string to both once shipped: the leg got its ``_SUCCESS``
marker and could never be reopened. So every leg here is published and then reopened from
disk, never queried out of the writer that built it.

It runs on a GPU node with the ``index`` extra installed, and skips loudly when the real
corpus or an engine library is absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout
from ragtime.common import io as common_io
from ragtime.common.io import (
    is_done,
    iter_parquet_batches,
    parquet_row_group_sizes,
    write_parquet_stream,
)
from ragtime.common.passage_store import RENDERINGS, iter_final_passages
from ragtime.common.schemas import (
    document_arrow_schema,
    final_passage_arrow_schema,
    sentence_arrow_schema,
    translation_final_arrow_schema,
)
from ragtime.config import all_hashes, load
from ragtime.orchestration import saturate
from ragtime.orchestration.cli import artifact_root
from ragtime.orchestration.run_identity import run_family
from ragtime.preprocess import index as index_mod
from ragtime.preprocess import packing as packing_mod
from ragtime.preprocess.index import (
    DENSE_LEG,
    LATE_INTERACTION_LEG,
    LEGS,
    SHARED_LANG,
    SPARSE_LEG,
    IndexAdapter,
    IndexIntegrityError,
    IndexShardSpec,
    default_legs,
    index_build_options,
    index_hash,
    kendall_tau_b,
    open_shard,
    query_leg,
    rbo,
    shard_agreement,
)
from ragtime.preprocess.reconcile import reconcile_hash

pytestmark = pytest.mark.full

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "config" / "e2e-original.yml"
#: The real artifact root. This file reads the corpus the preprocess stages published and
#: never writes into it; everything it creates lands under pytest's own temporary root.
#: It is resolved through `orchestration.cli.artifact_root` rather than a second hardcoded
#: "runs", because the store lives wherever `execution.artifact_root` or
#: `$RAGTIME_ARTIFACT_ROOT` says, and a test that looked only in the repo would skip itself
#: green the day the root moved. `_REPO_ROOT / <absolute>` is the absolute path itself, so
#: one expression covers the repo-relative default and an off-repo root alike.
_SOURCE_BASE = _REPO_ROOT / artifact_root(load(_CONFIG))

#: The bounded shard's size, in the same range the other corpus-scale tests use: the chunk
#: one takes 50 documents per language, the alignment and boilerplate measurements 200. Here
#: 200 per language is 800 documents, enough for a real PLAID codec to train on every shard
#: and small enough that the whole three-leg, three-rendering build finishes inside one GPU
#: allocation. This is not a scale test.
_DOCS_PER_LANG = 200

#: Sampled passages per (leg, rendering) in the self-retrieval battery.
_SELF_RETRIEVAL_SAMPLE = 5
#: Depth of every rank-agreement measurement. A window, not a quality threshold.
_TOP_K = 10

#: Corpus-scale extrapolation targets, measured at reconciliation.
_CORPUS_PASSAGES = 9_405_925
_CORPUS_RENDERINGS = 3

#: The measured facts this module reports, printed by the last test so the numbers land in
#: the job's own log rather than in a file nobody reads.
_MEASURED: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Bounded-shard carving: a document-atomic prefix of every real final/ table.
#
# The tables are co-ordered by document, which is what `iter_final_passages`'s co-walk join
# relies on, and laid out in contiguous per-language blocks: en, ru, es then zh in all five
# tables. So the first N documents of each language is a contiguous row range in every one
# of them. The range is found by binary-searching the Parquet footer's row groups, one row
# read per probe and about 13 probes for a 4,436-group table, rather than by scanning the
# 88.7 M-row sentence and translation tables. Only the small `documents.parquet`, at 201
# groups, is walked group by group, to learn the order the languages appear in.
#
# `preprocess.spine.plan_shards` is not reused here. It makes one full column pass over the
# work table to place its boundaries, which is the right cost for a production shard plan
# over a table about to be consumed entirely, and the wrong cost for lifting 800 documents
# out of the head of a 14 GB corpus: a first-row-per-group walk of all five tables took
# 5 min 51 s, where the binary search does the same job in seconds.
#
# The document ids are opaque, so language is always read from a table's own key column:
# `lang` on the corpus tables, and `source_lang` on the translation tables, which carries
# FLORES codes and interleaves zho_Hans and zho_Hant inside the one `zh` block.
# --------------------------------------------------------------------------- #
#: The key column carrying language, per table, and the FLORES to corpus-tag folding.
_LANG_COLUMN_DOCS = "lang"
_LANG_COLUMN_TRANSLATIONS = "source_lang"
_FLORES_PREFIX = {"eng": "en", "rus": "ru", "spa": "es", "zho": "zh"}


def _lang_of(row: dict[str, Any], key: str) -> str:
    """The corpus language tag of a row, folding FLORES codes (and zh Hans/Hant) onto it."""
    value = str(row.get(key) or "")
    return _FLORES_PREFIX.get(value[:3], value)


def _first_row_of_group(path: Path, group: int, columns: list[str]) -> dict[str, Any] | None:
    for batch in iter_parquet_batches(path, columns=columns, batch_size=1, row_groups=[group]):
        if batch:
            return batch[0]
    return None


def _lang_order(path: Path, key: str) -> list[str]:
    """The distinct language blocks in this table's own row order (one probe per group)."""
    order: list[str] = []
    for group in range(len(parquet_row_group_sizes(path))):
        row = _first_row_of_group(path, group, [key])
        if row is None:
            continue
        lang = _lang_of(row, key)
        if lang not in order:
            order.append(lang)
    return order


def _lang_start_group(path: Path, key: str, order: list[str], lang: str) -> int:
    """First row group that can hold ``lang``'s opening row, by binary search on the footer.

    It returns one group earlier than the match where it can, because a block may start in
    the middle of a row group and the caller drops the leading rows positionally. Erring
    early costs a few scanned and discarded rows. Erring late would silently truncate the
    slice, which is also why the caller asserts the document count it got.
    """
    target = order.index(lang)
    groups = len(parquet_row_group_sizes(path))
    lo, hi = 0, groups - 1
    while lo < hi:
        mid = (lo + hi) // 2
        row = _first_row_of_group(path, mid, [key])
        seen = order.index(_lang_of(row, key)) if row and _lang_of(row, key) in order else -1
        if seen >= target:
            hi = mid
        else:
            lo = mid + 1
    return max(0, lo - 1)


def _scan_lang_prefix(path: Path, wanted: set[str], start_group: int) -> list[dict[str, Any]]:
    """The contiguous run of rows whose ``document_id`` is in ``wanted``.

    ``wanted`` is by construction the first ``_DOCS_PER_LANG`` documents of one language
    block, so its rows are a prefix of that block. A non-member seen after the run has
    started means the block's wanted rows are exhausted.
    """
    rows: list[dict[str, Any]] = []
    started = False
    for group in range(start_group, len(parquet_row_group_sizes(path))):
        for batch in iter_parquet_batches(path, row_groups=[group]):
            for row in batch:
                if row["document_id"] in wanted:
                    rows.append(row)
                    started = True
                elif started:
                    return rows
    return rows


def _take_documents(path: Path, order: list[str], lang: str, n: int) -> list[dict[str, Any]]:
    """The first ``n`` documents of ``lang``, in table order (rows, text included)."""
    taken: list[dict[str, Any]] = []
    groups = len(parquet_row_group_sizes(path))
    for group in range(_lang_start_group(path, _LANG_COLUMN_DOCS, order, lang), groups):
        for batch in iter_parquet_batches(path, row_groups=[group]):
            for row in batch:
                if _lang_of(row, _LANG_COLUMN_DOCS) != lang:
                    if taken:
                        return taken
                    continue
                taken.append(row)
                if len(taken) >= n:
                    return taken
    return taken


@dataclass(slots=True)
class _Bounded:
    """The carved bounded shard: the config, the staged tables, and what is in them."""

    cfg: Any
    root: Path
    layout: Layout
    recon: str
    #: Which packing of that inventory the slice carries, resolved from the config the way
    #: the shipped adapter does. Every read of a passage table in this file goes through it.
    #: ``None`` names the older table from before the ``packing`` block, a different corpus
    #: packed to a wider content budget; reading that one indexes passages the shipped recipe
    #: does not produce.
    pack: str
    documents: list[dict[str, Any]]
    records: dict[str, dict[str, Any]]
    ids_by_lang: dict[str, set[str]]
    docs_by_lang: dict[str, set[str]]

    @property
    def all_ids(self) -> set[str]:
        return {pid for ids in self.ids_by_lang.values() for pid in ids}


def _carve(root: Path, cfg: Any) -> _Bounded:
    """Stage ``_DOCS_PER_LANG`` documents/language of the real corpus under ``root``."""
    recon = reconcile_hash(cfg)
    # The shipped packing, resolved from the config, never ``None``. ``None`` names the older
    # table from before the ``packing`` block, packed to a 510-token content budget against
    # the 509 this recipe ships. Carving from it hands every leg passages one token too long
    # for the late-interaction document path, since 510 plus two specials plus the
    # ``[unused1]`` marker is 513 against a 512 window: the truncation the re-pack removes.
    pack = packing_mod.packing_hash(cfg)
    family = run_family(cfg)
    chunker = all_hashes(cfg)["chunker"]
    src = Layout(run_dir=_SOURCE_BASE, base=_SOURCE_BASE, family=family, chunker_hash=chunker)
    dst = Layout(run_dir=root, base=root, family=family, chunker_hash=chunker)

    src_documents = src.documents_path()
    if not (is_done(src_documents) and is_done(src.final_passages_path(recon, pack))):
        pytest.skip(
            f"the real corpus node is absent at {src.final_dir(recon)} (packing {pack[:12]}) "
            "- these tests read an existing preprocess build, they never make one"
        )

    order = _lang_order(src_documents, _LANG_COLUMN_DOCS)
    assert sorted(order) == ["en", "es", "ru", "zh"], order
    documents: list[dict[str, Any]] = []
    docs_by_lang: dict[str, set[str]] = {}
    for lang in order:
        taken = _take_documents(src_documents, order, lang, _DOCS_PER_LANG)
        assert len(taken) == _DOCS_PER_LANG, (lang, len(taken))
        docs_by_lang[lang] = {row["document_id"] for row in taken}
        documents.extend(taken)
    write_parquet_stream(dst.documents_path(), documents, schema=document_arrow_schema())

    # ``total`` is the set of languages every one of whose documents must appear in the
    # table. It stopped being "all four" for the TRANSLATION tables when the English dedup
    # landed (``reconcile.store_identity_translations: false``, ``recon12`` f308301501d9):
    # English is identity pass-through, so no row is stored for it at all and every English
    # rendering resolves to the native ``documents.text`` slice. Expressing that as a
    # per-language set, rather than relaxing the check, keeps it strict where a short slice
    # would still be silent data loss, and turns English into its own assertion below:
    # exactly zero rows, which is stronger than "at most the sampled documents".
    tables: list[tuple[Path, Path, Any, str, frozenset[str]]] = [
        (
            src.final_sentences_path(recon),
            dst.final_sentences_path(recon),
            sentence_arrow_schema(),
            _LANG_COLUMN_DOCS,
            frozenset(order),  # totality: every document has at least one final sentence
        ),
        (
            src.final_passages_path(recon, pack),
            dst.final_passages_path(recon, pack),
            final_passage_arrow_schema(),
            _LANG_COLUMN_DOCS,
            frozenset(order),
        ),
    ]
    for variant in RENDERINGS:
        if variant == "original":
            continue  # the native rendering is a slice of documents.text, not a table
        tables.append(
            (
                src.final_translations_path(recon, variant),
                dst.final_translations_path(recon, variant),
                translation_final_arrow_schema(),
                _LANG_COLUMN_TRANSLATIONS,
                # One row per non-English final sentence; English is stored nowhere.
                frozenset(lang for lang in order if lang != SHARED_LANG),
            )
        )
    for src_path, dst_path, schema, key, total in tables:
        rows: list[dict[str, Any]] = []
        for lang in order:
            found = _scan_lang_prefix(
                src_path,
                docs_by_lang[lang],
                _lang_start_group(src_path, key, order, lang),
            )
            covered = {row["document_id"] for row in found}
            assert covered <= docs_by_lang[lang], (src_path, lang)
            if lang in total:
                # A short slice here would be silent data loss: the carve would hand the
                # index a corpus whose passages have no sentences (or no translation), and
                # `iter_final_passages` would then blame the corpus build, not the fixture.
                assert covered == docs_by_lang[lang], (
                    src_path.name,
                    lang,
                    len(covered),
                    _DOCS_PER_LANG,
                )
            else:
                # The one excluded case is English in a translation table. A row here would
                # mean an identity translation reappeared, which is what
                # `index.STAT_EN_DIVERGENCE` watches for, so this is asserted empty rather
                # than merely allowed to be short.
                assert not covered, (src_path.name, lang, len(covered))
            rows.extend(found)
        assert rows, src_path
        write_parquet_stream(dst_path, rows, schema=schema)

    # Compose once. Every test below reads this one materialised view of the slice, so a
    # co-walk failure surfaces in the fixture rather than inside an assertion.
    records = {r["passage_id"]: r for r in iter_final_passages(dst, recon, pack_hash=pack)}
    ids_by_lang: dict[str, set[str]] = {}
    for record in records.values():
        ids_by_lang.setdefault(record["lang"], set()).add(record["passage_id"])
    assert sorted(ids_by_lang) == ["en", "es", "ru", "zh"], sorted(ids_by_lang)
    return _Bounded(
        cfg=cfg,
        root=root,
        layout=dst,
        recon=recon,
        pack=pack,
        documents=documents,
        records=records,
        ids_by_lang=ids_by_lang,
        docs_by_lang=docs_by_lang,
    )


# --------------------------------------------------------------------------- #
# Clients + the timing wrapper.
# --------------------------------------------------------------------------- #
@dataclass
class _TimedLeg:
    """The real leg, wrapped only to measure it. Every call delegates and nothing is faked.

    Per-leg build time is one of the numbers reported here. GPU utilisation depends on the
    workload, so the translate stage's numbers do not carry over to the encode-heavy index
    build, and no counter records wall time. Delegating keeps the leg names exactly
    :data:`LEGS`, so ``IndexAdapter``'s leg-set check still rejects a fourth or a missing leg.
    """

    inner: Any
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.inner.name

    def _timed(self, phase: str, fn, *args):
        start = time.monotonic()
        try:
            return fn(*args)
        finally:
            key = f"{self.inner.name}.{phase}"
            self.seconds[key] = self.seconds.get(key, 0.0) + (time.monotonic() - start)

    def encode_docs(self, ctx, texts):
        return self._timed("encode", self.inner.encode_docs, ctx, texts)

    def encode_query(self, ctx, query):
        return self.inner.encode_query(ctx, query)

    def writer(self, out_dir: Path, ctx):
        return _TimedWriter(self.inner.writer(out_dir, ctx), self)

    def open(self, leg_dir: Path, ctx):
        return self._timed("open", self.inner.open, leg_dir, ctx)

    def search(self, reader, ctx, query_rep, top_k):
        return self.inner.search(reader, ctx, query_rep, top_k)


@dataclass
class _TimedWriter:
    inner: Any
    leg: _TimedLeg

    def add(self, ordinals, reps) -> None:
        self.leg._timed("write", self.inner.add, ordinals, reps)

    def finish(self) -> None:
        self.leg._timed("finish", self.inner.finish)

    def __getattr__(self, item):  # e.g. the sparse writer's ``empty`` counter
        return getattr(self.inner, item)


def _timed_legs() -> tuple[_TimedLeg, ...]:
    return tuple(_TimedLeg(inner=leg) for leg in default_legs())


def _rep_digest(rep: Any) -> bytes:
    """One encoded representation, canonicalised to bytes exactly rather than approximately.

    Two shapes reach this: a sparse ``{dim_id: weight}`` map, and an array, which is either
    the dense vector or the late-interaction token matrix. Both are hashed from their own
    values, so "the same vectors in the same order" is a byte comparison, not a tolerance.
    """
    import numpy as np

    if isinstance(rep, dict):
        return repr(sorted((str(k), float(v)) for k, v in rep.items())).encode()
    arr = np.asarray(rep)
    return f"{arr.dtype}{arr.shape}".encode() + arr.tobytes()


@dataclass
class _DigestLeg:
    """The real leg, wrapped only to hash what it hands the writer, in order.

    FL11 needs this for the one leg whose stored bytes cannot be reproduced, listed in
    :data:`index.UNSEEDABLE_LEGS`. Seismic's clustering takes no seed, so its index file
    differs between two builds, and what the deviation rests on is that the leg's input does
    not. That is only worth something measured, so this keeps a running sha256 over every
    ``(ordinals, reps)`` the build produces.
    """

    inner: Any
    _hash: Any = field(default_factory=lambda: hashlib.sha256())

    @property
    def name(self) -> str:
        return self.inner.name

    def reset(self) -> None:
        self._hash = hashlib.sha256()

    def digest(self) -> str:
        return self._hash.hexdigest()

    def encode_docs(self, ctx, texts):
        reps = self.inner.encode_docs(ctx, texts)
        for rep in reps:
            self._hash.update(_rep_digest(rep))
        return reps

    def encode_query(self, ctx, query):
        return self.inner.encode_query(ctx, query)

    def writer(self, out_dir: Path, ctx):
        return self.inner.writer(out_dir, ctx)

    def open(self, leg_dir: Path, ctx):
        return self.inner.open(leg_dir, ctx)

    def search(self, reader, ctx, query_rep, top_k):
        return self.inner.search(reader, ctx, query_rep, top_k)


@dataclass(slots=True)
class _Built:
    """The published bounded index plus everything the assertions read back."""

    bounded: _Bounded
    first_adapter: IndexAdapter
    adapter: IndexAdapter
    ctx: Any
    legs: tuple[_TimedLeg, ...]
    receipts: list[Path]
    manifest: dict[str, Any]

    def stat(self, metric: str, **slices: Any) -> float:
        """A counter summed over both adapters, since the build was split between them."""
        return self.first_adapter.stats.value(metric, **slices) + self.adapter.stats.value(
            metric, **slices
        )


def _drain(adapter: IndexAdapter, ctx: Any, wq: Any, limit: int | None = None) -> list[Path]:
    """Claim, work, validate and mark done until the queue empties or ``limit`` shards ran."""
    out: list[Path] = []
    while limit is None or len(out) < limit:
        shard = saturate.workqueue.claim(wq.pending, wq.running)
        if shard is None:
            break
        receipt = adapter.work(ctx, shard)
        assert adapter.validate(receipt), f"validate() rejected {shard.name}"
        saturate.workqueue.mark_done(shard, wq.done, "corpus")
        out.append(receipt)
    return out


@pytest.fixture(scope="session")
def bounded(tmp_path_factory: pytest.TempPathFactory) -> _Bounded:
    cfg = load(_CONFIG)
    root = tmp_path_factory.mktemp("m05d-full")
    start = time.monotonic()
    carved = _carve(root, cfg)
    _MEASURED["slice"] = {
        "docs_per_lang": _DOCS_PER_LANG,
        "reconcile_hash": carved.recon[:12],
        "packing_hash": carved.pack[:12],
        "documents": len(carved.documents),
        "passages": len(carved.records),
        "passages_by_lang": {k: len(v) for k, v in sorted(carved.ids_by_lang.items())},
        "carve_seconds": round(time.monotonic() - start, 1),
    }
    return carved


@pytest.fixture(scope="session")
def built(bounded: _Bounded) -> _Built:
    """Build all ten shards through the real work queue with the real models.

    The drain is split in two. The first adapter stops after four shards, standing in for a
    job array killed mid-flight, and a second, freshly constructed adapter drains the rest
    and publishes the manifest, with new ``Statistics``, an empty id-set cache and nothing
    carried over in memory. That makes resume a property of the artefact tree rather than of
    one long-lived process.
    """
    pytest.importorskip("faiss")
    pytest.importorskip("seismic")
    pytest.importorskip("pylate")
    pytest.importorskip("torch")

    cfg = bounded.cfg
    legs = _timed_legs()
    # ``for_config`` is the only construction path that resolves ``pack_hash`` from the
    # config. Hand-constructing the adapter with ``pack_hash=None`` is how these tests once
    # came to index the older, wider-budget passage table while the shipped code indexed the
    # current one.
    first = IndexAdapter.for_config(cfg, base=bounded.root, legs=legs)
    assert first.pack_hash == bounded.pack
    assert first.idx_hash == index_hash(cfg)
    start = time.monotonic()
    # bringup() itself checks that the three clients are the hashed recipe, taking the dense
    # identity from index_build.config.dense_model rather than the query-time retrieval
    # block, and that the encode window covers packing.pack_budget. So a mis-wired encoder,
    # or one that would truncate the packed passages, fails here rather than under a
    # manifest naming a model that never ran.
    ctx = first.bringup(cfg)
    _MEASURED["bringup_seconds"] = round(time.monotonic() - start, 1)

    wq = saturate.queue_for(cfg, first, base=bounded.root)
    assert saturate.seed(cfg, first, wq) == len(first.shard_specs(cfg)) == 10

    build_start = time.monotonic()
    receipts = _drain(first, ctx, wq, limit=4)
    resumed = IndexAdapter.for_config(cfg, base=bounded.root, legs=legs)
    receipts += _drain(resumed, replace(ctx), wq)
    wall = time.monotonic() - build_start

    manifest = common_io.read_jsonl(resumed.merge(cfg, receipts))[0]
    _MEASURED["build"] = {
        "shards": len(receipts),
        "wall_seconds": round(wall, 1),
        "per_leg_seconds": {
            key: round(value, 1)
            for leg in legs
            for key, value in sorted(leg.seconds.items())
        },
    }
    return _Built(
        bounded=bounded,
        first_adapter=first,
        adapter=resumed,
        ctx=ctx,
        legs=legs,
        receipts=receipts,
        manifest=manifest,
    )


# --------------------------------------------------------------------------- #
# Helpers shared by the assertions.
# --------------------------------------------------------------------------- #
def _shard_dir(built: _Built, variant: str | None, lang: str, part: int = 0) -> Path:
    return built.bounded.layout.index_shard_dir(
        built.bounded.recon, built.ctx.idx_hash, variant, lang, part=part
    )


def _lang_dir(built: _Built, variant: str | None, lang: str) -> Path:
    return built.bounded.layout.index_lang_dir(
        built.bounded.recon, built.ctx.idx_hash, variant, lang
    )


def _cell_legs(shard):
    """``(leg name, entry)`` over every part of one manifest cell.

    The manifest's per-language entry has been part-grained since the shard axis gained
    parts, so a test reading ``shard["legs"]`` would silently see nothing. That is the same
    blindness the census exists to prevent, so the traversal lives in one helper.
    """
    for part in shard["shard_parts"]:
        yield from part["legs"].items()


def _handle(built: _Built, variant: str | None, lang: str):
    return open_shard(_shard_dir(built, variant, lang), built.ctx)


def _longest(built: _Built, lang: str, n: int) -> list[dict[str, Any]]:
    """The ``n`` longest passages of ``lang``: deterministic, and not cherry-picked.

    Longest, because a three-word passage carries almost no lexical signal and a liveness
    probe over one would measure tie-breaking rather than plumbing. Length is a property of
    the slice, fixed before any result is seen, so it cannot be tuned to an outcome.
    """
    rows = [r for r in built.bounded.records.values() if r["lang"] == lang]
    rows.sort(key=lambda r: (-int(r["token_count"] or 0), r["passage_id"]))
    return rows[:n]


def _queries(built: _Built, n: int = 6) -> list[str]:
    """A fixed English query set drawn from the slice's own English passages."""
    return [" ".join(r["original"].split()[:12]) for r in _longest(built, "en", n)]


def _rank_of(hits: list[tuple[str, float]], passage_id: str) -> int:
    for i, (pid, _) in enumerate(hits):
        if pid == passage_id:
            return i
    return -1


def _explain_rank(
    built: _Built,
    leg: str,
    variant: str,
    lang: str,
    record: dict[str, Any],
    hits: list[tuple[str, float]],
    rank: int,
) -> str:
    """Why the winners won: the full failure context for a retrieval miss, unelided.

    A tuple passed as an assertion message is rendered through pytest's ``saferepr``, which
    elides the middle of a long repr, and a passage id is 45 characters, so the elision lands
    on exactly the ids needed. A truncated ``hits[:3]`` once read as "the target is first"
    when it was second. This returns a plain multi-line string, printed verbatim, and states
    per hit the two facts that decide what a miss means: whether the winner's text is the
    same string as the query, and whether it contains it.
    """
    target = str(record[variant])
    lines = [
        (
            f"{leg}/{variant}/{lang}: {record['passage_id']} came back at rank {rank} for its "
            f"OWN composed text (query = {len(target)} chars, "
            f"token_count={record.get('token_count')}, oversized={record.get('is_oversized')})"
        ),
    ]
    for i, (pid, score) in enumerate(hits[:5]):
        other = built.bounded.records.get(pid, {})
        text = str(other.get(variant, ""))
        lines.append(
            f"  #{i} {pid} score={score!r} self={pid == record['passage_id']} "
            f"same_text={text == target} contains_query={target in text and text != target} "
            f"chars={len(text)} tokens={other.get('token_count')} doc={other.get('document_id')}"
        )
    return "\n".join(lines)


def _sha_tree(root: Path) -> dict[str, str]:
    """``relative path -> sha256`` for every file under ``root``, with mtimes removed.

    fast-plaid's ``*.manifest.json`` sidecars are ``{"<file>": {"rows": N, "mtime": T}}``,
    where the mtime belongs to files whose content is hashed here anyway. Two builds of one
    shard are minutes apart, so those files always differ and would report a wall clock as a
    build difference. ``rows`` is kept; only the timestamp goes, and only in files that are
    nothing but an index of timestamps.
    """

    def _stable(path: Path) -> bytes:
        raw = path.read_bytes()
        if not path.name.endswith(".manifest.json"):
            return raw
        record = json.loads(raw)
        for entry in record.values():
            if isinstance(entry, dict):
                entry.pop("mtime", None)
        return json.dumps(record, sort_keys=True).encode()

    return {
        str(path.relative_to(root)): hashlib.sha256(_stable(path)).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _scratch_shard_file(built: _Built, spec: IndexShardSpec, tag: str) -> Path:
    path = built.bounded.root / "scratch" / tag / spec.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.payload()), encoding="utf-8")
    return path


def _scratch_adapter(built: _Built, tag: str, legs: tuple[Any, ...] | None = None):
    """An adapter and context over a scratch index hash, so a probe disturbs no real build.

    ``work`` on a hand-written shard payload is the entry point the queue drives, so a
    scratch build is the shipped path with a different output directory rather than a second
    build routine.

    It is built by ``for_config`` and then re-pointed at a scratch ``idx_hash``, so only the
    output location moves: ``recon_hash`` and ``pack_hash``, which decide the corpus this
    probe reads, stay as they ship.
    """
    idx_hash = hashlib.sha256(tag.encode()).hexdigest()
    legs = legs or built.legs
    adapter = replace(
        IndexAdapter.for_config(built.bounded.cfg, base=built.bounded.root, legs=legs),
        idx_hash=idx_hash,
    )
    return adapter, replace(built.ctx, idx_hash=idx_hash, legs=legs)


def _build_one(built: _Built, spec: IndexShardSpec, tag: str) -> Path:
    adapter, ctx = _scratch_adapter(built, tag)
    adapter.work(ctx, _scratch_shard_file(built, spec, tag))
    return adapter.shard_dir_from_ctx(ctx, spec)


# --------------------------------------------------------------------------- #
# FL17: the dense leg's loading path and its config wiring
# --------------------------------------------------------------------------- #
def test_fl17_bge_m3_dense_returns_1024d_through_the_existing_encoder(
    bounded: _Bounded,
) -> None:
    """FL17 [consistency]: ``Encoder.embed(texts, mode="dense")`` on the real BGE-M3.

    BGE-M3 is usually loaded through FlagEmbedding's tri-output API, and the dense leg uses
    the plain ``sentence-transformers`` dense-only path instead. That is enough, and the
    sparse and ColBERT heads stay out of the dense leg: they belong to the other two legs and
    come from different models, so ``Encoder`` serves ``"dense"`` and nothing else.

    It depends only on ``bounded`` and comes before the build fixture, so it still answers
    the question about the loading path on a run where a later leg's build fails.
    """
    pytest.importorskip("sentence_transformers")
    from ragtime.serving.registry import build_clients

    opts = index_build_options(bounded.cfg)
    encoder = build_clients(bounded.cfg).index_dense  # the registry's client, not a fresh one
    assert encoder.model == opts.dense_model
    texts = [r["original"] for r in list(bounded.records.values())[:4]]
    vectors = encoder.embed(texts, mode="dense")
    assert len(vectors) == 4
    assert len(vectors[0]) == 1024, len(vectors[0])
    norm = float(sum(float(v) * float(v) for v in vectors[0]) ** 0.5)
    assert abs(norm - 1.0) < 1e-2, norm
    assert "FlagEmbedding" not in sys.modules, "the dense path pulled in the tri-output model"
    _MEASURED["dense"] = {
        "model": opts.dense_model,
        "revision": opts.dense_revision,
        "dim": len(vectors[0]),
    }


def test_fl17b_index_client_identities_all_come_from_the_hashed_recipe(
    bounded: _Bounded,
) -> None:
    """[consistency] no stage picks a model; every leg's client comes from the config.

    All three identities live in the hashed ``index_build`` block that keys the index path
    and every leg's ``config_hash``, which is what makes the manifest a true record of what
    encoded the vectors. The dense leg is the case that went wrong: it has to come from
    ``index_build.config.dense_model`` and not from the query-time ``retrieval`` block, which
    no ``e2e-*`` config carries at all.
    """
    from ragtime.serving.registry import build_clients

    opts = index_build_options(bounded.cfg)
    clients = build_clients(bounded.cfg)
    assert clients.index_dense.model == opts.dense_model
    assert clients.index_dense.revision == opts.dense_revision
    assert clients.milco.model == opts.sparse_model
    assert clients.milco.revision == opts.sparse_model_revision
    assert clients.mtd_colbert.checkpoint == opts.spine_model
    assert clients.mtd_colbert.revision == opts.spine_model_revision


# --------------------------------------------------------------------------- #
# FL01: the whole build, end to end, on real reconciled data
# --------------------------------------------------------------------------- #
def test_fl01_nine_leg_variant_builds_over_the_bounded_shard(built: _Built) -> None:
    """FL01 [consistency]: ten shards by three legs, real models, published and validated.

    Nine leg-and-rendering builds plus the one shared English unit: the whole adapter, over
    real reconciled tables, with the real checkpoints.
    """
    specs = built.adapter.shard_specs(built.bounded.cfg)
    assert len(specs) == 10
    assert len(built.receipts) == 10
    for spec in specs:
        shard_dir = _shard_dir(built, spec.variant, spec.source_lang)
        for leg in LEGS:
            leg_dir = shard_dir / leg
            assert is_done(leg_dir), f"{spec.name}/{leg} was not published"
            assert (leg_dir / index_mod.IDMAP_FILENAME).exists()
            assert _dir_bytes(leg_dir) > 0, f"{spec.name}/{leg} published an empty artifact"
    assert built.manifest["legs"] == list(LEGS)
    for variant in RENDERINGS:
        section = built.manifest["variants"][variant]
        assert sorted(section["shards"]) == sorted({s.source_lang for s in specs})
        for shard in section["shards"].values():
            assert shard["parts"] == len(shard["shard_parts"]) >= 1
            for part in shard["shard_parts"]:
                assert sorted(part["legs"]) == sorted(LEGS)


# --------------------------------------------------------------------------- #
# FL02: id-set integrity at real scale, and the abort-on-drop half
# --------------------------------------------------------------------------- #
def test_fl02_id_set_integrity_per_leg_and_variant(built: _Built) -> None:
    """FL02 [consistency]: each (leg, rendering) indexes exactly the slice's passage ids."""
    expected = built.bounded.all_ids
    assert len(expected) == len(built.bounded.records)
    for variant in RENDERINGS:
        section = built.manifest["variants"][variant]
        assert section["passages"] == len(expected)
        covered: dict[str, set[str]] = {leg: set() for leg in LEGS}
        for shard in section["shards"].values():
            for leg, entry in _cell_legs(shard):
                ids = [
                    row["passage_id"]
                    for row in common_io.iter_parquet(
                        Path(entry["path"]) / index_mod.IDMAP_FILENAME
                    )
                ]
                assert len(ids) == len(set(ids)), (variant, leg)
                covered[leg] |= set(ids)
        for leg, ids in covered.items():
            assert ids == expected, (variant, leg, len(ids), len(expected))
    assert built.adapter.stats.value(
        index_mod.STAT_ID_INTEGRITY, variant="omt"
    ) == len(expected)


def test_fl02b_a_dropped_passage_aborts_publication(built: _Built) -> None:
    """FL02 [consistency]: a leg returning fewer vectors than passages fails the build.

    A batch killed for memory, or truncated, would make that rendering's index a different
    index from its siblings, and every cross-rendering comparison downstream would then be
    measuring the defect rather than the translation. Publication aborts with nothing
    partial written.
    """

    class _DropOne:
        name = DENSE_LEG

        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def encode_docs(self, ctx, texts):
            return list(self.inner.encode_docs(ctx, texts))[1:]

        def __getattr__(self, item):
            return getattr(self.inner, item)

    legs = tuple(_DropOne(leg) if leg.name == DENSE_LEG else leg for leg in built.legs)
    adapter, ctx = _scratch_adapter(built, "m05d-drop", legs=legs)
    spec = IndexShardSpec(variant="omt", source_lang="es")
    with pytest.raises(IndexIntegrityError, match="dropped passage"):
        adapter.work(ctx, _scratch_shard_file(built, spec, "m05d-drop"))
    shard_dir = adapter.shard_dir_from_ctx(ctx, spec)
    assert not is_done(shard_dir / DENSE_LEG)
    assert not adapter.manifest_path(built.bounded.cfg).exists()


# --------------------------------------------------------------------------- #
# FL03: idempotent resume across adapter instances
# --------------------------------------------------------------------------- #
def test_fl03_resume_skips_published_work_and_still_publishes_a_correct_manifest(
    built: _Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FL03 [consistency]: re-entering a finished shard re-streams and re-encodes nothing.

    The ``built`` fixture covers the cross-instance half, where a second, freshly built
    adapter drained the shards the first abandoned and published a complete manifest. This
    is the sharper half: on re-entry every leg is skipped before the passage text is read,
    so a retry after a late leg ran out of memory costs no encoding at all.
    """
    assert len(built.receipts) == 10
    adapter = IndexAdapter.for_config(
        built.bounded.cfg, base=built.bounded.root, legs=built.legs
    )
    assert adapter.idx_hash == built.ctx.idx_hash
    reads: list[int] = []
    monkeypatch.setattr(
        IndexAdapter, "load_shard", lambda self, ctx, spec: (reads.append(1), [])[1]
    )
    receipt = built.receipts[0]
    shard = Path(str(receipt).replace(f"{os.sep}out{os.sep}", f"{os.sep}done{os.sep}"))
    assert shard.exists(), shard
    spec = IndexShardSpec.from_payload(json.loads(shard.read_text(encoding="utf-8").strip()))
    before = _sha_tree(_shard_dir(built, spec.variant, spec.source_lang))
    adapter.work(built.ctx, shard)
    assert reads == [], "a completed shard re-read its passage text"
    assert adapter.stats.total(index_mod.STAT_LEG_RESUMED) == len(LEGS)
    assert _sha_tree(_shard_dir(built, spec.variant, spec.source_lang)) == before


# --------------------------------------------------------------------------- #
# FL04: self-retrieval, per leg per rendering
# --------------------------------------------------------------------------- #
def test_fl04_self_retrieval_returns_the_passage_itself(built: _Built) -> None:
    """FL04 [liveness]: a passage's own composed text retrieves itself, on every leg and
    rendering.

    The query is the document, so this says nothing about retrieval quality. It catches a
    mis-keyed idmap, an empty leg, or a save and load asymmetry that published an index
    nobody can reopen.

    Rank 1 is asserted for one leg only, for an arithmetic reason rather than a conventional
    one.

    - ``dense``: rank 0, asserted. The encoder L2-normalises and the index is an exact
      ``IndexFlatIP``, so a passage's score against itself is ``<v, v> = 1``, the largest
      value an inner product can take on unit vectors. No other document can exceed it, so a
      non-zero rank is a build defect, a mis-keyed idmap or a shifted ordinal, and never a
      property of the data.
    - ``sparse``: retrieval asserted, rank-1 rate recorded. Rank 0 is not guaranteed and
      asserting it was wrong. Measured on real data, and the same before the re-pack, so it
      is not an artefact of packing: the Russian ``omt_opus`` passage ``…_77697957#p1``, at
      508 tokens, comes back at rank 1 behind ``…_686759822#p2``, 507 tokens from a different
      document with the same source, at 304.97 against 302.86, a margin of 0.7 %. Two
      separate reasons make that expected. The sparse score is an unnormalised dot product
      over a SPLADE expansion, so ``<q, d> > <q, q>`` whenever a near-duplicate carries
      larger weights on the shared terms, since nothing bounds ``<q, q>`` from above the way
      unit-norm cosine does. And Seismic stores an approximation of the document:
      ``n_postings``, ``summary_energy`` and ``doc_cut`` prune postings at build time while
      ``query_cut`` and ``heap_factor`` prune the traversal, so the self score is
      systematically under-estimated and even a true ``<q, q>`` win can be lost to pruning.
      Demanding rank 1 from a pruned impact index would be a claim about SPLADE and Seismic
      rather than about this build.
    - ``late_interaction``: retrieval asserted, rank recorded. Its query encoder truncates at
      the checkpoint's 32-token ``query_length``, so demanding rank 1 from a 32-token prefix
      of a 500-token passage would likewise be a claim about MaxSim.

    What still bites for the two inexact legs is that the passage has to come back inside the
    measurement window on every one of the 45 probes, 5 passages by 3 languages by 3
    renderings. A mis-keyed idmap, an off-by-one ordinal or an unreadable reopened index
    fails that outright rather than surviving as rank 1 instead of 0.
    """
    ranks: dict[str, list[int]] = {}
    for variant in RENDERINGS:
        for lang in ("es", "ru", "zh"):
            handle = _handle(built, variant, lang)
            for record in _longest(built, lang, _SELF_RETRIEVAL_SAMPLE):
                for leg in LEGS:
                    hits = query_leg(handle, leg, record[variant], top_k=_TOP_K)
                    rank = _rank_of(hits, record["passage_id"])
                    ranks.setdefault(f"{leg}:{variant}", []).append(rank)
                    context = _explain_rank(built, leg, variant, lang, record, hits, rank)
                    if leg == DENSE_LEG:
                        assert rank == 0, context
                    else:
                        assert rank >= 0, context
    assert len(ranks) == len(LEGS) * len(RENDERINGS) == 9
    _MEASURED["self_retrieval"] = {
        key: {
            "rank1": sum(1 for r in value if r == 0),
            "found": sum(1 for r in value if r >= 0),
            "n": len(value),
            # Recorded rather than asserted. A leg whose self-retrieval rank drifts deep
            # into the window is worth reading even when nothing here fails.
            "worst_rank": max(value),
        }
        for key, value in sorted(ranks.items())
    }


# --------------------------------------------------------------------------- #
# FL05: cross-lingual sanity on the native index
# --------------------------------------------------------------------------- #
def test_fl05_an_english_query_finds_its_native_counterpart_in_the_original_index(
    built: _Built,
) -> None:
    """FL05 [liveness]: the native index is reachable from English at all.

    The query is the passage's own English rendering and the index searched is ``original``,
    the native text. This is not a cross-lingual retrieval quality measurement: the rank-1
    rate is recorded and compared against no published number, and what is asserted is only
    that the passage comes back inside the measurement window.
    """
    found: dict[str, list[int]] = {}
    for lang in ("es", "ru", "zh"):
        handle = _handle(built, "original", lang)
        for record in _longest(built, lang, 3):
            for leg in LEGS:
                hits = query_leg(handle, leg, record["omt"], top_k=_TOP_K)
                rank = _rank_of(hits, record["passage_id"])
                found.setdefault(f"{leg}:{lang}", []).append(rank)
                assert rank >= 0, (
                    f"{leg}/{lang}: the native passage {record['passage_id']} is not in the "
                    f"top {_TOP_K} for its own English text: the cross-lingual leg is dead, "
                    f"not merely imprecise; got {hits[:3]}"
                )
    _MEASURED["cross_lingual"] = {
        key: {"rank1": sum(1 for r in value if r == 0), "n": len(value), "ranks": value}
        for key, value in sorted(found.items())
    }


# --------------------------------------------------------------------------- #
# FL06, the cleanest comparison available: omt against omt_opus
# --------------------------------------------------------------------------- #
def test_fl06_cross_rendering_agreement_between_the_two_mt_tiers(built: _Built) -> None:
    """FL06 [consistency]: the same queries against ``omt`` and ``omt_opus``, measured.

    Both are English-to-English retrieval over the same passages, so a low overlap is
    attributable to translation quality alone. No pass or fail floor is asserted: none
    exists in advance, and inventing one would manufacture a quality claim there is no data
    for. The numbers are recorded so that a later run reading a much lower one has a
    baseline to read it against.
    """
    measured: dict[str, dict[str, float]] = {}
    queries = _queries(built)
    for lang in ("es", "ru", "zh"):
        left = _handle(built, "omt", lang)
        right = _handle(built, "omt_opus", lang)
        for leg in LEGS:
            rows = [shard_agreement(left, right, leg, q, _TOP_K) for q in queries]
            n = len(rows)
            measured[f"{leg}:{lang}"] = {
                "overlap": round(sum(r["overlap"] for r in rows) / n, 4),
                "rbo": round(sum(r["rbo"] for r in rows) / n, 4),
                "tau_b": round(sum(r["tau_b"] for r in rows) / n, 4),
            }
            assert rows and all(r["depth"] > 0 for r in rows), (leg, lang)
    _MEASURED["omt_vs_omt_opus"] = measured


# --------------------------------------------------------------------------- #
# FL07: no leg is degenerate; how much the legs agree
# --------------------------------------------------------------------------- #
def test_fl07_no_leg_returns_a_constant_ranking(built: _Built) -> None:
    """FL07 [consistency]: a leg whose top hit never moves is broken, whatever it scores."""
    handle = _handle(built, "omt", "es")
    queries = _queries(built)
    tops: dict[str, list[str]] = {}
    lists: dict[str, list[list[str]]] = {}
    for leg in LEGS:
        for query in queries:
            hits = query_leg(handle, leg, query, top_k=_TOP_K)
            assert hits, (leg, query)
            tops.setdefault(leg, []).append(hits[0][0])
            lists.setdefault(leg, []).append([pid for pid, _ in hits])
    for leg, top1 in tops.items():
        assert len(set(top1)) > 1, (
            f"{leg} returned the same top-1 for all {len(queries)} queries: that is a "
            "degenerate/constant-vector leg, not a ranking"
        )
    pairs: dict[str, dict[str, float]] = {}
    n = len(queries)
    for i, a in enumerate(LEGS):
        for b in LEGS[i + 1 :]:
            pairs[f"{a}|{b}"] = {
                "rbo": round(sum(rbo(lists[a][k], lists[b][k]) for k in range(n)) / n, 4),
                "tau_b": round(
                    sum(kendall_tau_b(lists[a][k], lists[b][k]) for k in range(n)) / n, 4
                ),
            }
    _MEASURED["between_legs"] = pairs


# --------------------------------------------------------------------------- #
# FL08: the English re-encode noise floor FL06 is read against
# --------------------------------------------------------------------------- #
def test_fl08_english_reencode_noise_floor(built: _Built) -> None:
    """FL08 [consistency]: the same English shard built three independent times.

    English is identity pass-through in both translation arms, so three independent forward
    passes over it can only disagree by floating-point noise on the GPU. Whatever residual
    disagreement shows up here is the floor FL06's cross-rendering numbers are read against:
    a difference at or below this floor says nothing about translation quality.
    """
    spec = IndexShardSpec(variant=None, source_lang=SHARED_LANG)
    handles = [
        open_shard(_build_one(built, spec, f"m05d-noise-{i}"), built.ctx) for i in range(3)
    ]
    queries = _queries(built)
    floor: dict[str, dict[str, float]] = {}
    identical: dict[str, bool] = {}
    for leg in LEGS:
        rows = [
            shard_agreement(handles[a], handles[b], leg, q, _TOP_K)
            for a, b in ((0, 1), (0, 2), (1, 2))
            for q in queries
        ]
        n = len(rows)
        floor[leg] = {
            "overlap": round(sum(r["overlap"] for r in rows) / n, 6),
            "rbo": round(sum(r["rbo"] for r in rows) / n, 6),
            "tau_b": round(sum(r["tau_b"] for r in rows) / n, 6),
        }
        identical[leg] = all(r["overlap"] == 1.0 for r in rows)
    _MEASURED["english_noise_floor"] = {"agreement": floor, "top_k_identical": identical}
    # A floor below total agreement is recorded. A floor that cannot be measured at all,
    # because the result lists are empty, is a failure.
    assert all(v["overlap"] > 0 for v in floor.values()), floor


# --------------------------------------------------------------------------- #
# FL09: per-leg storage, measured and extrapolated
# --------------------------------------------------------------------------- #
def test_fl09_per_leg_storage_measured_and_extrapolated(built: _Built) -> None:
    """FL09 [consistency]: bytes per passage per leg on real passages, extrapolated to 9.4 M.

    This answers what multi-vector storage costs at this corpus's scale with a number taken
    off a real build rather than a vendor estimate. The extrapolation is linear and is a
    planning figure, not a promise.
    """
    per_leg: dict[str, dict[str, float]] = {}
    for leg in LEGS:
        total_bytes = 0
        total_passages = 0
        for variant in RENDERINGS:
            for shard in built.manifest["variants"][variant]["shards"].values():
                if shard["shared"] and variant != RENDERINGS[0]:
                    continue  # the shared English build is one artefact, counted once
                for part in shard["shard_parts"]:
                    entry = part["legs"][leg]
                    total_bytes += int(entry["bytes"])
                    total_passages += int(entry["passages"])
        assert total_passages > 0 and total_bytes > 0, leg
        per_passage = total_bytes / total_passages
        per_leg[leg] = {
            "bytes": total_bytes,
            "passages": total_passages,
            "bytes_per_passage": round(per_passage, 1),
            "corpus_gb_x3_renderings": round(
                per_passage * _CORPUS_PASSAGES * _CORPUS_RENDERINGS / 1e9, 1
            ),
        }
    _MEASURED["storage"] = per_leg
    assert per_leg[LATE_INTERACTION_LEG]["bytes_per_passage"] > 0


# --------------------------------------------------------------------------- #
# FL10: provenance records the pins that decide reproducibility
# --------------------------------------------------------------------------- #
def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - a diagnostic must not fail the test
        return "<unknown>"


def test_fl10_manifest_provenance_matches_what_the_job_actually_ran(built: _Built) -> None:
    """FL10 [consistency]: the recorded architecture and batch-composition pins are real.

    The dense leg measured bitwise identical across three GPU architectures, so an empty
    ``gpu_arch_pin`` is the deliberate value. It still has to match the constraint the job
    requested, or the manifest describes a build that did not happen. The pin that does move
    results is batch composition, and the manifest records its hashed bounds.
    """
    provenance = built.manifest["provenance"]
    opts = index_build_options(built.bounded.cfg)
    assert provenance["gpu_arch_pin"] == opts.gpu_arch_pin
    requested = os.environ.get("SLURM_JOB_CONSTRAINTS", "").strip()
    if provenance["gpu_arch_pin"]:
        assert provenance["gpu_arch_pin"] in requested, (provenance["gpu_arch_pin"], requested)
    pin = provenance["batch_composition_pin"]
    assert pin["order"] == "(token_count, passage_id)"
    declared = dict(built.bounded.cfg.blocks["index_build"]["config"])
    assert pin["token_budget"] == int(declared["encode_batch_token_budget"])
    assert pin["max_items"] == int(declared["encode_max_items"])
    assert provenance["dense_model"] == opts.dense_model
    assert provenance["sparse_model"] == opts.sparse_model
    assert provenance["spine_model"] == opts.spine_model
    _MEASURED["provenance"] = {
        "gpu_arch_pin": provenance["gpu_arch_pin"]
        or "<none: dense leg measured architecture-invariant>",
        "slurm_constraint": requested or "<none>",
        "gpu": _gpu_name(),
        "batch_composition_pin": pin,
    }


# --------------------------------------------------------------------------- #
# FL11, determinism: same shard, same recipe, byte-identical artefacts
# --------------------------------------------------------------------------- #
def _engine_bytes_of_an_unpinnable_leg(relative_path: str) -> bool:
    """True for the engine file of a leg whose clustering cannot be seeded.

    Derived from :data:`index.UNSEEDABLE_LEGS`, the same dict the build logs and writes into
    ``provenance.train_seed``, so the exemption here and the admission in the code are one
    fact rather than two that can drift. Remove the entry in code and this test tightens by
    itself.

    The exemption is narrow: the engine's own file and nothing else. The leg's
    ``idmap.parquet`` is written by ``common.io`` from the pinned shard order and has nothing
    to do with the vendor's RNG, so it stays under the strict byte-identity assertion.
    """
    leg = relative_path.split("/", 1)[0]
    return leg in index_mod.UNSEEDABLE_LEGS and not relative_path.endswith(
        index_mod.IDMAP_FILENAME
    )


def test_fl11_the_same_shard_built_twice_is_byte_identical(built: _Built) -> None:
    """FL11 [consistency]: the same slice, recipe and batch composition give the same bytes.

    Bytes rather than rank order. An index that is rank-stable but byte-unstable stops the
    artefact tree being a checkpoint, because a resume would serve different vectors from
    the ones its sibling renderings were compared against.

    Both builds target the same directory and the first is moved aside once written, so any
    absolute path an engine embeds in its own metadata is identical in both, and the
    comparison measures the encoder and the index rather than the directory it ran in.

    One leg is exempt from the byte comparison, and the exemption lives in code rather than
    here. :data:`index.UNSEEDABLE_LEGS` records why: pyseismic-lsr 0.5.1 exposes no seed
    argument on ``build`` or ``build_from_dataset`` and no environment knob, so its index
    file cannot be reproduced by any means available to a caller. The file-level assertion is
    therefore scoped to the legs that can be pinned, and the sparse leg is held to what it
    can claim: byte-identical input across re-runs, measured directly, since
    :class:`_DigestLeg` hashes every vector handed to every writer.

    So what the sparse leg claims is byte-identical vectors in a byte-identical order and a
    byte-identical id map, which is everything this code controls, rather than a
    byte-identical stored index file. The other two legs are still compared byte for byte,
    and the late-interaction leg passes that only because its three RNG and kernel pins were
    found and fixed, not because it was exempted as well.
    """
    spec = IndexShardSpec(variant="omt", source_lang="es")
    legs = tuple(_DigestLeg(inner=leg) for leg in built.legs)
    adapter, ctx = _scratch_adapter(built, "m05d-det", legs=legs)
    shard_file = _scratch_shard_file(built, spec, "m05d-det")
    shard_dir = adapter.shard_dir_from_ctx(ctx, spec)

    adapter.work(ctx, shard_file)
    first_input = {leg.name: leg.digest() for leg in legs}
    aside = shard_dir.parent / f"{shard_dir.name}.first"
    shutil.move(str(shard_dir), str(aside))
    for leg in LEGS:  # the markers live beside the leg directory, so they move separately
        marker = common_io.success_marker(shard_dir / leg)
        if marker.exists():
            marker.unlink()
    for leg in legs:
        leg.reset()
    adapter.work(ctx, shard_file)
    second_input = {leg.name: leg.digest() for leg in legs}

    first, second = _sha_tree(aside), _sha_tree(shard_dir)
    assert sorted(first) == sorted(second), (sorted(first), sorted(second))
    differing = sorted(k for k in first if first[k] != second[k])
    unexpected = [k for k in differing if not _engine_bytes_of_an_unpinnable_leg(k)]
    _MEASURED["determinism"] = {
        "files": len(first),
        "differing": differing,
        "exempt_unseedable_engine_files": sorted(set(differing) - set(unexpected)),
        "encode_input_sha256": first_input,
        "encode_input_reproduced": first_input == second_input,
        "unseedable_legs": sorted(index_mod.UNSEEDABLE_LEGS),
    }

    # Every leg, the exempt one included, is fed identical vectors in an identical order.
    # That is what the batch-composition pin exists to give, and all the sparse leg claims.
    assert first_input == second_input, (
        "two builds of the same shard encoded different vectors (or the same vectors in a "
        f"different order): {first_input} vs {second_input}, the batch-composition pin is "
        "broken, which is a far larger failure than any engine's internal RNG"
    )
    # Every stored byte outside an unpinnable engine's own file matches.
    assert not unexpected, (
        f"two builds of the same shard under the same recipe differ in {unexpected}: the "
        "batch-composition pin does not fully determine the artifact. (The documented "
        f"exception, {sorted(index_mod.UNSEEDABLE_LEGS)}, is already excluded from this list.)"
    )


# --------------------------------------------------------------------------- #
# FL12: the truncation asymmetry between languages is counted, not silent
# --------------------------------------------------------------------------- #
def test_fl12_late_interaction_truncation_is_counted_per_language(built: _Built) -> None:
    """FL12 [consistency]: the per-language truncation asymmetry is measured, not silent.

    Measured on the corpus: untruncated lengths of en 176-208, ru 136-190, es 141-327 and zh
    312-343 against a 220-token document window, which drops about 35 % of Chinese tokens.
    That is a per-language bias introduced by the retrieval method rather than by the corpus.

    The problem is removed where it is created, by ``preprocess.packing``, which packs every
    passage to fit the retrieval window in every rendering. So what has to be visible here is
    what that mechanism can get wrong: the residual truncation counter per language, and the
    ``is_oversized`` tail. The counter stays sliced per language, because a per-language
    failure is what a corpus-wide total would hide.
    """
    langs = ("en", "es", "ru", "zh")
    passages = {
        lang: sum(built.stat(index_mod.STAT_PASSAGES, lang=lang, variant=v) for v in RENDERINGS)
        for lang in langs
    }
    truncated = {
        lang: sum(
            built.stat(
                index_mod.STAT_TRUNCATED, leg=LATE_INTERACTION_LEG, lang=lang, variant=variant
            )
            for variant in RENDERINGS
        )
        for lang in langs
    }
    oversized = {
        lang: sum(
            1
            for r in built.bounded.records.values()
            if r["lang"] == lang and r.get("is_oversized")
        )
        for lang in langs
    }
    _MEASURED["truncation_counters"] = {
        "passages_encoded": passages,
        "residual_truncated_tokens": truncated,
        "oversized_passages_by_lang": oversized,
        # The corpus-wide constant this shard is a sample of: 1,812 sentences whose own
        # len_max exceeds the content budget (en 846 / es 268 / ru 338 / zh 360, exact over
        # all 88,719,200). A packer cannot split a sentence, so these stay over budget.
        "irreducible_tail_corpus_wide": packing_mod.IRREDUCIBLE_TAIL_TOTAL,
    }
    # The counter is wired and sliced per language: every language encoded something.
    for lang in langs:
        assert passages[lang] > 0, (lang, passages)


def test_no_passage_is_truncated_in_any_language_or_rendering(built: _Built) -> None:
    """[consistency] no passage is truncated except the irreducible tail.

    Truncation is not acceptable for any passage a packer could have made fit. The one
    exception is the tail a packer cannot fix: a single sentence whose own ``len_max``
    exceeds the content budget, which no budget and no packing rule removes. Corpus-wide
    that is ``packing.IRREDUCIBLE_TAIL_TOTAL``, 1,812 sentences of 88,719,200, or 0.002 %:
    en 846, es 268, ru 338, zh 360. Those passages carry ``is_oversized`` and are the only
    ones allowed to lose a token here. Asserting zero would fail the first time it met real
    data.

    What is asserted is the property, not the mechanism. It held under MaxP windowing and it
    holds under rendering-invariant packing, which is why the mechanism was safe to replace.

    There are two independent readings: the build's own counter, which is what the whole
    bounded shard did, and a re-measurement with the checkpoint's own tokenizer over the real
    indexed units of sampled passages, which is what the config would have to be lying about
    for the counter to be wrong. Neither is read off the config. Truncation stayed invisible
    for so long because ``document_length: 220`` was a declared number nobody compared
    against real token lengths: measured corpus-wide it truncates 65.8-79.1 % of passages in
    every language, en 79.1, es 77.7, ru 65.8, zh 75.5.
    """
    langs = ("en", "es", "ru", "zh")
    counted = {
        f"{lang}:{variant}": built.stat(
            index_mod.STAT_TRUNCATED, leg=LATE_INTERACTION_LEG, lang=lang, variant=variant
        )
        for lang in langs
        for variant in RENDERINGS
    }
    offenders = {key: value for key, value in counted.items() if value}
    # Any loss at all has to be attributable to the irreducible tail. If this shard holds no
    # oversized passage the counter is exactly zero; if it does, the loss is allowed and is
    # re-measured below against the passages that may carry it.
    oversized = {
        r["passage_id"] for r in built.bounded.records.values() if r.get("is_oversized")
    }
    assert not offenders or oversized, (
        f"the late-interaction leg dropped tokens on {sorted(offenders)} while NO passage in "
        "this shard is oversized: every one of those passages is indexed as a prefix of "
        "itself, and no downstream measurement can tell that apart from a translation-quality "
        f"effect: {offenders}"
    )

    # Re-measured rather than inferred: tokenise the real indexed units, which are whole
    # passages, and require zero dropped tokens on every passage a packer could have made fit.
    mtd = built.ctx.mtd
    limit = mtd.effective_document_length()
    worst: dict[str, int] = {}
    tail: dict[str, int] = {}
    for lang in langs:
        for variant in RENDERINGS:
            for record in _longest(built, lang, 3):
                dropped = max(mtd.truncated_tokens([record[variant]]), default=0)
                key = f"{lang}:{variant}"
                if record["passage_id"] in oversized:
                    tail[key] = max(tail.get(key, 0), dropped)
                else:
                    worst[key] = max(worst.get(key, 0), dropped)
    _MEASURED["truncation"] = {
        "counter_by_lang_variant": counted,
        "document_length": limit,
        "max_dropped_tokens_resampled": worst,
        "max_dropped_tokens_oversized_tail": tail,
        "oversized_passages_in_shard": len(oversized),
        "irreducible_tail_corpus_wide": packing_mod.IRREDUCIBLE_TAIL_TOTAL,
    }
    assert not any(worst.values()), (
        f"re-measuring NON-OVERSIZED passages against the {limit}-token document window found "
        f"dropped tokens: {sorted(k for k, v in worst.items() if v)}, {worst}. Those are "
        "passages the packer could have made fit, so this is a packing or a document_length "
        "failure, not the irreducible tail."
    )


def test_late_interaction_encodes_whole_passages_within_a_covering_window(
    built: _Built,
) -> None:
    """[consistency] whole passages are encoded, and the window covers the pack budget.

    Rendering-invariant packing guarantees every passage fits the retrieval window in every
    rendering, so the encoder needs no windowing overflow path. Two things have to hold, and
    neither is readable off the config alone.

    The encode window covers the budget the passages were packed to. Otherwise the upstream
    guarantee is worthless and the encoder truncates the long passages again, per language.

    And the client carries no windowing surface. A dormant half-remedy with no config to
    read is a phantom knob.
    """
    mtd = built.ctx.mtd
    budget = int(built.bounded.cfg.blocks["packing"]["pack_budget"])
    assert mtd.effective_document_length() >= budget, (
        f"document_length {mtd.effective_document_length()} < packing.pack_budget {budget}: "
        "every passage above the window is truncated again, per-language, because only the "
        "long ones reach the limit"
    )
    for gone in ("window_texts", "window_bounds", "maxp_enabled"):
        assert not hasattr(mtd, gone), gone


# --------------------------------------------------------------------------- #
# FL13: the hashed index_build block is byte-identical across the whole roster
# --------------------------------------------------------------------------- #
def _block_text(text: str, name: str) -> str:
    """A YAML block's raw body: its column-0 key through the line before the next one."""
    lines = text.splitlines(keepends=True)
    start = next(
        (i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(name)}:", ln)), None
    )
    assert start is not None, f"top-level block {name!r} not found"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^[A-Za-z_][\w-]*:", lines[j]):
            end = j
            break
    return "".join(lines[start:end])


def test_fl13_index_build_block_is_byte_identical_across_all_six_configs() -> None:
    """FL13 [consistency]: the ``index_build`` block is identical in every shipped config.

    The block is shared across a run family in substance, since the family builds one index
    per rendering and this block's hash keys its path, but it is not yet listed in
    ``config.schema.SHARED_BLOCKS``. Until it is, this is the check, as
    ``tests/config/test_merge_translation_blocks_full.py`` is for the merge, translation and
    reconcile blocks.
    """
    paths = sorted((_REPO_ROOT / "config").glob("*.yml"))
    assert len(paths) == 6, [p.name for p in paths]
    bodies = {p.name: _block_text(p.read_text(encoding="utf-8"), "index_build") for p in paths}
    assert len(set(bodies.values())) == 1, (
        f"index_build differs across the roster: {sorted(bodies)} must be byte-identical"
    )
    hashes = {p.name: all_hashes(load(p))["index_build"] for p in paths}
    assert len(set(hashes.values())) == 1, hashes


# --------------------------------------------------------------------------- #
# FL15: English is encoded once, and the saving is real
# --------------------------------------------------------------------------- #
def test_fl15_english_is_encoded_once_for_all_three_renderings(built: _Built) -> None:
    """FL15 [consistency]: one shared English build, three manifests, measured saving.

    Reuse means publishing one build into three manifests, not skipping the shard and
    falling back at query time. So the shared path is string-equal in all three sections and
    every English ``passage_id`` is still covered by each rendering's own id set.
    """
    english = built.bounded.ids_by_lang["en"]
    shared_paths = {
        built.manifest["variants"][v]["shards"][SHARED_LANG]["path"] for v in RENDERINGS
    }
    assert len(shared_paths) == 1, shared_paths
    assert shared_paths.pop().endswith(f"_shared/{SHARED_LANG}")
    for variant in RENDERINGS:
        covered: set[str] = set()
        for shard in built.manifest["variants"][variant]["shards"].values():
            for part in shard["shard_parts"]:
                covered |= {
                    row["passage_id"]
                    for row in common_io.iter_parquet(
                        Path(part["legs"][DENSE_LEG]["path"]) / index_mod.IDMAP_FILENAME
                    )
                }
        assert english <= covered, variant

    encoded_en = sum(
        built.stat(index_mod.STAT_ENCODED, leg=DENSE_LEG, lang="en", variant=variant)
        for variant in RENDERINGS
    )
    assert encoded_en == len(english), (encoded_en, len(english))
    tokens_en = sum(int(built.bounded.records[pid]["token_count"] or 0) for pid in english)
    tokens_all = sum(int(r["token_count"] or 0) for r in built.bounded.records.values())
    saved = 2 * tokens_en / (tokens_all + 2 * tokens_en)
    assert tokens_en > 0 and 0.0 < saved < 1.0
    _MEASURED["english_once"] = {
        "en_passages": len(english),
        "en_dense_encodes": encoded_en,
        "en_tokens": tokens_en,
        "all_tokens": tokens_all,
        "encode_token_saving_fraction": round(saved, 4),
    }


# --------------------------------------------------------------------------- #
# FL16: a real kill between two legs
# --------------------------------------------------------------------------- #
_KILL_DRIVER = """
import json, sys
from dataclasses import replace
from pathlib import Path
from ragtime.config import load
from ragtime.preprocess.index import IndexAdapter, index_build_options
cfg = load(Path({config!r}))
# for_config resolves recon/pack/index hashes from the config; only the OUTPUT node is
# re-pointed, so the child reads exactly the corpus the parent gate built against.
adapter = replace(IndexAdapter.for_config(cfg, base={root!r}), idx_hash={idx!r})
assert adapter.recon_hash == {recon!r}, (adapter.recon_hash, {recon!r})
ctx = adapter.bringup(cfg)
if not getattr(ctx.dense, "model", ""):
    from ragtime.serving.encoders import Encoder
    ctx.dense = Encoder(model=index_build_options(cfg).dense_model)
adapter.work(ctx, Path({shard!r}))
"""


def test_fl16_a_killed_worker_loses_only_the_unfinished_leg(built: _Built) -> None:
    """FL16 [consistency]: a real kill between legs on real hardware, not a mock.

    One worker holds three multi-gigabyte encoders resident, so running out of memory in the
    third leg is the ordinary failure rather than an exotic one. Each leg is published
    independently, so a kill costs only the leg that was mid-flight. Anything else turns a
    retry into a full rebuild of a shard that had already paid for two thirds of its
    encoding.
    """
    idx_hash = hashlib.sha256(b"m05d-kill").hexdigest()
    spec = IndexShardSpec(variant="omt_opus", source_lang="es")
    root = built.bounded.root
    shard_file = _scratch_shard_file(built, spec, "m05d-kill")
    script = root / "scratch" / "m05d-kill" / "driver.py"
    script.write_text(
        _KILL_DRIVER.format(
            config=str(_CONFIG),
            recon=built.bounded.recon,
            idx=idx_hash,
            root=str(root),
            shard=str(shard_file),
        ),
        encoding="utf-8",
    )
    shard_dir = built.bounded.layout.index_shard_dir(
        built.bounded.recon, idx_hash, spec.variant, spec.source_lang, part=spec.part
    )
    proc = subprocess.Popen([sys.executable, str(script)], cwd=str(_REPO_ROOT))
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if is_done(shard_dir / DENSE_LEG):
                break
            if proc.poll() is not None:
                pytest.fail(
                    f"the kill probe's child exited (rc={proc.returncode}) before publishing "
                    f"{DENSE_LEG}: nothing was killed, so nothing was proven"
                )
            time.sleep(0.05)
        else:
            pytest.fail("the kill probe's child never published its first leg within 30 min")
        proc.kill()
    finally:
        proc.wait(timeout=300)

    survivor = _sha_tree(shard_dir / DENSE_LEG)
    assert survivor, "the completed leg did not survive the kill"
    unfinished = [leg for leg in LEGS if not is_done(shard_dir / leg)]
    assert unfinished, (
        "the child finished all three legs before the kill landed: the probe raced itself "
        "and proves nothing about per-leg atomicity"
    )
    assert DENSE_LEG not in unfinished

    adapter = replace(
        IndexAdapter.for_config(built.bounded.cfg, base=root, legs=built.legs),
        idx_hash=idx_hash,
    )
    adapter.work(replace(built.ctx, idx_hash=idx_hash), shard_file)
    assert _sha_tree(shard_dir / DENSE_LEG) == survivor, "a published leg was rebuilt"
    for leg in LEGS:
        assert is_done(shard_dir / leg), leg
    assert (
        adapter.stats.value(
            index_mod.STAT_LEG_RESUMED,
            leg=DENSE_LEG,
            lang=spec.source_lang,
            variant=spec.text_rendering,
        )
        == 1.0
    )
    _MEASURED["kill_probe"] = {
        "killed_after": DENSE_LEG,
        "rebuilt_on_retry": unfinished,
    }


# --------------------------------------------------------------------------- #
# FL18: search by display, all nine combinations, on real passages
# --------------------------------------------------------------------------- #
def test_fl18_search_times_display_grid_over_real_passages(built: _Built) -> None:
    """FL18 [consistency]: any id from any index displays in all three renderings.

    This is the contract retrieval's split between ``retrieve`` and ``display`` assumes,
    checked over real composed text, including multi-sentence Chinese passages where the
    native rendering is a slice and the two translated ones are joins. Searching one
    rendering and displaying another is an ordinary cell here, not an afterthought.
    """
    pytest.importorskip("lmdb")
    from ragtime.common.passage_store import LmdbPassageStore

    store = LmdbPassageStore.build_from_final(
        built.bounded.root / "passage_store.lmdb",
        built.bounded.layout,
        built.bounded.recon,
        # The same packing the index was built from. ``pack_hash`` has no default, so this
        # cannot fall back to a differently packed table: a store built from one grouping and
        # an index built from another would resolve half the ids and read as a display bug.
        pack_hash=built.bounded.pack,
    )
    combos = 0
    try:
        for variant in RENDERINGS:
            handle = _handle(built, variant, "zh")
            hits = query_leg(handle, DENSE_LEG, _queries(built)[0], top_k=3)
            assert hits, variant
            pid, score = hits[0]
            assert isinstance(pid, str) and isinstance(score, float)
            for rendering in RENDERINGS:
                text = store.render(pid, rendering)
                assert text and isinstance(text, str), (variant, rendering, pid)
                combos += 1
    finally:
        store.close()
    assert combos == 9


def test_search_returns_ids_and_scores_only_from_the_real_index(built: _Built) -> None:
    """[consistency] the index returns ids and scores, checked on the real artefacts.

    Retrieval depends on this, and it is guaranteed rather than merely observed: the only
    per-passage table any leg writes is ``idmap.parquet``, which has no text column.
    """
    texts = {r["original"] for r in built.bounded.records.values()}
    for variant in RENDERINGS:
        handle = _handle(built, variant, "ru")
        for leg in LEGS:
            hits = query_leg(handle, leg, _queries(built)[0], top_k=5)
            assert hits, leg
            for item in hits:
                assert type(item) is tuple and len(item) == 2
                pid, score = item
                assert isinstance(pid, str) and isinstance(score, float)
                assert pid in built.bounded.records
                assert pid not in texts
    for idmap in built.adapter.index_dir(built.bounded.cfg).rglob(index_mod.IDMAP_FILENAME):
        columns: set[str] = set()
        for row in common_io.iter_parquet(idmap):
            columns |= set(row)
        assert columns == {"ordinal", "passage_id", "document_id", "source_lang"}, idmap


def test_per_leg_config_hash_is_identical_across_the_three_renderings(built: _Built) -> None:
    """[consistency] the method is identical across renderings; only the text differs.

    Separately, no leg exists on one rendering and not another. Both halves are strict
    equalities, with no exception left for any leg.
    """
    for leg in LEGS:
        hashes = {
            entry["config_hash"]
            for variant in RENDERINGS
            for shard in built.manifest["variants"][variant]["shards"].values()
            for name, entry in _cell_legs(shard)
            if name == leg
        }
        assert len(hashes) == 1, (leg, hashes)
        assert hashes.pop() == built.manifest["leg_config_hash"][leg]
    assert len(set(built.manifest["leg_config_hash"].values())) == 3
    assert sorted(built.manifest["legs"]) == sorted((DENSE_LEG, SPARSE_LEG, LATE_INTERACTION_LEG))


# --------------------------------------------------------------------------- #
# FL19: the late-interaction leg's part buffer, on real embeddings.
#
# Without this there is no corpus-scale build. A shard's whole token-embedding buffer has to
# reach `add_documents` in one call, and for the English shard that is a measured 376 GiB in
# fp32, so the index is cut into parts of `plaid_part_passages` passages. What only real
# embeddings can give is bytes per passage, which is what the shipped part size costs in host
# RAM at corpus grain.
# --------------------------------------------------------------------------- #
def test_fl19_one_shard_part_is_one_plaid_part_and_its_buffer_is_measured(
    built: _Built,
) -> None:
    """FL19 [consistency+liveness]: one shard part is one PLAID part, and its buffer is sized.

    There is one part size. ``plaid_part_passages`` cuts the shard, through
    :meth:`IndexAdapter.part_bounds` and ``preprocess.assemble``, and also flushes the PLAID
    buffer in ``_PlaidWriter``. So a shard part holds at most one part's worth of passages,
    and its late-interaction leg holds exactly one PLAID part by construction.

    The plan is constructed from the part size this test forces rather than inherited from
    the one-part default. Forcing the size down while leaving the spec at ``parts=1``
    describes a build the recipe cannot produce, a shard claiming to be the whole cell while
    the table implies three parts, and ``_ordered_lang_keys`` is right to refuse it: that
    refusal is what stands between a stale plan and a language indexed as a prefix.

    The multi-part fan inside a leg stays in ``_PlaidWriter`` as the buffer bound and is
    covered by the small tests; config cannot reach it. The fan that matters at cell
    grain is FL20's. What only a real build can say is that this holds against a real PLAID
    build rather than a stand-in engine, and that one part's buffer is measured, so bytes per
    passage extrapolates the shipped part size to the host-RAM figure the corpus build is
    scheduled against.
    """
    lang, variant = "es", "omt"
    passages = len(built.bounded.ids_by_lang[lang])
    part_passages = max(1, passages // 3)
    parts = -(-passages // part_passages)  # >= 3 parts
    assert parts >= 3, (passages, part_passages, parts)
    # The plan is built from the part size this test forces rather than the one-part default:
    # a spec and a recipe that disagree describe no build the queue would ever seed.
    spec = IndexShardSpec(variant=variant, source_lang=lang, part=0, parts=parts)

    adapter, ctx = _scratch_adapter(built, "m05d-parts")
    ctx = replace(ctx, opts=replace(ctx.opts, plaid_part_passages=part_passages))

    # A forced part size paired with a default ``parts=1`` spec is the stale plan
    # ``_ordered_lang_keys`` exists to refuse, and that refusal is what stands between a
    # re-sized recipe and a language indexed as a prefix. It is asserted here on the real
    # table. The refusal happens inside ``load_shard`` before a single vector is encoded, so
    # nothing is published and this costs one column-pruned scan.
    stale = IndexShardSpec(variant=variant, source_lang=lang)
    assert (stale.part, stale.parts) == (0, 1)
    with pytest.raises(IndexIntegrityError, match=r"the shard plan says 1 part\(s\)"):
        adapter.work(ctx, _scratch_shard_file(built, stale, "m05d-parts-stale"))
    receipt = adapter.work(ctx, _scratch_shard_file(built, spec, "m05d-parts"))
    # validate re-derives this part's id set from the real passage table, so a boundary the
    # build computed differently from the plan fails here rather than at query time.
    assert adapter.validate(receipt), spec.name
    shard_dir = adapter.shard_dir_from_ctx(ctx, spec)
    leg_dir = shard_dir / LATE_INTERACTION_LEG

    census = common_io.read_jsonl(leg_dir / index_mod._PARTS_FILENAME)[0]
    # One PLAID part per shard part. Part 0 of a cell with three or more parts is a full
    # part, so the buffer measured below is a whole part's buffer.
    assert index_mod.plaid_part_count(leg_dir) == census["parts"] == 1
    assert census["units"] == census["part_passages"] == part_passages
    assert sorted(p.name for p in leg_dir.glob("plaid-*")) == sorted(census["dirs"])

    # Self-retrieval through the published part: first, middle and last of its ordinal space,
    # so a part written short at either end is visible.
    handle = open_shard(shard_dir, ctx)
    ordinals = [
        row["passage_id"] for row in common_io.read_parquet(leg_dir / index_mod.IDMAP_FILENAME)
    ]
    assert len(ordinals) == part_passages
    ranks = {}
    for probe in (ordinals[0], ordinals[len(ordinals) // 2], ordinals[-1]):
        text = built.bounded.records[probe][variant]
        hits = query_leg(handle, LATE_INTERACTION_LEG, text, top_k=10)
        ranks[probe] = _rank_of(hits, probe)
        # The merge is a total order, so one query twice is one answer twice.
        assert hits == query_leg(handle, LATE_INTERACTION_LEG, text, top_k=10)
    assert all(rank == 0 for rank in ranks.values()), ranks

    peak = int(census["peak_part_bytes"])
    per_passage = peak / max(1, int(census["units"]))
    shipped = index_mod._DEFAULT_PLAID_PART_PASSAGES
    _MEASURED["plaid_parts"] = {
        "bounded_cell": {
            "lang": lang,
            "passages": passages,
            "part_passages": part_passages,
            "cell_parts": parts,
            "plaid_parts_in_one_shard_part": int(census["parts"]),
            "peak_part_bytes": peak,
        },
        "bytes_per_passage": round(per_passage, 1),
        "shipped_part_passages": shipped,
        "projected_part_gib_at_shipped_size": round(per_passage * shipped / 2**30, 1),
    }


# --------------------------------------------------------------------------- #
# FL20: the shard part axis, on the real corpus tables and the real encoders.
#
# One level up from FL19: a part is a claimable work unit with its own three legs and its own
# ordinal space, sized so no language shard outruns a scheduler's wall-clock limit. What only
# a real build can show is that the partition is exact against the real passage table, with
# the boundary rule re-derived by `validate` rather than trusted from the artefact, and that
# the fan-merge still puts a passage's own text at rank 1 across independently built parts.
# --------------------------------------------------------------------------- #
def test_fl20_shard_parts_partition_the_cell_and_fan_merge_at_rank_1(built: _Built) -> None:
    """FL20 [consistency+liveness]: three or more real parts of one cell, exact and searchable.

    The part size is forced down through the resolved recipe until the bounded cell cuts into
    at least three parts, for the same reason FL19 forces the PLAID one: at one part the
    partition and the fan are both trivial and would show nothing.
    """
    lang, variant = "es", "omt"
    passages = len(built.bounded.ids_by_lang[lang])
    size = max(1, -(-passages // 3))
    parts = -(-passages // size)
    assert parts >= 3, (passages, size, parts)

    adapter, ctx = _scratch_adapter(built, "m05d-shard-parts")
    ctx = replace(ctx, opts=replace(ctx.opts, plaid_part_passages=size))
    specs = [
        IndexShardSpec(variant=variant, source_lang=lang, part=part, parts=parts)
        for part in range(parts)
    ]
    for spec in specs:
        out = adapter.work(ctx, _scratch_shard_file(built, spec, "m05d-shard-parts"))
        # validate re-derives this part's id set from the real passage table, so a boundary
        # the build computed differently from the plan fails here, on real data.
        assert adapter.validate(out), spec.name

    cell = ctx.layout.index_lang_dir(ctx.recon_hash, ctx.idx_hash, variant, lang)
    census = index_mod.read_shard_parts(cell)
    assert census["parts"] == parts
    assert census["dirs"] == sorted(p.name for p in cell.glob("part-*"))
    assert sum(census["passages"]) == passages

    # The partition is total and disjoint against the corpus's id set for this language.
    per_part = [
        {
            row["passage_id"]
            for row in common_io.iter_parquet(
                adapter.shard_dir_from_ctx(ctx, spec) / DENSE_LEG / index_mod.IDMAP_FILENAME
            )
        }
        for spec in specs
    ]
    assert sum(len(ids) for ids in per_part) == passages
    assert set().union(*per_part) == built.bounded.ids_by_lang[lang]

    # Self-retrieval through the fan, probing the first and the last part. A merge that
    # dropped the tail would still answer part 0 correctly.
    handle = index_mod.open_lang(cell, ctx)
    assert len(handle.parts) == parts
    probes = [min(per_part[0]), max(per_part[-1])]
    ranks: dict[str, list[int]] = {}
    for probe in probes:
        text = built.bounded.records[probe][variant]
        for leg in LEGS:
            hits = index_mod.query_lang_leg(handle, leg, text, top_k=_TOP_K)
            rank = _rank_of(hits, probe)
            ranks.setdefault(leg, []).append(rank)
            # The same rule as the single-part self-retrieval above: the exact leg is rank 1
            # and the two approximate ones only have to find the passage.
            if leg == DENSE_LEG:
                assert rank == 0, (leg, probe, rank, hits[:3])
            else:
                assert rank >= 0, (leg, probe, hits[:3])
            # The merge is a total order, so one query twice is one answer twice.
            assert hits == index_mod.query_lang_leg(handle, leg, text, top_k=_TOP_K)
            # Scores are non-increasing, so the fan merged on score rather than on rank.
            assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)

    _MEASURED["shard_parts"] = {
        "lang": lang,
        "passages": passages,
        "part_passages_used": size,
        "parts": parts,
        "part_passages": census["passages"],
        "shipped_plaid_part_passages": index_mod._DEFAULT_PLAID_PART_PASSAGES,
        "fan_ranks": {leg: value for leg, value in sorted(ranks.items())},
    }


# --------------------------------------------------------------------------- #
# The measured record, printed so the numbers live in the job's own log.
# --------------------------------------------------------------------------- #
def test_zz_report_measured_facts(built: _Built) -> None:
    """[consistency] every number this module measures, in one place."""
    _MEASURED["index_bytes_total"] = _dir_bytes(built.adapter.index_dir(built.bounded.cfg))
    print("\n=== full index gate: measured facts ===")
    print(json.dumps(_MEASURED, indent=2, sort_keys=True, default=str))
    print("=== end of measured facts ===")
    assert _MEASURED["index_bytes_total"] > 0
