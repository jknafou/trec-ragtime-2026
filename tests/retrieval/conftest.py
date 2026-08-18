"""Fixture machinery for the retrieval suite: a real published index, hermetically.

The corpus-side fixture generator is reused from ``tests/preprocess/test_index_small.py`` rather
than rewritten: its ``_Doc`` shape, its ``_tables`` writer under the reconciliation Arrow schemas,
its ``_write`` Layout binding, its ``_cfg`` hashed ``index_build`` block, and its ``_vec``
hash-of-text stand-in. The index itself is published by the real
:class:`~ragtime.preprocess.index.IndexAdapter` through the real work-queue (claim -> work ->
validate -> mark_done -> merge), so ``open_lang`` / ``query_lang_leg`` / ``read_shard_parts`` run
against real artifacts, real part censuses and real ``idmap.parquet`` files. Nothing on the query
path is simulated.

Two things are added here that the index build's own suite has no reason to carry.

1. An encoder trio routed through ``IndexCtx``, one distinct encoder per leg.
   ``test_index_small``'s leg trio hashes the text directly, ignores ``IndexCtx`` entirely, and
   returns three different shapes. That is fine for a build test and useless for two of the
   sharpest contracts here: SM-R13 (the query encoders must be
   ``ClientBundle.index_dense``/``.milco``/``.mtd_colbert``, asserted by object identity, which is
   impossible to test with a leg that never touches ``ctx``) and SM-R03 (cross-leg RRF is only
   observable if the legs disagree). So :class:`_ScaledLeg` mirrors the real legs' ``ctx`` call
   sites one for one, and each leg gets its own salted encoder and its own score scale, orders of
   magnitude apart.
2. An adversarial Spanish cell: 12 passages in 6 byte-identical pairs, cut into 3 parts of 5, so
   that one duplicate pair straddles the part-0/part-1 boundary and another opens part 2 at
   ordinal 0. That is where the real corpus concentrates its duplicates (0.78 % corpus-wide,
   13.7 % at ordinal 0 against 1.1 % mid-part), and it is the one place where the
   ``(-score, passage_id)`` tie-break in the within-cell merge is load-bearing.

The document generator is inherited unmodified rather than replaced with a convenient one, which
also keeps the fixture from reintroducing the ``token_count = i + 1`` assumption. Token order is
not row order in the real corpus: ``token_count`` decreases 228,834 times over 460,326 real ``es``
passages, and a fixture that quietly made the two agree would pass every test here while hiding a
real multi-part defect.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout, PassageStore, Statistics
from ragtime.common.passage_store import iter_final_passages
from ragtime.orchestration import saturate
from ragtime.preprocess.index import (
    DENSE_LEG,
    LATE_INTERACTION_LEG,
    LEGS,
    SPARSE_LEG,
    IndexAdapter,
    index_hash,
)

# The corpus-side fixture generator, reused verbatim (see the module docstring).
from tests.preprocess.test_index_small import (
    _FILLER_TEXT,
    _RECON,
    _base_docs,
    _cfg,
    _Doc,
    _FakeWriter,
    _tables,
    _vec,
    _write,
)

__all__ = [
    "DUP_PAIRS",
    "ES_PAIR_DOC",
    "PART_PASSAGES",
    "Built",
    "Clients",
    "SpyReranker",
    "build_index",
    "retrieval_cfg",
]

#: The part size the whole suite builds at. Small: a single-part cell cannot
#: exercise the cross-part raw-score merge, which is where the two-level fusion rule hides.
PART_PASSAGES = 5

#: Spanish duplicate pairs. ``2k`` and ``2k+1`` carry byte-identical text in all three
#: renderings; pair ``k`` is ``k + 3`` words long, so the adapter's pinned
#: ``(token_count, passage_id)`` cut puts pair ``k`` at ordinals ``(2k, 2k+1)``, which places
#: pair 2 across the part-0/part-1 boundary (ordinals 4 and 5) and pair 5 at ordinal 0 of
#: part 2 (ordinals 10 and 11).
DUP_PAIRS = 6
ES_PAIR_DOC = "spa-docs/92{:05d}"


# --------------------------------------------------------------------------- #
# Documents: the four shape-carrying ones + the adversarial Spanish cell.
# --------------------------------------------------------------------------- #
def _es_pair_docs() -> list[_Doc]:
    """12 Spanish passages in 6 byte-identical pairs, one pair per token length."""
    out: list[_Doc] = []
    for k in range(DUP_PAIRS):
        native = " ".join(f"muelle{k}" for _ in range(k + 3))
        english = " ".join(f"berth{k}" for _ in range(k + 3))
        low = " ".join(f"quay{k}" for _ in range(k + 3))
        for half in (0, 1):
            out.append(
                _Doc(
                    ES_PAIR_DOC.format(2 * k + half),
                    "es",
                    [native],
                    " ",
                    [(0,)],
                    omt=[english],
                    omt_opus=[low],
                )
            )
    return out


def _filler(lang: str, count: int, first: int = 0) -> list[_Doc]:
    """``count`` one-passage filler documents for ``lang``, from the shared filler material."""
    id_template, sep, native, omt, omt_opus = _FILLER_TEXT[lang]
    return [
        _Doc(
            id_template.format(n),
            lang,
            [s.format(n=n) for s in native],
            sep,
            [(0, 1)],
            omt=[s.format(n=n) for s in omt],
            omt_opus=[s.format(n=n) for s in omt_opus],
        )
        for n in range(first, first + count)
    ]


def retrieval_docs() -> list[_Doc]:
    """en 6 / ru 6 / zh 6 passages (2 parts each) + es 12 (3 parts), all four languages.

    Every non-Spanish language keeps its shape-carrying base document: the multi-sentence,
    no-separator Chinese one above all, since it is the case where ``original`` is a slice and
    the two MT renderings are joins, so it is the one that makes "rerank scored the searched
    rendering" observable rather than a tautology.
    """
    base = {doc.lang: doc for doc in _base_docs()}
    return [
        base["en"],  # 2 passages
        *_filler("en", 4),
        base["ru"],  # 2 passages
        *_filler("ru", 4),
        base["zh"],  # 2 passages
        *_filler("zh", 4),
        *_es_pair_docs(),  # 12 passages, 6 byte-identical pairs
    ]


# --------------------------------------------------------------------------- #
# A ctx-routed, per-leg-distinct encoder trio (see the module docstring, point 1).
# --------------------------------------------------------------------------- #
class SaltedEncoder:
    """One encoder stand-in carrying the four call shapes the three real clients expose.

    ``salt`` puts each leg in its own vector space, so two legs genuinely disagree and cross-leg
    RRF becomes observable. ``scale`` multiplies the returned similarity, so the three legs' score
    magnitudes differ by orders of magnitude while their ranks stay meaningful: the property RRF
    exists for and a score sum would destroy.

    The dense encoder is left at ``salt=""``/``scale=1.0`` so ``<v, v> == 1.0`` for
    a self-query, which is the dense-only rank-1 guarantee SM-R06 probes.
    """

    def __init__(self, name: str, salt: str = "", scale: float = 1.0) -> None:
        self.name = name
        self.salt = salt
        self.scale = scale
        self.doc_calls: list[int] = []
        self.query_calls: list[str] = []

    def _vector(self, text: str) -> list[float]:
        return _vec(self.salt + text)

    # dense (``Encoder.embed``)
    def embed(self, texts, mode: str = "dense"):
        self.doc_calls.append(len(texts))
        return [self._vector(t) for t in texts]

    # MILCO (``encode_text`` / ``encode_query``)
    def encode_text(self, texts):
        self.doc_calls.append(len(texts))
        return [self._vector(t) for t in texts]

    # MTD/PyLate (``encode_docs`` / ``encode_query``)
    def encode_docs(self, texts):
        self.doc_calls.append(len(texts))
        return [self._vector(t) for t in texts]

    def encode_query(self, query: str):
        self.query_calls.append(query)
        return self._vector(query)


@dataclass
class _ScaledLeg:
    """One leg with the real leg's ``ctx`` call sites and a toy engine.

    ``encode_docs``/``encode_query`` route through the same ``IndexCtx`` field the real leg of that
    name routes through (``ctx.dense.embed(..., mode="dense")`` / ``ctx.milco`` / ``ctx.mtd``),
    which is what lets a test assert by object identity that the query encoders are the index-build
    clients, rather than by inspection.
    """

    name: str
    scale: float = 1.0

    def _client(self, ctx):
        return {DENSE_LEG: ctx.dense, SPARSE_LEG: ctx.milco, LATE_INTERACTION_LEG: ctx.mtd}[
            self.name
        ]

    def encode_docs(self, ctx, texts):
        client = self._client(ctx)
        if self.name == DENSE_LEG:
            return client.embed(list(texts), mode="dense")
        if self.name == SPARSE_LEG:
            return client.encode_text(list(texts))
        return client.encode_docs(list(texts))

    def encode_query(self, ctx, query):
        return self._client(ctx).encode_query(query)

    def writer(self, out_dir: Path, ctx):
        return _FakeWriter(out_dir, [], self.name)

    def open(self, leg_dir: Path, ctx):
        import json

        return json.loads((leg_dir / "vectors.json").read_text(encoding="utf-8"))

    def search(self, reader, ctx, query_rep, top_k):
        scored = [
            (
                int(ordinal),
                self.scale * sum(a * b for a, b in zip(query_rep, vec, strict=True)),
            )
            for ordinal, vec in reader.items()
        ]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[: int(top_k)]


def scaled_legs() -> tuple[_ScaledLeg, ...]:
    """The three legs, in ``LEGS`` order, with deliberately non-comparable score scales."""
    return (
        _ScaledLeg(DENSE_LEG, scale=1.0),
        _ScaledLeg(SPARSE_LEG, scale=1_000.0),
        _ScaledLeg(LATE_INTERACTION_LEG, scale=0.001),
    )


class SpyReranker:
    """A cross-encoder stand-in that records every call and reorders deterministically.

    The score is the passage text's length, which is (a) a total order the test can recompute
    and (b) reliably different from the RRF order, so "the reranker actually decided the head"
    is observable rather than assumed.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [float(len(p)) for p in passages]


@dataclass
class Clients:
    """A ``ClientBundle``-shaped stub with four distinct encoder identities.

    ``embedder`` is a different object from ``index_dense`` and returns vectors in a different
    space. Reading it instead of ``index_dense`` is a silent bug: queries and stored vectors in
    different embedding spaces still produce plausible numbers. The fixture makes that mistake
    detectable rather than invisible.
    """

    index_dense: SaltedEncoder = field(
        default_factory=lambda: SaltedEncoder("index_dense", salt="", scale=1.0)
    )
    milco: SaltedEncoder = field(
        default_factory=lambda: SaltedEncoder("milco", salt="sparse::")
    )
    mtd_colbert: SaltedEncoder = field(
        default_factory=lambda: SaltedEncoder("mtd", salt="late::")
    )
    embedder: SaltedEncoder = field(
        default_factory=lambda: SaltedEncoder("embedder", salt="wrong-SPACE::")
    )
    reranker: Any = field(default_factory=SpyReranker)


# --------------------------------------------------------------------------- #
# Config: test_index_small's hashed blocks + the query-time `retrieval` block.
# --------------------------------------------------------------------------- #
def retrieval_cfg(
    tmp_path: Path,
    *,
    index: str = "original",
    passage_lang: str = "original",
    rrf_k: int | None = 60,
    depth: int = 100,
    quality_gated: bool | None = None,
    part_passages: int = PART_PASSAGES,
) -> types.SimpleNamespace:
    """``test_index_small._cfg`` + the ``retrieval`` block and the two resolved knobs.

    The ``retrieval`` block is not an input to ``index_hash`` (which hashes ``index_build`` +
    ``packing``), so two configs differing only in a query-time knob address the same published
    index: which is exactly what SM-R05/SM-R15/SM-R17 need in order to attribute a difference
    to the knob rather than to a different artifact.
    """
    cfg = _cfg(tmp_path)
    cfg.blocks["index_build"]["config"]["plaid_part_passages"] = int(part_passages)
    rrf: dict[str, Any] = {}
    if rrf_k is not None:
        rrf["k"] = rrf_k
    if quality_gated is not None:
        rrf["quality_gated"] = quality_gated
    cfg.blocks["retrieval"] = {
        "rrf": rrf,
        "reranker": {"model": "Qwen/Qwen3-Reranker-4B", "depth": depth},
        "index": index,
    }
    cfg.retrieval_index = index
    cfg.passage_lang = passage_lang
    return cfg


# --------------------------------------------------------------------------- #
# The published index.
# --------------------------------------------------------------------------- #
@dataclass
class Built:
    """One published fixture index plus everything a retrieval test reads back."""

    cfg: Any
    layout: Layout
    adapter: IndexAdapter
    legs: tuple[_ScaledLeg, ...]
    clients: Clients
    docs: list[_Doc]
    records: dict[str, dict[str, Any]]
    manifest_path: Path
    root: Path

    @property
    def recon_hash(self) -> str:
        return _RECON

    @property
    def idx_hash(self) -> str:
        return self.adapter.idx_hash

    def store(self) -> PassageStore:
        """An in-memory store over the same composed records the index encoded."""
        return PassageStore.from_records(self.records.values())

    def lang_dir(self, variant: str | None, lang: str) -> Path:
        return self.layout.index_lang_dir(_RECON, self.idx_hash, variant, lang)

    def ids(self, lang: str) -> list[str]:
        return sorted(pid for pid, r in self.records.items() if r["lang"] == lang)


def build_index(
    root: Path,
    *,
    cfg: Any = None,
    docs: list[_Doc] | None = None,
    clients: Clients | None = None,
) -> Built:
    """Seed the real work-queue, build every shard, validate each, publish the manifest.

    This is ``test_index_small._build`` with the leg and client trio substituted (see the module
    docstring), plus the ``merge`` call, because a query path may not open an index whose manifest
    was never published.
    """
    cfg = cfg if cfg is not None else retrieval_cfg(root)
    documents = docs if docs is not None else retrieval_docs()
    layout = _write(root, _tables(documents), cfg)
    legs = scaled_legs()
    bundle = clients if clients is not None else Clients()
    adapter = IndexAdapter(
        pack_hash=None,
        recon_hash=_RECON,
        idx_hash=index_hash(cfg),
        base=str(root),
        legs=legs,
    )
    ctx = _index_ctx(layout, cfg, legs, adapter, bundle)
    wq = saturate.queue_for(cfg, adapter, base=root)
    assert saturate.seed(cfg, adapter, wq) == len(adapter.shard_specs(cfg))
    receipts: list[Path] = []
    while True:
        shard = saturate.workqueue.claim(wq.pending, wq.running)
        if shard is None:
            break
        out = adapter.work(ctx, shard)
        assert adapter.validate(out), f"validate failed for {shard.name}"
        saturate.workqueue.mark_done(shard, wq.done, "corpus")
        receipts.append(out)
    manifest_path = adapter.merge(cfg, receipts)
    records = {
        r["passage_id"]: r for r in iter_final_passages(layout, _RECON, pack_hash=None)
    }
    return Built(
        cfg=cfg,
        layout=layout,
        adapter=adapter,
        legs=legs,
        clients=bundle,
        docs=documents,
        records=records,
        manifest_path=manifest_path,
        root=root,
    )


def _index_ctx(layout: Layout, cfg: Any, legs, adapter: IndexAdapter, clients: Clients):
    from ragtime.preprocess.index import IndexCtx, index_build_options
    from ragtime.serving.batching import Tier

    opts = index_build_options(cfg)
    return IndexCtx(
        dense=clients.index_dense,
        milco=clients.milco,
        mtd=clients.mtd_colbert,
        opts=opts,
        legs=tuple(legs),
        layout=layout,
        recon_hash=_RECON,
        pack_hash=None,
        idx_hash=adapter.idx_hash,
        tier=Tier(
            token_budget=opts.encode_batch_token_budget, max_items=opts.encode_max_items
        ),
    )


# --------------------------------------------------------------------------- #
# pytest fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def built(tmp_path: Path) -> Built:
    """The default published fixture index, rooted under a path that is not ``runs/``."""
    return build_index(tmp_path / "not-runs")


@pytest.fixture
def ctx(built: Built):
    """A :class:`RetrievalContext` over ``built``, with the in-memory store injected."""
    from ragtime.retrieval import bring_up

    return bring_up(
        built.cfg,
        built.clients,
        built.layout,
        recon_hash=built.recon_hash,
        pack_hash=None,
        idx_hash=built.idx_hash,
        passage_store=built.store(),
        legs=built.legs,
        stats=Statistics(),
    )


def context_for(
    built: Built, cfg: Any, *, clients: Clients | None = None, reranker: Any = None
):
    """A context over the same published index under a different query-time config.

    Each context gets its own :class:`SpyReranker` unless one is passed: several contexts in
    one test share the fixture's ``Clients`` bundle (that sharing is the point of SM-R13), and
    a shared call log would let one context's rerank call be read as another's: a test that
    passes for the wrong reason.
    """
    from ragtime.retrieval import bring_up

    ctx = bring_up(
        cfg,
        clients if clients is not None else built.clients,
        built.layout,
        recon_hash=built.recon_hash,
        pack_hash=None,
        idx_hash=built.idx_hash,
        passage_store=built.store(),
        legs=built.legs,
        stats=Statistics(),
    )
    ctx.reranker = reranker if reranker is not None else SpyReranker()
    return ctx


def leg_names() -> tuple[str, ...]:
    return tuple(LEGS)
