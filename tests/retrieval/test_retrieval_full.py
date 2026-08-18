"""The retrieval query stack against real artifacts and real checkpoints (FL-R01..FL-R13).

There are no qrels, no dev set and no training data, and none are coming. Every test here is a
liveness or consistency check: does the shipped query path return something structurally sane over
real data. None is a quality measurement. Nothing computes R@k, nDCG or precision, and no number
produced here may be quoted as "how good is retrieval". Where a number looks like a quality metric
(an overlap fraction, a rank, a Kendall tau) it is a liveness signal, and the docstring beside it
says so again.

Two fixtures, two blocking states, both labelled.

* FL-R01..FL-R07 run against the real, bounded-shard index ``tests/preprocess/test_index_full.py``
  publishes: 200 real documents per language carved out of the real ``final/<recon12>/`` tables,
  built with the real BGE-M3 / MILCO-650m / MTD-PyLate checkpoints through the real work-queue.
  They need no corpus-scale artifact.
* FL-R08..FL-R13 are [BLOCKED on corpus-scale index]. They need the ~9.9 M-passage build's own
  ``manifest.json``, and they skip with a message that names what is missing. None of them carries
  a weakened assertion standing in for the real one.

A second block is stated rather than absorbed: all six run files set
``retrieval.reranker.model: "Qwen3-Reranker-4B"``, which is not a resolvable checkpoint id (the
Hugging Face repo is ``Qwen/Qwen3-Reranker-4B``) and is not in the local cache. The tests that need
a live cross-encoder skip with that config leaf named rather than silently substituting another
model. See :func:`_require_reranker`.
"""

from __future__ import annotations

import json
import os
import time
import types
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout, Statistics
from ragtime.common.ids import doc_id_of
from ragtime.common.io import is_done, read_jsonl
from ragtime.common.passage_store import RENDERINGS, LmdbPassageStore
from ragtime.common.topics import CANONICAL_TOPICS_REL, Topic, load_topics
from ragtime.config import all_hashes
from ragtime.config import load as load_config
from ragtime.orchestration import saturate
from ragtime.orchestration.cli import artifact_root
from ragtime.orchestration.run_identity import run_family

# The production dense-gate predicate, imported and never re-implemented: FL-R11 and the
# corpus-acceptance pass (``acceptance.phase_b1``) must apply the identical "rank 0, or tied at
# the maximum with a byte-identical duplicate" rule, and a second copy of its float32 tolerance
# here is exactly how the two would silently drift apart.
from ragtime.preprocess.acceptance import duplicate_tie_explains_miss
from ragtime.preprocess.index import (
    DENSE_LEG,
    LATE_INTERACTION_LEG,
    LEGS,
    SHARED_LANG,
    SPARSE_LEG,
    default_legs,
    index_hash,
    kendall_tau_b,
    rbo,
    read_shard_parts,
)
from ragtime.preprocess.packing import packing_hash
from ragtime.preprocess.passage_store_build import PassageStoreAdapter
from ragtime.preprocess.reconcile import reconcile_hash
from ragtime.retrieval import bring_up, display, legs, retrieve
from ragtime.retrieval import service as service_mod
from ragtime.retrieval.concurrency import plan_query_concurrency

# The real bounded-shard index fixtures, reused rather than rebuilt (session-scoped, so the
# 800-document 9-leg-variant build happens once even when both files run).
from tests.harness import spy_through
from tests.preprocess.test_index_full import (  # noqa: F401 - re-exported as fixtures
    _CONFIG,
    bounded,
    built,
)

pytestmark = pytest.mark.full

_TOP_K = 10
_MEASURED: dict[str, Any] = {}
_TOPICS = Path(__file__).resolve().parents[2] / CANONICAL_TOPICS_REL


def _topic_query(topic: Topic) -> str:
    """The query text a topic contributes to the latency sweep.

    ``problem_statement`` and not ``background``: the loop's own ``search`` action is issued
    against the request's statement, and the background is prose the decomposer reads. The
    sweep exists to characterize real query lengths, so it uses the real field.
    """
    return topic.problem_statement


# --------------------------------------------------------------------------- #
# Timing reports + the two blocking preconditions
# --------------------------------------------------------------------------- #
def _perf(name: str, value: str, *, unit: str) -> str:
    """Report one measurement as ``<name>: <value> for <unit>``, and record it under ``name``.

    A duration is unreadable without what it covered, so the unit travels with the value and a
    long job says on its own what each measurement cost. This is the only way a number leaves
    this file.
    """
    line = f"{name}: {value} for {unit}"
    print(line)
    _MEASURED[name] = line
    return line


def _require_reranker(clients: Any) -> Any:
    """The real cross-encoder, or a labelled skip naming the config leaf that blocks it.

    Not substituting another checkpoint: a rerank test that quietly scored with
    ``ms-marco-MiniLM`` would be green and would prove nothing about the model the run record
    names.
    """
    reranker = clients.reranker
    # Resolve the loader as an attribute first, outside the try. An AttributeError here means the
    # class no longer exposes this entry point, which is a real failure of the suite and must never
    # be reported as a skip. That is not hypothetical: this probe used to call the private
    # ``_cross_encoder()``, which went away when the reranker moved from the
    # CrossEncoder/AutoModelForSequenceClassification path to AutoModelForCausalLM. The broad
    # ``except Exception`` below then swallowed the AttributeError and skipped, blaming
    # ``retrieval.reranker.model``, a config leaf that was already correct in all six run files,
    # so the integration test for the fusion fix did not run while the gate reported success.
    # Probe the public entry point, and let a missing one raise.
    load = reranker.load
    try:
        load()
    except Exception as exc:  # noqa: BLE001 - a genuine load failure is the same block
        pytest.skip(
            "[BLOCKED: reranker checkpoint] could not load the reranker named by "
            f"config/*.yml retrieval.reranker.model = {reranker.model!r} "
            f"({exc.__class__.__name__}: {exc}). Not substituting another "
            "checkpoint: a rerank test that quietly scored with ms-marco-MiniLM would be green "
            "and would prove nothing about the model the run record names. not a pass."
        )
    return reranker


#: The corpus-scale contexts, built once per pytest process, keyed on the rendering searched.
#: Value: ``(cfg, manifest, ctx, fingerprint)``, see :func:`_corpus_context`.
_CORPUS_CONTEXTS: dict[str, tuple[Any, Any, Any, tuple[Any, ...]]] = {}


def _ctx_fingerprint(ctx: Any) -> tuple[Any, ...]:
    """The configuration of a shared context, as a comparable tuple.

    Every field here is something a test could mutate on a context it was handed, thereby changing
    what the next test measures. ``index_ctx.query_workers`` is the sharp one: the handed-out view
    is a shallow copy, so writing ``ctx.leg_workers`` is isolated while writing
    ``ctx.index_ctx.query_workers`` reaches through to the shared object. That asymmetry is what
    ``_set_execution`` uses deliberately for the bounded-shard FL-R14, and it is why
    ``_set_execution`` must never be pointed at a corpus context.
    """
    return (
        ctx.index,
        ctx.passage_lang,
        ctx.leg_workers,
        ctx.index_ctx.query_workers,
        id(ctx.reranker),
        id(ctx.passage_store),
        tuple(sorted(ctx.cells)),
        tuple(ctx.leg_names),
        tuple((lang, len(ctx.cells[lang].parts)) for lang in sorted(ctx.cells)),
    )


def _corpus_context(index: str = "original"):
    """A context over the real corpus-scale index, or a labelled skip. FL-R08..FL-R13.

    Manifest-verified before anything is searched: every part of every cell must be on disk
    (``read_shard_parts``), which is the "did we search a complete index" check the production
    unit gate owes.

    Cached per rendering for the life of the pytest process: an uncached helper rebuilds the whole
    context on every call, so a five-test corpus gate would pay the cold open five times.

    The cache holds the expensive half: the opened cells, their readers, the ``IndexCtx`` and the
    LMDB passage store. Each caller gets a shallow copy (``dataclasses.replace``) carrying a fresh
    :class:`Statistics` and ``_owns_store=False``, so per-test counters cannot accumulate across
    tests and no test can close a store another test is still reading. Peak resident memory does
    not rise, because one rendering is resident either way.

    The leak is checked rather than assumed. :func:`_ctx_fingerprint` is recorded when the context
    is built and re-verified on every handout, so a test that reached through the shallow copy into
    a shared object (the ``index_ctx.query_workers`` case above) fails loudly here instead of
    silently changing what the next test measures. Its fast twin is
    ``test_full_harness_twins_small.py::test_the_shared_corpus_context_view_isolates_per_test_state``.

    One ordering consequence: FL-R08's reported numbers are the cold ones only because it is
    defined, and therefore collected, before the other corpus tests. Run alone or reordered after
    them, its wall time and ``ru_maxrss`` describe a warm process and must not be read as
    cold-open figures.
    """
    cached = _CORPUS_CONTEXTS.get(index)
    if cached is not None:
        cfg, manifest, ctx, fingerprint = cached
        assert _ctx_fingerprint(ctx) == fingerprint, (
            f"the shared corpus context for index={index!r} was MUTATED by an earlier test: "
            f"built as {fingerprint}, now {_ctx_fingerprint(ctx)}. A shared context must stay "
            "read-only: change a per-test knob on the shallow copy you were handed, or build "
            "your own context if the test genuinely needs a different one."
        )
        return cfg, manifest, replace(ctx, stats=Statistics(), _owns_store=False)

    cfg = load_config(_CONFIG)
    root = Path(__file__).resolve().parents[2] / artifact_root(cfg)
    layout = Layout(
        run_dir=root,
        base=root,
        family=run_family(cfg),
        chunker_hash=all_hashes(cfg)["chunker"],
    )
    recon, pack, idx = reconcile_hash(cfg), packing_hash(cfg), index_hash(cfg)
    manifest_path = layout.index_manifest_path(recon, idx)
    if not (manifest_path.exists() and is_done(manifest_path)):
        pytest.skip(
            f"[BLOCKED on corpus-scale index] {manifest_path} does not exist: the ~9.9 M-passage "
            "vectorize+assemble build has not published a manifest yet (the index build is `reopened`). "
            "This test is WRITTEN and UNRUN, not passing."
        )
    manifest = read_jsonl(manifest_path)[0]
    for source_lang, entry in manifest["variants"][index]["shards"].items():
        cell = layout.index_lang_dir(recon, idx, index, source_lang)
        census = read_shard_parts(cell)  # raises if a part is missing: never a short fan
        assert census["parts"] == entry["parts"], (source_lang, census, entry)
    store_path = layout.passage_store_path(recon, pack)
    if not (store_path.exists() and is_done(store_path)):
        pytest.skip(
            f"[BLOCKED on corpus-scale index] {store_path} does not exist: the by-id passage "
            "store has not been built corpus-wide (see FL-R10). not a pass."
        )
    from ragtime.serving.registry import build_clients

    clients = build_clients(cfg)
    ctx = bring_up(
        _query_cfg(cfg, index=index),
        clients,
        layout,
        recon_hash=recon,
        pack_hash=pack,
        idx_hash=idx,
        legs=default_legs(),
        stats=Statistics(),
    )
    _CORPUS_CONTEXTS[index] = (cfg, manifest, ctx, _ctx_fingerprint(ctx))
    return cfg, manifest, replace(ctx, stats=Statistics(), _owns_store=False)


# --------------------------------------------------------------------------- #
# Bounded-shard plumbing
# --------------------------------------------------------------------------- #
def _query_cfg(cfg: Any, *, index: str | None = None, passage_lang: str | None = None,
               depth: int | None = None, rrf_k: int | None = ...) -> Any:
    """A QUERY-time variation of ``cfg``. The hashed blocks are copied, never edited.

    Asserted below: ``index_hash`` is unchanged, so every variation addresses the same
    published index: which is what lets a difference be attributed to the knob.
    """
    # ``Mapping``, not ``dict``: a real ``RunConfig``'s blocks are nested ``MappingProxyType``,
    # so an ``isinstance(v, dict)`` copy would silently leave them frozen and the ``depth``
    # write below would raise on the real config while passing on a hand-built one.
    blocks = {
        k: dict(v) if isinstance(v, Mapping) else v for k, v in dict(cfg.blocks).items()
    }
    retrieval = {
        k: (dict(v) if isinstance(v, Mapping) else v)
        for k, v in dict(blocks.get("retrieval") or {}).items()
    }
    if depth is not None:
        retrieval["reranker"] = {**dict(retrieval.get("reranker") or {}), "depth": depth}
    if rrf_k is not ...:
        rrf_block = dict(retrieval.get("rrf") or {})
        if rrf_k is None:
            rrf_block.pop("k", None)
        else:
            rrf_block["k"] = rrf_k
        retrieval["rrf"] = rrf_block
    blocks["retrieval"] = retrieval
    out = types.SimpleNamespace(
        run_id=cfg.run_id,
        languages=tuple(cfg.languages),
        blocks=blocks,
        retrieval_index=index or cfg.retrieval_index,
        passage_lang=passage_lang or cfg.passage_lang,
    )
    assert index_hash(out) == index_hash(cfg), "a query-time knob moved the index recipe hash"
    return out


def _clients_for(built_fixture: Any, *, reranker: Any = None) -> types.SimpleNamespace:
    """A ``ClientBundle``-shaped view over the already-resident index encoders.

    The three encoders are the objects ``IndexAdapter.bringup`` took from the registry, so this
    stands up no second BGE-M3, MILCO or MTD: the single-serving rule applied to the harness as
    well as to the code. ``embedder`` is a deliberate sentinel, and touching it would be the silent
    wrong-embedding-space bug SM-R13 pins hermetically.
    """
    ctx = built_fixture.ctx
    return types.SimpleNamespace(
        index_dense=ctx.dense,
        milco=ctx.milco,
        mtd_colbert=ctx.mtd,
        embedder=object(),
        reranker=reranker,
    )


def _context(built_fixture: Any, *, index: str = "original", passage_lang: str = "original",
             depth: int = 0, store: Any = None, reranker: Any = None):
    b = built_fixture.bounded
    return bring_up(
        _query_cfg(b.cfg, index=index, passage_lang=passage_lang, depth=depth),
        _clients_for(built_fixture, reranker=reranker),
        b.layout,
        recon_hash=b.recon,
        pack_hash=b.pack,
        idx_hash=built_fixture.ctx.idx_hash,
        passage_store=store if store is not None else _memory_store(b),
        legs=default_legs(),
        stats=Statistics(),
    )


def _memory_store(bounded_fixture: Any):
    from ragtime.common import PassageStore

    return PassageStore.from_records(bounded_fixture.records.values())


def _queries(built_fixture: Any, lang: str, n: int = 3) -> list[dict[str, Any]]:
    """The ``n`` LONGEST passages of ``lang``: deterministic, fixed before any result is seen."""
    rows = [r for r in built_fixture.bounded.records.values() if r["lang"] == lang]
    rows.sort(key=lambda r: (-int(r["token_count"] or 0), r["passage_id"]))
    return rows[:n]


class _SpyReranker:
    """Wraps a reranker to RECORD what it was handed. Every call delegates; nothing is faked."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model(self) -> str:
        return self.inner.model

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return self.inner.score(query, passages)


# =========================================================================== #
# FL-R01..FL-R07: the real bounded-shard index, needing no corpus-scale artifact
# =========================================================================== #
def test_flr01_end_to_end_retrieve_over_real_bounded_shard_passages(built) -> None:  # noqa: F811
    """FL-R01 [liveness]: legs -> within-cell raw merge -> RRF, over real multilingual text.

    Ids and scores only, ``top_k`` honoured, and self-retrieval recorded as a measurement through
    the whole pipeline, which is a stronger integration claim than the index build's leg-level
    FL04: that one never crossed the fusion layer. It is a liveness signal, not a quality bar. The
    rank is reported, and only the dense leg's own rank 0 is asserted.
    """
    ctx = _context(built, depth=0)
    assert sorted(ctx.cells) == ["en", "es", "ru", "zh"]

    ranks: dict[str, int] = {}
    for lang in ("en", "es", "ru", "zh"):
        record = _queries(built, lang, 1)[0]
        out = retrieve(ctx, record["original"], top_k=_TOP_K)
        assert out and len(out) <= _TOP_K
        assert all(len(hit) == 2 for hit in out)
        assert all(isinstance(pid, str) and isinstance(score, float) for pid, score in out)
        assert all(pid in built.bounded.records for pid, _ in out)
        ranks[lang] = next(
            (i for i, (pid, _) in enumerate(out) if pid == record["passage_id"]), -1
        )
        # the fused list is deterministic: two runs of one query agree exactly
        assert retrieve(ctx, record["original"], top_k=_TOP_K) == out

        # the dense leg alone carries the rank-1 guarantee, and it is asserted per cell
        per_leg = legs(ctx, record["original"], top_k=_TOP_K, source_lang=lang)
        assert per_leg[DENSE_LEG][0][0] == record["passage_id"]
        assert per_leg[DENSE_LEG][0][1] == pytest.approx(1.0, abs=1e-4)
        for leg in (SPARSE_LEG, LATE_INTERACTION_LEG):
            assert per_leg[leg], f"{leg} returned nothing for a real {lang} query"
    _MEASURED["fl_r01_fused_self_rank"] = ranks


def test_flr02_search_one_rendering_display_another_over_real_text(built, tmp_path) -> None:  # noqa: F811
    """FL-R02 [consistency]: "search OMT, display Original" on real translated text.

    The store is the one ``PassageStoreAdapter`` built over this shard's real ``final/``
    tree, so the read half of the knob pair is exercised through its production backend rather
    than through an in-memory fixture.
    """
    store_path = _build_store(built)
    store = LmdbPassageStore(store_path)
    try:
        ids = [r["passage_id"] for r in _queries(built, "zh", 2)]
        searched = _context(built, index="omt", passage_lang="original", store=store)
        hits = retrieve(searched, "harbour freight report", top_k=_TOP_K)
        assert all(len(hit) == 2 for hit in hits)

        for lang in RENDERINGS:
            got = display(searched, ids, lang)
            assert [d for d, _ in got] == [doc_id_of(pid) for pid in ids]
            assert [t for _, t in got] == [
                built.bounded.records[pid][lang] for pid in ids
            ]
        # the same ids display identically no matter which index was searched
        for index in RENDERINGS:
            other = _context(built, index=index, passage_lang="original", store=store)
            assert display(other, ids, "original") == display(searched, ids, "original")
    finally:
        store.close()


def test_flr03_rerank_scores_the_searched_rendering_on_real_text(built) -> None:  # noqa: F811
    """FL-R03 [consistency]: the real cross-encoder is handed ``render(pid, retrieval.index)``.

    Real tokenisation, real truncation, real text: the thing the hermetic hash stand-in cannot
    see. Skips with a label if the configured reranker checkpoint does not resolve.
    """
    from ragtime.serving.registry import build_clients

    clients = build_clients(built.bounded.cfg)
    spy = _SpyReranker(_require_reranker(clients))
    ctx = _context(built, index="omt_opus", passage_lang="original", depth=5, reranker=spy)

    divergent = [
        pid
        for pid, record in built.bounded.records.items()
        if record["omt_opus"] != record["original"] and record["lang"] != SHARED_LANG
    ]
    assert divergent, "the real shard carries no passage whose two renderings differ"

    retrieve(ctx, "port authority berth allocation", top_k=5)
    (query, texts), = spy.calls
    assert query == "port authority berth allocation"
    searched = {r["omt_opus"] for r in built.bounded.records.values()}
    assert set(texts) <= searched
    native_only = {
        r["original"] for r in built.bounded.records.values() if r["lang"] != SHARED_LANG
    }
    assert not (set(texts) & (native_only - searched))


def test_flr05_cross_rendering_agreement_between_the_two_mt_tiers(built) -> None:  # noqa: F811
    """FL-R05 [recorded measurement, never a pass/fail quality bar].

    ``omt`` vs ``omt_opus`` top-k agreement through the full fused path, via the shared
    ``kendall_tau_b``/``rbo`` helpers. This is not a quality comparison of the two MT tiers. It is
    a degeneracy smoke test: two different translations of one corpus should produce rankings that
    are neither identical nor unrelated, and either extreme would indicate a plumbing defect. No
    threshold is asserted on the value.
    """
    high = _context(built, index="omt", depth=0)
    low = _context(built, index="omt_opus", depth=0)
    measurements = []
    for lang in ("es", "ru", "zh"):
        query = " ".join(_queries(built, lang, 1)[0]["omt"].split()[:12])
        a = [pid for pid, _ in retrieve(high, query, top_k=_TOP_K)]
        b = [pid for pid, _ in retrieve(low, query, top_k=_TOP_K)]
        assert a and b
        measurements.append(
            {
                "lang": lang,
                "kendall_tau_b": round(kendall_tau_b(a, b), 4),
                "rbo": round(rbo(a, b), 4),
                "overlap": len(set(a) & set(b)),
            }
        )
    _MEASURED["fl_r05_cross_rendering"] = measurements


def test_flr06_id_sets_are_identical_across_variants_through_m06s_own_entrypoint(built) -> None:  # noqa: F811
    """FL-R06 [consistency]: the producer/consumer contract, checked from the consumer side.

    Collected through ``bring_up`` and ``LangHandle``/``LegHandle.passage_ids``, the path a query
    actually walks, rather than by re-trusting the index build's own FL02 over the same artifacts.
    """
    by_variant: dict[str, dict[str, set[str]]] = {}
    for variant in RENDERINGS:
        ctx = _context(built, index=variant, depth=0)
        per_lang: dict[str, set[str]] = {}
        for lang, handle in ctx.cells.items():
            ids: set[str] = set()
            for part in handle.parts:
                for leg in LEGS:
                    leg_ids = set(part.passage_ids(leg))
                    assert leg_ids, (variant, lang, leg)
                    ids |= leg_ids
            per_lang[lang] = ids
        by_variant[variant] = per_lang

    for lang in by_variant["original"]:
        sets = [by_variant[v][lang] for v in RENDERINGS]
        assert sets[0] == sets[1] == sets[2], f"{lang}: id sets differ across renderings"
        assert sets[0] == built.bounded.ids_by_lang[lang]


def test_flr07_passage_store_adapter_over_the_real_final_tree(built) -> None:  # noqa: F811
    """FL-R07 [liveness + timing]: the store build driver over the bounded shard's real ``final/``.

    Atomic ``_SUCCESS``, keyed by ``(recon_hash, pack_hash)``, second run a no-op. The wall time
    is reported at this size as an anchor for FL-R10's corpus-scale number, never a substitute
    for it: the two differ by ~4 orders of magnitude in passage count.
    """
    b = built.bounded
    start = time.perf_counter()
    store_path = _build_store(built)
    elapsed = time.perf_counter() - start

    assert store_path == b.layout.passage_store_path(b.recon, b.pack)
    assert store_path.is_dir() and is_done(store_path)
    assert not list(store_path.parent.glob("*.tmp-*"))

    store = LmdbPassageStore(store_path)
    try:
        for record in list(b.records.values())[:50]:
            for rendering in RENDERINGS:
                assert store.render(record["passage_id"], rendering) == record[rendering]
    finally:
        store.close()

    mtimes = {p: p.stat().st_mtime_ns for p in sorted(store_path.rglob("*"))}
    assert _build_store(built) == store_path
    assert {p: p.stat().st_mtime_ns for p in sorted(store_path.rglob("*"))} == mtimes

    _perf(
        "fl_r07_store_build",
        f"{elapsed:.1f} s",
        unit=(
            f"{len(b.records)} passages x {len(RENDERINGS)} renderings, 1 process / 1 shard, "
            "PassageStoreAdapter over the bounded shard's real final/ tree"
        ),
    )


def test_flr14_the_leg_fan_is_result_identical_on_real_engines(built) -> None:  # noqa: F811
    """FL-R14 [equality proof]: the ``(query, leg, cell)`` fan changes latency, never the answer.

    SM-R21 proves this hermetically over the fixture's scaled stand-in encoders. This proves it
    where it can actually go wrong: over the real FAISS, Seismic and PyLate-PLAID engines, whose
    concurrency behaviour belongs to them, not to us. FAISS releases the GIL, ``pyseismic-lsr``
    does not, and torch does its own intra-op threading. Compared: the whole ranked list with its
    scores, and the pools before fusion, at five widths, on four real multilingual queries.

    The wall times are recorded but are not the fan's speed-up. This is a 200-document bounded
    shard with one part per cell, a unit too small to extrapolate from. The fan's real numbers come
    from the corpus-scale benchmark.
    """
    ctx_seq = _context(built, depth=0)
    _set_execution(ctx_seq, leg_workers=1)
    assert ctx_seq.leg_workers == 1

    for lang in ("en", "es", "ru", "zh"):
        query = _queries(built, lang, 1)[0]["original"]
        expected = retrieve(ctx_seq, query, top_k=_TOP_K)
        assert expected, f"{lang} retrieved nothing: the comparison would be vacuous"
        pools_expected = service_mod._pools(ctx_seq, [query], _TOP_K)
        assert len(pools_expected) == len(LEGS) * 4 == 12

        for width in (2, 3, 12, 32):
            ctx = _context(built, depth=0)
            _set_execution(ctx, leg_workers=width)
            assert ctx.leg_workers == width
            assert ctx.index_ctx.query_workers == 1, "the inner fan must be bounded off"
            assert retrieve(ctx, query, top_k=_TOP_K) == expected, (
                f"leg fan width {width} changed the {lang} answer on real engines"
            )
            assert service_mod._pools(ctx, [query], _TOP_K) == pools_expected

    # A multi-string call: two units of the fan then address the same (leg, cell), which is the
    # only shape that can race the lazily-opened readers `_warm_cells` exists to protect.
    multi = [_queries(built, lang, 1)[0]["original"] for lang in ("es", "ru")]
    expected = retrieve(ctx_seq, multi, top_k=_TOP_K)
    ctx = _context(built, depth=0)
    _set_execution(ctx, leg_workers=12)
    assert retrieve(ctx, multi, top_k=_TOP_K) == expected


def _set_execution(ctx: Any, *, leg_workers: int) -> None:
    """Re-plan ``ctx``'s two fan widths as if the config had named ``query_leg_workers``.

    ``bring_up`` resolves the pair once, so a width is changed the way a run changes it: through
    :func:`ragtime.retrieval.concurrency.plan_query_concurrency`, both halves together, rather than
    by poking ``leg_workers`` and leaving the inner width at whatever the previous plan chose.
    """
    plan = plan_query_concurrency(
        types.SimpleNamespace(blocks={"execution": {"query_leg_workers": leg_workers}})
    )
    ctx.leg_workers = plan.leg_workers
    ctx.index_ctx.query_workers = plan.part_workers


def _build_store(built_fixture: Any) -> Path:
    """Drive the real ``PassageStoreAdapter`` over the bounded shard (idempotent)."""
    b = built_fixture.bounded
    adapter = PassageStoreAdapter.for_config(b.cfg, base=b.root)
    wq = saturate.queue_for(b.cfg, adapter, base=b.root)
    saturate.seed(b.cfg, adapter, wq)
    ctx = adapter.bringup(b.cfg)
    receipts: list[Path] = []
    while True:
        shard = saturate.workqueue.claim(wq.pending, wq.running)
        if shard is None:
            break
        out = adapter.work(ctx, shard)
        assert adapter.validate(out)
        saturate.workqueue.mark_done(shard, wq.done, "corpus")
        receipts.append(out)
    return adapter.out_path(b.cfg) if not receipts else adapter.merge(b.cfg, receipts)


# =========================================================================== #
# FL-R08..FL-R13: [BLOCKED on corpus-scale index]
#
# Each of these skips with a message naming what is missing. None carries a weakened assertion
# standing in for the real one.
# =========================================================================== #
def test_flr08_the_production_unit_gate_one_real_query(  # [BLOCKED on corpus-scale index]
) -> None:
    """FL-R08 [production-unit gate]: one real query against the real corpus index.

    The unit is one ``service.search`` call, manifest-verified complete before it runs (every part
    of every cell present on disk, via ``read_shard_parts``, inside :func:`_corpus_context`), with
    a real return value and a reported wall time.

    No existing timing may be quoted as a prediction for this. The one number on record, "~0.5
    s/query", was measured against a 2,001-passage PLAID part, 65x smaller than a shipped part of
    131,072, and the 76.8 s PLAID ``add_documents`` figure is void here because it was measured on
    a GPU, solo, on the build side. This test supplies the real number.
    """
    import resource

    # Resident memory is measured here as well as wall time. `_corpus_context` -> `bring_up` opens
    # every language of the rendering at once and holds them for the context's life, so by the time
    # this test has a `ctx` the process already carries the full query-time footprint. That is the
    # number deciding whether a query node is feasible at all: the acceptance harness peaks at
    # 93-145 GiB for one (variant, lang) cell, and a rendering is four of them.
    #
    # ru_maxrss is a high-water mark for the whole process, in KiB on Linux (bytes on macOS; these
    # gates run on Linux only). It is read after the search, so it covers the index open and the
    # query-time allocations together.
    _cfg, manifest, ctx = _corpus_context()
    query = _topic_query(load_topics(_TOPICS)[0])

    parts = sum(
        entry["parts"] for entry in manifest["variants"][ctx.index]["shards"].values()
    )
    start = time.perf_counter()
    out = service_mod.search(ctx, query, top_k=_TOP_K)
    elapsed = time.perf_counter() - start
    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    langs = sorted(manifest["variants"][ctx.index]["shards"])

    assert out and len(out) <= _TOP_K
    assert all(len(hit) == 2 and isinstance(hit[0], str) for hit in out)
    _perf(
        "fl_r08_unit",
        f"{elapsed:.2f} s",
        unit=(
            f"1 query, sequential, {parts * len(LEGS)} leg-part searches on index={ctx.index} "
            f"(recon={ctx.recon_hash[:12]} idx={ctx.idx_hash[:12]})"
        ),
    )
    _perf(
        "fl_r08_resident",
        f"{rss_gib:.1f} GiB",
        unit=(
            f"one full rendering resident on index={ctx.index}: {len(langs)} cells "
            f"({','.join(langs)}) / {parts} parts, all {len(LEGS)} legs open"
        ),
    )
    # The ceiling is the allocation this job actually got, never a hardcoded constant. A hardcoded
    # one fails spuriously on a larger node: the GPU nodes serving this report 512000-773000 MB and
    # a big-memory node reports 1024000 MB, against a ~395 GiB rendering, so a limit inferred from
    # a CPU partition (251000 MB) would reject a machine with twice the headroom. Deriving from
    # SLURM_MEM_PER_NODE asks "did I fit in what I was given", which is the only question with a
    # stable answer across partitions.
    alloc_mb = os.environ.get("SLURM_MEM_PER_NODE")
    ceiling_gib = (float(alloc_mb) / 1024.0) if alloc_mb else None
    if ceiling_gib:
        assert rss_gib < ceiling_gib, (
            f"FULL-RENDERING resident set {rss_gib:.1f} GiB exceeded this job's own allocation "
            f"({ceiling_gib:.0f} GiB) for index={ctx.index} ({len(langs)} cells, {parts} parts). "
            "Serving this rendering needs a larger node, or mmap for the dense leg (which would "
            "also close MAT-1 and the paper's own mmap claim), before multi-machine sharding."
        )


def test_flr09_query_latency_sweep_over_all_103_topics(  # [BLOCKED on corpus-scale index]
) -> None:
    """FL-R09 [latency benchmark, kept distinct from FL-R08's unit gate].

    All 103 report requests of the canonical topics file, never a subset: a hand-picked sample
    would not stress the real query-length and language distribution the sweep exists to
    characterize. Per-leg and fused latency are reported separately, both timed, because the choice
    between the available levers (fewer and larger parts, resident rather than mmap'd, approximate
    search) depends on which of them dominates.
    """
    _cfg, _manifest, ctx = _corpus_context()
    topics = load_topics(_TOPICS)
    assert len(topics) == 103, f"the sweep reads all 103 topics, got {len(topics)}"

    # The corpus language distribution: the denominator for "is the result mix representative?".
    # Counted exactly off the already-open handles (every part's id map), never estimated from
    # part counts: parts are equal-sized except the last of each cell, so a parts-based estimate
    # would be systematically wrong per language by up to one partial part.
    corpus_by_lang: dict[str, int] = {}
    for lang, handle in ctx.cells.items():
        corpus_by_lang[lang] = sum(len(p.passage_ids(LEGS[0])) for p in handle.parts)
    corpus_total = sum(corpus_by_lang.values())

    fused: list[float] = []
    per_leg: dict[str, list[float]] = {leg: [] for leg in LEGS}
    retrieved_by_lang: Counter[str] = Counter()
    per_topic_lang: list[dict[str, int]] = []
    for topic in topics:
        query = _topic_query(topic)
        start = time.perf_counter()
        hits = retrieve(ctx, query, top_k=_TOP_K)
        fused.append(time.perf_counter() - start)
        # The language of a hit is a property of the passage, read from the store, not inferred
        # from which cell produced it (the fused list carries no provenance, by design: it returns
        # ids and scores only).
        langs_here: Counter[str] = Counter()
        for pid, _score in hits:
            langs_here[ctx.passage_store.passage(pid).lang] += 1
        retrieved_by_lang.update(langs_here)
        per_topic_lang.append(dict(langs_here))
        for leg in LEGS:
            leg_start = time.perf_counter()
            for lang in sorted(ctx.cells):
                service_mod.query_lang_leg(ctx.cells[lang], leg, query, _TOP_K)
            per_leg[leg].append(time.perf_counter() - leg_start)

    # Representativeness is reported, not asserted. Equal-depth pools plus RRF across languages
    # make the fused mix tend towards 1/n_languages whatever the corpus share, which is a design
    # choice, language balance for a multilingual task, and not a defect. Uniform RRF is also what
    # keeps retrieval identical across renderings, so multilingual information retrieval compares
    # translation quality rather than fusion policy. Recording the number here means a lift far
    # from 1.0 is a decision somebody made on purpose rather than a surprise found after scoring.
    retrieved_total = sum(retrieved_by_lang.values()) or 1
    representativeness = {
        lang: {
            "corpus_share": corpus_by_lang.get(lang, 0) / max(1, corpus_total),
            "result_share": retrieved_by_lang.get(lang, 0) / retrieved_total,
            "lift": (
                (retrieved_by_lang.get(lang, 0) / retrieved_total)
                / max(1e-9, corpus_by_lang.get(lang, 0) / max(1, corpus_total))
            ),
        }
        for lang in sorted(set(corpus_by_lang) | set(retrieved_by_lang))
    }
    _MEASURED["fl_r09_language_mix"] = json.dumps(representativeness, sort_keys=True)
    # A topic whose results are entirely one language is worth seeing separately from the mean:
    # averages hide per-topic collapse, and a monolingual result set for a multilingual request
    # is a retrieval failure the aggregate would not show.
    _MEASURED["fl_r09_monolingual_topics"] = str(
        sum(1 for d in per_topic_lang if len(d) == 1)
    )

    _perf(
        "fl_r09_fused",
        f"mean {sum(fused) / len(fused):.2f} s, max {max(fused):.2f} s",
        unit=(
            f"1 query (103 topics), sequential, fused path incl. rerank on index={ctx.index} "
            f"(idx={ctx.idx_hash[:12]})"
        ),
    )
    for leg, samples in per_leg.items():
        _perf(
            f"fl_r09_{leg}",
            f"mean {sum(samples) / len(samples):.2f} s",
            unit=f"1 query, sequential, all language cells of leg={leg} on index={ctx.index}",
        )


def test_flr10_corpus_scale_passage_store_build(  # [BLOCKED on corpus-scale index]
) -> None:
    """FL-R10 [timing]: the corpus-scale ``PassageStoreAdapter`` build time, reported.

    The ~952 s ``iter_final_passages`` figure is an anchor from a different code path (the
    same sequential read, with no LMDB writes) and is explicitly void as a prediction. This test
    supplies the real number, or it does not run.
    """
    cfg = load_config(_CONFIG)
    root = Path(__file__).resolve().parents[2] / artifact_root(cfg)
    adapter = PassageStoreAdapter.for_config(cfg, base=root)
    passages = adapter._layout(cfg).final_passages_path(adapter.recon_hash, adapter.pack_hash)
    if not is_done(passages):
        pytest.skip(
            f"[BLOCKED on corpus-scale index] {passages} is absent: the packed corpus this "
            "store derives from has not been published. not a pass."
        )
    start = time.perf_counter()
    ctx = adapter.bringup(cfg)
    shard = next(iter(adapter.shards(cfg)))
    shard_file = root / "wq-passage-store" / shard.name
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    import json

    shard_file.write_text(json.dumps(shard.payload), encoding="utf-8")
    receipt = adapter.work(ctx, shard_file)
    elapsed = time.perf_counter() - start
    assert adapter.validate(receipt)
    _perf(
        "fl_r10_store",
        f"{elapsed / 60:.1f} min",
        unit=(
            "whole corpus, ~9.4M passages x 3 renderings, 1 process / 1 shard (LMDB has one "
            f"writer), recon={adapter.recon_hash[:12]} pack={str(adapter.pack_hash)[:12]}"
        ),
    )


def test_flr11_corpus_scale_validation_without_qrels(  # [BLOCKED on corpus-scale index]
) -> None:
    """FL-R11 [recorded liveness and consistency battery, explicitly not a quality claim].

    Self-retrieval probes through ``retrieval.service`` at the corpus-acceptance pass's own
    sampling depth. Dense carries the only hard assertion, because rank 0 is a theorem only for the
    dense leg's L2-normalised exact inner product; sparse and late-interaction are recorded. The
    bounded gate measured dense 45/45, sparse 44/45, late-interaction 43/45, which are liveness
    signals and may never be read as retrieval quality.

    The dense gate is "rank 0, or tied at the maximum with a byte-identical duplicate", which is
    the full theorem rather than a softened one. A flat
    ``per_leg[DENSE_LEG][0][0] == passage_id`` asserts something the corpus makes false: BGE-M3
    vectors are unit-normalised and searched under exact ``IndexFlatIP``, so ``<v, d> <= 1`` for
    every indexed ``d``, with equality at every ``d`` whose vector equals ``v`` and not only at
    ``d == v``. The real corpus carries ~0.78 % exact duplicate vectors, enriched 12-18x at part
    boundaries, and the sampler below over-samples those boundaries, so a saturated top-k window is
    designed into this probe set. When it happens, ``query_lang_leg``'s ``(-score, passage_id)``
    tie-break picks the lexicographically smaller id deterministically, and which duplicate that is
    carries no information about the index.

    On the shipped corpus index this probe set reads 86 rank-0 of 90: 4 misses, 4 of 4 exact ties
    at the maximum, 0 unexplained. Each tie is a pair of passages with byte-identical text,
    byte-identical stored vectors and bit-identical scores, well clear of the next distinct score.
    The corpus-acceptance gate (FA01) sees the same picture on the same index: dense 497/500, 3
    tie-explained, 0 unexplained, ``dense_hard_gate_passed: true``.

    What still fails is the defect class that matters. The rule is not "ignore misses"; it is
    ``preprocess.acceptance.duplicate_tie_explains_miss``, imported rather than re-implemented so
    this test and the production validator cannot drift apart on the tolerance. A competitor at any
    score below the maximum means the probe's vector is not where the id map says it is, a mis-keyed
    idmap, and that raises, as does a miss with no recorded evidence, and as does a probe that falls
    out of a top-k window which is not saturated at the maximum.

    The battery is exhaustive: all 90 probes run, three legs each, with the legs sequential, and the
    late-interaction leg dominates whenever it falls back to CPU. It is also memory-bound, since one
    rendering stays resident throughout.
    """
    _cfg, _manifest, ctx = _corpus_context()
    import random

    rng = random.Random(20260805)  # a fixed literal: two runs probe the identical passages
    probes = []
    for lang in sorted(ctx.cells):
        handle = ctx.cells[lang]
        # boundary-adjacent parts are always sampled (that is where the real corpus
        # concentrates its byte-identical duplicates: 13.7 % at ordinal 0 vs 1.1 % mid-part)
        indexes = sorted(
            {0, len(handle.parts) - 1,
             *rng.sample(range(len(handle.parts)), min(3, len(handle.parts)))}
        )
        for part_index in indexes:
            part = handle.parts[part_index]
            ids = part.passage_ids(DENSE_LEG)
            ordinals = {0, len(ids) - 1, *rng.sample(range(len(ids)), min(3, len(ids)))}
            probes.extend((lang, ids[o]) for o in sorted(ordinals))

    hard, tied = 0, 0
    unexplained: list[dict[str, Any]] = []
    soft = {SPARSE_LEG: 0, LATE_INTERACTION_LEG: 0}
    for lang, passage_id in probes:
        text = ctx.passage_store.render(passage_id, ctx.index)
        per_leg = legs(ctx, text, top_k=_TOP_K, source_lang=lang)
        hits = per_leg[DENSE_LEG]
        ids = [pid for pid, _ in hits]
        rank = ids.index(passage_id) if passage_id in ids else -1
        if rank == 0:
            hard += 1
        else:
            # The row shape ``duplicate_tie_explains_miss`` reads, built exactly as
            # ``acceptance.phase_b1`` builds it: same fields, same "competitors are the hits
            # at or above the probe" rule, so one predicate judges both gates.
            row = {
                "passage_id": passage_id,
                "lang": lang,
                "rank": rank,
                "top_score": float(hits[0][1]) if hits else float("nan"),
                "own_score": float(hits[rank][1]) if rank >= 0 else float("nan"),
                "competitors": [
                    {"passage_id": pid, "score": score}
                    for pid, score in (hits[: rank + 1] if rank >= 0 else hits)
                ],
            }
            if duplicate_tie_explains_miss(row):
                tied += 1
            else:
                unexplained.append(row)
        for leg in (SPARSE_LEG, LATE_INTERACTION_LEG):
            soft[leg] += int(passage_id in {p for p, _ in per_leg[leg]})
    assert not unexplained, (
        "dense self-retrieval missed a probe that a byte-identical duplicate does not "
        f"explain: this IS a hard gate ({len(unexplained)} of {len(probes)} probes). A "
        "competitor below the maximum means the probe's vector is not where the id map says "
        f"it is (a mis-keyed idmap), which is the defect this gate exists to catch: "
        f"{json.dumps(unexplained, indent=2, default=str)}"
    )
    _MEASURED["fl_r11_battery"] = {
        "probes": len(probes),
        "dense_rank0": hard,
        "dense_misses_explained_by_duplicate_tie": tied,
        "dense_unexplained_misses": len(unexplained),
        **{f"{leg}_retrieved": n for leg, n in soft.items()},
        "NOTE": "liveness/consistency only, not a retrieval-quality measurement",
    }


def test_flr12_the_real_leg_part_fan_is_actually_reached(  # [BLOCKED on corpus-scale index]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FL-R12 [consistency]: one real query reaches every leg-part, counted, not assumed.

    A silently short fan (a ``read_shard_parts`` defect, a truncated cell) returns a
    full-looking result list that nothing downstream could distinguish from a complete one.
    """
    _cfg, manifest, ctx = _corpus_context()
    parts = {
        lang: entry["parts"]
        for lang, entry in manifest["variants"][ctx.index]["shards"].items()
    }
    expected = sum(parts.values()) * len(LEGS)

    searched: list[tuple[str, str, int]] = []

    # `spy_through` captures the real function before patching and hands the body only a `before`
    # hook, so a self-call is not expressible here. Looking the name up on the module inside the spy
    # resolves to the spy itself once `monkeypatch.setattr` has run, which is infinite recursion and
    # is how this failed the first time it ran at corpus scale: `RecursionError: maximum recursion
    # depth exceeded`, 39 minutes in, after the whole rendering had been loaded. The defect was
    # latent for as long as the test was written but skipped. Its fast twin is SM-R29 in
    # tests/retrieval/test_query_fan_small.py, which runs the identical harness over the fixture
    # index in milliseconds.
    def count(handle, leg, query, top_k):
        searched.extend((leg, handle.lang_dir.name, i) for i in range(len(handle.parts)))

    spy = spy_through(monkeypatch, service_mod, "query_lang_leg", before=count)
    retrieve(ctx, "port authority berth allocation", top_k=_TOP_K)
    assert spy.count == len(parts) * len(LEGS), (
        f"the spy delegated {spy.count} times for {len(parts)} cells x {len(LEGS)} legs"
    )
    assert len(searched) == expected == len(set(searched)), (
        f"the fan reached {len(searched)} leg-parts, the manifest implies {expected}"
    )
    _MEASURED["fl_r12_fan"] = {"parts_by_lang": parts, "leg_part_searches": expected}


def test_flr13_corpus_scale_id_set_integrity_through_m06s_consumption_path(  # [BLOCKED]
) -> None:
    """FL-R13 [consistency]: FL-R06 widened to the full manifest, a census read, not a scan.

    Every rendering's per-language passage count and part census must agree, which is cheap once
    the manifest exists and is exactly the producer/consumer contract retrieval depends on.
    """
    _cfg, manifest, _ctx = _corpus_context()
    per_variant = {
        variant: {
            lang: (entry["passages"], entry["parts"], tuple(entry["part_dirs"]))
            for lang, entry in manifest["variants"][variant]["shards"].items()
        }
        for variant in RENDERINGS
    }
    for lang in per_variant["original"]:
        values = {per_variant[v][lang] for v in RENDERINGS}
        assert len(values) == 1, f"{lang}: the three renderings' cells disagree: {values}"
    # and the digests the build recorded per part agree too (the Task-2 comparison rests on it)
    for lang, entry in manifest["variants"]["original"]["shards"].items():
        for part in range(entry["parts"]):
            digests = {
                manifest["variants"][v]["shards"][lang]["shard_parts"][part]["id_digest"]
                for v in RENDERINGS
            }
            assert len(digests) == 1, (lang, part, digests)
    _MEASURED["fl_r13_census"] = per_variant["original"]


def test_zz_report_measured_facts() -> None:
    """Print every measurement this gate produced, so the numbers land in the job's own log."""
    import json

    print("retrieval measured facts:\n" + json.dumps(_MEASURED, indent=2, default=str, sort_keys=True))
