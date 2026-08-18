"""The retrieval query stack, hermetic, over a published fixture index (SM-R01..SM-R20).

There are no qrels, no dev set and no training data, and none are coming. Nothing here computes
R@k, nDCG or any other quality number, and nothing here may be turned into one later.
Self-retrieval is a plumbing proof through a deterministic hash-of-text stand-in encoder: it says
the id map addresses the right vectors and the fan reaches every part, and nothing at all about
retrieval quality.

The fixture (``conftest.py``) is ``tests/preprocess/test_index_small.py``'s own corpus
generator, published through the real ``IndexAdapter`` + work-queue, with an adversarial
Spanish cell: 12 passages in 6 byte-identical pairs cut into 3 parts of 5, so one duplicate
pair straddles the part-0/part-1 boundary and another opens part 2 at ordinal 0.
"""

from __future__ import annotations

import ast
import inspect
import threading
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout, Statistics
from ragtime.common.fusion import rrf
from ragtime.common.ids import doc_id_of
from ragtime.common.passage_store import RENDERINGS
from ragtime.config import all_hashes
from ragtime.orchestration.run_identity import run_family
from ragtime.preprocess.index import (
    DENSE_LEG,
    LATE_INTERACTION_LEG,
    LEGS,
    SPARSE_LEG,
    IndexIntegrityError,
    query_lang_leg,
    query_leg,
)
from ragtime.retrieval import _index_api, bring_up, display, legs, retrieve
from ragtime.retrieval import context as context_mod
from ragtime.retrieval import display as display_mod
from ragtime.retrieval import service as service_mod
from tests.retrieval.conftest import (
    DUP_PAIRS,
    ES_PAIR_DOC,
    Built,
    Clients,
    build_index,
    context_for,
    retrieval_cfg,
)

pytestmark = pytest.mark.small

_DEPTH = 100
_TOP = 30  # the whole fixture corpus, so nothing below is a truncation artifact
_RETRIEVAL_DIR = Path(inspect.getfile(service_mod)).parent


def _cells(ctx) -> list[str]:
    return sorted(ctx.cells)


def _pools(ctx, queries: list[str], depth: int) -> list[list[tuple[str, float]]]:
    """Recompute the RRF layer's input independently of ``service``, in the same nesting order."""
    return [
        query_lang_leg(ctx.cells[lang], leg, query, depth)
        for query in queries
        for leg in LEGS
        for lang in _cells(ctx)
    ]


# --------------------------------------------------------------------------- #
# SM-R01 / SM-R02: the two-level fusion rule (the silent-ranking-bug guard)
# --------------------------------------------------------------------------- #
def test_smr01_within_a_cell_the_merge_is_a_score_sort_and_rrf_would_differ(
    built: Built, ctx
) -> None:
    """SM-R01: parts of one cell merge on raw score, and a per-part RRF is provably different.

    Both orders are computed on the same pools and compared, so the test fails if the two
    levels are ever swapped. It also pins the ``(-score, passage_id)`` tie-break where the real
    corpus concentrates its ties: a byte-identical duplicate pair straddling a part boundary.
    """
    handle = ctx.cells["es"]
    assert len(handle.parts) == 3, "a single-part cell cannot exercise the cross-part merge"

    differed: list[str] = []
    for pid in built.ids("es"):
        query = built.records[pid]["original"]
        per_part = [query_leg(part, DENSE_LEG, query, _TOP) for part in handle.parts]
        flat = [hit for pool in per_part for hit in pool]
        assert len(flat) == 12, "the fan did not reach every part"

        merged = query_lang_leg(handle, DENSE_LEG, query, _TOP)
        assert merged == sorted(flat, key=lambda kv: (-kv[1], kv[0]))[:_TOP]
        assert merged[0][1] == max(score for _, score in flat)

        naive = rrf(per_part)
        if [p for p, _ in naive] != [p for p, _ in merged]:
            differed.append(pid)
    assert differed, (
        "no query distinguished the score merge from a per-part RRF on this fixture: the "
        "test would then be vacuous, which is worse than red"
    )

    # The boundary duplicate pair: ordinals 4 (part 0) and 0 (part 1) hold identical text.
    left = f"{ES_PAIR_DOC.format(4)}#p0"
    right = f"{ES_PAIR_DOC.format(5)}#p0"
    assert built.records[left]["original"] == built.records[right]["original"]
    merged = query_lang_leg(handle, DENSE_LEG, built.records[left]["original"], _TOP)
    assert [pid for pid, _ in merged[:2]] == sorted([left, right])
    assert merged[0][1] == merged[1][1]


def test_smr02_service_delegates_the_within_cell_merge_and_never_re_implements_it(
    ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM-R02: ``query_lang_leg`` is called once per (leg, cell, query) and its pools are rrf's.

    Identity, not equality: the objects handed to ``common.fusion.rrf`` must be the very lists
    ``query_lang_leg`` returned, which is what rules out a second, private per-part pool path
    living inside ``service``.

    The spy's own bookkeeping is a ``dict`` keyed by unit and both comparisons are
    order-insensitive, because ``_pools`` fans and a spy that ran in worker threads would
    otherwise be recording completion order. That the pools nevertheless reach ``rrf`` in
    submission order is the separate, sharper assertion of
    ``test_the_leg_fan_hands_rrf_its_pools_in_submission_order_not_completion_order``
    (``test_query_fan_small.py``): it is not weakened here, it is proved there.
    """
    seen: list[tuple[str, str, str]] = []
    returned: list[Any] = []
    real = service_mod.query_lang_leg
    lock = threading.Lock()

    def spy(handle, leg, query, top_k):
        lang = handle.lang_dir.name
        out = real(handle, leg, query, top_k)
        with lock:
            seen.append((leg, lang, query))
            returned.append(out)
        return out

    fused_args: list[Any] = []
    real_rrf = service_mod.rrf

    def rrf_spy(pools, **kwargs):
        fused_args.append(list(pools))
        return real_rrf(pools, **kwargs)

    monkeypatch.setattr(service_mod, "query_lang_leg", spy)
    monkeypatch.setattr(service_mod, "rrf", rrf_spy)
    retrieve(ctx, "muelle2 muelle2 muelle2 muelle2 muelle2", top_k=_TOP)

    assert sorted(seen) == sorted(
        (leg, lang, "muelle2 muelle2 muelle2 muelle2 muelle2")
        for leg in LEGS
        for lang in _cells(ctx)
    )
    assert len(seen) == len(set(seen)), "a (leg, cell, query) was searched twice"
    assert len(fused_args) == 1
    assert sorted(id(p) for p in fused_args[0]) == sorted(id(p) for p in returned)


def test_smr03_fusion_across_legs_and_languages_is_rank_based_rrf(built: Built) -> None:
    """SM-R03: the fused order is ``common.fusion.rrf``'s, never a raw-score combination.

    The three legs are scaled 1 / 1e3 / 1e-3 apart, so any score-additive fusion
    would be dominated by the sparse leg alone and would differ from the asserted order.
    """
    cfg = retrieval_cfg(built.root, depth=0)  # no rerank: isolate the fusion layer
    ctx = context_for(built, cfg)
    query = built.records[built.ids("ru")[0]]["original"]

    got = retrieve(ctx, query, top_k=_TOP)
    pools = _pools(ctx, [query], _TOP)
    assert got == rrf(pools, k=60)[:_TOP]

    # ...and it is not what a score sum would produce (the scales are non-comparable).
    summed: dict[str, float] = {}
    for pool in pools:
        for pid, score in pool:
            summed[pid] = summed.get(pid, 0.0) + score
    by_sum = [pid for pid, _ in sorted(summed.items(), key=lambda kv: (-kv[1], kv[0]))]
    assert [pid for pid, _ in got] != by_sum

    # every leg and every language contributed a pool (a 1-cell fixture proves nothing here)
    assert len(pools) == len(LEGS) * len(_cells(ctx)) == 12


# --------------------------------------------------------------------------- #
# SM-R04 / SM-R05: search returns no text; the two knobs are independent
# --------------------------------------------------------------------------- #
class _RaisingStore:
    """A passage store that turns any text access into an exception, not a silent read."""

    def render(self, passage_id: str, passage_lang: str) -> str:  # pragma: no cover - raises
        raise AssertionError(f"the search path read text for {passage_id!r}")

    def passage(self, passage_id: str):  # pragma: no cover - raises
        raise AssertionError(f"the search path read a passage record for {passage_id!r}")


def test_smr04_the_search_path_cannot_reach_passage_text(built: Built) -> None:
    """SM-R04: with rerank off, text is structurally unreachable: the store raises and search works.

    Rerank is the one text touch in this folder, it is a by-id fetch of the searched rendering,
    and SM-R11 pins it. Everything else, legs, the part fan, the cell merge, RRF, runs here
    against a store that would explode if it were touched.
    """
    cfg = retrieval_cfg(built.root, depth=0)
    ctx = context_for(built, cfg)
    ctx.passage_store = _RaisingStore()

    out = retrieve(ctx, built.records[built.ids("zh")[0]]["original"], top_k=5)
    assert out and all(isinstance(pid, str) and isinstance(s, float) for pid, s in out)
    assert all(len(hit) == 2 for hit in out), "search returned more than (passage_id, score)"
    assert legs(ctx, "muelle0 muelle0 muelle0", top_k=3, source_lang="es")


def test_smr05_the_two_knobs_are_independent(built: Built) -> None:
    """SM-R05: ``passage_lang`` cannot reach search, and the searched index cannot reach display."""
    # (a) retrieve's signature carries no rendering parameter at all.
    params = set(inspect.signature(retrieve).parameters)
    assert not params & {"passage_lang", "rendering", "index", "variant"}

    # (b) display() of one id is identical whichever index was searched.
    pid = f"{ES_PAIR_DOC.format(0)}#p0"
    seen = set()
    for index in RENDERINGS:
        ctx = context_for(built, retrieval_cfg(built.root, index=index))
        assert ctx.index == index
        seen.add(tuple(display(ctx, [pid], "original")))
    assert len(seen) == 1, "display leaked the searched index"

    # (c) two configs differing only in passage_lang return byte-identical search results.
    query = built.records[pid]["original"]
    outs = [
        retrieve(context_for(built, retrieval_cfg(built.root, passage_lang=lang)), query, top_k=_TOP)
        for lang in RENDERINGS
    ]
    assert outs[0] == outs[1] == outs[2]


# --------------------------------------------------------------------------- #
# SM-R06 / SM-R07: self-retrieval is dense-only; a partial leg set is refused
# --------------------------------------------------------------------------- #
def test_smr06_self_retrieval_at_rank_1_is_a_dense_only_guarantee(built: Built, ctx) -> None:
    """SM-R06: dense is a hard rank-0 at score 1.0; sparse/late-interaction assert presence only.

    The asymmetry is not conservatism: only the dense leg's L2-normalized exact inner product
    makes ``<v, v>`` the maximum attainable score. A rank-1 assertion on the other two would be
    asserting something the shipped engines do not guarantee (measured on the real bounded gate:
    dense 45/45, sparse 44/45, late-interaction 43/45).
    """
    pid = built.ids("ru")[0]
    query = built.records[pid]["original"]
    per_leg = legs(ctx, query, top_k=6, source_lang="ru")

    dense = per_leg[DENSE_LEG]
    assert dense[0][0] == pid, "dense self-retrieval must return the passage itself at rank 0"
    assert dense[0][1] == pytest.approx(1.0, abs=1e-9)
    assert all(score <= dense[0][1] + 1e-12 for _, score in dense)

    for leg in (SPARSE_LEG, LATE_INTERACTION_LEG):
        assert pid in {p for p, _ in per_leg[leg]}, f"{leg} did not retrieve the passage at all"

    # The FA01 tie-break exception: a byte-identical duplicate pair both self-retrieve at 1.0,
    # and the order between them is the deterministic passage-id tie-break, not chance.
    left = f"{ES_PAIR_DOC.format(2 * (DUP_PAIRS - 1))}#p0"
    right = f"{ES_PAIR_DOC.format(2 * (DUP_PAIRS - 1) + 1)}#p0"
    dup = legs(ctx, built.records[left]["original"], top_k=4, source_lang="es")[DENSE_LEG]
    assert [p for p, _ in dup[:2]] == sorted([left, right])
    assert dup[0][1] == pytest.approx(1.0, abs=1e-9) == pytest.approx(dup[1][1], abs=1e-9)


def test_smr07_a_cell_missing_a_leg_raises_at_read_time(ctx) -> None:
    """SM-R07: the ``LEGS``-exactly invariant is re-checked when QUERYING, not only when building.

    A cell fused over two of three legs returns a full-looking, silently under-retrieved list;
    nothing downstream, not the id maps, not a score, not the manifest, could tell.
    """
    ctx.cells["ru"].parts[0].legs.pop(SPARSE_LEG)
    with pytest.raises(IndexIntegrityError, match="silently under-retrieved"):
        retrieve(ctx, "housing costs", top_k=5)
    with pytest.raises(IndexIntegrityError):
        legs(ctx, "housing costs", top_k=5, source_lang="ru")


# --------------------------------------------------------------------------- #
# SM-R10: statelessness
# --------------------------------------------------------------------------- #
def test_smr10_retrieval_is_stateless_across_calls(built: Built, ctx) -> None:
    """SM-R10: no candidate/score cache bleeds between calls; the caller owns ``retrieved[]``."""
    a = built.records[built.ids("zh")[0]]["original"]
    b = built.records[built.ids("es")[0]]["original"]

    retrieve(ctx, a, top_k=_TOP)
    after = retrieve(ctx, b, top_k=_TOP)
    fresh = retrieve(context_for(built, built.cfg), b, top_k=_TOP)
    assert after == fresh


# --------------------------------------------------------------------------- #
# SM-R11 / SM-R12: what the cross-encoder is handed
# --------------------------------------------------------------------------- #
def test_smr11_rerank_scores_the_searched_rendering_never_passage_lang(built: Built) -> None:
    """SM-R11: every text handed to the reranker is ``render(pid, retrieval.index)``.

    The context searches ``omt_opus`` while reading ``original``, which is the one
    combination that would expose a rerank pool built from the reading knob.
    """
    cfg = retrieval_cfg(built.root, index="omt_opus", passage_lang="original", depth=10)
    ctx = context_for(built, cfg)
    store = ctx.passage_store

    retrieve(ctx, "quay2 quay2 quay2 quay2 quay2", top_k=5)
    (query, texts), = ctx.reranker.calls
    assert query == "quay2 quay2 quay2 quay2 quay2"
    assert texts, "the reranker was called with an empty pool"

    searched = {store.render(pid, "omt_opus") for pid in built.records}
    assert set(texts) <= searched
    # and at least one of them genuinely differs from the original rendering, so the assertion
    # above is not satisfied by the two renderings happening to coincide
    divergent = [
        pid
        for pid in built.records
        if store.render(pid, "omt_opus") != store.render(pid, "original")
    ]
    assert divergent
    assert any(t not in {store.render(pid, "original") for pid in built.records} for t in texts)


def test_smr12_several_query_strings_become_one_reranker_call(built: Built) -> None:
    """SM-R12: multi-query rerank is one call, with the query strings joined by a single space."""
    ctx = context_for(built, retrieval_cfg(built.root, depth=10))
    retrieve(ctx, ["query one", "query two"], top_k=5)
    assert len(ctx.reranker.calls) == 1
    assert ctx.reranker.calls[0][0] == "query one query two"

    # duplicates are dropped before fusion (a repeated pool would double its own RRF weight)
    ctx2 = context_for(built, retrieval_cfg(built.root, depth=10))
    retrieve(ctx2, ["query one", "query one"], top_k=5)
    assert ctx2.reranker.calls[0][0] == "query one"


# --------------------------------------------------------------------------- #
# SM-R13 / SM-R14: the encoder contract
# --------------------------------------------------------------------------- #
def test_smr13_query_encoders_are_the_index_build_clients_not_the_embedder(
    tmp_path: Path,
) -> None:
    """SM-R13: queries are encoded by ``index_dense``/``milco``/``mtd_colbert``, by identity.

    The sharpest cross-stage contract in this suite. ``ClientBundle.embedder`` is the
    query-time ``retrieval.dense`` encoder. Every shipped config names the same model id there
    as ``index_build.config.dense_model``, which is exactly why the confusion is dangerous:
    only ``index_build`` also pins ``dense_revision``, so ``embedder`` is the unpinned twin and
    reading it here could put the query and the stored vectors in different embedding spaces -
    no crash, no exception, just plausible wrong numbers that no liveness check would catch.
    Asserted by object identity rather than by model id, since the ids agree.
    """
    clients = Clients()
    built = build_index(tmp_path / "not-runs", clients=clients)
    ctx = context_for(built, built.cfg, clients=clients)

    assert ctx.index_ctx.dense is clients.index_dense
    assert ctx.index_ctx.milco is clients.milco
    assert ctx.index_ctx.mtd is clients.mtd_colbert
    assert clients.embedder is not clients.index_dense

    before = list(clients.embedder.query_calls)
    retrieve(ctx, "berth0 berth0 berth0", top_k=5)
    assert clients.embedder.query_calls == before, "a query was encoded by ClientBundle.embedder"
    for client in (clients.index_dense, clients.milco, clients.mtd_colbert):
        assert client.query_calls, f"{client.name} encoded nothing"


def test_smr14_the_query_is_encoded_per_leg_and_cell_never_per_part(tmp_path: Path) -> None:
    """SM-R14: the encode count is independent of the part count; the fan must not multiply it.

    The shipped count is ``len(LEGS) x cells x distinct queries``, not ``len(LEGS) x distinct
    queries``. :func:`~ragtime.preprocess.index.query_lang_leg` takes a query string and encodes it
    once per cell, and retrieval reuses that function verbatim, because re-implementing the
    within-cell merge is the silent ranking bug this suite exists to prevent. One encode per leg
    would need a ``query_lang_leg_with_rep`` sibling beside ``search_with_rep`` in
    ``preprocess/index.py``, that is, an edit to the index build's own module. What is asserted
    instead is the property that guards the hazard, which is stronger than the formula: four times
    the parts, the same number of encodes.
    """
    fine = Clients()
    coarse = Clients()
    a = build_index(tmp_path / "fine", clients=fine, cfg=retrieval_cfg(tmp_path / "fine"))
    b = build_index(
        tmp_path / "coarse",
        clients=coarse,
        cfg=retrieval_cfg(tmp_path / "coarse", part_passages=1_000),
    )
    ctx_a, ctx_b = context_for(a, a.cfg, clients=fine), context_for(b, b.cfg, clients=coarse)
    assert len(ctx_a.cells["es"].parts) == 3 and len(ctx_b.cells["es"].parts) == 1

    for client in (*_encoders(fine), *_encoders(coarse)):
        client.query_calls.clear()
    retrieve(ctx_a, ["berth0 berth0 berth0", "quay1 quay1 quay1 quay1"], top_k=5)
    retrieve(ctx_b, ["berth0 berth0 berth0", "quay1 quay1 quay1 quay1"], top_k=5)

    per_leg_fine = [len(c.query_calls) for c in _encoders(fine)]
    per_leg_coarse = [len(c.query_calls) for c in _encoders(coarse)]
    assert per_leg_fine == per_leg_coarse
    assert per_leg_fine == [len(_cells(ctx_a)) * 2] * len(LEGS) == [8, 8, 8]


def _encoders(clients: Clients):
    return (clients.index_dense, clients.milco, clients.mtd_colbert)


# --------------------------------------------------------------------------- #
# SM-R15 / SM-R16 / SM-R17: the three config knobs
# --------------------------------------------------------------------------- #
def test_smr15_rrf_k_is_config_driven_and_falls_back_to_the_one_canonical_default(
    built: Built,
) -> None:
    """SM-R15: ``k`` comes from ``retrieval.rrf.k``; absent, ``rrf``'s own default applies.

    The absent case is checked against ``rrf(pools)`` called with no ``k`` at all, so a second
    hardcoded 60 anywhere in ``retrieval/`` would have to disagree with this to survive: and
    the source is scanned for the literal as well.
    """
    query = built.records[built.ids("zh")[0]]["original"]
    for k in (1, 60):
        ctx = context_for(built, retrieval_cfg(built.root, rrf_k=k, depth=0))
        assert retrieve(ctx, query, top_k=_TOP) == rrf(_pools(ctx, [query], _TOP), k=k)[:_TOP]

    absent = context_for(built, retrieval_cfg(built.root, rrf_k=None, depth=0))
    assert absent.knobs.rrf_k is None
    assert retrieve(absent, query, top_k=_TOP) == rrf(_pools(absent, [query], _TOP))[:_TOP]

    # k=1 and k=60 must actually produce different orders here, or the test is vacuous.
    one = context_for(built, retrieval_cfg(built.root, rrf_k=1, depth=0))
    sixty = context_for(built, retrieval_cfg(built.root, rrf_k=60, depth=0))
    assert retrieve(one, query, top_k=_TOP) != retrieve(sixty, query, top_k=_TOP)

    for path in sorted(_RETRIEVAL_DIR.glob("*.py")):
        assert "60" not in _code_literals(path), f"{path.name} carries a hardcoded rrf k"


def test_smr16_reranker_depth_bounds_the_pool_and_is_config_driven(built: Built) -> None:
    """SM-R16: exactly ``depth`` candidates reach the cross-encoder, never a hardcoded 100."""
    query = built.records[built.ids("es")[0]]["original"]
    for depth in (3, 7):
        ctx = context_for(built, retrieval_cfg(built.root, depth=depth))
        out = retrieve(ctx, query, top_k=_TOP)
        (_, texts), = ctx.reranker.calls
        assert len(texts) == depth
        # the reranked head really decided the head of the result (score = text length here)
        assert [s for _, s in out[:depth]] == sorted((float(len(t)) for t in texts), reverse=True)
        assert len(out) == len(built.records), "the sub-depth tail was dropped, not kept"

    # depth 0 disables the rerank entirely rather than silently reranking everything
    off = context_for(built, retrieval_cfg(built.root, depth=0))
    retrieve(off, query, top_k=5)
    assert off.reranker.calls == []

    with pytest.raises(context_mod.RetrievalConfigError, match="reranker.depth is missing"):
        cfg = retrieval_cfg(built.root)
        del cfg.blocks["retrieval"]["reranker"]["depth"]
        context_mod.retrieval_knobs(cfg)


def test_smr17_quality_gated_is_read_and_ignored_never_invented(built: Built) -> None:
    """SM-R17: an undefined knob changes nothing, a canary for the day someone wires it up.

    ``retrieval.rrf.quality_gated`` has no definition in any config, doc, code path or paper
    section. Nothing here invents one; if something later does, this test goes red
    rather than the behaviour changing silently under a knob nobody documented.
    """
    query = built.records[built.ids("ru")[0]]["original"]
    outs = []
    for flag in (True, False):
        ctx = context_for(built, retrieval_cfg(built.root, quality_gated=flag))
        assert ctx.knobs.rrf_quality_gated is flag
        outs.append(retrieve(ctx, query, top_k=_TOP))
    assert outs[0] == outs[1]

    for path in sorted(_RETRIEVAL_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)) and "quality_gated" in ast.dump(node.test):
                pytest.fail(f"{path.name} branches on the undefined quality_gated knob")


def _code_literals(path: Path) -> set[str]:
    """Numeric/str literals in real CODE (comments and docstrings excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            out.add(str(node.value))
    return out


# --------------------------------------------------------------------------- #
# SM-R18 / SM-R19 / SM-R20: citation, paths, top_k
# --------------------------------------------------------------------------- #
def test_smr18_display_always_carries_the_original_doc_id(built: Built) -> None:
    """SM-R18: the doc-id is ``common.doc_id_of``'s, for every rendering, for a non-English doc."""
    pid = f"{ES_PAIR_DOC.format(3)}#p0"
    assert built.records[pid]["lang"] == "es"
    for lang in RENDERINGS:
        ctx = context_for(built, retrieval_cfg(built.root, index="omt", passage_lang=lang))
        (doc_id, text), = display(ctx, [pid])
        assert doc_id == doc_id_of(pid) == ES_PAIR_DOC.format(3)
        assert text == ctx.passage_store.render(pid, lang)
        # the explicit-rendering form agrees with the configured default
        assert display(ctx, [pid], lang) == [(doc_id, text)]

    # order and duplicates are preserved, so a caller can zip ids back onto the result
    ctx = context_for(built, built.cfg)
    ids = [pid, built.ids("zh")[0], pid]
    assert [d for d, _ in display(ctx, ids)] == [doc_id_of(i) for i in ids]


def test_smr19_every_path_comes_from_the_injected_layout_never_a_runs_default(
    built: Built, ctx, tmp_path: Path
) -> None:
    """SM-R19: the artifact root is the caller's ``Layout``; nothing falls back to ``runs/``."""
    root = built.root.resolve()
    assert "runs" not in root.parts, "the fixture root must not be the default artifact root"
    assert built.manifest_path.resolve().is_relative_to(root)
    for handle in ctx.cells.values():
        assert handle.lang_dir.resolve().is_relative_to(root)
        for part in handle.parts:
            assert part.shard_dir.resolve().is_relative_to(root)

    # A Layout rooted somewhere else finds no manifest and raises: it does not quietly
    # resolve the index it just read from a default root.
    elsewhere = Layout(
        run_dir=tmp_path / "empty",
        base=tmp_path / "empty",
        family=run_family(built.cfg),
        chunker_hash=all_hashes(built.cfg)["chunker"],
    )
    with pytest.raises(IndexIntegrityError, match="no published index manifest"):
        bring_up(
            built.cfg,
            built.clients,
            elsewhere,
            recon_hash=built.recon_hash,
            pack_hash=None,
            idx_hash=built.idx_hash,
            passage_store=built.store(),
            legs=built.legs,
        )


def test_smr20_top_k_is_caller_supplied_and_never_a_hardcoded_20(built: Built) -> None:
    """SM-R20: ``top_k`` bounds the result and nothing internal caps or pads it.

    The context carries a rerank depth of 10, which is what makes the prefix assertion below
    meaningful rather than accidental: :func:`candidate_depth` is ``max(top_k, depth)``, so for
    any ``top_k <= depth`` every pool is fetched to the same depth and the answer is
    prefix-stable in ``top_k``. (With rerank disabled the pools are only ``top_k`` deep, and a
    rank fusion over differently-truncated pools legitimately reorders: a property of RRF, not
    a defect, and not something a caller may assume.)
    """
    ctx = context_for(built, retrieval_cfg(built.root, depth=10))
    query = built.records[built.ids("en")[0]]["original"]
    three = retrieve(ctx, query, top_k=3)
    seven = retrieve(ctx, query, top_k=7)
    assert len(three) == 3 and len(seven) == 7
    assert seven[:3] == three
    assert three != seven[:len(three) + 1]
    assert len(retrieve(ctx, query, top_k=len(built.records) + 50)) == len(built.records)
    assert retrieve(ctx, "", top_k=5) == [] == retrieve(ctx, ["", "  "], top_k=5)

    assert "top_k" in inspect.signature(retrieve).parameters
    assert inspect.signature(retrieve).parameters["top_k"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------- #
# The spine exception itself: the allowlist is data, and it is enforced
# --------------------------------------------------------------------------- #
def test_the_preprocess_import_exception_is_exactly_twelve_symbols_in_one_file() -> None:
    """The one allowed reach into ``preprocess`` is enumerable, and enforced by inspection."""
    assert _index_api.ALLOWED_INDEX_SYMBOLS == frozenset(
        {
            "IndexCtx",
            "IndexIntegrityError",
            "LangHandle",
            "Leg",
            "LegHandle",
            "default_legs",
            "index_build_options",
            "index_hash",
            "leg_config_hash",
            "open_lang",
            "query_lang_leg",
            "read_shard_parts",
        }
    )
    exported = set(_index_api.__all__) - {"ALLOWED_INDEX_SYMBOLS", "FORBIDDEN_INDEX_MODULES"}
    assert exported == set(_index_api.ALLOWED_INDEX_SYMBOLS)

    api_file = Path(inspect.getfile(_index_api)).name
    for path in sorted(_RETRIEVAL_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        touching = {m for m in modules if m.split(".")[0:2] == ["ragtime", "preprocess"]}
        if path.name == api_file:
            assert touching == {"ragtime.preprocess.index"}
        else:
            assert not touching, f"{path.name} reaches into preprocess outside the allowlist"
        assert not (modules & _index_api.FORBIDDEN_INDEX_MODULES)


def test_the_imported_query_surface_still_has_the_signatures_m06_composes() -> None:
    """The index build -> retrieval contract, pinned from the consumer side.

    SM-R02, SM-R07, SM-R13 and SM-R14 all rest on these signatures.
    """
    assert list(inspect.signature(_index_api.query_lang_leg).parameters) == [
        "handle",
        "leg",
        "query",
        "top_k",
    ]
    assert list(inspect.signature(_index_api.open_lang).parameters) == ["lang_dir", "ctx"]
    assert tuple(leg.name for leg in _index_api.default_legs()) == tuple(LEGS)
    # search returns ids + scores; no leg exposes a text-bearing surface at all
    for leg in _index_api.default_legs():
        assert not [m for m in dir(leg) if "text" in m and not m.startswith("encode")]


def test_display_and_service_never_import_each_other_into_a_text_carrying_search() -> None:
    """``service`` reads text at exactly one call site: the rerank pool (SM-R11's subject)."""
    source = Path(inspect.getfile(service_mod)).read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]
    assert body.count("passage_store.render") == 1
    assert "passage_store.passage" not in body
    # ...and the search module does not import the DISPLAY module at all, so a text fetch
    # cannot arrive through Knob 2's side door either (docstring mentions are fine).
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "display" not in imported and ".display" not in str(imported)


def test_stats_are_emitted_on_canonical_slices_only(built: Built) -> None:
    """Counter-only discipline: this folder emits counters, never a metric, on legal slices."""
    ctx = context_for(built, retrieval_cfg(built.root, depth=5))
    assert isinstance(ctx.stats, Statistics)
    retrieve(ctx, "berth0 berth0 berth0", top_k=5)
    display(ctx, [built.ids("zh")[0]])

    from ragtime.retrieval import stats as stats_mod

    emitted = {rec.metric_id for rec in ctx.stats.records()}
    declared = {getattr(stats_mod, name) for name in stats_mod.__all__}
    assert emitted <= declared and emitted
    # One retrieve call and one display call, counted on their own knob: the query counter is
    # sliced by the SEARCHED rendering and the display counter by the READ one.
    assert ctx.stats.value(stats_mod.STAT_QUERIES, variant=ctx.index) == 1.0
    assert ctx.stats.value(stats_mod.STAT_LEG_CELL_SEARCHES, leg=DENSE_LEG, lang="es",
                           variant=ctx.index) == 1.0
    assert ctx.stats.value(stats_mod.STAT_DISPLAY_LOOKUPS, variant=ctx.passage_lang) == 1.0
    assert ctx.stats.total(stats_mod.STAT_RERANKED) == 5.0


def test_display_module_holds_no_search_knowledge() -> None:
    """Knob 2's module cannot see Knob 1: ``display`` never reads ``ctx.index``."""
    body = Path(inspect.getfile(display_mod)).read_text(encoding="utf-8").split('"""', 2)[-1]
    assert "ctx.index" not in body and "retrieval_index" not in body
