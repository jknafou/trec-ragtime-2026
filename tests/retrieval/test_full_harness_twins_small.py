"""Fast twins for the harness of ``tests/retrieval/test_retrieval_full.py``: no index, no GPU.

FL-R08..FL-R14 can only answer their data questions over the corpus-scale index, and a cold open
of one rendering takes tens of minutes. Those tests are not made only of data assertions, though.
They also carry harness: skip predicates, spies, manifest key paths and attribute walks over the
retrieval context, each of which is right or wrong independently of how much data is behind it.

This module exercises that same harness against the fixture-scale index built by the shared
``built`` fixture. The helpers are imported from the full module itself rather than
re-implemented, so a twin cannot drift away from the code it protects.

The twins replace nothing: every corpus-scale assertion stands exactly as written. What they
remove is the long index open that would otherwise stand between a KeyError, an AttributeError or
a self-calling spy and the report of it.
"""

from __future__ import annotations

import inspect
import types
from dataclasses import replace
from typing import Any

import pytest

from ragtime.common import Statistics
from ragtime.common.io import is_done, read_jsonl
from ragtime.common.passage_store import RENDERINGS
from ragtime.preprocess.acceptance import duplicate_tie_explains_miss
from ragtime.preprocess.index import DENSE_LEG, LEGS, read_shard_parts
from ragtime.retrieval import bring_up

# The subject under test is the full gate's own machinery. Importing it is ~1.7 s and touches no
# artifact: the corpus reads all live inside the test bodies and the `_corpus_context` helper.
from tests.retrieval import test_retrieval_full as full
from tests.retrieval.conftest import Built, context_for, retrieval_cfg

pytestmark = pytest.mark.small


# --------------------------------------------------------------------------- #
# 1. `_require_reranker`: the skip that must not absorb a missing entry point.
# --------------------------------------------------------------------------- #
def test_require_reranker_lets_a_vanished_entry_point_blow_up_instead_of_skipping() -> None:
    """A reranker without the probed entry point must raise, never skip.

    Probing a private method from inside the ``try`` is how a vacuous green happens: a broad
    ``except Exception`` swallows the ``AttributeError`` that a moved entry point raises --
    ``reranker._cross_encoder()`` belongs to the CrossEncoder /
    ``AutoModelForSequenceClassification`` path rather than to the ``AutoModelForCausalLM`` one
    that ships -- and the gate emits a labelled skip blaming ``retrieval.reranker.model``, a
    config leaf that is correct in all six run files, while FL-R03, the integration test for the
    fusion fix, does not run at all and the gate reports success.

    The shipped shape resolves the attribute outside the ``try``, so a missing entry point is the
    suite failure it is.

    The explicit try/except is load-bearing and is not interchangeable with
    ``pytest.raises(AttributeError)``. ``pytest.skip`` raises ``Skipped``, which derives from
    ``BaseException`` and so passes straight through ``pytest.raises``: against the swallowing
    shape a ``raises``-based twin reports SKIPPED, reproducing the vacuous green this test exists
    to catch. That was measured rather than argued: in a scratch copy carrying the swallowing
    shape, a ``raises`` twin came back "1 skipped" and this form came back failed.
    """
    clients = types.SimpleNamespace(reranker=types.SimpleNamespace(model="Qwen/Qwen3-Reranker-4B"))
    try:
        full._require_reranker(clients)
    except AttributeError:
        return  # the only correct outcome: a vanished entry point is a suite failure
    except pytest.skip.Exception as skipped:
        pytest.fail(
            "_require_reranker converted a MISSING entry point into a skip, which is "
            f"a test that did not run while the gate reads RC=0: {skipped}"
        )
    pytest.fail("_require_reranker accepted a reranker that exposes no entry point at all")


def test_require_reranker_still_skips_when_the_real_load_genuinely_fails() -> None:
    """The skip the guard exists for: a resolvable entry point that cannot load the checkpoint.

    Asserted with its label intact, because a skip whose message stops naming the blocking config
    leaf is how a block gets mistaken for a pass.
    """

    class _Unloadable:
        model = "Qwen3-Reranker-4B"

        def load(self) -> None:
            raise OSError("no such checkpoint in the local cache")

    with pytest.raises(pytest.skip.Exception) as caught:
        full._require_reranker(types.SimpleNamespace(reranker=_Unloadable()))
    message = str(caught.value)
    assert "BLOCKED: reranker checkpoint" in message
    assert "retrieval.reranker.model" in message
    assert "not a pass" in message


def test_require_reranker_returns_the_reranker_when_the_load_succeeds() -> None:
    """The happy path, so the guard cannot become unconditionally-skip and stay unnoticed."""
    loaded: list[int] = []

    class _Ok:
        model = "Qwen/Qwen3-Reranker-4B"

        def load(self) -> None:
            loaded.append(1)

    reranker = _Ok()
    assert full._require_reranker(types.SimpleNamespace(reranker=reranker)) is reranker
    assert loaded == [1], "the guard must actually drive the load it claims to probe"


def test_the_entry_point_the_full_gate_probes_is_public_on_the_production_reranker() -> None:
    """``Reranker.load`` exists and is public: the contract the guard is allowed to lean on.

    A private helper can vanish in any refactor (``_cross_encoder`` did); a public entry point is
    part of what the class owes its callers. Pinning it here means the probe and the production
    class are checked against each other without a GPU.
    """
    from ragtime.serving.reranker import Reranker

    assert callable(Reranker.load)
    assert not hasattr(Reranker, "_cross_encoder"), (
        "the deleted private probe target is back: if a guard starts calling it again, the "
        "vacuous-skip failure returns with it"
    )


# --------------------------------------------------------------------------- #
# 2. `_corpus_context`: the manifest key paths FL-R08/12/13 evaluate after the load.
# --------------------------------------------------------------------------- #
def test_the_manifest_key_paths_the_corpus_gate_reads_exist_in_a_real_manifest(
    built: Built,
) -> None:
    """Every ``manifest[...]`` expression in FL-R08/FL-R12/FL-R13, over a really-published manifest.

    These reads happen *after* ``bring_up`` has opened the rendering, so a drifted key is a
    ``KeyError`` 30-40 minutes into a corpus job. The fixture index publishes the same manifest
    through the same ``IndexAdapter.merge``, so the key paths are checkable in milliseconds:

    * ``manifest["variants"][index]["shards"][lang]["parts"]``: FL-R08's leg-part unit count and
      FL-R12's expected fan width;
    * ``entry["passages"] / ["parts"] / ["part_dirs"]``: FL-R13's per-cell census tuple;
    * ``["shard_parts"][part]["id_digest"]``: FL-R13's cross-rendering digest agreement.

    The counts are the fixture's and are asserted only for internal agreement; the corpus-scale
    numbers remain FL-R13's alone.
    """
    manifest = read_jsonl(built.manifest_path)[0]
    assert is_done(built.manifest_path), (
        "`_corpus_context` skips unless the manifest is `_SUCCESS`-marked: if a published "
        "manifest did not satisfy that predicate, every corpus test would skip forever"
    )

    assert sorted(manifest["variants"]) == sorted(RENDERINGS)
    for variant in RENDERINGS:
        shards = manifest["variants"][variant]["shards"]
        assert sorted(shards) == ["en", "es", "ru", "zh"], (variant, sorted(shards))
        for lang, entry in shards.items():
            assert isinstance(entry["parts"], int) and entry["parts"] >= 1
            assert isinstance(entry["passages"], int) and entry["passages"] >= 1
            assert len(entry["part_dirs"]) == entry["parts"], (variant, lang)
            assert len(entry["shard_parts"]) == entry["parts"], (variant, lang)
            for part in range(entry["parts"]):
                assert entry["shard_parts"][part]["id_digest"]

    # FL-R13's own comparison, on the fixture: the three renderings' cells must agree.
    for lang in manifest["variants"]["original"]["shards"]:
        values = {
            (
                manifest["variants"][v]["shards"][lang]["passages"],
                manifest["variants"][v]["shards"][lang]["parts"],
                tuple(manifest["variants"][v]["shards"][lang]["part_dirs"]),
            )
            for v in RENDERINGS
        }
        assert len(values) == 1, (lang, values)


def test_the_corpus_context_part_census_agrees_with_the_manifest_on_a_real_cell(
    built: Built,
) -> None:
    """``read_shard_parts(cell)["parts"] == entry["parts"]``: ``_corpus_context``'s own gate.

    This is the "did we search a COMPLETE index" precondition every corpus test inherits, and it
    is the only part of ``_corpus_context`` that runs before the expensive open. Its two moving
    pieces, the ``"parts"`` key of the census dict, and the layout call that names the cell, are
    checked here against a real published cell, so a rename cannot make the precondition raise
    only on the cluster.
    """
    manifest = read_jsonl(built.manifest_path)[0]
    for variant in RENDERINGS:
        for lang, entry in manifest["variants"][variant]["shards"].items():
            cell = built.lang_dir(variant, lang)
            census = read_shard_parts(cell)
            assert census["parts"] == entry["parts"], (variant, lang, census, entry)
            assert len(census["dirs"]) == entry["parts"]


def test_the_shared_corpus_context_view_isolates_per_test_state(built: Built) -> None:
    """The caching must not let one corpus test change what the next one measures.

    ``_corpus_context`` builds the 255-410 GiB context once per pytest process; without the
    cache a tier logs ``retrieval.context.up`` twice and pays the whole load again. Each caller
    is handed a shallow ``dataclasses.replace`` view carrying a fresh ``Statistics``. That
    mechanism has two properties worth checking, both size-independent, so both are checked on
    a fixture context:

    1. the view shares the expensive half (the same cells, the same ``IndexCtx``, the same
       passage store), or the cache saves nothing;
    2. the view's own ``stats`` and ``leg_workers`` are its own, so a query in one test does not
       show up in another test's counters.
    """
    ctx = context_for(built, retrieval_cfg(built.root, depth=0))
    view = replace(ctx, stats=Statistics(), _owns_store=False)

    assert view.cells is ctx.cells and view.index_ctx is ctx.index_ctx
    assert view.passage_store is ctx.passage_store
    assert view.stats is not ctx.stats
    assert view._owns_store is False

    view.leg_workers = 7
    assert ctx.leg_workers != 7, "a per-test width write leaked into the shared context"


def test_the_shared_context_fingerprint_catches_a_reach_through_mutation(built: Built) -> None:
    """``_ctx_fingerprint`` is the tripwire, so it must actually trip.

    A shallow copy isolates the context's own fields but not the objects it points at:
    ``view.index_ctx.query_workers = N`` reaches through to the shared ``IndexCtx``, which is
    precisely how a shared context would silently change what the next test measures. Both
    directions are asserted: an unmutated context matches and a reached-through one does not, so
    the tripwire cannot pass by always agreeing.
    """
    ctx = context_for(built, retrieval_cfg(built.root, depth=0))
    fingerprint = full._ctx_fingerprint(ctx)
    assert full._ctx_fingerprint(replace(ctx, stats=Statistics())) == fingerprint

    view = replace(ctx, stats=Statistics(), _owns_store=False)
    view.index_ctx.query_workers = (view.index_ctx.query_workers or 0) + 5
    assert full._ctx_fingerprint(ctx) != fingerprint, (
        "a reach-through write to the shared IndexCtx was invisible to the fingerprint: the "
        "cached corpus context could then change what the next test measures, silently"
    )


def test_corpus_context_serves_the_cache_without_rebuilding_and_refuses_a_mutated_one(
    built: Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_corpus_context``'s cache branch itself, driven with a fixture context in the cache.

    The expensive branch needs the corpus; the cache branch does not, and it is the branch that
    decides whether a five-test gate pays the ~30-40 minute open once or five times. Seeding
    ``_CORPUS_CONTEXTS`` lets the real code path run here: two calls, the same shared cells, two
    independent ``Statistics``, and no rebuild (``load_config`` is fenced off: if the helper fell
    through to the build path it would raise instead of quietly re-reading the corpus).
    """
    ctx = context_for(built, retrieval_cfg(built.root, depth=0))
    manifest = read_jsonl(built.manifest_path)[0]
    monkeypatch.setattr(
        full, "_CORPUS_CONTEXTS", {"original": (built.cfg, manifest, ctx, full._ctx_fingerprint(ctx))}
    )
    monkeypatch.setattr(full, "load_config", _must_not_rebuild)

    cfg_a, manifest_a, ctx_a = full._corpus_context()
    cfg_b, manifest_b, ctx_b = full._corpus_context()

    assert cfg_a is cfg_b is built.cfg and manifest_a is manifest_b is manifest
    assert ctx_a is not ctx_b, "each caller must get its own view, not one shared object"
    assert ctx_a.cells is ctx_b.cells is ctx.cells, "the EXPENSIVE half must be shared"
    assert ctx_a.stats is not ctx_b.stats, "per-test counters must not accumulate across tests"
    assert ctx_a.index == ctx.index

    # ...and the tripwire fires on the next handout once something reached through.
    ctx.index_ctx.query_workers = (ctx.index_ctx.query_workers or 0) + 3
    with pytest.raises(AssertionError, match="was MUTATED by an earlier test"):
        full._corpus_context()


def _must_not_rebuild(*_args: Any, **_kwargs: Any):
    raise AssertionError(
        "_corpus_context fell through to the BUILD path with a warm cache: the corpus-scale "
        "open would be paid again"
    )


def test_bring_up_accepts_the_exact_keyword_set_the_corpus_context_passes() -> None:
    """``_corpus_context`` calls ``bring_up`` with a fixed kwarg set: bind it without calling it.

    A renamed or reordered parameter is a ``TypeError`` that a corpus job would raise only after
    ``build_clients`` has stood up the three real encoders. ``Signature.bind`` answers it for free.
    """
    inspect.signature(bring_up).bind(
        object(),  # cfg
        object(),  # clients
        object(),  # layout
        recon_hash="r",
        pack_hash="p",
        idx_hash="i",
        legs=(),
        stats=object(),
    )


# --------------------------------------------------------------------------- #
# 3. The context attribute walks FL-R08/09/11/12 perform after the load.
# --------------------------------------------------------------------------- #
def test_the_context_attribute_paths_the_corpus_gate_walks_are_real(built: Built) -> None:
    """Each attribute chain the blocked tests dereference, over a real (fixture-scale) context.

    None of these is a data claim; each is an API-shape claim that currently costs a full load to
    falsify:

    * ``ctx.index`` must be a key of ``manifest["variants"]`` (FL-R08, FL-R12 index by it);
    * ``ctx.cells[lang].parts[i].passage_ids(leg)`` (FL-R09's corpus census, FL-R11's probe set);
    * ``handle.lang_dir.name`` (FL-R12's spy labels every unit with it);
    * ``ctx.passage_store.passage(pid).lang`` (FL-R09's per-hit language mix);
    * ``ctx.passage_store.render(pid, ctx.index)`` (FL-R11's self-retrieval query text);
    * ``ctx.recon_hash`` / ``ctx.idx_hash`` (FL-R08 names the index its timing covered: a report
      line that raises is a lost measurement at the worst possible moment).
    """
    manifest = read_jsonl(built.manifest_path)[0]
    ctx = context_for(built, retrieval_cfg(built.root, depth=0))

    assert ctx.index in manifest["variants"]
    assert sorted(ctx.cells) == ["en", "es", "ru", "zh"]
    assert isinstance(ctx.recon_hash, str) and isinstance(ctx.idx_hash, str)

    for lang, handle in ctx.cells.items():
        assert handle.lang_dir.name == lang
        assert handle.parts, lang
        for part in handle.parts:
            ids = part.passage_ids(LEGS[0])
            assert ids and all(isinstance(pid, str) for pid in ids)
            assert list(part.passage_ids(DENSE_LEG))

    pid = built.ids("es")[0]
    assert ctx.passage_store.passage(pid).lang == "es"
    assert ctx.passage_store.render(pid, ctx.index)


def test_the_flr11_row_shape_is_the_one_the_shared_tie_predicate_reads(built: Built) -> None:
    """FL-R11 builds a dict and hands it to ``acceptance.duplicate_tie_explains_miss``.

    The predicate is imported rather than re-implemented so that the corpus gate and the
    production validator cannot drift on the tolerance, but that only holds if the row shape
    FL-R11 constructs is the shape the predicate reads. A renamed field would make the predicate
    silently answer about a missing key, at corpus scale, inside the one hard gate of the tier.
    Both verdicts are exercised, so the twin cannot pass by always returning the same answer.
    """
    tied = {
        "passage_id": "a#p0",
        "lang": "es",
        "rank": 1,
        "top_score": 0.9999999403953552,
        "own_score": 0.9999999403953552,
        "competitors": [
            {"passage_id": "b#p0", "score": 0.9999999403953552},
            {"passage_id": "a#p0", "score": 0.9999999403953552},
        ],
    }
    assert duplicate_tie_explains_miss(tied) is True

    below_the_maximum = {**tied, "own_score": 0.635, "competitors": [
        {"passage_id": "b#p0", "score": 0.9999999403953552},
        {"passage_id": "a#p0", "score": 0.635},
    ]}
    assert duplicate_tie_explains_miss(below_the_maximum) is False


# --------------------------------------------------------------------------- #
# 4. The remaining shared helpers of the full module.
# --------------------------------------------------------------------------- #
def test_query_cfg_varies_a_query_knob_without_moving_the_index_recipe(built: Built) -> None:
    """``_query_cfg`` carries its own ``index_hash`` assert, so run it on a real config, cheaply.

    One hazard is reachable only with a real config: a ``RunConfig``'s blocks are nested
    ``MappingProxyType``, so a ``dict``-only copy leaves them frozen and the ``depth`` write raises
    on the cluster while passing on a hand-built namespace. The fixture config is a namespace, so
    this twin pins the other half, that every knob path executes and the recipe hash is preserved.
    The ``Mapping`` hazard stays uncovered here.
    """
    varied = full._query_cfg(built.cfg, index="omt", passage_lang="original", depth=5, rrf_k=17)
    assert varied.retrieval_index == "omt"
    assert varied.blocks["retrieval"]["reranker"]["depth"] == 5
    assert varied.blocks["retrieval"]["rrf"]["k"] == 17
    # the source config is untouched: several contexts in one test read it after this returns
    assert built.cfg.retrieval_index == "original"

    dropped = full._query_cfg(built.cfg, rrf_k=None)
    assert "k" not in dropped.blocks["retrieval"]["rrf"]


def test_set_execution_replans_both_fan_widths_together(built: Built) -> None:
    """FL-R14 changes the fan width through ``plan_query_concurrency``, never by poking one field.

    The failure it guards against is a half-applied plan: the outer width moves and the inner one
    keeps the previous plan's value, so "the fan changed the answer" would be measured against a
    context that is not the one the width names.
    """
    ctx = context_for(built, retrieval_cfg(built.root, depth=0))
    full._set_execution(ctx, leg_workers=1)
    assert (ctx.leg_workers, ctx.index_ctx.query_workers) == (1, None)
    full._set_execution(ctx, leg_workers=12)
    assert ctx.leg_workers == 12
    assert ctx.index_ctx.query_workers == 1, "the inner fan must be bounded off, as FL-R14 asserts"


def test_clients_for_names_the_index_ctx_fields_that_actually_exist(built: Built) -> None:
    """``_clients_for`` reads ``ctx.dense`` / ``.milco`` / ``.mtd`` off the built index context.

    It stands up no second encoder, which is the single-serving rule applied to the harness, and
    that only works if those three field names are the real ``IndexCtx``'s. Checked against a real
    one. ``embedder`` stays the deliberate sentinel that makes a wrong-space read loud.
    """
    ctx = context_for(built, retrieval_cfg(built.root, depth=0))
    bundle = full._clients_for(types.SimpleNamespace(ctx=ctx.index_ctx))
    assert bundle.index_dense is ctx.index_ctx.dense
    assert bundle.milco is ctx.index_ctx.milco
    assert bundle.mtd_colbert is ctx.index_ctx.mtd
    assert bundle.reranker is None
    assert bundle.embedder is not bundle.index_dense


def test_the_timing_report_carries_what_it_measured() -> None:
    """``<name>: <value> for <unit>``, the comparable form.

    A number without what it covered cannot be compared against another run, so a ``_perf``
    that dropped a field would produce unusable measurements at the end of a long job. The
    report is also recorded under its own name, so the end-of-tier dump carries every number.
    """
    try:
        line = full._perf("probe_timing", "1.5 s", unit="1 query on index=original")
        assert line == "probe_timing: 1.5 s for 1 query on index=original"
        for field in ("probe_timing", "1.5 s", "1 query on index=original"):
            assert field in line
        assert full._MEASURED["probe_timing"] == line
    finally:
        # The full module's measured-facts dump is shared process state; this probe is not one
        # of its measurements, so it does not get to appear in it.
        full._MEASURED.pop("probe_timing", None)


def test_topic_query_reads_the_problem_statement_not_the_background(small_topics: Any) -> None:
    """FL-R08/FL-R09 issue ``topic.problem_statement``: the field the loop's own search uses."""
    topic = small_topics[0]
    assert full._topic_query(topic) == topic.problem_statement
    assert full._topic_query(topic).strip()
    assert full._topic_query(topic) != getattr(topic, "background", None)
