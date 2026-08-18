"""The outer ``(query, leg, cell)`` query fan, over the real fixture index (SM-R21..SM-R28).

The fan is a throughput change and nothing else, so every test here restates one sentence: the
answer at any width is the answer at width 1. They differ only in how they try to break it:

* by comparing whole ranked lists, scores included, at five widths (SM-R21);
* by making completion order provably disagree with submission order and checking that what
  reaches ``rrf`` is still submission order (SM-R22): the concrete hazard, since ``rrf``
  accumulates a float per id across pools and float addition is not associative;
* by counting the threads that actually exist during a fanned call, so "bounded" is measured
  rather than asserted (SM-R25);
* by reading the config leaf the width comes from (SM-R23/24), because a hardcoded width is a
  run choice made in code.

The fixture is ``conftest.py``'s real published index (12 units: 3 legs x 4 language cells),
not a simulation of one: a fan test over stubbed engines would prove nothing about the code
path that actually runs.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from ragtime.config import ConfigError
from ragtime.preprocess.index import LEGS
from ragtime.retrieval import concurrency as conc
from ragtime.retrieval import retrieve
from ragtime.retrieval import service as service_mod
from tests.retrieval.conftest import Built, context_for, retrieval_cfg

pytestmark = pytest.mark.small

_DEPTH = 100
_TOP = 30
#: 1 (no pool at all), 2 and 3 (fewer threads than units, so units queue), 12 (one thread per
#: unit on the shipped shape), 32 (more threads than units, the ``min`` guard).
_WIDTHS = (1, 2, 3, 12, 32)


def _cfg(built: Built, *, leg_workers: int | None = None, part_workers: int | None = None,
         depth: int = 0) -> Any:
    cfg = retrieval_cfg(built.root, depth=depth)
    block = cfg.blocks.setdefault("execution", {})
    if leg_workers is not None:
        block[conc.QUERY_LEG_WORKERS_KEY] = leg_workers
    if part_workers is not None:
        block[conc.QUERY_PART_WORKERS_KEY] = part_workers
    return cfg


# --------------------------------------------------------------------------- #
# SM-R21, the equality proof: fanned output is sequential output
# --------------------------------------------------------------------------- #
def test_smr21_every_fan_width_returns_the_byte_identical_ranked_list(built: Built) -> None:
    """SM-R21: ``retrieve`` at widths 1/2/3/12/32 returns the same list, scores included.

    Equality of ``(passage_id, score)`` pairs, not of id order alone. A fan that perturbed the RRF
    accumulation would show up first in a float, and only later, on some other query, as a swapped
    rank.

    Four queries in four languages, plus one multi-string call, which is the case that makes two
    units of the fan address the same ``(leg, cell)`` and is therefore the one that can race a
    lazily-opened reader. Rerank is on (``depth=100``), so the compared list is the one a caller
    actually receives, through every stage.
    """
    baseline_ctx = context_for(built, _cfg(built, leg_workers=0, depth=100))
    assert baseline_ctx.leg_workers == 1, "width 0 must degrade to sequential, not to a no-op"

    queries: list[str | list[str]] = []
    for lang in ("en", "es", "ru", "zh"):
        queries.append(built.records[built.ids(lang)[0]]["original"])
    queries.append([queries[0], queries[1], queries[2]])  # one call, three query strings

    for query in queries:
        expected = retrieve(baseline_ctx, query, top_k=_TOP)
        assert expected, "a query that retrieves nothing would make this test vacuous"
        for width in _WIDTHS:
            ctx = context_for(built, _cfg(built, leg_workers=width, depth=100))
            assert ctx.leg_workers == width
            got = retrieve(ctx, query, top_k=_TOP)
            assert got == expected, f"width {width} changed the answer for {query!r}"


def test_smr21b_the_pools_themselves_are_identical_element_for_element(built: Built) -> None:
    """SM-R21b: the fan's output, not just what survives fusion, is width-invariant.

    ``retrieve`` could hide a reordering behind ``rrf``'s own sort on a fixture where no tie
    exists. Comparing ``_pools`` directly removes that escape route: the same lists, in the same
    order, which is what ``rrf``'s float accumulation depends on.
    """
    query = built.records[built.ids("es")[0]]["original"]
    sequential = service_mod._pools(context_for(built, _cfg(built, leg_workers=1)), [query], _DEPTH)
    assert len(sequential) == len(LEGS) * 4 == 12
    for width in _WIDTHS:
        fanned = service_mod._pools(
            context_for(built, _cfg(built, leg_workers=width)), [query], _DEPTH
        )
        assert fanned == sequential, f"width {width} reordered or changed the pools"


# --------------------------------------------------------------------------- #
# SM-R22: submission order, proved against an inverted completion order
# --------------------------------------------------------------------------- #
def test_the_leg_fan_hands_rrf_its_pools_in_submission_order_not_completion_order(
    built: Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM-R22: with completion order forced to be the reverse of submission order, ``rrf``
    still receives submission order.

    A stub ``query_lang_leg`` returns a unit-labelled pool and sleeps longer the earlier it was
    submitted, so the 12 units finish back to front, and the recorded completion order is asserted
    to differ from submission order rather than assumed to. If ``_pools`` collected by completion
    instead (an ``as_completed`` loop, a future set, a queue) the pools handed to ``rrf`` would
    come out reversed, and this test fails loudly rather than leaving a rare, data-dependent tie
    flip in a run.
    """
    ctx = context_for(built, _cfg(built, leg_workers=12))
    submitted: list[tuple[str, str]] = []
    completed: list[tuple[str, str]] = []
    lock = threading.Lock()
    expected_units = [(leg, lang) for leg in ctx.leg_names for lang in sorted(ctx.cells)]
    order = {unit: i for i, unit in enumerate(expected_units)}

    def stub(handle, leg, query, top_k):
        unit = (leg, handle.lang_dir.name)
        with lock:
            submitted.append(unit)
        # Earlier units sleep longer -> completion order is the reverse of submission order.
        time.sleep(0.02 * (len(order) - order[unit]))
        with lock:
            completed.append(unit)
        return [(f"{leg}-{unit[1]}#p0", 1.0)]

    fused_pools: list[Any] = []
    real_rrf = service_mod.rrf

    def rrf_spy(pools, **kwargs):
        fused_pools.append(list(pools))
        return real_rrf(pools, **kwargs)

    monkeypatch.setattr(service_mod, "query_lang_leg", stub)
    monkeypatch.setattr(service_mod, "rrf", rrf_spy)
    retrieve(ctx, "a query", top_k=_TOP)

    assert completed != expected_units, (
        "the units did not finish out of order: the test cannot distinguish submission from "
        "completion order and would pass vacuously"
    )
    assert len(fused_pools) == 1
    assert [pool[0][0] for pool in fused_pools[0]] == [
        f"{leg}-{lang}#p0" for leg, lang in expected_units
    ]


def test_the_stats_the_fan_emits_are_per_unit_and_width_invariant(built: Built) -> None:
    """The counter bus is not thread-safe (a dict read-modify-write), so emission is deferred out
    of the workers, and the deferral may neither lose nor duplicate a unit."""
    from ragtime.retrieval import stats as stats_mod

    query = built.records[built.ids("ru")[0]]["original"]
    for width in (1, 12):
        ctx = context_for(built, _cfg(built, leg_workers=width))
        retrieve(ctx, query, top_k=_TOP)
        assert ctx.stats.total(stats_mod.STAT_LEG_CELL_SEARCHES) == 12.0
        for leg in ctx.leg_names:
            for lang in sorted(ctx.cells):
                assert ctx.stats.value(
                    stats_mod.STAT_LEG_CELL_SEARCHES, leg=leg, lang=lang, variant=ctx.index
                ) == 1.0
                assert ctx.stats.value(
                    stats_mod.STAT_LEG_SECONDS, leg=leg, lang=lang, variant=ctx.index
                ) >= 0.0


# --------------------------------------------------------------------------- #
# SM-R23/24: the width comes from config, and the two widths are one decision
# --------------------------------------------------------------------------- #
def test_smr23_the_leg_fan_width_is_a_config_leaf_never_a_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SM-R23: ``execution.query_leg_workers``, declared in the schema, three-case contract."""
    from ragtime.config.schema import _ALLOWED

    assert conc.QUERY_LEG_WORKERS_KEY == "query_leg_workers"
    assert "query_leg_workers" in _ALLOWED["execution"]
    # ...and it is a width, so it must not be in the hashed, fairness-shared build recipe.
    from ragtime.preprocess.index import INDEX_BUILD_CONFIG

    assert "query_leg_workers" not in INDEX_BUILD_CONFIG
    assert "query_leg_workers" not in _ALLOWED["retrieval"]

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    assert conc.default_leg_workers() == conc.LEG_WORKERS_CAP == 12
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert conc.default_leg_workers() == 4  # not halved: see the module docstring
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert conc.default_leg_workers() == 1

    assert conc.resolve_leg_workers(None) == conc.default_leg_workers()
    assert conc.resolve_leg_workers(7) == 7
    assert conc.resolve_leg_workers(0) == 1  # sequential, never a pool that cannot run
    assert conc.resolve_leg_workers(-3) == 1
    with pytest.raises(ConfigError, match="query_leg_workers"):
        conc.resolve_leg_workers("wide")


def test_smr24_the_two_fan_widths_are_bounded_as_one_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SM-R24: the nesting cannot silently multiply, an absent inner width becomes 1.

    This is the whole bound. Without it, the shipped defaults compose to 12 outer x 8 inner =
    96 of our own threads (plus FAISS OpenMP / Seismic Rayon / torch intra-op) on a node that
    is also hosting the shared vLLM.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")

    def cfg(**execution: Any) -> Any:
        import types

        return types.SimpleNamespace(blocks={"execution": dict(execution)})

    # Both absent: the whole budget goes to the outer axis and the inner one is pinned to 1.
    plan = conc.plan_query_concurrency(cfg())
    assert (plan.leg_workers, plan.part_workers) == (12, 1)
    assert plan.bounded

    # Outer explicitly sequential: the inner width goes back to being absent (None), i.e.
    # `preprocess.index` derives it exactly as it did before this fan existed. `None` and `1`
    # must stay distinguishable all the way to that single owner.
    plan = conc.plan_query_concurrency(cfg(query_leg_workers=0))
    assert (plan.leg_workers, plan.part_workers) == (1, None)
    assert plan.bounded

    # An operator who names the inner width is obeyed verbatim, with no silent override, and the
    # resulting unbounded product is visible rather than mysterious.
    plan = conc.plan_query_concurrency(cfg(query_leg_workers=4, query_part_workers=6))
    assert (plan.leg_workers, plan.part_workers) == (4, 6)
    assert not plan.bounded

    plan = conc.plan_query_concurrency(cfg(query_part_workers=8))
    assert (plan.leg_workers, plan.part_workers) == (12, 8)
    assert not plan.bounded


def test_smr24b_bring_up_forwards_the_planned_inner_width_onto_the_index_ctx(
    built: Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM-R24b: the plan reaches both consumers, the context and ``IndexCtx.query_workers``.

    ``query_workers`` stays raw on the way down (``preprocess.index.resolve_query_workers`` is
    still the only thing that says what a value means); what this pins is which raw value.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    ctx = context_for(built, _cfg(built))
    assert ctx.leg_workers == 12
    assert ctx.index_ctx.query_workers == 1

    ctx = context_for(built, _cfg(built, leg_workers=1))
    assert ctx.leg_workers == 1
    assert ctx.index_ctx.query_workers is None

    ctx = context_for(built, _cfg(built, leg_workers=3, part_workers=5))
    assert (ctx.leg_workers, ctx.index_ctx.query_workers) == (3, 5)


# --------------------------------------------------------------------------- #
# SM-R25: the bound, counted
# --------------------------------------------------------------------------- #
def test_smr25_the_fan_creates_at_most_the_configured_number_of_threads(
    built: Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM-R25: the threads that actually run are <= the width, and no inner pool is created.

    Measured from inside the fan (every unit records its own thread name) rather than argued
    from the code, because the failure this guards against, a nested pool multiplying out -
    is invisible in a diff and obvious in a thread census.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    ctx = context_for(built, _cfg(built, leg_workers=3))
    names: list[str] = []
    lock = threading.Lock()
    real = service_mod.query_lang_leg

    def spy(handle, leg, query, top_k):
        with lock:
            names.append(threading.current_thread().name)
        return real(handle, leg, query, top_k)

    monkeypatch.setattr(service_mod, "query_lang_leg", spy)
    retrieve(ctx, built.records[built.ids("zh")[0]]["original"], top_k=_TOP)

    assert len(names) == 12
    workers = set(names)
    assert len(workers) <= 3, f"the fan ran on {len(workers)} threads, width was 3"
    assert all(n.startswith(service_mod._LEG_THREAD_PREFIX) for n in workers), workers
    # The inner fan is off under this plan, so no part-level pool thread may exist at all.
    assert not [
        t for t in threading.enumerate() if t.name.startswith("ragtime-query-fan")
    ]


def test_the_sequential_width_creates_no_pool_at_all(
    built: Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Width 1 is the original code path, not a one-thread emulation of it: same discipline
    as ``preprocess.index._fan_parts``' degenerate case."""
    ctx = context_for(built, _cfg(built, leg_workers=1))
    names: list[str] = []
    real = service_mod.query_lang_leg

    def spy(handle, leg, query, top_k):
        names.append(threading.current_thread().name)
        return real(handle, leg, query, top_k)

    monkeypatch.setattr(service_mod, "query_lang_leg", spy)
    retrieve(ctx, built.records[built.ids("en")[0]]["original"], top_k=_TOP)
    assert set(names) == {threading.current_thread().name}


# --------------------------------------------------------------------------- #
# SM-R26: the diagnostic probe stays sequential
# --------------------------------------------------------------------------- #
def test_smr26_the_legs_diagnostic_probe_does_not_fan(
    built: Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM-R26: ``legs()`` reports per-leg cost, so it must not measure itself under contention."""
    from ragtime.retrieval import legs as legs_probe

    ctx = context_for(built, _cfg(built, leg_workers=12))
    names: list[str] = []
    real = service_mod.query_lang_leg

    def spy(handle, leg, query, top_k):
        names.append(threading.current_thread().name)
        return real(handle, leg, query, top_k)

    monkeypatch.setattr(service_mod, "query_lang_leg", spy)
    legs_probe(ctx, built.records[built.ids("es")[0]]["original"], top_k=5)
    assert set(names) == {threading.current_thread().name}


# --------------------------------------------------------------------------- #
# SM-R29: the fast twin of FL-R12's counting spy
#
# FL-R12 ("the real leg-part fan is actually reached") is the corpus-scale test whose harness once
# killed a corpus-scale run with a RecursionError after the whole rendering had been loaded: its
# spy delegated by calling ``service_mod.query_lang_leg(...)``, the very module attribute
# ``monkeypatch.setattr`` had just rebound to the spy. Nothing about that defect needed the corpus.
# The fan mechanism is identical over this fixture's 3 legs x 4 cells and answers in milliseconds,
# so it is answered here.
# --------------------------------------------------------------------------- #
def test_smr29_a_counting_spy_over_the_leg_part_fan_delegates_and_terminates(
    built: Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM-R29: FL-R12's counting harness, run at fixture scale, same spy, same manifest arithmetic.

    Three things are asserted, and each is one way FL-R12 can fail for harness reasons alone:

    1. It terminates. A spy that re-resolves the patched name self-calls forever. Under the old
       shape this test does not fail an assertion; it raises ``RecursionError``, as FL-R12 did, and
       it does so in well under a second.
    2. It delegates exactly once per unit. The recorded call count equals the number of
       ``(leg, cell)`` units, so a spy that swallowed the call by returning a fake pool, or doubled
       it, is caught here rather than as a wrong corpus-scale census.
    3. The manifest arithmetic FL-R12 does is the arithmetic that matches reality.
       ``sum(parts per cell) * len(LEGS)`` is compared against the leg-parts the fan actually
       walked: the same expression, over a real published manifest, four orders of magnitude
       cheaper. A drift in the manifest key path (``variants``/``shards``/``parts``) is a KeyError
       that FL-R12 would otherwise raise only after the corpus index was open.

    This does not replace FL-R12: only the corpus index can say whether the shipped build's 234
    leg-parts are all reachable. It replaces the 39 minutes it took to learn that the spy was
    written wrong.
    """
    from ragtime.common.io import read_jsonl
    from tests.harness import spy_through

    manifest = read_jsonl(built.manifest_path)[0]
    ctx = context_for(built, _cfg(built, leg_workers=1))
    parts_by_lang = {
        lang: entry["parts"]
        for lang, entry in manifest["variants"][ctx.index]["shards"].items()
    }
    expected = sum(parts_by_lang.values()) * len(LEGS)
    assert expected > len(LEGS), "a one-part-per-cell fixture would make the census vacuous"

    searched: list[tuple[str, str, int]] = []

    def count(handle, leg, query, top_k):  # FL-R12's own `before` body, verbatim in shape
        searched.extend((leg, handle.lang_dir.name, i) for i in range(len(handle.parts)))

    spy = spy_through(monkeypatch, service_mod, "query_lang_leg", before=count)
    out = retrieve(ctx, built.records[built.ids("zh")[0]]["original"], top_k=_TOP)

    assert out, "a fan census over a query that retrieves nothing would be vacuous"
    assert spy.count == len(parts_by_lang) * len(LEGS), (
        f"the spy delegated {spy.count} times for {len(parts_by_lang)} cells x {len(LEGS)} legs"
    )
    assert len(searched) == expected == len(set(searched)), (
        f"the fan reached {len(searched)} leg-parts, the manifest implies {expected}"
    )


def test_smr29b_spy_through_captures_the_original_before_the_patch(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property that makes the self-calling spy unwritable, pinned on the helper itself.

    ``spy_through`` reads the attribute before ``monkeypatch.setattr`` runs and closes over the
    value, so ``Spy.real`` is the pre-patch function and the installed spy is a different object.
    Moving the capture after the patch would make ``real`` the spy itself, and every user of the
    helper would then silently self-call, so it is asserted rather than reviewed.
    """
    from tests.harness import spy_through

    before = service_mod.query_lang_leg
    spy = spy_through(monkeypatch, service_mod, "query_lang_leg")

    assert spy.real is before
    assert service_mod.query_lang_leg is spy.spy is not before
    # ...and the spy is a plain wrapper: no call has happened, and delegation is one deep.
    assert spy.count == 0
    assert spy.spy.__wrapped__ is before
