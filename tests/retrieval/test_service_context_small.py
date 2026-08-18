"""The service-backed ``RetrievalContext``, entirely on CPU.

Every test here is a CPU client or a pure unit test. Nothing loads a model, opens an index or needs
a GPU: the "service" is ``tests/devkit/conftest.FakeService``, a thread writing JSON into a tmp
directory that mirrors ``rsvc.Service._handle``'s reply shape. Keeping the file on CPU is a
deliberate design choice, so that a defect found here costs one cheap re-run rather than another
pass through a GPU job. It is also the harness-versus-data split: the transport, the refusals and
the context construction are harness properties, answerable in milliseconds, and a
``RecursionError`` first met the slow way cost 39 minutes of index load but reproduces standalone
in 0.1 s.

The fake is reused rather than re-written. The client half under test is the real
``retrieval.endpoints`` code: resolution, the rendering and ``index_hash`` filter, the abortable
wait, the reply checks. Only the ranked list is faked, which is the one part a GPU job would
otherwise have to provide. A test that faked the client would prove nothing about the refusals,
because the refusals are the client.

What this file covers:

1. One dataclass, two constructors. ``bring_up`` and ``service_context`` return the same
   :class:`~ragtime.retrieval.context.RetrievalContext`, and ``retrieval.service.retrieve`` stays
   the single entry point. No stage body branches on which one built it.
2. The four Knob-1 refusals, each witness pinned separately. A wrong-lane answer raises nothing
   downstream, because the three renderings share passage ids, which makes it the most expensive
   failure available here.
3. ``_NoLocalCells`` raises rather than returning empty. ``service._pools`` iterates
   ``sorted(ctx.cells)``, so an empty dict there fuses zero pools and returns a full-looking empty
   ranked list: a confident answer to a question nobody could answer.

What this file does not cover: that the two transports return the same ranked list. Transport
identity, fleet-width invariance and dispatch-order invariance are properties of a live service,
and only a CPU client run against a real one can witness them. A fake returning a canned list
cannot.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ragtime.common import Statistics
from ragtime.retrieval import (
    RetrievalContext,
    RetrievalServiceError,
    ServiceBackend,
    StaleServiceError,
    search_action,
    service_context,
)
from ragtime.retrieval.service import legs, retrieve
from ragtime.retrieval.stats import (
    STAT_CANDIDATES_FUSED,
    STAT_QUERIES,
    STAT_QUERY_SECONDS,
    STAT_QUERY_STRINGS,
)
from tests.devkit.conftest import INDEX_HASH, OTHER_INDEX_HASH, FakeService, dev_cfg

pytestmark = pytest.mark.small

SEARCH = {"rationale": "because", "action": "search", "query": "nordic walking poles"}

#: Tokens that only the transport seam has any reason to spell. Not the bare word
#: "backend", ``serving`` and ``preprocess`` legitimately say "the SaT CPU backend", "the ORT
#: backend", and a rule that fires on those is a rule nobody keeps.
BACKEND_MARKERS = ("ctx.backend", 'ctx, "backend"', "ctx.transport", "ServiceBackend",
                   "RankingBackend")

#: The only modules under ``src/ragtime/`` allowed to spell one.
#:
#: * ``retrieval/endpoints.py`` is the transport; ``retrieval/context.py`` binds one;
#:   ``retrieval/service.py`` holds the one dispatch; ``retrieval/__init__.py`` re-exports them.
#: * ``devkit/`` is exempt for a stated reason rather than by oversight: choosing where ranking
#:   happens is its job (``search_backend``), and it records that choice rather than branching a
#:   stage body on it.
BACKEND_AWARE = {
    "retrieval/__init__.py",
    "retrieval/context.py",
    "retrieval/endpoints.py",
    "retrieval/service.py",
}


@pytest.fixture
def layout(tmp_path: Path):
    """A dev-shaped Layout. Nothing here reads the corpus: the store is always injected."""
    from ragtime.common import Layout

    return Layout(run_dir=tmp_path / "svcrun", base=tmp_path / "base")


@pytest.fixture
def service(tmp_path: Path, passage_records):
    """A live ``original`` lane whose ranked list is the fixture store's own passage ids.

    Wired here rather than imported from ``tests/devkit/conftest.py``: pytest does not apply that
    conftest to this directory, but the fake itself is imported, not re-written: a second fake
    would be a second reading of ``rsvc``'s reply contract, and the drift between them would be
    invisible until a live run.

    Scores descend with rank, so the order the service returned is observable in the result and
    a client that re-sorted would be caught.
    """
    ids = [str(r["passage_id"]) for r in passage_records]
    svc = FakeService(
        root=tmp_path / "rsvc",
        hits=tuple((pid, 1.0 - i / 100.0) for i, pid in enumerate(ids)),
    ).start()
    yield svc
    svc.stop()


def _ctx(service: FakeService, layout, store, **kw) -> RetrievalContext:
    """A service-backed context over a fake lane, with the in-memory store injected.

    ``stats`` is forwarded as-is and never ``or``-defaulted. An empty
    :class:`~ragtime.common.Statistics` is falsy, so ``stats or Statistics()`` silently swaps a
    caller's counter bus for a fresh one and every counter assertion then reads 0.
    """
    cfg = kw.pop("cfg", None) or dev_cfg()
    return service_context(
        cfg,
        layout,
        registry=service.registry,
        recon_hash="recon0",
        pack_hash=None,
        idx_hash=kw.pop("idx_hash", INDEX_HASH),
        passage_store=store,
        **kw,
    )


# --------------------------------------------------------------------------- #
# 1. One dataclass, two constructors, one entry point.
# --------------------------------------------------------------------------- #
def test_the_service_constructor_returns_the_production_dataclass(
    service: FakeService, layout, passage_store
) -> None:
    """Not a subclass, not a look-alike, not a Protocol: the same class ``bring_up`` returns.

    A look-alike would be the "second production path" this design exists not to build: every
    property a caller reads (``index``, ``passage_lang``, ``idx_hash``) has to be the production
    property reading the production knobs, or the two paths can drift on what a knob means.
    """
    ctx = _ctx(service, layout, passage_store)
    try:
        assert type(ctx) is RetrievalContext
        assert ctx.transport == "service"
        assert isinstance(ctx.backend, ServiceBackend)
        # Knob 2 and the config-derived knobs are genuinely present: reading is local.
        assert ctx.index == "original" and ctx.passage_lang == "original"
        assert ctx.idx_hash == INDEX_HASH
        # Knob 1's machinery is absent: it lives in the service.
        assert ctx.index_ctx is None and ctx.leg_names == () and ctx.reranker is None
    finally:
        ctx.close()


def test_an_in_process_context_reports_the_other_transport(ctx) -> None:
    """The same property, from the other constructor, so ``transport`` is a real discriminator.

    ``ctx`` is the suite's real ``bring_up`` over a real (tiny) published index, not a hand-built
    dataclass: a default that only ever reads ``"in_process"`` because nothing ever sets it would
    be a vacuous green.
    """
    assert type(ctx) is RetrievalContext
    assert ctx.transport == "in_process"
    assert ctx.backend is None
    # And the local machinery the service-backed twin lacks is genuinely present here.
    assert ctx.leg_names and ctx.languages


def test_the_tool_seam_is_never_patched(service: FakeService, layout, passage_store) -> None:
    """No global is rebound: the context carries the backend, so nothing has to be monkeypatched.

    An earlier dev-only search path rebound ``retrieval.tool.retrieve`` for the duration of one
    call. It had to be a local patch because production has no injection point and must not grow
    one. Putting the backend on the context needs no injection point at all, so ``tool.retrieve``
    is ``service.retrieve`` before, during and after a remote search.
    """
    from ragtime.retrieval import service as retrieval_service
    from ragtime.retrieval import tool

    assert tool.retrieve is retrieval_service.retrieve
    ctx = _ctx(service, layout, passage_store)
    try:
        search_action(ctx, SEARCH, top_k=3)
        assert tool.retrieve is retrieval_service.retrieve
    finally:
        ctx.close()
    assert tool.retrieve is retrieval_service.retrieve


def test_search_action_is_production_code_over_a_shared_ranker(
    service: FakeService, layout, passage_store, passage_records
) -> None:
    """The full tool call: production ``search_action``, ranked remotely, read locally.

    This is the shape a shared retriever serves: one resident index, many clients. What is asserted
    is that the model's query reached the service unmodified, carrying this run's ``rerank_depth``
    from the config rather than a literal, and naming its lane.
    """
    ctx = _ctx(service, layout, passage_store)
    try:
        result = search_action(ctx, SEARCH, top_k=3)
    finally:
        ctx.close()

    assert [req["query"] for req in service.seen] == ["nordic walking poles"]
    assert service.seen[0]["rerank_depth"] == 100
    assert service.seen[0]["index"] == "original"
    # Ids and scores came back in the service's order and were not re-sorted here.
    expected = [str(r["passage_id"]) for r in passage_records][:3]
    assert [pid for pid, _ in result.hits] == expected
    # Text was fetched locally, by id, in Knob 2's rendering.
    assert all(text for _, _, text in result.passages)


def test_knob_1_and_knob_2_stay_decoupled_across_the_transport(
    service: FakeService, layout, passage_store
) -> None:
    """Search ``original`` remotely while reading ``omt`` locally: the two never entangle."""
    ctx = _ctx(service, layout, passage_store, cfg=dev_cfg(index="original", passage_lang="omt"))
    try:
        result = search_action(ctx, SEARCH, top_k=2)
    finally:
        ctx.close()

    assert ctx.index == "original"  # what was searched, remotely
    assert result.passage_lang == "omt"  # what was read, locally
    for pid, _, text in result.passages:
        assert text == passage_store.render(pid, "omt")


def test_no_stage_body_knows_a_transport_exists() -> None:
    """The transport is decided in one place, and that is checked mechanically.

    The system a test exercises has to be the system that serves a run. The moment a stage body
    writes ``if ctx.backend is None`` there are two systems to keep in agreement instead of one
    system with a parameter, every fix has to be made twice, and the suite stops being evidence
    about what runs.

    Read as text over the whole package, because the property is about what nobody wrote, and the
    failure it prevents raises nothing: a stage that took the local branch on a service-backed
    context would return a full-looking empty list.
    """
    root = Path(__file__).resolve().parents[2] / "src" / "ragtime"
    scanned, offenders = 0, []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in BACKEND_AWARE or rel.split("/")[0] == "devkit":
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in BACKEND_MARKERS):
            offenders.append(rel)
    assert offenders == [], f"{offenders} know about the transport; only {sorted(BACKEND_AWARE)} may"
    # A glob that silently matched nothing is its own vacuous green.
    assert scanned > 40, f"only {scanned} modules were scanned: the rule is inert"


# --------------------------------------------------------------------------- #
# 2. The four Knob-1 refusals. A wrong lane is not an error: it is the wrong experiment.
# --------------------------------------------------------------------------- #
def test_witness_1_a_lane_that_disagrees_with_the_config_is_refused_before_anything_is_written(
    service: FakeService, layout, passage_store
) -> None:
    """``rendering=`` may state Knob 1; it may never override it. The config is the run record."""
    with pytest.raises(RetrievalServiceError, match="disagrees with this config"):
        _ctx(service, layout, passage_store, rendering="omt")
    assert service.seen == []


def test_witness_1b_a_backend_pointed_at_another_lane_refuses_at_query_time(
    service: FakeService, layout, passage_store
) -> None:
    """The same check, at the point of use, for a context assembled by hand rather than built.

    ``service_context`` cannot produce this state, it forces the two equal, so the guard would
    be untested by construction if it were only exercised through the constructor. It is not dead
    code: it is what stops a context being handed a backend by any future assembler (a pair
    registry, a supervisor) that resolved the wrong lane.
    """
    ctx = _ctx(service, layout, passage_store)
    try:
        ctx.backend = ServiceBackend(
            registry=service.registry, rendering="omt", index_hash=INDEX_HASH
        )
        with pytest.raises(RetrievalServiceError, match="retrieval.index"):
            retrieve(ctx, "poles", top_k=3)
        assert service.seen == []
    finally:
        ctx.close()


def test_witness_2_the_registry_filter_refuses_before_a_request_file_exists(
    tmp_path: Path, layout, passage_store
) -> None:
    """Only an ``omt`` lane is live and the run searches ``original``: nothing is even queued."""
    omt = FakeService(
        root=tmp_path / "rsvc", name="omt-0", rendering="omt", hits=(("x#p0", 1.0),)
    ).start()
    try:
        ctx = _ctx(omt, layout, passage_store)
        with pytest.raises(StaleServiceError):
            search_action(ctx, SEARCH, top_k=3)
        ctx.close()
        assert omt.seen == []
        assert list((omt.queue / "in").glob("*.json")) == []
    finally:
        omt.stop()


def test_witness_2b_the_same_rendering_over_a_different_build_is_not_reachable(
    service: FakeService, layout, passage_store
) -> None:
    """Right rendering, wrong ``index_hash``: still the wrong artifact, still refused.

    Filtering on the pair is not hygiene. Two builds of ``original`` produce different rankings
    for the same query, and nothing downstream would report it.
    """
    ctx = _ctx(service, layout, passage_store, idx_hash=OTHER_INDEX_HASH)
    try:
        with pytest.raises(StaleServiceError):
            search_action(ctx, SEARCH, top_k=3)
    finally:
        ctx.close()
    assert service.seen == []


def test_witness_3_a_reply_from_the_wrong_lane_is_refused(
    tmp_path: Path, layout, passage_store, passage_records
) -> None:
    """A service that advertised ``original`` and ANSWERED as ``omt`` is not believed.

    This is the failure with no downstream symptom: the ids are valid, the scores are plausible,
    and the experiment is silently the wrong one.
    """
    liar = FakeService(
        root=tmp_path / "rsvc",
        hits=((str(passage_records[0]["passage_id"]), 1.0),),
        reply_rendering="omt",
    ).start()
    try:
        ctx = _ctx(liar, layout, passage_store)
        with pytest.raises(RetrievalServiceError, match="silently the wrong"):
            search_action(ctx, SEARCH, top_k=3)
        ctx.close()
    finally:
        liar.stop()


def test_witness_3b_the_reply_lane_is_checked_BEFORE_ok(
    tmp_path: Path, layout, passage_store
) -> None:
    """Order matters, so it is asserted rather than left to reading order.

    A reply that is both from the wrong lane and ``ok: false`` must report the lane. Checking
    ``ok`` first would let a wrong-lane reply that happened to succeed slip past on the day
    somebody reorders the two ifs, and the test would still pass on the ``ok: false`` case.
    """
    liar = FakeService(
        root=tmp_path / "rsvc", reply_rendering="omt", ok=False, error="RuntimeError: cell ru died"
    ).start()
    try:
        ctx = _ctx(liar, layout, passage_store)
        with pytest.raises(RetrievalServiceError, match="silently the wrong"):
            search_action(ctx, SEARCH, top_k=3)
        ctx.close()
    finally:
        liar.stop()


def test_witness_4_the_server_refuses_a_lane_it_does_not_serve(
    tmp_path: Path, layout, passage_store
) -> None:
    """The server-side witness, and the one most easily deleted by an innocent-looking edit.

    A descriptor is a CLAIM the service writes about itself, so a client that only checked the
    descriptor would be checking the service's own word. ``ask_service`` therefore preserves the
    caller-supplied ``index`` instead of defaulting it from the descriptor it just resolved, and
    the service compares that against what it actually searches. Overwrite it and the comparison
    becomes a tautology: the server's refusal silently stops being a second witness, with no test
    failing anywhere unless this one exists.

    The fake advertises ``original`` and actually searches ``omt``, which no client-side check can
    detect.
    """
    liar = FakeService(root=tmp_path / "rsvc", rendering="original", searches="omt").start()
    try:
        ctx = _ctx(liar, layout, passage_store)
        with pytest.raises(RetrievalServiceError, match="refusing to answer from a different"):
            search_action(ctx, SEARCH, top_k=3)
        ctx.close()
        # The client's half of that contract: the request named the lane it meant.
        assert liar.seen[0]["index"] == "original"
    finally:
        liar.stop()


def test_the_request_carries_the_callers_lane_not_the_descriptors(
    service: FakeService, layout, passage_store
) -> None:
    """Witness 4's precondition, stated directly so a regression is one line to read."""
    ctx = _ctx(service, layout, passage_store)
    try:
        search_action(ctx, SEARCH, top_k=1)
    finally:
        ctx.close()
    assert service.seen[0]["index"] == ctx.index == "original"


# --------------------------------------------------------------------------- #
# 3. Liveness is checked, never assumed. A worker must not queue into a black hole.
# --------------------------------------------------------------------------- #
def test_no_live_service_at_all_is_refused_immediately(
    tmp_path: Path, layout, passage_store
) -> None:
    empty = FakeService(root=tmp_path / "rsvc", serve=False)
    empty.registry.mkdir(parents=True, exist_ok=True)
    ctx = _ctx(empty, layout, passage_store)
    started = time.perf_counter()
    try:
        with pytest.raises(StaleServiceError):
            search_action(ctx, SEARCH, top_k=3)
    finally:
        ctx.close()
    assert time.perf_counter() - started < 10.0


def test_a_stale_descriptor_is_refused_not_queued_into(
    tmp_path: Path, layout, passage_store
) -> None:
    """``ready: true`` is not liveness. A descriptor that stopped beating is a corpse.

    The lane publishes ``ready: true`` with a dead beat and answers nothing, the exact shape that
    once served a ``FileNotFoundError`` reading like a build failure, because a tmpfs-staged queue
    directory evaporates with its job.
    """
    dead = FakeService(root=tmp_path / "rsvc", serve=False, beat_age_s=10_000).start()
    ctx = _ctx(dead, layout, passage_store)
    started = time.perf_counter()
    try:
        with pytest.raises(StaleServiceError):
            search_action(ctx, SEARCH, top_k=3)
    finally:
        ctx.close()
    assert time.perf_counter() - started < 10.0
    assert list((dead.queue / "in").glob("*.json")) == []


def test_a_service_that_dies_mid_request_aborts_the_wait(
    tmp_path: Path, layout, passage_store
) -> None:
    """A live-then-killed lane fails in seconds, not after the full 900 s per-query timeout.

    This is the difference between a clear error and a mystery, and it is what makes a supervisor
    able to distinguish "the infrastructure is down" from "this topic is poison": an outage that
    presents as a 15-minute hang looks like slow work, not like a fault.
    """
    dying = FakeService(root=tmp_path / "rsvc", serve=False).start()
    dying.kill_descriptor(after_s=0.3)
    ctx = _ctx(dying, layout, passage_store, timeout_s=120.0)
    started = time.perf_counter()
    try:
        with pytest.raises(StaleServiceError):
            search_action(ctx, SEARCH, top_k=3)
    finally:
        ctx.close()
    assert time.perf_counter() - started < 30.0


def test_a_service_error_surfaces_as_a_refusal(tmp_path: Path, layout, passage_store) -> None:
    """``ok: false`` from the RIGHT lane is an error, and it carries the service's own message."""
    broken = FakeService(
        root=tmp_path / "rsvc", ok=False, error="RuntimeError: cell ru died"
    ).start()
    try:
        ctx = _ctx(broken, layout, passage_store)
        with pytest.raises(RetrievalServiceError, match="cell ru died"):
            search_action(ctx, SEARCH, top_k=3)
        ctx.close()
    finally:
        broken.stop()


def test_stale_and_service_errors_stay_distinguishable() -> None:
    """Two sibling classes: a supervisor classifies on exactly this distinction.

    ``StaleServiceError`` means nothing answered: retryable infrastructure, so requeue the report
    request untouched. ``RetrievalServiceError`` means the fleet answered and the answer was
    unusable. The fault-attribution predicate turns the first into "do not increment the attempt
    counter", and a subclass relationship would silently merge the two branches.
    """
    assert not issubclass(StaleServiceError, RetrievalServiceError)
    assert not issubclass(RetrievalServiceError, StaleServiceError)


# --------------------------------------------------------------------------- #
# 4. _NoLocalCells: raising, not returning empty. The sharpest failure available here.
# --------------------------------------------------------------------------- #
def test_local_ranking_on_a_service_backed_context_raises_instead_of_answering_empty(
    service: FakeService, layout, passage_store
) -> None:
    """The load-bearing test in this file.

    ``service._pools`` starts with ``sorted(ctx.cells)``. Hand it a plain empty ``dict`` and it
    fuses zero pools and returns an empty ranked list: no exception, no counter, no log line, and a
    confident answer to a question nobody could answer, which a RAG loop reads as "no evidence
    exists" and abstains on. Raising turns that into the failure it is.

    The backend is cleared to reach the local path on a context that has no cells, which is the
    state a future bug would produce: a supervisor that forgot to bind a lane, or a
    ``dataclasses.replace`` that dropped a field.
    """
    ctx = _ctx(service, layout, passage_store)
    try:
        ctx.backend = None
        with pytest.raises(RetrievalServiceError, match="service-backed"):
            retrieve(ctx, "anything", top_k=3)
    finally:
        ctx.close()


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(lambda cells: iter(cells), id="__iter__"),
        pytest.param(lambda cells: len(cells), id="__len__"),
        pytest.param(lambda cells: "es" in cells, id="__contains__"),
        pytest.param(lambda cells: cells["es"], id="__getitem__"),
        pytest.param(lambda cells: cells.get("es"), id="get"),
        pytest.param(lambda cells: cells.keys(), id="keys"),
        pytest.param(lambda cells: cells.values(), id="values"),
        pytest.param(lambda cells: cells.items(), id="items"),
    ],
)
def test_every_read_accessor_on_cells_refuses(
    probe, service: FakeService, layout, passage_store
) -> None:
    """Not only ``__iter__``. A partial refusal leaves one path silently answering from nothing.

    ``legs(ctx, q, source_lang="es")`` probes with ``in`` and ``[]``, neither of which touches
    ``__iter__``, so a guard that covered iteration alone would let the per-leg diagnostic report
    "this cell returned nothing" about a cell that lives in another job.
    """
    ctx = _ctx(service, layout, passage_store)
    try:
        with pytest.raises(RetrievalServiceError, match="service-backed"):
            probe(ctx.cells)
    finally:
        ctx.close()


@pytest.mark.parametrize("source_lang", [None, "es"])
def test_the_per_leg_diagnostic_has_no_remote_meaning_and_says_so(
    source_lang, service: FakeService, layout, passage_store
) -> None:
    """``legs`` reports each leg's raw pool from local handles. There are none; it must not lie."""
    ctx = _ctx(service, layout, passage_store)
    try:
        with pytest.raises(RetrievalServiceError, match="service-backed"):
            legs(ctx, "poles", top_k=3, source_lang=source_lang)
    finally:
        ctx.close()


def test_languages_has_no_remote_answer(service: FakeService, layout, passage_store) -> None:
    """"Which language cells does this process hold" is not a question about the corpus."""
    ctx = _ctx(service, layout, passage_store)
    try:
        with pytest.raises(RetrievalServiceError, match="service-backed"):
            _ = ctx.languages
    finally:
        ctx.close()


# --------------------------------------------------------------------------- #
# 5. One ranking policy, one counter bus, one provenance record.
# --------------------------------------------------------------------------- #
def test_several_query_strings_are_refused_rather_than_fused_differently(
    service: FakeService, layout, passage_store
) -> None:
    """Production rrfs (query x leg x cell) pools as one fusion; per-query fusion is a different
    ranking policy, so the service is never asked to approximate it."""
    action = {"action": "search", "query": ["poles", "finland"], "rationale": "two"}
    ctx = _ctx(service, layout, passage_store)
    try:
        with pytest.raises(RetrievalServiceError, match="one query per request"):
            search_action(ctx, action, top_k=3)
    finally:
        ctx.close()


def test_the_counter_bus_is_transport_blind(
    service: FakeService, layout, passage_store
) -> None:
    """The same four counters, emitted by the same code, whichever transport answered.

    They are emitted in ``service.retrieve`` around the dispatch, not inside either branch, so a
    monitoring rollup cannot tell the two apart, and a change that adds a transport cannot
    forget to instrument it.
    """
    stats = Statistics()
    ctx = _ctx(service, layout, passage_store, stats=stats)
    try:
        search_action(ctx, SEARCH, top_k=2)
    finally:
        ctx.close()

    assert stats.total(STAT_QUERIES) == 1.0
    assert stats.total(STAT_QUERY_STRINGS) == 1.0
    assert stats.total(STAT_QUERY_SECONDS) > 0.0
    # The fused-pool size comes from the reply, not from a local count that does not exist.
    assert stats.total(STAT_CANDIDATES_FUSED) == float(len(service.hits))


def test_every_search_records_which_service_answered(
    service: FakeService, layout, passage_store
) -> None:
    """The provenance the config hash cannot carry, so the artifact must.

    ``config.fairness.family_guard`` does not compare the ``execution`` block, so nothing in the
    hash proves two members of a run family searched the same lane over the same transport. The
    per-search record does: rendering, ``index_hash``, and the answering service's identity.
    ``queued_s`` rides along because it is the one number separating "the query is slow" from "the
    query queued", which is what an admission ceiling is sized from.
    """
    service.queued_s = 0.25
    ctx = _ctx(service, layout, passage_store)
    try:
        search_action(ctx, SEARCH, top_k=2)
    finally:
        ctx.close()

    (call,) = ctx.backend.calls
    assert call["rendering"] == "original" and call["index_hash"] == INDEX_HASH
    assert call["service"]["rendering"] == "original"
    assert call["service"]["index_hash"] == INDEX_HASH
    assert call["queued_s"] == 0.25
    assert call["client_wall_s"] >= 0.0 and call["hits"] == 2
    assert ctx.transport == "service"


def test_the_reply_contract_this_client_reads_is_the_one_the_service_writes() -> None:
    """A cheap drift alarm on a contract that has no shared type and costs ~900 s to test live.

    The client (``retrieval.endpoints``) and the service (``devkit.rsvc``) are separate files in
    separate packages, so nothing but a real run would notice a renamed key, and a real run costs
    two GPUs and an 854-906 s cold load. Read as text, so a service file caught mid-edit is not
    misread as a contract break. If this fails, follow ``rsvc``'s shape; do not loosen the client.
    """
    source = (
        Path(__file__).resolve().parents[2] / "src" / "ragtime" / "devkit" / "rsvc.py"
    ).read_text(encoding="utf-8")
    for key in ('"ok"', '"rendering"', '"index_hash"', '"runs"', '"hits"', '"fused_pool"',
                '"summary"', '"queued_s"', '"wall_s"'):
        assert key in source, f"the service no longer writes {key}; this client reads it"
    for key in ('"top_k"', '"rerank_depth"', '"query"'):
        assert key in source, f"the service no longer reads {key}; this client sends it"


def test_the_client_half_has_exactly_one_implementation() -> None:
    """``devkit.rsvc_registry`` re-exports the client; it does not keep a second copy.

    Two implementations of a resolution, liveness and wait protocol drift silently and in the worst
    direction: the supervisor writes descriptors one way while a client reads them another, and a
    fleet that is up reports as "no live lane". The client therefore has one owner, and both names
    resolve to the same objects.
    """
    from ragtime.devkit import rsvc_registry
    from ragtime.retrieval import endpoints

    for name in ("ask", "ask_service", "is_stale", "live_services", "read_descriptors",
                 "service_stamp", "StaleServiceError", "RetrievalServiceError"):
        assert getattr(rsvc_registry, name) is getattr(endpoints, name), name
