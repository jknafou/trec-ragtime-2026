"""The client half of the filesystem protocol: which lane was asked, and which lane answered.

Two properties belong to ``ask_service`` itself rather than to the search path built on it, so no
other test in this folder can reach them.

* The server-side Knob-1 refusal is reachable. A request names the rendering the caller means and
  ``ask_service`` preserves it. Overwriting it with the resolved service's own advertised rendering
  made ``rsvc._handle``'s mismatch check a tautology, because the server then compared its
  rendering against a value derived from its own descriptor, and left only the client-side witness
  that a query hit the index the caller meant. Knob 1 (``retrieval.index``) is the multilingual
  information retrieval axis, and a silent substitution there corrupts the comparison with no error
  anywhere, so both witnesses have to be live.
* The reply says which service answered. The resolver is the only code that knows which descriptor
  it picked, and provenance is stated over ``(rendering, index_hash)``, so a run record has to name
  the lane rather than infer it.

The "service" is a thread writing JSON into a tmp directory (``conftest.FakeService``, whose
``_answer`` mirrors ``rsvc.Service._handle``, refusal included).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragtime.devkit.rsvc_registry import (
    DEFAULT_REGISTRY,
    RetrievalServiceError,
    ask_service,
    search,
    service_stamp,
)
from tests.devkit.conftest import INDEX_HASH, FakeService

pytestmark = pytest.mark.small

HITS = (("eng-docs/0007421#p0", 0.9), ("spa-docs/0451820#p1", 0.5))


def _lane(tmp_path: Path, **kw) -> FakeService:
    return FakeService(root=tmp_path / "rsvc", hits=HITS, **kw)


# --------------------------------------------------------------------------- #
# 1. The server's refusal is a real witness, not a tautology.
# --------------------------------------------------------------------------- #
def test_a_request_naming_a_rendering_the_service_does_not_serve_is_refused_by_the_server(
    tmp_path: Path,
) -> None:
    """Ask ``original`` of an ``omt`` lane with no rendering filter: the service must refuse.

    When ``ask_service`` overwrote ``index`` with the resolved service's rendering, this request was
    rewritten to ``omt`` and answered: a plausible ranked list from the wrong index, which is the
    one failure with no downstream symptom.
    """
    omt = _lane(tmp_path, name="omt-0", rendering="omt").start()
    try:
        reply = ask_service(omt.registry, {"query": "who bombed what", "index": "original"})
    finally:
        omt.stop()

    assert reply["ok"] is False
    assert "refusing to answer from a different rendering" in reply["error"]
    assert "runs" not in reply
    # The service saw the caller's rendering, unmodified: that is the whole fix.
    assert omt.seen[0]["index"] == "original"


def test_a_descriptor_that_lies_about_its_rendering_is_caught_by_the_service(
    tmp_path: Path,
) -> None:
    """The registry filter believes the descriptor; the service does not have to.

    A descriptor is a claim a service writes about itself, so the client-side filter cannot detect
    one that is wrong. Here the lane advertises ``original`` and actually searches ``omt``: the
    filter resolves it, the request names ``original``, and the service refuses, which surfaces
    as a :class:`RetrievalServiceError` rather than a ranked list from the wrong index.
    """
    liar = _lane(tmp_path, rendering="original", searches="omt").start()
    try:
        with pytest.raises(RetrievalServiceError, match="failed request"):
            search("who bombed what", rendering="original", registry=liar.registry, top_k=2)
        assert liar.seen[0]["index"] == "original"
    finally:
        liar.stop()


def test_the_rendering_filter_still_names_the_lane_when_the_request_does_not(
    tmp_path: Path,
) -> None:
    """``rendering=`` names Knob 1 for a caller that passes no ``index`` in the request body."""
    lane = _lane(tmp_path, rendering="omt", searches="original").start()
    try:
        reply = ask_service(lane.registry, {"query": "q"}, rendering="omt")
    finally:
        lane.stop()
    assert lane.seen[0]["index"] == "omt"  # the caller's declared lane, not the service's truth
    assert reply["ok"] is False  # ...and the service, which searches `original`, refuses it


def test_an_absent_index_defaults_to_the_resolved_service(tmp_path: Path) -> None:
    """No ``index`` and no ``rendering=``: the descriptor's own rendering is the default.

    Defaulting is the only case where the server's check degenerates to a tautology, and it is
    deliberate: it keeps a caller that resolved its lane purely through the registry working.
    """
    lane = _lane(tmp_path, rendering="omt_opus").start()
    try:
        reply = ask_service(lane.registry, {"query": "q", "top_k": 2})
    finally:
        lane.stop()
    assert lane.seen[0]["index"] == "omt_opus"
    assert reply["ok"] is True


# --------------------------------------------------------------------------- #
# 2. Provenance: which service answered.
# --------------------------------------------------------------------------- #
def test_the_reply_names_the_service_that_answered(tmp_path: Path) -> None:
    """``_service`` carries the lane's identity: name, rendering, ``index_hash``, queue."""
    lane = _lane(tmp_path, name="e2e-original-original-0@a100").start()
    try:
        reply = ask_service(lane.registry, {"query": "q", "top_k": 1}, rendering="original")
    finally:
        lane.stop()

    stamp = reply["_service"]
    assert stamp["name"] == "e2e-original-original-0@a100"
    assert stamp["rendering"] == "original"
    assert stamp["index_hash"] == INDEX_HASH
    assert stamp["queue"] == str(lane.queue)
    assert stamp["descriptor"].endswith("e2e-original-original-0@a100.json")


def test_the_stamp_is_a_projection_and_survives_json(tmp_path: Path) -> None:
    """Only the provenance fields, and nothing that is not JSON-safe (it lands in a run record)."""
    import json

    doc = {
        "name": "n",
        "slot": "s",
        "tier": "a100",
        "rendering": "omt",
        "index_hash": "abc",
        "node": "gpu003",
        "job_id": "4293985",
        "queue": "/q",
        "_path": "/reg/n.json",
        "heartbeat": 1.0,
        "ready": True,
    }
    stamp = service_stamp(doc)
    assert stamp == {
        "name": "n",
        "slot": "s",
        "tier": "a100",
        "rendering": "omt",
        "index_hash": "abc",
        "node": "gpu003",
        "job_id": "4293985",
        "queue": "/q",
        "descriptor": "/reg/n.json",
    }
    assert json.loads(json.dumps(stamp)) == stamp
    assert service_stamp({"name": "bare"}) == {"name": "bare"}  # absent fields are omitted


# --------------------------------------------------------------------------- #
# 3. One constant, one owner.
# --------------------------------------------------------------------------- #
def test_the_registry_path_has_exactly_one_definition() -> None:
    """Every devkit module imports the registry path; none of them re-spells it.

    Two copies of a shared constant drift, and the drift is silent in the worst way: the supervisor
    writes descriptors into one directory while clients scan another, so a fleet that is up reports
    as "no live lane".
    """
    from ragtime.devkit import rsvc_registry

    devkit = Path(rsvc_registry.__file__).resolve().parent
    literals = (f'"{DEFAULT_REGISTRY}"', f"'{DEFAULT_REGISTRY}'")
    offenders = [
        path.name
        for path in sorted(devkit.glob("*.py"))
        if path.name != "rsvc_registry.py"
        and any(lit in path.read_text(encoding="utf-8") for lit in literals)
    ]
    assert offenders == [], f"{offenders} re-spell the registry path instead of importing it"
