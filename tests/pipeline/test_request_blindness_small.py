"""Request blindness: the RAG loop behaves as a function of its nugget question alone.

The organisers' topic ``title`` field is part of the report request and enters decompose only.
Nothing enforced that "only": ``title`` is one keyword argument away from every stage it must
not reach, and the stage it must not reach is the one least likely to be re-read.

The property is stated in ``rag_loop/prompts.py``, in ``rag_loop/loop.py`` and in the RAG-loop
section of ``docs/pipeline.md``. Request blindness is what makes a loop's behaviour attributable
to its nugget question, which is what makes the k loops comparable to each other and across the
three translation arms. A title threaded in would delete that, and nothing on disk would record
it, because no provenance field carries what a loop was handed.

Three layers, deliberately redundant, because each fails in a different way.

1. Structural: the ``rag_loop/`` source contains no ``title`` identifier at all. This fails on
   the first line of a leak -- a parameter, an attribute, a dict key -- before any wiring exists
   to observe. Parsed rather than grepped, because the docstrings that document this invariant
   name the very fields it forbids, so a text grep would fail on correct code.
2. Boundary: ``round_loop._as_mapping``, the one function that decides what a loop is handed,
   yields exactly ``{nugget_id, question}``.
3. Behavioural: drive the real ``run_rounds`` -> ``run_round`` -> ``_as_mapping`` fan with a
   sentinel title, and assert the sentinel reaches every decompose prompt and nothing a loop was
   given. This is the layer that survives a refactor of the other two.

Each layer is paired with a control that proves it can fail. A scanner reporting "no leak"
against a payload that is leaking is decoration, so
:func:`test_the_leak_detector_detects_a_leak` runs the detector over a poisoned mapping.
"""

from __future__ import annotations

import ast
import asyncio
import types
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Answer, Layout, Retrieved, Statistics, Support
from ragtime.pipeline import round_loop
from tests.pipeline.conftest import audit_response, make_bundle, nuggets_response

pytestmark = pytest.mark.small

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAG_LOOP = _REPO_ROOT / "src" / "ragtime" / "pipeline" / "rag_loop"

#: A string no real prompt, corpus passage or config value can contain, so its presence
#: anywhere is unambiguous evidence of the plumbing under test and never a coincidence.
SENTINEL = "ZZQ-TITLE-SENTINEL-7413"

#: The request fields the loop is forbidden to see. ``title`` is the one this file was written
#: for; the other three are listed because the same delete-the-invariant mistake is available
#: for each and the scan costs nothing extra.
FORBIDDEN = ("title", "problem_statement", "background")


# --------------------------------------------------------------------------- #
# Layer 1, structural: `rag_loop/` does not name any request field in code.
# --------------------------------------------------------------------------- #
def _code_identifiers(path: Path) -> set[str]:
    """Every name this module uses as code, and never a word occurring in prose.

    Collected: parameter names, bare names including the interpolated slots of an f-string,
    attribute names, keyword-argument names, and string literals used as a subscript or dict key.
    That last group is the one a naive walk misses and the realistic leak needs:
    ``round_loop._as_mapping`` hands the loop a dict, so a leak reads ``nugget["title"]``.

    Not collected: docstrings and comments. ``rag_loop/prompts.py``'s own module docstring names
    ``background``, ``problem_statement`` and ``limit`` in the sentence that forbids them, so a
    text-level grep would fail on the correct file.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            found.add(node.arg)
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                found.add(node.slice.value)
        elif isinstance(node, ast.Dict):
            found.update(
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            )
    return found


def test_the_rag_loop_package_never_names_a_request_field_in_code() -> None:
    """No module under ``rag_loop/`` uses ``title``, the problem statement or the background."""
    modules = sorted(_RAG_LOOP.rglob("*.py"))
    assert modules, f"no modules found under {_RAG_LOOP}: the scan would be vacuously green"

    leaks = {
        f"{p.relative_to(_REPO_ROOT)}:{name}"
        for p in modules
        for name in FORBIDDEN
        if name in _code_identifiers(p)
    }
    assert not leaks, (
        "a request field is named as CODE inside the request-blind RAG loop: "
        f"{sorted(leaks)}. The loop's behaviour must be a function of its nugget question "
        "alone (rag_loop/prompts.py:5, loop.py:137); "
        "`title` in particular enters DECOMPOSE only (TOPICS-FIX-and-TITLE.md)."
    )


def test_the_identifier_scanner_sees_code_and_ignores_prose() -> None:
    """The control for the scan above: it must find a real use and not a mentioned word.

    Without this, a ``_code_identifiers`` that returned ``set()`` would make the structural layer
    pass forever.
    """
    tmp_names = _code_identifiers_from_source(
        '"""mentions title only in prose."""\n'
        "def f(nugget, background=1):\n"
        "    return nugget['title'] + f'{problem_statement}'\n"
    )
    assert {"title", "background", "problem_statement"} <= tmp_names, (
        "the scanner must see a dict key, a parameter and an f-string slot"
    )

    prose_only = _code_identifiers_from_source('"""title problem_statement background."""\n')
    assert not ({"title", "problem_statement", "background"} & prose_only), (
        "the scanner must not see a word that only occurs inside a docstring"
    )


def _code_identifiers_from_source(src: str) -> set[str]:
    """:func:`_code_identifiers` over a literal string (same walk, no file)."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = Path(fh.name)
    try:
        return _code_identifiers(path)
    finally:
        path.unlink()


# --------------------------------------------------------------------------- #
# Layer 2, the boundary: `_as_mapping` is the whole of what a loop is handed.
# --------------------------------------------------------------------------- #
def test_as_mapping_hands_a_loop_exactly_its_nugget_id_and_question() -> None:
    """Equality on the key set, not a containment check.

    ``assert "title" not in mapping`` would keep passing if ``problem_statement`` were added
    instead, and the invariant is about the whole request, not about one field of it.
    """
    rich = {
        "nugget_id": "2000#n0",
        "question": "Which agency enforces it?",
        "title": SENTINEL,
        "problem_statement": SENTINEL,
        "background": SENTINEL,
        "weight": 0.9,
        "vital": True,
    }
    assert set(round_loop._as_mapping(rich)) == {"nugget_id", "question"}
    assert SENTINEL not in str(round_loop._as_mapping(rich))


def test_as_mapping_is_equally_narrow_for_a_record_shaped_nugget() -> None:
    """The attribute branch: a real ``common.Nugget`` carries weight/vital/status too."""
    record = types.SimpleNamespace(
        nugget_id="2000#n0",
        question="Which agency enforces it?",
        title=SENTINEL,
        weight=0.9,
        vital=True,
    )
    assert set(round_loop._as_mapping(record)) == {"nugget_id", "question"}
    assert SENTINEL not in str(round_loop._as_mapping(record))


# --------------------------------------------------------------------------- #
# Layer 3, behavioural: the real fan, with a sentinel title.
# --------------------------------------------------------------------------- #
def _leaks(payload: Any, sentinel: str) -> bool:
    """Does ``sentinel`` appear anywhere in ``payload``, however nested?

    ``repr`` rather than a structured walk, because the question is whether the model could
    possibly have been shown this string, and a value that survives ``repr`` is a value that
    could reach a prompt. :func:`test_the_leak_detector_detects_a_leak` shows it can say yes.
    """
    return sentinel in repr(payload)


def test_the_leak_detector_detects_a_leak() -> None:
    """The control. A detector that can only say "clean" certifies nothing."""
    assert _leaks({"nugget_id": "n0", "title": SENTINEL}, SENTINEL)
    assert _leaks([{"a": [("b", SENTINEL)]}], SENTINEL)
    assert not _leaks({"nugget_id": "n0", "question": "who?"}, SENTINEL)


def _cfg(decompose_cfg):
    """The real ``decomposition`` block plus the two leaves ``run_rounds`` reads."""
    decompose_cfg.blocks["decomposition"]["dedup"]["llm_paraphrase_merge"] = False
    decompose_cfg.blocks["decomposition"]["R_max"] = 1
    decompose_cfg.blocks["decomposition"]["min_new"] = 1
    decompose_cfg.blocks["decomposition"]["low_streak"] = 1
    decompose_cfg.blocks["retrieval"] = {"index": "original"}
    decompose_cfg.retrieval_index = "original"
    return decompose_cfg


def _loop_result(nugget_id: str) -> Any:
    return types.SimpleNamespace(
        nugget_id=nugget_id,
        question="q",
        status="answered",
        answers=(
            Answer(
                answer="the Food Act",
                sentence="The Food Act applies.",
                quoted_span="the Food Act",
                support=(Support(passage_id="d1#p0", lang="en"),),
            ),
        ),
        retrieved=(Retrieved(passage_id="d1#p0", score=9.0),),
        search_trail=[],
        turns=1,
        searches=1,
        claims_committed=1,
        claims_rejected=0,
        contested=False,
        closed_by="model",
    )


def test_a_topic_title_reaches_every_decompose_call_and_no_loop(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """One sentinel, two destinations, opposite verdicts, in one real run.

    ``run_rounds`` is driven through round 0 and one audit round, so all four decompose calls
    that take the request are exercised -- seed draft, self-critique, coverage audit, on-topic
    gate -- and the real ``run_round`` -> ``_as_mapping`` fan runs unmodified. Only ``run_loop``
    is replaced, by a recorder, since it is the boundary being measured.

    Both halves are asserted together. "The title never reached a loop" is trivially true of a
    run where nothing was decomposed and no loop ran, so the same test proves the title reached
    decompose and that loops ran.
    """
    cfg = _cfg(decompose_cfg)
    clients = make_bundle(
        nuggets=[
            nuggets_response(["What law applies?"]),
            nuggets_response(["What law applies?"], weights=[0.9]),
        ],
        audit=[audit_response(add=[("Which agency enforces it?", None, 0.8)])],
        on_topic={"rationale": "a genuine facet", "on_topic": True},
    )

    given_to_loops: list[dict[str, Any]] = []

    async def _recording_run_loop(nugget, cfg_, *, llm, ctx, seed, stats, passage_lang):
        given_to_loops.append(dict(nugget))
        return _loop_result(str(nugget["nugget_id"]))

    monkeypatch.setattr(round_loop, "run_loop", _recording_run_loop)

    asyncio.run(
        round_loop.run_rounds(
            cfg,
            "2061",
            "Report on food law.",
            "You brief a committee.",
            5000,
            llm=object(),
            ctx=object(),
            clients=clients,
            layout=Layout(run_dir=tmp_path / "run", base=tmp_path),
            seed=0,
            ceiling=4,
            stats=Statistics(),
            title=SENTINEL,
        )
    )

    # (a) Loops actually ran; without this, (b) is vacuous.
    assert given_to_loops, "no loop ran, so the blindness assertion below would prove nothing"

    # (b) Nothing a loop was given carries the title.
    leaking = [n for n in given_to_loops if _leaks(n, SENTINEL)]
    assert not leaking, (
        f"the topic title reached {len(leaking)} RAG loop(s): {leaking}. The loop is "
        "request-blind by construction; `title` enters DECOMPOSE only "
        "(TOPICS-FIX-and-TITLE.md). Look at round_loop._as_mapping."
    )
    assert all(set(n) == {"nugget_id", "question"} for n in given_to_loops)

    # (c) The title reached decompose: every prompt built from the request carries it.
    by_schema = {name: clients.llm.calls_for(name) for name in ("nuggets", "coverage_audit", "on_topic_gate")}
    for name, calls in by_schema.items():
        assert calls, f"the {name} call never happened, so its half of this test is vacuous"
        assert all(SENTINEL in c.prompt for c in calls), (
            f"a {name} prompt did not carry the title; decompose must see the whole request"
        )
    assert len(by_schema["nuggets"]) >= 2, "round 0 issues a draft AND a self-critique call"


def test_the_dedup_confirm_call_is_the_one_decompose_call_that_stays_request_free(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """The scope control for the test above: decompose seeing the title is not everything
    seeing it.

    ``dedup_confirm_prompt`` compares two nugget questions and takes no request field at all;
    handing it a title would be scope creep dressed as consistency. Pinned so that anyone
    threading a field through this package has to justify skipping this one.
    """
    from ragtime.pipeline.decompose.prompts import dedup_confirm_prompt

    text = dedup_confirm_prompt("Which agency enforces it?", "Who enforces the law?")
    assert SENTINEL not in text
    assert "PROBLEM STATEMENT" not in text
    assert "TITLE" not in text
