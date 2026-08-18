"""The topic ``title`` enters the decompose prompts, and only those.

The organisers' ``-fix`` topics file added a ``title`` to all 103 topics. The title is part of
the report request and enters decompose only. The half of that sentence after "only" lives in
``test_request_blindness_small.py``; this file owns the half before it.

Four groups, in the order they can fail:

1. Placement. The title is rendered after the problem statement and before the background, and
   is never labelled primary. Position encodes precedence here, and ``prompts._title_section``
   carries the argument, so a test that only checked that the title appears somewhere in the
   prompt would pass against the headline layout the design rejects.
2. Omission and additivity. No title -- an absent key, ``""``, or whitespace -- renders no
   section, which is the rule ``background`` already follows, and the resulting prompt is
   byte-identical to what the pre-title code produced. That is what makes the change strictly
   additive, and why the anti-drift clause lives inside the title section rather than being
   folded into the existing sentence about not decomposing the background alone.
3. Threading. All four request-bearing decompose calls receive it: seed draft, self-critique,
   coverage audit and on-topic gate; and the driver reads ``Topic.title``. Each is asserted on
   the prompt the stub LLM received, never on a call signature.
4. What was not changed. The frozen few-shot exemplar set gains no title. Editing
   ``_AUTONUGGETIZER_V1`` in place is forbidden by its own module contract and would move
   ``provenance.prompt_hash`` for reasons unrelated to the request.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from ragtime.common import Layout, Statistics
from ragtime.pipeline import driver, round_loop
from ragtime.pipeline.decompose import grow_nuggets
from ragtime.pipeline.decompose.exemplars import EXEMPLAR_SETS, Exemplar
from ragtime.pipeline.decompose.prompts import (
    gap_audit_prompt,
    on_topic_prompt,
    seed_prompt,
    self_critique_prompt,
)
from tests.pipeline.conftest import audit_response, make_bundle, nuggets_response

pytestmark = pytest.mark.small

TITLE = "MARKER_TITLE"
SPINE = "MARKER_PROBLEM"
PERSONA = "MARKER_BACKGROUND"

#: The four request-bearing prompts, each as ``(name, render)``. Parameterising over the set is
#: what turns "one of them was forgotten" into a failing test rather than something nobody looks
#: for.
_REQUEST_PROMPTS = (
    (
        "seed",
        lambda **kw: seed_prompt(SPINE, PERSONA, (4, 8), **kw),
    ),
    (
        "self_critique",
        lambda **kw: self_critique_prompt(SPINE, PERSONA, ["Who enforces it?"], (4, 8), **kw),
    ),
    (
        "gap_audit",
        lambda **kw: gap_audit_prompt(
            SPINE,
            PERSONA,
            [{"nugget_id": "2000#n0", "question": "Who enforces it?", "aggregator_type": "OR"}],
            (),
            round=1,
            **kw,
        ),
    ),
    (
        "on_topic",
        lambda **kw: on_topic_prompt("Who enforces it?", SPINE, None, **kw),
    ),
)


# --------------------------------------------------------------------------- #
# 1. Placement: after the problem statement, before the background, never primary.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,render", _REQUEST_PROMPTS, ids=[n for n, _ in _REQUEST_PROMPTS])
def test_the_title_is_rendered_after_the_spine(name: str, render) -> None:
    """The problem statement keeps first position and the ``primary`` label; the title follows.

    Ordering, not mere presence. A headline layout, with the title above the problem statement,
    would satisfy a presence check and is what ``_title_section`` argues against: a two- to
    four-word subject label in first-read position is the most decomposable thing on the page,
    while the facets live in the 300-word problem statement.
    """
    text = render(title=TITLE)
    assert TITLE in text, f"{name} did not render the title at all"
    assert text.index(SPINE) < text.index(TITLE), (
        f"{name} rendered the title BEFORE the problem statement; the spine comes first and "
        "outranks the label (01_decompose.md §8, prompts.py module docstring)"
    )
    head = text[: text.index(SPINE)]
    if name == "on_topic":
        # The gate labels the problem statement "the information need" rather than "primary:
        # the coverage spine", because it decomposes nothing. The ordering claim above is the
        # one that transfers; asserting the seed prompt's wording here would pin a label this
        # prompt has never carried.
        assert "the information need" in head.lower()
    else:
        assert "primary" in head.lower() and "spine" in head.lower()


@pytest.mark.parametrize(
    "name,render",
    [p for p in _REQUEST_PROMPTS if p[0] != "on_topic"],
    ids=[p[0] for p in _REQUEST_PROMPTS if p[0] != "on_topic"],
)
def test_the_title_is_rendered_before_the_background(name: str, render) -> None:
    """Between the problem statement and the background: precedence matches label order.

    ``on_topic`` is excluded because it renders no background at all. The gate never sees the
    persona, which is the asymmetry the title's placement argument turns on.
    """
    text = render(title=TITLE)
    assert text.index(TITLE) < text.index(PERSONA), (
        f"{name} rendered the title after the background; a title names the SUBJECT of the "
        "request and outranks a persona"
    )


def test_the_title_section_is_never_labelled_primary_and_forbids_decomposing_itself() -> None:
    """The anti-drift clause travels with the field it governs.

    Without it the title is just another labelled block of request text, and the model has no
    reason to prefer the problem statement's facets to the label's words.
    """
    text = seed_prompt(SPINE, PERSONA, (4, 8), title=TITLE)
    section = text[text.index("TITLE (") : text.index("BACKGROUND (")]
    assert "primary" not in section.lower()
    assert "not itself the information need" in section.lower()
    assert "never as the thing to decompose" in section.lower()
    # ... and it must not let the label narrow the request.
    assert "even if the title does not mention it" in section.lower()


# --------------------------------------------------------------------------- #
# 2. Omission and additivity: no title means the pre-title bytes, exactly.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,render", _REQUEST_PROMPTS, ids=[n for n, _ in _REQUEST_PROMPTS])
@pytest.mark.parametrize("empty", ["", "   ", "\n\t "])
def test_an_empty_title_renders_no_section_at_all(name: str, render, empty: str) -> None:
    """Omitted entirely, never rendered blank: the rule ``background`` follows, unchanged.

    A labelled but empty field is something to hallucinate into, and the pre-``-fix`` topics
    file carries no ``title`` key at all, so this path is real rather than defensive.
    """
    text = render(title=empty)
    assert "TITLE (" not in text, f"{name} rendered a labelled-but-blank TITLE section"


@pytest.mark.parametrize("name,render", _REQUEST_PROMPTS, ids=[n for n, _ in _REQUEST_PROMPTS])
def test_a_title_less_prompt_is_byte_identical_to_omitting_the_argument(name: str, render) -> None:
    """The additivity property: a topic with no title gets exactly the old prompt.

    This is what makes the question "did the title change the bank?" answerable. If the no-title
    path had drifted by a word, a title-less arm and the earlier code would be incomparable, and
    a change in the round-0 baseline would have two causes instead of one.
    """
    assert render(title="") == render()
    assert render(title="   ") == render()


# --------------------------------------------------------------------------- #
# 3. Threading: every request-bearing decompose call, and the driver.
# --------------------------------------------------------------------------- #
def _cfg(decompose_cfg):
    decompose_cfg.blocks["decomposition"]["dedup"]["llm_paraphrase_merge"] = False
    decompose_cfg.blocks["retrieval"] = {"index": "original"}
    decompose_cfg.retrieval_index = "original"
    return decompose_cfg


def test_round_zero_threads_the_title_into_both_of_its_calls(decompose_cfg) -> None:
    """The seed draft and the self-critique. Asymmetry between them is the failure to catch.

    Self-critique re-judges coverage and weight against the request. Shown a smaller request
    than the draft was, it would edit for a difference that is an artefact of the plumbing.
    """
    clients = make_bundle(
        nuggets=[
            nuggets_response(["What law applies?"]),
            nuggets_response(["What law applies?"], weights=[0.9]),
        ]
    )
    asyncio.run(
        grow_nuggets(
            "Report on food law.", "You brief a committee.", (), None, 0,
            cfg=_cfg(decompose_cfg), clients=clients, topic_id="2061",
            limit=5000, seed=7, title=TITLE,
        )
    )
    calls = clients.llm.calls_for("nuggets")
    assert len(calls) == 2, f"round 0 issues a draft and a self-critique; saw {len(calls)}"
    assert all(TITLE in c.prompt for c in calls)


def test_round_zero_without_a_title_shows_no_title_section(decompose_cfg) -> None:
    """The negative twin: without it the test above passes against a hardcoded string."""
    clients = make_bundle(
        nuggets=[
            nuggets_response(["What law applies?"]),
            nuggets_response(["What law applies?"], weights=[0.9]),
        ]
    )
    asyncio.run(
        grow_nuggets(
            "Report on food law.", "You brief a committee.", (), None, 0,
            cfg=_cfg(decompose_cfg), clients=clients, topic_id="2061", limit=5000, seed=7,
        )
    )
    assert all("TITLE (" not in c.prompt for c in clients.llm.calls_for("nuggets"))


def test_the_audit_round_threads_the_title_into_the_audit_and_the_on_topic_gate(
    decompose_cfg,
) -> None:
    """From round 1 on, the auditor proposes request-driven gaps and the gate admits them.

    Both must see the same request the seed saw. A gate judging against a smaller request would
    reject proposals for reasons that have nothing to do with topicality.
    """
    from ragtime.pipeline.decompose import bank as bank_ops
    from ragtime.pipeline.decompose import coverage_audit

    bank = bank_ops.mint("2061", ["What law applies?"], origin_round=0)
    evidence = coverage_audit.evidence_from_results(
        [
            types.SimpleNamespace(
                nugget_id=bank[0].nugget_id,
                question="q",
                status="answered",
                answers=(),
                retrieved=(),
                search_trail=[],
            )
        ]
    )
    clients = make_bundle(
        nuggets=[nuggets_response([])],
        audit=[audit_response(add=[("Which agency enforces it?", None, 0.8)])],
        on_topic={"rationale": "a genuine facet", "on_topic": True},
    )
    asyncio.run(
        grow_nuggets(
            "Report on food law.", "You brief a committee.", bank, evidence, 1,
            cfg=_cfg(decompose_cfg), clients=clients, topic_id="2061",
            limit=5000, seed=7, title=TITLE,
        )
    )
    for schema in ("coverage_audit", "on_topic_gate"):
        calls = clients.llm.calls_for(schema)
        assert calls, f"the {schema} call never happened; this assertion would be vacuous"
        assert all(TITLE in c.prompt for c in calls), f"{schema} did not see the title"


def test_solve_topic_reads_topic_title_and_hands_it_to_decompose(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """The driver half: ``Topic.title`` into ``grow_nuggets``, through the real ``solve_topic``.

    Asserted on the prompt the stub LLM received, not on a call signature. The field being read
    off the topic and the field reaching the model are different claims, and only the second one
    matters.
    """
    clients = make_bundle(
        nuggets=[
            nuggets_response(["What law applies?"]),
            nuggets_response(["What law applies?"], weights=[0.9]),
        ]
    )

    async def _no_loops(bank, cfg, **kw):
        return round_loop.RoundResult(
            round_no=0, results=(), wall_s=0.0, sequential_wall_s=0.0
        )

    monkeypatch.setattr(driver, "run_round", _no_loops)

    topic = types.SimpleNamespace(
        topic_id="2061",
        problem_statement="Report on food law.",
        background="You brief a committee.",
        limit=5000,
        title=TITLE,
    )
    asyncio.run(
        driver.solve_topic(
            _cfg(decompose_cfg), topic,
            llm=object(), ctx=object(), clients=clients,
            layout=Layout(run_dir=tmp_path / "run", base=tmp_path),
            seed=0, ceiling=1, stats=Statistics(),
        )
    )
    calls = clients.llm.calls_for("nuggets")
    assert calls, "decompose never ran"
    assert all(TITLE in c.prompt for c in calls)


def test_solve_topic_survives_a_topic_that_has_no_title(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """The pre-``-fix`` topics file, and every plain-dict test double, has no ``title``.

    ``_field`` returns ``None``, which becomes ``""``, which renders no section. A crash here
    would make a title-less request unusable through the driver.
    """
    clients = make_bundle(
        nuggets=[
            nuggets_response(["What law applies?"]),
            nuggets_response(["What law applies?"], weights=[0.9]),
        ]
    )

    async def _no_loops(bank, cfg, **kw):
        return round_loop.RoundResult(
            round_no=0, results=(), wall_s=0.0, sequential_wall_s=0.0
        )

    monkeypatch.setattr(driver, "run_round", _no_loops)

    result = asyncio.run(
        driver.solve_topic(
            _cfg(decompose_cfg),
            {
                "topic_id": "2061",
                "problem_statement": "Report on food law.",
                "background": "You brief a committee.",
                "limit": 5000,
            },
            llm=object(), ctx=object(), clients=clients,
            layout=Layout(run_dir=tmp_path / "run", base=tmp_path),
            seed=0, ceiling=1, stats=Statistics(),
        )
    )
    assert result.nuggets > 0
    assert all("TITLE (" not in c.prompt for c in clients.llm.calls_for("nuggets"))


# --------------------------------------------------------------------------- #
# 5. What was not changed.
# --------------------------------------------------------------------------- #
def test_the_frozen_exemplar_set_gains_no_title() -> None:
    """``Exemplar`` stays ``(problem_statement, background, nuggets)``.

    ``exemplars.py``'s contract is that a v2 is a new constant and never an in-place edit of v1,
    because the ``decomposition`` block's hash is ``provenance.prompt_hash`` and is shared
    byte-identically across a run family. Adding a title to the demonstrations is therefore an
    ``autonuggetizer_v2`` plus a change to ``decomposition.few_shot`` in all six configs: a
    fairness-family decision, not a prompt edit.
    """
    assert set(Exemplar.__dataclass_fields__) == {"problem_statement", "background", "nuggets"}
    assert set(EXEMPLAR_SETS) == {"autonuggetizer_v1"}

    text = seed_prompt(SPINE, PERSONA, (4, 8), EXEMPLAR_SETS["autonuggetizer_v1"][:2], title=TITLE)
    example_block = text[text.index("Worked examples") :]
    assert "TITLE (" not in example_block, "the few-shot examples must not sprout a title"
    assert TITLE not in example_block


def test_run_round_takes_no_title_parameter() -> None:
    """The fan over the request-blind loops takes no title argument, by design.

    A ``title=`` here would be a parameter with nowhere legitimate to go, and its existence is
    the invitation to use it. ``run_rounds``, the coverage loop that owns decompose, has one;
    ``run_round``, the fan, must not.
    """
    import inspect

    assert "title" in inspect.signature(round_loop.run_rounds).parameters
    assert "title" not in inspect.signature(round_loop.run_round).parameters
