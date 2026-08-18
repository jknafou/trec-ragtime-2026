"""A one-action menu requires that action's own fields, so an empty `submit_claim` is unemittable.

The defect: `{"rationale": ..., "action": "submit_claim"}`, with no `claim`, no `passage_id`
and no `span`, was grammar-legal, passed `jsonschema.validate` and arrived as an empty span,
so it died on `commit_span`'s first precondition. The bounded retry re-decoded against the
same permissive grammar, so both retries were structurally guaranteed to reproduce it. Of 30
dropped claims, 24 were `empty_span` and none were `not_verbatim`, so the commit gate itself
never fired.

This does not relax the commit gate. Every check in `commit.py` still runs, unchanged, on the
same symmetric NFC comparison. It only stops the model spending a generation on an object
that could never have passed them: `action_schema_for` exists to make an option
unrepresentable for one generation rather than reject it afterwards, and this applies that
one level down from the enum.

What is asserted here is schema shape. Whether xgrammar honours that shape is a separate
question, because on this stack a structured-output request can be accepted and silently
ignored. It is probed live instead, each probe with a permissive control showing the model
would emit the bad shape when allowed: `required` is enforced (8/8 against the control's
0/8), `maxLength` is enforced, and for the multi-action menu `oneOf` is enforced (0/6
against a flat control's 6/6) while `if`/`then` is accepted and ignored.

Not covered here:
  - whether forcing the span changes the commit rate per language, which needs a real cell.
  - whether it changes the abstain rate. Making a shape unrepresentable does not make the
    model able to quote, so it may now abstain where it used to emit an empty claim. That is
    still better, since an abstain costs one generation and an empty claim cost three, but it
    is a behaviour change and it is not measured here.
"""

from __future__ import annotations

import pytest

from ragtime.pipeline.rag_loop.phase_gate import ABSTAIN, SEARCH, SUBMIT_ANSWER, SUBMIT_CLAIM
from ragtime.serving.schemas import action_schema_for, actions_of


def _schema(menu) -> dict:
    """The raw JSON-schema dict behind a CompiledSchema."""
    cs = action_schema_for(tuple(menu))
    for attr in ("schema", "raw", "payload", "json_schema", "spec", "_schema"):
        value = getattr(cs, attr, None)
        if isinstance(value, dict):
            return value
    raise AssertionError("could not reach the compiled schema's raw dict")


@pytest.mark.small
def test_a_lone_submit_claim_cannot_be_emitted_empty() -> None:
    """24 of the 30 dropped claims were exactly this shape."""
    req = _schema((SUBMIT_CLAIM,))["required"]
    for field in ("claim", "passage_id", "span"):
        assert field in req, (
            f"{field!r} is not required when the menu is submit_claim ONLY: an object without it "
            "is grammar-legal and dies on commit_span's first precondition, wasting a generation"
        )


@pytest.mark.small
def test_an_EMPTY_STRING_span_is_refused_too() -> None:
    """`commit_span`'s first check is `if not span`, which an empty string satisfies exactly as an
    absent key does. `required` alone would close half the hole."""
    props = _schema((SUBMIT_CLAIM,))["properties"]
    for field in ("claim", "passage_id", "span"):
        assert props[field].get("minLength") == 1, f"{field!r} may still be emitted as ''"


@pytest.mark.small
def test_a_lone_search_must_carry_its_query() -> None:
    """Same defect, different action: `{"action": "search"}` with no query is schema-valid and
    raises `ValueError` in `retrieval.tool.search_action`: a whole turn spent on a malformed
    object. `run_loop` catches it (MAT-9), but catching is not the same as preventing."""
    assert "query" in _schema((SEARCH,))["required"]


@pytest.mark.small
@pytest.mark.parametrize("action", [SUBMIT_ANSWER, ABSTAIN])
def test_the_PAYLOADLESS_terminals_gain_no_requirement(action: str) -> None:
    """`submit_answer` carries no value of its own: the short answer and the report sentence ride
    with each CLAIM during GATHER, and `abstain` carries nothing either.

    Requiring a field of them would make the terminal turn UNEMITTABLE, which turns the budget
    backstop into a hang: `forced_terminal_menu` offers exactly one action, and if the grammar
    cannot express it the loop cannot end. This test is the guard on that mistake.
    """
    assert _schema((action,))["required"] == ["rationale", "action"]


@pytest.mark.small
def test_a_MULTI_action_menu_binds_each_action_to_ITS_OWN_fields() -> None:
    """A multi-action menu is a `oneOf` of per-action branches, and each branch is strict.

    A flat union has to stay permissive, because the only alternative it offers is requiring
    `span` for every action, which would make searching impossible.
    `oneOf` removes that trade-off: `search` gets a branch requiring `query` and not `span`, while
    `submit_claim` gets one requiring a non-empty `span`. Both properties can now hold at once, so
    the test asserts the thing that actually matters: no action is forced to carry another's
    fields, and no action may be emitted empty.

    With the flat multi-action menu, `empty_span` was 45 of 60 claim rejections and 45 of 78
    claim turns carried no span at all. `if`/`then` is accepted and then ignored on this
    stack; `oneOf` is enforced (0/6), confirmed against the real compiled menu, with a flat
    control that produced 6/6 empty.
    """
    multi = _schema((SEARCH, SUBMIT_CLAIM, SUBMIT_ANSWER))
    branches = {b["properties"]["action"]["enum"][0]: b for b in multi["oneOf"]}
    assert set(branches) == {SEARCH, SUBMIT_CLAIM, SUBMIT_ANSWER}

    # submit_claim cannot be emitted empty ...
    claim = branches[SUBMIT_CLAIM]
    assert {"claim", "passage_id", "span"} <= set(claim["required"])
    assert claim["properties"]["span"]["minLength"] == 1
    # ... and search is not dragged along with it, which is the property the old test protected.
    search = branches[SEARCH]
    assert "span" not in search["required"]
    assert "query" in search["required"]


@pytest.mark.small
def test_narrowing_still_narrows_the_enum() -> None:
    """The original guarantee must survive the new one: the phase gate is enforced at the GRAMMAR
    level, not only in code.

    Read through `actions_of`, which is the one owner of "which menu is this schema": a
    single-action menu is a plain object and a multi-action menu is a `oneOf`, and hand-reading
    `["properties"]["action"]["enum"]` breaks on the second shape.
    """
    assert actions_of(action_schema_for((SUBMIT_CLAIM,))) == (SUBMIT_CLAIM,)
    assert actions_of(action_schema_for((SEARCH, ABSTAIN))) == (SEARCH, ABSTAIN)


@pytest.mark.small
def test_the_cache_does_not_serve_a_stale_shape() -> None:
    """`action_schema_for` memoizes by menu tuple. A cache populated before the requirement was
    added would hand back the permissive schema forever, and the symptom would be indistinguishable
    from the fix not working."""
    first = _schema((SUBMIT_CLAIM,))
    second = _schema((SUBMIT_CLAIM,))
    assert first["required"] == second["required"]
    assert "span" in second["required"]
