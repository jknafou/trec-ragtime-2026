"""Small tier for the RAG loop: its control flow, against a scripted mock LLM.

Every test here derives from the RAG-loop specification in ``docs/pipeline.md`` and exercises the
loop's logic, never data or scale. The full tier (``test_ragloop_full.py``) runs the same loop
against the real vLLM and the real index; a test that can fail without either belongs here, where
it costs a fraction of a second instead of a service.

The three text fixtures, and the corpus property each one carries:

* ``EN_TEXT``, an ordinary English passage. A quoted span is a substring of it, which the corpus
  satisfies trivially.
* ``ZH_TEXT``, Chinese with no space between sentences. A passage is one contiguous slice of its
  document, so a span may cross what looks like a sentence boundary with no separator. This is
  why joining member sentences with ``" ".join`` corrupts most multi-sentence Chinese passages:
  the source has no separator between CJK sentences. A fixture of space-separated English would
  pass while asserting the opposite of the corpus.
* ``NFD_SPAN``, a span in NFD facing a passage stored in NFC. The commit check normalises both
  sides, so it matches. An ASCII-only fixture would make the symmetric ``nfc`` call look
  redundant and let a one-sided normalisation ship green, which would then behave differently per
  language, since NFC/NFD divergence is concentrated in the non-English renderings.
"""

from __future__ import annotations

import asyncio
import unicodedata
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar

import pytest

from ragtime.pipeline.rag_loop import (
    Budget,
    CommittedClaim,
    LoopState,
    PhaseGateViolation,
    Rejection,
    assert_allowed,
    cluster_key,
    commit_span,
    fan_in,
    forced_terminal_menu,
    phase_menu,
    run_loop,
)
from ragtime.pipeline.rag_loop import actions as actions_mod
from ragtime.serving.schemas import actions_of

pytestmark = pytest.mark.small

EN_TEXT = "Reactions attributed to the MSG symptom complex most commonly include headache."
# No space between the two sentences -- the real CJK shape (see the module docstring).
ZH_TEXT = "谷氨酸钠症候群的反应包括头痛。也有人报告出汗和潮红。"
# "café" decomposed: e + U+0301. The stored passage is NFC; the model quotes NFD.
NFD_SPAN = unicodedata.normalize("NFD", "café au lait")
NFC_TEXT = unicodedata.normalize("NFC", "They served café au lait in the lobby.")

SHOWN = {
    "d1#p1": EN_TEXT,
    "d2#p3": ZH_TEXT,
    "d3#p7": NFC_TEXT,
}


# --------------------------------------------------------------------------- #
# The scripted mock LLM + a fake search, so the loop runs with no GPU and no index.
# --------------------------------------------------------------------------- #
class ScriptedLlm:
    """Emits a fixed action trail, and records the menu it was constrained to on every turn.

    Several tests below assert on the grammar the loop offered, which is what actually enforces
    the phase gate. Asserting only on the outcome would pass even with a wrong menu, as long as
    the model happened to behave.
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.menus: list[tuple[str, ...]] = []
        self.calls = 0

    async def generate(self, schema: Any, prompt: str, seed: int, **kw: Any) -> dict[str, Any]:
        self.calls += 1
        # Via `actions_of`, not by reaching into the schema: a multi-action menu is a `oneOf`
        # of per-action branches and has no top-level `properties`, so reading the schema by
        # hand here turns one deliberate schema change into a wall of false loop failures.
        self.menus.append(actions_of(schema))
        if not self.script:
            # A loop that outruns its script is a fault in the test, and must look like one
            # rather than abstaining forever.
            raise AssertionError("the loop asked for more turns than the script provides")
        return self.script.pop(0)


@dataclass
class FakeSearchResult:
    queries: tuple[str, ...]
    hits: tuple[tuple[str, float], ...]
    passages: tuple[tuple[str, str, str], ...]
    passage_lang: str
    dropped_for_budget: int = 0

    @property
    def passage_ids(self) -> tuple[str, ...]:
        return tuple(p[0] for p in self.passages)

    @property
    def shown(self) -> int:
        return len(self.passages)


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the retrieval tool call with a fixed result.

    A substitution rather than a spy: a spy that re-looks-up the name it replaced recurses.
    Nothing here calls the real ``search_action``, so there is no name to re-resolve.
    """

    def _fake(ctx: Any, action: Any, *, top_k: int, char_budget: Any, passage_lang: Any):
        return FakeSearchResult(
            queries=(str(action.get("query", "")),),
            hits=(("d1#p1", 0.94), ("d2#p3", 0.88), ("d3#p7", 0.71)),
            passages=tuple((pid, pid.split("#", 1)[0], SHOWN[pid]) for pid in SHOWN),
            passage_lang=passage_lang or "original",
        )

    monkeypatch.setattr(actions_mod, "search_action", _fake)
    monkeypatch.setattr(actions_mod, "format_passages", lambda r, **kw: f"PASSAGES({r.shown})")


class FakeCtx:
    """A retrieval context stub exposing only what the loop actually touches."""

    class _Store:
        _LANG: ClassVar[dict[str, str]] = {"d1#p1": "en", "d2#p3": "zh", "d3#p7": "es"}

        def passage(self, pid: str) -> Any:
            # `self._LANG`, not `self._Store._LANG`: `_Store` is a nested class of `FakeCtx`, so
            # the latter does not resolve from inside it and raises AttributeError, which
            # `actions._native_lang` catches and turns into "". The stub then reports every
            # passage as language-unknown and every `Support.lang` here reads "". A stub that
            # fails into the same value its caller uses for "unknown" is invisible by
            # construction.
            return type("P", (), {"lang": self._LANG[pid]})()

    passage_store = _Store()


class Cfg:
    """The shipped rag_loop block, as `config.load` would hand it over."""

    def __init__(self, **over: Any) -> None:
        budget = {
            "min_searches": 2,
            "max_iters": 8,
            "max_searches": 5,
            "token_budget": 20000,
        }
        budget.update(over.pop("budget", {}))
        retries = over.pop("claim_commit_max_retries", 2)
        self.blocks = {
            "rag_loop": {"budget": budget, "search_top_k": 20, **over},
            "claim_commit": {"max_retries": retries},
        }


def _run(script: list[dict[str, Any]], cfg: Cfg | None = None) -> tuple[Any, ScriptedLlm]:
    llm = ScriptedLlm(script)
    result = asyncio.run(
        run_loop(
            {"nugget_id": "2061#n2", "question": "What reactions are attributed to MSG?"},
            cfg or Cfg(),
            llm=llm,
            ctx=FakeCtx(),
        )
    )
    return result, llm


def _act(action: str, **kw: Any) -> dict[str, Any]:
    return {"rationale": "because", "action": action, **kw}


# --------------------------------------------------------------------------- #
# The phase gate, as a pure function: no loop, no model.
# --------------------------------------------------------------------------- #
def test_abstain_is_absent_on_turn_one_premature_abstain_guard():
    """At n_search=0 and n_answer_fail=0, abstain is not in the grammar. A nugget with no
    evidence must cost a real search before it can be given up."""
    assert "abstain" not in phase_menu(LoopState(), Budget())


def test_abstain_needs_BOTH_halves_of_the_effort_floor():
    """F = (n_search >= min_searches) and (n_answer_fail >= 1). Either half alone is not enough,
    and the conjunction is what keeps abstain reachable only with no committed claim."""
    b = Budget(min_searches=2)
    assert "abstain" not in phase_menu(LoopState(n_search=5, n_answer_fail=0), b)
    assert "abstain" not in phase_menu(LoopState(n_search=0, n_answer_fail=3), b)
    assert "abstain" in phase_menu(LoopState(n_search=2, n_answer_fail=1), b)


def test_search_leaves_the_menu_at_the_cap_but_closing_never_does():
    b = Budget(max_searches=5)
    assert "search" not in phase_menu(LoopState(n_search=5), b)
    menu = phase_menu(LoopState(n_search=5), b)
    assert "submit_claim" in menu and "submit_answer" in menu


def test_forced_terminal_offers_exactly_one_action_and_never_a_closed_book_answer():
    """The backstop forces exactly one terminal. With no committed claim the only option is
    abstain, so an ungrounded guess is unreachable rather than merely discouraged."""
    assert forced_terminal_menu(False) == ("abstain",)
    assert forced_terminal_menu(True) == ("submit_answer",)


def test_out_of_menu_action_raises_rather_than_being_tolerated():
    """A structured-output request can be silently ignored on this stack, through the dead
    `guided_json` field, so the code-side re-check is a real guard rather than a duplicate."""
    with pytest.raises(PhaseGateViolation):
        assert_allowed("abstain", ("submit_claim", "submit_answer"))


# --------------------------------------------------------------------------- #
# The verbatim NFC span commit, and the always-original doc-id.
# --------------------------------------------------------------------------- #
def test_verbatim_span_commits_and_derives_the_original_doc_id():
    out = commit_span(
        claim="Headache is a reaction.",
        span="most commonly include headache",
        passage_id="d1#p1",
        shown_text=EN_TEXT,
        native_lang="en",
    )
    assert isinstance(out, CommittedClaim)
    assert out.doc_id == "d1"  # derived, never model-supplied


def test_non_verbatim_span_is_rejected_and_never_repaired():
    out = commit_span(
        claim="c", span="most commonly  include headache", passage_id="d1#p1", shown_text=EN_TEXT
    )
    assert isinstance(out, Rejection) and out.reason == "not_verbatim"


def test_span_crossing_a_CJK_sentence_boundary_commits():
    """A passage is one contiguous slice; there is no separator between Chinese sentences, so a
    span may legitimately cross the boundary. A fixture of space-separated English would never
    exercise this and would let a sentence-joining implementation pass."""
    span = "包括头痛。也有人报告"
    assert span in ZH_TEXT
    out = commit_span(claim="c", span=span, passage_id="d2#p3", shown_text=ZH_TEXT, native_lang="zh")
    assert isinstance(out, CommittedClaim)


def test_normalization_is_symmetric_so_an_NFD_quote_of_NFC_text_commits():
    out = commit_span(claim="c", span=NFD_SPAN, passage_id="d3#p7", shown_text=NFC_TEXT)
    assert isinstance(out, CommittedClaim), "nfc must be applied to BOTH sides, not just one"


def test_citing_a_passage_never_shown_is_rejected():
    out = commit_span(claim="c", span="anything", passage_id="d9#p9", shown_text=None)
    assert isinstance(out, Rejection) and out.reason == "unseen_passage"


# --------------------------------------------------------------------------- #
# Fan-in.
# --------------------------------------------------------------------------- #
def test_fan_in_clusters_paraphrases_into_one_answer_with_unioned_doc_ids():
    a = fan_in([
        CommittedClaim("Headache is a reaction.", "headache", "d1#p1", "d1", "en",
                       answer="headache", sentence="Headache is a reaction."),
        CommittedClaim("headache is a reaction.", "headache", "d2#p3", "d2", "zh",
                       answer="HEADACHE ", sentence="Headache is a reaction."),
    ])
    assert len(a) == 1
    assert sorted(a[0].references) == ["d1", "d2"]
    assert {s.lang for s in a[0].support} == {"en", "zh"}


def test_fan_in_keeps_disagreement_as_separate_answers():
    a = fan_in([
        CommittedClaim("Headache is a reaction.", "h", "d1#p1", "d1", "en",
                       answer="headache", sentence="Headache is a reaction."),
        CommittedClaim("Chest pain is a reaction.", "c", "d3#p7", "d3", "es",
                       answer="chest pain", sentence="Chest pain is a reaction."),
    ])
    assert len(a) == 2, "corpus disagreement must surface as a multi-valued answer, never merge"


def test_cluster_key_ignores_case_and_spacing_but_not_content():
    assert cluster_key("Headache  is a REACTION.") == cluster_key("headache is a reaction.")
    assert cluster_key("headache") != cluster_key("chest pain")


def test_references_carry_a_placeholder_score_for_the_post_hoc_citation_scorer():
    """The keys are the RAG loop's output and the scores are the citation scorer's. 0.0 means not
    scored yet; it is not a confidence anything upstream computed."""
    a = fan_in([CommittedClaim("x", "x", "d1#p1", "d1", "en", answer="x", sentence="x")])
    assert a[0].references == {"d1": 0.0} and a[0].score == 0.0


# --------------------------------------------------------------------------- #
# The loop end to end, on the scripted trail.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("fake_search")
def test_happy_path_search_commit_answer():
    result, _llm = _run([
        _act("search", query="MSG reactions"),
        _act("submit_claim", claim="Headache is a reaction.", answer="headache",
             sentence="Headache is a reaction attributed to the MSG symptom complex.",
             passage_id="d1#p1", span="most commonly include headache"),
        _act("submit_answer"),
    ])
    assert result.status == "answered"
    assert result.claims_committed == 1
    assert result.answers[0].references == {"d1": 0.0}
    assert result.answers[0].answer == "headache"
    assert result.action_trail == ["search", "submit_claim", "submit_answer"]
    assert result.closed_by == "model"


@pytest.mark.usefixtures("fake_search")
def test_zero_claim_answer_bounces_to_gather_and_only_then_unlocks_abstain():
    """A claimless `submit_answer` is a failed attempt that returns the agent to gathering, and
    abstain appears only once F holds, never before."""
    result, llm = _run([
        _act("search", query="q1"),
        _act("submit_answer"),          # 0 claims -> failed attempt (n_search=1, F not yet held)
        _act("search", query="q2"),     # now n_search=2
        _act("abstain"),
    ])
    assert result.status == "unanswered"
    assert result.turns == 4
    assert result.answer_failures == 1
    # Turn 2 offered no abstain (only 1 search); turn 4 did (2 searches and 1 failed answer).
    assert "abstain" not in llm.menus[1]
    assert "abstain" in llm.menus[3]


@pytest.mark.usefixtures("fake_search")
def test_a_rejected_claim_does_not_terminate_the_loop_and_feeds_back():
    result, _ = _run([
        _act("search", query="q"),   # without this, both claims are rejected as `unseen_passage`
        _act("submit_claim", claim="c", answer="a", sentence="s",
             passage_id="d1#p1", span="not IN the PASSAGE"),
        _act("submit_claim", claim="Headache is a reaction.", answer="headache",
             sentence="Headache is a reaction attributed to the MSG symptom complex.",
             passage_id="d1#p1", span="most commonly include headache"),
        _act("submit_answer"),
    ])
    # The rejection triggers a bounded re-decode within the same turn, so the corrected claim is
    # the retry of turn 2 and does not cost a third turn of budget.
    assert result.claim_retries == 1
    assert result.claims_committed == 1
    assert result.claims_rejected == 0, "the claim was recovered by the retry, not dropped"
    assert result.turns == 3, "a span slip must not consume a turn of the search budget"
    assert result.status == "answered"


@pytest.mark.usefixtures("fake_search")
def test_budget_backstop_forces_a_terminal_when_max_iters_is_reached():
    """Termination by the monotonic counters alone: the model never chooses to stop."""
    cfg = Cfg(budget={"max_iters": 3, "min_searches": 1, "max_searches": 99})
    result, llm = _run([_act("search", query=f"q{i}") for i in range(3)] + [_act("abstain")], cfg)
    assert result.turns == 4, "3 free turns, then ONE forced terminal turn"
    assert llm.menus[-1] == ("abstain",), "the backstop must offer exactly one action"
    assert result.closed_by == "budget"
    assert result.status == "unanswered"


@pytest.mark.usefixtures("fake_search")
def test_budget_backstop_accepts_when_a_claim_is_committed():
    cfg = Cfg(budget={"max_iters": 2, "min_searches": 1, "max_searches": 99})
    result, llm = _run([
        _act("search", query="q"),
        _act("submit_claim", claim="Headache is a reaction.", answer="headache",
             sentence="Headache is a reaction attributed to the MSG symptom complex.",
             passage_id="d1#p1", span="most commonly include headache"),
        _act("submit_answer"),
    ], cfg)
    assert llm.menus[-1] == ("submit_answer",)
    assert result.status == "answered" and result.closed_by == "budget"


@pytest.mark.usefixtures("fake_search")
def test_the_task2_trail_is_retained_on_abstain():
    """Task-2 keeps the evidence even when the nugget is unanswered."""
    result, _ = _run([
        _act("search", query="q1"),
        _act("submit_answer"),
        _act("search", query="q2"),
        _act("abstain"),
    ])
    assert result.status == "unanswered"
    assert [r.passage_id for r in result.retrieved] == ["d1#p1", "d2#p3", "d3#p7"]
    assert result.retrieved[0].score == 0.94, "best score first"
    assert len(result.search_trail) == 2


@pytest.mark.usefixtures("fake_search")
def test_contested_flag_is_set_but_changes_no_control_flow():
    result, _ = _run([
        _act("search", query="q"),
        _act("submit_claim", claim="Headache is a reaction.",
             passage_id="d1#p1", span="most commonly include headache"),
        _act("submit_claim", claim="Sweating is a reaction.", answer="sweating",
             sentence="Sweating is also reported.",
             passage_id="d2#p3", span="也有人报告出汗和潮红"),
        _act("submit_answer"),
    ])
    assert result.contested is True and len(result.answers) == 2
    assert result.status == "answered", "conflict closes as a multi-answer, never as an abstain"


@pytest.mark.usefixtures("fake_search")
def test_the_context_only_ever_grows_and_is_never_reset():
    """The loop is one growing conversation: no new session per turn or per phase, no reset.

    Asserted on the prompt the model actually received, which is the only observable that
    distinguishes an appended context from a rebuilt one.
    """
    seen: list[str] = []

    class Recording(ScriptedLlm):
        async def generate(self, schema, prompt, seed, **kw):
            seen.append(prompt)
            return await super().generate(schema, prompt, seed, **kw)

    llm = Recording([
        _act("search", query="q"),
        _act("submit_claim", claim="c", answer="headache", sentence="s",
             passage_id="d1#p1", span="headache"),
        _act("submit_answer", answer="headache"),
    ])
    asyncio.run(run_loop({"nugget_id": "n", "question": "q?"}, Cfg(), llm=llm, ctx=FakeCtx()))
    assert len(seen) == 3
    for earlier, later in pairwise(seen):
        assert later.startswith(earlier), "each turn must EXTEND the previous context, not replace it"


@pytest.mark.usefixtures("fake_search")
def test_the_loop_instantiates_no_model_and_opens_no_index():
    """Asserted structurally: the loop's only collaborators are the two injected objects."""
    src = (
        __import__("pathlib").Path(actions_mod.__file__).parent / "loop.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("build_clients(", "bring_up(", "AsyncOpenAI(", "faiss", "torch"):
        assert forbidden not in src, f"loop.py must not reference {forbidden!r}"


@pytest.mark.usefixtures("fake_search")
def test_the_claim_retry_is_bounded_and_then_drops_the_claim():
    """A claim that keeps failing is dropped. An unbounded re-decode would be a second, invisible
    way for the loop never to terminate, reachable by a model that cannot quote."""
    bad = _act("submit_claim", claim="c", answer="a", sentence="s",
               passage_id="d1#p1", span="never IN the PASSAGE")
    result, _llm = _run(
        [
            _act("search", query="q"),          # turn 1
            bad, dict(bad), dict(bad),          # turn 2 + its two bounded re-decodes
            _act("submit_answer"),              # turn 3: 0 claims -> failed attempt, unlocks F
            _act("abstain"),                    # turn 4: F now holds
        ],
        Cfg(claim_commit_max_retries=2, budget={"min_searches": 1}),
    )
    assert result.claim_retries == 2, "exactly `claim_commit.max_retries` re-decodes, then stop"
    assert result.claims_rejected == 1 and result.claims_committed == 0
    assert result.status == "unanswered"


@pytest.mark.usefixtures("fake_search")
def test_the_short_answer_and_the_report_sentence_are_distinct_fields():
    """`common.schemas.Answer` carries two different strings: ``answer`` is the short value for
    Task 3, such as "headache", and ``sentence`` is the citeable English report sentence for
    Task 1.

    An earlier implementation took one short value from ``submit_answer`` and, whenever the
    nugget was multi-valued, filled ``answer`` with the whole claim sentence, degrading Task 3
    exactly when the nugget was most interesting. No other test noticed.
    """
    result, _llm = _run([
        _act("search", query="q"),
        _act("submit_claim", claim="Headache is a reaction.", answer="headache",
             sentence="Headache is a reaction attributed to the MSG symptom complex.",
             passage_id="d1#p1", span="most commonly include headache"),
        _act("submit_claim", claim="Sweating too.", answer="sweating",
             sentence="Sweating is also reported.",
             passage_id="d2#p3", span="也有人报告出汗和潮红"),
        _act("submit_answer"),
    ])
    assert len(result.answers) == 2, "two distinct values must stay two answers"
    for a in result.answers:
        assert a.answer != a.sentence, f"answer must be the SHORT value, got {a.answer!r}"
        assert len(a.answer) < len(a.sentence)
    assert {a.answer for a in result.answers} == {"headache", "sweating"}


@pytest.mark.usefixtures("fake_search")
def test_a_generation_timeout_ends_the_loop_with_a_terminal_not_an_exception():
    """`run_loop` must always return a LoopResult. The round loop fans k of these concurrently,
    and a bare exception out of one would cancel every sibling in the same `asyncio.gather`."""
    from ragtime.serving.llm import GenerationTimeout

    class TimingOut(ScriptedLlm):
        # A separate counter: `ScriptedLlm.generate` increments `calls` itself, so incrementing
        # it here too would double-count and the timeout would never fire on the intended turn.
        turn = 0

        async def generate(self, schema, prompt, seed, **kw):
            TimingOut.turn += 1
            if TimingOut.turn == 2:
                raise GenerationTimeout("simulated degenerate generation")
            return await super().generate(schema, prompt, seed, **kw)

    llm = TimingOut([_act("search", query="q"), _act("submit_answer")])
    result = asyncio.run(
        run_loop({"nugget_id": "n", "question": "q?"}, Cfg(), llm=llm, ctx=FakeCtx())
    )
    assert result.status == "unanswered", "no committed claim -> the honest outcome is unanswered"
    assert result.closed_by == "timeout", "recorded distinctly from a spent budget"
    assert result.retrieved, "evidence gathered before the timeout is retained for Task-2"


@pytest.mark.usefixtures("fake_search")
def test_a_timeout_on_the_RETRY_generation_also_ends_with_a_terminal():
    """The gap a per-call-site guard leaves, and the reason there is one guard instead.

    The bounded claim re-decode is a second `llm.generate()`. With the timeout guard wrapping
    only the main call, a timeout raised here propagated straight out of `run_loop`, and the
    existing timeout test could not see it because that one fires on a `search` turn. Both call
    sites now sit inside one `try`, so this holds for future call sites too.
    """
    from ragtime.serving.llm import GenerationTimeout

    class TimeoutOnRetry(ScriptedLlm):
        seen = 0

        async def generate(self, schema, prompt, seed, **kw):
            TimeoutOnRetry.seen += 1
            # 1 = search, 2 = the rejected claim, 3 = its re-decode.
            if TimeoutOnRetry.seen == 3:
                raise GenerationTimeout("simulated degenerate generation during the re-decode")
            return await super().generate(schema, prompt, seed, **kw)

    llm = TimeoutOnRetry([
        _act("search", query="q"),
        _act("submit_claim", claim="c", answer="a", sentence="s",
             passage_id="d1#p1", span="not IN the PASSAGE"),
        _act("submit_answer"),
    ])
    result = asyncio.run(
        run_loop({"nugget_id": "n", "question": "q?"}, Cfg(), llm=llm, ctx=FakeCtx())
    )
    assert result.closed_by == "timeout"
    assert result.status == "unanswered"
    assert result.retrieved, "the trail gathered before the timeout is retained"


@pytest.mark.usefixtures("fake_search")
def test_a_missing_short_value_falls_back_to_the_claim_and_is_COUNTED():
    """`answer` and `sentence` are optional in the flat union, so a schema-valid turn can omit
    them. Falling back to the claim is right, since the loop never invents a value, but doing it
    silently is not: a high fallback rate is the signal that the optionality costs Task 3."""
    from ragtime.common import Statistics

    stats = Statistics()
    llm = ScriptedLlm([
        _act("search", query="q"),
        _act("submit_claim", claim="Headache is a reaction.",   # no answer, no sentence
             passage_id="d1#p1", span="most commonly include headache"),
        _act("submit_answer"),
    ])
    result = asyncio.run(
        run_loop({"nugget_id": "n", "question": "q?"}, Cfg(), llm=llm, ctx=FakeCtx(), stats=stats)
    )
    assert result.answers[0].answer == "Headache is a reaction.", "falls back, never invents"
    emitted = {m for (m, _slices) in getattr(stats, "counters", {})} if hasattr(stats, "counters") else set()
    assert result.claims_committed == 1
    # The counter must exist and have fired; either public shape of the counter bus is accepted.
    dumped = str(getattr(stats, "counters", "")) + str(getattr(stats, "_counters", ""))
    assert "claim_value_fallback" in dumped or "claim_value_fallback" in str(emitted), (
        "the fallback must be COUNTABLE, not silent"
    )


@pytest.mark.usefixtures("fake_search")
def test_the_grounding_span_survives_fan_in_into_the_answer():
    """MAT-6: `Answer.quoted_span` is the commit evidence, and each loop's
    `rag_loop/{nugget_id}.jsonl` record carries it per answer. `CommittedClaim` always had it,
    but `fan_in` used to drop it, so the evidence died at fan-in and the driver could not write a
    compliant record."""
    result, _llm = _run([
        _act("search", query="q"),
        _act("submit_claim", claim="Headache is a reaction.", answer="headache",
             sentence="Headache is a reaction attributed to the MSG symptom complex.",
             passage_id="d1#p1", span="most commonly include headache"),
        _act("submit_answer"),
    ])
    assert result.answers[0].quoted_span == "most commonly include headache"
    # ... and it really is the span: verbatim in the passage the model was shown.
    assert result.answers[0].quoted_span in EN_TEXT


@pytest.mark.usefixtures("fake_search")
def test_claim_counters_carry_lang_and_variant_so_m11_can_slice_them():
    """MAT-7: the central language-fairness diagnostic is `verbatim_commit_rate_by_language`. A
    counter emitted with only `nugget=` cannot be rolled up into it, however many are emitted, so
    the slices are asserted rather than assumed.

    `lang` is the cited passage's native language, zh here from the CJK fixture; `variant` is the
    rendering the loop read. They are different axes and both are needed, because the diagnostic
    asks how the commit rate varies by source language within one arm.
    """
    from ragtime.common import Statistics

    stats = Statistics()
    llm = ScriptedLlm([
        _act("search", query="q"),
        _act("submit_claim", claim="Sweating is reported.", answer="sweating", sentence="S.",
             passage_id="d2#p3", span="也有人报告出汗和潮红"),
        _act("submit_answer"),
    ])
    asyncio.run(run_loop({"nugget_id": "n", "question": "q?"}, Cfg(), llm=llm, ctx=FakeCtx(),
                         stats=stats, passage_lang="omt"))
    dumped = str(getattr(stats, "counters", "")) + str(getattr(stats, "_counters", ""))
    assert "claims_committed" in dumped
    assert "'lang', 'zh'" in dumped or "zh" in dumped, "the passage's NATIVE lang must be sliced"
    assert "omt" in dumped, "the READ rendering must be sliced"


# --------------------------------------------------------------------------- #
# A dropped claim must not end the loop.
#
# `closed_by: malformed_action` terminated 48 of 316 loop executions across two fleets, three of
# them after they had already committed claims, and it cost at least one weight-1.0 nugget
# outright: `2001#n7`, on the risks of Nordic walking, was killed in both rounds it was
# attempted, and the report carries no risks even though the request asks for them.
#
# The cause was not the model. `run_loop` emits its rejection histogram as
# `stats.emit(STAT_CLAIMS_REJECTED, 1.0, nugget=.., lang=.., variant=.., reason=..)`, and
# `reason` was not in `common.stats.CANONICAL_SLICE_KEYS`, so the emit raised `ValueError`, which
# `run_loop`'s outer handler catches and labels a model fault. Every loop therefore died on the
# first claim it dropped, discarding the rest of its turn budget and every search it had left.
#
# It fired unconditionally: over those 316 records, `claims_rejected >= 1` and
# `closed_by == malformed_action` picked out the same 48 records in both directions, and
# `claims_rejected` never exceeded 1, while records from the previous build reach
# `claims_rejected: 7` and close normally.
#
# The defect needed no GPU, no index and no scale to reproduce, which is why these tests live in
# the small tier.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("fake_search")
def test_a_dropped_claim_does_not_terminate_the_loop():
    """Dropping a claim costs a claim, not the loop.

    The script drops one claim, through `max_retries` re-decodes that are all non-verbatim, then
    commits a good one and answers. With the defect present the loop dies at the drop with
    `closed_by=malformed_action` and `claims_committed=0`, leaving the last script entries
    unreached.
    """
    bad = {"claim": "Not in any passage.", "answer": "x", "sentence": "X.",
           "passage_id": "d1#p1", "span": "this span is not in the passage"}
    result, llm = _run([
        _act("search", query="q"),
        _act("submit_claim", **bad),   # first attempt -> Rejection
        _act("submit_claim", **bad),   # re-decode 1   -> Rejection
        _act("submit_claim", **bad),   # re-decode 2   -> Rejection, claim DROPPED
        _act("submit_claim", claim="Headache is a reaction.", answer="headache",
             sentence="Headache is a reaction.", passage_id="d1#p1",
             span="most commonly include headache"),
        _act("submit_answer"),
    ])
    assert result.closed_by != "malformed_action", (
        f"a dropped claim ended the loop: closed_by={result.closed_by!r} "
        f"detail={result.closed_detail!r}"
    )
    assert result.closed_by == "model"
    assert result.claims_rejected == 1
    assert result.claims_committed == 1, "the loop must survive the drop and keep gathering"
    assert result.status == "answered"
    assert llm.script == [], "the loop stopped early -- entries left unconsumed"


@pytest.mark.usefixtures("fake_search")
def test_the_rejection_histogram_survives_the_loop():
    """`rejection_reasons` is what the crashing emit existed to build, so it must be populated.

    An `empty_span` is a grammar hole, an `unseen_passage` is the model citing a doc-id instead
    of a passage handle, and a `not_verbatim` is the commit gate doing its job. The aggregate
    `claims_rejected` cannot tell them apart, and the three want opposite fixes.
    """
    unseen = {"claim": "c", "answer": "a", "sentence": "S.",
              "passage_id": "d9#p9", "span": "whatever"}
    result, _llm = _run([
        _act("search", query="q"),
        _act("search", query="q2"),     # min_searches=2, so the effort floor needs both
        _act("submit_claim", **unseen),
        _act("submit_claim", **unseen),
        _act("submit_claim", **unseen),
        _act("submit_answer"),          # zero committed -> bounces (n_answer_fail=1)
        _act("abstain"),                # ...which, with 2 searches, unlocks abstain
    ])
    assert result.rejection_reasons, "the histogram was computed and then lost"
    assert sum(result.rejection_reasons.values()) == result.claims_rejected
    assert "unseen_passage" in result.rejection_reasons


def test_the_claims_rejected_counter_is_emittable_at_all():
    """The emit that killed 48 loops, isolated from the loop.

    `common.stats.Statistics.emit` correctly rejects an unknown slice key with `ValueError`. What
    was wrong is that `reason` was missing from the vocabulary while the caller emitted it on
    every dropped claim, inside a `try` that reads a `ValueError` as a model fault. A counter
    must never be able to end real work, so the emit is pinned here on its own, with no loop, no
    script and no mock in the way.
    """
    from ragtime.common import Statistics
    from ragtime.pipeline.rag_loop import stats as S

    Statistics().emit(
        S.STAT_CLAIMS_REJECTED, 1.0,
        nugget="2001#n7", lang="ru", variant="original", reason="not_verbatim",
    )


@pytest.mark.usefixtures("fake_search")
def test_an_aborted_loop_records_WHICH_exception_and_from_WHERE():
    """`closed_by: malformed_action` with nothing else is not a diagnosis.

    The handler catches three unrelated failure classes: a `ValueError` from the search tool, a
    `GuidedJsonError` from an exhausted schema retry, and a `ValueError` from our own counter bus.
    `closed_by` alone names none of them, so `closed_detail` must carry both the exception type
    and the raise site; the type on its own cannot separate the three.
    """
    class Boom:
        async def generate(self, schema: Any, prompt: str, seed: int, **kw: Any) -> dict[str, Any]:
            raise ValueError("synthetic fault")

    result = asyncio.run(
        run_loop({"nugget_id": "n", "question": "q?"}, Cfg(), llm=Boom(), ctx=FakeCtx())
    )
    assert result.closed_by == "malformed_action"
    assert result.closed_detail.startswith("ValueError at "), result.closed_detail
    assert "synthetic fault" in result.closed_detail
    assert ".py:" in result.closed_detail, "the raise SITE is the load-bearing half"
