"""Fast tests for the coverage audit and the loop of rounds. No model, no index.

Budget: under 30 s for the whole file. Everything that is size-independent is answered here in
seconds, so the one expensive test only has to answer what needs a live service.

Every assertion is on observable state -- a bank's statuses, a novelty curve, a counter, a file
on disk -- never on "no exception was raised". Each tripwire is paired with the negative case
that makes it able to fail: `audit` marking a nugget answered is worthless as a test unless the
same fixture can also fail to mark one.

The four properties this file exists for, all of which fail invisibly in production:

* the audit's over-claim guard: a `full` label with no committed answer behind it closes a
  nugget forever, and over-claimed coverage is the dominant risk in this stage;
* saturation firing on the configured streak: a loop that never stops burns k loops per round
  up to `R_max`, and one that stops at the first quiet round is the premature stop the anti-flap
  floor exists to prevent;
* a raising loop in a later round not cancelling its siblings, since rounds are where the
  gathering wrap is re-entered;
* the artifact tree reading correctly as the checkpoint: a half-run topic that reads as
  complete is silently a dropped topic.
"""

from __future__ import annotations

import asyncio
import json
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Answer, Layout, Nugget, Retrieved, Statistics, Support
from ragtime.pipeline import driver, records, round_loop
from ragtime.pipeline.decompose import bank as bank_ops
from ragtime.pipeline.decompose import coverage_audit, grow_nuggets
from ragtime.pipeline.decompose.grow_nuggets import GAP_EVIDENCE_DRIVEN, GAP_REQUEST_DRIVEN
from tests.pipeline.conftest import audit_response, make_bundle, nuggets_response

pytestmark = pytest.mark.small


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bank(*questions: str, topic: str = "2061") -> tuple[Nugget, ...]:
    return bank_ops.mint(topic, list(questions), origin_round=0)


def _result(nugget_id: str, *, answers=(), retrieved=(), status="answered") -> Any:
    """A `LoopResult`-shaped namespace: `evidence_from_results` is duck-typed."""
    return types.SimpleNamespace(
        nugget_id=nugget_id,
        question="q",
        status=status,
        answers=tuple(answers),
        retrieved=tuple(Retrieved(passage_id=p, score=s) for p, s in retrieved),
        search_trail=[],
        turns=1,
        searches=1,
        claims_committed=len(answers),
        claims_rejected=0,
        contested=False,
        closed_by="model",
    )


def _answer(value: str, *, span: str = "", passage: str | None = None) -> Answer:
    return Answer(
        answer=value,
        sentence=f"{value}.",
        quoted_span=span or value,
        support=(Support(passage_id=passage, lang="en"),) if passage else (),
    )


def _evidence(results) -> coverage_audit.Evidence:
    return coverage_audit.evidence_from_results(results)


# --------------------------------------------------------------------------- #
# 1. mark-answered: the audit closes what is answered, and refuses what is not.
# --------------------------------------------------------------------------- #
def test_apply_delta_closes_a_full_and_answered_nugget() -> None:
    bank = _bank("What law applies?", "Who enforces it?")
    nid = bank[0].nugget_id
    evidence = _evidence([_result(nid, answers=[_answer("the Food Act")])])
    delta = coverage_audit.AuditDelta(
        rationale="", coverage={nid: coverage_audit.COVERAGE_FULL}, gaps=(), prune=()
    )

    out = coverage_audit.apply_delta(coverage_audit.attach_evidence(bank, evidence), delta, evidence)

    status = {n.nugget_id: n.status for n in out}
    assert status[nid] == bank_ops.STATUS_ANSWERED
    assert status[bank[1].nugget_id] == bank_ops.STATUS_UNANSWERED, (
        "an unlabelled nugget must stay OPEN: the audit closes only what it judged"
    )


@pytest.mark.parametrize(
    "label", [coverage_audit.COVERAGE_PARTIAL, coverage_audit.COVERAGE_NONE]
)
def test_only_full_closes_a_nugget(label: str) -> None:
    """`partial` and `none` keep a nugget open, so it keeps driving retrieval."""
    bank = _bank("What law applies?")
    nid = bank[0].nugget_id
    evidence = _evidence([_result(nid, answers=[_answer("partly")])])
    delta = coverage_audit.AuditDelta(rationale="", coverage={nid: label}, gaps=(), prune=())

    out = coverage_audit.apply_delta(coverage_audit.attach_evidence(bank, evidence), delta, evidence)
    assert out[0].status == bank_ops.STATUS_UNANSWERED


def test_a_full_label_with_no_committed_answer_is_refused_and_counted() -> None:
    """The over-claim guard. Closing a nugget is permanent, and the judge does over-claim."""
    bank = _bank("What law applies?")
    nid = bank[0].nugget_id
    evidence = _evidence([_result(nid, answers=[], status="abstained")])
    delta = coverage_audit.AuditDelta(
        rationale="", coverage={nid: coverage_audit.COVERAGE_FULL}, gaps=(), prune=()
    )
    stats = Statistics()

    out = coverage_audit.apply_delta(bank, delta, evidence, stats=stats, round=1)

    assert out[0].status == bank_ops.STATUS_UNANSWERED, (
        "a nugget with zero committed answers was closed on the model's say-so"
    )
    assert stats.value(coverage_audit.AUDIT_OVERCLAIM_BLOCKED, round=1, nugget=nid) == 1.0


# --------------------------------------------------------------------------- #
# 2. prune: off-topic goes, answered never goes.
# --------------------------------------------------------------------------- #
def test_prune_flags_an_unanswered_nugget() -> None:
    bank = _bank("What law applies?", "What is the capital of France?")
    off_topic = bank[1].nugget_id
    evidence = _evidence([])
    delta = coverage_audit.AuditDelta(rationale="", coverage={}, gaps=(), prune=(off_topic,))
    stats = Statistics()

    out = coverage_audit.apply_delta(bank, delta, evidence, stats=stats, round=1)

    assert {n.nugget_id: n.status for n in out}[off_topic] == bank_ops.STATUS_PRUNED
    assert stats.value(coverage_audit.AUDIT_PRUNED, round=1) == 1.0
    assert len(out) == len(bank), "a pruned nugget stays in the bank, flagged: never deleted"


def test_pruning_a_nugget_that_has_answers_is_refused_and_counted() -> None:
    """Answered is not deleted: that nugget carries Task 1 sentences and Task 3 answers."""
    bank = _bank("What law applies?")
    nid = bank[0].nugget_id
    evidence = _evidence([_result(nid, answers=[_answer("the Food Act")])])
    delta = coverage_audit.AuditDelta(rationale="", coverage={}, gaps=(), prune=(nid,))
    stats = Statistics()

    out = coverage_audit.apply_delta(
        coverage_audit.attach_evidence(bank, evidence), delta, evidence, stats=stats, round=1
    )

    assert out[0].status != bank_ops.STATUS_PRUNED
    assert stats.value(coverage_audit.AUDIT_PRUNE_REFUSED, round=1, nugget=nid) == 1.0


# --------------------------------------------------------------------------- #
# 3. The audit call: labels, gaps, prunes, and what it throws away.
# --------------------------------------------------------------------------- #
def test_audit_emits_gaps_and_labels_from_one_call(decompose_cfg) -> None:
    bank = _bank("What law applies?")
    nid = bank[0].nugget_id
    evidence = _evidence(
        [_result(nid, answers=[_answer("the Food Act", passage="d1#p0")], retrieved=[("d1#p0", 9.0)])]
    )
    clients = make_bundle(
        audit=[
            audit_response(
                coverage=[(nid, "partial")],
                add=[("Which agency enforces it?", "d1#p0", 0.8), ("When did it start?", None, 0.4)],
                prune=[],
            )
        ]
    )

    delta = asyncio.run(
        coverage_audit.audit(
            bank, evidence, problem_statement="Report on food law.", background="",
            cfg=decompose_cfg, clients=clients, seed=0, round=1,
        )
    )

    assert len(clients.llm.calls_for("coverage_audit")) == 1, "the audit is ONE call, not one per nugget"
    assert delta.coverage == {nid: "partial"}
    assert [g.question for g in delta.gaps] == [
        "Which agency enforces it?",
        "When did it start?",
    ]
    assert delta.gaps[0].evidence_driven and not delta.gaps[1].evidence_driven


def test_audit_drops_a_label_for_a_nugget_that_is_not_open(decompose_cfg) -> None:
    """A settled nugget cannot be re-opened or re-closed by a later round's label."""
    bank = bank_ops.close(_bank("What law applies?", "Who enforces it?"), [])
    bank = bank_ops.prune(bank, [bank[1].nugget_id])
    evidence = _evidence([])
    clients = make_bundle(
        audit=[audit_response(coverage=[(bank[0].nugget_id, "none"), (bank[1].nugget_id, "full")])]
    )

    delta = asyncio.run(
        coverage_audit.audit(
            bank, evidence, problem_statement="p", background="",
            cfg=decompose_cfg, clients=clients, seed=0, round=1,
        )
    )

    assert bank[0].nugget_id in delta.coverage, "the OPEN nugget's label survives"
    assert bank[1].nugget_id not in delta.coverage, "a pruned nugget's label must be discarded"


def test_audit_drops_a_hallucinated_trigger_passage_and_counts_it(decompose_cfg) -> None:
    """A fabricated provenance id reads downstream as evidence. Drop it to null and count it."""
    bank = _bank("What law applies?")
    evidence = _evidence([_result(bank[0].nugget_id, retrieved=[("d1#p0", 9.0)])])
    clients = make_bundle(
        audit=[audit_response(add=[("Which agency enforces it?", "not-A-real-PASSAGE", 0.5)])]
    )
    stats = Statistics()

    delta = asyncio.run(
        coverage_audit.audit(
            bank, evidence, problem_statement="p", background="",
            cfg=decompose_cfg, clients=clients, seed=0, round=1, stats=stats,
        )
    )

    assert delta.gaps[0].trigger_passage_id is None
    assert not delta.gaps[0].evidence_driven
    assert stats.value(coverage_audit.AUDIT_TRIGGER_UNKNOWN, round=1) == 1.0


def test_audit_keeps_a_trigger_the_round_really_retrieved(decompose_cfg) -> None:
    """The negative twin of the test above: without it, dropping every trigger would pass."""
    bank = _bank("What law applies?")
    evidence = _evidence([_result(bank[0].nugget_id, retrieved=[("d1#p0", 9.0)])])
    clients = make_bundle(audit=[audit_response(add=[("Which agency enforces it?", "d1#p0", 0.5)])])

    delta = asyncio.run(
        coverage_audit.audit(
            bank, evidence, problem_statement="p", background="",
            cfg=decompose_cfg, clients=clients, seed=0, round=1,
        )
    )
    assert delta.gaps[0].trigger_passage_id == "d1#p0"


def test_audit_discards_a_question_carrying_the_models_deliberation(decompose_cfg) -> None:
    """The round-0 rule, unchanged in later rounds: reject and count, never repair."""
    bank = _bank("What law applies?")
    clients = make_bundle(
        audit=[audit_response(add=[("Not a question at all", None, 0.5)])]
    )
    delta = asyncio.run(
        coverage_audit.audit(
            bank, _evidence([]), problem_statement="p", background="",
            cfg=decompose_cfg, clients=clients, seed=0, round=1,
        )
    )
    assert delta.gaps == ()


# --------------------------------------------------------------------------- #
# 4. attach_evidence: the answers accumulate, they do not replace.
# --------------------------------------------------------------------------- #
def test_attach_evidence_accumulates_across_rounds_and_does_not_double_count() -> None:
    bank = _bank("What law applies?")
    nid = bank[0].nugget_id
    r1 = _evidence([_result(nid, answers=[_answer("the Food Act")], retrieved=[("d1#p0", 9.0)])])
    r2 = _evidence(
        [_result(nid, answers=[_answer("the Food Act"), _answer("the Safety Act")],
                 retrieved=[("d1#p0", 9.0), ("d2#p1", 8.0)])]
    )

    after1 = coverage_audit.attach_evidence(bank, r1)
    after2 = coverage_audit.attach_evidence(after1, r2)

    assert [a.answer for a in after2[0].answers] == ["the Food Act", "the Safety Act"]
    assert [r.passage_id for r in after2[0].retrieved] == ["d1#p0", "d2#p1"]


# --------------------------------------------------------------------------- #
# 5. grow_nuggets' round->=1 branch: the audit wired to the gate, the mint and the dedup.
# --------------------------------------------------------------------------- #
def _audit_cfg(decompose_cfg):
    """The real `decomposition` block with the dedup gate off, so no LLM confirm calls land."""
    decompose_cfg.blocks["decomposition"]["dedup"]["llm_paraphrase_merge"] = False
    return decompose_cfg


def test_round_one_admits_an_on_topic_gap_and_mints_it_with_its_trigger(decompose_cfg) -> None:
    cfg = _audit_cfg(decompose_cfg)
    bank = _bank("What law applies?")
    nid = bank[0].nugget_id
    evidence = _evidence([_result(nid, answers=[_answer("the Food Act")], retrieved=[("d1#p0", 9.0)])])
    clients = make_bundle(
        audit=[audit_response(coverage=[(nid, "full")], add=[("Which agency enforces it?", "d1#p0", 0.8)])],
        on_topic={"rationale": "a genuine facet", "on_topic": True},
    )
    stats = Statistics()

    out = asyncio.run(
        grow_nuggets(
            "Report on food law.", "", bank, evidence, 1,
            cfg=cfg, clients=clients, topic_id="2061", limit=5000, seed=0, stats=stats,
        )
    )

    minted = [n for n in out if n.origin_round == 1]
    assert [n.question for n in minted] == ["Which agency enforces it?"]
    assert minted[0].trigger_passage_id == "d1#p0"
    assert minted[0].nugget_id != nid, "a gap nugget gets a FRESH id, never a recycled one"
    assert {n.nugget_id: n.status for n in out}[nid] == bank_ops.STATUS_ANSWERED
    assert stats.value(GAP_EVIDENCE_DRIVEN, round=1) == 1.0
    assert stats.value(GAP_REQUEST_DRIVEN, round=1) == 0.0, (
        "a gap WITH a trigger passage is evidence-driven; counting it as request-driven would "
        "make the two origin rates unreadable"
    )


def test_a_gap_is_never_absorbed_into_a_pruned_or_answered_nugget(decompose_cfg) -> None:
    """B1: dedup from round 1 on runs over a bank of mixed status, and must respect that status.

    The stub embedder is one-hot on the exact question string, so a gap whose question is
    identical to a bank nugget's is a cosine-1.0 candidate and the confirm gate calls it a
    duplicate. The fixture therefore forces the merge that must not happen. Either survivor
    would be corrupt:

    * a `pruned` survivor makes the facet permanently unaskable: never fanned, never in Task 3;
    * an `answered` survivor keeps `status="answered"` while never having been asked, so a facet
      with no evidence ships as answered.

    Only redundant and off-topic nuggets leave the bank; an answered one never does.
    """
    cfg = decompose_cfg
    cfg.blocks["decomposition"]["dedup"]["llm_paraphrase_merge"] = True
    bank = _bank("What law applies?", "Who enforces it?")
    pruned_id, answered_id = bank[0].nugget_id, bank[1].nugget_id
    bank = bank_ops.prune(bank, [pruned_id])
    evidence = _evidence([_result(answered_id, answers=[_answer("the Food Act")])])
    bank = bank_ops.close(coverage_audit.attach_evidence(bank, evidence), [answered_id])

    clients = make_bundle(
        # Two gaps, each a verbatim duplicate of a non-open nugget.
        audit=[audit_response(add=[("What law applies?", None, 0.6), ("Who enforces it?", None, 0.6)])],
        dedup={"duplicate": True, "paraphrase_match": True, "entity_match": True, "reason": "same"},
    )

    out = asyncio.run(
        grow_nuggets(
            "Report on food law.", "", bank, evidence, 1,
            cfg=cfg, clients=clients, topic_id="2061", limit=5000, seed=0,
        )
    )

    by_id = {n.nugget_id: n for n in out}
    assert by_id[pruned_id].status == bank_ops.STATUS_PRUNED
    assert by_id[answered_id].status == bank_ops.STATUS_ANSWERED
    minted = [n for n in out if n.origin_round == 1]
    assert len(minted) == 2, (
        f"both gaps must survive as OPEN nuggets; got {len(minted)}: one was absorbed into a "
        "pruned/answered survivor, which deletes the facet"
    )
    assert all(n.status == bank_ops.STATUS_UNANSWERED for n in minted)
    assert round_loop.open_nuggets(out) == tuple(minted), (
        "the gaps must be fannable next round: that is the point of minting them"
    )


def test_two_open_duplicates_still_merge_at_round_one(decompose_cfg) -> None:
    """The negative twin of B1: holding out non-open nuggets must not disable dedup itself.

    Without this, never merging anything would pass the test above.
    """
    cfg = decompose_cfg
    cfg.blocks["decomposition"]["dedup"]["llm_paraphrase_merge"] = True
    bank = _bank("What law applies?")
    clients = make_bundle(
        audit=[audit_response(add=[("What law applies?", None, 0.6)])],
        dedup={"duplicate": True, "paraphrase_match": True, "entity_match": True, "reason": "same"},
    )

    out = asyncio.run(
        grow_nuggets(
            "Report on food law.", "", bank, _evidence([]), 1,
            cfg=cfg, clients=clients, topic_id="2061", limit=5000, seed=0,
        )
    )

    assert len(out) == 1, "two OPEN duplicates must still merge: dedup was not disabled"
    assert out[0].nugget_id == bank[0].nugget_id, "the OLDER open nugget survives"


def test_round_one_rejects_an_off_topic_gap_through_m08as_gate(decompose_cfg) -> None:
    """A gap faces the same `on_topic` gate as a round-0 nugget: later rounds get no own gate."""
    cfg = _audit_cfg(decompose_cfg)
    bank = _bank("What law applies?")
    clients = make_bundle(
        audit=[audit_response(add=[("What is the capital of France?", None, 0.5)])],
        on_topic={"rationale": "drifts to an adjacent subject", "on_topic": False},
    )
    stats = Statistics()

    out = asyncio.run(
        grow_nuggets(
            "Report on food law.", "", bank, _evidence([]), 1,
            cfg=cfg, clients=clients, topic_id="2061", limit=5000, seed=0, stats=stats,
        )
    )

    assert [n.question for n in out] == ["What law applies?"]
    assert stats.total("decompose.on_topic_rejected") == 1.0
    assert len(clients.llm.calls_for("on_topic_gate")) == 1


def test_round_one_refuses_to_run_without_evidence(decompose_cfg) -> None:
    """An audit round with no evidence is a second seed round. Refuse rather than degrade."""
    with pytest.raises(ValueError, match="requires passages"):
        asyncio.run(
            grow_nuggets(
                "p", "", _bank("q?"), None, 1,
                cfg=decompose_cfg, clients=make_bundle(), topic_id="2061",
                limit=5000, seed=0,
            )
        )


def test_round_one_refuses_a_text_blob_instead_of_the_loops_own_evidence(decompose_cfg) -> None:
    """Under `audit_evidence: reuse` the audit reads the loops' evidence, nothing else."""
    with pytest.raises(TypeError, match="Evidence"):
        asyncio.run(
            grow_nuggets(
                "p", "", _bank("q?"), "some passages I found somewhere", 1,
                cfg=decompose_cfg, clients=make_bundle(), topic_id="2061",
                limit=5000, seed=0,
            )
        )


def test_round_zero_still_refuses_passages(decompose_cfg) -> None:
    """The mirror assertion: neither branch may silently run the other's contract."""
    with pytest.raises(ValueError, match="passages=None"):
        asyncio.run(
            grow_nuggets(
                "p", "", (), _evidence([]), 0,
                cfg=decompose_cfg, clients=make_bundle(nuggets=[nuggets_response(["q?"])]),
                topic_id="2061", limit=5000, seed=0,
            )
        )


def test_an_audit_config_asking_for_a_searching_auditor_is_refused(decompose_cfg) -> None:
    decompose_cfg.blocks["decomposition"]["audit_evidence"] = "query"
    with pytest.raises(ValueError, match="audit_evidence"):
        asyncio.run(
            grow_nuggets(
                "p", "", _bank("q?"), _evidence([]), 1,
                cfg=decompose_cfg, clients=make_bundle(), topic_id="2061", limit=5000, seed=0,
            )
        )


# --------------------------------------------------------------------------- #
# 6. The loop of rounds: it stops, and it stops for the configured reason.
# --------------------------------------------------------------------------- #
class _ScriptedRound:
    """A `run_round` stand-in returning one canned result per open nugget, and counting rounds."""

    def __init__(self, *, answers_for=lambda nid: [_answer("v")], raise_on=()) -> None:
        self.rounds: list[tuple[int, tuple[str, ...]]] = []
        self._answers_for = answers_for
        self._raise_on = set(raise_on)

    async def __call__(
        self, bank, cfg, *, llm, ctx, ceiling, round_no, seed, stats, passage_lang, only=None
    ):
        # `only` is honoured here, not merely accepted. The sweep round fans exactly the nuggets
        # the final audit minted after the last fan, and a double that ignored the filter would
        # let a sweep-scoped test pass while the real fan re-ran the whole bank.
        targets = round_loop.open_nuggets(bank)
        ids = tuple(n.nugget_id for n in targets)
        if only is not None:
            wanted = set(only)
            ids = tuple(i for i in ids if i in wanted)
        self.rounds.append((round_no, ids))
        if round_no in self._raise_on:
            raise AssertionError(f"round {round_no} was not supposed to run")
        return round_loop.RoundResult(
            round_no=round_no,
            results=tuple(_result(nid, answers=self._answers_for(nid)) for nid in ids),
            wall_s=0.01,
            sequential_wall_s=0.01,
        )


def _loop_cfg(decompose_cfg, *, r_max=6, min_new=1, low_streak=2):
    cfg = _audit_cfg(decompose_cfg)
    cfg.blocks["decomposition"]["R_max"] = r_max
    cfg.blocks["decomposition"]["min_new"] = min_new
    cfg.blocks["decomposition"]["low_streak"] = low_streak
    cfg.blocks["retrieval"] = {"index": "original"}
    cfg.retrieval_index = "original"
    return cfg


def _run_rounds(cfg, clients, layout, *, monkeypatch, scripted) -> Any:
    monkeypatch.setattr(round_loop, "run_round", scripted)
    return asyncio.run(
        round_loop.run_rounds(
            cfg, "2061", "Report on food law.", "", 5000,
            llm=object(), ctx=object(), clients=clients, layout=layout,
            seed=0, ceiling=4, stats=Statistics(),
        )
    )


def test_the_loop_stops_on_the_configured_low_streak_and_not_before(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """Two quiet rounds are required (`low_streak: 2`); one is not enough."""
    cfg = _loop_cfg(decompose_cfg, low_streak=2)
    clients = make_bundle(
        nuggets=[nuggets_response(["What law applies?"]), nuggets_response(["What law applies?"], weights=[0.9])],
        # round 1 adds two, round 2 adds none, round 3 adds none -> the streak fires at 3.
        audit=[
            audit_response(add=[("Who enforces it?", None, 0.7), ("When did it start?", None, 0.5)]),
            audit_response(),
        ],
    )
    scripted = _ScriptedRound()
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)

    out = _run_rounds(cfg, clients, layout, monkeypatch=monkeypatch, scripted=scripted)

    assert out.novelty == (2, 0, 0), f"novelty curve was {out.novelty}"
    assert out.saturated_at == 3, "the streak must fire on the SECOND consecutive low round"
    assert not out.coverage_incomplete
    assert [r for r, _ in scripted.rounds] == [1, 2, 3], "rounds ran sequentially, 1..3"


def test_a_longer_streak_config_really_delays_the_stop(decompose_cfg, tmp_path, monkeypatch) -> None:
    """The negative twin: the same data under `low_streak: 3` must run one round more.

    Without this, a hardcoded `2` would pass the test above and nothing would notice.
    """
    cfg = _loop_cfg(decompose_cfg, low_streak=3)
    clients = make_bundle(
        nuggets=[nuggets_response(["What law applies?"]), nuggets_response(["What law applies?"], weights=[0.9])],
        audit=[
            audit_response(add=[("Who enforces it?", None, 0.7), ("When did it start?", None, 0.5)]),
            audit_response(),
        ],
    )
    scripted = _ScriptedRound()
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)

    out = _run_rounds(cfg, clients, layout, monkeypatch=monkeypatch, scripted=scripted)

    assert out.novelty == (2, 0, 0, 0)
    assert out.saturated_at == 4, (
        "the same data saturated at 3 under low_streak=2; a hardcoded streak would not move"
    )


def test_novelty_counts_new_facets_not_the_bank_size_delta(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """B2: a round that mints 3 gaps while dedup merges 3 old nuggets has novelty 3, not 0.

    A bank-size delta reads that round as `3 - 3 = 0`. Two such rounds fire
    `saturated(min_new=1, low_streak=2)` with six freshly minted, never-fanned facets in the
    bank: they ship as unanswered Task 3 questions with no Task 1 content, and the novelty curve
    -- the recorded evidence for saturation -- says nothing new was found in the rounds that
    found them.

    The stub embedder is one-hot per exact question, so seed questions are merged away by
    verbatim-duplicate proposals while genuinely new ones are added.
    """
    cfg = _loop_cfg(decompose_cfg, r_max=1, low_streak=1, min_new=1)
    cfg.blocks["decomposition"]["dedup"]["llm_paraphrase_merge"] = True
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)
    # Round 0 is seeded on disk, through the resume path, so the fixture controls the bank
    # exactly: two of the three seed nuggets are verbatim duplicates of each other, so the merge
    # at round 1 removes a pre-existing nugget rather than a freshly minted one. That is what
    # makes the two definitions differ: 2 minted - 1 merged = size delta 1, net-new 2.
    seed_bank = _bank("dup?", "dup?", "keep?")
    bank_ops.write_bank(layout.decompose_round(0), seed_bank)
    clients = make_bundle(
        nuggets=[],
        audit=[audit_response(add=[("brand new a?", None, 0.5), ("brand new b?", None, 0.5)])],
        dedup={"duplicate": True, "paraphrase_match": True, "entity_match": True, "reason": "same"},
    )

    out = _run_rounds(cfg, clients, layout, monkeypatch=monkeypatch, scripted=_ScriptedRound())

    minted = [n for n in out.bank if n.origin_round == 1]
    size_delta = len(out.bank) - len(seed_bank)
    assert len(minted) == 2 and size_delta == 1, (
        f"fixture precondition: expected 2 minted survivors and a size delta of 1; got "
        f"{len(minted)} and {size_delta}: the two definitions would not be distinguishable"
    )
    assert out.novelty[0] == 2, (
        f"novelty reported {out.novelty[0]}, which is the bank-SIZE delta ({size_delta}). It "
        "must be net-new-after-dedup: 2 new facets were found this round, and reporting 1 (or, "
        "with more merges, 0) is what fires saturation over never-fanned nuggets"
    )
    assert all(n >= 0 for n in out.novelty), "novelty can never be negative"


def test_a_resumed_novelty_curve_is_identical_to_the_uninterrupted_one(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """B2, resume half: both branches must compute novelty the same way, from the same bank."""
    cfg = _loop_cfg(decompose_cfg, low_streak=2)
    clients = make_bundle(
        nuggets=[nuggets_response(["What law applies?"]), nuggets_response(["What law applies?"], weights=[0.9])],
        audit=[
            audit_response(add=[("Who enforces it?", None, 0.7), ("When did it start?", None, 0.5)]),
            audit_response(),
        ],
    )
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)
    first = _run_rounds(cfg, clients, layout, monkeypatch=monkeypatch, scripted=_ScriptedRound())

    resumed = _run_rounds(
        cfg, make_bundle(nuggets=[], audit=[]), layout,
        monkeypatch=monkeypatch, scripted=_ScriptedRound(),
    )
    assert resumed.novelty == first.novelty
    assert resumed.saturated_at == first.saturated_at


def test_the_loop_stops_at_r_max_and_flags_coverage_incomplete(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """A never-saturating topic must terminate at `R_max` and say the bank may be incomplete."""
    cfg = _loop_cfg(decompose_cfg, r_max=3, low_streak=2)
    # Every round adds two new nuggets, so novelty never falls to the floor.
    clients = make_bundle(
        nuggets=[nuggets_response(["What law applies?"]), nuggets_response(["What law applies?"], weights=[0.9])],
        audit=[
            audit_response(add=[("Who enforces it?", None, 0.7), ("When did it start?", None, 0.5)]),
            audit_response(add=[("What penalties apply?", None, 0.6), ("Who reviews it?", None, 0.4)]),
            audit_response(add=[("Which court hears it?", None, 0.3), ("What is the fine?", None, 0.2)]),
        ],
    )
    stats = Statistics()
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)
    # `raise_on=(5,)`, not `(4,)`. Round 4 is the sweep: the bounded one-shot fan over the
    # nuggets the final audit minted after the last round's fan (this topic's audit adds two on
    # every round, including the R_max one, so there are always some). The tripwire has to stay,
    # because it is the only thing between a bug and an unbounded loop; it moves to 5, the round
    # that could only exist if the sweep had audited and re-entered the loop.
    scripted = _ScriptedRound(raise_on=(5,))
    monkeypatch.setattr(round_loop, "run_round", scripted)

    out = asyncio.run(
        round_loop.run_rounds(
            cfg, "2061", "Report on food law.", "", 5000,
            llm=object(), ctx=object(), clients=clients, layout=layout,
            seed=0, ceiling=4, stats=stats,
        )
    )

    assert len(out.novelty) == 3, "the loop must stop AT R_max, not run forever"
    assert out.saturated_at is None and out.coverage_incomplete
    assert stats.value(round_loop.COVERAGE_INCOMPLETE, seed=0) == 1.0
    assert len(out.bank) > 1, "the bank at R_max is still emitted, never silently truncated"

    # The sweep ran once, and only over the never-fanned nuggets. Both halves matter: without the
    # first, a nugget the last audit added reaches Task 3 with no answers and was never tried;
    # without the second, the sweep is a whole extra round of work on every topic.
    numbers = [r for r, _ in scripted.rounds]
    assert numbers == [1, 2, 3, 4], f"expected 3 rounds + 1 sweep, got {numbers}"
    swept = dict(scripted.rounds)[4]
    fanned_before = {nid for r, ids in scripted.rounds if r < 4 for nid in ids}
    assert swept, "the sweep fanned nothing, so it proves nothing here"
    assert not (set(swept) & fanned_before), (
        f"the sweep re-ran already-fanned nuggets {set(swept) & fanned_before}: it must fan ONLY "
        "the ones the final audit added"
    )

    # And no nugget is left unattempted: every open nugget in the emitted bank has a loop record.
    open_ids = {n.nugget_id for n in round_loop.open_nuggets(out.bank)}
    unattempted = {nid for nid in open_ids if not layout.rag_loop(nid).exists()}
    assert not unattempted, f"nuggets reached the bank with no loop record at all: {unattempted}"


def test_an_empty_open_set_is_a_novelty_observation_not_an_early_break(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """A round with nothing to fan still costs one novelty observation.

    Breaking immediately would end the loop one round before `low_streak` says, which is the
    premature stop the anti-flap floor exists to prevent.
    """
    cfg = _loop_cfg(decompose_cfg, low_streak=2)
    clients = make_bundle(
        nuggets=[nuggets_response(["What law applies?"]), nuggets_response(["What law applies?"], weights=[0.9])],
        audit=[audit_response(coverage=[("2061#n0", "full")])],
    )
    scripted = _ScriptedRound()
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)

    out = _run_rounds(cfg, clients, layout, monkeypatch=monkeypatch, scripted=scripted)

    assert scripted.rounds[1][1] == (), "round 2 had nothing open: and still ran as a round"
    assert out.novelty == (0, 0) and out.saturated_at == 2


# --------------------------------------------------------------------------- #
# 7. One raising loop must not cancel its siblings, in a later round.
# --------------------------------------------------------------------------- #
def test_a_raising_loop_in_round_two_does_not_cancel_its_siblings(monkeypatch) -> None:
    """The round-0 fan already holds this; the wrap must hold when a later round re-enters."""
    started: list[str] = []

    async def fake_run_loop(nugget, cfg, **kw):
        nid = nugget["nugget_id"]
        started.append(nid)
        await asyncio.sleep(0.01)
        if nid == "2061#n1":
            raise RuntimeError("round-2 loop exploded")
        return _result(nid, answers=[_answer("v")])

    monkeypatch.setattr(round_loop, "run_loop", fake_run_loop)
    bank = _bank("a?", "b?", "c?")
    cfg = types.SimpleNamespace(blocks={"retrieval": {"index": "original"}}, passage_lang="omt")

    res = asyncio.run(
        round_loop.run_round(bank, cfg, llm=object(), ctx=object(), ceiling=3, round_no=2)
    )

    assert sorted(started) == ["2061#n0", "2061#n1", "2061#n2"], "every sibling still ran"
    assert {r.nugget_id: r.status for r in res.results}["2061#n1"] == "error"
    assert res.round_no == 2 and res.errors and "round-2 loop exploded" in res.errors[0]


# --------------------------------------------------------------------------- #
# 8. open_nuggets speaks the bank's status vocabulary, not a literal.
# --------------------------------------------------------------------------- #
def test_open_nuggets_sees_a_real_bank() -> None:
    """A real `Nugget` is `unanswered`, not `open`; a literal `open` selects nothing at all."""
    bank = _bank("a?", "b?", "c?")
    bank = bank_ops.close(bank, [bank[1].nugget_id])
    bank = bank_ops.prune(bank, [bank[2].nugget_id])

    got = round_loop.open_nuggets(bank)

    assert [n.nugget_id for n in got] == ["2061#n0"], (
        "open_nuggets must read bank.STATUS_UNANSWERED: a literal 'open' fans ZERO real nuggets"
    )


# --------------------------------------------------------------------------- #
# 9. The artifact tree is the checkpoint.
# --------------------------------------------------------------------------- #
def test_each_round_writes_its_bank_and_a_resume_reads_it_back(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    cfg = _loop_cfg(decompose_cfg, low_streak=2)
    clients = make_bundle(
        nuggets=[nuggets_response(["What law applies?"]), nuggets_response(["What law applies?"], weights=[0.9])],
        audit=[audit_response(add=[("Who enforces it?", None, 0.7)]), audit_response()],
    )
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)
    first = _run_rounds(cfg, clients, layout, monkeypatch=monkeypatch, scripted=_ScriptedRound())

    for r in range(len(first.novelty) + 1):
        assert layout.decompose_round(r).exists(), f"round {r}'s bank was not written"

    # A second launch over the finished tree must re-run nothing: no LLM call, same bank.
    second_clients = make_bundle(nuggets=[], audit=[])
    scripted = _ScriptedRound()
    second = _run_rounds(cfg, second_clients, layout, monkeypatch=monkeypatch, scripted=scripted)

    assert scripted.rounds == [], "a completed round was re-fanned on resume"
    assert second_clients.llm.calls == [], "a completed round issued a fresh generation on resume"
    assert [n.question for n in second.bank] == [n.question for n in first.bank]
    assert second.novelty == first.novelty


def test_a_topic_that_dies_mid_loop_leaves_its_rounds_but_no_completion_marker(
    decompose_cfg, tmp_path, monkeypatch
) -> None:
    """A topic killed during round 2 must not read as done, and must keep round 1.

    The failure is driven through the real `drive`, by making the round-2 audit raise, rather
    than by writing round banks by hand. A hand-written tree exercises no code that could have
    created the completion marker, so asserting the marker is absent would pass for any
    implementation, including one that never writes a marker.

    The assertions pin the exact half-run state: round 0 and round 1 durable, round 2 absent and
    `<run>/_SUCCESS` absent, so the next launch re-runs round 2 and only round 2.
    """
    cfg = _drive_cfg(decompose_cfg, driver.KIND_E2E)
    # Round 1 must not saturate, or the loop stops before the crash it is here to survive.
    cfg.blocks["decomposition"]["R_max"] = 3
    cfg.blocks["decomposition"]["low_streak"] = 2
    clients = make_bundle(
        nuggets=[nuggets_response(["What law applies?"]), nuggets_response(["What law applies?"], weights=[0.9])],
        audit=[audit_response(add=[("Who enforces it?", None, 0.7)])],
    )
    clients.llm = _RaisesOnNthAudit(clients.llm, n=2)
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)
    monkeypatch.setattr(round_loop, "run_round", _ScriptedRound())

    with pytest.raises(RuntimeError, match="the node died"):
        driver.drive(
            cfg, {"topic_id": "2061", "problem_statement": "Report on food law.", "limit": 5000},
            "original", 0, ctx=object(), clients=clients, llm=object(), layout=layout,
        )

    assert layout.decompose_round(0).exists(), "round 0 must survive the crash"
    assert layout.decompose_round(1).exists(), "round 1 completed and must survive the crash"
    assert not layout.decompose_round(2).exists(), "round 2 died mid-audit and must NOT be durable"
    assert not layout.success().exists(), (
        "the topic died mid-loop and must not read as complete: a marker here silently drops "
        "every remaining round on the next launch"
    )


class _RaisesOnNthAudit:
    """A stub LLM that dies on the Nth `coverage_audit` call: a node preempted mid-round."""

    def __init__(self, inner, *, n: int) -> None:
        self._inner = inner
        self._n = n
        self._seen = 0
        self.model = inner.model
        self.calls = inner.calls

    async def generate(self, schema, prompt, seed, **kwargs):
        if getattr(schema, "name", "") == "coverage_audit":
            self._seen += 1
            if self._seen >= self._n:
                raise RuntimeError("the node died mid-audit")
        return await self._inner.generate(schema, prompt, seed, **kwargs)


def test_bank_round_trip_preserves_every_answer_field(tmp_path) -> None:
    """B4: `write_bank` then `read_bank` must be the identity over the whole record.

    Asserted as a whole-object equality rather than a field-by-field spot check, because a field
    list written into a test falls behind the schema exactly as `_answer_from_dict`'s did.
    `Nugget` and `Answer` are frozen dataclasses, so `==` compares every field, including the
    nested `Answer.quoted_span` that was being dropped on read.
    """
    bank = _bank("What law applies?")
    bank = (
        replace(
            bank[0],
            status=bank_ops.STATUS_ANSWERED,
            weight=0.75,
            vital=True,
            aggregator_type="and",
            origin_round=2,
            trigger_passage_id="d1#p0",
            answers=(
                Answer(
                    answer="the Food Act",
                    sentence="The Food Act applies.",
                    quoted_span="under the Food Act of 1998",
                    score=0.5,
                    references={"d1": 0.9},
                    support=(Support(passage_id="d1#p0", lang="en"),),
                ),
            ),
            retrieved=(Retrieved(passage_id="d1#p0", score=9.0),),
        ),
    )

    path = tmp_path / "round_9.jsonl"
    bank_ops.write_bank(path, bank)
    assert bank_ops.read_bank(path) == bank, (
        "a written-then-read bank must be identical; a dropped field makes a RESUMED run "
        "diverge from an uninterrupted one"
    )


def test_a_resumed_answer_is_not_duplicated_by_the_next_round(tmp_path) -> None:
    """B4's consequence: idempotency keys on `(answer, quoted_span)`.

    With `quoted_span` lost on read, a re-committed claim's key misses the seen-set key and the
    same answer is appended twice, so the auditor sees duplicated answers and a resumed run's
    labels, gaps and prunes all differ.
    """
    bank = _bank("What law applies?")
    nid = bank[0].nugget_id
    evidence = _evidence([_result(nid, answers=[_answer("the Food Act", span="under the Act")])])
    attached = coverage_audit.attach_evidence(bank, evidence)

    path = tmp_path / "round_1.jsonl"
    bank_ops.write_bank(path, attached)
    reloaded = bank_ops.read_bank(path)

    again = coverage_audit.attach_evidence(reloaded, evidence)
    assert len(again[0].answers) == 1, (
        f"the same committed answer was appended {len(again[0].answers)} times after a resume"
    )


def test_the_loop_record_keeps_one_line_per_round(tmp_path) -> None:
    """A nugget re-fanned next round must not erase its earlier round's record."""
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)
    cfg = types.SimpleNamespace(blocks={"retrieval": {"index": "original"}}, passage_lang="omt")

    records.write_loop_record(layout, _result("2061#n0"), cfg, topic_id="2061", seed=0, round_no=1)
    path = records.write_loop_record(
        layout, _result("2061#n0", answers=[_answer("v")]), cfg, topic_id="2061", seed=0, round_no=2
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [r["round"] for r in rows] == [1, 2]

    # ... and re-writing round 2 (a resume) replaces its line rather than appending a second.
    records.write_loop_record(
        layout, _result("2061#n0", answers=[_answer("v")]), cfg, topic_id="2061", seed=0, round_no=2
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["round"] for r in rows] == [1, 2]


# --------------------------------------------------------------------------- #
# 10. `drive` dispatches on run.kind and reaches the right sub-stages.
# --------------------------------------------------------------------------- #
def _drive_cfg(decompose_cfg, kind: str):
    cfg = _loop_cfg(decompose_cfg, r_max=1, low_streak=1)
    cfg.kind = kind
    cfg.run_id = "test-run"
    cfg.outputs = ()
    cfg.blocks.setdefault("rag_loop", {})["search_top_k"] = 7
    return cfg


def test_drive_dispatches_e2e_to_the_coverage_loop(decompose_cfg, tmp_path, monkeypatch) -> None:
    cfg = _drive_cfg(decompose_cfg, driver.KIND_E2E)
    clients = make_bundle(
        nuggets=[nuggets_response(["What law applies?"]), nuggets_response(["What law applies?"], weights=[0.9])],
        audit=[audit_response()],
    )
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)
    scripted = _ScriptedRound()
    monkeypatch.setattr(round_loop, "run_round", scripted)

    run_dir = driver.drive(
        cfg, {"topic_id": "2061", "problem_statement": "Report on food law.", "limit": 5000},
        "original", 0, ctx=object(), clients=clients, llm=object(), layout=layout,
    )

    assert run_dir == layout.run_dir
    assert scripted.rounds, "the e2e path must reach the RAG-loop fan"
    assert layout.decompose_round(0).exists()
    assert layout.success().exists(), "a completed topic must be marked done"


def test_drive_refuses_an_unknown_run_kind(decompose_cfg, tmp_path) -> None:
    cfg = _drive_cfg(decompose_cfg, "something_new")
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)
    with pytest.raises(ValueError, match="run.kind"):
        driver.drive(
            cfg, {"topic_id": "2061", "problem_statement": "p", "limit": 5000},
            "original", 0, ctx=object(), clients=make_bundle(), llm=object(), layout=layout,
        )


def test_drive_over_a_completed_topic_is_a_no_op(decompose_cfg, tmp_path) -> None:
    cfg = _drive_cfg(decompose_cfg, driver.KIND_E2E)
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.success().write_text("", encoding="utf-8")
    clients = make_bundle(nuggets=[], audit=[])

    run_dir = driver.drive(
        cfg, {"topic_id": "2061", "problem_statement": "p", "limit": 5000},
        "original", 0, ctx=object(), clients=clients, llm=object(), layout=layout,
    )

    assert run_dir == layout.run_dir
    assert clients.llm.calls == [], "a done topic must not be recomputed"


def test_topic_layout_gives_each_topic_its_own_rag_loop_dir(decompose_cfg) -> None:
    """Two topics of one cell must not share `rag_loop/`, or their nugget ids collide.

    The root comes from `orchestration.cli.artifact_root`, i.e. from the config record, never
    from a literal. The assertions are therefore about the shape below the root and not about
    the root itself, which differs per site.
    """
    cfg = _drive_cfg(decompose_cfg, driver.KIND_E2E)
    a = driver.topic_layout(cfg, "original", 0, "2061")
    b = driver.topic_layout(cfg, "original", 0, "2062")
    assert a.run_dir != b.run_dir
    assert a.rag_loop("2061#n0").parent != b.rag_loop("2061#n0").parent
    assert a.run_dir.parent == b.run_dir.parent, "both topics live under the SAME cell dir"
    assert a.run_dir.name == "2061" and b.run_dir.name == "2062"
    assert Path(str(a.base)) in a.run_dir.parents, "the cell hangs under the artifact root"
