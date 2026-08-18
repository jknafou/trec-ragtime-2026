"""Small tier for select and serialize: the pure projections on real record classes.

Every fixture is built by :mod:`tests.pipeline.serialize_fixtures` from the production writers and
the real frozen records; the config is the real ``config/e2e-omt.yml``. Nothing here loads a model,
opens an index or reaches a network: the whole tier is arithmetic and JSON.

These are the fast twins of the full tier: every claim the full tier makes about output shape, id
syntax and the budget rule is re-made here from the same builders, so a defect reproduces in
seconds rather than behind a validator subprocess.

The estimate beside each test is stated so ``--durations=0`` can falsify it.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from ragtime.common import Retrieved, Statistics, nfkc_len
from ragtime.common.layout import Layout
from ragtime.pipeline.select_serialize import (
    budget,
    dedup,
    load,
    task3,
)
from ragtime.pipeline.select_serialize.knobs import SerializeConfigMissing, SerializeKnobs
from ragtime.pipeline.select_serialize.submission import trec_selfcheck
from ragtime.pipeline.select_serialize.submission.envelope import EnvelopeError, envelopes
from ragtime.pipeline.select_serialize.submission.validate import (
    validate as run_validator,
)
from tests.pipeline import serialize_fixtures as fx

pytestmark = pytest.mark.small

DOC_A = "eng-docs/0007421"
DOC_B = "rus-docs/0330915"
DOC_C = "spa-docs/0451820"


@pytest.fixture
def cfg():
    return fx.cfg_with_serialize()


@pytest.fixture
def knobs(cfg):
    return SerializeKnobs.from_cfg(cfg)


def _bank_two_nuggets():
    """Two answered nuggets. The top one carries two answers, which is what separates the
    coverage-first selector from a plain importance greedy at a two-sentence budget."""
    return (
        fx.nugget(
            "2000#n0", "Which countries regulate e-cigarettes?", weight=0.9, vital=True,
            answers=(
                fx.answer("A", "AAAA", refs={DOC_A: 0.9}, span="sa"),
                fx.answer("B", "BBBB", refs={DOC_B: 0.8}, span="sb"),
            ),
            retrieved=(Retrieved(f"{DOC_A}#p0", -0.1), Retrieved(f"{DOC_A}#p1", -0.2)),
        ),
        fx.nugget(
            "2000#n1", "What do those laws contain?", weight=0.4,
            answers=(fx.answer("C", "CCCC", refs={DOC_C: 0.7}, span="sc"),),
            retrieved=(Retrieved(f"{DOC_C}#p0", -0.3), Retrieved(f"{DOC_A}#p0", -1.5)),
        ),
    )


# --------------------------------------------------------------------------- #
# Budget: the NFKC joiner rule  (< 5 s)
# --------------------------------------------------------------------------- #
def test_t1_budget_counts_the_join_space():
    """``joined_len`` is ``nfkc_len(" ".join(...))``, the joiner costs exactly +1 per gap, and
    ``<=`` is inclusive at the boundary: the three properties read out of the validator."""
    texts = ["0123456789", "abcdefghij"]
    assert budget.joined_len(texts) == nfkc_len(" ".join(texts)) == 21
    # n sentences cost n-1 extra characters over the bare sum.
    assert budget.joined_len(texts) - sum(nfkc_len(t) for t in texts) == len(texts) - 1

    # The incremental accounting agrees with the whole-report measure.
    used = budget.fits(0, texts[0], 0, 21)
    assert used == 10
    assert budget.fits(used, texts[1], 1, 21) == 11  # 10 for the text + 1 for the joiner
    # `<=` is inclusive: 21 fits, 20 does not. This is the pair the validator was demonstrated
    # on (`--limit 20` -> `[Error] 21 > 20`; `--limit 21` -> rc 0).
    assert budget.fits(used, texts[1], 1, 20) is None

    # NFKC, not NFC: a ligature is two characters to the scorer and one to NFC.
    assert nfkc_len("ﬁ") == 2
    assert budget.joined_len(["ﬁ" * 3000]) == 6000


def test_t1_budget_trim_re_measures_the_whole_report():
    """``trim_to_limit`` drops from the tail until the real measure fits. NFKC is not additive
    across a join, so the incremental accounting is not the last word."""
    texts = ["aaaa", "bbbb", "cccc"]
    assert budget.trim_to_limit(texts, 14) == texts
    assert budget.trim_to_limit(texts, 13) == ["aaaa", "bbbb"]
    assert budget.trim_to_limit(texts, 1) == []


# --------------------------------------------------------------------------- #
# Task 3 (< 5 s each)
# --------------------------------------------------------------------------- #
def test_t3_caps_by_weight_then_by_score_with_deterministic_ties(knobs):
    """Every cut is by a score this stage owns, with a deterministic tie on lower ``nugget_id``."""
    tied = tuple(fx.nugget(f"2000#n{i}", f"q{i}", weight=0.5) for i in (3, 1, 2))
    stats = Statistics()
    kept = task3.apply_caps(tied, dataclasses.replace(knobs, k_t3=2), stats=stats)
    assert [n.nugget_id for n in kept] == ["2000#n1", "2000#n2"]
    assert stats.total(task3.T3_NUGGETS_DROPPED) == 1
    assert stats.total(task3.T3_DROPPED_TAIL_MEAN_WEIGHT) == 0.5


# --------------------------------------------------------------------------- #
# Dedup (< 10 s)
# --------------------------------------------------------------------------- #
def test_and_nugget_is_exempt_from_answer_dedup():
    """``and`` answers are required components and are all preserved; an ``OR`` duplicate merges
    only when both gates agree, and the confirm gate sees a canonicalised pair order."""
    answers = (
        fx.answer("headache pain", "S1", refs={DOC_A: 0.4}, span="x"),
        fx.answer("headache pain", "S2", refs={DOC_B: 0.9}, span="y"),
    )
    and_nugget = fx.nugget("2000#n0", "q", aggregator_type="and", answers=answers)
    or_nugget = fx.nugget("2000#n1", "q", aggregator_type="OR", answers=answers)

    kept_and = asyncio.run(
        dedup.merge_answers(
            and_nugget, 0.5, embed=fx.stub_embed, confirm=fx.stub_confirm_always
        )
    )
    assert len(kept_and.answers) == 2

    seen: list[tuple[str, str]] = []
    kept_or = asyncio.run(
        dedup.merge_answers(
            or_nugget, 0.5, embed=fx.stub_embed, confirm=fx.recording_confirm(seen),
        )
    )
    assert len(kept_or.answers) == 1
    # The union of the doc-ids, keeping the highest score per doc-id.
    assert kept_or.answers[0].references == {DOC_A: 0.4, DOC_B: 0.9}
    assert seen == [tuple(sorted(("headache pain", "headache pain")))]

    # Either gate may veto: with the confirm gate saying no, both answers survive.
    vetoed = asyncio.run(
        dedup.merge_answers(or_nugget, 0.5, embed=fx.stub_embed, confirm=fx.stub_confirm_never)
    )
    assert len(vetoed.answers) == 2


def test_question_dedup_merges_a_component_keeping_the_lower_nugget_id():
    """The merge keeps the highest ``weight``, the union of ``answers`` and ``retrieved``, and
    the lower id as representative. A high merge rate here indicates a defect upstream, which is
    why the counter exists."""
    bank = (
        fx.nugget("2000#n1", "which countries regulate e cigarettes", weight=0.4,
                  answers=(fx.answer("a", "A"),), retrieved=(Retrieved(f"{DOC_A}#p0", -1.0),)),
        fx.nugget("2000#n0", "which countries regulate e cigarettes", weight=0.9,
                  answers=(fx.answer("b", "B"),), retrieved=(Retrieved(f"{DOC_B}#p0", -2.0),)),
        fx.nugget("2000#n2", "entirely different subject matter here", weight=0.5),
    )
    stats = Statistics()
    merged = dedup.merge_questions_embedding(bank, 0.9, embed=fx.stub_embed, stats=stats)
    assert [n.nugget_id for n in merged] == ["2000#n0", "2000#n2"]
    kept = merged[0]
    assert kept.weight == 0.9
    assert {a.answer for a in kept.answers} == {"a", "b"}
    assert {r.passage_id for r in kept.retrieved} == {f"{DOC_A}#p0", f"{DOC_B}#p0"}
    assert stats.total(dedup.SERIALIZE_QUESTIONS_MERGED) == 1


# --------------------------------------------------------------------------- #
# Input assembly (< 5 s each)
# --------------------------------------------------------------------------- #
def test_assemble_bank_enriches_from_loop_records_and_ignores_later_rounds(tmp_path, cfg):
    """The persisted bank has no ``references``, because the audit strips them. The enrichment
    restores them from the loop records, and a loop round beyond the last complete bank is
    ignored."""
    bank = _bank_two_nuggets()
    root = fx.write_cell(tmp_path, topic_id="2000", bank=bank, cfg=cfg, rounds=2,
                         extra_loop_round=True)
    layout = Layout(run_dir=root / "topics" / "2000", outputs=cfg.outputs)
    assert load.last_complete_round(layout) == 1

    assembled = load.assemble_bank(layout, topic_id="2000")
    by_id = {n.nugget_id: n for n in assembled}
    assert by_id["2000#n0"].answers[0].references == {DOC_A: 0.9}
    # `answer_score: max_reference`, derived from an already-computed number, never invented.
    assert by_id["2000#n0"].answers[0].score == 0.9
    # The round-2 loop record's answer never appears.
    texts = {a.sentence for n in assembled for a in n.answers}
    assert "A sentence from a round with no complete bank." not in texts


def test_bank_answer_without_a_loop_twin_is_a_hard_error(tmp_path, cfg):
    """A missing twin means an empty citation dict on a real sentence, so it refuses to emit."""
    bank = _bank_two_nuggets()
    root = fx.write_cell(tmp_path, topic_id="2000", bank=bank, cfg=cfg)
    (root / "topics" / "2000" / "rag_loop" / "2000#n0.jsonl").write_text("", encoding="utf-8")
    layout = Layout(run_dir=root / "topics" / "2000", outputs=cfg.outputs)
    with pytest.raises(load.MissingLoopEvidence):
        load.assemble_bank(layout, topic_id="2000")


def test_citation_scores_absent_is_none_not_empty(tmp_path, cfg):
    """``None`` means the citation scorer never ran and ``{}`` means it ran and scored nothing. The
    two have opposite consequences and are never collapsed into each other."""
    root = fx.write_cell(tmp_path, topic_id="2000", bank=_bank_two_nuggets(), cfg=cfg)
    layout = Layout(run_dir=root / "topics" / "2000", outputs=cfg.outputs)
    assert load.citation_scores(layout) is None

    key = load.answer_key(_bank_two_nuggets()[0].answers[0])
    root2 = fx.write_cell(
        tmp_path / "scored", topic_id="2000", bank=_bank_two_nuggets(), cfg=cfg,
        scores={("2000#n0", key, DOC_A): 0.42},
    )
    layout2 = Layout(run_dir=root2 / "topics" / "2000", outputs=cfg.outputs)
    scores = load.citation_scores(layout2)
    assert scores == {("2000#n0", key, DOC_A): 0.42}
    assembled = load.assemble_bank(layout2, topic_id="2000", scores=scores)
    assert assembled[0].answers[0].references[DOC_A] == 0.42


# --------------------------------------------------------------------------- #
# Envelope + knobs (< 5 s each)
# --------------------------------------------------------------------------- #
def test_run_id_over_25_chars_is_refused_on_every_track(cfg):
    """The nuggets schema enforces ``maxLength: 25`` and the report schema does not, so the same
    26-character id passes as a report and is rejected as nuggets. It is refused here instead,
    before anything is written."""
    for env in envelopes(cfg):
        assert len(env.run_id) <= 25
    long_outputs = tuple(
        {**dict(o), "run_id": "x" * 26} for o in cfg.outputs
    )
    bad = dataclasses.replace(cfg, outputs=long_outputs)
    with pytest.raises(EnvelopeError):
        envelopes(bad)


def test_assessment_priority_comes_from_the_declared_path(cfg):
    """``submissions/<track>/run_N.<ext>`` carries the priority. A separate priority field in the
    config would be free to disagree with the file name the organisers actually see."""
    for env in envelopes(cfg):
        assert env.priority == int(env.path.rsplit("run_", 1)[1].split(".")[0])


def test_serialize_knobs_refuse_to_default_a_tuned_value():
    """A tuned knob has no code default: its value is part of the experiment's record."""
    from types import SimpleNamespace

    with pytest.raises(SerializeConfigMissing, match="no 'serialize' block"):
        SerializeKnobs.from_cfg(SimpleNamespace(blocks={}))
    with pytest.raises(SerializeConfigMissing, match="tuned knob"):
        SerializeKnobs.from_cfg(SimpleNamespace(blocks={"serialize": {"k_t3": 5}}))
    with pytest.raises(SerializeConfigMissing, match="unknown serialize knob"):
        SerializeKnobs.from_cfg(
            SimpleNamespace(blocks={"serialize": {**fx.SERIALIZE_BLOCK, "nope": 1}})
        )
    # The specified defaults hold when the block does not restate them.
    k = SerializeKnobs.from_cfg(SimpleNamespace(blocks={"serialize": fx.SERIALIZE_BLOCK}))
    assert (k.rrf_k, k.top_docs, k.dedup_q_cutoff) == (60, 1000, 0.90)


# --------------------------------------------------------------------------- #
# The refusals around the validator, and the Task 2 self-check  (< 5 s each)
#
# These are not twins of a slower tier: there is no serialize full tier, and no test in this
# repository invokes the organisers' validator, which is a separate checkout this repository
# does not carry. What is tested here is the code around it -- the refusal to point it at a
# Task 2 file, and every tripwire of the Task 2 self-check, which is ours and needs nothing
# external. That the validator itself accepts our Task 1 and Task 3 files is evidenced by the
# submitted runs having passed it, not by anything here.
# --------------------------------------------------------------------------- #
def test_validator_is_never_pointed_at_a_trec_run(tmp_path):
    """There is no Task 2 validator -- ``--format`` accepts only ``{report, nuggets}`` -- so
    asking for one is refused by name rather than quietly reported as a pass."""
    with pytest.raises(ValueError, match="no Task-2 validator"):
        run_validator(tmp_path / "t2.txt", "trec", topics_path=tmp_path / "topics.jsonl")


def test_trec_selfcheck_can_go_red_on_every_rule_it_claims(tmp_path):
    """Each tripwire is shown failing, because a check only ever seen green is not evidence."""
    good = "2000 Q0 eng-docs/0000001 1 2.0 r\n2000 Q0 eng-docs/0000002 2 1.0 r\n"
    (tmp_path / "ok.txt").write_text(good, encoding="utf-8")
    assert trec_selfcheck.check_run_file(tmp_path / "ok.txt", top_docs=10) == 2

    cases = {
        "columns": "2000 Q0 eng-docs/0000001 1 2.0\n",
        "q0": "2000 XX eng-docs/0000001 1 2.0 r\n",
        "increasing": "2000 Q0 eng-docs/0000001 1 1.0 r\n2000 Q0 eng-docs/0000002 2 2.0 r\n",
        "contiguous": (
            "2000 Q0 eng-docs/0000001 1 2.0 r\n2001 Q0 eng-docs/0000002 1 2.0 r\n"
            "2000 Q0 eng-docs/0000003 2 1.0 r\n"
        ),
        "tie_order": "2000 Q0 eng-docs/0000009 1 2.0 r\n2000 Q0 eng-docs/0000001 2 2.0 r\n",
        "passage_id": "2000 Q0 eng-docs/0000001#p0 1 2.0 r\n",
        "rank": "2000 Q0 eng-docs/0000001 7 2.0 r\n",
        "blank": "2000 Q0 eng-docs/0000001 1 2.0 r\n\n",
    }
    for name, text in cases.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(trec_selfcheck.TrecFormatError):
            trec_selfcheck.check_run_file(path, top_docs=10)
    over = "".join(f"2000 Q0 eng-docs/{i:07d} {i + 1} {10 - i}.0 r\n" for i in range(3))
    (tmp_path / "over.txt").write_text(over, encoding="utf-8")
    with pytest.raises(trec_selfcheck.TrecFormatError):
        trec_selfcheck.check_run_file(tmp_path / "over.txt", top_docs=2)


# --------------------------------------------------------------------------- #
# Task 2 has no official validator, so its own check must be able to refuse  (< 10 s each)
# --------------------------------------------------------------------------- #
def test_t2_selfcheck_refuses_a_ZERO_ROW_run_file(tmp_path):
    """A zero-row TREC run is refused, not returned as ``0``.

    The per-line loop body never executes over an empty file, so every rule in this module is
    vacuously satisfied and ``check_run_file`` returns ``0`` without raising. That is the Task 2
    analogue of an empty run file exiting 0, which the Task 1 and Task 3 side closes by making
    ``--topics`` mandatory; Task 2 has no validator to make mandatory.
    """
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    with pytest.raises(trec_selfcheck.TrecFormatError, match="0 rows"):
        trec_selfcheck.check_run_file(tmp_path / "empty.txt", top_docs=10)


def test_t2_selfcheck_refuses_a_run_missing_a_declared_topic(tmp_path):
    """``topics=`` makes the check capable of failing on absent content. It is the same set
    difference the official validator takes for Task 1 and Task 3, and one-directional in the
    same way."""
    good = "2000 Q0 eng-docs/0000001 1 2.0 r\n2000 Q0 eng-docs/0000002 2 1.0 r\n"
    (tmp_path / "one.txt").write_text(good, encoding="utf-8")
    # Without `topics=`, the file's own rules are all it can check.
    assert trec_selfcheck.check_run_file(tmp_path / "one.txt", top_docs=10) == 2
    # With the declared set, a missing topic is a refusal naming it.
    with pytest.raises(trec_selfcheck.TrecFormatError, match="2001"):
        trec_selfcheck.check_run_file(
            tmp_path / "one.txt", top_docs=10, topics=["2000", "2001"]
        )
    # Extra topics outside the declared set are accepted: the check is one-directional.
    assert trec_selfcheck.check_run_file(tmp_path / "one.txt", top_docs=10, topics=["2000"]) == 2
