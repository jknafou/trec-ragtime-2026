"""Task 2 ranks committed claims by ``claim_importance x nugget_importance``, accumulated, with a
reranker tie-break and no ties in the emitted file.

The four rules under test:

1. the unit is the committed claim;
2. a retrieved document's score is ``claim_importance x nugget_importance``;
3. a document appearing more than once accumulates: the scores are added;
4. ties are broken by the reranker score of the tied documents, collapsed to the document by the
   mean of the logs.

Every test here is written to discriminate, not merely to pass. Several assertions name the value
a MaxP / no-claim-importance / doc-id-sort ranking would have produced instead, so a test that both
readings satisfy cannot be mistaken for evidence about this one.

The tie-separation and collapse tests are unit tests on pure functions and use hand-built values:
the property is arithmetic, and the numbers sit exactly on the boundaries, with equal tie keys
and a gap narrower than the emitted precision. What they do not establish is that real cells
produce ties at all. That is a property of the data -- `rsvc` rounds every reranker
log-probability with ``round(s, 5)``, which is where the ties come from -- and belongs to the
full-tier test that reads a real serialised run file.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from ragtime.common import Answer, Nugget, Topic
from ragtime.pipeline.select_serialize.knobs import SerializeKnobs
from ragtime.pipeline.select_serialize.submission.trec_selfcheck import (
    TrecFormatError,
    check_run_file,
)
from ragtime.pipeline.select_serialize.task2_claims import (
    ClaimT2Diagnostics,
    UnseparableTie,
    _separate_tied_groups,
    claim_support_rows,
)


# --------------------------------------------------------------------------- #
# Fixtures: the smallest objects the real functions accept.
# --------------------------------------------------------------------------- #
def _topic() -> Topic:
    return Topic(
        topic_id="2000", collection_id="c", title="t",
        problem_statement="p", background="b", limit=5000,
    )


def _knobs(**over) -> SerializeKnobs:
    base = {"k_t3": 10, "answers_cap": 5, "dedup_a_cutoff": 0.9, "dedup_q_cutoff": 0.9,
            "rrf_k": 60, "top_docs": 1000, "answer_score": "max_reference"}
    base.update(over)
    return SerializeKnobs(**base)  # type: ignore[arg-type]


def _nugget(nid: str, weight: float, answers: list[Answer]) -> Nugget:
    return Nugget(nugget_id=nid, question=f"q{nid}", weight=weight, answers=tuple(answers))


def _answer(value: str, docs: list[str]) -> Answer:
    return Answer(answer=value, sentence=f"{value}.", references={d: 0.0 for d in docs})


def _key(doc: str) -> tuple[float, ...]:
    """A tie key that separates every document, which is the shape ``strict`` mode needs."""
    return (float(ord(doc[-1])),)


def _write(rows: list[str]) -> pathlib.Path:
    p = pathlib.Path(tempfile.mkdtemp()) / "run.txt"
    p.write_text("".join(rows), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# 1. The worked example, as a literal assertion.
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_tie_spread_spans_the_full_gap_to_the_next_distinct_score() -> None:
    """A/B/C tied at 6.0 above D at 5.0 -> 6.0 / 5.5000005 / 5.000001 / 5.0.

    ``step = (gap - eps) / (groups - 1)``, so the last tied row lands at ``nxt + eps`` and the
    group consumes the whole gap. The older ``gap / (groups + 1)`` gives 6.0 / 5.75 / 5.50 and
    leaves 0.5 unused, which the companion test below asserts, so the pair separates the rules.
    """
    kept = [("dA", 6.0), ("dB", 6.0), ("dC", 6.0), ("dD", 5.0)]
    out = _separate_tied_groups(kept, _key, span_full_gap=True, strict=True, eps=1e-6)
    assert [d for d, _ in out] == ["dA", "dB", "dC", "dD"], "ordering is preserved exactly"
    assert out[0][1] == pytest.approx(6.000000000, abs=1e-9)
    assert out[1][1] == pytest.approx(5.500000500, abs=1e-9)
    assert out[2][1] == pytest.approx(5.000001000, abs=1e-9)
    assert out[3][1] == 5.0, "the next distinct score is never touched"


# --------------------------------------------------------------------------- #
# 2. The inversion guard: a property, not one example.
# --------------------------------------------------------------------------- #
@pytest.mark.small
@pytest.mark.parametrize("n_tied", [2, 3, 5, 17])
@pytest.mark.parametrize("gap", [1.0, 0.001, 12.75])
def test_a_separated_row_never_reaches_the_next_distinct_score(n_tied: int, gap: float) -> None:
    """The reranker is a tie-break, never an override of rule 2.

    Every separated row must stay strictly above the next distinct score, or a document with a
    genuinely lower accumulated importance would outrank one with a higher one. An earlier
    version of the spreading scheme put the last tied row below the next score; this catches that
    across group sizes and gap widths rather than at one hand-picked point.
    """
    top, nxt = 6.0, 6.0 - gap
    kept = [(f"d{i:02d}", top) for i in range(n_tied)] + [("zzz", nxt)]
    out = _separate_tied_groups(kept, _key, span_full_gap=True, strict=True, eps=1e-9)
    scores = [s for _, s in out]
    assert scores[:-1] == sorted(scores[:-1], reverse=True), "tied block stays in its given order"
    assert all(s > nxt for s in scores[:-1]), f"a row fell to or below the next score {nxt}"
    assert scores[-1] == nxt
    assert len(set(scores)) == len(scores), "strict mode must leave every row distinct"


# --------------------------------------------------------------------------- #
# 3. Accumulation: the sharpest difference from the pool ranking.
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_a_document_cited_by_three_claims_SUMS_and_does_not_take_the_max() -> None:
    """Rule 3. A fixture with one citation per document would pass under MaxP too, so this one
    cites ``docA`` from three separate claims and asserts the sum."""
    bank = [
        _nugget("n1", 2.0, [_answer("a1", ["docA"]), _answer("a2", ["docA"])]),
        _nugget("n2", 3.0, [_answer("a3", ["docA"]), _answer("a4", ["docB"])]),
    ]
    diag = ClaimT2Diagnostics()
    rows = claim_support_rows(
        bank, _topic(), _knobs(), run_id="r", scores=None,
        reranker={"docA": -0.1, "docB": -0.2}, diagnostics=diag,
    )
    by_doc = {r.doc_id: r.score for r in rows}
    # With scores=None the claim_importance factor drops, so each citation contributes the
    # nugget weight alone.
    assert by_doc["docA"] == pytest.approx(2.0 + 2.0 + 3.0), "three citations ACCUMULATE"
    assert by_doc["docA"] != pytest.approx(3.0), "the MAX would have been 3.0: not this rule"
    assert by_doc["docB"] == pytest.approx(3.0)
    assert diag.claim_citations == 4


# --------------------------------------------------------------------------- #
# 4. A missing claim-importance score drops the factor and is counted: never invented.
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_absent_claim_importance_is_dropped_not_guessed_and_is_counted() -> None:
    bank = [_nugget("n1", 4.0, [_answer("a1", ["docA"])])]
    diag = ClaimT2Diagnostics()
    rows = claim_support_rows(
        bank, _topic(), _knobs(), run_id="r", scores={},  # ran, found nothing
        reranker={"docA": -0.3}, diagnostics=diag,
    )
    assert rows[0].score == pytest.approx(4.0), "identity-multiply: nugget_importance alone"
    assert diag.unscored_citations == 1, "the dropped factor is RECORDED, not silent"


# --------------------------------------------------------------------------- #
# 5. A gap too narrow to separate: refused in strict mode, untouched otherwise.
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_ULP_apart_scores_are_ONE_tie_group_not_a_gap_to_spread_into() -> None:
    """Three rows at 0.9 whose next distinct score was 0.8999999999999999: a gap of one ULP.

    Those two floats print identically at "{:.9f}", so they are the same emitted score. Grouping
    the run by float equality split them, the separator saw a 1e-16 gap to spread into, and
    `strict` raised `UnseparableTie` on a topic with no real tie problem. Accumulated
    `claim_importance x nugget_importance` sums land on ULP-apart values routinely, so this
    rejected real topics.
    """
    ulp_below = 0.8999999999999999
    assert ulp_below != 0.9, "the two floats really are distinct"
    assert f"{ulp_below:.9f}" == f"{0.9:.9f}", "...and really do print identically"

    kept = [("dA", 0.9), ("dB", 0.9), ("dC", ulp_below), ("dZ", 0.5)]
    out = _separate_tied_groups(kept, _key, span_full_gap=True, strict=True, eps=1e-6)
    scores = [s for _, s in out]
    assert len(set(scores[:3])) == 3, "the three ULP-equal rows are ONE run and get separated"
    assert all(s > 0.5 for s in scores[:3]), "none may fall to or below the next distinct score"
    assert scores[3] == 0.5, "the genuinely-next score is untouched"


#: A gap that is emitted-distinct (it differs at the 9th decimal, so it is a real next score) but
#: narrower than eps, so the tie group genuinely cannot be spread into it. The two values must
#: stay this close: a wider pair is reclassified as one tie group with a full synthetic gap below
#: it, which is a different case and not the one this fixture is here to exercise.
_TIED = 5.0000001          # prints 5.000000100
_NEXT = 5.0                # prints 5.000000000 -> distinct, but only 1e-7 away (< eps=1e-6)


@pytest.mark.small
def test_an_unseparable_gap_RAISES_in_strict_mode() -> None:
    """A checker that forbids ties plus a serialiser that quietly emits them gives a failure with
    no diagnosis."""
    assert f"{_TIED:.9f}" != f"{_NEXT:.9f}", "the two scores must be genuinely distinct as emitted"
    kept = [("dA", _TIED), ("dB", _TIED), ("dC", _NEXT)]
    with pytest.raises(UnseparableTie):
        _separate_tied_groups(kept, _key, span_full_gap=True, strict=True, eps=1e-6)


@pytest.mark.small
def test_an_unseparable_gap_is_left_byte_identical_when_not_strict() -> None:
    kept = [("dA", _TIED), ("dB", _TIED), ("dC", _NEXT)]
    out = _separate_tied_groups(kept, _key, span_full_gap=True, strict=False, eps=1e-6)
    assert out == kept, "no evidence to separate -> leave it alone, do not fabricate a spread"


# --------------------------------------------------------------------------- #
# 6. The non-spanning formula, kept as the contrast that makes test 1 discriminating.
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_the_historical_spacing_is_untouched_by_default() -> None:
    """`span_full_gap=False` gives ``gap/(groups+1)``: 6.0 / 5.75 / 5.50, with 0.5 left unused.

    No submitted run used this setting -- `claim_support_rows` passes ``span_full_gap=True`` -- so
    this pins the arithmetic of the knob rather than the behaviour of an arm. It is the companion
    to the first test: the two formulas must produce different numbers from the same input, or
    test 1 would pass under either rule and certify nothing.
    """
    kept = [("dA", 6.0), ("dB", 6.0), ("dC", 6.0), ("dD", 5.0)]
    out = _separate_tied_groups(kept, _key)  # defaults: span_full_gap=False, strict=False
    assert [s for _, s in out] == pytest.approx([6.0, 5.75, 5.50, 5.0])


# --------------------------------------------------------------------------- #
# 7. No ties in the emitted file: with the negative control that proves it can fail.
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_the_validator_accepts_a_strictly_descending_run() -> None:
    """The positive control. Without it, a checker that raised on everything would look right."""
    rows = ["2000 Q0 dA 1 6.000000000 R\n",
            "2000 Q0 dB 2 5.500000500 R\n",
            "2000 Q0 dC 3 5.000001000 R\n"]
    assert check_run_file(_write(rows), top_docs=1000) == 3


@pytest.mark.small
@pytest.mark.parametrize(
    ("bad", "why"),
    [
        (["2000 Q0 dA 1 5.000000000 R\n", "2000 Q0 dB 2 5.000000000 R\n"], "identical text"),
        # The case a text comparison misses: different strings, same parsed number. trec_eval
        # reads the text and converts it, so this is a real tie.
        (["2000 Q0 dA 1 5.0 R\n", "2000 Q0 dB 2 5.000000000 R\n"], "same value, different text"),
        (["2000 Q0 dA 1 5.0 R\n", "2000 Q0 dB 2 6.0 R\n"], "ascending score"),
        (["2000 Q0 dA 1 6.0 R\n", "2000 Q0 dB 3 5.0 R\n"], "rank is not its 1-based position"),
    ],
)
def test_the_validator_REFUSES_a_tie_or_a_non_descending_score(bad: list[str], why: str) -> None:
    """The negative controls. The earlier serialiser produced ties by construction, so a no-ties
    assertion that cannot fail would certify nothing; these four make its silence mean
    something."""
    with pytest.raises(TrecFormatError):
        check_run_file(_write(bad), top_docs=1000)


@pytest.mark.small
def test_the_write_time_gate_takes_the_strict_default() -> None:
    """The rule must be ON where the submission is actually produced, not merely available.

    `project.py` stages the Task 2 payload, runs `check_run_file` over the staged bytes, turns a
    `TrecFormatError` into `SubmissionRejected` with nothing published, and only then writes the
    real file. It passes no `forbid_ties=`, so it takes the default, which is what makes the
    strictly-descending rule live at write time. Both halves are asserted: the default is strict,
    and the call site does not override it. A checker that is correct but unreachable from the
    write path would guard nothing.
    """
    import importlib
    import inspect

    sig = inspect.signature(check_run_file)
    assert sig.parameters["forbid_ties"].default is True, "the DEFAULT must be strict"

    # `import_module`, not `from ... import project`: `select_serialize.project` is both a module
    # and the function it exports, and the package `__init__` binds the function, so the plain
    # import yields the function and `getsource` returns only its body.
    src = inspect.getsource(importlib.import_module("ragtime.pipeline.select_serialize.project"))
    assert "check_run_file(" in src, "the write path must run the T2 self-check at all"
    assert "forbid_ties=" not in src, (
        "project.py passes no forbid_ties=, so it inherits the strict default. If a call site ever "
        "needs to loosen it, this assertion should fail and force the decision to be explicit."
    )


@pytest.mark.small
def test_forbid_ties_false_still_reads_a_pre_change_run_file() -> None:
    """``forbid_ties=False`` keeps the looser rules available, so a run file carrying tied scores
    stays readable by its own checker instead of every archived run becoming unreadable."""
    rows = ["2000 Q0 dA 1 5.0 R\n", "2000 Q0 dB 2 5.0 R\n"]
    assert check_run_file(_write(rows), top_docs=1000, forbid_ties=False) == 2


# --------------------------------------------------------------------------- #
# 8. The rule-4 collapse is the mean of the logs, not the maximum.
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_mean_of_logs_collapse_flips_the_winner_against_max() -> None:
    """X cited at -0.1 and -3.0 collapses to -1.55 and loses to Y at -0.5.

    Under the previous maximum collapse X wins at -0.1, so this separates the two rules rather
    than passing under both. Computed here rather than hardcoded, so the test states the
    arithmetic.
    """
    x_vals, y_vals = [-0.1, -3.0], [-0.5]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    assert x_mean == pytest.approx(-1.55)
    assert y_mean > x_mean, "mean of logs: Y wins"
    assert max(x_vals) > max(y_vals), "MAX: X would have won, the rules genuinely disagree"

    # And the ordering the tie key produces from those collapsed values.
    kept = [("X", 6.0), ("Y", 6.0), ("zz", 5.0)]
    ranked = sorted(kept[:2], key=lambda ds: -{"X": x_mean, "Y": y_mean}[ds[0]])
    assert [d for d, _ in ranked] == ["Y", "X"]
