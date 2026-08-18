"""The citation join: the citations live in the loop records, not in the nugget bank.

`coverage_audit._merge_answers` rebuilds every bank answer from an `AnswerView` carrying no
`references`, so a scorer that reads the bank alone sees `references == {}` on every answer,
skips them all, and writes an empty-but-present `scores.jsonl`: which `select_serialize` then
reports as `citation_scores: present`. A wrong result with a positive provenance claim.

The fixture reproduces that split: the bank's answers have empty references and the loop record
carries the real ones. A scorer reading the bank alone scores nothing here, and only the join
scores every citation. A fixture whose bank happened to carry references would pass either way.
"""

from __future__ import annotations

import json

import pytest

from ragtime.pipeline.records import answer_key, loop_evidence


def _write(layout, nugget_id: str, *, answer: str, span: str, refs: dict[str, float]) -> None:
    """One loop record carrying real references, the shape `fan_in` writes."""
    path = layout.rag_loop(nugget_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"round": 0, "answers": [
            {"answer": answer, "quoted_span": span, "references": refs, "score": 0.0}
        ]}) + "\n",
        encoding="utf-8",
    )


@pytest.mark.small
def test_loop_evidence_recovers_citations_the_bank_does_not_have(tmp_path) -> None:
    from ragtime.common.layout import Layout

    layout = Layout(run_dir=tmp_path)
    _write(layout, "n1", answer="42", span="the answer is 42",
           refs={"docA": 0.0, "docB": 0.0})

    ev = loop_evidence(layout, "n1", through_round=0)
    key = answer_key({"answer": "42", "quoted_span": "the answer is 42"})
    assert key in ev, "the join key must match fan_in's (answer, quoted_span) spelling"
    assert set(ev[key]["references"]) == {"docA", "docB"}


@pytest.mark.small
def test_rounds_beyond_through_round_are_dropped(tmp_path) -> None:
    """A cell interrupted between a fan and the audit has loop records for a round with no bank."""
    from ragtime.common.layout import Layout

    layout = Layout(run_dir=tmp_path)
    path = layout.rag_loop("n1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"round": 0, "answers": [
            {"answer": "a", "quoted_span": "s", "references": {"early": 0.0}}]}) + "\n"
        + json.dumps({"round": 5, "answers": [
            {"answer": "a", "quoted_span": "s", "references": {"late": 0.0}}]}) + "\n",
        encoding="utf-8",
    )
    ev = loop_evidence(layout, "n1", through_round=0)
    refs = ev[answer_key({"answer": "a", "quoted_span": "s"})]["references"]
    assert "early" in refs and "late" not in refs, "a round past the authoritative bank leaked in"


@pytest.mark.small
def test_a_passage_id_in_a_loop_record_is_normalised_to_its_doc_id(tmp_path) -> None:
    """A citation names the original doc-id, whatever rendering was read or searched."""
    from ragtime.common.layout import Layout

    layout = Layout(run_dir=tmp_path)
    _write(layout, "n1", answer="a", span="s", refs={"doc-xyz_123#p4": 0.0})
    refs = loop_evidence(layout, "n1", through_round=0)[
        answer_key({"answer": "a", "quoted_span": "s"})
    ]["references"]
    assert all("#" not in d for d in refs), f"a passage id reached the citation set: {refs}"


@pytest.mark.small
def test_the_same_doc_cited_twice_keeps_the_MAX_score(tmp_path) -> None:
    from ragtime.common.layout import Layout

    layout = Layout(run_dir=tmp_path)
    path = layout.rag_loop("n1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"round": 0, "answers": [
            {"answer": "a", "quoted_span": "s", "references": {"d": 0.2}}]}) + "\n"
        + json.dumps({"round": 0, "answers": [
            {"answer": "a", "quoted_span": "s", "references": {"d": 0.7}}]}) + "\n",
        encoding="utf-8",
    )
    ev = loop_evidence(layout, "n1", through_round=0)
    assert ev[answer_key({"answer": "a", "quoted_span": "s"})]["references"]["d"] == 0.7


@pytest.mark.small
def test_citation_score_is_the_product_and_records_which_factors_ran() -> None:
    """The scoring contract: score = nugget_importance x claim_importance, and the row says so."""
    from ragtime.pipeline.citation_scoring.aggregate import citation_score

    score, factors = citation_score(0.8, 0.5)
    assert score == pytest.approx(0.40)
    assert factors == ("nugget_importance", "claim_importance"), (
        "every filed scores.jsonl row carries exactly these two names, in this order"
    )

    # It is a product, so either factor vetoes: an irrelevant nugget cannot ride on a strong claim.
    assert citation_score(0.0, 1.0)[0] == 0.0
    assert citation_score(1.0, 0.0)[0] == 0.0


@pytest.mark.small
def test_a_factor_outside_0_1_raises_rather_than_clamping() -> None:
    """A clamp turns a caller's bug into a plausible number that ranks a submission."""
    from ragtime.pipeline.citation_scoring.aggregate import citation_score

    for bad in (-0.1, 1.5):
        with pytest.raises(Exception):
            citation_score(bad, 0.5)
