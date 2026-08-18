"""The Task-2 self-check, since the official validator covers no part of a TREC run.

``rag_run_validator``'s ``--format`` accepts only ``report`` and ``nuggets``, and nothing in the
clone touches TREC run syntax, so for the Task-2 submissions this file is the only gate that exists.

It re-reads the emitted file bytes rather than the in-memory row list: a serialization bug in
``T2RunRow.to_line`` would sail past an in-memory assertion, and the bytes are what is submitted.

What each rule catches:

* Six whitespace-separated columns and the literal ``Q0``: a column-count slip shifts every field
  for the whole file.
* Scores non-increasing within a topic, and rows contiguous per topic: ``trec_eval`` would
  otherwise mis-rank silently.
* At most ``top_docs`` rows per request, since the track truncates beyond its own limit and an
  over-long run loses exactly the tail we thought we submitted.
* Lexicographic doc-id tie order, because the evaluation software re-orders equal scores by doc-id,
  so a file that does not already match its own re-run is a reproducibility hole.
* No ``#`` in any doc-id. The report schema's ``references`` is a bare string, so a passage id such
  as ``rus-docs/0330915#p1`` would be accepted by the validator and then silently fail to resolve at
  assessment time.

Every rule above is a statement about a line, and each lives inside the per-line loop, so over a
zero-row file they all pass vacuously and :func:`check_run_file` would return 0 for a run with no
content. Two checks therefore live outside the loop:

* Zero rows is an unconditional refusal. It is not a partial state to be judged against a policy
  switch; it is the absence of a submission.
* ``topics=`` makes the check capable of failing on absent content. It is the same set difference
  the official tool performs for Tasks 1 and 3, and it is one-directional in the same way: extra
  topics and a differing topic order are accepted, only missing ones are refused. Passing it is the
  caller's decision, so this module stays a checker of the file and the policy for when the gate is
  armed lives in :mod:`..project`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ragtime.common import doc_id_of

__all__ = ["TrecFormatError", "check_ids", "check_run_file"]

_COLUMNS = 6
_Q0 = "Q0"


class TrecFormatError(ValueError):
    """The emitted TREC run violates a rule the official validator does not check."""


def check_run_file(
    path: str | Path,
    *,
    top_docs: int,
    topics: Iterable[str] | None = None,
    forbid_ties: bool = True,
) -> int:
    """Assert every TREC run rule over the emitted bytes and return the row count.

    ``topics`` is the set of topic ids the run is declared to cover, taken from the run's own topics
    file. When given, a topic with no row in the file is refused; when omitted, only the file's own
    per-line rules are checked.

    ``forbid_ties=True``, the default, requires scores to be strictly descending within a topic, so
    no two rows may share a score. The claim-support ranking pushes its rule-4 reranker order into
    the score column, via ``task2_claims._separate_tied_groups(strict=True)``, precisely because
    row order does not survive the evaluator, so a surviving tie means the separation did not
    happen and the ordering would fall back to ``trec_eval``'s orthographic doc-id sort, which is
    not neutral in a multilingual corpus because doc-id prefixes track source and therefore
    language.

    Rank is covered by the stronger rule below: it must equal its 1-based position within the topic,
    so gaps and repeats are both refused without a separate check.

    ``forbid_ties=False`` restores the older pair of rules, non-increasing scores with lexicographic
    tie order, for a caller reading a file an earlier serializer produced.
    """
    text = Path(path).read_text(encoding="utf-8")
    rows = 0
    seen_topics: list[str] = []
    current: str | None = None
    per_topic = 0
    last_score: float | None = None
    last_doc: str | None = None
    #: The score field as emitted, used in the no-ties error message; the float and the text can
    #: disagree below the format's precision.
    last_score_s: str | None = None

    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise TrecFormatError(f"{path}:{lineno}: blank line in a TREC run file")
        fields = line.split()
        if len(fields) != _COLUMNS:
            raise TrecFormatError(
                f"{path}:{lineno}: {len(fields)} whitespace-separated fields, expected {_COLUMNS} "
                f"(topic_id Q0 doc_id rank score run_id): {line!r}"
            )
        topic_id, q0, doc_id, rank, score_s, _run_id = fields
        if q0 != _Q0:
            raise TrecFormatError(f"{path}:{lineno}: field 2 is {q0!r}, expected the literal 'Q0'")
        check_ids([doc_id])
        try:
            score = float(score_s)
            rank_i = int(rank)
        except ValueError as exc:
            raise TrecFormatError(f"{path}:{lineno}: unparsable rank/score: {line!r}") from exc

        if topic_id != current:
            if topic_id in seen_topics:
                raise TrecFormatError(
                    f"{path}:{lineno}: topic {topic_id!r} reappears after another topic; "
                    "entries for each topic must be contiguous"
                )
            seen_topics.append(topic_id)
            current, per_topic, last_score, last_doc = topic_id, 0, None, None
            last_score_s = None
        per_topic += 1
        if per_topic > top_docs:
            raise TrecFormatError(
                f"{path}:{lineno}: topic {topic_id!r} has more than {top_docs} rows; the track "
                "truncates beyond that, so the tail would be silently lost"
            )
        if rank_i != per_topic:
            raise TrecFormatError(
                f"{path}:{lineno}: rank {rank_i} is not the 1-based position {per_topic} "
                "within its topic"
            )
        if last_score is not None:
            # Compare the parsed value, not the score field as written: `trec_eval` decides the
            # ordering after parsing, so two rows printing "5.000000000" from floats 1e-12 apart
            # and the pair "5.0" / "5.000000000" must both count as tied.
            if forbid_ties and score >= last_score:
                raise TrecFormatError(
                    f"{path}:{lineno}: topic {topic_id!r} score {score_s!r} is not strictly less "
                    f"than the previous row's {last_score_s!r} ({doc_id!r} after {last_doc!r}). "
                    "Scores must strictly descend: the rule-4 reranker order is pushed into the "
                    "score column because trec_eval discards row order, so a tie means the "
                    "separation did not happen and the ordering would fall back to an orthographic "
                    "doc-id sort, which is not neutral when doc-id prefixes track source and "
                    "therefore language."
                )
            if not forbid_ties:
                if score > last_score:
                    raise TrecFormatError(
                        f"{path}:{lineno}: score {score} is above the previous {last_score}; "
                        "scores must be non-increasing within a topic"
                    )
                if score == last_score and last_doc is not None and doc_id < last_doc:
                    raise TrecFormatError(
                        f"{path}:{lineno}: tied scores out of lexicographic doc-id order "
                        f"({doc_id!r} after {last_doc!r}); trec_eval re-orders ties by doc-id, so "
                        "this file would not match its own re-run"
                    )
        last_score, last_doc, last_score_s = score, doc_id, score_s
        rows += 1

    # Both checks live outside the loop: everything above is a statement about a line that exists,
    # so a file with no lines satisfies all of it vacuously.
    if rows == 0:
        raise TrecFormatError(
            f"{path}: 0 rows. An empty TREC run is not a partial submission, it is the absence of "
            "one, and every per-line rule above passes vacuously over it. The official validator "
            "has the same trap for Tasks 1 and 3, where an empty run file exits 0; Task 2 has no "
            "official validator to fall back on."
        )
    if topics is not None:
        missing = sorted(set(map(str, topics)) - set(seen_topics))
        if missing:
            shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
            raise TrecFormatError(
                f"{path}: {len(missing)} of {len(set(map(str, topics)))} declared topic(s) have "
                f"no row in this run file: {shown}. The official validator refuses exactly this "
                "for Tasks 1 and 3; for Task 2 this is the only gate."
            )
    return rows


def check_ids(ids: Iterable[str]) -> None:
    """Check that every emitted id is an original doc-id and not a passage or sentence id.

    ``doc_id_of`` is the arbiter rather than a hand-written ``'#' not in id``, so the id round-trips
    through the same primitive the whole pipeline cites with and this check cannot drift from the
    definition it protects.
    """
    for raw in ids:
        if doc_id_of(raw) != raw:
            raise TrecFormatError(
                f"{raw!r} is not an original doc-id (doc_id_of gives {doc_id_of(raw)!r}). The "
                "validator's schema is a bare string and would accept this, after which every "
                "citation silently fails to resolve at assessment time."
            )
