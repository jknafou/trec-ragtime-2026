"""The per-loop artifact: ``rag_loop/{nugget_id}.jsonl``, one record per answered nugget.

Each loop writes a record carrying the two translation knobs and, per answer, the verbatim
span the claim was grounded on. The knobs are resolved from ``cfg`` rather than from the loop,
because ``run_loop`` is request-blind and knows neither which index was searched nor which
rendering it read.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import doc_id_of
from ragtime.common.io import read_jsonl, write_jsonl

if TYPE_CHECKING:
    from ragtime.common import Layout

__all__ = ["loop_record", "write_loop_record", "write_round_records"]


def _knobs(cfg: Any) -> tuple[str, str]:
    """Return ``(retrieval.index, passage_lang)``, the two translation knobs of the run.

    Both are read defensively: an empty string is visibly missing, whereas a wrong value would
    silently read as an arm the run was not.
    """
    index = ""
    retrieval = getattr(cfg, "blocks", {}).get("retrieval", {}) if hasattr(cfg, "blocks") else {}
    if isinstance(retrieval, dict):
        index = str(retrieval.get("index", "") or "")
    index = index or str(getattr(cfg, "retrieval_index", "") or "")
    passage_lang = str(getattr(cfg, "passage_lang", "") or "")
    return index, passage_lang


def _answer_dict(answer: Any) -> dict[str, Any]:
    if is_dataclass(answer) and not isinstance(answer, type):
        return asdict(answer)
    if isinstance(answer, dict):
        return dict(answer)
    return {"value": str(answer)}


def loop_record(result: Any, cfg: Any, *, topic_id: str, seed: int, round_no: int) -> dict[str, Any]:
    """Build the record for one loop as a plain JSON-safe dict.

    Kept separate from the writer so the shape can be asserted without a filesystem, and so the
    round writer can build every record before opening a file.
    """
    index, passage_lang = _knobs(cfg)
    answers = [_answer_dict(a) for a in (getattr(result, "answers", ()) or ())]
    return {
        "nugget_id": getattr(result, "nugget_id", ""),
        "question": getattr(result, "question", ""),
        "status": getattr(result, "status", ""),
        "closed_by": getattr(result, "closed_by", ""),
        "topic_id": topic_id,
        "seed": seed,
        "round": round_no,
        # Written even when empty, so a reader can tell "not recorded" from "searched original".
        "retrieval_index": index,
        "passage_lang": passage_lang,
        "answers": answers,
        # The evidence trail is retained even on abstain: a nugget the corpus could not answer
        # still produced real evidence about what is in it.
        "retrieved": [_plainable(r) for r in (getattr(result, "retrieved", ()) or ())],
        "search_trail": list(getattr(result, "search_trail", ()) or ()),
        "action_trail": list(getattr(result, "action_trail", ()) or ()),
        "rationale_trail": list(getattr(result, "rationale_trail", ()) or ()),
        "turns": getattr(result, "turns", 0),
        "searches": getattr(result, "searches", 0),
        "claims_committed": getattr(result, "claims_committed", 0),
        "claims_rejected": getattr(result, "claims_rejected", 0),
        # These two make a `closed_by: malformed_action` diagnosable from disk: which exception
        # ended the loop, and why individual claims were refused.
        # Copied because a `LoopResult` field is a live dict serialized later.
        "rejection_reasons": dict(getattr(result, "rejection_reasons", {}) or {}),
        "closed_detail": str(getattr(result, "closed_detail", "") or ""),
        # Half of the abstain predicate: (n_search >= min_searches) and (answer_failures >= 1).
        "answer_failures": getattr(result, "answer_failures", 0),
        "claim_retries": getattr(result, "claim_retries", 0),
        "contested": bool(getattr(result, "contested", False)),
    }


def _plainable(obj: Any) -> Any:
    """Return a JSON-safe view of one ``Retrieved`` (or an already-plain mapping)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    return str(obj)


def write_loop_record(
    layout: Layout,
    result: Any,
    cfg: Any,
    *,
    topic_id: str,
    seed: int,
    round_no: int = 0,
) -> Path:
    """Write one loop's record to ``Layout.rag_loop(nugget_id)``.

    The write goes through ``common.io.write_jsonl``, which does temp -> fsync -> replace ->
    fsync(dir); a local temp-then-rename would leave a truncated file behind a preempted node.
    The path is per nugget and carries no round level, so the file holds one line per round,
    keyed by ``round``: earlier rounds are preserved and this round replaces its own line, which
    keeps a resumed round idempotent instead of doubled.
    """
    nugget_id = str(getattr(result, "nugget_id", "") or "unknown")
    path = layout.rag_loop(nugget_id)
    record = loop_record(result, cfg, topic_id=topic_id, seed=seed, round_no=round_no)
    rows = [r for r in _existing(path) if r.get("round") != round_no]
    rows.append(record)
    rows.sort(key=lambda r: int(r.get("round", 0)))
    return write_jsonl(path, rows, skip_if_done=False)


def _existing(path: Path) -> list[dict[str, Any]]:
    """Return the records already on disk, or ``[]``.

    A malformed line is dropped rather than raised on: this file is a trajectory log, and
    refusing to record round 2 because round 1's line was truncated would turn a logging problem
    into a lost round of GPU work.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_round_records(
    layout: Layout,
    round_result: Any,
    cfg: Any,
    *,
    topic_id: str,
    seed: int,
) -> tuple[Path, ...]:
    """Write every loop record for one round. Returns the paths, in the round's own order."""
    return tuple(
        write_loop_record(
            layout, r, cfg, topic_id=topic_id, seed=seed, round_no=round_result.round_no
        )
        for r in round_result.results
    )


# Reading loop records back lives beside the writer because both `citation_scoring` (a producer)
# and `select_serialize` (its consumer) need this exact join, and a producer must not import its
# consumer. One implementation, so "what counts as cited" cannot drift between the two.
def answer_key(answer: Any) -> str:
    """Return the join key for one answer: its short value plus the span that grounded it.

    This is the same pair ``rag_loop`` keys idempotence on, so the bank's copy and the loop
    record's copy of an answer address each other exactly.
    """
    value = str(getattr(answer, "answer", "") if not isinstance(answer, dict) else answer.get("answer", ""))
    span = str(
        getattr(answer, "quoted_span", "") if not isinstance(answer, dict) else answer.get("quoted_span", "")
    )
    return f"{value}\x00{span}"


def loop_evidence(layout: Layout, nugget_id: str, *, through_round: int) -> dict[str, dict[str, Any]]:
    """Return ``answer_key -> {references, score}`` from one nugget's loop records.

    The bank's ``answer.references`` are always empty, because ``coverage_audit._merge_answers``
    does not carry them, so the real citations live only in the loop records. Rounds strictly
    greater than ``through_round`` are dropped. References accumulate across the retained rounds
    keeping the maximum score per doc-id, matching the answer-dedup merge rule.
    """
    path = layout.rag_loop(nugget_id)
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        if int(row.get("round", 0)) > through_round:
            continue
        for raw in row.get("answers", ()) or ():
            key = answer_key(raw)
            slot = out.setdefault(key, {"references": {}, "score": 0.0})
            for doc, score in (raw.get("references", {}) or {}).items():
                # Citations are always resolved to the original doc-id, whatever was read.
                doc = doc_id_of(str(doc))
                value = float(score)
                if doc not in slot["references"] or value > slot["references"][doc]:
                    slot["references"][doc] = value
            slot["score"] = max(float(slot["score"]), float(raw.get("score", 0.0) or 0.0))
    return out
