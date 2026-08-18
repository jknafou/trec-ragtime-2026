"""The nugget bank: an accumulating ``tuple[Nugget, ...]`` and its content fingerprint.

The bank is the growing set of question-nuggets decompose owns across rounds. It is a plain
immutable tuple of frozen, slotted ``common.schemas.Nugget`` records: every operation here returns
a new tuple and every field change goes through ``dataclasses.replace``. That is what lets a caller
hold the previous round's bank for the trajectory log without it being rewritten underneath.

``bank_fingerprint`` is the runtime fairness anchor's hash. It reuses
``config.hashing.config_hash``, the one canonicalizing hasher (recursive NFC, finite-guard,
canonical JSON, sha256), rather than building a second one. It lives here rather than in ``common``
because ``common`` may not import ``config``, while ``pipeline`` already depends on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import (
    Answer,
    Nugget,
    Retrieved,
    Support,
    get_logger,
    nugget_id,
    read_jsonl,
    write_jsonl,
)
from ragtime.common.io import is_done, success_marker
from ragtime.config.hashing import config_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ragtime.common import Layout

_log = get_logger("pipeline.decompose.bank")

__all__ = [
    "STATUS_ANSWERED",
    "STATUS_PRUNED",
    "STATUS_UNANSWERED",
    "add",
    "bank_fingerprint",
    "close",
    "mint",
    "next_index",
    "open_nuggets",
    "prune",
    "read_bank",
    "write_bank",
]

STATUS_UNANSWERED = "unanswered"
STATUS_ANSWERED = "answered"
STATUS_PRUNED = "pruned"


# Accumulation ops: all replace-based, never mutating.
def add(bank: tuple[Nugget, ...], new: Iterable[Nugget]) -> tuple[Nugget, ...]:
    """Append ``new`` to ``bank``, skipping any ``nugget_id`` already present.

    Append-only and order-preserving: existing entries keep their position and their identity, so
    a later dedup pass walking the bank in order always resolves a collision in favour of the older
    nugget, whose id an earlier round's answers already reference.
    """
    seen = {n.nugget_id for n in bank}
    out = list(bank)
    for n in new:
        if n.nugget_id in seen:
            continue
        seen.add(n.nugget_id)
        out.append(n)
    return tuple(out)


def open_nuggets(bank: tuple[Nugget, ...]) -> tuple[Nugget, ...]:
    """Return the still-open nuggets, those with ``status == "unanswered"``.

    This is the set the round loop fans out to the k RAG loops; a pruned or already-answered
    nugget is never re-fanned.
    """
    return tuple(n for n in bank if n.status == STATUS_UNANSWERED)


def close(bank: tuple[Nugget, ...], nugget_ids: str | Iterable[str]) -> tuple[Nugget, ...]:
    """Mark the named nuggets ``answered`` (a ``pruned`` nugget is left alone)."""
    targets = _id_set(nugget_ids)
    return tuple(
        replace(n, status=STATUS_ANSWERED)
        if n.nugget_id in targets and n.status != STATUS_PRUNED
        else n
        for n in bank
    )


def prune(bank: tuple[Nugget, ...], nugget_ids: str | Iterable[str]) -> tuple[Nugget, ...]:
    """Mark the named nuggets ``pruned``. They stay in the bank, flagged.

    A pruned nugget is not deleted: its id must never be reusable, and the trajectory log needs the
    record of what was dropped and when. Membership of the bank tuple is therefore stable under
    ``prune``; only ``status`` moves.
    """
    targets = _id_set(nugget_ids)
    return tuple(
        replace(n, status=STATUS_PRUNED) if n.nugget_id in targets else n for n in bank
    )


def _id_set(nugget_ids: str | Iterable[str]) -> frozenset[str]:
    if isinstance(nugget_ids, str):
        return frozenset({nugget_ids})
    return frozenset(nugget_ids)


def next_index(bank: tuple[Nugget, ...]) -> int:
    """Return the next free ``#n{j}`` ordinal for ``bank``: the highest seen plus one.

    Ids are monotone, not dense: dedup may drop a nugget, and its ordinal is retired with it. A
    dense re-numbering would let a merged-away id resurface on a different question, which would
    make an earlier round's answers point at the wrong nugget.
    """
    highest = -1
    for n in bank:
        _, _, suffix = n.nugget_id.rpartition("#n")
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def mint(
    topic_id: str,
    questions: Sequence[str] | Sequence[tuple[str, float]],
    *,
    start_index: int = 0,
    origin_round: int = 0,
    trigger_passage_id: str | None = None,
) -> tuple[Nugget, ...]:
    """Mint nuggets for ``questions`` with sequential ``<topic_id>#n{j}`` ids.

    ``questions`` is either plain strings or ``(question, weight)`` pairs. Every minted nugget
    starts ``status="unanswered"`` with the schema's default ``aggregator_type="OR"``, since the
    answer-set-informed AND/OR confirmation needs answers that round 0 does not have, and with
    empty ``retrieved`` and ``answers``, which the RAG loops own.
    """
    out: list[Nugget] = []
    j = start_index
    for item in questions:
        if isinstance(item, str):
            question, weight = item, 0.0
        else:
            question, weight = item[0], float(item[1])
        text = question.strip()
        if not text:
            continue
        out.append(
            Nugget(
                nugget_id=nugget_id(topic_id, j),
                question=text,
                weight=weight,
                origin_round=origin_round,
                trigger_passage_id=trigger_passage_id,
            )
        )
        j += 1
    return tuple(out)


def bank_fingerprint(bank: tuple[Nugget, ...]) -> str:
    """Return the sha256 of the bank's question set, the seed-integrity hash.

    The canonical form is the nuggets sorted by ``nugget_id``, each reduced to ``{"question",
    "aggregator_type"}``, wrapped as ``{"nuggets": [...]}`` and run through
    ``config.hashing.config_hash``, which NFC-normalizes every string leaf, so two banks differing
    only in Unicode composition hash the same.

    It excludes ``weight`` and ``vital``, which are LLM-scored floats, because the
    fairness claim is that the three variants enumerate the same facets from the same request
    rather than that a sampled score is bit-identical. It also excludes ``status``, ``answers``
    and ``retrieved``, which are later-round state and legitimately diverge per variant. It is
    sensitive to ``question`` and ``aggregator_type``, which is what a leak of variant state into
    the seed would move.

    Sorting by ``nugget_id`` makes the hash order-independent: the artifact is a set, so two banks
    holding the same nuggets in a different order are the same bank.
    """
    payload = {
        "nuggets": [
            {"question": n.question, "aggregator_type": n.aggregator_type}
            for n in sorted(bank, key=lambda n: n.nugget_id)
        ]
    }
    return config_hash(payload)


# Artifact IO: the round-r bank JSONL.
def write_bank(path: str | os.PathLike[str], bank: tuple[Nugget, ...]) -> Path:
    """Write ``bank`` to ``path``, normally ``Layout.decompose_round(r)``, as atomic JSONL.

    Serialization is ``dataclasses.asdict``, not ``n.__dict__``, which does not exist on a slotted
    frozen dataclass; it recurses through the nested ``Retrieved``, ``Answer`` and ``Support``
    records and converts tuples to lists. ``common.io.write_jsonl`` supplies the atomic
    temp-then-rename, ``_SUCCESS`` and skip-if-done contract, so the artifact tree stays the
    checkpoint.
    """
    return write_jsonl(path, [asdict(n) for n in bank])


def seed_bank_hash(cfg: Any) -> str:
    """Return ``H(decomposition, llm, topics)``, the identity of a round-0 seed bank.

    Exactly the three blocks round 0 can depend on, and no others. Round 0 is retrieval-free by
    construction, so neither translation knob reaches it: ``passage_lang`` decides what the loops
    read and ``retrieval.index`` what they search, and this call happens before either applies.
    Including one of them would partition the cache by arm and give up the sharing entirely.

    Built from :func:`ragtime.config.all_hashes` rather than by hashing the blocks again here,
    because a second hashing scheme is how two parts of a system come to disagree about whether
    two configs are the same.
    """
    from ragtime.config import all_hashes

    h = all_hashes(cfg)
    digest = hashlib.sha256()
    for name in ("decomposition", "llm", "topics"):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(h.get(name, "").encode())
        digest.update(b"\0")
    return digest.hexdigest()


def publish_seed_bank(path: str | os.PathLike[str], bank: tuple[Nugget, ...]) -> tuple[Nugget, ...]:
    """Publish ``bank`` as the canonical round-0 artifact, first writer wins, and return the winner.

    Not ``write_bank``, which is temp-then-rename: atomic, but last-writer-wins. Two
    arms can reach the same ``(topic, seed)`` concurrently and their banks need not be equal, since
    sampled decoding is not batch-invariant and the two calls land on different pairs with
    different batching. Under last-wins the canonical bank would depend on scheduling order, and an
    arm that had already read the earlier version would be working from a bank the artifact no
    longer shows: a fairness violation that leaves no trace.

    So the publish is ``os.link``, which fails if the destination exists. The loser discards its
    own bank and reads the winner's, and every arm converges on one artifact. This is the same
    first-wins primitive ``workqueue.claim`` uses, applied to a file instead of a shard.

    The caller must use the returned bank rather than the one it passed in: on a lost race they
    differ, which is precisely the case this exists to make consistent.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(
            "".join(json.dumps(asdict(n), sort_keys=True) + "\n" for n in bank), encoding="utf-8"
        )
        try:
            os.link(tmp, dst)  # first wins; raises if another arm already published
        except FileExistsError:
            _log.info("decompose.seed_bank.lost_race", path=str(dst))
    finally:
        tmp.unlink(missing_ok=True)
    # Written by winner and loser alike, and idempotent: the marker is what `is_done` reads, and a
    # winner that died between link and marker would leave a complete file nobody trusts.
    marker = success_marker(dst)
    if not marker.exists():
        marker.write_bytes(b"")
    return read_bank(dst)


def read_bank(path: str | os.PathLike[str]) -> tuple[Nugget, ...]:
    """Read a bank artifact back into ``Nugget`` records, the inverse of :func:`write_bank`.

    This is the shape the round loop resumes from and select-and-serialize consumes; keeping the
    reader next to the writer is what stops two stages from disagreeing about the file's nesting.
    """
    return tuple(_nugget_from_dict(row) for row in read_jsonl(path))


def _nugget_from_dict(row: dict[str, Any]) -> Nugget:
    return Nugget(
        nugget_id=row["nugget_id"],
        question=row["question"],
        weight=float(row.get("weight", 0.0)),
        vital=bool(row.get("vital", False)),
        aggregator_type=row.get("aggregator_type", "OR"),
        status=row.get("status", STATUS_UNANSWERED),
        origin_round=int(row.get("origin_round", 0)),
        trigger_passage_id=row.get("trigger_passage_id"),
        retrieved=tuple(
            Retrieved(passage_id=r["passage_id"], score=float(r["score"]))
            for r in row.get("retrieved", ()) or ()
        ),
        answers=tuple(_answer_from_dict(a) for a in row.get("answers", ()) or ()),
    )


def _answer_from_dict(row: dict[str, Any]) -> Answer:
    # Every persisted field of `Answer` must round-trip. `coverage_audit._merge_answers` keys
    # answer identity on `(answer, quoted_span)`, so dropping `quoted_span` on read makes a
    # re-committed claim miss the seen-set and append the same answer twice, which diverges a
    # resumed run's labels, gaps and prunes from an uninterrupted one.
    return Answer(
        answer=row["answer"],
        sentence=row.get("sentence", ""),
        quoted_span=row.get("quoted_span", ""),
        score=float(row.get("score", 0.0)),
        references=dict(row.get("references", {}) or {}),
        support=tuple(
            Support(passage_id=s["passage_id"], lang=s["lang"])
            for s in row.get("support", ()) or ()
        ),
    )


def last_complete_round(layout: Layout, *, max_rounds: int = 64) -> int:
    """Return the highest round whose bank artifact carries its ``_SUCCESS`` companion.

    Scans upward from 0 and stops at the first gap, so a stray ``round_9.jsonl`` from an unrelated
    experiment cannot be mistaken for the end of a two-round run. Raises when round 0 itself is
    absent: a topic directory with no complete bank has nothing to serialize.
    """
    last = -1
    for r in range(max_rounds):
        if not is_done(layout.decompose_round(r)):
            break
        last = r
    if last < 0:
        raise FileNotFoundError(
            f"no complete decompose bank under {layout.run_dir}: "
            f"{layout.decompose_round(0)} has no _SUCCESS companion"
        )
    return last
