"""Shared fixtures for the select-and-serialize tiers, built from the real record classes.

Nothing here is a hand-written dict. A cell is materialised by the same writers production
uses (``decompose.bank.write_bank``, ``pipeline.records.write_loop_record``) from the same frozen
records (``common.Nugget``/``Answer``/``Retrieved``/``Support``, ``rag_loop.LoopResult``), and the
config is the real ``config/e2e-omt.yml`` loaded through ``config.load``. Three separate defects
have hidden behind fixtures that spoke a vocabulary the real system does not: ``"open"`` where a
real bank carries ``STATUS_UNANSWERED``; a mocked payload carrying a ``weight`` key the real
grammar forbids; an assertion the bug already satisfied.

The one thing that cannot come from a real file is the ``serialize`` config block: it exists in no
``config/*.yml`` and in no ``config.schema._ALLOWED`` entry yet, because promoting it into
``SHARED_BLOCKS`` is a fairness-family decision taken once across the whole family, not an
implementer's edit. So :func:`cfg_with_serialize` takes the real loaded config and adds the block
in memory via ``dataclasses.replace``: the shipped file is never edited, and the day the block
lands this helper becomes a one-line passthrough.

Both tiers import this module: every claim the full tier makes about shapes is re-made in the
small tier from the same builders, so a ``RecursionError``-class defect reproduces in a fraction
of a second instead of only behind a long, index-loading run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ragtime.common import Answer, Layout, Nugget, Retrieved, Support
from ragtime.common.io import write_jsonl
from ragtime.config import load as load_config
from ragtime.pipeline.decompose.bank import write_bank
from ragtime.pipeline.rag_loop import LoopResult
from ragtime.pipeline.records import write_loop_record
from ragtime.pipeline.select_serialize.load import SCORES_FILE

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_E2E_CONFIG = REPO_ROOT / "config" / "e2e-omt.yml"
REAL_MLIR_CONFIG = REPO_ROOT / "config" / "mlir-original.yml"

#: A minimal, explicit ``serialize`` block. Every tuned knob is stated, because the code refuses to
#: default one, and every defaulted knob is omitted, so the tests exercise the same "which knobs may
#: be absent" contract a real config would.
SERIALIZE_BLOCK: dict[str, Any] = {
    "k_t3": 20,
    "answers_cap": 3,
    "dedup_a_cutoff": 0.85,
}


def cfg_with_serialize(path: Path = REAL_E2E_CONFIG, **overrides: Any) -> Any:
    """The real config, plus the ``serialize`` block that no ``config/*.yml`` carries yet."""
    cfg = load_config(path)
    block = {**SERIALIZE_BLOCK, **overrides}
    return dataclasses.replace(cfg, blocks={**dict(cfg.blocks), "serialize": block})


# --------------------------------------------------------------------------- #
# A deterministic stand-in for the resident embedder + the shared vLLM confirm gate
# --------------------------------------------------------------------------- #
_VOCAB_DIM = 64


def stub_embed(texts: list[str]) -> list[list[float]]:
    """A deterministic, normalised bag-of-words embedding: one vector per text.

    Not a hash-identity stub. Near-duplicate sentences have to score a high cosine, or an
    assertion that the merge fired would only be testing string equality. The real
    ``serving.embed(texts, mode="dense")`` also returns normalised vectors, which is why neither
    this nor the production path normalises again.
    """
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _VOCAB_DIM
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=2).digest()
            vec[int.from_bytes(digest, "big") % _VOCAB_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


async def stub_confirm_never(_a: str, _b: str) -> bool:
    """An answer-dedup confirm gate that always vetoes: the and-gate's recall-safe side."""
    return False


async def stub_confirm_always(_a: str, _b: str) -> bool:
    """A confirm gate that always agrees, so the embedding candidate alone decides the merge."""
    return True


def recording_confirm(seen: list[tuple[str, str]], *, agree: bool = True):
    """A confirm gate that keeps the pair it was asked about, so a test can check pair order."""

    async def confirm(a: str, b: str) -> bool:
        seen.append((a, b))
        return agree

    return confirm


# --------------------------------------------------------------------------- #
# Materializing a cell on disk with the production writers
# --------------------------------------------------------------------------- #
def write_cell(
    root: Path,
    *,
    topic_id: str,
    bank: tuple[Nugget, ...],
    cfg: Any,
    rounds: int = 1,
    scores: dict[tuple[str, str, str], float] | None = None,
    extra_loop_round: bool = False,
) -> Path:
    """Write ``<root>/topics/<topic_id>/`` exactly as ``pipeline.driver`` would, and return root.

    ``extra_loop_round`` writes a loop record for a round beyond the last complete bank. That is
    the shape a run interrupted mid-round leaves behind, and :func:`assemble_bank` must ignore it
    rather than ship evidence the coverage audit never saw.

    The bank is written with ``references`` and ``score`` stripped from the answers while the loop
    records keep both, which is what production produces:
    ``decompose.coverage_audit._merge_answers`` rebuilds each bank answer from an ``AnswerView``
    that carries neither field. A fixture that persisted the full answer into the bank would leave
    the enrichment join untested, and would pass even if that join were deleted.
    """
    topic_dir = root / "topics" / topic_id
    layout = Layout(run_dir=topic_dir, outputs=getattr(cfg, "outputs", None))
    persisted = tuple(
        dataclasses.replace(
            n,
            answers=tuple(
                dataclasses.replace(a, references={}, score=0.0) for a in n.answers
            ),
        )
        for n in bank
    )
    for r in range(rounds):
        write_bank(layout.decompose_round(r), persisted)
    for nugget in bank:
        for r in range(rounds):
            write_loop_record(
                layout,
                LoopResult(
                    nugget_id=nugget.nugget_id,
                    question=nugget.question,
                    status=nugget.status,
                    answers=nugget.answers,
                    retrieved=nugget.retrieved,
                ),
                cfg,
                topic_id=topic_id,
                seed=0,
                round_no=r,
            )
        if extra_loop_round:
            write_loop_record(
                layout,
                LoopResult(
                    nugget_id=nugget.nugget_id,
                    question=nugget.question,
                    status="answered",
                    answers=(
                        Answer(
                            answer="evidence the audit never saw",
                            sentence="A sentence from a round with no complete bank.",
                            quoted_span="never-audited",
                            references={"eng-docs/9999999": 0.99},
                        ),
                    ),
                ),
                cfg,
                topic_id=topic_id,
                seed=0,
                round_no=rounds,
            )
    if scores is not None:
        write_jsonl(
            layout.citation_scores() / SCORES_FILE,
            [
                {
                    "nugget_id": nugget_id,
                    "answer": key.split("\x00", 1)[0],
                    "quoted_span": key.split("\x00", 1)[1],
                    "doc_id": doc_id,
                    "score": score,
                }
                for (nugget_id, key, doc_id), score in sorted(scores.items())
            ],
            skip_if_done=False,
        )
    (topic_dir / "_SUCCESS").write_text("", encoding="utf-8")
    return root


def answer(
    value: str,
    sentence: str,
    *,
    refs: dict[str, float] | None = None,
    span: str = "span",
    support: tuple[Support, ...] = (),
) -> Answer:
    """One real ``common.Answer``. ``refs`` values are what the loop records carry."""
    return Answer(
        answer=value,
        sentence=sentence,
        quoted_span=span,
        references=dict(refs or {}),
        support=support,
    )


def nugget(
    nugget_id: str,
    question: str,
    *,
    weight: float = 0.5,
    status: str = "answered",
    aggregator_type: str = "OR",
    vital: bool = False,
    answers: tuple[Answer, ...] = (),
    retrieved: tuple[Retrieved, ...] = (),
) -> Nugget:
    """One real ``common.Nugget``."""
    return Nugget(
        nugget_id=nugget_id,
        question=question,
        weight=weight,
        vital=vital,
        aggregator_type=aggregator_type,
        status=status,
        answers=answers,
        retrieved=retrieved,
    )


def read_lines(path: Path) -> list[str]:
    """The emitted file's own bytes, as lines (never the in-memory rows)."""
    return path.read_text(encoding="utf-8").splitlines()


def read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    """Parse an emitted JSONL submission the way the validator's reader does: line by line."""
    return [json.loads(line) for line in read_lines(path)]


def record_measurements(name: str, payload: dict[str, Any]) -> Path:
    """Write a tier's measurements to a JSON file, never to the terminal reporter.

    pytest's capture swallows anything printed by a test that passes, so measurements reported
    that way are lost exactly when the run succeeded. A file survives both the capture and the
    job log.
    """
    out = REPO_ROOT / "logs" / "gates" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
