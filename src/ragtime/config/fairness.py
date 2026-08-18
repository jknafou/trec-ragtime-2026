"""The single place the fairness invariant is machine-checked, on raw config text.

Attribution must come from shared infrastructure rather than frozen evidence. Within
each run family, resolved from the run id, ``family_guard`` asserts:

  (a) the shared blocks are byte-identical, compared on the raw source text, which is
      a stronger and more diff-auditable guarantee than equality after parsing. The
      list is ``schema.SHARED_BLOCKS`` and is not restated here, so that
      a second enumeration cannot go stale;
  (b) exactly one knob moved for the family: controlled ``e2e`` runs keep
      ``retrieval.index == original`` and vary only ``passage_lang``, while
      controlled ``mlir`` runs keep ``passage_lang == original`` and vary only
      ``retrieval.index``;
  (c) a run that moves both knobs is allowed only when ``status == optional``, which
      excludes it from the controlled three-way contrast;
  (d) the round-0 seed-bank precondition, namely that ``seeds`` is equal within the
      family. The seed bank derives purely from the byte-identical ``llm`` and
      ``decomposition`` blocks plus ``seeds``, so config-time equality of those is
      the guarantee; comparing the seed-bank hash itself is a runtime assertion.

``top_level_blocks`` and ``shared_block_hash`` are the canonical raw-text algorithm,
kept byte-identical to the standalone fairness-check tooling and pinned by a
regression test, so the copies cannot silently drift. The gate is CPU-only and runs
before any planning or GPU step.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .loader import load
from .schema import SHARED_BLOCKS, RunConfig

__all__ = ["FairnessError", "check", "family_guard", "shared_block_hash", "top_level_blocks"]

_TOP_KEY = re.compile(r"^([A-Za-z_][\w-]*):")


class FairnessError(RuntimeError):
    """Raised when a run family violates the fairness invariant, before any GPU work."""


# --------------------------------------------------------------------------- #
# The canonical raw-text algorithm, byte-identical to the standalone check.
# --------------------------------------------------------------------------- #
def top_level_blocks(text: str) -> dict[str, str]:
    """Map each column-0 ``key:`` to the raw lines of its block, header plus body.

    A top-level block is the text from a column-0 ``key:`` line up to, but not
    including, the next column-0 non-comment key.
    """
    blocks: dict[str, str] = {}
    cur: list[str] = []
    key: str | None = None
    for line in text.splitlines():
        m = _TOP_KEY.match(line)
        is_top = bool(m) and not line.startswith((" ", "\t"))
        if is_top:
            if key is not None:
                blocks[key] = "\n".join(cur)
            key, cur = m.group(1), [line]  # type: ignore[union-attr]
        elif key is not None:
            cur.append(line)
    if key is not None:
        blocks[key] = "\n".join(cur)
    return blocks


def shared_block_hash(text: str) -> str:
    """sha256 over ``SHARED_BLOCKS`` in fixed order."""
    blocks = top_level_blocks(text)
    h = hashlib.sha256()
    for name in SHARED_BLOCKS:
        h.update(name.encode())
        h.update(b"\0")
        h.update(blocks.get(name, "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# The fairness gate.
# --------------------------------------------------------------------------- #
def _family_of(run_id: str) -> str:
    if run_id.startswith("e2e"):
        return "e2e"
    if run_id.startswith("mlir"):
        return "mlir"
    return f"other:{run_id}"


def _check_shared_blocks(fam: str, members: list[RunConfig]) -> None:
    """Check (a): every shared block must be byte-identical across the family."""
    # Keyed by position rather than by run_id: two members that happen to share a run
    # id must each be compared, or a genuine divergence between them would be dropped
    # by a run_id-keyed dict.
    per_member = [(cfg.run_id, top_level_blocks(cfg.source_text)) for cfg in members]
    for name in SHARED_BLOCKS:
        texts = [(rid, blocks.get(name, "")) for rid, blocks in per_member]
        distinct = {t for _, t in texts}
        if len(distinct) > 1:
            majority_text, _ = Counter(t for _, t in texts).most_common(1)[0]
            offenders = sorted({rid for rid, t in texts if t != majority_text})
            raise FairnessError(
                f"[{fam} family] shared block '{name}' diverges: "
                f"run(s) {offenders} differ from the family majority. "
                f"The shared blocks ({'/'.join(SHARED_BLOCKS)}) must be "
                f"byte-identical across the family: apply the identical change to every "
                f"run, or the translation delta becomes an infrastructure artifact."
            )


def _check_knobs(fam: str, members: list[RunConfig]) -> None:
    """Checks (b) and (c): one knob per family, and both-moved runs must be optional."""
    for cfg in members:
        moved_search = cfg.retrieval_index != "original"
        moved_reading = cfg.passage_lang != "original"
        if moved_search and moved_reading:
            if cfg.status != "optional":
                raise FairnessError(
                    f"[{fam} family] run '{cfg.run_id}' moves BOTH knobs "
                    f"(retrieval.index={cfg.retrieval_index}, passage_lang={cfg.passage_lang}) "
                    f"but status={cfg.status!r}; a run that moves both knobs MUST be "
                    f"status: optional (excluded from the controlled 3-way comparison)."
                )
            continue  # the optional max-performance entry may move both
        # Which knob may move is a property of the FAMILY, not of `kind`. The two axes
        # are independent: the family says which knob the experiment varies (e2e varies
        # passage_lang, mlir varies retrieval.index), while `kind` says which code path
        # executes it. Keying these branches on `kind` would misread an mlir arm that
        # runs the agentic path as a Knob-2 experiment.
        if fam == "e2e" and moved_search:
            raise FairnessError(
                f"[{fam} family] controlled run '{cfg.run_id}' moved the search knob "
                f"retrieval.index={cfg.retrieval_index} (must stay 'original'); "
                f"e2e-family runs vary only the reading knob passage_lang."
            )
        if fam == "mlir" and moved_reading:
            raise FairnessError(
                f"[{fam} family] controlled run '{cfg.run_id}' moved the reading knob "
                f"passage_lang={cfg.passage_lang} (must stay 'original'); "
                f"mlir-family runs vary only the search knob retrieval.index."
            )


def _check_seed_bank(fam: str, members: list[RunConfig]) -> None:
    """Check (d): the round-0 seed-bank precondition, seeds equal within the family."""
    if len({cfg.seeds for cfg in members}) > 1:
        detail = ", ".join(f"{cfg.run_id}={cfg.seeds}" for cfg in members)
        raise FairnessError(
            f"[{fam} family] seed-bank precondition violated: seeds differ within the family "
            f"({detail}); the round-0 seed bank derives from the byte-identical "
            f"llm+decomposition blocks + seeds, so seeds must be equal across variants."
        )


def family_guard(configs: Sequence[RunConfig]) -> None:
    """Raise ``FairnessError`` unless every run family upholds the fairness invariant.

    Groups by family, resolved from the run id, and applies checks (a) to (d). Each
    config must carry its ``source_text``, since the shared-block check hashes the raw
    file bytes; a synthesized ``RunConfig`` without it fails loudly rather than hashing
    an empty string. No model or GPU is touched.

    This validates the internal consistency of whatever set it is handed. Family
    completeness, meaning that all expected runs of a family are present, is the
    caller's responsibility. Rule (c) is applied per config rather than restricted to a
    designated run id.
    """
    families: dict[str, list[RunConfig]] = {}
    for cfg in configs:
        if not cfg.source_text:
            raise FairnessError(
                f"run '{cfg.run_id}': source_text is missing, so byte-identical "
                f"shared blocks cannot be verified. Obtain configs via config.load(path) "
                f"rather than synthesizing them."
            )
        families.setdefault(_family_of(cfg.run_id), []).append(cfg)

    for fam, members in families.items():
        _check_shared_blocks(fam, members)
        _check_knobs(fam, members)
        _check_seed_bank(fam, members)


def check(paths: Sequence[str | Path]) -> list[RunConfig]:
    """Load every path, run ``family_guard`` over the result, and return the configs.

    The shape ``orchestration.cli.run`` calls as one hard pre-launch gate: it raises,
    from ``load``, ``validate`` or ``family_guard``, before any planning or GPU step.
    """
    configs = [load(p) for p in paths]
    family_guard(configs)
    return configs
