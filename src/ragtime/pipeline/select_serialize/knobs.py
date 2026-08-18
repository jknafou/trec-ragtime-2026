"""``SerializeKnobs``: the one place the ``serialize`` config block is read.

Every knob of that block arrives here and nowhere else, so "is this number a run choice or a code
constant?" has one answer per knob, written next to it.

The rule this module encodes is that a tuned knob has no code default. The config file is the
complete record, with no hardcoded budgets, but some knobs are tuned and the rest have an explicit
pinned value, and the two are treated differently:

* Required, never defaulted: ``k_t3``, ``answers_cap`` and ``dedup_a_cutoff``. Each is tuned, so
  its value is the experiment, and a code default would be a run parameter the reproducible record
  does not carry, silently applied to every submission.
* Defaulted to a pinned value: ``rrf_k`` (60, the same constant ``common.fusion.rrf`` defaults
  to), ``top_docs`` (1000, a track rule rather than our choice), ``dedup_q_cutoff`` (0.90),
  ``empty_fallback`` and ``validate`` (on), ``collection_id`` and ``run_mode``. Writing these into
  a config changes nothing, and omitting them cannot change the experiment.

If a config carries no ``serialize`` block, :func:`from_cfg` raises
:class:`SerializeConfigMissing` naming exactly what to add rather than falling back to a default
block: a submission produced from knobs no config carries would be unreproducible, which is the
one thing this stage exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ANSWER_SCORE_POLICIES",
    "COLLECTION_ID",
    "RUN_ID_MAX",
    "SPEC_DEFAULTS",
    "TUNED_KNOBS",
    "SerializeConfigMissing",
    "SerializeKnobs",
]

#: The Track-Guidelines cap on ``metadata.run_id``. The nuggets schema enforces it and the report
#: schema does not, so it is enforced here for Tasks 1 and 2.
RUN_ID_MAX = 25

#: The only collection in RAGTIME 2026; asserted equal to ``topic.collection_id``.
COLLECTION_ID = "ragtime/2/all"

#: How ``answer.score`` is obtained for the three places the design sorts on it. Nothing upstream
#: writes one: ``rag_loop.fanin`` records 0.0 with an explicit note that a placeholder confidence
#: would be a fabricated number.
#:
#: * ``max_reference`` takes ``max(references.values(), default=0.0)``, a derivation from an
#:   already-computed number rather than a new estimate.
#: * ``as_written`` takes ``answer.score`` verbatim, which is the policy to switch to once
#:   something upstream writes one; today it makes every score 0.0.
ANSWER_SCORE_POLICIES = ("max_reference", "as_written")

#: The knobs whose value the design pins: writing them into a config changes nothing and omitting
#: them cannot change the experiment. Kept as one mapping rather than as bare dataclass defaults,
#: because ``slots=True`` makes ``cls.<field>`` a ``member_descriptor`` rather than the default,
#: so :meth:`SerializeKnobs.from_cfg` could not read them back, and a hand-copied second fallback
#: list is exactly the drift the single-owner rule exists to prevent.
SPEC_DEFAULTS: dict[str, Any] = {
    # The same constant as the shared indexation fusion; `common.fusion.rrf` defaults to it.
    "rrf_k": 60,
    # A track rule rather than our choice: up to 1,000 results per query are accepted.
    "top_docs": 1000,
    # High and recall-safe.
    "dedup_q_cutoff": 0.90,
    "empty_fallback": True,
    "validate": True,
    "answer_score": "max_reference",
    "collection_id": COLLECTION_ID,
    "run_mode": "automatic",
}

#: The tuned knobs: no code default, ever. See the module docstring.
TUNED_KNOBS = ("k_t3", "answers_cap", "dedup_a_cutoff")


class SerializeConfigMissing(ValueError):
    """Raised when the run config carries no ``serialize`` block (or a required knob)."""


@dataclass(frozen=True, slots=True)
class SerializeKnobs:
    """The validated ``serialize`` block. Built only by :meth:`from_cfg`."""

    k_t3: int
    answers_cap: int
    dedup_a_cutoff: float
    dedup_q_cutoff: float = SPEC_DEFAULTS["dedup_q_cutoff"]
    rrf_k: int = SPEC_DEFAULTS["rrf_k"]
    top_docs: int = SPEC_DEFAULTS["top_docs"]
    empty_fallback: bool = SPEC_DEFAULTS["empty_fallback"]
    validate: bool = SPEC_DEFAULTS["validate"]
    answer_score: str = SPEC_DEFAULTS["answer_score"]
    collection_id: str = SPEC_DEFAULTS["collection_id"]
    run_mode: str = SPEC_DEFAULTS["run_mode"]

    @classmethod
    def from_cfg(cls, cfg: Any) -> SerializeKnobs:
        """Read and validate the ``serialize`` block off a loaded ``RunConfig``.

        ``cfg.blocks`` is the full validated top-level map, so this reads the same bytes
        ``config.hashing.all_hashes`` fingerprints: the hash the manifest records and the knobs
        the run used cannot disagree.
        """
        blocks = getattr(cfg, "blocks", None) or {}
        block = blocks.get("serialize")
        if not isinstance(block, dict) and not hasattr(block, "get"):
            raise SerializeConfigMissing(
                "the run config carries no 'serialize' block. The select-and-serialize knobs "
                "(k_t3, answers_cap, dedup_a_cutoff and the rest) are run choices and are never "
                "defaulted in code: add the block to every config/*.yml, to "
                "config.schema._ALLOWED, and to SHARED_BLOCKS so it stays byte-identical per "
                "family."
            )
        unknown = sorted(set(block) - set(cls.__dataclass_fields__))
        if unknown:
            raise SerializeConfigMissing(
                f"unknown serialize knob(s) {unknown}; the file must stay the complete record"
            )
        missing = [name for name in TUNED_KNOBS if block.get(name) is None]
        if missing:
            raise SerializeConfigMissing(
                f"serialize block is missing tuned knob(s) {missing}; each is tuned, so its "
                "value is the experiment and code must not supply one"
            )

        def pick(name: str) -> Any:
            return block.get(name, SPEC_DEFAULTS[name])

        knobs = cls(
            k_t3=int(block["k_t3"]),
            answers_cap=int(block["answers_cap"]),
            dedup_a_cutoff=float(block["dedup_a_cutoff"]),
            dedup_q_cutoff=float(pick("dedup_q_cutoff")),
            rrf_k=int(pick("rrf_k")),
            top_docs=int(pick("top_docs")),
            empty_fallback=bool(pick("empty_fallback")),
            validate=bool(pick("validate")),
            answer_score=str(pick("answer_score")),
            collection_id=str(pick("collection_id")),
            run_mode=str(pick("run_mode")),
        )
        if knobs.answer_score not in ANSWER_SCORE_POLICIES:
            raise SerializeConfigMissing(
                f"serialize.answer_score must be one of {list(ANSWER_SCORE_POLICIES)}; "
                f"got {knobs.answer_score!r}"
            )
        if knobs.k_t3 < 1 or knobs.answers_cap < 1 or knobs.top_docs < 1:
            raise SerializeConfigMissing(
                "serialize k_t3 / answers_cap / top_docs are caps and must be >= 1"
            )
        return knobs
