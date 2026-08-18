"""Pipeline stage 1: the question-nugget bank.

One operation parameterized by round (``grow_nuggets``), plus the bank machinery it needs. Round 0
is the seed: retrieval-free facet enumeration from the request alone, which is what makes it
byte-identical across the three translation renderings and gives the runtime fairness anchor
(``fairness_anchor.assert_seed_parity``). Rounds 1 and above take the coverage-audit branch of the
same function.

Two predicates ship standalone: ``on_topic.on_topic``, the admission gate over every proposed gap
nugget, and ``saturation.saturated``, the loop's stop condition, which is called by
``pipeline.round_loop`` and never from inside this package. The sequential round loop lives one
level up because it invokes the RAG loop and this package may not.

This package imports nothing from ``ragtime.retrieval`` or ``ragtime.pipeline.rag_loop``: round
0's retrieval-freeness is a property of the import graph rather than of a comment.
"""

from __future__ import annotations

from .bank import (
    add,
    bank_fingerprint,
    close,
    mint,
    open_nuggets,
    prune,
    read_bank,
    write_bank,
)
from .coverage_audit import (
    AuditDelta,
    Evidence,
    Gap,
    apply_delta,
    attach_evidence,
    audit,
    evidence_from_results,
)
from .dedup import dedup_nuggets
from .exemplars import EXEMPLAR_SETS, Exemplar, select_exemplars
from .fairness_anchor import SeedParityError, assert_seed_parity
from .grow_nuggets import grow_nuggets
from .kband import limit_to_k
from .on_topic import on_topic
from .prompts import seed_prompt, self_critique_prompt
from .saturation import saturated
from .weighting import weight_and_dedup

__all__ = [
    "EXEMPLAR_SETS",
    "AuditDelta",
    "Evidence",
    "Exemplar",
    "Gap",
    "SeedParityError",
    "add",
    "apply_delta",
    "assert_seed_parity",
    "attach_evidence",
    "audit",
    "bank_fingerprint",
    "close",
    "dedup_nuggets",
    "evidence_from_results",
    "grow_nuggets",
    "limit_to_k",
    "mint",
    "on_topic",
    "open_nuggets",
    "prune",
    "read_bank",
    "saturated",
    "seed_prompt",
    "select_exemplars",
    "self_critique_prompt",
    "weight_and_dedup",
    "write_bank",
]
