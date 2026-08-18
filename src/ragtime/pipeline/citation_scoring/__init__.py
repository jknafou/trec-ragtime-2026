"""The post-hoc hierarchical citation scorer.

``citation_score = nugget_importance x claim_importance``

The score has exactly two factors, and every row of every filed ``scores.jsonl`` names those two.

It ranks citations and never gates them. A threshold that discards a claim whose score is too
low is not implemented here, so nothing in this package can drop a claim or a
citation: the higher the score, the higher the assessment priority.

Only ``score_citations`` is exported, keeping the coupling point with select-and-serialize stable.
"""

from __future__ import annotations

from .scorer import score_citations

__all__ = ["score_citations"]
