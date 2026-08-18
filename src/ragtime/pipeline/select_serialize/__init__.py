"""Select and serialize: the terminal deterministic projection into the three TREC deliverables.

The output format is the one the official validator accepts, verified against it with both an
accept and a reject demonstration.

The public surface is narrow:
:func:`~ragtime.pipeline.select_serialize.project.project` is the entry point, and everything else
is exported for tests and for the scorer and monitoring to key against.
"""

from __future__ import annotations

from .budget import fits, joined_len, trim_to_limit
from .knobs import RUN_ID_MAX, SerializeConfigMissing, SerializeKnobs
from .load import assemble_bank, citation_scores, topic_dirs
from .project import ProjectResult, project
from .task1 import task1_report
from .task3 import apply_caps, task3_bank

__all__ = [
    "RUN_ID_MAX",
    "ProjectResult",
    "SerializeConfigMissing",
    "SerializeKnobs",
    "apply_caps",
    "assemble_bank",
    "citation_scores",
    "fits",
    "joined_len",
    "project",
    "task1_report",
    "task3_bank",
    "topic_dirs",
    "trim_to_limit",
]
