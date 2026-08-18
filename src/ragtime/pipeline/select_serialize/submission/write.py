"""The submission writer, built on ``Layout.submission(track)`` and ``common.io``.

Paths are config-driven: ``Layout.submission(track)`` returns the ``config.outputs`` entry's
``path`` verbatim and raises ``KeyError`` on an unrouted track, so the reproducible record and the
emitted file cannot disagree. Nothing in this module joins a directory name.

Root resolution is explicit. Those paths are repo-relative, such as
``submissions/report_generation/run_1.jsonl``, and what they are relative to is pinned nowhere.
Inheriting the process working directory would make the same config write to a different place
depending on where the job started, so ``root`` is a required keyword argument resolved by the
caller, and an already-absolute config path passes through untouched.

Writes are atomic: Tasks 1 and 3 go through ``common.io.write_jsonl`` and the Task-2 text run
through ``common.io.write_lines``, both under the same temp, fsync, rename, fsync-directory,
``_SUCCESS`` contract.

A submission is written with ``skip_if_done=False``. The artifact tree is the checkpoint for
internal artifacts, but a submission file is a projection of them and must reflect the current bank
rather than whatever a previous partial run left behind.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragtime.common.io import write_jsonl, write_lines
from ragtime.common.layout import Layout

__all__ = ["resolve", "write_jsonl_rows", "write_trec_run"]


def resolve(layout: Layout, track: str, *, root: str | Path) -> Path:
    """Return the absolute path this track's submission is written to."""
    declared = layout.submission(track)
    return declared if declared.is_absolute() else Path(root) / declared


def write_jsonl_rows(path: str | Path, rows: Sequence[Any]) -> Path:
    """Write Task-1 and Task-3 rows as newline-delimited JSON, one object per line.

    The concatenated single-line style the topics file uses would crash the validator with
    ``JSONDecodeError: Extra data``, and so would a blank line; ``write_jsonl`` emits neither.
    """
    return write_jsonl(path, rows, skip_if_done=False)


def write_trec_run(path: str | Path, rows: Sequence[Any]) -> Path:
    """Write the 6-column Task-2 run, rows already grouped and ordered by the caller."""
    return write_lines(path, [row.to_line() for row in rows], skip_if_done=False)
