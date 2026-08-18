"""The submission manifest: our upload and reproducibility record, not a track file.

It captures the submission-time attributes that live outside the deliverable files: the collection,
the task, automatic versus manual, the assessment-priority order, the ``serialize`` config hash with
the variant and seed each file came from, and the validator's verdict verbatim.

Three things it is careful about:

* ``collection_id`` is asserted equal to ``topic.collection_id`` rather than copied from config, so
  a config claiming a different collection cannot describe a corpus we did not search.
* The validator verdict has three states, ``passed``, ``failed`` and not run because the tool was
  unavailable. A skip is not a pass, and recording an unrun check as passed would be worse than
  recording nothing.
* Nothing about the submission portal is verifiable from the validator: the task-name strings, the
  file-naming convention and the upload interface are portal-side. This file records what we intend
  to upload, not what the portal will accept.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ragtime.common.io import write_lines

__all__ = [
    "REDUCE_COMMAND",
    "CollectionMismatch",
    "FamilyManifestError",
    "assemble_family_manifest",
    "assert_collection",
    "manifest_document",
    "reduce_command",
    "write_manifest",
]

#: The single spelling of the reduce, so the instruction a fragment carries and the entry point that
#: performs it cannot drift. It lives here, beside `assemble_family_manifest`, rather than in
#: `..assemble_manifest`, which imports this module; the other direction would be a cycle.
REDUCE_COMMAND = (
    "python -m ragtime.pipeline.select_serialize.assemble_manifest --manifest-root {root}"
)


def reduce_command(manifest_root: str | Path) -> str:
    """Return the command that turns the fragments under ``manifest_root`` into the family file.

    ``manifest_root`` is the directory holding ``manifest.d/``, which for the shipped configs is
    ``<submission_root>/submissions``, one level below ``submission_root``. Passing the wrong one
    finds no fragments rather than failing loudly, so the command is generated from a resolved path
    here instead of being retyped from a usage example.
    """
    return REDUCE_COMMAND.format(root=manifest_root)


class FamilyManifestError(ValueError):
    """The per-run fragments cannot be reduced into one coherent family manifest."""


class CollectionMismatch(ValueError):
    """A topic's ``collection_id`` disagrees with the configured one."""


def assert_collection(topics: Sequence[Any], collection_id: str) -> None:
    """Check that the manifest's collection is the one the topics themselves name."""
    mismatched = sorted({str(t.collection_id) for t in topics} - {collection_id})
    if mismatched:
        raise CollectionMismatch(
            f"topics declare collection_id(s) {mismatched} but serialize.collection_id is "
            f"{collection_id!r}; the manifest would describe a corpus we did not search"
        )


def manifest_document(
    *,
    entries: Sequence[Mapping[str, Any]],
    run_mode: str,
    collection_id: str,
    config_hashes: Mapping[str, Any],
    coverage: Mapping[str, Any],
    manifest_root: str | Path,
) -> dict[str, Any]:
    """Return one run's manifest fragment as a plain dict, so a test can read the shape.

    ``entries`` are already-built per-file records, sorted here by assessment priority, taken from
    the ``run_N`` in the declared path, because only the first few runs are guaranteed to be pooled
    and scored. The ordering changes no file's content, so the fairness invariant holds.

    This document is not the deliverable manifest and says so in its own bytes. Its
    ``assessment_priority_order`` is degenerate by construction, one entry per file, so a run
    emitting Tasks 1 and 3 reads ``["e2e-omt", "e2e-omt"]``; the deliverable wants run_ids in
    priority order, a list of runs, and that dedupe happens only in
    :func:`assemble_family_manifest`. The ``family_manifest`` block is how a reader holding only
    this file learns what is missing and exactly what to run, with ``manifest_root`` already
    resolved.
    """
    ordered = sorted(entries, key=lambda e: (int(e.get("priority", 0)), str(e.get("track", ""))))
    return {
        "collection_id": collection_id,
        "run_mode": run_mode,
        "assessment_priority_order": [e.get("run_id") for e in ordered],
        "config_hashes": dict(config_hashes),
        "coverage": dict(coverage),
        "family_manifest": {
            "assembled": False,
            "note": (
                "This file is one run's fragment, not the family manifest. Its "
                "assessment_priority_order lists one entry per file and is deduped to a list of "
                "runs only by the reduce below, which is a separate step: run it after every cell "
                "has serialized, and re-run it at any time, since it is total and idempotent."
            ),
            "command": reduce_command(manifest_root),
            "writes": str(Path(manifest_root) / "manifest.json"),
        },
        "submissions": [dict(e) for e in ordered],
    }


def write_manifest(path: str | Path, document: Mapping[str, Any]) -> Path:
    """Write the manifest atomically, through ``common.io`` like every other artifact."""
    return write_lines(
        path,
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True).splitlines(),
        skip_if_done=False,
    )


def assemble_family_manifest(fragment_dir: str | Path) -> dict[str, Any]:
    """Reduce every per-run fragment into the single family manifest.

    ``project()`` is per-run while the manifest is per-family, so each run writes only
    ``manifest.d/<run_id>.json`` and concurrent cells never share a mutable file. This function is
    the pure, idempotent reduce over that directory: re-running it always yields the same answer for
    the same fragments, which makes it safe to call at any time, including by hand before an upload.

    Doing it opportunistically at the end of each run would race the fragment writes: run A could
    list the directory before run B's fragment lands and then win the rename, leaving a manifest
    that omits B permanently. A total reduce is only correct when it runs after the writes it
    covers, so this is an explicit step rather than a side effect.

    ``assessment_priority_order`` is deduped here: a run emits one or two files and the per-run
    document repeats its ``run_id`` once per file, whereas the deliverable wants run_ids in priority
    order, a list of runs. ``submissions`` keeps one entry per emitted file, which is also required,
    so both readings are served by different keys instead of one wrong list.

    Raises :class:`FamilyManifestError` rather than guessing when the fragments disagree on
    ``collection_id`` or ``run_mode``: those are family-wide facts, so a disagreement means the
    fragments came from two different experiments and no single manifest can describe them.
    """
    directory = Path(fragment_dir)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FamilyManifestError(
            f"no per-run manifest fragments under {directory}. Each run writes one at "
            f"`Layout.submission_manifest_fragment(run_id)` when it serializes; an empty directory "
            f"means no run has serialized yet, not that the family is empty."
        )

    entries: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    collections: dict[str, str] = {}
    modes: dict[str, str] = {}
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        collections[path.stem] = str(doc.get("collection_id", ""))
        modes[path.stem] = str(doc.get("run_mode", ""))
        entries.extend(dict(e) for e in doc.get("submissions", ()))
        runs[path.stem] = {
            "config_hashes": doc.get("config_hashes", {}),
            "coverage": doc.get("coverage", {}),
        }

    for field, seen in (("collection_id", collections), ("run_mode", modes)):
        distinct = sorted(set(seen.values()))
        if len(distinct) > 1:
            raise FamilyManifestError(
                f"fragments disagree on {field}: "
                + ", ".join(f"{k}={v!r}" for k, v in sorted(seen.items()))
                + ". That is a family-wide fact, so these fragments describe two different "
                "experiments and no single manifest can cover them."
            )

    ordered = sorted(entries, key=lambda e: (int(e.get("priority", 0)), str(e.get("track", ""))))
    priority_order: list[str] = []
    for entry in ordered:
        run_id = entry.get("run_id")
        if run_id is not None and run_id not in priority_order:
            priority_order.append(str(run_id))

    return {
        "collection_id": next(iter(collections.values()), ""),
        "run_mode": next(iter(modes.values()), ""),
        "assessment_priority_order": priority_order,
        "runs": dict(sorted(runs.items())),
        "submissions": ordered,
    }
