"""Reduce the per-run manifest fragments into the single family manifest.

The submission manifest is a family artifact, while ``project()`` is a per-run call, so the reduce
is its own step rather than a side effect of serializing. Each run writes only
``manifest.d/<cell_key>.json``; reducing those into ``submissions/manifest.json`` has to happen
after the writes it covers, and a run that listed the directory before a sibling's fragment landed
would win the rename and drop that sibling permanently. Being a separate, idempotent, total reduce
also means it can be re-run at any time, including by hand immediately before an upload.

The argument is the directory holding ``manifest.d/``, which for the shipped configs is
``<submission_root>/submissions`` and not ``submission_root`` itself::

    python -m ragtime.pipeline.select_serialize.assemble_manifest \\
        --manifest-root /path/to/artifacts/submissions

The reduce is not a node in the job DAG. Three things name it instead: every fragment carries
``family_manifest.command`` generated from its own resolved root, so a reader holding one fragment
is told what is missing; ``project()`` logs ``select_serialize.manifest_fragment`` with the same
command as ``next_step``; and :func:`check`, exposed as ``--check``, answers whether the family
manifest is current without writing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragtime.common.layout import MANIFEST_FRAGMENT_DIR

from .submission.manifest import (
    FamilyManifestError,
    assemble_family_manifest,
    reduce_command,
    write_manifest,
)

__all__ = ["assemble", "check", "main"]


def assemble(manifest_root: str | Path) -> Path:
    """Reduce ``<manifest_root>/manifest.d/*.json`` into ``<manifest_root>/manifest.json``.

    ``manifest_root`` is the directory that holds ``manifest.d``, that is the parent of
    ``Layout.submission_manifest()`` and the common parent of the declared submission outputs. For
    the shipped configs, whose outputs are ``submissions/<track>/run_N.<ext>``, that is
    ``<submission_root>/submissions`` and not ``submission_root`` itself. Passing the wrong one
    finds no fragments rather than failing loudly, so derive it from
    ``root / layout.submission_manifest()`` and take ``.parent`` instead of assuming the names.
    """
    root = Path(manifest_root)
    document = assemble_family_manifest(root / MANIFEST_FRAGMENT_DIR)
    return write_manifest(root / "manifest.json", document)


def check(manifest_root: str | Path) -> list[str]:
    """Report whether ``<manifest_root>/manifest.json`` covers every fragment on disk.

    Returns a list of human-readable problems; an empty list means current. Writes nothing, so it
    is safe to call at any time. The reduce is not a DAG node, so the failure mode it leaves open
    is that nobody ran it, and that is only detectable by asking.

    Staleness is decided by membership rather than by mtime: the family document keys ``runs`` by
    the fragment's stem, so ``set(runs) != {p.stem for p in fragments}`` says exactly which cell is
    described and which is not. An mtime comparison over a shared filesystem written by concurrent
    cells on different nodes would depend on clock skew.
    """
    root = Path(manifest_root)
    fragment_dir = root / MANIFEST_FRAGMENT_DIR
    manifest = root / "manifest.json"
    fragments = sorted(fragment_dir.glob("*.json"))
    problems: list[str] = []
    if not fragments:
        problems.append(
            f"no fragments under {fragment_dir}: no run has serialized yet, so there is nothing "
            "to reduce (this is not the same as an empty family)"
        )
        return problems
    if not manifest.exists():
        problems.append(
            f"{manifest} does not exist: {len(fragments)} fragment(s) are on disk and the family "
            f"manifest was never assembled. Run: {reduce_command(root)}"
        )
        return problems
    try:
        described = set(json.loads(manifest.read_text(encoding="utf-8")).get("runs", {}))
    except (OSError, ValueError) as exc:
        problems.append(f"{manifest} is not readable as a family manifest: {exc}")
        return problems
    on_disk = {p.stem for p in fragments}
    if described != on_disk:
        missing = sorted(on_disk - described)
        extra = sorted(described - on_disk)
        problems.append(
            f"{manifest} is stale: it describes {sorted(described)}, the fragments on disk are "
            f"{sorted(on_disk)}"
            + (f"; not described: {missing}" if missing else "")
            + (f"; described but gone: {extra}" if extra else "")
            + f". Re-run: {reduce_command(root)}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ragtime.pipeline.select_serialize.assemble_manifest")
    ap.add_argument(
        "--manifest-root",
        required=True,
        help=(
            "the directory holding manifest.d/, the common parent of the declared submission "
            "paths, i.e. <submission_root>/submissions for the shipped configs, not submission_root"
        ),
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help=(
            "do not write: report whether manifest.json exists and covers every fragment, and "
            "exit 3 if it does not. Run this before an upload"
        ),
    )
    args = ap.parse_args(argv)
    if args.check:
        problems = check(args.manifest_root)
        for problem in problems:
            # A family manifest nobody assembled must not read as a successful run.
            print(f"NOT CURRENT: {problem}", file=sys.stderr)
        if problems:
            return 3
        print(f"current: {Path(args.manifest_root) / 'manifest.json'}")
        return 0
    try:
        path = assemble(args.manifest_root)
    except FamilyManifestError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
