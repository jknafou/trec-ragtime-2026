"""The official ``hltcoe/rag-run-validator`` as a hard gate on every emitted submission.

Invoking an external, pinned tool as a subprocess is designed once, here, rather than re-invoked ad
hoc from several call sites. The validator is pinned at commit
``eb0811b229736746306d234081b348f77b2f646b`` (tag ``v0.1``); its only dependency is
``jsonschema>=4.0.0``, which the project environment already has, so it runs from a clone on
``PYTHONPATH`` with nothing added to the lock file.

Six properties of the tool that a naive wrapper gets wrong, each read out of its source or
demonstrated against it:

1. ``python -m rag_run_validator`` does not work: the package ships no ``__main__.py``, so ``-m``
   exits 1. A reject suite driven that way reports every deliberately broken file as correctly
   rejected while the validator never ran. This module invokes ``python -c`` on its ``cli`` entry
   point instead. The way to tell the two apart is to check that an accept case returns 0, not
   only that a reject returns 1 -- which is how these six properties were established by hand. No
   test in this repository runs the validator subprocess: the tool is an external clone that is
   not vendored here, so nothing standing enforces the six below.
2. ``--topics`` is mandatory. Without it an empty run file exits 0, because the missing-topics
   check is the only thing that makes the validator capable of failing on absent content.
3. Passing the shipped topics file directly crashes it. Its reader is ``map(json.loads, f)``, line
   by line, while the shipped file is TREC's concatenated single-line form with no newlines, giving
   an uncaught ``JSONDecodeError`` that also exits 1 and so looks like a clean rejection.
   :func:`normalize_topics` writes a true one-object-per-line copy first; the source file is never
   touched.
4. The same reader reads the run file, so a blank line or concatenated style crashes it
   identically. Crash and verdict are distinguished: a real rejection prints ``[Error`` on stdout
   while a crash writes a ``Traceback`` to stderr, and a traceback raises
   :class:`ValidatorHarnessError` rather than being reported as a format verdict.
5. Success prints nothing; silence with return code 0 is the pass signal.
6. It does not cover Task 2, since ``--format`` accepts only ``report`` and ``nuggets``. Pointing it
   at a TREC run is a category error, so :func:`validate` refuses it by name.

If the tool cannot be resolved at all, :func:`validate` returns a verdict with ``ran=False`` and the
manifest records that it did not run, never that it passed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ragtime.common.topics import load_topics

__all__ = [
    "FORMAT_NUGGETS",
    "FORMAT_REPORT",
    "VALIDATOR_COMMIT",
    "VALIDATOR_HOME_ENV",
    "SubmissionRejected",
    "ValidatorHarnessError",
    "Verdict",
    "normalize_topics",
    "validate",
    "validator_home",
]

#: The pinned commit, stated in the verdict and the manifest so a submission records which
#: validator accepted it.
VALIDATOR_COMMIT = "eb0811b229736746306d234081b348f77b2f646b"
#: Where the clone is expected. It is not vendored into this repository; obtain it separately.
VALIDATOR_HOME_ENV = "RAGTIME_RRV_HOME"
_DEFAULT_HOME = Path("tools") / "rag-run-validator"

FORMAT_REPORT = "report"
FORMAT_NUGGETS = "nuggets"

# src/ragtime/pipeline/select_serialize/submission/validate.py -> parents[5] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[5]

#: The single supported way to start the tool; see point 1 of the module docstring.
_ENTRY = "from rag_run_validator import cli; cli()"


class SubmissionRejected(RuntimeError):
    """The validator rejected the file, so nothing is published."""


class ValidatorHarnessError(RuntimeError):
    """The validator crashed or could not be started, which is not a format verdict."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the validator said, verbatim, plus whether it ran at all."""

    path: str
    fmt: str
    ran: bool
    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    command: tuple[str, ...]
    commit: str = VALIDATOR_COMMIT

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "format": self.fmt,
            # "not run" is a third state, never collapsed into "failed" or "passed".
            "verdict": "passed" if self.ok else ("not run: tool unavailable" if not self.ran else "failed"),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "command": list(self.command),
            "validator_commit": self.commit,
        }


def validator_home() -> Path | None:
    """Return the validator clone, or ``None`` when it is not on this machine."""
    env = os.environ.get(VALIDATOR_HOME_ENV)
    candidates = [Path(env)] if env else []
    candidates.append(_REPO_ROOT / _DEFAULT_HOME)
    for candidate in candidates:
        if (candidate / "rag_run_validator" / "__init__.py").exists():
            return candidate
    return None


def normalize_topics(topics_path: str | Path, dest: str | Path) -> Path:
    """Re-emit the topics file as newline-delimited JSON at ``dest``.

    Reading goes through ``common.topics.load_topics``, the one place TREC's concatenated
    single-line quirk is absorbed, so the source file is never edited and the parser is not
    reimplemented here. The validator reads only ``topic_id`` and ``limit``, but all six fields are
    re-emitted so the copy is faithful rather than a validator-shaped subset. Later topics files are
    already strict JSONL, which makes this an idempotent re-emit; it stays because the contract is
    that the copy the validator reads is one we produced.
    """
    import json

    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "topic_id": t.topic_id,
                "collection_id": t.collection_id,
                "title": t.title,
                "problem_statement": t.problem_statement,
                "background": t.background,
                "limit": t.limit,
            },
            ensure_ascii=False,
        )
        for t in load_topics(topics_path)
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def validate(
    path: str | Path,
    fmt: str,
    *,
    topics_path: str | Path,
    strict_on_length: bool = True,
    timeout_s: float = 300.0,
) -> Verdict:
    """Run the validator on ``path`` and return a :class:`Verdict`, publishing nothing.

    ``topics_path`` is the normalized topics file from :func:`normalize_topics` and is mandatory.
    Raises :class:`ValidatorHarnessError` if the tool crashed, so a traceback cannot be mistaken
    for a clean rejection.
    """
    if fmt not in (FORMAT_REPORT, FORMAT_NUGGETS):
        raise ValueError(
            f"format must be {FORMAT_REPORT!r} or {FORMAT_NUGGETS!r}; got {fmt!r}. There is no "
            "Task-2 validator: the tool covers no part of a TREC run file."
        )
    run_path = Path(path)
    home = validator_home()
    argv = (
        sys.executable, "-c", _ENTRY, str(run_path),
        "--format", fmt, "--topics", str(topics_path),
    )
    if strict_on_length and fmt == FORMAT_REPORT:
        # This changes no measurement; it only turns a `[Warning]` with rc 0 into an `[Error]`
        # with rc 1. Length is never checked for `--format nuggets`.
        argv = (*argv, "--strict_on_length")

    if home is None:
        return Verdict(
            path=str(run_path), fmt=fmt, ran=False, ok=False, returncode=None,
            stdout="", stderr=(
                f"validator not found; set ${VALIDATOR_HOME_ENV} or clone "
                f"hltcoe/rag-run-validator at {VALIDATOR_COMMIT} to {_DEFAULT_HOME}"
            ),
            command=argv,
        )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(home), env.get("PYTHONPATH", "")]))
    proc = subprocess.run(
        argv, capture_output=True, text=True, env=env, timeout=timeout_s, check=False
    )
    if "Traceback (most recent call last)" in proc.stderr:
        raise ValidatorHarnessError(
            f"the validator crashed on {run_path} (exit {proc.returncode}); this is a harness "
            f"failure, not a format verdict:\n{proc.stderr}"
        )
    return Verdict(
        path=str(run_path), fmt=fmt, ran=True, ok=proc.returncode == 0,
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, command=argv,
    )
