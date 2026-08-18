"""The NIST envelope: ``team_id`` / ``run_id`` / ``run_desc`` / the assessment priority.

Every value comes from the ``config.outputs`` entry whose ``track`` matches, so the emitted metadata
and the reproducible record cannot disagree.

``run_id <= 25`` is enforced here, before anything is written. The validator's asymmetry makes that
necessary: the nuggets schema sets ``maxLength: 25`` and rejects a 26-character ``run_id`` under
``--format nuggets``, while the default report schema has no such constraint and the same 26
characters pass. The Track Guidelines require the cap for both, so on Tasks 1 and 2 this is the only
check that exists.

Assessment priority is read from the output path rather than from a second source: the file name
encodes it, ``run_1`` being priority 1, and only the first few runs are guaranteed to be pooled and
scored. A separate priority field in the config would be free to disagree with the file name NIST
actually sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..knobs import RUN_ID_MAX

__all__ = ["Envelope", "EnvelopeError", "envelopes"]

#: ``submissions/<track>/run_<N>.<ext>`` -> the priority N.
_RUN_N = re.compile(r"run_(\d+)\.")


class EnvelopeError(ValueError):
    """The outputs entry is missing, or its ``run_id`` would get the run rejected."""


@dataclass(frozen=True, slots=True)
class Envelope:
    """One emitted deliverable's identity, resolved from one ``config.outputs`` entry."""

    track: str
    task: int
    path: str
    team_id: str
    run_id: str
    run_desc: str
    priority: int


def envelopes(cfg: Any) -> tuple[Envelope, ...]:
    """Return every declared deliverable of this run, in ``config.outputs`` order."""
    return tuple(_from_entry(cfg, entry) for entry in (getattr(cfg, "outputs", ()) or ()))


def _from_entry(cfg: Any, entry: Any) -> Envelope:
    run_id = str(entry["run_id"])
    if len(run_id) > RUN_ID_MAX:
        raise EnvelopeError(
            f"run_id {run_id!r} is {len(run_id)} characters; the Track Guidelines cap it at "
            f"{RUN_ID_MAX}. The nuggets schema enforces this and the report schema does not, so "
            "an over-long run_id would pass validation as a report and be rejected as nuggets "
            "from the same run."
        )
    team_id = str(getattr(cfg, "team_id", "") or "")
    if not team_id:
        raise EnvelopeError("config carries no team_id; every submission line requires one")
    path = str(entry["path"])
    match = _RUN_N.search(path)
    return Envelope(
        track=str(entry["track"]),
        task=int(entry["task"]),
        path=path,
        team_id=team_id,
        run_id=run_id,
        run_desc=str(entry.get("run_desc", "") or ""),
        priority=int(match.group(1)) if match else 0,
    )
