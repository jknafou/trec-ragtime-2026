"""Topics loader, tolerant of the two shapes the released topics file takes.

The canonical request set is ``topics/topics.all.2026.v0625-fix.jsonl``, strict
one-object-per-line JSONL carrying a ``title`` field. Its superseded predecessor
is a single physical line holding 103 JSON objects back to back with no
separators. Source files are never edited; instead ``load_topics`` walks the
buffer with ``json.JSONDecoder().raw_decode`` in a whitespace-skipping loop, so
both shapes read identically. Topic text is taken verbatim, with no Unicode
normalization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CANONICAL_TOPICS_REL", "Topic", "TopicsError", "load_topics", "parse_topics"]

#: The canonical request set, as a repo-relative path; resolving it against a root
#: is the caller's decision. A run normally takes its topics file from its own
#: config (``topics.path``); this constant is the default for tools that have no
#: config to read.
CANONICAL_TOPICS_REL = "topics/topics.all.2026.v0625-fix.jsonl"


class TopicsError(ValueError):
    """Raised when a topics file cannot supply a complete request."""


@dataclass(frozen=True, slots=True)
class Topic:
    """One report request (topic). Fields per decompose I/O schema (input)."""

    topic_id: str
    collection_id: str
    title: str
    problem_statement: str
    background: str
    limit: int


def _topic_from_obj(obj: dict) -> Topic:
    """Build a :class:`Topic` from one decoded JSON object.

    Fields are picked explicitly rather than by ``Topic(**obj)``, so an added
    upstream key is a deliberate schema change here. ``title`` is required and its
    absence is a hard error rather than a default: the decompose prompt reads it, so
    an empty default would silently truncate the request. Only the superseded pre-fix
    topics file can trigger that, so the message names it; the remaining five keys
    are present in both releases and keep a bare ``KeyError``.
    """
    if "title" not in obj:
        raise TopicsError(
            f"topic {obj.get('topic_id', '<no topic_id>')!r} has no 'title'; this is the "
            f"superseded pre-fix topics file. Use {CANONICAL_TOPICS_REL} (or point the run's "
            f"`topics.path` at it): `title` is part of the request and must not default to ''."
        )
    return Topic(
        topic_id=obj["topic_id"],
        collection_id=obj["collection_id"],
        title=obj["title"],
        problem_statement=obj["problem_statement"],
        background=obj["background"],
        limit=obj["limit"],
    )


def parse_topics(text: str) -> list[Topic]:
    """Parse concatenated JSON objects from ``text`` into :class:`Topic`s.

    Handles both the real single-line concatenated form and ordinary
    newline-delimited JSONL (whitespace, including newlines, between objects is
    simply skipped).
    """
    decoder = json.JSONDecoder()
    topics: list[Topic] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = decoder.raw_decode(text, i)
        topics.append(_topic_from_obj(obj))
        i = end
    return topics


def load_topics(path: str | Path) -> list[Topic]:
    """Load all topics from ``path`` (the source file is never modified)."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_topics(text)
