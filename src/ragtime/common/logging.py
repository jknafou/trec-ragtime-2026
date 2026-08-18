"""Structured logging: one JSON line per event, on the standard library only.

``get_logger(name)`` returns a :class:`StructuredLogger` whose ``debug`` / ``info``
/ ``warning`` / ``error`` methods take a message plus arbitrary keyword fields and
emit one machine-parseable JSON line per record. It wraps the stdlib ``logging``
module, so level filtering and handlers behave as usual.
"""

from __future__ import annotations

import json
import logging
from typing import Any

__all__ = ["StructuredLogger", "get_logger"]

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("ragtime")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    _CONFIGURED = True


class StructuredLogger:
    """Emits ``{"event": msg, <fields>}`` JSON lines at the stdlib log levels."""

    __slots__ = ("_logger",)

    def __init__(self, name: str) -> None:
        _configure_root()
        self._logger = logging.getLogger(f"ragtime.{name}")

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        record = {"event": event, **fields}
        self._logger.log(
            level, json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        )

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)


def get_logger(name: str) -> StructuredLogger:
    """Return a :class:`StructuredLogger` scoped under ``ragtime.<name>``."""
    return StructuredLogger(name)
