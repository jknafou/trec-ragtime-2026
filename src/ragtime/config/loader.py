"""The config loader, and the only IO in this package.

Parses ``config/<run>.yml`` with a module-level ``ruamel.yaml`` ``YAML(typ="safe")``
instance, chosen over PyYAML for two stricter behaviours that matter across seven
near-identical hand-maintained files: it raises ``DuplicateKeyError`` where
``yaml.safe_load`` silently keeps the last of a duplicated block, and it defaults to
YAML 1.2, so bare ``no``, ``on`` and ``off`` stay strings. Only reads happen here, so
ruamel's round-trip mode is not needed. ``DuplicateKeyError`` propagates: it is a
launch-gating error.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ruamel.yaml import YAML

from .schema import RunConfig, validate

__all__ = ["load"]

# One shared safe-loader instance; only `.load()` is ever called on it.
_YAML = YAML(typ="safe")


def load(path: str | Path) -> RunConfig:
    """Load, parse, validate and freeze one ``config/<run>.yml`` into a ``RunConfig``.

    Retains the exact UTF-8 ``source_text``, which is the input to ``family_guard``'s
    byte-identical shared-block check, and the ``source_path``. A duplicate top-level
    block raises ``ruamel.yaml.constructor.DuplicateKeyError``, which is neither caught
    nor re-wrapped.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    raw = _YAML.load(text)  # DuplicateKeyError propagates by design
    cfg = validate(raw)
    return dataclasses.replace(cfg, source_path=p, source_text=text)
