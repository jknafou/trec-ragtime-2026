"""The canonicalizing block hasher: a semantic content fingerprint.

``config_hash`` and ``all_hashes`` sit downstream of ``load()``, so their input is the
parsed, defaulted, resolved block and raw-text hashing is impossible here by
construction, the applied defaults not being present in the raw text. The canonical
form reuses the project's JSON byte-stability convention: recursively NFC-normalize
every string leaf, run ``common.io.ensure_finite``, then
``json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":"))`` and sha256.
That is robust to comment, whitespace and key-order reflow, which keeps the
``config_hash``-keyed artifact tree from being invalidated by a cosmetic edit, while a
real value change flips the hash. ``family_guard`` uses the separate raw-text path in
``fairness.py``; the two notions of sameness are not unified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ragtime.common import nfc
from ragtime.common.io import ensure_finite

if TYPE_CHECKING:
    from .schema import RunConfig

__all__ = ["all_hashes", "config_hash"]

_JSON_SEP = (",", ":")


def _canonicalize(obj: Any) -> Any:
    """NFC every string leaf and turn mappings and tuples into JSON-ready dicts and lists."""
    if isinstance(obj, str):
        return nfc(obj)
    if isinstance(obj, Mapping):
        return {k: _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


def config_hash(block: Mapping[str, Any] | Any) -> str:
    """Return the canonical sha256, as full 64-hex, of a parsed config ``block``.

    Recursive NFC, then the finite-float guard, then canonical JSON, so a comment,
    whitespace, quoting or key-order reflow of the same semantic content hashes
    identically while any value change flips the hash.
    """
    canonical = ensure_finite(_canonicalize(block))
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=_JSON_SEP)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def all_hashes(cfg: RunConfig) -> dict[str, str]:
    """Return ``{block_name: config_hash(block)}`` over every top-level block.

    A flat mapping from block name to 64-character lowercase hex, computed once here
    and consumed by the orchestration plan, ``provenance.json`` and the
    ``config_hash``-keyed artifact tree, rather than recomputed per stage. The keys
    always include the shared fairness blocks.
    """
    return {name: config_hash(cfg.blocks[name]) for name in cfg.blocks}
