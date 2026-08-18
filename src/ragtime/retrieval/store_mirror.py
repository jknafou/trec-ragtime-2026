"""Serve the by-id passage store from a staged mirror: the root redirects, the path re-resolves.

The depth-100 passage fetch the reranker blocks on is the largest single cost on the query path,
and it is a filesystem read rather than a model. Off shared storage its tail is seconds per fetch,
because nothing pins the store in page cache and any eviction puts the next query back on the cold
path; off a node-local tmpfs copy the tail collapses to milliseconds, since tmpfs pages are
unevictable. ``execution.passage_store_mirror_root`` is how a scored run asks for the staged copy.

Four properties:

1. The root is redirected and the path is re-resolved, never a free-form path to a ``data.mdb``.
   ``Layout.passage_store_path(recon_hash, pack_hash)`` takes ``pack_hash`` as a
   required-but-nullable argument so no caller can silently address a differently-packed table,
   and this is the one call site where being wrong is silent: the three renderings share passage
   ids, so a store built from the wrong packing answers every id and errors nowhere. Hence
   ``mirrored = mirror_root / origin.relative_to(layout.base)``.
2. Verify, then fall back loudly. The mirror is accepted only when it exists, carries its own
   ``_SUCCESS`` marker, and its ``data.mdb`` has the same size as the origin's. Any other outcome
   opens the origin and emits ``retrieval.store_mirror_refused`` plus a warning carrying the
   reason, because a silent downgrade would leave nothing in the record to explain the slowdown.
3. It is declared in ``execution``, not in a hashed block. The mirror changes no returned byte
   (the same LMDB, the same keys, the same values, a different inode), so it must not enter
   ``config_hash`` or re-key an artifact. ``config.fairness.family_guard`` does not compare
   ``execution``, which is why the resolved path is written into the run record as a witness.
4. Staging itself is not here. This module only decides which of the two copies gets opened; the
   copy is made at service bring-up, inside the allocation, before the first query, because tmpfs
   dies with the node.

For a mirror to be accepted the store must sit under ``<mirror_root>/`` at the same relative path
it has under the artifact root, with its marker::

    <mirror_root>/corpus/<family>/<chunker12>/<node>/final/<recon12>/passages/<pack12>/passages.lmdb
    <...>/passages.lmdb._SUCCESS

Copying only ``data.mdb`` and forgetting the marker is a refusal, which is the half-written case
property 2 exists for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common.io import is_done

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ragtime.common import Layout

__all__ = [
    "PASSAGE_STORE_MIRROR_ROOT_KEY",
    "STORE_DATA_FILE",
    "StoreLocation",
    "mirror_root",
    "resolve_store_location",
]

#: The ``execution`` key naming the root a staged passage store is served from: a root, never a
#: store path.
PASSAGE_STORE_MIRROR_ROOT_KEY = "passage_store_mirror_root"

#: LMDB's data file inside the store directory (``common.io.lmdb_open`` opens with
#: ``subdir=True``). Its size is the cheapest whole-store integrity witness available that does
#: not re-read the whole table.
STORE_DATA_FILE = "data.mdb"


@dataclass(frozen=True, slots=True)
class StoreLocation:
    """Which copy of the by-id store was opened, and why. A run-record witness.

    ``path`` is what was actually opened; ``origin`` is always the canonical ``Layout``-resolved
    path. ``mirrored`` is true only when a mirror was offered and passed verification.
    ``refusal`` is set exactly when a mirror was offered and rejected, and carries the reason.
    """

    origin: Path
    path: Path
    mirrored: bool = False
    refusal: str | None = None

    @property
    def witness(self) -> dict[str, Any]:
        """The per-search record's store field, beside the transport witness."""
        out: dict[str, Any] = {"path": str(self.path), "mirrored": self.mirrored}
        if self.refusal is not None:
            out["refusal"] = self.refusal
        return out


def _execution(cfg: Any) -> dict[str, Any]:
    return dict((getattr(cfg, "blocks", None) or {}).get("execution") or {})


def mirror_root(cfg: Any) -> Path | None:
    """``execution.passage_store_mirror_root`` as a path, or ``None`` when unset.

    An empty string is the same statement as absent (a config template that leaves the key in
    place with no value must not resolve to the filesystem root), and both mean "open the
    canonical store", which is the shipped default.
    """
    raw = _execution(cfg).get(PASSAGE_STORE_MIRROR_ROOT_KEY)
    if raw is None:
        return None
    text = str(raw).strip()
    return Path(text) if text else None


def _relative_to_base(layout: Layout, origin: Path) -> Path | None:
    """``origin``'s path relative to the artifact root, or ``None`` if it is not beneath it."""
    try:
        return origin.relative_to(Path(layout.base))
    except ValueError:
        return None


def _size_of(store_dir: Path) -> int | None:
    try:
        return (store_dir / STORE_DATA_FILE).stat().st_size
    except OSError:
        return None


def _verify(origin: Path, candidate: Path) -> str | None:
    """``None`` when ``candidate`` may be served in place of ``origin``, else why it may not.

    Three checks in the order that makes the message useful: presence, publication, then size.
    The size comparison is against the origin rather than a recorded constant, because the origin
    is the artifact the run is entitled to read, and it costs two ``stat`` calls.
    """
    if not candidate.exists():
        return f"mirror {candidate} does not exist"
    if not is_done(candidate):
        return (
            f"mirror {candidate} carries no _SUCCESS marker; a staged copy is not published "
            "until its stager says so, and a half-written store is indistinguishable from a "
            "whole one by path"
        )
    mirror_size = _size_of(candidate)
    if mirror_size is None:
        return f"mirror {candidate} has no readable {STORE_DATA_FILE}"
    origin_size = _size_of(origin)
    if origin_size is None:
        return f"origin {origin} has no readable {STORE_DATA_FILE} to size the mirror against"
    if mirror_size != origin_size:
        return (
            f"mirror {candidate}/{STORE_DATA_FILE} is {mirror_size} bytes but the origin is "
            f"{origin_size}; a truncated or differently-built store answers every passage id "
            "and errors nowhere, so it is refused rather than served"
        )
    return None


def resolve_store_location(
    layout: Layout,
    recon_hash: str,
    pack_hash: str | None,
    *,
    root: Path | None = None,
) -> StoreLocation:
    """Which by-id store to open: the canonical one, or a verified mirror of it.

    ``root`` is :func:`mirror_root`'s answer. ``None`` (the shipped default) returns the canonical
    location untouched, so this costs one ``is None`` on every path that does not stage.

    This function never raises and never falls back silently: an unusable mirror comes back as the
    origin plus a ``refusal`` string, and the caller counts and logs it. Whether the origin itself
    is published is ``context._open_store``'s check, applied before either copy is opened, so a
    mirror can speed a published store up but never substitute for one that was never built.
    """
    origin = Path(layout.passage_store_path(recon_hash, pack_hash))
    if root is None:
        return StoreLocation(origin=origin, path=origin)
    relative = _relative_to_base(layout, origin)
    if relative is None:
        return StoreLocation(
            origin=origin,
            path=origin,
            refusal=(
                f"the store path {origin} is not beneath the artifact root {layout.base}, so "
                "the recon12/pack12 keying cannot be re-resolved under the mirror root, and "
                "a mirror addressed any other way could serve a differently-packed table"
            ),
        )
    candidate = Path(root) / relative
    if candidate == origin:
        return StoreLocation(origin=origin, path=origin)
    refusal = _verify(origin, candidate)
    if refusal is not None:
        return StoreLocation(origin=origin, path=origin, refusal=refusal)
    return StoreLocation(origin=origin, path=candidate, mirrored=True)
