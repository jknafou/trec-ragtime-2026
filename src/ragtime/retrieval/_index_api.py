"""The single import site for ``preprocess.index``'s query surface.

The dependency spine makes ``preprocess`` and ``retrieval`` siblings, so ``retrieval`` does not
import ``preprocess`` in general. The index query surface is the one narrow exception: it is a
shared concern (``preprocess.acceptance`` consumes four of the same symbols) and it is already
Protocol-shaped, so concrete-engine coupling is confined to the :func:`default_legs` factory.

The exception is exactly twelve symbols wide and lives in this one file so its scope can be
checked by inspection. The build side of ``preprocess`` (``vectorize``, ``assemble``, ``chunk``,
``merge``, ``translate_omt``, ``reconcile``, ``packing``) is never importable through here; a
query-time module needing one of those is asking for something the index artifact should carry.

The corpus keys ``recon_hash`` and ``pack_hash`` are not in the allowlist, so
:func:`ragtime.retrieval.context.bring_up` takes both as required keyword arguments from its
caller rather than re-deriving them.
"""

from __future__ import annotations

from ragtime.preprocess.index import (
    IndexCtx,
    IndexIntegrityError,
    LangHandle,
    Leg,
    LegHandle,
    default_legs,
    index_build_options,
    index_hash,
    leg_config_hash,
    open_lang,
    query_lang_leg,
    read_shard_parts,
)

#: The allowlist as data, so a test can compare it against this module's ``__all__`` and against
#: what ``ragtime/retrieval/`` actually imports.
ALLOWED_INDEX_SYMBOLS: frozenset[str] = frozenset(
    {
        "IndexCtx",
        "IndexIntegrityError",
        "LangHandle",
        "Leg",
        "LegHandle",
        "default_legs",
        "index_build_options",
        "index_hash",
        "leg_config_hash",
        "open_lang",
        "query_lang_leg",
        "read_shard_parts",
    }
)

#: The build-side modules this exception must never grow to cover, named as data so a test can
#: assert no module under ``ragtime/retrieval/`` imports any of them.
FORBIDDEN_INDEX_MODULES: frozenset[str] = frozenset(
    {
        "ragtime.preprocess.assemble",
        "ragtime.preprocess.chunk",
        "ragtime.preprocess.merge",
        "ragtime.preprocess.packing",
        "ragtime.preprocess.reconcile",
        "ragtime.preprocess.translate_omt",
        "ragtime.preprocess.vectorize",
    }
)

__all__ = [
    "ALLOWED_INDEX_SYMBOLS",
    "FORBIDDEN_INDEX_MODULES",
    "IndexCtx",
    "IndexIntegrityError",
    "LangHandle",
    "Leg",
    "LegHandle",
    "default_legs",
    "index_build_options",
    "index_hash",
    "leg_config_hash",
    "open_lang",
    "query_lang_leg",
    "read_shard_parts",
]
