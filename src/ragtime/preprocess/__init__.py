"""Offline, work-queued corpus builds: the raw collection into three indexed renderings.

Nine substages run in a fixed order, each of them a pull work queue::

    chunk -> merge -> translate_omt_identity -> translate_omt -> reconcile
          -> len_max -> packing -> vectorize -> assemble

``download`` fetches the raw corpus byte-exactly and ``chunk`` mints the native
sentence-id spine (deterministic SaT segmentation into a ``documents.parquet`` +
``sentences.parquet`` pair, with sentence text stored once as a span of its
document). ``merge`` maps short sentences onto translation units, ``translate_omt``
produces the raw English of both translated renderings, and ``reconcile`` decides per
unit whether the ``§`` split survived, fuses the unit where it did not, renumbers each
document densely and scrubs the marker.

Reconciliation settles which sentences exist and stops there. ``packing`` is the sole
producer of a passage: it groups the reconciled sentences against the per-sentence length
sidecar ``len_max`` builds, so passage boundaries are identical in all three renderings
and one passage id addresses the same content whichever text is read. ``vectorize`` and
``assemble`` then encode those passages and build the dense, sparse and late-interaction
legs. Everything after reconciliation reads its ``final/<recon12>/`` node rather than the
spine or the raw translation table, and everything after packing addresses passages by the
ids minted there.

All model access flows through ``serving``, paths through ``common.Layout`` and
ids through ``common.ids``. Depends on ``common``, ``config``, ``serving`` and
``orchestration`` only.
"""

from __future__ import annotations

from .chunk import ChunkAdapter, chunk, chunk_document, segment_document, segment_documents
from .corpus import lang_of, read_native, read_trans
from .download import download
from .merge import MergeAdapter, merge_document, merge_short_sentences
from .merge_join import MARKER, join_constituents, scrub_markers, split_translation
from .reconcile import ReconcileAdapter, build_document, reconcile_document, reconcile_hash
from .tokenizer import PackTokenizer, load_pack_tokenizer
from .translate_omt import OmtIdentityAdapter, OmtTranslateAdapter

__all__ = [
    "MARKER",
    "ChunkAdapter",
    "MergeAdapter",
    "OmtIdentityAdapter",
    "OmtTranslateAdapter",
    "PackTokenizer",
    "ReconcileAdapter",
    "build_document",
    "chunk",
    "chunk_document",
    "download",
    "join_constituents",
    "lang_of",
    "load_pack_tokenizer",
    "merge_document",
    "merge_short_sentences",
    "read_native",
    "read_trans",
    "reconcile_document",
    "reconcile_hash",
    "scrub_markers",
    "segment_document",
    "segment_documents",
    "split_translation",
]
