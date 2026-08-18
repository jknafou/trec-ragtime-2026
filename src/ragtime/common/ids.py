"""ID algebra: the join keys shared across the three corpus renderings.

Passage, sentence and document ids are identical across the Original, Organizer
and OMT renderings, so these mint/parse helpers are the single home for the
citation join keys. ``doc_id_of`` resolves any id back to the original document
id, whichever rendering was read or searched.

    document_id   "eng-docs/0007421"      (never contains '#')
    sentence_id   "eng-docs/0007421#s0"   <doc>#s{j}
    passage_id    "eng-docs/0007421#p0"   <doc>#p{k}
    nugget_id     "2061#n03"              <topic>#n{j}

Sentences and passages are siblings under the document rather than nested: a
sentence is a span of its document, and under passage overlap the same sentence
appears in two adjacent passages. Both suffixes are first-level, so ``doc_id_of``
is a single split on the stored string.
"""

from __future__ import annotations

__all__ = ["doc_id_of", "nugget_id", "passage_id", "sentence_id"]

_SEP = "#"


def doc_id_of(ref_id: str) -> str:
    """Strip the ``#p{k}`` / ``#s{j}`` suffix, returning the original doc-id.

    Splits on the first ``#`` only, so a passage id and a sentence id both map
    back to the same doc-id. Document ids never contain ``#``, so this round-trips
    for both minters:
    ``doc_id_of(passage_id(d, k)) == doc_id_of(sentence_id(d, j)) == d``.
    """
    return ref_id.split(_SEP, 1)[0]


def passage_id(doc_id: str, k: int) -> str:
    """Mint ``<doc_id>#p{k}`` - the 0-based, doc-scoped passage id."""
    return f"{doc_id}{_SEP}p{k}"


def sentence_id(doc_id: str, j: int) -> str:
    """Mint ``<doc_id>#s{j}`` - the 0-based, document-scoped sentence id.

    Scoped to the document rather than the passage: the id is minted once at
    segmentation time, before packing exists, so a sentence carried into an
    adjacent passage as overlap keeps the same id.
    """
    return f"{doc_id}{_SEP}s{j}"


def nugget_id(topic_id: str, j: int) -> str:
    """Mint ``<topic_id>#n{j}`` - the nugget join key, stable across rounds."""
    return f"{topic_id}{_SEP}n{j}"
