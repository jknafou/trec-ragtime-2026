"""Pins for reconciliation's final corpus node.

``common`` owns the paths and the pinned Arrow schemas because producer and consumers
live in different folders: ``preprocess.reconcile`` writes these four tables and the
index build, the passage store and retrieval read them. A hand-rolled schema or a
hand-built path on either side is how a column silently changes type, or how a v1
inventory ends up sharing a directory with a v2 translation.

They pin properties rather than the digests themselves: a digest pin would break on any
legitimate config edit, whereas "the key moves when input X moves" is the property that
protects the artifact tree.
"""

from __future__ import annotations

import types

import pytest

from ragtime.common import Layout
from ragtime.common.schemas import (
    final_passage_arrow_schema,
    passage_arrow_schema,
    sentence_arrow_schema,
    sentence_remap_arrow_schema,
    translation_final_arrow_schema,
)

pytestmark = pytest.mark.small

_FAMILY = "e2e"
_CHUNKER = "c" * 64
_RECON = "9f3a1b2c4d5e6f70" + "0" * 48


def _layout(tmp_path):
    return Layout(run_dir=tmp_path, base=tmp_path, family=_FAMILY, chunker_hash=_CHUNKER)


# --------------------------------------------------------------------------- #
# Paths.
# --------------------------------------------------------------------------- #
def test_every_final_artifact_hangs_off_the_one_composite_keyed_node(tmp_path):
    layout = _layout(tmp_path)
    node = layout.final_dir(_RECON)
    assert node.name == _RECON[:12]
    assert node.parent.name == "final"
    assert node.parent.parent == layout.corpus_dir(_FAMILY, _CHUNKER)

    assert layout.final_sentences_path(_RECON) == node / "sentences.parquet"
    assert layout.final_passages_path(_RECON, None) == node / "passages.parquet"
    assert layout.final_translations_path(_RECON, "omt") == node / "translations/omt.parquet"
    assert layout.sentence_remap_path(_RECON) == node / "remap.parquet"
    assert layout.final_manifest_path(_RECON) == node / "manifest.json"


def test_final_translations_are_a_different_path_from_the_raw_ones(tmp_path):
    """Two names, two meanings: nothing can read the marker-bearing table by accident."""
    layout = _layout(tmp_path)
    final = layout.final_translations_path(_RECON, "omt")
    raw = layout.translations_raw_path("omt", "t" * 64, "m" * 64)
    assert final != raw
    assert "translations_raw" in str(raw)
    assert "translations_raw" not in str(final)


def test_a_different_reconcile_key_is_a_different_directory(tmp_path):
    """The composite key exists so that an incoherent pairing is unaddressable."""
    layout = _layout(tmp_path)
    other = "a" * 64
    assert layout.final_dir(_RECON) != layout.final_dir(other)
    assert layout.final_sentences_path(_RECON) != layout.final_sentences_path(other)


# --------------------------------------------------------------------------- #
# Schemas.
# --------------------------------------------------------------------------- #
def test_final_sentences_reuse_the_spine_schema_rather_than_cloning_it():
    """A fused sentence is a first-class sentence: same columns, same contract."""
    assert sentence_arrow_schema().names == [
        "sentence_id",
        "document_id",
        "sentence_index",
        "lang",
        "start",
        "end",
        "paragraph_index",
        "token_count",
    ]


def test_the_stored_passage_table_is_text_free():
    """Persisting passage text would put a second copy of the corpus on disk.

    Distinct from ``passage_arrow_schema``, which mirrors the in-memory record the packer
    returns; on disk a passage is an ordered list of sentence ids and nothing else.
    """
    names = final_passage_arrow_schema().names
    assert names == [
        "passage_id",
        "document_id",
        "lang",
        "sentence_ids",
        "token_count",
        "is_oversized",
        "paragraph_index",
    ]
    assert "text" not in names
    assert "sentence_char_spans" not in names
    assert "text" in passage_arrow_schema().names  # the in-memory shape is untouched


def test_final_translations_are_keyed_by_sentence_and_carry_no_raw_column():
    names = translation_final_arrow_schema().names
    assert names == ["sentence_id", "document_id", "variant", "text", "source_lang"]
    assert "text_raw" not in names  # `text_raw` means "may contain a marker"
    assert "passage_id" not in names


def test_remap_is_the_audit_trail_of_an_otherwise_irreversible_renumbering():
    assert sentence_remap_arrow_schema().names == [
        "sentence_id",
        "final_sentence_id",
        "document_id",
        "fused",
    ]


@pytest.mark.parametrize(
    ("schema_fn", "column", "arrow_type"),
    [
        (final_passage_arrow_schema, "sentence_ids", "list<item: string>"),
        (final_passage_arrow_schema, "is_oversized", "bool"),
        (final_passage_arrow_schema, "token_count", "int32"),
        (translation_final_arrow_schema, "text", "string"),
        (sentence_remap_arrow_schema, "fused", "bool"),
    ],
)
def test_column_types_are_pinned(schema_fn, column, arrow_type):
    schema = schema_fn()
    assert str(schema.field(column).type) == arrow_type


# --------------------------------------------------------------------------- #
# The composite key itself.
# --------------------------------------------------------------------------- #
def _cfg(**overrides):
    blocks = {
        "chunker": {"config": {"token_budget": 512}, "hash": ""},
        "merge": {"min_sentence_tokens": 16},
        "translation": {"config": {"beam_size": 4}, "hash": ""},
        "reconcile": {"split_back": True, "min_segment_chars": 1},
    }
    blocks.update(overrides)
    return types.SimpleNamespace(run_id="e2e-omt", blocks=blocks)


def test_reconcile_hash_is_a_function_of_all_four_inputs_and_only_those():
    from ragtime.preprocess.reconcile import reconcile_hash

    base = reconcile_hash(_cfg())
    assert reconcile_hash(_cfg()) == base  # deterministic
    # An unrelated block must not move it: the key names its real dependencies.
    assert reconcile_hash(_cfg(llm={"model": "whatever"})) == base
    # Each of the four inputs must.
    assert reconcile_hash(_cfg(chunker={"config": {"token_budget": 256}, "hash": ""})) != base
    assert reconcile_hash(_cfg(merge={"min_sentence_tokens": 8})) != base
    assert reconcile_hash(_cfg(translation={"config": {"beam_size": 1}, "hash": ""})) != base
    assert reconcile_hash(_cfg(reconcile={"split_back": False, "min_segment_chars": 1})) != base


def test_reconcile_hash_uses_the_one_canonical_hasher():
    """Full 64-character lowercase hex, like every other block hash."""
    from ragtime.preprocess.reconcile import reconcile_hash

    h = reconcile_hash(_cfg())
    assert len(h) == 64
    assert h == h.lower()
    assert all(c in "0123456789abcdef" for c in h)
