"""Artifact-shape and path pins for the merge map and the raw translation table.

``merge`` writes the merge map and ``translate_omt`` reads it; ``translate_omt`` writes
``translations_raw`` and reconciliation reads it. Producer and consumer are different
stages, so the schema field names, their order, their widths and the path each table
resolves to are pinned here: a hand-rolled schema or a hand-built filename on either
side is how a column silently changes type or an artifact silently moves.
"""

from __future__ import annotations

import pytest

from ragtime.common import Layout
from ragtime.common.schemas import merge_map_arrow_schema, translation_raw_arrow_schema

pytestmark = pytest.mark.small


def test_merge_map_schema_fields_and_order_are_pinned():
    schema = merge_map_arrow_schema()
    assert schema.names == [
        "sentence_id",
        "document_id",
        "sentence_index",
        "lang",
        "start",
        "end",
        "merge_unit_id",
        "merge_constituent_index",
        "merge_constituent_count",
        "merge_unit_token_count",
    ]
    # int32 across every offset/count column: the sentence tables are the project's largest
    # row count and the longest real document is 23,917 code points (~90,000x margin).
    for name in (
        "sentence_index",
        "start",
        "end",
        "merge_constituent_index",
        "merge_constituent_count",
        "merge_unit_token_count",
    ):
        assert str(schema.field(name).type) == "int32", name


def test_merge_unit_token_count_makes_reconciliation_tokenizer_free():
    """Reconciliation decides fuse vs split-back from a stored number, not by re-tokenizing."""
    assert "merge_unit_token_count" in merge_map_arrow_schema().names


def test_translations_raw_schema_fields_and_order_are_pinned():
    schema = translation_raw_arrow_schema()
    assert schema.names == [
        "sentence_id",
        "document_id",
        "variant",
        "text_raw",
        "source_lang",
        "merge_unit_id",
        "merge_constituent_count",
        "model_id",
        "model_config_hash",
        "merge_hash",
        "created_at",
    ]
    # The column is `text_raw`, not `text`, so no consumer can mistake a marker-bearing
    # string for display-ready text.
    assert "text" not in schema.names


def test_neither_table_carries_a_passage_id():
    """Passages do not exist yet at this point in the build; reconciliation produces them."""
    assert "passage_id" not in merge_map_arrow_schema().names
    assert "passage_id" not in translation_raw_arrow_schema().names


def test_merge_map_path_is_keyed_by_family_chunker_and_merge_hash():
    layout = Layout(run_dir="runs", base="runs", family="e2e", chunker_hash="c" * 64)
    p = layout.merge_map_path("m" * 64)
    assert p == (
        layout.corpus_dir("e2e", "c" * 64) / "merge_map" / ("m" * 12) / "data.parquet"
    )
    # A rule edit resolves to a fresh map beside the same spine.
    assert layout.merge_map_path("n" * 64) != p


def test_translations_raw_path_is_keyed_by_both_semantic_hashes():
    """Keying by the translation block alone would let a re-run reuse a stale grouping."""
    layout = Layout(run_dir="runs", base="runs", family="e2e", chunker_hash="c" * 64)
    p = layout.translations_raw_path("omt", "t" * 64, "m" * 64)
    assert p == (
        layout.corpus_dir("e2e", "c" * 64)
        / "translations_raw"
        / "omt"
        / f"{'t' * 12}-{'m' * 12}"
        / "data.parquet"
    )
    assert layout.translations_raw_path("omt", "t" * 64, "n" * 64) != p
    assert layout.translations_raw_path("omt", "u" * 64, "m" * 64) != p
    # The two producers of one variant never write one file.
    assert layout.translations_raw_path("omt", "t" * 64, "m" * 64, part="identity") != p


def test_translations_raw_is_not_the_reconciled_translations_name():
    """Two names, two meanings: ``translations/`` is reconciliation's marker-free output."""
    layout = Layout(run_dir="runs", base="runs", family="e2e", chunker_hash="c" * 64)
    assert "translations_raw" in str(layout.translations_raw_path("omt", "t" * 64, "m" * 64))


def test_both_paths_require_a_family_bound_layout():
    layout = Layout(run_dir="runs")
    with pytest.raises(ValueError, match="family"):
        layout.merge_map_path("m" * 64)
    with pytest.raises(ValueError, match="family"):
        layout.translations_raw_path("omt", "t" * 64, "m" * 64)
