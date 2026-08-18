"""``preprocess.spine``: document-atomic row ranges over the co-ordered corpus tables.

Merge and translate both shard their work with this mechanism, so its two contracts are
pinned directly: a shard does not split a document, and a shard's ``documents.parquet``
range holds exactly the documents its work-table range covers. Everything else in merge and
translate assumes both.
"""

from __future__ import annotations

import itertools

import pytest

from ragtime.preprocess import spine

pytestmark = pytest.mark.small


def _docs(n_docs: int, sents_per_doc: int) -> list[dict]:
    return [
        {
            "document_id": f"spa-docs/{i:07d}",
            "lang": "es",
            "paragraphs": [[f"doc{i} sentence {j} here" for j in range(sents_per_doc)]],
        }
        for i in range(n_docs)
    ]


def test_shards_are_document_atomic_and_tile_every_row(build_corpus_tables) -> None:
    paths = build_corpus_tables(_docs(9, 4))
    ranges, boundary_docs = spine.plan_shards(paths["sentences"], n_shards=4)

    assert len(ranges) == len(boundary_docs)
    # tiling: contiguous, starts at 0, ends at the row count, no gap and no overlap
    assert ranges[0].row_start == 0
    assert ranges[-1].row_end == 9 * 4
    for a, b in itertools.pairwise(ranges):
        assert a.row_end == b.row_start

    # atomicity: every document's rows live in exactly one shard
    aligned = spine.align_documents(paths["documents"], ranges, boundary_docs)
    seen: dict[str, int] = {}
    for k, r in enumerate(aligned):
        for row in spine.iter_row_range(paths["sentences"], r.row_start, r.row_end):
            assert seen.setdefault(row["document_id"], k) == k
    assert len(seen) == 9


def test_document_range_covers_exactly_the_shards_documents(build_corpus_tables) -> None:
    paths = build_corpus_tables(_docs(9, 4))
    ranges, boundary_docs = spine.plan_shards(paths["sentences"], n_shards=4)
    aligned = spine.align_documents(paths["documents"], ranges, boundary_docs)

    for r in aligned:
        from_sentences = {
            row["document_id"]
            for row in spine.iter_row_range(paths["sentences"], r.row_start, r.row_end)
        }
        assert set(spine.load_document_texts(paths["documents"], r)) == from_sentences


def test_alignment_refuses_a_documents_table_from_another_build(build_corpus_tables) -> None:
    """A stale ``documents.parquet`` is a hard error, never a silently shifted range."""
    a = build_corpus_tables(_docs(4, 3), name="a")
    b = build_corpus_tables(
        [{"document_id": "zho-docs/9999999", "lang": "zh", "paragraphs": [["x y z"]]}],
        name="b",
    )
    ranges, boundary_docs = spine.plan_shards(a["sentences"], n_shards=2)
    with pytest.raises(LookupError, match="not found"):
        spine.align_documents(b["documents"], ranges, boundary_docs)


def test_row_range_read_matches_a_full_scan_slice(build_corpus_tables) -> None:
    """The footer-pruned read is the same rows as slicing a full scan: no off-by-one."""
    from ragtime.common.io import iter_parquet

    paths = build_corpus_tables(_docs(6, 5))
    everything = list(iter_parquet(paths["sentences"]))
    for start, end in ((0, 7), (7, 19), (19, len(everything)), (3, 3)):
        got = list(spine.iter_row_range(paths["sentences"], start, end))
        assert [r["sentence_id"] for r in got] == [
            r["sentence_id"] for r in everything[start:end]
        ]


def test_shard_name_sorts_in_row_order(build_corpus_tables) -> None:
    """``merge`` relies on the lexicographic shard-name sort restoring work-table order."""
    paths = build_corpus_tables(_docs(9, 4))
    ranges, _ = spine.plan_shards(paths["sentences"], n_shards=4)
    names = [r.name for r in ranges]
    assert names == sorted(names)


def test_empty_table_plans_no_shards(build_corpus_tables) -> None:
    paths = build_corpus_tables([])
    ranges, boundary_docs = spine.plan_shards(paths["sentences"], n_shards=4)
    assert ranges == [] and boundary_docs == []
    assert spine.align_documents(paths["documents"], ranges, boundary_docs) == []
