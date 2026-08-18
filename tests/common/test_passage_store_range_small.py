"""The composition pushdown: ``iter_final_passages`` over one document row range.

Without a pushdown ``iter_final_passages`` opens all five final tables at row 0 and
composes every passage of every document, so a caller wanting one shard's worth composes
the corpus and filters afterwards. On a sharded index build that fixed re-read is a
substantial share of the total cost, and it makes finer sharding progressively worse:
the smaller the shard, the larger the fraction spent re-reading the whole corpus.

The bar is byte-identity rather than "looks right". A subtly mis-aligned slice would
produce a silently incomplete index, so every test below compares the ranged composition
against the un-ranged stream's own records for the same documents, field for field and in
order. The five shapes that can break alignment each get their own case:

1. a range starting mid-corpus rather than at row 0,
2. a range whose boundary document has no passages, so nothing anchors the passage cursor,
3. a range of English documents, which has zero rows in either translation table, the
   normal state since ``reconcile.store_identity_translations: false``,
4. a range of non-English documents, with rows in both translation tables,
5. the degenerate single-document range.

Plus the two properties that make the pushdown safe to build on: the ranges of a shard
plan partition the corpus exactly, so concatenating them reproduces the full stream, and
the un-ranged call is byte-unaffected. The equivalent proof on the real corpus is
``test_passage_store_range_full.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout
from ragtime.common import io as common_io
from ragtime.common.ids import passage_id, sentence_id
from ragtime.common.passage_store import (
    PASSAGES_TABLE,
    RENDERINGS,
    SENTENCES_TABLE,
    FinalRange,
    LmdbPassageStore,
    PassageCompositionError,
    iter_final_passages,
    plan_final_ranges,
    translations_table,
)
from ragtime.common.schemas import (
    document_arrow_schema,
    final_passage_arrow_schema,
    sentence_arrow_schema,
    translation_final_arrow_schema,
)

pytestmark = pytest.mark.small

_FAMILY = "e2e"
_CHUNKER = "c" * 64
_RECON = "9f3a1b2c4d5e6f70" + "0" * 48


# --------------------------------------------------------------------------- #
# Fixture: a corpus laid out like the real one, with an English block, then a
# non-English block, and a passage-less document planted on a range boundary.
# --------------------------------------------------------------------------- #
@dataclass
class _Doc:
    document_id: str
    lang: str
    sentences: list[str]
    sep: str
    passages: list[tuple[int, ...]]
    omt: list[str] = field(default_factory=list)
    omt_opus: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.sep.join(self.sentences)

    def spans(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        cursor = 0
        for i, sentence in enumerate(self.sentences):
            if i:
                cursor += len(self.sep)
            out.append((cursor, cursor + len(sentence)))
            cursor += len(sentence)
        return out


def _en(i: int) -> _Doc:
    """An English document: two spaces between sentences, no translation rows at all."""
    sentences = [f"English sentence {i}a.", f"English sentence {i}b.", f"English tail {i}."]
    return _Doc(
        document_id=f"eng-docs/{i:07d}",
        lang="en",
        sentences=sentences,
        sep="  ",  # two spaces: the separator a join would silently normalise to one
        passages=[(0, 1), (2,)],
    )


def _zh(i: int) -> _Doc:
    """A Chinese document: nothing separates the sentences, rows in both variants."""
    sentences = [f"改編自日本歌曲的粵語流行曲{i}。", f"到90年代初，社會愈趨繁榮{i}。", f"躺平運動很流行{i}。"]
    return _Doc(
        document_id=f"zho-docs/{i:07d}",
        lang="zh",
        sentences=sentences,
        sep="",
        passages=[(0, 1), (2,)],
        omt=[f"Cantonese pop songs {i}.", f"By the early 1990s {i}.", f"Lying flat {i}."],
        omt_opus=[f"Canto songs {i}.", f"By the 1990s {i}.", f"Lying-flat {i}."],
    )


def _es_no_passages(i: int) -> _Doc:
    """A document with sentences (and translations) but no passage rows.

    Planted on a range boundary: the passage cursor gets nothing to anchor on,
    so a range that resolved the passage table by "where this document's rows start" rather
    than by "the first row at or after this document's ordinal" would mis-align here.
    """
    return _Doc(
        document_id=f"spa-docs/{i:07d}",
        lang="es",
        sentences=[f"El café {i}.", f"Es popular {i}."],
        sep=" ",
        passages=[],
        omt=[f"The coffee {i}.", f"It is popular {i}."],
        omt_opus=[f"Coffee {i}.", f"Popular {i}."],
    )


#: Document order on disk, mirroring the real corpus's contiguous per-language blocks.
#: Ordinals: 0-3 en, 4 es (no passages), 5-9 zh, 10 es (no passages), 11-12 en.
def _fixture_docs() -> list[_Doc]:
    return [
        *[_en(i) for i in range(4)],
        _es_no_passages(400),
        *[_zh(i) for i in range(5)],
        _es_no_passages(401),
        *[_en(i) for i in (90, 91)],
    ]


def _tables(docs: list[_Doc]) -> dict[str, list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    translations: dict[str, list[dict[str, Any]]] = {"omt": [], "omt_opus": []}
    for doc in docs:
        documents.append({"document_id": doc.document_id, "lang": doc.lang, "text": doc.text})
        for j, (start, end) in enumerate(doc.spans()):
            sentences.append(
                {
                    "sentence_id": sentence_id(doc.document_id, j),
                    "document_id": doc.document_id,
                    "sentence_index": j,
                    "lang": doc.lang,
                    "start": start,
                    "end": end,
                    "paragraph_index": 0,
                    "token_count": len(doc.sentences[j].split()) or 1,
                }
            )
        for k, members in enumerate(doc.passages):
            passages.append(
                {
                    "passage_id": passage_id(doc.document_id, k),
                    "document_id": doc.document_id,
                    "lang": doc.lang,
                    "sentence_ids": [sentence_id(doc.document_id, j) for j in members],
                    "token_count": sum(len(doc.sentences[j].split()) or 1 for j in members),
                    "is_oversized": False,
                    "paragraph_index": [0],
                }
            )
        for variant, texts in (("omt", doc.omt), ("omt_opus", doc.omt_opus)):
            for j, text in enumerate(texts):
                translations[variant].append(
                    {
                        "sentence_id": sentence_id(doc.document_id, j),
                        "document_id": doc.document_id,
                        "variant": variant,
                        "text": text,
                        "source_lang": doc.lang,
                    }
                )
    return {
        "documents": documents,
        "sentences": sentences,
        "passages": passages,
        "omt": translations["omt"],
        "omt_opus": translations["omt_opus"],
    }


def _write(tmp_path: Path, tables: dict[str, list[dict[str, Any]]], *, row_group: int) -> Layout:
    """Write the five tables at their Layout paths, in small row groups.

    ``row_group`` is tiny so a fixture of a dozen documents still spans many
    row groups. The footer-index pruning this feature rests on is only exercised when the
    range's rows are a strict subset of the file's groups.
    """
    layout = Layout(run_dir=tmp_path, base=tmp_path, family=_FAMILY, chunker_hash=_CHUNKER)
    common_io.write_parquet_stream(
        layout.documents_path(),
        tables["documents"],
        schema=document_arrow_schema(),
        row_group_size=row_group,
    )
    common_io.write_parquet_stream(
        layout.final_sentences_path(_RECON),
        tables["sentences"],
        schema=sentence_arrow_schema(),
        row_group_size=row_group,
    )
    common_io.write_parquet_stream(
        layout.final_passages_path(_RECON, None),
        tables["passages"],
        schema=final_passage_arrow_schema(),
        row_group_size=row_group,
    )
    for variant in ("omt", "omt_opus"):
        common_io.write_parquet_stream(
            layout.final_translations_path(_RECON, variant),
            tables[variant],
            schema=translation_final_arrow_schema(),
            row_group_size=row_group,
        )
    return layout


@pytest.fixture(params=[2, 3, 7], ids=["rg2", "rg3", "rg7"])
def final_layout(tmp_path: Path, request: pytest.FixtureRequest) -> Layout:
    """The same corpus at three row-group sizes.

    Row groups are the unit the footer prunes on, so a range boundary that happens to land
    on a group edge and one that lands mid-group are different code paths (the trailing rows
    of a boundary group are dropped positionally). Parametrizing the size sweeps both without
    hand-picking offsets.
    """
    return _write(tmp_path, _tables(_fixture_docs()), row_group=request.param)


def _full(layout: Layout, **kwargs: Any) -> list[dict[str, Any]]:
    return list(iter_final_passages(layout, _RECON, pack_hash=None, **kwargs))


def _ranged(layout: Layout, start: int, end: int, **kwargs: Any) -> list[dict[str, Any]]:
    (rng,) = plan_final_ranges(
        layout,
        _RECON,
        pack_hash=None,
        document_ranges=[(start, end)],
        renderings=kwargs.get("renderings", RENDERINGS),
    )
    return list(iter_final_passages(layout, _RECON, pack_hash=None, document_range=rng, **kwargs))


def _reference(layout: Layout, start: int, end: int) -> list[dict[str, Any]]:
    """What the un-ranged stream yields for documents ``[start, end)``: the oracle."""
    wanted = [d.document_id for d in _fixture_docs()[start:end]]
    keep = set(wanted)
    return [r for r in _full(layout) if r["document_id"] in keep]


# --------------------------------------------------------------------------- #
# The five byte-identity cases.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("start", "end", "why"),
    [
        (5, 9, "mid-corpus: neither table is opened at row 0"),
        (4, 8, "the boundary document (ordinal 4) has NO passages"),
        (9, 11, "the range's last document has no passages"),
        (0, 4, "English only: ZERO rows in either translation table"),
        (11, 13, "English only, and it is the corpus TAIL"),
        (5, 10, "non-English only: rows in both translation tables"),
        (6, 7, "degenerate single-document range"),
        (4, 5, "degenerate single-document range, and that document has no passages"),
        (0, 13, "the whole corpus, expressed as a range"),
    ],
)
def test_ranged_composition_is_byte_identical_to_the_full_stream(
    final_layout: Layout, start: int, end: int, why: str
) -> None:
    """Field for field, in order, for every shape of range."""
    assert _ranged(final_layout, start, end) == _reference(final_layout, start, end), why


def test_an_empty_range_yields_nothing(final_layout: Layout) -> None:
    """A zero-width range is a legitimate answer (an empty shard), not an error."""
    assert _ranged(final_layout, 6, 6) == []


def test_the_shard_plan_partitions_the_corpus_exactly(final_layout: Layout) -> None:
    """Concatenating a plan's ranges reproduces the full stream: nothing lost, nothing twice.

    This is the property a sharded build actually depends on: 'each shard is right' does not
    imply 'the shards together are the corpus'. The cuts land on the
    passage-less documents (ordinals 4 and 10) and on the language block boundaries.
    """
    cuts = [(0, 4), (4, 5), (5, 9), (9, 11), (11, 13)]
    ranges = plan_final_ranges(final_layout, _RECON, pack_hash=None, document_ranges=cuts)
    assert [(r.doc_row_start, r.doc_row_end) for r in ranges] == cuts

    stitched = [
        record
        for rng in ranges
        for record in iter_final_passages(
            final_layout, _RECON, pack_hash=None, document_range=rng
        )
    ]
    assert stitched == _full(final_layout)

    # ... and the sub-ranges themselves tile each table with no gap and no overlap.
    for table in (SENTENCES_TABLE, PASSAGES_TABLE, translations_table("omt")):
        spans = [r.rows_for(table) for r in ranges]
        assert spans[0][0] == 0
        for (_, prev_end), (start, _) in pairwise(spans):
            assert start == prev_end


def test_english_ranges_own_no_translation_rows_at_all(final_layout: Layout) -> None:
    """Absence is the normal state for English, and the plan says so with an empty range.

    The English block is not merely "no rows read": the planned range is empty, which is
    what makes the read free rather than merely filtered.
    """
    (head,) = plan_final_ranges(
        final_layout, _RECON, pack_hash=None, document_ranges=[(0, 4)]
    )
    for variant in ("omt", "omt_opus"):
        start, end = head.rows_for(translations_table(variant))
        assert start == end, "an all-English range must own zero translation rows"
    # The passages and sentences of those same documents are not empty.
    assert head.rows_for(PASSAGES_TABLE) == (0, 8)
    assert head.rows_for(SENTENCES_TABLE)[1] > 0
    # And every rendering of an English passage is still the native slice, falsifiably so
    # on the multi-sentence passages, where a ``" ".join`` of identity segments would
    # normalise the fixture's two-space separator down to one.
    records = _ranged(final_layout, 0, 4)
    assert records
    multi = [r for r in records if len(r["sentence_ids"]) > 1]
    assert multi
    for record in records:
        assert record["original"] == record["omt"] == record["omt_opus"]
    for record in multi:
        assert "  " in record["original"], "the two-space separator survives (no join)"


def test_a_non_english_range_owns_rows_in_both_translation_tables(final_layout: Layout) -> None:
    (rng,) = plan_final_ranges(final_layout, _RECON, pack_hash=None, document_ranges=[(5, 10)])
    for variant in ("omt", "omt_opus"):
        start, end = rng.rows_for(translations_table(variant))
        assert end - start == 15, "5 zh documents x 3 sentences"
    records = _ranged(final_layout, 5, 10)
    assert records and all(r["lang"] == "zh" for r in records)
    for record in records:
        assert record["omt"] != record["original"] != record["omt_opus"]
        assert " " not in record["original"], "the CJK slice rule survives the range"


def test_a_passage_less_boundary_document_contributes_nothing_and_shifts_nothing(
    final_layout: Layout,
) -> None:
    """Ordinal 4 has sentences and translations but no passages, the hardest boundary."""
    (rng,) = plan_final_ranges(final_layout, _RECON, pack_hash=None, document_ranges=[(4, 5)])
    assert rng.rows_for(PASSAGES_TABLE)[0] == rng.rows_for(PASSAGES_TABLE)[1]
    assert rng.rows_for(SENTENCES_TABLE)[1] - rng.rows_for(SENTENCES_TABLE)[0] == 2
    assert _ranged(final_layout, 4, 5) == []
    # The range after it still opens on the right passage row.
    assert _ranged(final_layout, 5, 6) == _reference(final_layout, 5, 6)


# --------------------------------------------------------------------------- #
# The un-ranged call is byte-unaffected, and the read really is pruned.
# --------------------------------------------------------------------------- #
def test_the_unranged_stream_is_unchanged(
    final_layout: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing callers get the identical records through the identical reader.

    ``document_range=None`` must still go through ``iter_parquet`` (whole-table streaming),
    not through a degenerate full-width range: a silent change of reader would be a change
    of default behaviour smuggled in as an optimisation.
    """
    docs = _fixture_docs()
    expected = [
        passage_id(doc.document_id, k) for doc in docs for k in range(len(doc.passages))
    ]
    assert [r["passage_id"] for r in _full(final_layout)] == expected

    seen: list[str] = []
    real = common_io.iter_parquet

    def spy(path: Any, **kwargs: Any) -> Any:
        seen.append(Path(path).name)
        return real(path, **kwargs)

    monkeypatch.setattr(common_io, "iter_parquet", spy)
    _full(final_layout)
    assert set(seen) == {
        "documents.parquet",
        "sentences.parquet",
        "passages.parquet",
        "omt.parquet",
        "omt_opus.parquet",
    }


def _table_paths(layout: Layout) -> dict[str, Path]:
    return {
        "documents.parquet": layout.documents_path(),
        "sentences.parquet": layout.final_sentences_path(_RECON),
        "passages.parquet": layout.final_passages_path(_RECON, None),
        "omt.parquet": layout.final_translations_path(_RECON, "omt"),
        "omt_opus.parquet": layout.final_translations_path(_RECON, "omt_opus"),
    }


def test_a_ranged_read_opens_only_the_row_groups_it_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the range: the read is O(range), not O(corpus) with a filter after it.

    Asserted structurally, since a ranged stream never calls the whole-table
    ``iter_parquet`` and the row groups it does open are a strict subset of each file's,
    rather than by timing, which a shared filesystem makes unreproducible. The real-corpus
    timing is in ``test_passage_store_range_full.py``.
    """
    layout = _write(tmp_path, _tables(_fixture_docs()), row_group=2)
    opened: list[tuple[str, tuple[int, ...]]] = []
    real_batches = common_io.iter_parquet_batches

    def spy_batches(path: Any, *, row_groups: Any = None, **kwargs: Any) -> Any:
        if row_groups is not None:
            opened.append((Path(path).name, tuple(row_groups)))
        return real_batches(path, row_groups=row_groups, **kwargs)

    def forbidden(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a ranged read must never stream a whole table")

    (rng,) = plan_final_ranges(layout, _RECON, pack_hash=None, document_ranges=[(5, 9)])
    monkeypatch.setattr(common_io, "iter_parquet_batches", spy_batches)
    monkeypatch.setattr(common_io, "iter_parquet", forbidden)
    records = list(iter_final_passages(layout, _RECON, pack_hash=None, document_range=rng))

    assert records, "the range is non-empty"
    by_file = dict(opened)
    paths = _table_paths(layout)
    assert set(by_file) == set(paths)
    for name, groups in by_file.items():
        total = len(common_io.parquet_row_group_sizes(paths[name]))
        assert 0 < len(groups) < total, f"{name}: opened {len(groups)} of {total} row groups"


# --------------------------------------------------------------------------- #
# A range cannot be used wrongly without saying so.
# --------------------------------------------------------------------------- #
def test_a_range_planned_for_fewer_renderings_refuses_a_wider_stream(
    final_layout: Layout,
) -> None:
    """A missing sub-range is a loud error, never a silently unread table."""
    (rng,) = plan_final_ranges(
        final_layout,
        _RECON,
        pack_hash=None,
        document_ranges=[(5, 9)],
        renderings=("original", "omt"),
    )
    assert translations_table("omt_opus") not in rng.table_rows
    with pytest.raises(PassageCompositionError, match="carries no row range"):
        list(
            iter_final_passages(
                final_layout, _RECON, pack_hash=None, document_range=rng
            )
        )
    # ... while the renderings it was planned for compose exactly as the full stream does.
    got = list(
        iter_final_passages(
            final_layout,
            _RECON,
            pack_hash=None,
            document_range=rng,
            renderings=("original", "omt"),
        )
    )
    ref = [
        r
        for r in _full(final_layout, renderings=("original", "omt"))
        if r["document_id"] in {d.document_id for d in _fixture_docs()[5:9]}
    ]
    assert got == ref


def test_a_hand_built_range_that_does_not_describe_its_documents_fails_loudly(
    final_layout: Layout,
) -> None:
    """The leftover-rows check still fires: a mis-ranged read is a crash, not a short index."""
    (rng,) = plan_final_ranges(final_layout, _RECON, pack_hash=None, document_ranges=[(5, 9)])
    wrong = FinalRange(
        doc_row_start=rng.doc_row_start,
        doc_row_end=rng.doc_row_end,
        # one row too many in the passage table: the extra belongs to a later document
        table_rows={
            **rng.table_rows,
            PASSAGES_TABLE: (rng.rows_for(PASSAGES_TABLE)[0], rng.rows_for(PASSAGES_TABLE)[1] + 1),
        },
    )
    with pytest.raises(PassageCompositionError, match="rows remain after this document range"):
        list(
            iter_final_passages(final_layout, _RECON, pack_hash=None, document_range=wrong)
        )


def test_plan_rejects_inverted_or_overlapping_ranges(final_layout: Layout) -> None:
    with pytest.raises(ValueError, match="inverted"):
        plan_final_ranges(final_layout, _RECON, pack_hash=None, document_ranges=[(6, 3)])
    with pytest.raises(ValueError, match="non-overlapping"):
        plan_final_ranges(final_layout, _RECON, pack_hash=None, document_ranges=[(0, 6), (3, 9)])
    with pytest.raises(ValueError, match="unknown rendering"):
        plan_final_ranges(
            final_layout,
            _RECON,
            pack_hash=None,
            document_ranges=[(0, 1)],
            renderings=("original", "sockeye"),
        )
    assert plan_final_ranges(final_layout, _RECON, pack_hash=None, document_ranges=[]) == []


def test_a_range_round_trips_through_a_shard_payload(final_layout: Layout) -> None:
    """A plan is resolved once at seed and carried to a worker, as JSON, like ShardRange."""
    import json

    (rng,) = plan_final_ranges(final_layout, _RECON, pack_hash=None, document_ranges=[(5, 9)])
    restored = FinalRange.from_payload(json.loads(json.dumps(rng.payload())))
    assert restored == rng
    assert list(
        iter_final_passages(final_layout, _RECON, pack_hash=None, document_range=restored)
    ) == _reference(final_layout, 5, 9)


# --------------------------------------------------------------------------- #
# Cross-stage I/O: the stores consume a ranged stream unchanged.
# --------------------------------------------------------------------------- #
def test_an_lmdb_store_built_from_stitched_ranges_equals_one_built_from_the_corpus(
    tmp_path: Path, final_layout: Layout
) -> None:
    """The pushdown is transparent to every consumer of the record stream."""
    whole = tmp_path / "whole.lmdb"
    sharded = tmp_path / "sharded.lmdb"
    LmdbPassageStore.build(whole, _full(final_layout)).close()
    ranges = plan_final_ranges(
        final_layout, _RECON, pack_hash=None, document_ranges=[(0, 5), (5, 13)]
    )
    LmdbPassageStore.build(
        sharded,
        (
            record
            for rng in ranges
            for record in iter_final_passages(
                final_layout, _RECON, pack_hash=None, document_range=rng
            )
        ),
    ).close()
    with LmdbPassageStore(whole) as a, LmdbPassageStore(sharded) as b:
        for doc in _fixture_docs():
            for k in range(len(doc.passages)):
                pid = passage_id(doc.document_id, k)
                for rendering in RENDERINGS:
                    assert a.render(pid, rendering) == b.render(pid, rendering)
                assert a.passage(pid) == b.passage(pid)
