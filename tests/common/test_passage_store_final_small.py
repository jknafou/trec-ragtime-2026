"""The passage store over the text-free final corpus tables.

A passage no longer carries text: it is an ordered list of final ``sentence_id``s, so
``common.passage_store`` composes its renderings from ``documents.parquet``,
``final/<recon12>/sentences.parquet`` and
``final/<recon12>/translations/<variant>.parquet`` every time the store is built. The two
kinds of rendering follow different rules, and the tests below keep them from collapsing
into one:

- ``original`` is one contiguous slice of the document. The load-bearing case is Chinese:
  nothing separates CJK sentences in the source, so a ``" ".join`` over the members
  fabricates a space in both the text the retriever encodes and the verbatim NFC
  span-commit target.
- ``omt`` and ``omt_opus`` are a single-space join, unconditionally and for every source
  language, because those segments are English.

English has no translation row at all (``reconcile.store_identity_translations: false``),
so its ``omt`` and ``omt_opus`` text resolves to the native slice. The fixture's English
document separates its sentences by two spaces, because that is where a stored-identity
design is observably wrong: the join normalises the separator to one space, so an English
passage would render differently under ``original`` and ``omt``. Under the shipped policy
all three renderings are the same string.

Also pinned here: the corpus-scale discipline (streamed, column-pruned reads, never
``read_parquet`` and never the whole text column), that a partial translation raises
instead of silently falling back to the original, and that a non-English document with no
rows at all raises too, since absence is the normal state for English and an unguarded
fall-back would serve untranslated source text as a translated rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout
from ragtime.common import io as common_io
from ragtime.common.ids import doc_id_of, passage_id, sentence_id
from ragtime.common.passage_store import (
    LmdbPassageStore,
    PassageCompositionError,
    PassageStore,
    compose_passage_text,
    iter_final_passages,
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

_DOC_EN = "eng-docs/0007421"
_DOC_ES = "spa-docs/0451820"
_DOC_ZH = "zho-docs/2288104"


# --------------------------------------------------------------------------- #
# Fixture: the four co-ordered final tables, built by hand.
# --------------------------------------------------------------------------- #
@dataclass
class _Doc:
    """One fixture document: its sentences, their separator, and its passages.

    ``sep`` is what separates the sentences inside ``documents.text``: ``" "`` for
    Latin/Cyrillic prose and ``""`` for Chinese, exactly as in the real corpus.
    ``passages`` lists each passage's member sentence indices (a contiguous run).
    """

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


def _fixture_docs() -> list[_Doc]:
    """en (identity translations) + es + zh (the no-separator CJK case), in document order."""
    en_sentences = [
        "Nordic walking uses poles.",
        "It started in Finland.",
        "Poles engage the upper body.",
    ]
    es_sentences = ["El café contiene 2 g de cafeína.", "Es muy popular."]
    zh_sentences = [
        "改編自日本歌曲的粵語流行曲陸續面世。",
        "到90年代初，社會愈趨繁榮。",
        "躺平運動很流行。",
    ]
    return [
        _Doc(
            document_id=_DOC_EN,
            lang="en",
            sentences=en_sentences,
            # two spaces: the separator that makes the English case falsifiable. A join
            # over identity segments would normalise it to one; the native slice keeps
            # both.
            sep="  ",
            passages=[(0, 1), (2,)],
            # no rows in either translation table. English is not translated, so
            # `reconcile.store_identity_translations: false` stores nothing for it and the
            # store resolves all three renderings to the native slice.
            omt=[],
            omt_opus=[],
        ),
        _Doc(
            document_id=_DOC_ES,
            lang="es",
            sentences=es_sentences,
            sep=" ",
            passages=[(0, 1)],
            omt=["Coffee contains 2 g of caffeine.", "It is very popular."],
            omt_opus=["The coffee has 2 g of caffeine.", "It's very popular."],
        ),
        _Doc(
            document_id=_DOC_ZH,
            lang="zh",
            sentences=zh_sentences,
            sep="",  # nothing between CJK sentences, which is why the slice rule exists
            passages=[(0, 1), (2,)],
            omt=[
                "Cantonese pop songs adapted from Japanese songs kept appearing.",
                "By the early 1990s, society had grown more prosperous.",
                "Lying flat is popular.",
            ],
            omt_opus=[
                "Cantonese songs adapted from Japanese ones appeared.",
                "By the start of the 1990s, society was more prosperous.",
                "The lying-flat movement is popular.",
            ],
        ),
    ]


def _tables(docs: list[_Doc]) -> dict[str, list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    translations: dict[str, list[dict[str, Any]]] = {"omt": [], "omt_opus": []}
    for doc in docs:
        documents.append({"document_id": doc.document_id, "lang": doc.lang, "text": doc.text})
        spans = doc.spans()
        for j, (start, end) in enumerate(spans):
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


def _write(tmp_path: Path, tables: dict[str, list[dict[str, Any]]]) -> Layout:
    """Write the four tables at their Layout paths (never a hand-built path)."""
    layout = Layout(run_dir=tmp_path, base=tmp_path, family=_FAMILY, chunker_hash=_CHUNKER)
    common_io.write_parquet_stream(
        layout.documents_path(), tables["documents"], schema=document_arrow_schema()
    )
    common_io.write_parquet_stream(
        layout.final_sentences_path(_RECON), tables["sentences"], schema=sentence_arrow_schema()
    )
    common_io.write_parquet_stream(
        layout.final_passages_path(_RECON, None),
        tables["passages"],
        schema=final_passage_arrow_schema(),
    )
    for variant in ("omt", "omt_opus"):
        common_io.write_parquet_stream(
            layout.final_translations_path(_RECON, variant),
            tables[variant],
            schema=translation_final_arrow_schema(),
        )
    return layout


@pytest.fixture
def final_layout(tmp_path: Path) -> Layout:
    return _write(tmp_path, _tables(_fixture_docs()))


@pytest.fixture
def final_records(final_layout: Layout) -> dict[str, dict[str, Any]]:
    return {rec["passage_id"]: rec for rec in iter_final_passages(final_layout, _RECON, pack_hash=None)}


# --------------------------------------------------------------------------- #
# The composition rule itself.
# --------------------------------------------------------------------------- #
def test_original_rendering_is_one_slice_not_a_join_chinese(final_records) -> None:
    """The defect the slice rule exists for: no fabricated space between CJK sentences."""
    docs = {d.document_id: d for d in _fixture_docs()}
    zh = docs[_DOC_ZH]
    record = final_records[passage_id(_DOC_ZH, 0)]  # sentences 0 and 1

    expected = zh.sentences[0] + zh.sentences[1]  # the document region, verbatim
    assert record["original"] == expected
    assert " " not in record["original"], "a join would insert a space the source never had"
    assert record["original"] != " ".join(zh.sentences[:2]), "the join is the wrong rule here"


def test_translated_renderings_are_a_single_space_join(final_records) -> None:
    docs = {d.document_id: d for d in _fixture_docs()}
    zh = docs[_DOC_ZH]
    record = final_records[passage_id(_DOC_ZH, 0)]

    assert record["omt"] == " ".join(zh.omt[:2])
    assert record["omt_opus"] == " ".join(zh.omt_opus[:2])
    assert record["omt"] != record["omt_opus"]  # two tiers, two texts


def test_latin_passage_agrees_with_the_join_which_is_why_the_bug_hid(final_records) -> None:
    docs = {d.document_id: d for d in _fixture_docs()}
    es = docs[_DOC_ES]
    record = final_records[passage_id(_DOC_ES, 0)]
    assert record["original"] == es.text
    assert record["original"] == " ".join(es.sentences)  # agrees only because sep == " "


def test_sentence_char_spans_are_rebased_onto_the_passage(final_records) -> None:
    """``passage.text[a:b]`` must be the member sentence, including for a passage
    that does not start at document offset 0 (zh #p1 starts after two sentences)."""
    docs = {d.document_id: d for d in _fixture_docs()}
    zh = docs[_DOC_ZH]
    record = final_records[passage_id(_DOC_ZH, 1)]  # sentence 2 only
    assert record["sentence_char_spans"] == [[0, len(zh.sentences[2])]]
    (start, end) = record["sentence_char_spans"][0]
    assert record["original"][start:end] == zh.sentences[2]

    multi = final_records[passage_id(_DOC_ZH, 0)]
    for i, (start, end) in enumerate(multi["sentence_char_spans"]):
        assert multi["original"][start:end] == zh.sentences[i]


def test_compose_passage_text_translated_spans_are_cumulative_over_the_join() -> None:
    segments = ["First one.", "Second one."]
    text, spans = compose_passage_text("ignored", [(0, 5), (5, 10)], segments, "omt")
    assert text == "First one. Second one."
    assert spans == ((0, 10), (11, 22))
    for (start, end), segment in zip(spans, segments, strict=True):
        assert text[start:end] == segment


def test_compose_passage_text_original_preserves_a_multi_space_separator() -> None:
    """The two rules are genuinely different: the slice keeps the source separator
    verbatim, while the translated join normalises to exactly one space."""
    document = "First.   Second."
    spans = [(0, 6), (9, 16)]
    native, _ = compose_passage_text(document, spans, None, "original")
    joined, _ = compose_passage_text(document, spans, ["First.", "Second."], "omt")
    assert native == document
    assert joined == "First. Second."
    assert native != joined


def test_compose_passage_text_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="unknown rendering"):
        compose_passage_text("abc", [(0, 3)], None, "sockeye")
    with pytest.raises(PassageCompositionError, match="needs the member sentences"):
        compose_passage_text("abc", [(0, 3)], None, "omt")
    with pytest.raises(PassageCompositionError, match="correspond 1:1"):
        compose_passage_text("abc", [(0, 1), (1, 3)], ["only one"], "omt")
    with pytest.raises(PassageCompositionError, match="contiguous run"):
        compose_passage_text("abcdef", [(3, 6), (0, 3)], None, "original")
    assert compose_passage_text("abc", [], None, "original") == ("", ())


# --------------------------------------------------------------------------- #
# The record stream and the stores over it.
# --------------------------------------------------------------------------- #
def test_stream_covers_every_passage_and_carries_the_text_free_columns(final_records) -> None:
    expected_ids = [
        passage_id(doc.document_id, k)
        for doc in _fixture_docs()
        for k in range(len(doc.passages))
    ]
    assert list(final_records) == expected_ids  # passages.parquet row order, nothing skipped

    record = final_records[passage_id(_DOC_EN, 0)]
    assert record["lang"] == "en"
    assert record["document_id"] == _DOC_EN
    assert record["sentence_ids"] == [sentence_id(_DOC_EN, 0), sentence_id(_DOC_EN, 1)]
    assert record["is_oversized"] is False
    assert record["paragraph_index"] == [0]
    assert record["token_count"] > 0
    assert "text" not in record  # the table is text-free; the store composes `original`


def test_in_memory_store_over_the_stream_renders_all_three(final_layout) -> None:
    store = PassageStore.from_records(iter_final_passages(final_layout, _RECON, pack_hash=None))
    pid = passage_id(_DOC_ZH, 0)
    assert len(store) == 5
    assert pid in store
    texts = {name: store.render(pid, name) for name in ("original", "omt", "omt_opus")}
    assert all(texts.values())
    assert len(set(texts.values())) == 3
    assert store.passage(pid).lang == "zh"
    assert doc_id_of(pid) == _DOC_ZH  # citation always resolves to the original doc-id


def test_lmdb_store_build_from_final_round_trips(tmp_path, final_layout) -> None:
    db = tmp_path / "passages.lmdb"
    LmdbPassageStore.build_from_final(db, final_layout, _RECON, pack_hash=None).close()

    docs = {d.document_id: d for d in _fixture_docs()}
    with LmdbPassageStore(db) as store:  # reopened readonly, lock=False
        zh_pid = passage_id(_DOC_ZH, 0)
        assert store.render(zh_pid, "original") == docs[_DOC_ZH].sentences[0] + (
            docs[_DOC_ZH].sentences[1]
        )
        assert " " not in store.render(zh_pid, "original")
        assert store.render(zh_pid, "omt") == " ".join(docs[_DOC_ZH].omt[:2])
        assert store.render(zh_pid, "omt_opus") == " ".join(docs[_DOC_ZH].omt_opus[:2])

        p = store.passage(zh_pid)
        assert p.passage_id == zh_pid
        assert p.document_id == _DOC_ZH
        assert p.sentence_ids == (sentence_id(_DOC_ZH, 0), sentence_id(_DOC_ZH, 1))
        assert p.sentence_char_spans[0] == (0, len(docs[_DOC_ZH].sentences[0]))
        assert p.text[slice(*p.sentence_char_spans[1])] == docs[_DOC_ZH].sentences[1]

        with pytest.raises(KeyError):
            store.passage("eng-docs/nope#p0")
        with pytest.raises(ValueError, match="unknown rendering"):
            store.render(zh_pid, "sockeye")


def test_english_has_no_stored_translation_and_all_three_renderings_are_one_text(
    final_records, final_layout
) -> None:
    """The rendering axis has exactly one English text, and it is the native slice.

    Nothing is stored for English, so ``omt`` cannot be a second string. The two-space
    separator is what makes the claim falsifiable: a ``" ".join`` over identity segments
    would produce ``"...poles. It started..."`` against the document's own
    ``"...poles.  It started..."``.
    """
    docs = {d.document_id: d for d in _fixture_docs()}
    en = docs[_DOC_EN]
    record = final_records[passage_id(_DOC_EN, 0)]
    assert record["original"] == record["omt"] == record["omt_opus"]
    assert record["original"] == en.sentences[0] + "  " + en.sentences[1]
    assert record["omt"] != " ".join(en.sentences[:2]), (
        "a space join must not silently produce the native text"
    )
    # ... and the table on disk really has no English row.
    rows = list(common_io.iter_parquet(final_layout.final_translations_path(_RECON, "omt")))
    assert rows, "the non-English documents still have their rows"
    assert not [r for r in rows if r["document_id"] == _DOC_EN]


# --------------------------------------------------------------------------- #
# Broken builds fail loudly (the fall-back must never mask one).
# --------------------------------------------------------------------------- #
def test_a_non_english_document_with_no_rows_is_a_hard_error(tmp_path) -> None:
    """Absence means "not translated", which only English may be.

    No rows is the normal state for the ~37 % of the corpus that is English, so an
    unguarded fall-back would quietly hand back untranslated Chinese as this passage's
    ``omt`` rendering the moment a table were truncated or mis-ranged.
    """
    docs = _fixture_docs()
    docs[2].omt = []  # the zh document has no `omt` rows at all
    layout = _write(tmp_path, _tables(docs))
    with pytest.raises(PassageCompositionError, match="has no rows for this 'zh' passage"):
        list(iter_final_passages(layout, _RECON, pack_hash=None))


def test_a_partially_covered_passage_is_a_hard_error(tmp_path) -> None:
    """The fall-back is for English identity, not for a half-translated passage."""
    tables = _tables(_fixture_docs())
    dropped = sentence_id(_DOC_ZH, 1)
    tables["omt"] = [row for row in tables["omt"] if row["sentence_id"] != dropped]
    layout = _write(tmp_path, tables)
    with pytest.raises(PassageCompositionError, match="partial rendering must never"):
        list(iter_final_passages(layout, _RECON, pack_hash=None))


def test_a_passage_naming_an_unknown_sentence_is_a_hard_error(tmp_path) -> None:
    tables = _tables(_fixture_docs())
    tables["sentences"] = [
        row for row in tables["sentences"] if row["sentence_id"] != sentence_id(_DOC_ZH, 1)
    ]
    layout = _write(tmp_path, tables)
    with pytest.raises(PassageCompositionError, match="not in the final sentence inventory"):
        list(iter_final_passages(layout, _RECON, pack_hash=None))


def test_rows_for_a_document_absent_from_documents_parquet_fail_loudly(tmp_path) -> None:
    """A leftover row means the four tables are not from one corpus build."""
    tables = _tables(_fixture_docs())
    ghost = "zho-docs/9999999"
    tables["sentences"].append(
        {
            "sentence_id": sentence_id(ghost, 0),
            "document_id": ghost,
            "sentence_index": 0,
            "lang": "zh",
            "start": 0,
            "end": 3,
            "paragraph_index": 0,
            "token_count": 1,
        }
    )
    layout = _write(tmp_path, tables)
    with pytest.raises(PassageCompositionError, match="rows remain for document"):
        list(iter_final_passages(layout, _RECON, pack_hash=None))


# --------------------------------------------------------------------------- #
# Corpus-scale discipline (9.4 M / 88.7 M rows; a 3.9 GB text column).
# --------------------------------------------------------------------------- #
def test_the_build_streams_column_pruned_reads_and_never_materializes_a_table(
    final_layout, monkeypatch
) -> None:
    calls: list[tuple[str, tuple[str, ...] | None, int]] = []
    real_iter_parquet = common_io.iter_parquet

    def spy(path, *, columns=None, batch_size=common_io.PARQUET_BATCH_SIZE):
        calls.append((Path(path).name, tuple(columns) if columns else None, batch_size))
        return real_iter_parquet(path, columns=columns, batch_size=batch_size)

    def forbidden(*args, **kwargs):
        raise AssertionError("read_parquet materializes the whole table, never at this scale")

    monkeypatch.setattr(common_io, "iter_parquet", spy)
    monkeypatch.setattr(common_io, "read_parquet", forbidden)

    stream = iter_final_passages(final_layout, _RECON, pack_hash=None)
    first = next(stream)  # lazily: one record before any table is finished
    assert first["passage_id"] == passage_id(_DOC_EN, 0)
    rest = list(stream)
    assert len(rest) == 4

    by_name = {name: (columns, batch) for name, columns, batch in calls}
    assert set(by_name) == {
        "documents.parquet",
        "sentences.parquet",
        "passages.parquet",
        "omt.parquet",
        "omt_opus.parquet",
    }
    # documents.parquet is the one table carrying corpus text: pruned to two columns
    # and read in small batches, so the large text column is never resident.
    doc_columns, doc_batch = by_name["documents.parquet"]
    assert doc_columns == ("document_id", "text")
    assert doc_batch <= 1024
    # the passage table is text-free: no read may even ask for a text column
    passage_columns, _ = by_name["passages.parquet"]
    assert "text" not in passage_columns
    assert by_name["sentences.parquet"][0] == ("sentence_id", "document_id", "start", "end")


def test_paths_come_from_layout_only(final_layout) -> None:
    """Every table this module reads is a Layout-resolved path (no second path system)."""
    node = final_layout.final_dir(_RECON)
    assert final_layout.final_passages_path(_RECON, None) == node / "passages.parquet"
    assert final_layout.final_sentences_path(_RECON) == node / "sentences.parquet"
    assert final_layout.final_translations_path(_RECON, "omt") == node / "translations/omt.parquet"
    assert final_layout.documents_path().name == "documents.parquet"
    for path in (
        final_layout.documents_path(),
        final_layout.final_passages_path(_RECON, None),
        final_layout.final_sentences_path(_RECON),
        final_layout.final_translations_path(_RECON, "omt"),
        final_layout.final_translations_path(_RECON, "omt_opus"),
    ):
        assert path.exists() and common_io.is_done(path)
