"""The passage-text derivation rule, pinned so a consumer cannot re-invent it as a join.

A ``" ".join`` over member slices differs from the correct single slice for the large
majority of Chinese passages, fabricating a space that is not in the source. That space
would reach the retriever's encoded text and the verbatim NFC span-commit target.
"""

from __future__ import annotations

import pytest

from ragtime.common.schemas import passage_native_text

pytestmark = pytest.mark.small


def test_cjk_passage_has_no_fabricated_space() -> None:
    # Two adjacent Chinese sentences with nothing between them, as in the real corpus,
    # where about a third of zh adjacent-sentence separators are empty.
    doc = "改編自日本歌曲的粵語流行曲陸續面世。到90年代初，社會愈趨繁榮。"
    first = doc.index("到")
    spans = [(0, first), (first, len(doc))]

    got = passage_native_text(doc, spans)

    assert got == doc, "the slice must reproduce the document region verbatim"
    assert " " not in got, "a join would insert a space the source never had"
    assert got != " ".join(doc[a:b] for a, b in spans), "the join is the wrong rule"


def test_latin_passage_preserves_the_single_source_space() -> None:
    doc = "He arrived late. The train had left."
    mid = doc.index("The")
    spans = [(0, mid - 1), (mid, len(doc))]

    got = passage_native_text(doc, spans)

    assert got == doc
    # Here the join happens to agree, which is why the wrong rule is easy to miss.
    assert got == " ".join(doc[a:b] for a, b in spans)


def test_multi_space_separator_is_preserved_verbatim() -> None:
    # A join would normalise this to one space; the slice must not.
    doc = "First.   Second."
    spans = [(0, 6), (9, len(doc))]

    got = passage_native_text(doc, spans)

    assert got == doc
    assert got != " ".join(doc[a:b] for a, b in spans)


def test_single_sentence_and_empty_passage() -> None:
    doc = "Only one."
    assert passage_native_text(doc, [(0, len(doc))]) == doc
    assert passage_native_text(doc, []) == ""


def test_slice_spans_the_whole_contiguous_run() -> None:
    doc = "A. B. C. D."
    spans = [(0, 2), (3, 5), (6, 8)]  # A. B. C. -- D. belongs to the next passage
    assert passage_native_text(doc, spans) == "A. B. C."
