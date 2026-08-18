"""``Passage.sentence_char_spans``: exact per-sentence recovery from the passage text.

A packed passage is its sentences joined by a single space, so the paragraph structure is
gone and re-segmenting that text does not give the chunker's boundaries back. The spans are
what keeps the sentence grain recoverable, for sentence-level translation and for the
translated renderings the passage store composes from it.

For every passage: ``len(sentence_char_spans) == len(sentence_ids)``, the spans are in the
same order, and ``text[start:end]`` is that sentence verbatim. That includes the two cases
where naive offsets break, a passage that spans paragraphs, and an overlap sentence, which
keeps its home passage's ``sentence_id`` while sitting at the head of the next passage so
its span is relative to that passage.

The fake segmenter and tokenizer make the expected sentences exactly predictable, and keep
torch and the network out of it.
"""

from __future__ import annotations

import pytest

from ragtime.common import nfc
from ragtime.preprocess.boilerplate import boilerplate_rules
from ragtime.preprocess.chunk import _paragraphs, chunk_document

pytestmark = pytest.mark.small


def _doc(doc_id: str, text: str, lang: str = "en") -> dict:
    return {"id": doc_id, "text": text, "url": "u", "date": "d", "lang": lang}


def _expected_sentences(doc: dict, segmenter) -> list[str]:
    """The document's NFC sentences in document order, derived independently."""
    paras = _paragraphs(
        nfc(doc["text"]), strip_boilerplate=True, rules=boilerplate_rules("v1")
    )
    return [nfc(s) for p in paras for s in segmenter.split(p)]


def _assert_spans_recover_sentences(passages, expected: list[str]) -> None:
    """Every passage slices back to its own sentences, and the homed sentences are the
    document's sentences in order."""
    homed: dict[str, str] = {}  # sentence_id -> the text seen at its home passage
    homed_order: list[str] = []
    for p in passages:
        assert len(p.sentence_char_spans) == len(p.sentence_ids)
        slices = [p.text[s:e] for s, e in p.sentence_char_spans]
        # The spans tile the passage text exactly, single-space joiner included.
        assert " ".join(slices) == p.text
        for sid, text in zip(p.sentence_ids, slices, strict=True):
            assert text  # no empty slice
            if sid in homed:
                # An overlap sentence: same text, span relative to this passage.
                assert text == homed[sid]
            else:
                homed[sid] = text
                homed_order.append(text)
    # The sentences homed across the document are the segmenter's, in order.
    assert homed_order == expected


# --------------------------------------------------------------------------- #
# Slice round trip over the multilingual fixtures (en, ru, es, zh).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("token_budget", [10, 20, 512])
def test_spans_recover_every_sentence_across_langs(
    tiny_native_docs, fake_segmenter, fake_tokenizer, token_budget: int
) -> None:
    for doc in tiny_native_docs:
        passages = chunk_document(
            dict(doc), fake_segmenter, fake_tokenizer,
            token_budget=token_budget, overlap_frac=0.15,
        )
        _assert_spans_recover_sentences(
            passages, _expected_sentences(doc, fake_segmenter)
        )


def test_spans_recover_every_sentence_with_overlap(
    overlap_doc, fake_segmenter, fake_tokenizer
) -> None:
    passages = chunk_document(
        dict(overlap_doc), fake_segmenter, fake_tokenizer,
        token_budget=100, overlap_frac=0.15,
    )
    assert len(passages) >= 2
    _assert_spans_recover_sentences(
        passages, _expected_sentences(overlap_doc, fake_segmenter)
    )


# --------------------------------------------------------------------------- #
# A passage that spans paragraphs: the joined text has lost the break.
# --------------------------------------------------------------------------- #
def test_spans_are_exact_across_a_paragraph_spanning_passage(
    fake_segmenter, fake_tokenizer
) -> None:
    doc = _doc(
        "eng-docs/0000300",
        "Para zero sentence.|Para zero second.\n\nPara one sentence.\n\nPara two sentence.",
    )
    # A large budget packs all four sentences from three paragraphs into one passage.
    (p,) = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=512, overlap_frac=0.0
    )
    assert p.paragraph_index == (0, 1, 2)  # it really does cross the paragraph breaks
    assert [p.text[s:e] for s, e in p.sentence_char_spans] == [
        "Para zero sentence.",
        "Para zero second.",
        "Para one sentence.",
        "Para two sentence.",
    ]
    assert len(p.sentence_char_spans) == len(p.sentence_ids) == 4


# --------------------------------------------------------------------------- #
# An overlap sentence: it opens the next passage, and its span is local to that passage.
# --------------------------------------------------------------------------- #
def test_overlap_sentence_span_is_relative_to_its_new_passage(
    fake_segmenter, fake_tokenizer
) -> None:
    # With budget 10, s0 (4) and s1 (5) fill p0 and s2 (4) overflows, so p1 opens with the
    # overlap tail.
    doc = _doc(
        "eng-docs/0000301",
        "a0 a1 a2 a3|b0 b1 b2 b3 b4|c0 c1 c2 c3",
    )
    p0, p1 = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=10, overlap_frac=0.15
    )
    shared = [sid for sid in p1.sentence_ids if sid in set(p0.sentence_ids)]
    # Sentence ids are document-scoped, so the overlap head carries the same id in both
    # passages; there is no home-passage id to carry.
    assert shared == ["eng-docs/0000301#s1"]
    # In p0 the shared sentence sits after s0, in p1 it is at offset 0, so a span taken
    # from the home passage would slice the wrong text here.
    i0 = p0.sentence_ids.index(shared[0])
    i1 = p1.sentence_ids.index(shared[0])
    assert p0.sentence_char_spans[i0][0] > 0
    assert p1.sentence_char_spans[i1] == (0, len("b0 b1 b2 b3 b4"))
    assert p0.text[slice(*p0.sentence_char_spans[i0])] == "b0 b1 b2 b3 b4"
    assert p1.text[slice(*p1.sentence_char_spans[i1])] == "b0 b1 b2 b3 b4"


# --------------------------------------------------------------------------- #
# An oversized single-sentence passage: one span covering the whole text.
# --------------------------------------------------------------------------- #
def test_oversized_passage_has_one_whole_text_span(
    tiny_native_docs, fake_segmenter, fake_tokenizer
) -> None:
    doc = dict(next(d for d in tiny_native_docs if d["id"] == "rus-docs/0000002"))
    passages = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=10, overlap_frac=0.0
    )
    (p,) = [x for x in passages if x.is_oversized]
    assert p.sentence_char_spans == ((0, len(p.text)),)
    assert p.text[0 : len(p.text)] == p.text  # kept whole rather than split


# --------------------------------------------------------------------------- #
# The spans are additive: they change no packing decision and no existing field.
# --------------------------------------------------------------------------- #
def test_spans_do_not_disturb_text_ids_or_counts(
    overlap_doc, fake_segmenter, fake_tokenizer
) -> None:
    passages = chunk_document(
        dict(overlap_doc), fake_segmenter, fake_tokenizer,
        token_budget=100, overlap_frac=0.15,
    )
    for p in passages:
        # The text is still the sentences joined by one space.
        assert p.text == " ".join(p.text[s:e] for s, e in p.sentence_char_spans)
        assert p.token_count == fake_tokenizer.count(p.text)
        assert len(p.sentence_ids) == len(p.sentence_char_spans)
