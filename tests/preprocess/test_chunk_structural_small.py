"""The structural half of the chunker: boilerplate stripping, paragraphs,
prefer_paragraph_break, paragraph_index, the wall-of-text fallback, and batch invariance.

The fake segmenter splits paragraph text on ``|`` and the fake tokenizer counts whitespace
words, so paragraph boundaries, which are blank lines, and sentence boundaries are both
under the fixture's control, and nothing loads torch or the network. Determinism, the pack
seam and the oversized rule are in ``test_chunk_small.py``.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from ragtime.preprocess.boilerplate import (
    BOILERPLATE_RULES_V1,
    boilerplate_rules,
    is_boilerplate,
)
from ragtime.preprocess.chunk import chunk_document, chunk_documents

pytestmark = pytest.mark.small


def _doc(doc_id: str, text: str, lang: str = "en") -> dict:
    return {"id": doc_id, "text": text, "url": "u", "date": "d", "lang": lang}


# --------------------------------------------------------------------------- #
# The v1 boilerplate rules, line by line.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "line",
    [
        "Home › News › Politics",  # breadcrumb shape: two or more chevron separators
        "Home > News > World",
        "Accept all cookies",  # cookie-banner CTA
        "Manage cookie preferences",
        "This website uses cookies to enhance your experience.",
        "We use cookies. By continuing you agree to our terms.",
        "Share | Tweet | Email | Print",
        "© 2023 SiteName. All rights reserved.",
        "© 2024 Example News. All rights reserved.",
        "Copyright 2023 Acme Corp. All rights reserved.",
        "Read more",
        "Continue reading »",
        "By Jane A. Doe",
        "LONDON, June 5, 2024",
        "Advertisement",
    ],
)
def test_v1_flags_pure_chrome(line: str) -> None:
    assert is_boilerplate(line, BOILERPLATE_RULES_V1)


@pytest.mark.parametrize(
    "line",
    [
        # Prose that contains a chrome token without being chrome. The rules favour
        # precision, so every one of these has to survive.
        "Home sales rose 8%/10% year-over-year in the two largest metro markets.",
        "Start-up costs rose from 20/30 percent to 45/60 percent among small businesses.",
        (
            "A federal judge ruled the AI company violated copyright law by training on "
            "the artist's work without a license."
        ),
        "The bill has three main clauses: (a) taxation, (b) benefits, and (c) enforcement.",
        (
            "The EU cookie consent rules take effect Monday, forcing sites to ask "
            "permission before tracking users."
        ),
        # More keywords sitting inside ordinary prose.
        "The government updated its cookie policy after a public consultation this year.",
        "By Monday the deal between the two firms had already been signed.",
        "The council will share the plan with residents before the vote next week.",
        "In 2024 the company reported record revenue across every regional market.",
        "The minister announced a new policy today.",
    ],
)
def test_v1_keeps_real_prose(line: str) -> None:
    assert not is_boilerplate(line, BOILERPLATE_RULES_V1)


def test_boilerplate_rules_resolves_v1_and_rejects_unknown() -> None:
    assert boilerplate_rules("v1") is BOILERPLATE_RULES_V1
    with pytest.raises(ValueError):
        boilerplate_rules("v999")


def test_long_line_with_chrome_word_is_never_stripped() -> None:
    # A line longer than the short-line guard is prose even when it opens with a
    # banner-shaped "we use cookies" clause. Precision matters more than recall here, so a
    # long line is left alone.
    long_prose = "We use cookies " + ("and analytics " * 40) + "on this platform."
    assert len(long_prose) > 120
    assert not is_boilerplate(long_prose, BOILERPLATE_RULES_V1)


# --------------------------------------------------------------------------- #
# Boilerplate lines are absent from the emitted passages.
# --------------------------------------------------------------------------- #
def test_boilerplate_lines_absent_from_passages(fake_segmenter, fake_tokenizer) -> None:
    doc = _doc(
        "eng-docs/0000100",
        "Home > News > World\n"
        "This website uses cookies to enhance your experience.\n"
        "\n"
        "The minister announced a new policy today.|It takes effect next week.\n"
        "\n"
        "© 2024 Example News. All rights reserved.",
    )
    passages = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=512, overlap_frac=0.0
    )
    joined = " ".join(p.text for p in passages)
    assert "minister announced" in joined  # real content survives
    for chrome in ("Home >", "cookies", "©", "rights reserved"):
        assert chrome not in joined  # every chrome fragment was stripped


def test_entirely_boilerplate_document_is_dropped(fake_segmenter, fake_tokenizer) -> None:
    doc = _doc(
        "eng-docs/0000101",
        "Home › News › Politics\n"
        "© 2024 Example News. All rights reserved.\n"
        "Share | Tweet | Email",
    )
    passages = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=512, overlap_frac=0.0
    )
    assert passages == []  # no content left, so the document is dropped


# --------------------------------------------------------------------------- #
# paragraph_index carries the source paragraphs a passage came from.
# --------------------------------------------------------------------------- #
def test_paragraph_index_spans_source_paragraphs(fake_segmenter, fake_tokenizer) -> None:
    doc = _doc(
        "eng-docs/0000102",
        "Para zero sentence.\n\nPara one sentence.\n\nPara two sentence.",
    )
    # A large budget packs all three one-sentence paragraphs into one passage.
    (p,) = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=512, overlap_frac=0.0
    )
    assert p.paragraph_index == (0, 1, 2)
    assert len(p.sentence_ids) == 3


def test_paragraph_index_is_per_paragraph_when_split(fake_segmenter, fake_tokenizer) -> None:
    doc = _doc(
        "eng-docs/0000103",
        "one two three\n\nfour five six\n\nseven eight nine",
    )
    # With budget 3 each three-word paragraph fills a passage exactly, one per paragraph.
    passages = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=3, overlap_frac=0.0
    )
    assert [p.paragraph_index for p in passages] == [(0,), (1,), (2,)]


# --------------------------------------------------------------------------- #
# prefer_paragraph_break closes a passage at a paragraph boundary once it is near budget.
# --------------------------------------------------------------------------- #
def _prefer_doc() -> dict:
    # Paragraph 0 packs to 7 tokens, which clears the 0.6 * 10 = 6 fill threshold. The
    # first sentence of paragraph 1 is 2 tokens and would still fit at 9, but it opens a
    # new paragraph, which is the case the option is about.
    return _doc("eng-docs/0000104", "a0 a1 a2|a3 a4 a5 a6\n\nb0 b1")


def test_prefer_paragraph_break_closes_at_boundary(fake_segmenter, fake_tokenizer) -> None:
    passages = chunk_document(
        _prefer_doc(), fake_segmenter, fake_tokenizer,
        token_budget=10, overlap_frac=0.0,
        prefer_paragraph_break=True, prefer_min_fill=0.6,
    )
    assert [p.token_count for p in passages] == [7, 2]  # closed at the paragraph seam
    assert [p.paragraph_index for p in passages] == [(0,), (1,)]


def test_without_prefer_paragraph_break_fills_to_limit(fake_segmenter, fake_tokenizer) -> None:
    passages = chunk_document(
        _prefer_doc(), fake_segmenter, fake_tokenizer,
        token_budget=10, overlap_frac=0.0,
        prefer_paragraph_break=False,
    )
    # b0 b1 is under the token limit, so it packs across the paragraph break.
    assert len(passages) == 1
    assert passages[0].token_count == 9
    assert passages[0].paragraph_index == (0, 1)


# --------------------------------------------------------------------------- #
# A document with no blank lines falls back to a single paragraph.
# --------------------------------------------------------------------------- #
def test_wall_of_text_falls_back_to_one_paragraph(fake_segmenter, fake_tokenizer) -> None:
    text = "|".join(f"sent{i} word word word" for i in range(6))  # no newlines at all
    passages = chunk_document(
        _doc("eng-docs/0000105", text), fake_segmenter, fake_tokenizer,
        token_budget=8, overlap_frac=0.0,
    )
    assert len(passages) > 1  # several passages, all from the one fallback paragraph
    for p in passages:
        assert p.paragraph_index == (0,)


# --------------------------------------------------------------------------- #
# The batched path gives the same bytes as the per-document reference.
# --------------------------------------------------------------------------- #
def test_batched_equals_reference_over_structural_docs(
    fake_segmenter, fake_tokenizer
) -> None:
    docs = [
        _doc(
            "eng-docs/0000106",
            "Home › News › Politics\nFirst para sentence one.|First para sentence two.\n\n"
            "Second para only.\n\n© 2024 X. All rights reserved.",
        ),
        _doc("rus-docs/0000107", "wall of text one|wall of text two|wall of text three", "ru"),
        _doc("spa-docs/0000108", "Solo|un|par|de|frases cortas", "es"),
    ]
    reference = [
        asdict(p)
        for d in docs
        for p in chunk_document(
            d, fake_segmenter, fake_tokenizer, token_budget=6, overlap_frac=0.15
        )
    ]
    for bs in (1, 2, 10):  # the batch size changes no byte of the output
        batched = [
            asdict(p)
            for p in chunk_documents(
                docs, fake_segmenter, fake_tokenizer,
                token_budget=6, overlap_frac=0.15, batch_size=bs,
            )
        ]
        assert batched == reference
