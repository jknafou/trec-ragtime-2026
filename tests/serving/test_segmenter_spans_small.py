"""``Segmenter.split_spans``: the span walk that the corpus data model rests on.

The passage model stores sentence text exactly once, as a span, so the walk that turns a
segmenter's strings into ``(start, end)`` offsets is load-bearing:
``[text[a:b] for a, b in split_spans(t)]`` must equal ``split(t)`` byte for byte, with the
same drops. These exercise :func:`ragtime.serving.segmenter.spans_of` directly on synthetic
segment lists, with no wtpsplit, no model and no network, because the interesting cases are
whitespace shapes rather than sentence boundaries. Equality against the real model's own
segments is pinned in ``tests/preprocess/test_chunk_byte_identity_full.py``.
"""

from __future__ import annotations

import pytest

from ragtime.serving.segmenter import Segmenter, SpanMismatchError, spans_of

pytestmark = pytest.mark.small


def _clean(segs: list[str]) -> list[str]:
    """The reference: exactly what ``Segmenter._clean`` (i.e. ``split``) returns."""
    return Segmenter._clean(segs)


@pytest.mark.parametrize(
    "segs",
    [
        ["One sentence."],
        ["First. ", "Second. ", "Third."],
        ["Leading space kept? ", "  indented sentence.", " trailing  "],
        ["Ends with newline.\n", "Next para line.\n\n", "Last."],
        ["第一 句子。", "第二 句子。"],  # CJK: no spaces between sentences
        ["Пе́рвое предложение. ", "Второе."],  # Cyrillic + combining marks
        ["café: résumé. ", "ligature ﬁ stays."],
        ["   ", "content after a whitespace-only segment.", "\t\n"],  # dropped segments
        ["a", "", "b"],  # an empty segment (falsy): dropped, cursor unmoved
    ],
)
def test_spans_slice_back_to_exactly_what_split_returns(segs: list[str]) -> None:
    text = "".join(segs)
    spans = spans_of(text, segs)
    assert [text[a:b] for a, b in spans] == _clean(segs)


def test_spans_are_ordered_non_overlapping_and_gap_only_whitespace() -> None:
    segs = ["First one.  ", "\n Second one. ", "   ", "Third."]
    text = "".join(segs)
    pos = 0
    for a, b in spans_of(text, segs):
        assert pos <= a < b <= len(text)
        assert not text[pos:a].strip()  # whatever we skipped was whitespace
        pos = b
    assert not text[pos:].strip()


def test_whitespace_only_and_empty_segments_are_dropped_exactly_like_clean() -> None:
    segs = ["", "   ", "\n\t", "real.", "  "]
    text = "".join(segs)
    assert len(spans_of(text, segs)) == len(_clean(segs)) == 1


def test_segments_that_do_not_reproduce_the_input_are_a_hard_error() -> None:
    """No fallback, no `find`: a lossy segmentation aborts rather than mis-addressing.

    (`find` would silently pick the first occurrence of a repeated sentence: exactly the
    class of bug this data model must not admit.)
    """
    text = "Alpha. Beta. Alpha."
    with pytest.raises(SpanMismatchError):
        spans_of(text, ["Alpha. ", "Beta. "])  # cursor lands short of len(text)
    with pytest.raises(SpanMismatchError):
        spans_of(text, ["Alpha. ", "Beta. ", "Alpha. ", "extra"])  # walks past the end


def test_a_segment_that_is_not_a_substring_at_the_cursor_is_a_hard_error() -> None:
    text = "Alpha. Beta."
    # same total length, but the second segment is not what the text holds there.
    with pytest.raises(SpanMismatchError):
        spans_of(text, ["Alpha. ", "Gamma."])


def test_split_spans_uses_the_one_resident_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``split_spans``/``split_spans_batch`` go through the same ``_backend()`` as
    ``split``, never a second model, and are the span form of the same call."""
    calls: list[str] = []

    class _Backend:
        def split(self, text: str) -> list[str]:
            calls.append(text)
            return [f"{p} " for p in text.split(" ") if p][:-1] + [text.split(" ")[-1]]

    seg = Segmenter()
    monkeypatch.setattr(Segmenter, "_backend", lambda self: _Backend())
    text = "alpha beta gamma"
    assert [text[a:b] for a, b in seg.split_spans(text)] == seg.split(text)
    assert seg.split_spans_batch([text, text]) == [seg.split_spans(text)] * 2
    assert calls  # the resident backend did the segmenting, not a private copy
