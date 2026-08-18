"""Chunker byte-identity: the rebuild changed the representation, not the segmentation.

The corpus data model moved sentence text from a copy inside a passage to a span of its
document. What could not move is which sentences exist. So this file compares the new
span-addressed inventory against an independent reference implementation of what the
previous chunker produced: paragraphs, then ``segmenter.split`` per paragraph, then
per-sentence ``nfc``:

    reference = [nfc(s) for p in _paragraphs(nfc(doc.text)) for s in segmenter.split(p)]
    rebuilt   = [document.text[start:end] for each sentence row]

Count and texts, in order, for every document. A difference is a result, not something to
reconcile away.

The derived passages are compared too, since `chunk_document` still packs and only stopped
writing. That is the stronger check: identical passage text, ids and spans means both the
sentence inventory and the packing survived the move. The same comparison against the real
segmenter and a slice of the real corpus is in `test_chunk_byte_identity_full.py`.

The fake segmenter splits on ``|`` and the fake tokenizer counts words, so this needs
neither torch nor the network.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from ragtime.common import nfc
from ragtime.preprocess.boilerplate import boilerplate_rules
from ragtime.preprocess.chunk import (
    _pack_sentences,
    _paragraphs,
    _sents_of,
    chunk_document,
    segment_document,
    segment_documents,
)

pytestmark = pytest.mark.small

_RULES = "v1"
_OPTS = {"strip_boilerplate": True, "boilerplate_rules_version": _RULES}
_PACK = {"token_budget": 12, "overlap_frac": 0.15}


def _doc(doc_id: str, text: str, lang: str = "en") -> dict:
    return {"id": doc_id, "text": text, "url": "u", "date": "d", "lang": lang}


def _corpus() -> list[dict]:
    """Documents covering every shape the two representations could disagree on."""
    long_sentence = " ".join(f"q{i}" for i in range(30))  # oversized vs budget 12
    many = "|".join(" ".join(f"s{i}_{j}" for j in range(5)) for i in range(9))
    return [
        # multi-paragraph + boilerplate stripped from the head and the tail
        _doc(
            "eng-docs/0000001",
            "Home › News › Politics\nFirst para one.|First para two.\n\n"
            "Second para only.\n\n© 2024 X. All rights reserved.",
        ),
        # an oversized sentence between two normal ones
        _doc("rus-docs/0000002", f"Первое предложение.|{long_sentence}|Последнее.", "ru"),
        # decomposed é: normalising the document once has to give what per-sentence nfc gave
        _doc("spa-docs/0000003", "El café tiene dos gramos|Segunda frase corta", "es"),
        # CJK, plus a paragraph break
        _doc("zho-docs/0000004", "第一 句子|第二 句子\n\n第三 句子", "zh"),
        # many passages -> exercises overlap carry
        _doc("eng-docs/0000005", many),
        # a single sentence
        _doc("spa-docs/0000006", "Una sola frase.", "es"),
        # wall of text, no structure at all
        _doc("eng-docs/0000007", "no markers at all in this document whatsoever"),
        # ragged whitespace around the separators
        _doc("eng-docs/0000008", "  padded one.   |\t tabbed two. |   third   "),
        # a whitespace-only "sentence" between two real ones (dropped by both paths)
        _doc("eng-docs/0000009", "real one.|   |real two."),
        # repeated identical sentences (a `find`-based span lookup would mis-address these)
        _doc("eng-docs/0000010", "Alpha beta.|Gamma delta.|Alpha beta."),
    ]


def _main_sentence_inventory(doc: dict, segmenter) -> list[str]:
    """The previous chunker's sentence inventory, re-implemented here.

    This is not a call into the new code path. The old expression, paragraph split then
    ``segmenter.split`` then per-sentence ``nfc`` then drop blanks, is written out, so the
    comparison is against the previous behaviour rather than against itself.
    """
    paras = _paragraphs(
        nfc(doc["text"]), strip_boilerplate=True, rules=boilerplate_rules(_RULES)
    )
    return [nfc(s) for p in paras for s in segmenter.split(p) if s.strip()]


def _rebuilt_sentence_inventory(record: dict) -> list[str]:
    """The new inventory: every sentence sliced out of its document."""
    return [record["text"][s["start"] : s["end"]] for s in record["sentences"]]


@pytest.mark.parametrize("doc", _corpus(), ids=lambda d: d["id"])
def test_sentence_inventory_is_identical_to_mains_chunker(
    doc, fake_segmenter, fake_tokenizer
) -> None:
    """Count and texts, in order, one document at a time."""
    expected = _main_sentence_inventory(doc, fake_segmenter)
    record = segment_document(dict(doc), fake_segmenter, fake_tokenizer, **_OPTS)
    rebuilt = _rebuilt_sentence_inventory(record)
    assert len(rebuilt) == len(expected)
    assert rebuilt == expected


def test_sentence_inventory_is_identical_corpus_wide(fake_segmenter, fake_tokenizer) -> None:
    """The same over the whole batched stream, where a per-document pass could not see an
    off-by-one in the batch regrouping."""
    docs = _corpus()
    expected = [s for d in docs for s in _main_sentence_inventory(d, fake_segmenter)]
    records = list(
        segment_documents(
            [dict(d) for d in docs], fake_segmenter, fake_tokenizer, batch_size=3, **_OPTS
        )
    )
    rebuilt = [s for r in records for s in _rebuilt_sentence_inventory(r)]
    assert len(rebuilt) == len(expected)
    assert rebuilt == expected


def test_derived_passages_are_identical_to_mains_packer(fake_segmenter, fake_tokenizer) -> None:
    """Packing the new span-addressed sentences reproduces the previous passage text,
    paragraph_index, spans and token counts exactly.

    Only the sentence ids change form, from ``#p{k}#s{j}`` to ``#s{j}``, so they are
    compared separately against the document-scoped expectation.
    """
    for doc in _corpus():
        expected_sents = _main_sentence_inventory(doc, fake_segmenter)
        # The old packer's input: (sentence_text, paragraph_index) pairs.
        paras = _paragraphs(
            nfc(doc["text"]), strip_boilerplate=True, rules=boilerplate_rules(_RULES)
        )
        pairs = [(nfc(s), i) for i, p in enumerate(paras) for s in fake_segmenter.split(p)]
        assert [t for t, _ in pairs] == expected_sents  # fixture sanity

        got = chunk_document(dict(doc), fake_segmenter, fake_tokenizer, **_PACK, **_OPTS)
        # Rebuild the old _Sent list, with the document-scoped ids that replaced them.
        from ragtime.preprocess.chunk import _Sent

        ref_sents = [
            _Sent(text=t, count=fake_tokenizer.count(t), sid=f"{doc['id']}#s{j}", para=para)
            for j, (t, para) in enumerate(pairs)
        ]
        want = _pack_sentences(
            doc["id"], doc["lang"], ref_sents, fake_tokenizer,
            token_budget=_PACK["token_budget"], overlap_frac=_PACK["overlap_frac"],
            prefer_paragraph_break=True, prefer_min_fill=0.6,
        )
        assert [asdict(p) for p in got] == [asdict(p) for p in want], doc["id"]


def test_repeated_sentences_get_distinct_spans_not_the_first_occurrence(
    fake_segmenter, fake_tokenizer
) -> None:
    """Why the cursor walk exists rather than a `find`: a document whose first and last
    sentences are byte-identical has to address them at different offsets."""
    doc = _doc("eng-docs/0000010", "Alpha beta.|Gamma delta.|Alpha beta.")
    rec = segment_document(doc, fake_segmenter, fake_tokenizer, **_OPTS)
    first, last = rec["sentences"][0], rec["sentences"][-1]
    assert rec["text"][first["start"] : first["end"]] == rec["text"][last["start"] : last["end"]]
    assert first["start"] != last["start"]  # a `find` would have collapsed these
    assert first["sentence_id"] != last["sentence_id"]


def test_sents_of_slices_never_re_normalizes(fake_segmenter, fake_tokenizer) -> None:
    """The packer's view is pure slicing. There is no per-sentence ``nfc()`` call any more,
    and the NFC property holds because the document was normalised once."""
    doc = _doc("spa-docs/0000003", "El café tiene dos gramos|Segunda frase corta", "es")
    rec = segment_document(doc, fake_segmenter, fake_tokenizer, **_OPTS)
    for se in _sents_of(rec):
        assert nfc(se.text) == se.text
        assert se.text in rec["text"]
