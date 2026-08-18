"""Byte-identity of the chunker against the previous implementation, with the real SaT.

`test_chunk_byte_identity_small.py` runs the same comparison on a controlled fixture with a
fake segmenter. Here the real wtpsplit segmenter runs over a bounded slice of the real
four-language corpus, and the new span-addressed sentence inventory has to match, count and
texts in order, what the previous chunker produced: `_paragraphs`, then `segmenter.split`
per paragraph, then `nfc`.

A difference means the rebuild changed the segmentation rather than only its
representation, which is a result rather than something to reconcile away.

The two properties the data model rests on are re-checked on real text as well:
`documents.text[start:end]` is the segment verbatim, and that slice is NFC. The worker
checks both as it writes, but here they are compared against an independent reference
rather than against the producer's own bookkeeping.

The slice is `_DOCS_PER_LANG` documents per language, read from an already-downloaded raw
store, so the test needs no network. Skipped when the corpus or wtpsplit is unavailable.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from ragtime.common import Layout, doc_id_of, nfc
from ragtime.config import all_hashes, load
from ragtime.orchestration.run_identity import run_family
from ragtime.preprocess import corpus
from ragtime.preprocess.boilerplate import boilerplate_rules
from ragtime.preprocess.chunk import _chunk_options, _paragraphs, segment_documents
from ragtime.preprocess.tokenizer import load_pack_tokenizer

pytestmark = pytest.mark.full

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_PER_LANG = 200  # enough real length and script variance; minutes on a CPU node
_STEMS = ("eng-docs", "spa-docs", "rus-docs", "zho-docs")


@pytest.fixture(scope="module")
def bounded_cfg():
    return load(_REPO_ROOT / "config" / "e2e-original.yml")


def _raw_dir(cfg) -> Path:
    """The already-downloaded raw store for this run family; nothing is fetched here."""
    fam, ch = run_family(cfg), all_hashes(cfg)["chunker"]
    root = _REPO_ROOT / "runs"
    return Layout(run_dir=root, base=root).corpus_raw_dir(fam, ch)


def _bounded_docs(raw: Path) -> list[dict]:
    docs: list[dict] = []
    for stem in _STEMS:
        f = raw / f"{stem}.jsonl.gz"
        if not f.exists():
            pytest.skip(f"bounded corpus slice unavailable: {f} missing")
        for i, d in enumerate(corpus.read_native(f)):
            if i >= _DOCS_PER_LANG:
                break
            docs.append(d)
    return docs


def _main_sentence_inventory(doc: dict, segmenter, opts: dict) -> list[str]:
    """The previous chunker's sentence inventory, re-implemented here as the reference.

    The old expression is written out in full: paragraph split, per-paragraph
    ``segmenter.split``, per-sentence ``nfc``, drop blanks. The comparison is therefore
    against the previous behaviour rather than against the new code path calling itself.
    """
    rules = (
        boilerplate_rules(opts["boilerplate_rules_version"])
        if opts["strip_boilerplate"]
        else ()
    )
    paras = _paragraphs(
        nfc(doc["text"]), strip_boilerplate=opts["strip_boilerplate"], rules=rules
    )
    return [nfc(s) for p in paras for s in segmenter.split(p) if s.strip()]


def test_real_sat_sentence_inventory_is_identical_to_mains_chunker(bounded_cfg) -> None:
    pytest.importorskip("wtpsplit")
    from ragtime.serving.segmenter import Segmenter

    opts = _chunk_options(bounded_cfg)
    seg_opts = {k: opts[k] for k in ("strip_boilerplate", "boilerplate_rules_version")}
    docs = _bounded_docs(_raw_dir(bounded_cfg))
    assert len(docs) == len(_STEMS) * _DOCS_PER_LANG

    seg_model = bounded_cfg.blocks["chunker"]["config"]["segmenter_model"]
    segmenter = Segmenter(model=seg_model)
    tokenizer = load_pack_tokenizer(bounded_cfg)

    records = list(
        segment_documents(
            [dict(d) for d in docs], segmenter, tokenizer, batch_size=32, **seg_opts
        )
    )
    by_id = {r["document_id"]: r for r in records}

    total = 0
    for doc in docs:
        expected = _main_sentence_inventory(doc, segmenter, opts)
        rec = by_id.get(doc["id"])
        if not expected:  # a document that reduces to zero content is dropped by both
            assert rec is None, doc["id"]
            continue
        assert rec is not None, doc["id"]
        rebuilt = [rec["text"][s["start"] : s["end"]] for s in rec["sentences"]]
        # Compare counts first, so a length mismatch reports as one rather than as a diff.
        assert len(rebuilt) == len(expected), doc["id"]
        assert rebuilt == expected, doc["id"]
        total += len(rebuilt)
    assert total > 1000  # the slice really did exercise thousands of real sentences


def test_real_sat_spans_are_verbatim_nfc_and_tile_their_document(bounded_cfg) -> None:
    """The checks the worker runs as it writes, repeated here on real multilingual text:
    verbatim spans, NFC, dense ids, and spans that tile the document leaving only
    whitespace between them."""
    pytest.importorskip("wtpsplit")
    from ragtime.serving.segmenter import Segmenter

    opts = _chunk_options(bounded_cfg)
    seg_opts = {k: opts[k] for k in ("strip_boilerplate", "boilerplate_rules_version")}
    docs = _bounded_docs(_raw_dir(bounded_cfg))
    segmenter = Segmenter(model=bounded_cfg.blocks["chunker"]["config"]["segmenter_model"])
    tokenizer = load_pack_tokenizer(bounded_cfg)

    langs: set[str] = set()
    for rec in segment_documents(
        [dict(d) for d in docs], segmenter, tokenizer, batch_size=32, **seg_opts
    ):
        text = rec["text"]
        langs.add(rec["lang"])
        assert unicodedata.is_normalized("NFC", text)
        pos = 0
        for j, s in enumerate(rec["sentences"]):
            assert s["sentence_index"] == j  # dense, no gaps, no duplicates
            assert s["sentence_id"] == f"{rec['document_id']}#s{j}"
            assert doc_id_of(s["sentence_id"]) == rec["document_id"]
            assert 0 <= s["start"] <= s["end"] <= len(text)
            span = text[s["start"] : s["end"]]
            assert span and span.strip() == span
            assert nfc(span) == span
            assert s["start"] >= pos and not text[pos : s["start"]].strip()  # tiling
            pos = s["end"]
        assert not text[pos:].strip()
    assert langs == {"en", "es", "ru", "zh"}  # all four scripts really were exercised


def test_real_sat_split_spans_equals_split(bounded_cfg) -> None:
    """The segmenter's two representations agree on real text in all four scripts:
    ``[text[a:b] for a, b in split_spans(t)] == split(t)``."""
    pytest.importorskip("wtpsplit")
    from ragtime.serving.segmenter import Segmenter

    segmenter = Segmenter(model=bounded_cfg.blocks["chunker"]["config"]["segmenter_model"])
    opts = _chunk_options(bounded_cfg)
    rules = boilerplate_rules(opts["boilerplate_rules_version"])
    checked = 0
    for doc in _bounded_docs(_raw_dir(bounded_cfg)):
        for para in _paragraphs(nfc(doc["text"]), strip_boilerplate=True, rules=rules):
            spans = segmenter.split_spans(para)
            assert [para[a:b] for a, b in spans] == segmenter.split(para)
            checked += 1
    assert checked > 1000
