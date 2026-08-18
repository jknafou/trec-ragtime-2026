"""Chunk determinism, the pack boundary, the oversized rule, and the cross-stage row shapes.

The deterministic core, ``chunk_document``, runs against a fake segmenter that splits on
``|`` and a fake tokenizer that counts whitespace words, so the expected packing is known
exactly and neither torch nor the network is involved. ``chunk(cfg)`` is also run end to end
over a tiny staged raw store with the same fakes injected.
"""

from __future__ import annotations

import gzip
import importlib
import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path

import pytest

from ragtime.common import Document, Layout, Passage, Sentence, doc_id_of, iter_parquet
from ragtime.config import all_hashes
from ragtime.preprocess import corpus
from ragtime.preprocess.chunk import chunk, chunk_document

pytestmark = pytest.mark.small

ck = importlib.import_module("ragtime.preprocess.chunk")

_PID_RE = re.compile(r"^.+#p\d+$")
_SID_RE = re.compile(r"^[^#]+#s\d+$")  # document-scoped, not nested under a passage


def _by_id(docs: list[dict], doc_id: str) -> dict:
    return next(d for d in docs if d["id"] == doc_id)


def _stage(base: Path, cfg, docs: list[dict]) -> Layout:
    """Write ``docs`` into the raw store for this family, keyed by the chunker hash."""
    fam = "e2e"
    ch = all_hashes(cfg)["chunker"]
    layout = Layout(run_dir=base, base=base, family=fam, chunker_hash=ch)
    raw = layout.corpus_raw_dir(fam, ch)
    raw.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = {}
    for d in docs:
        groups.setdefault(d["id"].split("/", 1)[0], []).append(d)
    for stem, rows in groups.items():
        with gzip.open(raw / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                # Drop the fixture 'lang'; read_native re-tags it from the filename.
                f.write(json.dumps({k: v for k, v in r.items() if k != "lang"}) + "\n")
    return layout


# --------------------------------------------------------------------------- #
# Determinism, NFC, ids, overlap.
# --------------------------------------------------------------------------- #
def test_chunk_is_byte_deterministic_and_writes_no_passages(
    tmp_path, make_cfg, tiny_native_docs, fake_segmenter, fake_tokenizer
) -> None:
    cfg = make_cfg(token_budget=512, overlap_frac=0.15)
    docs = [_by_id(tiny_native_docs, i) for i in (
        "eng-docs/0000001", "rus-docs/0000002", "spa-docs/0000003", "zho-docs/0000004",
    )]
    a, b = tmp_path / "a", tmp_path / "b"
    la = _stage(a, cfg, docs)
    _stage(b, cfg, docs)
    docs_a, sents_a = chunk(cfg, base=a, segmenter=fake_segmenter, tokenizer=fake_tokenizer)
    docs_b, sents_b = chunk(cfg, base=b, segmenter=fake_segmenter, tokenizer=fake_tokenizer)
    assert (docs_a, sents_a) == (la.documents_path(), la.sentences_path())
    # Byte-identical output: there is no random state, and the writer options are pinned.
    assert docs_a.read_bytes() == docs_b.read_bytes()
    assert sents_a.read_bytes() == sents_b.read_bytes()

    # Documents and sentences only. Chunk writes no passage artefact, because reconcile is
    # the sole producer of passages and two meanings of `passages.parquet` cannot coexist.
    corpus_dir = docs_a.parent
    emitted = sorted(p.name for p in corpus_dir.iterdir() if p.is_file())
    assert emitted == [
        "documents.parquet",
        "documents.parquet._SUCCESS",
        "sentences.parquet",
        "sentences.parquet._SUCCESS",
    ]
    assert not (corpus_dir / "passages").exists()


def test_oversized_sentence_is_its_own_untouched_passage(
    tiny_native_docs, fake_segmenter, fake_tokenizer
) -> None:
    doc = dict(_by_id(tiny_native_docs, "rus-docs/0000002"))
    passages = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=10, overlap_frac=0.0
    )
    oversized = [p for p in passages if p.is_oversized]
    assert len(oversized) == 1
    (p,) = oversized
    assert len(p.sentence_ids) == 1  # not split
    assert p.token_count > 10  # over the budget
    assert p.token_count == fake_tokenizer.count(p.text)  # the whole sentence, intact
    # It is flushed as its own passage: the preceding short sentence stays in a separate,
    # non-oversized passage rather than being merged into it.
    idx = passages.index(p)
    assert idx >= 1 and not passages[idx - 1].is_oversized
    assert p.text not in passages[idx - 1].text


def test_nfc_precomposed_and_decomposed_chunk_identically(
    tiny_native_docs, fake_segmenter, fake_tokenizer
) -> None:
    # tiny_native_docs holds both é copies under one id: [0] precomposed, [-1] decomposed.
    copies = [d for d in tiny_native_docs if d["id"] == "spa-docs/0000003"]
    assert unicodedata.normalize("NFD", copies[0]["text"]) == unicodedata.normalize(
        "NFD", copies[-1]["text"]
    )  # the two copies differ only in precomposed versus decomposed form
    pre = chunk_document(
        dict(copies[0]), fake_segmenter, fake_tokenizer, token_budget=512, overlap_frac=0.0
    )
    dec = chunk_document(
        dict(copies[-1]), fake_segmenter, fake_tokenizer, token_budget=512, overlap_frac=0.0
    )
    assert [asdict(p) for p in pre] == [asdict(p) for p in dec]  # same bytes, ids and text
    for p in pre:
        assert unicodedata.is_normalized("NFC", p.text)


def test_passage_and_sentence_ids_are_well_formed(
    tiny_native_docs, fake_segmenter, fake_tokenizer
) -> None:
    for doc in tiny_native_docs:
        for p in chunk_document(
            dict(doc), fake_segmenter, fake_tokenizer, token_budget=20, overlap_frac=0.15
        ):
            assert _PID_RE.match(p.passage_id)
            assert doc_id_of(p.passage_id) == p.document_id == doc["id"]
            for sid in p.sentence_ids:
                # Sentence ids are document-scoped, a sibling of the passage id, and
                # `doc_id_of` splits on the first '#', so both forms resolve back to the
                # original doc-id that a citation has to carry.
                assert _SID_RE.match(sid)
                assert doc_id_of(sid) == doc["id"]


def test_adjacent_passages_share_overlap_sentences(
    overlap_doc, fake_segmenter, fake_tokenizer
) -> None:
    budget, frac = 100, 0.15
    passages = chunk_document(
        dict(overlap_doc), fake_segmenter, fake_tokenizer,
        token_budget=budget, overlap_frac=frac,
    )
    assert len(passages) >= 2
    shared = set(passages[0].sentence_ids) & set(passages[1].sentence_ids)
    assert shared  # the overlap head of passage 1 is the tail of passage 0
    # The overlap lands in a band around the configured 0.15, not on an exact figure,
    # because whole sentences are carried.
    counts = {sid: se_count for p in passages for sid, se_count in _sid_counts(p)}
    overlap_tokens = sum(counts[s] for s in shared)
    assert 0.08 * budget <= overlap_tokens <= 0.22 * budget


def _sid_counts(passage: Passage):
    # Each fake sentence is 10 whitespace words, so per-sentence counts divide evenly.
    words = passage.text.split()
    per = len(words) // max(1, len(passage.sentence_ids))
    return [(sid, per) for sid in passage.sentence_ids]


# --------------------------------------------------------------------------- #
# The pack boundary and the oversized own-passage rule.
# --------------------------------------------------------------------------- #
def test_pack_closes_exactly_at_the_budget_seam(
    tiny_native_docs, fake_segmenter, fake_tokenizer
) -> None:
    # In the boundary document s0 (4) and s1 (6) exactly fill a budget of 10, and s2 (1)
    # would take it to 11, so it opens the next passage.
    doc = dict(_by_id(tiny_native_docs, "eng-docs/0000001"))
    passages = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=10, overlap_frac=0.0
    )
    assert len(passages) == 2
    assert passages[0].token_count == 10 and len(passages[0].sentence_ids) == 2  # s0+s1
    assert passages[1].token_count == 1 and len(passages[1].sentence_ids) == 1  # s2 alone
    for p in passages:
        assert p.token_count <= 10 or p.is_oversized  # only an oversized passage may exceed


def test_every_non_oversized_passage_is_within_budget(
    overlap_doc, fake_segmenter, fake_tokenizer
) -> None:
    passages = chunk_document(
        dict(overlap_doc), fake_segmenter, fake_tokenizer, token_budget=100, overlap_frac=0.15
    )
    for p in passages:
        if not p.is_oversized:
            assert p.token_count <= 100


# --------------------------------------------------------------------------- #
# The overlap is a cap of about 15%: whole trailing sentences, stopping before it is passed.
# --------------------------------------------------------------------------- #
def test_overlap_tail_is_capped_at_target_paper_example() -> None:
    # The worked example: budget 512, so the overlap cap is 0.15 * 512 = 76.8. With trailing
    # sentences s14=40, s15=29, s16=17 and s17=17, the tail is {s15, s16, s17} = 63, because
    # adding s14 would reach 103 and the cap stops before it is passed. The next sentence is
    # small, so there is plenty of room and the cap rather than the room decides.
    prev = [
        ck._Sent(text="s14", count=40, sid="d#p0#s14"),
        ck._Sent(text="s15", count=29, sid="d#p0#s15"),
        ck._Sent(text="s16", count=17, sid="d#p0#s16"),
        ck._Sent(text="s17", count=17, sid="d#p0#s17"),
    ]
    tail = ck._overlap_tail(prev, overlap_frac=0.15, budget=512, next_count=10)
    assert [s.sid for s in tail] == ["d#p0#s15", "d#p0#s16", "d#p0#s17"]  # original order
    assert sum(s.count for s in tail) == 63  # within the 76.8 cap
    assert sum(s.count for s in tail) <= 0.15 * 512


def test_overlap_tail_floor_holds_only_when_it_fits_in_room() -> None:
    # The floor: a single trailing sentence bigger than the soft cap is still carried, as
    # long as it fits the room left by budget - next_count.
    prev = [ck._Sent(text="big", count=200, sid="d#p0#s0")]
    fits = ck._overlap_tail(prev, overlap_frac=0.15, budget=512, next_count=10)
    assert [s.sid for s in fits] == ["d#p0#s0"]  # 200 fits the room of 502

    # Room outranks the floor. If that one sentence would break the budget together with
    # the next sentence, the overlap is empty; carrying it would either exceed the budget
    # or force the next sentence to be split.
    busts = ck._overlap_tail(prev, overlap_frac=0.15, budget=512, next_count=400)
    assert busts == []  # 200 does not fit the room of 112


def test_non_oversized_passage_stays_within_budget_after_large_overlap(
    fake_segmenter, fake_tokenizer
) -> None:
    # The defect this pins: a passage ends on a large but not oversized sentence and a
    # normal sentence follows. The overlap floor carried that large trailing sentence into
    # the next passage, which then held 553 tokens against a budget of 512. With the room
    # cap the overlap is dropped and every non-oversized passage stays within budget.
    doc = {
        "id": "eng-docs/0000200",
        "text": f"{' '.join(['w'] * 5)}|{' '.join(['w'] * 90)}|{' '.join(['w'] * 30)}",
        "url": "u", "date": "d", "lang": "en",
    }
    passages = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=100, overlap_frac=0.15
    )
    for p in passages:
        assert not p.is_oversized  # no single sentence exceeds the 100-token budget
        assert p.token_count <= 100
    assert [p.token_count for p in passages] == [95, 30]  # p0 = w5 + w90, p1 = w30, no overlap


class _SpecialTokenizer:
    """Whitespace counter that reserves two special tokens, as bge-m3 does."""

    def count(self, text: str) -> int:
        return len(text.split())

    def num_special(self) -> int:
        return 2


def test_special_tokens_are_reserved_so_real_encoded_fits_the_window(fake_segmenter) -> None:
    # A budget of 12 with 2 reserved leaves a content budget of 10. Of three 4-word
    # sentences, s0 and s1 pack to 8, and s2 would make 12, so it opens a new passage. Every
    # non-oversized passage therefore encodes, content plus specials, inside the window the
    # retriever actually has.
    tok = _SpecialTokenizer()
    budget = 12
    doc = {
        "id": "eng-docs/0000021",
        "text": "a0 a1 a2 a3|b0 b1 b2 b3|c0 c1 c2 c3",
        "url": "u", "date": "d", "lang": "en",
    }
    passages = chunk_document(doc, fake_segmenter, tok, token_budget=budget, overlap_frac=0.0)
    for p in passages:
        if not p.is_oversized:
            assert p.token_count <= budget - tok.num_special()  # content within the budget
            assert p.token_count + tok.num_special() <= budget   # encoded within the window
    # The reservation bit: without it all three sentences would fit one 12-token passage.
    assert len(passages) == 2
    assert passages[0].token_count == 8 and passages[1].token_count == 4


# --------------------------------------------------------------------------- #
# The row shapes later stages read.
# --------------------------------------------------------------------------- #
def test_spine_rows_map_1to1_onto_common_document_and_sentence(
    tmp_path, make_cfg, tiny_native_docs, fake_segmenter, fake_tokenizer
) -> None:
    """The merged output is two Parquet tables read with ``iter_parquet``. Each row carries
    exactly the field set of a ``Document`` or a ``Sentence``, which is what every later
    stage relies on, and every sentence slices its document verbatim."""
    cfg = make_cfg()
    docs = [_by_id(tiny_native_docs, i) for i in ("eng-docs/0000001", "zho-docs/0000004")]
    _stage(tmp_path, cfg, docs)
    docs_path, sents_path = chunk(
        cfg, base=tmp_path, segmenter=fake_segmenter, tokenizer=fake_tokenizer
    )
    assert docs_path.suffix == sents_path.suffix == ".parquet"

    doc_rows = list(iter_parquet(docs_path))
    assert doc_rows
    by_id: dict[str, str] = {}
    for rec in doc_rows:
        assert set(rec) == {"document_id", "lang", "text"}
        assert isinstance(rec["text"], str) and rec["text"]
        d = Document(**rec)
        assert d.document_id == rec["document_id"]  # the row round-trips into the dataclass
        by_id[rec["document_id"]] = rec["text"]

    sent_rows = list(iter_parquet(sents_path))
    assert sent_rows
    seen: dict[str, list[int]] = {}
    for rec in sent_rows:
        assert set(rec) == {
            "sentence_id", "document_id", "sentence_index", "lang",
            "start", "end", "paragraph_index", "token_count",
        }
        for k in ("sentence_index", "start", "end", "paragraph_index", "token_count"):
            assert isinstance(rec[k], int)
        s = Sentence(**rec)
        assert s.sentence_id == rec["sentence_id"]
        # The sentence is a span of its document: verbatim, NFC and non-empty.
        text = by_id[rec["document_id"]]
        assert 0 <= rec["start"] <= rec["end"] <= len(text)
        span = text[rec["start"] : rec["end"]]
        assert span and span.strip() == span and unicodedata.is_normalized("NFC", span)
        assert rec["sentence_id"] == f"{rec['document_id']}#s{rec['sentence_index']}"
        seen.setdefault(rec["document_id"], []).append(rec["sentence_index"])

    # Ids are dense per document, 0 to n-1, with no gaps and no duplicates.
    for doc_id, idxs in seen.items():
        assert idxs == list(range(len(idxs))), doc_id


def test_spans_tile_the_document_leaving_no_content_behind(
    tmp_path, make_cfg, tiny_native_docs, fake_segmenter, fake_tokenizer
) -> None:
    """Consecutive sentence spans do not overlap and do not skip content. Whatever lies
    between two adjacent spans, or before the first and after the last, is separator
    material only. A gap carrying content would mean a sentence had been dropped.

    The fake segmenter consumes its ``|`` markers, whereas the real SaT reproduces its
    input verbatim, so here the gaps are ``|`` where a real document's would be whitespace,
    and the marker is excluded from the check. The stricter whitespace-only form of the
    same assertion runs against the real segmenter in ``test_chunk_byte_identity_full.py``.
    """
    cfg = make_cfg()
    # The four distinct ids; the fixture carries two copies of one of them.
    docs = [_by_id(tiny_native_docs, i) for i in (
        "eng-docs/0000001", "rus-docs/0000002", "spa-docs/0000003", "zho-docs/0000004",
    )]
    _stage(tmp_path, cfg, docs)
    docs_path, sents_path = chunk(
        cfg, base=tmp_path, segmenter=fake_segmenter, tokenizer=fake_tokenizer
    )
    texts = {r["document_id"]: r["text"] for r in iter_parquet(docs_path)}
    per_doc: dict[str, list[tuple[int, int]]] = {}
    for r in iter_parquet(sents_path):
        per_doc.setdefault(r["document_id"], []).append((r["start"], r["end"]))
    assert per_doc
    for doc_id, spans in per_doc.items():
        text = texts[doc_id]
        pos = 0
        for start, end in spans:  # stored in document order
            assert start >= pos and end > start  # ordered, non-overlapping, non-empty
            assert not text[pos:start].strip("| \t\n")  # nothing but separators in the gap
            pos = end
        assert not text[pos:].strip("| \t\n")


def test_arrow_schemas_cannot_drift_from_their_dataclasses() -> None:
    """Each schema's column order is its dataclass's field order. A field added to one and
    not the other would null-fill or drop a column at merge time, without an error."""
    import dataclasses

    from ragtime.common import (
        document_arrow_schema,
        passage_arrow_schema,
        sentence_arrow_schema,
    )

    for schema, record in (
        (document_arrow_schema(), Document),
        (sentence_arrow_schema(), Sentence),
        (passage_arrow_schema(), Passage),
    ):
        assert list(schema.names) == [f.name for f in dataclasses.fields(record)]


def test_sentence_offsets_are_int32() -> None:
    """int32 rather than int64. The longest document in the collection is 23,917 code
    points, so the margin is large, and at about 95 M rows this is the biggest table
    written, where halving the width is worth having."""
    import pyarrow as pa

    from ragtime.common import sentence_arrow_schema

    schema = sentence_arrow_schema()
    for name in ("sentence_index", "start", "end", "paragraph_index", "token_count"):
        assert schema.field(name).type == pa.int32()


def test_chunk_consumes_read_native_dicts_directly(
    gzip_jsonl_fixture, fake_segmenter, fake_tokenizer
) -> None:
    """The reader's dicts go straight into the chunker, with no renaming layer between."""
    doc = next(iter(corpus.read_native(gzip_jsonl_fixture["docs"])))
    passages = chunk_document(
        doc, fake_segmenter, fake_tokenizer, token_budget=512, overlap_frac=0.15
    )
    assert passages and passages[0].document_id == doc["id"]


def test_default_base_constant_matches_cli_local_root() -> None:
    from ragtime.orchestration.cli import _LOCAL_ROOT

    assert ck._DEFAULT_BASE == _LOCAL_ROOT
