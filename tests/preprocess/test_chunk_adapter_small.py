"""Document-balanced sharding and reassembly, through the driver.

The real ``ChunkAdapter`` runs under the real ``saturate`` driver, with a fake segmenter and
tokenizer injected so nothing loads torch or hits the network. The point of the file is that
sharding and merging give byte-identical output to a whole-corpus ``chunk(cfg)``: running in
parallel does not change the ids or their order.
"""

from __future__ import annotations

import gzip
import importlib
import itertools
import json
from pathlib import Path

import pytest

from ragtime.common import Layout, iter_parquet
from ragtime.config import all_hashes
from ragtime.orchestration import saturate
from ragtime.preprocess.chunk import ChunkAdapter, _ChunkCtx, _shard_slices

pytestmark = pytest.mark.small

ck = importlib.import_module("ragtime.preprocess.chunk")


def _stage(base: Path, cfg, docs_by_stem: dict[str, list[dict]]) -> Layout:
    fam = "e2e"
    ch = all_hashes(cfg)["chunker"]
    layout = Layout(run_dir=base, base=base, family=fam, chunker_hash=ch)
    raw = layout.corpus_raw_dir(fam, ch)
    raw.mkdir(parents=True, exist_ok=True)
    for stem, rows in docs_by_stem.items():
        with gzip.open(raw / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return layout


def test_shard_slices_are_doc_balanced_with_no_overlap_or_gap() -> None:
    counts = {"eng-docs.jsonl.gz": 10, "spa-docs.jsonl.gz": 7}
    slices = list(_shard_slices(counts, corpus_shards=6))
    per = -(-17 // 6)  # ceil(17/6) == 3
    assert len(slices) > 2  # many shards, not one-per-file
    per_file: dict[str, list[tuple[int, int]]] = {}
    for fname, start, end in slices:
        assert end - start <= per
        per_file.setdefault(fname, []).append((start, end))
    for fname, rs in per_file.items():
        rs.sort()
        assert rs[0][0] == 0 and rs[-1][1] == counts[fname]
        for (a0, a1), (b0, b1) in itertools.pairwise(rs):
            assert a1 == b0  # contiguous


def test_adapter_sharded_merge_equals_whole_corpus_chunk(
    tmp_path, make_cfg, fake_segmenter, fake_tokenizer, monkeypatch
) -> None:
    cfg = make_cfg(token_budget=20, overlap_frac=0.15)
    # Enough documents to span many shards.
    docs = {
        "eng-docs": [
            {"id": f"eng-docs/{i:07d}", "text": "a b c|d e|f g h i", "url": "u", "date": "d"}
            for i in range(12)
        ],
        "spa-docs": [
            {"id": f"spa-docs/{i:07d}", "text": "uno dos|tres cuatro cinco", "url": "u", "date": "d"}
            for i in range(9)
        ],
    }
    direct_base = tmp_path / "direct"
    _stage(direct_base, cfg, docs)
    direct_docs, direct_sents = ck.chunk(
        cfg, base=direct_base, segmenter=fake_segmenter, tokenizer=fake_tokenizer
    )
    direct_doc_records = list(iter_parquet(direct_docs))
    direct_sent_records = list(iter_parquet(direct_sents))
    assert direct_doc_records and direct_sent_records

    # Force many shards so the document-balanced path is taken. corpus_shards lives in the
    # execution block, not the chunker one, so the chunker hash and the corpus path stay
    # the same as in the direct run and the two outputs remain comparable.
    cfg.blocks["execution"]["corpus_shards"] = 8
    ch = all_hashes(cfg)["chunker"]
    adapter_base = tmp_path / "adapter"
    layout = _stage(adapter_base, cfg, docs)
    adapter = ChunkAdapter(base=str(adapter_base))
    ctx = _ChunkCtx(
        segmenter=fake_segmenter,
        tokenizer=fake_tokenizer,
        raw_dir=layout.corpus_raw_dir("e2e", ch),
        batch_size=4,  # small enough that each shard makes several batched split calls
        # Read each shard from the slice file the seed materialised, which is the fast read
        # path, so the equality below also covers slice-versus-whole-file agreement.
        slices_dir=layout.corpus_slices_dir("e2e", ch),
    )
    # Drive the real lifecycle, with the fake context standing in for model bring-up.
    monkeypatch.setattr(ChunkAdapter, "bringup", lambda self, _cfg: ctx)
    wq = saturate.queue_for(cfg, adapter, base=str(adapter_base))
    n = saturate.seed(cfg, adapter, wq)
    assert n > 2  # genuinely many shards
    saturate.run_worker(cfg, adapter, wq, backoff_s=0.0, max_iters=200)
    saturate.drive(cfg, adapter, wq, poll_s=0.0, max_polls=5)

    # Sharded and merged equals whole-corpus for both tables: the same records in the same
    # order, and since both sides are Parquet, the same bytes.
    merged_docs, merged_sents = layout.documents_path(), layout.sentences_path()
    assert list(iter_parquet(merged_docs)) == direct_doc_records
    assert list(iter_parquet(merged_sents)) == direct_sent_records
    assert merged_docs.read_bytes() == direct_docs.read_bytes()
    assert merged_sents.read_bytes() == direct_sents.read_bytes()
    # The work-queue path writes no passage artefact either.
    assert not (merged_docs.parent / "passages").exists()


def _spine_rec(doc_id: str, sents: list[str], paras: list[int], lang: str = "en") -> dict:
    """One nested shard record: the document text plus its sentences as spans of it."""
    text = " ".join(sents)
    rows, pos = [], 0
    for j, (s, para) in enumerate(zip(sents, paras, strict=True)):
        if j:
            pos += 1  # the single-space joiner
        rows.append(
            {
                "sentence_id": f"{doc_id}#s{j}",
                "document_id": doc_id,
                "sentence_index": j,
                "lang": lang,
                "start": pos,
                "end": pos + len(s),
                "paragraph_index": para,
                "token_count": len(s.split()),
            }
        )
        pos += len(s)
    return {"document_id": doc_id, "lang": lang, "text": text, "sentences": rows}


def test_merge_stream_equals_the_shard_record_stream_in_sorted_order(
    tmp_path, make_cfg, monkeypatch
) -> None:
    """The merge turns nested JSONL shards into two Parquet tables in constant memory.

    Its output is exactly the projections of the shard record stream, in shard-sorted
    order, written through the temp-then-rename and ``_SUCCESS`` contract.
    """
    from collections.abc import Iterator

    from ragtime.common import iter_parquet, read_jsonl, write_jsonl
    from ragtime.common.io import is_done

    cfg = make_cfg()
    # Shard outputs as write_jsonl produces them, carrying the cases where byte stability
    # is easiest to lose: non-ASCII, CJK, Cyrillic and multi-paragraph spans.
    shard_records = [
        [_spine_rec("eng-docs/0000001", ["café: résumé"], [0])],
        [
            _spine_rec("spa-docs/0000002", ["第二个 doc"], [0], lang="zh"),
            _spine_rec("spa-docs/0000004", ["una línea.", "y otra"], [0, 1], lang="es"),
        ],
        [_spine_rec("rus-docs/0000003", ["тест"], [2], lang="ru")],
    ]
    out_dir = tmp_path / "wq" / "out"
    shard_paths = []
    for i, recs in enumerate(shard_records):
        p = out_dir / f"shard_{i:04d}"
        write_jsonl(p, recs)
        shard_paths.append(p)

    # The reference: parse every shard in sorted order, then project.
    nested: list[dict] = []
    for p in sorted(shard_paths):
        nested.extend(read_jsonl(p))
    want_docs = [{k: r[k] for k in ("document_id", "lang", "text")} for r in nested]
    want_sents = [s for r in nested for s in r["sentences"]]

    adapter = ChunkAdapter(base=str(tmp_path / "family"))
    # Pass the shards unsorted; sorting them is merge's job.
    seen: list[object] = []
    real_writer = ck.write_parquet_stream

    def _spy(path, records, **kw):
        seen.append(records)
        return real_writer(path, records, **kw)

    monkeypatch.setattr(ck, "write_parquet_stream", _spy)
    adapter.merge(cfg, list(reversed(shard_paths)))
    monkeypatch.undo()
    # Constant memory: merge hands the writer lazy streams rather than materialised lists.
    assert len(seen) == 2 and all(isinstance(s, Iterator) for s in seen)

    layout = adapter._layout(cfg)
    merged_docs, merged_sents = layout.documents_path(), layout.sentences_path()
    assert merged_docs.suffix == merged_sents.suffix == ".parquet"
    assert list(iter_parquet(merged_docs)) == want_docs  # same records, same order
    assert list(iter_parquet(merged_sents)) == want_sents
    assert is_done(merged_docs) and is_done(merged_sents)
    # The sentence grain that verbatim claim commit depends on survives the round trip.
    texts = {r["document_id"]: r["text"] for r in iter_parquet(merged_docs)}
    for rec in iter_parquet(merged_sents):
        assert rec["sentence_id"] == f"{rec['document_id']}#s{rec['sentence_index']}"
        assert texts[rec["document_id"]][rec["start"] : rec["end"]]
