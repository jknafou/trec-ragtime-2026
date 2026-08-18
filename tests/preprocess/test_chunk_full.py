"""Bounded-shard chunking through the real work queue.

Everything here is the shipped machinery: the SaT segmenter on onnx CPU, the pinned
``BAAI/bge-m3`` tokenizer, real multilingual text, the ``saturate`` driver and the
``ChunkAdapter``. Only the corpus is cut down. Skipped where the corpus or SaT is
unavailable.

The slice is 50 documents per language, 200 in all. That is enough real length variance,
oversized sentences and multi-shard draining to exercise the lifecycle, and it finishes in a
few minutes on a CPU node. Throughput at corpus scale is a capacity question and is measured
elsewhere.
"""

from __future__ import annotations

import gzip
import itertools
import json
import unicodedata
from pathlib import Path

import pytest

from ragtime.common import Layout, doc_id_of, iter_parquet
from ragtime.common.io import is_done
from ragtime.config import all_hashes, load
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family

pytestmark = pytest.mark.full

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_PER_LANG = 50  # 200 documents in all, sized for a few minutes on a CPU node
_STEMS = ("eng-docs", "spa-docs", "rus-docs", "zho-docs")


def _stage_bounded_slice(base: Path, fam: str, chunker_hash: str) -> None:
    """Head-slice ``_DOCS_PER_LANG`` whole docs per language into the family raw store."""
    from huggingface_hub import hf_hub_download

    raw = Layout(run_dir=base, base=base).corpus_raw_dir(fam, chunker_hash)
    raw.mkdir(parents=True, exist_ok=True)
    for stem in _STEMS:
        src = hf_hub_download(
            repo_id="trec-ragtime/ragtime2",
            repo_type="dataset",
            filename=f"{stem}.jsonl.gz",
        )
        out = raw / f"{stem}.jsonl.gz"
        with gzip.open(src, "rt", encoding="utf-8") as fin, gzip.open(
            out, "wt", encoding="utf-8"
        ) as fout:
            for i, line in enumerate(fin):
                if i >= _DOCS_PER_LANG:
                    break
                json.loads(line)  # slice whole records, never mid-record
                fout.write(line)
    # Mark the bounded store done so `download` treats it as complete and does not re-fetch.
    (raw.parent / f"{raw.name}._SUCCESS").write_bytes(b"")


@pytest.fixture
def bounded_cfg():
    return load(_REPO_ROOT / "config" / "e2e-original.yml")


def test_bounded_chunk_through_real_workqueue(tmp_path, bounded_cfg, monkeypatch) -> None:
    pytest.importorskip("wtpsplit")
    # Declare the pool width rather than inherit it. `_chunk_workers` falls through to
    # `default_chunk_workers()`, which returns SLURM_CPUS_PER_TASK as-is, so the memory this
    # test wants would be whatever node it landed on: each worker holds a resident SaT plus
    # bge-m3 tokenizer at a measured 1,966 MiB, so a 48-CPU node would want around 92 GiB.
    # Past the memory grant the cgroup kills a pool child and the parent then blocks in
    # `imap` on a result that can never arrive, which looks from outside like a hang.
    # Two workers keep the real pool path, since `bringup` takes it whenever workers > 1 and
    # a width of 1 is the sequential path this test is not about, at around 3.89 GiB.
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    from ragtime.preprocess import ChunkAdapter

    fam = run_family(bounded_cfg)
    ch = all_hashes(bounded_cfg)["chunker"]
    try:
        _stage_bounded_slice(tmp_path, fam, ch)
    except Exception as exc:  # noqa: BLE001, offline / repo unavailable -> skip
        pytest.skip(f"bounded corpus slice unavailable: {exc}")

    adapter = ChunkAdapter(base=str(tmp_path))
    wq = saturate.queue_for(bounded_cfg, adapter, base=str(tmp_path))

    # Document-balanced sharding: many shards, each within the configured document budget,
    # together covering every document with no overlap and no gap. That is the property
    # that lets N workers run in parallel without duplicating or losing work.
    specs = list(adapter.shards(bounded_cfg))
    corpus_shards = int(bounded_cfg.blocks["execution"]["corpus_shards"])
    total_docs = len(_STEMS) * _DOCS_PER_LANG
    per = -(-total_docs // corpus_shards)  # ceil doc budget
    assert len(specs) > len(_STEMS)  # many shards, not one-per-language
    ranges: dict[str, list[tuple[int, int]]] = {}
    for s in specs:
        assert s.payload["end"] - s.payload["start"] <= per  # each within the doc budget
        ranges.setdefault(s.payload["docs_file"], []).append(
            (s.payload["start"], s.payload["end"])
        )
    for rs in ranges.values():
        rs.sort()
        assert rs[0][0] == 0 and rs[-1][1] == _DOCS_PER_LANG  # covers the whole file
        for (a0, a1), (b0, b1) in itertools.pairwise(rs):
            assert a1 == b0  # contiguous: no overlap, no gap

    n = saturate.seed(bounded_cfg, adapter, wq)
    assert n == len(specs)
    saturate.run_worker(bounded_cfg, adapter, wq)
    saturate.drive(bounded_cfg, adapter, wq)

    layout = Layout(run_dir=tmp_path, base=tmp_path, family=fam, chunker_hash=ch)
    docs_out, sents_out = layout.documents_path(), layout.sentences_path()
    assert is_done(docs_out) and is_done(sents_out)
    assert docs_out.suffix == sents_out.suffix == ".parquet"  # Parquet+zstd spine
    assert not (docs_out.parent / "passages").exists()  # chunk writes no passages

    # Read as a stream; the corpus is never materialised whole.
    texts = {r["document_id"]: r["text"] for r in iter_parquet(docs_out)}
    assert texts
    sents = list(iter_parquet(sents_out))
    assert sents
    seen: dict[str, int] = {}
    for r in sents:
        text = texts[r["document_id"]]
        # On real multilingual text: every sentence is a verbatim, NFC, in-bounds span of
        # its own document, and the ids are dense in document order.
        assert 0 <= r["start"] <= r["end"] <= len(text)
        span = text[r["start"] : r["end"]]
        assert span and span.strip() == span
        assert unicodedata.is_normalized("NFC", span)
        assert r["sentence_id"] == f"{r['document_id']}#s{r['sentence_index']}"
        assert r["sentence_index"] == seen.get(r["document_id"], 0)
        seen[r["document_id"]] = r["sentence_index"] + 1
        assert doc_id_of(r["sentence_id"]) == r["document_id"]  # citations resolve to the doc

    # Re-driving a finalised queue does no work at all.
    before = (docs_out.read_bytes(), sents_out.read_bytes())
    saturate.seed(bounded_cfg, adapter, wq)
    saturate.run_worker(bounded_cfg, adapter, wq)
    saturate.drive(bounded_cfg, adapter, wq)
    assert (docs_out.read_bytes(), sents_out.read_bytes()) == before


def test_pipeline_array_size_is_not_a_hardcoded_one(bounded_cfg) -> None:
    """The PIPELINE array size comes from the _topic_shards hook, not a literal 1."""
    from ragtime.orchestration import plan

    dag = plan.build_plan(bounded_cfg)
    n_seeds = len(plan.expand_seeds(bounded_cfg))
    assert dag.node(plan.PIPELINE).array_size == n_seeds * plan._topic_shards(bounded_cfg)
    assert callable(plan._topic_shards)  # one source for the count, not an inline constant
