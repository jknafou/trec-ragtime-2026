"""The multiprocessing chunk path gives the same bytes as the sequential one.

``_ChunkPool`` has to produce the same records in the same order as the sequential core:
the ``{document, sentences}`` records it emits, and the passages derived from them, down to
sentence ids, spans, ``paragraph_index``, NFC and overlap.

The fakes are injected into real ``spawn`` worker processes through the ``model_factory``
seam. They are module-level classes so the child can unpickle them by reference, and they
pull in neither wtpsplit nor the network, which leaves the real pool, imap and ordering
machinery as the thing under test. Worker-count resolution is covered separately at the
bottom of the file.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from ragtime.preprocess.chunk import (
    _chunk_workers,
    _ChunkPool,
    _pool_init,
    _sents_of,
    chunk_documents,
    default_chunk_workers,
    segment_documents,
)

pytestmark = pytest.mark.small


# --------------------------------------------------------------------------- #
# Module-level fakes. The spawn child imports this module to unpickle the factory, so they
# have to live here rather than in conftest, and they have to be dependency-free.
# --------------------------------------------------------------------------- #
class _FakeSegmenter:
    """Splits on the ``|`` marker, so a fixture controls its own sentence boundaries."""

    def split(self, text: str) -> list[str]:
        return [s.strip() for s in text.split("|") if s.strip()]

    def split_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.split(t) for t in texts]

    def split_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        pos = 0
        for raw in text.split("|"):
            seg = raw.strip()
            if seg:
                a = pos + (len(raw) - len(raw.lstrip()))
                spans.append((a, a + len(seg)))
            pos += len(raw) + 1  # +1 for the consumed "|" separator
        return spans

    def split_spans_batch(self, texts: list[str]) -> list[list[tuple[int, int]]]:
        return [self.split_spans(t) for t in texts]


class _FakeTokenizer:
    """Whitespace word counts, and no special tokens, so the content budget is the budget."""

    def count(self, text: str) -> int:
        return len(text.split())

    def num_special(self) -> int:
        return 0


def _fake_models() -> tuple[_FakeSegmenter, _FakeTokenizer]:
    return _FakeSegmenter(), _FakeTokenizer()


def _doc(doc_id: str, text: str, lang: str = "en") -> dict:
    return {"id": doc_id, "text": text, "url": "u", "date": "d", "lang": lang}


def _nontrivial_docs() -> list[dict]:
    """Documents covering paragraphs, boilerplate, overlap, oversized sentences and NFC."""
    cafe_nfd = "café"  # decomposed é: NFC normalisation has to agree across processes
    long_sentence = " ".join(f"q{i}" for i in range(30))  # oversized vs budget 12
    many = "|".join(" ".join(f"s{i}_{j}" for j in range(5)) for i in range(9))
    return [
        _doc(
            "eng-docs/0000001",
            "Home › News › Politics\nFirst para sentence one.|First para two.\n\n"
            "Second para only.\n\n© 2024 X. All rights reserved.",
        ),
        _doc("rus-docs/0000002", f"Первое предложение.|{long_sentence}|Последнее.", "ru"),
        _doc("spa-docs/0000003", f"El {cafe_nfd} tiene dos gramos|Segunda frase corta", "es"),
        _doc("zho-docs/0000004", "第一 句子|第二 句子\n\n第三 句子", "zh"),
        _doc("eng-docs/0000005", many),  # several passages, so overlap is carried
        _doc(
            "eng-docs/0000006",
            "Share | Tweet | Email\n© 2024 Y. All rights reserved.",
        ),  # entirely boilerplate, so it is dropped
        _doc("spa-docs/0000007", "Una sola frase.", "es"),
    ]


_PACK = {"token_budget": 12, "overlap_frac": 0.15}
_OPTS = {
    "strip_boilerplate": True,
    "boilerplate_rules_version": "v1",
    "prefer_paragraph_break": True,
    "prefer_min_fill": 0.6,
}


_SEG_OPTS = {k: _OPTS[k] for k in ("strip_boilerplate", "boilerplate_rules_version")}


def _record_lines(passages) -> list[str]:
    """The serialised passage lines the byte comparison runs over."""
    return [json.dumps(asdict(p), ensure_ascii=False) for p in passages]


def _spine_lines(records) -> list[str]:
    """The serialised record lines, as ``ChunkAdapter.work`` writes them."""
    return [json.dumps(r, ensure_ascii=False) for r in records]


@pytest.fixture(scope="module")
def shared_pool():
    """One spawn pool for the whole module. Bring-up is the slow part, and reusing it also
    matches the resident pool the adapter keeps."""
    pool = _ChunkPool(2, "unused-model", "unused-tokenizer", model_factory=_fake_models)
    yield pool
    pool.close()


@pytest.mark.parametrize("batch_size", [2, 3, 64])
def test_pool_output_byte_identical_to_sequential(shared_pool, batch_size: int) -> None:
    """Records and order out of a real spawn pool match the sequential path exactly."""
    docs = _nontrivial_docs()
    seg, tok = _fake_models()
    reference = _spine_lines(
        segment_documents(docs, seg, tok, batch_size=batch_size, **_SEG_OPTS)
    )
    parallel = _spine_lines(
        shared_pool.segment_documents(docs, batch_size=batch_size, **_SEG_OPTS)
    )
    assert parallel == reference  # the same lines in the same order


def test_pool_derived_passages_are_byte_identical_to_sequential(shared_pool) -> None:
    """The passages derived from the pool's records match the sequential packer.

    The records are what ships, but reconcile derives passages from them, and a divergence
    in span offsets between the two paths would show up here as different passage text even
    if the rows themselves compared equal.
    """
    docs = _nontrivial_docs()
    seg, tok = _fake_models()
    reference = _record_lines(chunk_documents(docs, seg, tok, batch_size=3, **_PACK, **_OPTS))
    parallel = _record_lines(
        _pack(shared_pool.segment_documents(docs, batch_size=3, **_SEG_OPTS))
    )
    assert parallel == reference


def _pack(records):
    """Derive passages from records with the same settings as ``chunk_documents``."""
    from ragtime.preprocess.chunk import _pack_sentences

    tok = _FakeTokenizer()
    for rec in records:
        yield from _pack_sentences(
            rec["document_id"], rec["lang"], _sents_of(rec), tok,
            token_budget=_PACK["token_budget"], overlap_frac=_PACK["overlap_frac"],
            prefer_paragraph_break=_OPTS["prefer_paragraph_break"],
            prefer_min_fill=_OPTS["prefer_min_fill"],
        )


def test_pool_emits_exact_sentence_spans(shared_pool) -> None:
    """The parallel path's sentence spans slice their own document exactly.

    The comparisons above check the record bytes. This one checks the meaning across the
    process boundary: every sentence is a verbatim, NFC span of its own document text. A
    worker cannot pass by emitting offsets that happen to equal a broken reference.
    """
    from ragtime.common import nfc

    records = list(shared_pool.segment_documents(_nontrivial_docs(), batch_size=3,
                                                 **_SEG_OPTS))
    assert records
    assert any(len(r["sentences"]) > 1 for r in records)  # multi-sentence documents included
    for rec in records:
        text = rec["text"]
        assert [s["sentence_index"] for s in rec["sentences"]] == list(
            range(len(rec["sentences"]))
        )
        for s in rec["sentences"]:
            slice_ = text[s["start"] : s["end"]]
            assert slice_ and slice_.strip() == slice_ and nfc(slice_) == slice_
            assert s["sentence_id"] == f"{rec['document_id']}#s{s['sentence_index']}"


def test_pool_reuse_across_calls_stays_identical(shared_pool) -> None:
    """A pool held across shards, as the adapter holds it, keeps producing the same output."""
    docs = _nontrivial_docs()
    seg, tok = _fake_models()
    reference = _spine_lines(segment_documents(docs, seg, tok, batch_size=3, **_SEG_OPTS))
    first = _spine_lines(shared_pool.segment_documents(docs, batch_size=3, **_SEG_OPTS))
    second = _spine_lines(shared_pool.segment_documents(docs, batch_size=3, **_SEG_OPTS))
    assert first == reference
    assert second == reference


def test_pool_init_pins_every_native_thread_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each worker really is one thread.

    Importing SaT pulls in scipy and sklearn, which bring up OpenBLAS, and OpenBLAS
    defaults to 128 threads on a large node whatever ``intra_op_threads`` says. Before this
    pin a worker measured 129 threads and 1.3 GB of RSS, so a pool of them oversubscribed
    the cpuset and was killed inside the memory cgroup.
    """
    for var in (
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(var, "128")  # restored on teardown
    _pool_init("unused-model", "unused-tokenizer", model_factory=_fake_models)
    import os

    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        assert os.environ[var] == "1"


# --------------------------------------------------------------------------- #
# Worker count: the config knob wins, then the cpuset, then a capped cpu_count.
# --------------------------------------------------------------------------- #
def test_default_workers_prefers_slurm_cpuset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert default_chunk_workers() == 4


def test_default_workers_fallback_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    import importlib

    chunk_mod = importlib.import_module("ragtime.preprocess.chunk")
    assert 1 <= default_chunk_workers() <= chunk_mod._FALLBACK_WORKER_CAP


def test_chunk_workers_config_knob_wins(monkeypatch: pytest.MonkeyPatch, make_cfg) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    cfg = make_cfg()
    cfg.blocks["execution"]["chunk_workers"] = 3
    assert _chunk_workers(cfg) == 3
    del cfg.blocks["execution"]["chunk_workers"]
    assert _chunk_workers(cfg) == 4  # falls back to the cpuset default
