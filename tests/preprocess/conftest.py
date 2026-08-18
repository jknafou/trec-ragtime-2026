"""Fixtures for the corpus-spine tests (chunk, download, corpus, tokenizer).

Nothing here imports torch, transformers or wtpsplit at collection time, and nothing
touches the network or SLURM. The chunk core runs against a fake segmenter and a fake
tokenizer that counts whitespace-separated words; ``download`` runs against a mocked
``snapshot_download``. The tracked ``config/*.yml`` are opened read-only, and a config
that needs a filled ``chunker.config`` block is built in memory as a ``SimpleNamespace``
so no tracked file is ever edited.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import types
from pathlib import Path

import pytest

# The bge-m3 revision pinned by the shared chunker block.
_TOKENIZER_ID = "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181"
_SEGMENTER_MODEL = "sat-3l-sm"


# --------------------------------------------------------------------------- #
# Fake models: deterministic, dependency-free stand-ins for SaT + bge-m3.
# --------------------------------------------------------------------------- #
class FakeSegmenter:
    """Splits on the ``|`` marker, so a fixture controls its own sentence boundaries.

    It mirrors the real ``Segmenter``'s four-method surface: ``split``/``split_batch``
    return sentence strings, ``split_spans``/``split_spans_batch`` return ``(start, end)``
    offsets into the input, and the two agree, ``[t[a:b] for a, b in split_spans(t)] ==
    split(t)``. The span walk cannot reuse ``serving.segmenter.spans_of`` because this fake
    consumes its ``|`` separators whereas a real SaT segmentation reproduces its input
    verbatim, so the cursor advances one extra character per separator.
    """

    def split(self, text: str) -> list[str]:
        return [s.strip() for s in text.split("|") if s.strip()]

    def split_batch(self, texts: list[str]) -> list[list[str]]:
        """Batched shape of the real Segmenter: ``split`` applied per text."""
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


class FakeTokenizer:
    """Counts whitespace-separated tokens: ``count("") == 0``, monotone under concatenation.

    It adds no special tokens, so the content budget equals ``token_budget`` and the
    boundary fixtures below stay exact. The retriever's two-token reservation for bge-m3 is
    covered separately by a fake that reports a non-zero ``num_special``.
    """

    def count(self, text: str) -> int:
        return len(text.split())

    def num_special(self) -> int:
        return 0


@pytest.fixture
def fake_segmenter() -> FakeSegmenter:
    return FakeSegmenter()


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def _cfg(token_budget: int = 512, overlap_frac: float = 0.15) -> types.SimpleNamespace:
    """A minimal cfg exposing ``run_id`` and a filled ``chunker.config`` block.

    ``run_family(cfg)`` reads only ``cfg.run_id`` and chunk/tokenizer read
    ``cfg.blocks["chunker"]["config"]``, so a ``SimpleNamespace`` covers everything the
    stages touch without loading a tracked config.
    """
    return types.SimpleNamespace(
        run_id="e2e-original",
        blocks={
            "chunker": {
                "config": {
                    "token_budget": token_budget,
                    "overlap_frac": overlap_frac,
                    "segmenter_model": _SEGMENTER_MODEL,
                    "tokenizer_id": _TOKENIZER_ID,
                    # Semantic knobs, mirroring the shared chunker.config.
                    "strip_boilerplate": True,
                    "boilerplate_rules": "v1",
                    "prefer_paragraph_break": True,
                    "prefer_paragraph_break_min_fill": 0.6,
                }
            },
            # Execution knobs live in their own block; they are not chunking semantics.
            "execution": {"corpus_shards": 100, "oversubscription": 5},
        },
    )


@pytest.fixture
def make_cfg():
    return _cfg


@pytest.fixture
def chunk_cfg() -> types.SimpleNamespace:
    return _cfg()


# --------------------------------------------------------------------------- #
# tiny_native_docs: a handful of synthetic native docs across zh/en/ru/es with the
# three documented shapes: (a) oversized sentence, (b) clean 2-passage boundary,
# (c) precomposed-vs-decomposed é for NFC symmetry. Sentences are '|'-separated so
# FakeSegmenter splits them exactly; token counts are whitespace-word counts.
# --------------------------------------------------------------------------- #
def _words(n: int, w: str = "w") -> str:
    return " ".join(f"{w}{i}" for i in range(n))


@pytest.fixture
def tiny_native_docs() -> list[dict]:
    # (b) boundary doc: with budget=10, s0=4 + s1=6 -> exactly 10 (one passage); s2=1
    # would overflow to 11 -> starts the second passage.
    boundary = {
        "id": "eng-docs/0000001",
        "text": f"{_words(4, 'a')}|{_words(6, 'b')}|{_words(1, 'c')}",
        "url": "http://x/1",
        "date": "2026-01-01",
        "lang": "en",
    }
    # (a) oversized doc: a single 15-word sentence > budget(10), between two short ones.
    oversized = {
        "id": "rus-docs/0000002",
        "text": f"{_words(3, 'p')}|{_words(15, 'q')}|{_words(2, 'r')}",
        "url": "http://x/2",
        "date": "2026-01-02",
        "lang": "ru",
    }
    # (c) NFC pair: same content, precomposed 'é' vs decomposed 'é'.
    cafe = "café"  # U+00E9
    cafe_nfd = "café"  # e + U+0301
    nfc_pre = {
        "id": "spa-docs/0000003",
        "text": f"El {cafe} tiene dos gramos|Segunda oracion corta",
        "url": "http://x/3",
        "date": "2026-01-03",
        "lang": "es",
    }
    nfc_dec = {**nfc_pre, "text": f"El {cafe_nfd} tiene dos gramos|Segunda oracion corta"}
    # a zh doc so all four langs are represented.
    zho = {
        "id": "zho-docs/0000004",
        "text": f"{_words(2, 'z')}|{_words(2, 'y')}",
        "url": "http://x/4",
        "date": "2026-01-04",
        "lang": "zh",
    }
    return [boundary, oversized, nfc_pre, nfc_dec, zho]


# --------------------------------------------------------------------------- #
# overlap_docs: a doc that packs into >=2 passages with ~10-20% sentence overlap.
# budget=100, ten 10-token sentences fill passage0; the 11th overflows; the overlap
# tail carries the trailing ~20 tokens (2 sentences) into passage1.
# --------------------------------------------------------------------------- #
@pytest.fixture
def overlap_doc() -> dict:
    sents = "|".join(_words(10, f"s{i}_") for i in range(14))
    return {
        "id": "eng-docs/0000009",
        "text": sents,
        "url": "http://x/9",
        "date": "2026-01-09",
        "lang": "en",
    }


# --------------------------------------------------------------------------- #
# gzip_jsonl_fixture: a tiny real gzip JSONL pair (native + trans) for corpus reads.
# --------------------------------------------------------------------------- #
@pytest.fixture
def gzip_jsonl_fixture(tmp_path: Path) -> dict[str, Path]:
    native = [
        {"id": "spa-docs/0000010", "text": "Hola mundo.", "url": "http://a", "date": "2026-01-01"},
        {"id": "spa-docs/0000011", "text": "Segundo doc.", "url": "http://b", "date": "2026-01-02"},
        {"id": "spa-docs/0000012", "text": "Tercero.", "url": "http://c", "date": "2026-01-03"},
    ]
    trans = [
        {"id": "spa-docs/0000010", "text": "Hello world.", "lang": "es"},
        {"id": "spa-docs/0000011", "text": "Second doc.", "lang": "es"},
    ]
    docs_path = tmp_path / "spa-docs.jsonl.gz"
    trans_path = tmp_path / "spa-trans.jsonl.gz"
    with gzip.open(docs_path, "wt", encoding="utf-8") as f:
        for r in native:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(trans_path, "wt", encoding="utf-8") as f:
        for r in trans:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"docs": docs_path, "trans": trans_path}


# --------------------------------------------------------------------------- #
# corpus_tables: the (documents, sentences) spine of the passage model, as real Parquet.
#
# The merge and translate stages read only these two tables, so the fixture builds them the
# way `chunk` does: document text is `" ".join(paragraphs)`, a paragraph is
# `" ".join(sentences)`, and every sentence's (start, end) is a verbatim span of that text.
# The span property is asserted while building, so a fixture that drifts from the chunker's
# contract fails here rather than quietly teaching the stages a shape the corpus never has.
# --------------------------------------------------------------------------- #
def _build_corpus_tables(root: Path, docs: list[dict]) -> dict[str, Path]:
    """Write ``documents.parquet`` + ``sentences.parquet`` for ``docs``.

    Each entry is ``{"document_id", "lang", "paragraphs": [[sentence, ...], ...]}``.
    Token counts are whitespace-word counts (matching :class:`FakeTokenizer`).
    """
    from ragtime.common.io import write_parquet_stream
    from ragtime.common.schemas import document_arrow_schema, sentence_arrow_schema

    doc_rows: list[dict] = []
    sent_rows: list[dict] = []
    for d in docs:
        paragraphs = [" ".join(p) for p in d["paragraphs"]]
        text = " ".join(paragraphs)
        j = 0
        offset = 0
        for pi, para in enumerate(d["paragraphs"]):
            for sent in para:
                start = text.index(sent, offset)
                end = start + len(sent)
                assert text[start:end] == sent, (d["document_id"], sent)
                sent_rows.append(
                    {
                        "sentence_id": f"{d['document_id']}#s{j}",
                        "document_id": d["document_id"],
                        "sentence_index": j,
                        "lang": d["lang"],
                        "start": start,
                        "end": end,
                        "paragraph_index": pi,
                        "token_count": len(sent.split()),
                    }
                )
                offset = end
                j += 1
        doc_rows.append({"document_id": d["document_id"], "lang": d["lang"], "text": text})

    documents = root / "documents.parquet"
    sentences = root / "sentences.parquet"
    write_parquet_stream(documents, doc_rows, schema=document_arrow_schema())
    write_parquet_stream(sentences, sent_rows, schema=sentence_arrow_schema())
    return {"documents": documents, "sentences": sentences}


@pytest.fixture
def build_corpus_tables(tmp_path: Path):
    """Factory: ``build_corpus_tables(docs) -> {"documents": Path, "sentences": Path}``."""

    def _make(docs: list[dict], *, name: str = "corpus") -> dict[str, Path]:
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        return _build_corpus_tables(root, docs)

    return _make


# --------------------------------------------------------------------------- #
# download mock: patches the `snapshot_download` imported by download.py so it writes a
# small fixture set into local_dir, and records the call arguments and count, which is how
# a no-op re-run is shown. `lfs_blob_shas` matches the bytes written here.
# --------------------------------------------------------------------------- #
_FIXTURE_FILES = {
    "eng-docs.jsonl.gz": b"eng-native-bytes\n",
    "spa-trans.jsonl.gz": b"spa-trans-bytes\n",
    "README.md": b"readme\n",
}


@pytest.fixture
def lfs_blob_shas() -> dict[str, str]:
    return {name: hashlib.sha256(data).hexdigest() for name, data in _FIXTURE_FILES.items()}


@pytest.fixture
def mocked_snapshot_download(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    import importlib

    dl = importlib.import_module("ragtime.preprocess.download")

    state = types.SimpleNamespace(calls=[])

    def _fake_snapshot(repo_id, repo_type, allow_patterns, local_dir):
        state.calls.append(
            {"repo_id": repo_id, "repo_type": repo_type, "local_dir": str(local_dir)}
        )
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        for name, data in _FIXTURE_FILES.items():
            (root / name).write_bytes(data)

    # Patch the two seams inside download.py rather than huggingface_hub globally.
    monkeypatch.setattr(dl, "snapshot_download", _fake_snapshot, raising=False)
    monkeypatch.setattr(
        dl,
        "_fetch_expected_shas",
        lambda repo_id: {
            name: hashlib.sha256(data).hexdigest() for name, data in _FIXTURE_FILES.items()
        },
    )
    return state
