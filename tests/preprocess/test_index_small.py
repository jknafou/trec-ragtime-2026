"""The three index legs over the three renderings, with no real checkpoints.

There are no qrels, no dev set and no training data, so every test here checks that the
build produces something structurally sound, never that it retrieves well. The
self-retrieval tests run through a deterministic hash-of-text stand-in encoder: they show
the idmap addresses the right vectors, and say nothing about retrieval quality.

The fixture follows the ``_Doc``/``_tables``/``_write`` pattern of
``tests/common/test_passage_store_final_small.py``, widened to all four language shards with
a multi-sentence Chinese passage for the no-separator case and an oversized sentence, then
padded with dull filler so every shard is big enough for a real PLAID codec to train (see
``_PASSAGES_PER_LANG``). The passage-store tests are not repeated here; this suite builds on
them, and SM09 shows the index composes text with the same ``compose_passage_text`` rather
than a second copy of the join-versus-slice rule.

Legs are exercised two ways: a deterministic in-test trio for every structural property, and
the real ``FaissDenseLeg``, ``SeismicSparseLeg`` and ``PlaidLateInteractionLeg`` against fake
clients wherever the engine libraries are installed. Those libraries live in the ``index``
extra, so the tests that need them skip rather than pretend when it is absent.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import shutil
import sys
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout
from ragtime.common import io as common_io
from ragtime.common.ids import passage_id, sentence_id
from ragtime.common.passage_store import (
    RENDERINGS,
    PassageStore,
    compose_passage_text,
    iter_final_passages,
)
from ragtime.common.schemas import (
    document_arrow_schema,
    final_passage_arrow_schema,
    sentence_arrow_schema,
    translation_final_arrow_schema,
)
from ragtime.config import ConfigError, all_hashes
from ragtime.config.schema import INDEX_BUILD_CONFIG
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family
from ragtime.preprocess import index as index_mod
from ragtime.preprocess.index import (
    DENSE_LEG,
    LATE_INTERACTION_LEG,
    LEGS,
    SHARED_LANG,
    SPARSE_LEG,
    IndexAdapter,
    IndexBuildOptions,
    IndexCtx,
    IndexIntegrityError,
    IndexShardSpec,
    default_legs,
    idmap_arrow_schema,
    index_build_options,
    index_hash,
    kendall_tau_b,
    leg_config_hash,
    leg_encode_hash,
    leg_recipe_keys,
    open_shard,
    query_leg,
    rbo,
)
from ragtime.serving.batching import Tier

pytestmark = pytest.mark.small

_RECON = "f8f20fe2cf17" + "0" * 52
_DOC_EN = "eng-docs/0007421"
_DOC_ES = "spa-docs/0451820"
_DOC_RU = "rus-docs/0330915"
_DOC_ZH = "zho-docs/2288104"

#: How big a language shard has to be for a real PLAID build, and how it has to be batched.
#:
#: fast-plaid trains its residual codec on a held-out slice of the embeddings passed to the
#: first ``add_documents`` call, and that slice is empty below ten vectors: 9 vectors fail
#: with "Cannot train codec: no heldout samples were generated" and 10 pass, at every
#: vectors-per-document ratio tried (1x10, 2x5, 3x4, 4x3, 8x2). The floor is therefore a
#: vector count, not a document count, and shard size alone cannot fix it: with 2-item
#: buckets the build still failed at 256 passages, because ``create`` only ever sees the
#: first bucket. The fixture needs both a bucket that holds enough documents and a shard
#: large enough to fill it. ``_FakeMtd`` emits 2 vectors per document, putting the floor at
#: 5 documents in the first bucket; ``_BATCH_ITEMS`` sits at 12, which is 24 vectors or
#: about 2.4 times the floor, and each shard carries 16 passages, so a second partial bucket
#: exists and the batch-composition assertions see more than one bucket. Smaller buckets
#: afterwards are fine, since they take fast-plaid's ``update`` path, probed at
#: [12, 2, 2, 2] and [12, 1, 1].
_PASSAGES_PER_LANG = 16
_BATCH_ITEMS = 12
_BATCH_TOKENS = 4096


# --------------------------------------------------------------------------- #
# Fixture: the four co-ordered final tables over all four language shards.
# --------------------------------------------------------------------------- #
@dataclass
class _Doc:
    """One fixture document: the passage-store fixture's shape plus ``oversized``.

    ``sep`` is what separates sentences inside ``documents.text``: a space for Latin and
    Cyrillic prose and nothing for Chinese, as in the real corpus. That difference is what
    makes the slice-versus-join distinction observable.
    """

    document_id: str
    lang: str
    sentences: list[str]
    sep: str
    passages: list[tuple[int, ...]]
    omt: list[str] = field(default_factory=list)
    omt_opus: list[str] = field(default_factory=list)
    oversized: tuple[int, ...] = ()

    @property
    def text(self) -> str:
        return self.sep.join(self.sentences)

    def spans(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        cursor = 0
        for i, sentence in enumerate(self.sentences):
            if i:
                cursor += len(self.sep)
            out.append((cursor, cursor + len(sentence)))
            cursor += len(sentence)
        return out


def _base_docs() -> list[_Doc]:
    """One document per language: en, es, ru with an oversized sentence, and zh with no
    sentence separator.

    These four carry every shape the suite reasons about. ``_filler_docs`` then pads each
    language shard out to :data:`_PASSAGES_PER_LANG` so a real PLAID codec can train.
    """
    en = ["Nordic walking uses poles.", "It started in Finland.", "Poles engage the body."]
    es = ["El café contiene 2 g de cafeína.", "Es muy popular."]
    ru = ["Стоимость жилья выросла.", "Это " + "очень " * 40 + "длинное предложение."]
    zh = ["改編自日本歌曲的粵語流行曲陸續面世。", "到90年代初，社會愈趨繁榮。", "躺平運動很流行。"]
    return [
        # English has no translation rows. Nothing is stored for it, so all three renderings
        # resolve to the same native slice, which is what makes one shared `_shared/en`
        # index correct rather than an approximation.
        _Doc(_DOC_EN, "en", en, " ", [(0, 1), (2,)], omt=[], omt_opus=[]),
        _Doc(
            _DOC_ES,
            "es",
            es,
            " ",
            [(0, 1)],
            omt=["Coffee contains 2 g of caffeine.", "It is very popular."],
            omt_opus=["The coffee has 2 g of caffeine.", "It's very popular."],
        ),
        _Doc(
            _DOC_RU,
            "ru",
            ru,
            " ",
            [(0,), (1,)],
            omt=["Housing costs rose.", "This is a " + "very " * 40 + "long sentence."],
            omt_opus=["Housing prices went up.", "That is a " + "very " * 40 + "long one."],
            oversized=(1,),  # the second passage carries the over-budget sentence
        ),
        _Doc(
            _DOC_ZH,
            "zh",
            zh,
            "",  # nothing between CJK sentences, which is why `original` is a slice
            [(0, 1), (2,)],
            omt=[
                "Cantonese pop songs adapted from Japanese songs kept appearing.",
                "By the early 1990s, society had grown more prosperous.",
                "Lying flat is popular.",
            ],
            omt_opus=[
                "Cantonese songs adapted from Japanese ones appeared.",
                "By the start of the 1990s, society was more prosperous.",
                "The lying-flat movement is popular.",
            ],
        ),
    ]


#: Per-language filler material: ``(document-id template, separator, native, omt, omt_opus)``.
#: English carries empty translation tuples because it is not translated, so all three
#: renderings resolve to the native span, as the base English document does. Every string is
#: distinct per language and per index ``{n}``, so a passage's own text retrieves only itself.
_FILLER_TEXT: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "en": (
        "eng-docs/91{:05d}",
        " ",
        ("Harbour report {n} lists the new berths.", "Freight traffic grew by {n} percent."),
        (),
        (),
    ),
    "es": (
        "spa-docs/91{:05d}",
        " ",
        (
            "El informe portuario {n} enumera los nuevos atraques.",
            "El tráfico de mercancías creció un {n} por ciento.",
        ),
        ("Port report {n} lists the new berths.", "Goods traffic grew by {n} percent."),
        ("The harbour report {n} lists new moorings.", "Freight rose {n} percent."),
    ),
    "ru": (
        "rus-docs/91{:05d}",
        " ",
        (
            "Портовый отчёт {n} перечисляет новые причалы.",
            "Грузопоток вырос на {n} процентов.",
        ),
        ("Harbour bulletin {n} enumerates the new quays.", "Cargo flow rose by {n} percent."),
        ("Port bulletin {n} lists new quays.", "The cargo turnover grew {n} percent."),
    ),
    "zh": (
        "zho-docs/91{:05d}",
        "",  # nothing between CJK sentences, as in the base Chinese document
        ("第{n}號港口報告列出新泊位。", "貨運量增長了{n}個百分點。"),
        ("Harbour notice {n} sets out the new moorings.", "Shipping volume grew {n} percent."),
        ("Port notice {n} gives the new moorings.", "The shipping volume rose {n} percent."),
    ),
}


def _filler_docs() -> list[_Doc]:
    """One-passage documents padding every language shard to :data:`_PASSAGES_PER_LANG`.

    They exist so a real PLAID build has a trainable codec, and nothing else, so they are
    deliberately dull: two sentences, one passage, never oversized. Every shape the suite
    asserts on lives in ``_base_docs``.
    """
    base_counts: dict[str, int] = {}
    for doc in _base_docs():
        base_counts[doc.lang] = base_counts.get(doc.lang, 0) + len(doc.passages)
    out: list[_Doc] = []
    for lang, (id_template, sep, native, omt, omt_opus) in _FILLER_TEXT.items():
        for n in range(_PASSAGES_PER_LANG - base_counts[lang]):
            out.append(
                _Doc(
                    id_template.format(n),
                    lang,
                    [s.format(n=n) for s in native],
                    sep,
                    [(0, 1)],
                    omt=[s.format(n=n) for s in omt],
                    omt_opus=[s.format(n=n) for s in omt_opus],
                )
            )
    return out


def _fixture_docs() -> list[_Doc]:
    """The four shape-carrying documents plus enough filler for a real PLAID build."""
    return [*_base_docs(), *_filler_docs()]


def _tables(docs: list[_Doc]) -> dict[str, list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    translations: dict[str, list[dict[str, Any]]] = {"omt": [], "omt_opus": []}
    for doc in docs:
        documents.append({"document_id": doc.document_id, "lang": doc.lang, "text": doc.text})
        for j, (start, end) in enumerate(doc.spans()):
            sentences.append(
                {
                    "sentence_id": sentence_id(doc.document_id, j),
                    "document_id": doc.document_id,
                    "sentence_index": j,
                    "lang": doc.lang,
                    "start": start,
                    "end": end,
                    "paragraph_index": 0,
                    "token_count": len(doc.sentences[j].split()) or 1,
                }
            )
        for k, members in enumerate(doc.passages):
            passages.append(
                {
                    "passage_id": passage_id(doc.document_id, k),
                    "document_id": doc.document_id,
                    "lang": doc.lang,
                    "sentence_ids": [sentence_id(doc.document_id, j) for j in members],
                    "token_count": sum(len(doc.sentences[j].split()) or 1 for j in members),
                    "is_oversized": k in doc.oversized,
                    "paragraph_index": [0],
                }
            )
        # English contributes no translation rows, because the shipped config sets
        # `reconcile.store_identity_translations: false`. Its `omt` and `omt_opus` text is
        # resolved from the native span by `compose_passage_text`, so the fixture leaves
        # `_Doc.omt` and `_Doc.omt_opus` empty for English documents.
        for variant, texts in (("omt", doc.omt), ("omt_opus", doc.omt_opus)):
            for j, text in enumerate(texts):
                translations[variant].append(
                    {
                        "sentence_id": sentence_id(doc.document_id, j),
                        "document_id": doc.document_id,
                        "variant": variant,
                        "text": text,
                        "source_lang": doc.lang,
                    }
                )
    return {
        "documents": documents,
        "sentences": sentences,
        "passages": passages,
        "omt": translations["omt"],
        "omt_opus": translations["omt_opus"],
    }


def _cfg(tmp_path: Path) -> types.SimpleNamespace:
    """A minimal cfg carrying what the adapter reads: run id, languages, three blocks.

    The ``index_build`` block is a fixture rather than a copy of a shipped config, but the
    adapter still reads it through the normal path, ``cfg.blocks["index_build"]["config"]``.
    """
    return types.SimpleNamespace(
        run_id="e2e-original",
        languages=("zh", "en", "ru", "es"),
        blocks={
            "chunker": {"config": {"tokenizer_id": "BAAI/bge-m3@abc", "token_budget": 512}},
            "index_build": {
                "config": {
                    "dense_model": "BAAI/bge-m3",
                    "dense_revision": "5617a9f",
                    "ann_factory": "Flat",
                    "sparse_model": "omai-research/milco-650m",
                    "sparse_model_revision": "567aeb0",
                    "spine_model": "hltcoe/plaidx-large-neuclir-mtd",
                    "spine_model_revision": "1fb92d4",
                    "spine_engine": "pylate+fast_plaid",
                    "plaid_nbits": 2,
                    "dense_tokenizer_id": "BAAI/bge-m3@abc",
                    "sparse_pivot_tokenizer_id": "bert-base-uncased",
                    "late_interaction_tokenizer_id": "FacebookAI/xlm-roberta-large",
                    # Batch composition lives in the hashed block, as the shipped configs
                    # declare it, not in the unshared ``execution`` block, which has no such
                    # keys and would reject them.
                    "encode_batch_token_budget": _BATCH_TOKENS,
                    "encode_max_items": _BATCH_ITEMS,
                },
                "hash": "<logged at runtime>",
            },
            "execution": {},
        },
    )


def _write(tmp_path: Path, tables: dict[str, list[dict[str, Any]]], cfg: Any) -> Layout:
    """Write the four final tables at the same Layout paths the adapter will resolve."""
    layout = Layout(
        run_dir=tmp_path,
        base=tmp_path,
        family=run_family(cfg),
        chunker_hash=all_hashes(cfg)["chunker"],
    )
    common_io.write_parquet_stream(
        layout.documents_path(), tables["documents"], schema=document_arrow_schema()
    )
    common_io.write_parquet_stream(
        layout.final_sentences_path(_RECON), tables["sentences"], schema=sentence_arrow_schema()
    )
    common_io.write_parquet_stream(
        layout.final_passages_path(_RECON, None),
        tables["passages"],
        schema=final_passage_arrow_schema(),
    )
    for variant in ("omt", "omt_opus"):
        common_io.write_parquet_stream(
            layout.final_translations_path(_RECON, variant),
            tables[variant],
            schema=translation_final_arrow_schema(),
        )
    return layout


# --------------------------------------------------------------------------- #
# The deterministic in-test leg trio: no engine library and no checkpoint.
# --------------------------------------------------------------------------- #
_DIM = 8


def _vec(text: str) -> list[float]:
    """A deterministic near-unit vector from the text's sha256. A stand-in, not a model."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i] / 255.0 for i in range(_DIM)]
    norm = sum(v * v for v in raw) ** 0.5 or 1.0
    return [v / norm for v in raw]


class _FakeWriter:
    """Writes the leg's 'engine' file: ``{ordinal: vector}`` as JSON."""

    def __init__(self, out_dir: Path, calls: list[tuple[str, ...]], name: str) -> None:
        self.out_dir = out_dir
        self.calls = calls
        self.name = name
        self.vectors: dict[str, list[float]] = {}

    def add(self, ordinals, reps) -> None:
        for ordinal, rep in zip(ordinals, reps, strict=True):
            self.vectors[str(ordinal)] = list(rep)

    def finish(self) -> None:
        self.calls.append(("finish", self.name))
        (self.out_dir / "vectors.json").write_text(json.dumps(self.vectors), encoding="utf-8")


@dataclass
class _FakeLeg:
    """One leg with the real contract and a toy engine, instrumented for call counts."""

    name: str
    calls: list[tuple[str, ...]] = field(default_factory=list)
    #: When set, the encoder returns fewer reps than passages: the dropped-passage case.
    drop_first: bool = False

    def encode_docs(self, ctx, texts):
        self.calls.append(("encode", self.name))
        reps = [_vec(t) for t in texts]
        return reps[1:] if self.drop_first else reps

    def encode_query(self, ctx, query):
        return _vec(query)

    def writer(self, out_dir: Path, ctx):
        self.calls.append(("build", self.name))
        return _FakeWriter(out_dir, self.calls, self.name)

    def open(self, leg_dir: Path, ctx):
        return json.loads((leg_dir / "vectors.json").read_text(encoding="utf-8"))

    def search(self, reader, ctx, query_rep, top_k):
        scored = [
            (int(ordinal), sum(a * b for a, b in zip(query_rep, vec, strict=True)))
            for ordinal, vec in reader.items()
        ]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[: int(top_k)]


def _fake_legs() -> tuple[_FakeLeg, ...]:
    return tuple(_FakeLeg(name=name) for name in LEGS)


class _FakeDense:
    """A dense client stand-in with the real ``Encoder.embed`` call shape."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def embed(self, texts, mode="dense"):
        self.calls.append((mode, len(texts)))
        return [_vec(t) for t in texts]


class _FakeMilco:
    """A sparse client stand-in returning ``{dim_id: weight}`` per text, as the real one does."""

    @staticmethod
    def _weights(text: str) -> dict[int, float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return {int(digest[i]) + 1: 1.0 + digest[i] / 255.0 for i in range(4)}

    def encode_text(self, texts):
        return [self._weights(t) for t in texts]

    def encode_query(self, query):
        return self._weights(query)


class _FakeMtd:
    """A late-interaction client stand-in: one small multi-vector matrix per text."""

    def encode_docs(self, texts):
        import numpy as np

        return [np.asarray([_vec(t), _vec(t + "#2")], dtype="float32") for t in texts]

    def encode_query(self, query):
        import numpy as np

        return np.asarray([_vec(query)], dtype="float32")

    def truncated_tokens(self, texts):
        # A crude stand-in for the real tokenizer measurement. It only has to be non-zero
        # for a long text, so that the counter's wiring shows up.
        return [max(0, len(t.split()) - 20) for t in texts]


def _ctx(layout: Layout, cfg: Any, legs, *, opts: IndexBuildOptions | None = None) -> IndexCtx:
    resolved = opts if opts is not None else index_build_options(cfg)
    return IndexCtx(
        dense=_FakeDense(),
        milco=_FakeMilco(),
        mtd=_FakeMtd(),
        opts=resolved,
        legs=tuple(legs),
        layout=layout,
        recon_hash=_RECON,
        pack_hash=None,
        idx_hash=index_hash(cfg),
        # read from the same hashed keys the adapter reads, so the two cannot drift apart
        tier=Tier(
            token_budget=resolved.encode_batch_token_budget,
            max_items=resolved.encode_max_items,
        ),
    )


@dataclass
class _Build:
    """One fully built fixture index + everything a test needs to inspect it."""

    cfg: Any
    layout: Layout
    adapter: IndexAdapter
    ctx: IndexCtx
    legs: tuple[Any, ...]
    wq: Any
    receipts: list[Path]
    root: Path


def _build(
    tmp_path: Path,
    *,
    legs=None,
    docs=None,
    part_passages: int | None = None,
) -> _Build:
    """Seed the real work queue, run every shard through ``work``, and ``validate`` each.

    ``part_passages`` sets ``index_build.config.plaid_part_passages``, which is the only
    part size left. It goes through the config rather than through the resolved options, so
    a multi-part build here is the real path with a different declared recipe. Forcing the
    writer directly would say nothing about the config wiring.
    """
    cfg = _cfg(tmp_path)
    if part_passages is not None:
        cfg.blocks["index_build"]["config"]["plaid_part_passages"] = int(part_passages)
    layout = _write(tmp_path, _tables(docs or _fixture_docs()), cfg)
    legs = tuple(_fake_legs() if legs is None else legs)
    adapter = IndexAdapter(pack_hash=None, 
        recon_hash=_RECON, idx_hash=index_hash(cfg), base=str(tmp_path), legs=legs
    )
    wq = saturate.queue_for(cfg, adapter, base=tmp_path)
    assert saturate.seed(cfg, adapter, wq) == len(adapter.shard_specs(cfg))
    ctx = _ctx(layout, cfg, legs)
    receipts: list[Path] = []
    while True:
        shard = saturate.workqueue.claim(wq.pending, wq.running)
        if shard is None:
            break
        out = adapter.work(ctx, shard)
        assert adapter.validate(out), f"validate failed for {shard.name}"
        saturate.workqueue.mark_done(shard, wq.done, "corpus")
        receipts.append(out)
    return _Build(
        cfg=cfg,
        layout=layout,
        adapter=adapter,
        ctx=ctx,
        legs=legs,
        wq=wq,
        receipts=receipts,
        root=tmp_path,
    )


def _manifest(build: _Build) -> dict[str, Any]:
    return common_io.read_jsonl(build.adapter.manifest_path(build.cfg))[0]


def _all_passage_ids(docs=None) -> set[str]:
    return {
        passage_id(doc.document_id, k)
        for doc in (docs or _fixture_docs())
        for k in range(len(doc.passages))
    }


# --------------------------------------------------------------------------- #
# SM01: the base fixture itself
# --------------------------------------------------------------------------- #
def test_fixture_round_trips_through_layout_and_m05g_schemas(tmp_path: Path) -> None:
    """SM01/SM16: the tables are written with reconciliation's own schemas, and read back."""
    cfg = _cfg(tmp_path)
    layout = _write(tmp_path, _tables(_fixture_docs()), cfg)
    records = {r["passage_id"]: r for r in iter_final_passages(layout, _RECON, pack_hash=None)}
    assert set(records) == _all_passage_ids()
    langs = {r["lang"] for r in records.values()}
    assert langs == {"en", "es", "ru", "zh"}  # all four shards present
    for lang in langs:  # every shard is big enough for PLAID, not just the ones a test opens
        assert sum(r["lang"] == lang for r in records.values()) == _PASSAGES_PER_LANG, lang
    assert any(r["is_oversized"] for r in records.values())  # the oversized case exists
    zh = records[passage_id(_DOC_ZH, 0)]
    assert len(zh["sentence_ids"]) == 2 and " " not in zh["original"]
    # Every table sits at its Layout path with a _SUCCESS marker; no path is hand-built.
    for path in (
        layout.documents_path(),
        layout.final_sentences_path(_RECON),
        layout.final_passages_path(_RECON, None),
        layout.final_translations_path(_RECON, "omt"),
        layout.final_translations_path(_RECON, "omt_opus"),
    ):
        assert path.exists() and common_io.is_done(path)


def test_adapter_accepts_m05g_schema_tables_without_a_shim(tmp_path: Path) -> None:
    """SM16: reconciliation's tables are the index build's input, with no shim between."""
    build = _build(tmp_path)
    assert len(build.receipts) == len(build.adapter.shard_specs(build.cfg))
    build.adapter.merge(build.cfg, build.receipts)
    assert common_io.is_done(build.adapter.manifest_path(build.cfg))


# --------------------------------------------------------------------------- #
# SM02: the shard plan
# --------------------------------------------------------------------------- #
def test_shards_are_three_variants_times_three_langs_plus_one_shared_english(
    tmp_path: Path,
) -> None:
    """SM02: exactly 10 specs (one part per language at the shipped size), no duplicates."""
    cfg = _cfg(tmp_path)
    _write(tmp_path, _tables(_fixture_docs()), cfg)  # the plan reads the corpus
    adapter = IndexAdapter(recon_hash=_RECON, pack_hash=None, idx_hash="x" * 64, base=str(tmp_path))
    specs = adapter.shard_specs(cfg)
    assert len(specs) == 10 == len(RENDERINGS) * 3 + 1
    assert len({s.name for s in specs}) == 10
    assert {s.parts for s in specs} == {1}  # 16 passages/lang, one part each
    assert {s.part for s in specs} == {0}
    non_english = {(s.variant, s.source_lang) for s in specs if s.source_lang != SHARED_LANG}
    assert non_english == {(v, lang) for v in RENDERINGS for lang in ("es", "ru", "zh")}
    shared = [s for s in specs if s.source_lang == SHARED_LANG]
    assert len(shared) == 1 and shared[0].variant is None
    # the languages come from cfg, not from a hardcoded set
    trimmed = types.SimpleNamespace(**{**vars(cfg), "languages": ("en", "es")})
    assert len(adapter.shard_specs(trimmed)) == len(RENDERINGS) + 1


def test_the_part_axis_multiplies_the_plan_and_partitions_the_pinned_order(
    tmp_path: Path,
) -> None:
    """The shard grain is ``(variant, source_lang, part)``, cut on the pinned ordinal.

    Two properties matter. The plan is ``ceil(passages / size)`` parts per language, which
    is what shortens the longest shard. And the parts of one language are contiguous ordinal
    slices of the text-free table's ``(token_count, passage_id)`` order, which is what makes
    part membership identical across the three renderings by construction rather than by a
    rule someone has to keep applying.
    """
    cfg = _cfg(tmp_path)
    cfg.blocks["index_build"]["config"]["plaid_part_passages"] = 5
    layout = _write(tmp_path, _tables(_fixture_docs()), cfg)
    adapter = IndexAdapter(recon_hash=_RECON, pack_hash=None, idx_hash="x" * 64, base=str(tmp_path))
    specs = adapter.shard_specs(cfg)
    # 16 passages per language -> 4 parts (5, 5, 5, 1)
    assert len(specs) == (len(RENDERINGS) * 3 + 1) * 4
    assert {s.parts for s in specs} == {4}
    assert sorted({s.part for s in specs}) == [0, 1, 2, 3]
    assert len({s.name for s in specs}) == len(specs)

    pinned = sorted(
        (int(r["token_count"] or 0), r["passage_id"])
        for r in iter_final_passages(layout, _RECON, pack_hash=None)
        if r["lang"] == "zh"
    )
    seen: list[str] = []
    for part in range(4):
        spec = IndexShardSpec(variant="omt", source_lang="zh", part=part, parts=4)
        ids = adapter.expected_ids_at(
            layout.final_passages_path(_RECON, None), "zh", part=part, parts=4, part_size=5
        )
        # exactly the ordinal slice, and nothing else
        assert ids == {pid for _, pid in pinned[part * 5 : part * 5 + 5]}
        assert spec.name.endswith(f"_p{part:05d}")
        seen.extend(sorted(ids))
    assert sorted(seen) == sorted(pid for _, pid in pinned)  # total, disjoint


# --------------------------------------------------------------------------- #
# SM03 / SM13: id-set integrity, and the abort-on-drop half
# --------------------------------------------------------------------------- #
def test_every_leg_variant_covers_exactly_the_corpus_id_set(tmp_path: Path) -> None:
    """SM03: per ``(leg, variant)``, the indexed ids equal the fixture's full id set."""
    build = _build(tmp_path)
    build.adapter.merge(build.cfg, build.receipts)
    manifest = _manifest(build)
    expected = _all_passage_ids()
    for variant in RENDERINGS:
        section = manifest["variants"][variant]
        assert section["passages"] == len(expected)
        covered: dict[str, set[str]] = {leg: set() for leg in LEGS}
        for lang, shard in section["shards"].items():
            for part in shard["shard_parts"]:
                for leg, entry in part["legs"].items():
                    ids = [
                        row["passage_id"]
                        for row in common_io.iter_parquet(
                            Path(entry["path"]) / "idmap.parquet"
                        )
                    ]
                    assert ids, (variant, lang, part["part"], leg)
                    covered[leg] |= set(ids)
        for leg, ids in covered.items():
            assert ids == expected, (variant, leg)


def test_a_leg_that_drops_a_passage_fails_the_shard(tmp_path: Path) -> None:
    """A leg silently returning fewer reps than passages is a hard failure in ``work``."""
    legs = tuple(
        _FakeLeg(name=name, drop_first=(name == SPARSE_LEG)) for name in LEGS
    )
    cfg = _cfg(tmp_path)
    layout = _write(tmp_path, _tables(_fixture_docs()), cfg)
    adapter = IndexAdapter(pack_hash=None, 
        recon_hash=_RECON, idx_hash=index_hash(cfg), base=str(tmp_path), legs=legs
    )
    wq = saturate.queue_for(cfg, adapter, base=tmp_path)
    saturate.seed(cfg, adapter, wq)
    shard = saturate.workqueue.claim(wq.pending, wq.running)
    with pytest.raises(IndexIntegrityError, match="dropped passage"):
        adapter.work(_ctx(layout, cfg, legs), shard)


def test_a_tampered_idmap_fails_validate_and_publishes_no_manifest(tmp_path: Path) -> None:
    """SM13: an id set that is not the corpus's aborts publication, nothing partial."""
    build = _build(tmp_path)
    receipt = json.loads(build.receipts[0].read_text(encoding="utf-8").strip())
    idmap = Path(receipt["legs"][DENSE_LEG]["path"]) / "idmap.parquet"
    rows = [dict(r) for r in common_io.iter_parquet(idmap)][:-1]  # silently lose one passage
    common_io.success_marker(idmap).unlink()
    idmap.unlink()
    common_io.write_parquet_stream(idmap, rows, schema=idmap_arrow_schema())

    # Force a fresh read of the authority table; the cache is a same-process detail.
    build.adapter._expected.clear()
    assert build.adapter.validate(build.receipts[0]) is False
    with pytest.raises(IndexIntegrityError, match="nothing partial is published"):
        build.adapter.merge(build.cfg, build.receipts)
    assert not build.adapter.manifest_path(build.cfg).exists()


# --------------------------------------------------------------------------- #
# SM04 / SM05: no leg may be missing, and there is no fourth leg
# --------------------------------------------------------------------------- #
def test_a_missing_leg_variant_is_a_hard_failure_not_an_exception(tmp_path: Path) -> None:
    """SM04: no leg is allowed to be absent, so a missing one raises."""
    build = _build(tmp_path)
    receipt_path = build.receipts[0]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8").strip())
    receipt["legs"].pop(SPARSE_LEG)
    common_io.success_marker(receipt_path).unlink()
    receipt_path.unlink()
    common_io.write_jsonl(receipt_path, [receipt])
    with pytest.raises(IndexIntegrityError, match="missing leg"):
        build.adapter.merge(build.cfg, build.receipts)
    assert not build.adapter.manifest_path(build.cfg).exists()


def test_a_whole_cell_missing_from_the_plan_is_a_hard_failure(tmp_path: Path) -> None:
    build = _build(tmp_path)
    with pytest.raises(IndexIntegrityError, match="no shard output"):
        build.adapter.merge(build.cfg, build.receipts[1:])
    assert not build.adapter.manifest_path(build.cfg).exists()


def test_bm25_is_structurally_gone_three_legs_per_variant_never_four(tmp_path: Path) -> None:
    """SM05: the leg set is checked rather than merely omitting BM25."""
    assert LEGS == (DENSE_LEG, SPARSE_LEG, LATE_INTERACTION_LEG)
    assert not any("bm25" in leg for leg in LEGS)
    cfg = _cfg(tmp_path)
    assert not any("bm25" in key for key in cfg.blocks["index_build"]["config"])
    for field_name in IndexBuildOptions.__dataclass_fields__:
        assert "bm25" not in field_name
    build = _build(tmp_path)
    build.adapter.merge(build.cfg, build.receipts)
    manifest = _manifest(build)
    assert manifest["legs"] == list(LEGS)
    for variant in RENDERINGS:
        for shard in manifest["variants"][variant]["shards"].values():
            for part in shard["shard_parts"]:
                assert sorted(part["legs"]) == sorted(LEGS)
    # a fourth (or a missing) leg cannot even be constructed
    with pytest.raises(ValueError, match="exactly these legs"):
        IndexAdapter(
            recon_hash=_RECON,
            pack_hash=None,
            idx_hash="x" * 64,
            legs=(*_fake_legs(), _FakeLeg(name="bm25")),
        )
    with pytest.raises(ValueError, match="exactly these legs"):
        IndexAdapter(recon_hash=_RECON, pack_hash=None, idx_hash="x" * 64, legs=_fake_legs()[:2])


# --------------------------------------------------------------------------- #
# SM06 / SM07: the no-text contract and self-retrieval
# --------------------------------------------------------------------------- #
def _handles(build: _Build) -> dict[str, Any]:
    """One opened handle per variant, using that variant's own non-English shard."""
    return {
        variant: open_shard(
            build.layout.index_shard_dir(
                _RECON, build.ctx.idx_hash, variant, "zh", part=0
            ),
            build.ctx,
        )
        for variant in RENDERINGS
    }


def test_search_with_rep_is_query_leg_without_the_encode(tmp_path: Path) -> None:
    """SM06b: ``query_leg`` is ``encode_query`` followed by ``search_with_rep``.

    The corpus-acceptance pass in :mod:`ragtime.preprocess.acceptance` queries with a
    passage's own stored vector, so the primitive it calls has to be the same search and
    ordinal resolution ``query_leg`` runs, rather than a second query path that could drift.
    """
    from ragtime.preprocess.index import search_with_rep

    build = _build(tmp_path)
    handle = next(iter(_handles(build).values()))
    for leg in LEGS:
        impl = handle.leg_impl(leg)
        rep = impl.encode_query(handle.ctx, "housing prices")
        assert search_with_rep(handle, leg, rep, top_k=3) == query_leg(
            handle, leg, "housing prices", top_k=3
        )


def test_query_leg_returns_passage_id_and_score_only(tmp_path: Path) -> None:
    """SM06: three legs by three renderings, ``(str, float)`` pairs, and no text anywhere."""
    build = _build(tmp_path)
    texts = {
        r["passage_id"]: r["original"] for r in iter_final_passages(build.layout, _RECON, pack_hash=None)
    }
    calls = 0
    for handle in _handles(build).values():
        for leg in LEGS:
            hits = query_leg(handle, leg, "housing prices", top_k=3)
            calls += 1
            assert hits, leg
            for item in hits:
                assert type(item) is tuple and len(item) == 2
                pid, score = item
                assert isinstance(pid, str) and isinstance(score, float)
                assert pid in texts  # an id, never a text
                assert pid not in texts.values()
    assert calls == 9
    # The structural guarantee: the only per-passage table a leg writes has no text column.
    assert [f.name for f in idmap_arrow_schema()] == [
        "ordinal",
        "passage_id",
        "document_id",
        "source_lang",
    ]
    for idmap in build.adapter.index_dir(build.cfg).rglob("idmap.parquet"):
        columns = set()
        for row in common_io.iter_parquet(idmap):
            columns |= set(row)
        assert "text" not in columns and not (columns - set(_IDMAP_COLUMNS))


_IDMAP_COLUMNS = ("ordinal", "passage_id", "document_id", "source_lang")


def test_self_retrieval_returns_the_passage_itself_at_rank_one(tmp_path: Path) -> None:
    """SM07: over all nine combinations, a passage's own composed text retrieves itself first.

    This catches a broken index, a mis-keyed idmap or a leg that built empty. It says
    nothing about retrieval quality, since the encoder is a hash stand-in.
    """
    build = _build(tmp_path)
    records = {r["passage_id"]: r for r in iter_final_passages(build.layout, _RECON, pack_hash=None)}
    checks = 0
    for variant, handle in _handles(build).items():
        for leg in LEGS:
            for pid, record in records.items():
                if record["lang"] != "zh":
                    continue
                hits = query_leg(handle, leg, record[variant], top_k=3)
                assert hits[0][0] == pid, (variant, leg, pid, hits)
                checks += 1
    # 3 legs x 3 renderings x every zh passage in the shard
    assert checks == 9 * _PASSAGES_PER_LANG


# --------------------------------------------------------------------------- #
# SM08: search/display decoupling, all 9 combinations
# --------------------------------------------------------------------------- #
def test_any_id_from_any_index_displays_in_all_three_renderings(tmp_path: Path) -> None:
    """SM08: search one rendering and display another, over all nine combinations."""
    build = _build(tmp_path)
    store = PassageStore.from_records(iter_final_passages(build.layout, _RECON, pack_hash=None))
    combos = 0
    for variant, handle in _handles(build).items():
        hits = query_leg(handle, DENSE_LEG, "society prosperity", top_k=2)
        assert hits
        pid = hits[0][0]
        for rendering in RENDERINGS:
            text = store.render(pid, rendering)
            assert text and isinstance(text, str)
            combos += 1
        assert variant in RENDERINGS
    assert combos == 9


# --------------------------------------------------------------------------- #
# SM09: one join-versus-slice rule
# --------------------------------------------------------------------------- #
def test_index_text_is_byte_identical_to_compose_passage_text(tmp_path: Path) -> None:
    """SM09: the encode step composes with the shared composer, not a second copy of it."""
    build = _build(tmp_path)
    zh = next(d for d in _fixture_docs() if d.document_id == _DOC_ZH)
    spans = zh.spans()[:2]  # the multi-sentence passage
    expected_native, _ = compose_passage_text(zh.text, spans, None, "original")
    expected_omt, _ = compose_passage_text(zh.text, spans, zh.omt[:2], "omt")
    pid = passage_id(_DOC_ZH, 0)

    native_items = {
        it.passage_id: it.text
        for it in build.adapter.load_shard(
            build.ctx, IndexShardSpec(variant="original", source_lang="zh")
        )
    }
    omt_items = {
        it.passage_id: it.text
        for it in build.adapter.load_shard(
            build.ctx, IndexShardSpec(variant="omt", source_lang="zh")
        )
    }
    assert native_items[pid] == expected_native
    assert " " not in native_items[pid], "a join would fabricate a separator the source lacks"
    assert omt_items[pid] == expected_omt
    assert native_items[pid] != omt_items[pid]


# --------------------------------------------------------------------------- #
# SM10: English once, and still fully covered
# --------------------------------------------------------------------------- #
def test_english_is_built_once_and_referenced_by_every_variant(tmp_path: Path) -> None:
    """SM10: one shared path in all three manifests and every English id still covered."""
    build = _build(tmp_path)
    build.adapter.merge(build.cfg, build.receipts)
    manifest = _manifest(build)
    shared_paths = {
        manifest["variants"][variant]["shards"][SHARED_LANG]["path"] for variant in RENDERINGS
    }
    assert len(shared_paths) == 1, shared_paths  # string-equal, not three copies
    assert shared_paths.pop().endswith(f"_shared/{SHARED_LANG}")
    for variant in RENDERINGS:
        assert manifest["variants"][variant]["shards"][SHARED_LANG]["shared"] is True

    english = {
        passage_id(doc.document_id, k)
        for doc in _fixture_docs()
        if doc.lang == "en"
        for k in range(len(doc.passages))
    }
    assert len(english) == _PASSAGES_PER_LANG
    for variant in RENDERINGS:
        covered: set[str] = set()
        for shard in manifest["variants"][variant]["shards"].values():
            for part in shard["shard_parts"]:
                covered |= {
                    row["passage_id"]
                    for row in common_io.iter_parquet(
                        Path(part["legs"][DENSE_LEG]["path"]) / "idmap.parquet"
                    )
                }
        assert english <= covered, variant  # one build, referenced by three manifests
        assert covered == _all_passage_ids()

    # The English text is encoded once, not once per rendering.
    encoded_en = sum(
        build.adapter.stats.value(
            index_mod.STAT_ENCODED, leg=DENSE_LEG, lang="en", variant=variant
        )
        for variant in RENDERINGS
    )
    assert encoded_en == len(english)


def test_english_is_one_text_in_all_three_renderings_and_the_canary_stays_zero(
    tmp_path: Path,
) -> None:
    """SM10b: why building English once is correct rather than an approximation.

    English has no stored translation, so ``compose_passage_text`` resolves all three
    renderings to the same native slice. The shared shard is not close enough, it is the
    same bytes. ``index.shared_en_composition_divergence`` is therefore structurally zero,
    and it is still emitted, because a non-zero value means an identity row has reappeared
    in a translations table, which is the one way this property could break.
    """
    build = _build(tmp_path)
    for record in iter_final_passages(build.layout, _RECON, pack_hash=None):
        if record["lang"] != "en":
            continue
        assert record["omt"] == record["original"], record["passage_id"]
        assert record["omt_opus"] == record["original"], record["passage_id"]
    assert build.adapter.stats.total(index_mod.STAT_EN_DIVERGENCE) == 0


# --------------------------------------------------------------------------- #
# SM11 / SM25: identical method across renderings; per-leg tokenizer provenance
# --------------------------------------------------------------------------- #
def test_per_leg_config_hash_is_identical_across_the_three_renderings(tmp_path: Path) -> None:
    """SM11: the same embedder, tokenizer and ANN parameters on every leg and rendering."""
    build = _build(tmp_path)
    build.adapter.merge(build.cfg, build.receipts)
    manifest = _manifest(build)
    for leg in LEGS:
        hashes = {
            part["legs"][leg]["config_hash"]
            for variant in RENDERINGS
            for lang in manifest["variants"][variant]["shards"]
            for part in manifest["variants"][variant]["shards"][lang]["shard_parts"]
        }
        assert len(hashes) == 1, (leg, hashes)
        assert hashes.pop() == manifest["leg_config_hash"][leg]
    # The three legs are different recipes; one shared hash would make the check vacuous.
    assert len(set(manifest["leg_config_hash"].values())) == 3
    # The rendering is not an input to a leg hash.
    opts = index_build_options(build.cfg)
    assert leg_config_hash(opts, DENSE_LEG) == leg_config_hash(opts, DENSE_LEG)


def test_three_distinct_tokenizer_identities_are_recorded_per_leg(tmp_path: Path) -> None:
    """SM25: three vocabularies give three recorded values, not one shared leaf."""
    build = _build(tmp_path)
    build.adapter.merge(build.cfg, build.receipts)
    tokenizers = _manifest(build)["provenance"]["tokenizers"]
    assert set(tokenizers) == set(LEGS)
    assert len(set(tokenizers.values())) == 3, tokenizers
    opts = index_build_options(build.cfg)
    assert tokenizers[DENSE_LEG] == opts.dense_tokenizer_id
    assert tokenizers[SPARSE_LEG] == opts.sparse_pivot_tokenizer_id
    assert tokenizers[LATE_INTERACTION_LEG] == opts.late_interaction_tokenizer_id


def test_manifest_provenance_carries_the_pins_that_matter(tmp_path: Path) -> None:
    """The batch-composition pin, and the field recording the GPU architecture.

    The pin reports the hashed record rather than a code default. Batch composition decides
    which passages are encoded together, and that changes the vectors: only 60.97 % stay
    bitwise identical when a shard is batched differently. It is therefore shared across a
    run family and lives in ``index_build.config``.
    """
    build = _build(tmp_path)
    build.adapter.merge(build.cfg, build.receipts)
    provenance = _manifest(build)["provenance"]
    assert "gpu_arch_pin" in provenance
    declared = build.cfg.blocks["index_build"]["config"]
    pin = provenance["batch_composition_pin"]
    assert pin["order"] == "(token_count, passage_id)"
    assert pin["token_budget"] == declared["encode_batch_token_budget"] == _BATCH_TOKENS
    assert pin["max_items"] == declared["encode_max_items"] == _BATCH_ITEMS
    assert provenance["dense_model"] == "BAAI/bge-m3"
    # The late-interaction window is part of the recipe the manifest states and of the
    # per-leg hash, so indexes built at 220 and at 512 cannot share a leg identity.
    assert provenance["document_length"] == index_build_options(build.cfg).document_length
    opts = index_build_options(build.cfg)
    assert leg_config_hash(replace(opts, document_length=512), LATE_INTERACTION_LEG) != (
        leg_config_hash(opts, LATE_INTERACTION_LEG)
    )


#: The four hardware facts ``saturate.worker_provenance`` reads off a live job, in the shape
#: a real GPU worker produces them: the launcher fills ``RAGTIME_GPU_MODEL`` from
#: ``nvidia-smi`` and forwards the three SLURM variables to the worker.
_WORKER_ENV = {
    "RAGTIME_GPU_MODEL": "NVIDIA GeForce RTX 5090",
    "SLURMD_NODENAME": "gpu009",
    "SLURM_JOB_ID": "4185997",
    "SLURM_ARRAY_TASK_ID": "3",
}


def test_the_manifest_records_which_worker_built_each_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-shard hardware mapping belongs in the durable artefact, not only the queue.

    A build can run across several GPU architectures at once. The dense leg is measured
    bitwise invariant across three of them, but the sparse and late-interaction legs are not
    measured on that axis, so any later question about cross-architecture reproducibility
    can only be answered from a per-part record of which card built what. The queue's
    ``out/`` and ``done/`` carry it too, but they are scratch and are abandoned when the
    stage re-keys. The manifest is what survives.
    """
    for name, value in _WORKER_ENV.items():
        monkeypatch.setenv(name, value)
    build = _build(tmp_path)
    build.adapter.merge(build.cfg, build.receipts)
    manifest = _manifest(build)
    expected = saturate.worker_provenance(dict(_WORKER_ENV))
    assert expected  # against an empty provenance the check below would be vacuous
    parts = [
        part
        for variant in RENDERINGS
        for cell in manifest["variants"][variant]["shards"].values()
        for part in cell["shard_parts"]
    ]
    assert parts, "no part entries: the assertion below would be vacuous"
    assert all(part["worker"] == expected for part in parts), parts
    # The de-duplicated set stays as well. It answers "was this build homogeneous?" at a
    # glance, and it does not replace the mapping above, which says which shard went where.
    assert manifest["provenance"]["workers"] == [json.dumps(expected, sort_keys=True)]


def test_a_receipt_without_worker_provenance_yields_an_explicit_empty_mapping() -> None:
    """Absence is recorded as ``{}`` rather than as a missing key.

    Off SLURM ``worker_provenance`` returns nothing. A key present for some parts and absent
    for others is worse than uniform absence, because a reader walking the manifest would
    see a complete-looking record with a silent hole in it.
    """
    entry = index_mod._manifest_shard(
        {"shard_dir": "/x/part-00000", "part": 0, "variant": None, "legs": {}}
    )
    assert entry["worker"] == {}


def test_batch_composition_is_identical_across_renderings(tmp_path: Path) -> None:
    """Bucket membership does not vary by rendering.

    Measured on one shard: batching it differently leaves 60.97 % of vectors bitwise
    identical and 99.95 % of the top ten unchanged. Bucketing on the rendering-invariant
    ``(token_count, passage_id)`` key removes that as a source of cross-rendering
    difference, so a rank difference is attributable to the translation alone.
    """
    build = _build(tmp_path)
    memberships = []
    for variant in RENDERINGS:
        items = build.adapter.load_shard(
            build.ctx, IndexShardSpec(variant=variant, source_lang="zh")
        )
        memberships.append(
            [[it.passage_id for it in bucket] for bucket in build.adapter.buckets(build.ctx, items)]
        )
    assert memberships[0] == memberships[1] == memberships[2]
    assert sum(len(b) for b in memberships[0]) == _PASSAGES_PER_LANG  # every zh passage
    # More than one bucket, or identical membership would be one trivial bucket.
    assert len(memberships[0]) > 1, memberships[0]


# --------------------------------------------------------------------------- #
# SM12: per-leg atomic outputs / resume
# --------------------------------------------------------------------------- #
def test_a_published_leg_is_not_rebuilt_when_another_leg_is_retried(tmp_path: Path) -> None:
    """SM12: after a kill between legs, re-entering rebuilds only the missing leg."""
    build = _build(tmp_path)
    shard_dir = build.layout.index_shard_dir(_RECON, build.ctx.idx_hash, "omt", "zh", part=0)
    before = [tuple(leg.calls) for leg in build.legs]

    # The killed leg: its output is gone and the other two survive.
    late = shard_dir / LATE_INTERACTION_LEG
    common_io.success_marker(late).unlink()
    import shutil

    shutil.rmtree(late)

    receipt = next(
        p
        for p in build.receipts
        if json.loads(p.read_text(encoding="utf-8").strip())["shard_dir"] == str(shard_dir)
    )
    shard = build.wq.done / receipt.name
    assert shard.exists()
    out = build.adapter.work(build.ctx, shard)

    after = {leg.name: tuple(leg.calls) for leg in build.legs}
    for leg, prior in zip(build.legs, before, strict=True):
        if leg.name == LATE_INTERACTION_LEG:
            assert len(after[leg.name]) > len(prior)  # rebuilt
        else:
            assert after[leg.name] == prior, f"{leg.name} was rebuilt on resume"
    assert build.adapter.validate(out) is True
    assert common_io.is_done(late)


def test_re_entering_a_complete_shard_reads_no_passage_text_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully published shard re-streams nothing: the skip happens before the read."""
    build = _build(tmp_path)
    receipt = build.receipts[0]
    shard = build.wq.done / receipt.name
    called: list[int] = []

    def forbidden(self, ctx, spec):
        called.append(1)
        return []

    monkeypatch.setattr(IndexAdapter, "load_shard", forbidden)
    build.adapter.work(build.ctx, shard)
    assert called == []
    resumed = build.adapter.stats.total(index_mod.STAT_LEG_RESUMED)
    assert resumed >= len(LEGS)


# --------------------------------------------------------------------------- #
# SM14 / SM15 / SM17: the frozen cross-stage surfaces
# --------------------------------------------------------------------------- #
def test_index_adapter_matches_the_stage_adapter_protocol_exactly(tmp_path: Path) -> None:
    """SM14: one adapter for the whole stage, with the driver's stable method surface."""
    adapter = IndexAdapter(recon_hash=_RECON, pack_hash=None, idx_hash="x" * 64, base=str(tmp_path))
    assert adapter.stage.startswith("index_")
    assert adapter.template.endswith(".sbatch")
    for name in ("bringup", "shards", "work", "validate", "merge"):
        expected = [
            p for p in inspect.signature(getattr(saturate.StageAdapter, name)).parameters
            if p != "self"
        ]
        got = [
            p for p in inspect.signature(getattr(adapter, name)).parameters if p != "self"
        ]
        assert got == expected, f"{name}{tuple(got)} != {name}{tuple(expected)}"
    assert isinstance(adapter, saturate.StageAdapter)


def test_query_leg_signature_is_the_pinned_m06_contract() -> None:
    """SM15: ``(handle, leg, query, top_k) -> list[tuple[str, float]]``, as retrieval calls it."""
    sig = inspect.signature(query_leg)
    assert list(sig.parameters) == ["handle", "leg", "query", "top_k"]
    assert sig.return_annotation == "list[tuple[str, float]]"


def test_search_with_rep_signature_is_pinned() -> None:
    """SM15b: the second half of the query path, same shape with ``rep`` for ``query``."""
    from ragtime.preprocess.index import search_with_rep

    sig = inspect.signature(search_with_rep)
    assert list(sig.parameters) == ["handle", "leg", "rep", "top_k"]
    assert sig.return_annotation == "list[tuple[str, float]]"


def test_passage_store_call_site_signatures_are_unchanged() -> None:
    """SM17: retrieval's ``display()`` and the pipeline citation path both call these."""
    from ragtime.common.passage_store import LmdbPassageStore

    render = inspect.signature(LmdbPassageStore.render)
    assert list(render.parameters) == ["self", "passage_id", "passage_lang"]
    passage = inspect.signature(LmdbPassageStore.passage)
    assert list(passage.parameters) == ["self", "passage_id"]
    build = inspect.signature(LmdbPassageStore.build_from_final)
    assert list(build.parameters)[:3] == ["path", "layout", "reconcile_hash"]


# --------------------------------------------------------------------------- #
# SM19: the shipped default trains nothing, and the compressed path stays a config value
# --------------------------------------------------------------------------- #
def test_flat_is_the_default_and_the_compressed_string_round_trips_as_non_default(
    tmp_path: Path,
) -> None:
    """SM19: ``OPQ64,IVF16384_HNSW32,PQ64`` is a valid non-default config value."""
    cfg = _cfg(tmp_path)
    assert index_build_options(cfg).exact is True
    compressed = types.SimpleNamespace(**vars(cfg))
    compressed.blocks = {
        **cfg.blocks,
        "index_build": {
            "config": {
                **cfg.blocks["index_build"]["config"],
                "ann_factory": "OPQ64,IVF16384_HNSW32,PQ64",
                "nprobe": 256,
            }
        },
    }
    opts = index_build_options(compressed)
    assert opts.ann_factory == "OPQ64,IVF16384_HNSW32,PQ64"
    assert opts.exact is False and opts.nprobe == 256
    # A recipe edit moves the index hash, so it cannot resolve onto the flat build's tree.
    assert index_hash(compressed) != index_hash(cfg)


def test_the_flat_dense_path_never_calls_a_training_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM19: with ``ann_factory="Flat"`` nothing is trained or clustered.

    ``index_factory`` is the only route to a trainable index in this module. A spy that
    counts zero calls to it, plus a read-back index that is a plain ``IndexFlatIP`` and
    already ``is_trained``, is what "the shipped default has no quantiser step" means.
    """
    faiss = pytest.importorskip("faiss")
    from ragtime.preprocess.index import FaissDenseLeg

    calls: list[str] = []
    real_factory = faiss.index_factory

    def spy_factory(*args, **kwargs):
        calls.append("index_factory")
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(faiss, "index_factory", spy_factory)
    cfg = _cfg(tmp_path)
    leg = FaissDenseLeg()
    ctx = _ctx(
        Layout(run_dir=tmp_path, base=tmp_path, family="e2e", chunker_hash="c" * 64),
        cfg,
        default_legs(),
    )
    out = tmp_path / "denseleg"
    out.mkdir()
    writer = leg.writer(out, ctx)
    texts = ["alpha text", "beta text", "gamma text"]
    writer.add([0, 1, 2], leg.encode_docs(ctx, texts))
    writer.finish()
    assert calls == []  # no factory call, so no trainable index and nothing to train
    reader = leg.open(out, ctx)
    assert isinstance(reader, faiss.IndexFlatIP) and reader.is_trained
    hits = leg.search(reader, ctx, leg.encode_query(ctx, texts[1]), 2)
    assert hits[0][0] == 1  # self-retrieval through the real faiss index


# --------------------------------------------------------------------------- #
# The declared training seed reaches every leg that trains, and the one leg that cannot
# take it says so, with the vendor signature as evidence.
#
# The defect: ``quantizer_train_seed`` was hashed into the dense leg's recipe and then
# applied to nothing, so three renderings could each cluster from different random state
# while the manifest recorded a seed. That is the same kind of hidden, rendering-correlated
# confound the batch-composition pin exists to remove.
# --------------------------------------------------------------------------- #
class _FakeClusteringParameters:
    """Stands in for FAISS's ``ClusteringParameters``; all that matters is the ``seed``."""

    def __init__(self) -> None:
        self.seed = -1


class _FakeTrainableIndex:
    """An index shaped like ``OPQ...,IVF...,PQ...``: a coarse ``cp`` plus the PQ's own one."""

    def __init__(self) -> None:
        self.cp = _FakeClusteringParameters()
        self.pq = types.SimpleNamespace(cp=_FakeClusteringParameters())
        self.is_trained = False


def test_the_declared_seed_reaches_every_faiss_clustering_knob() -> None:
    """Both ``cp.seed`` values come from the config before ``train``, not from chance."""
    opts = IndexBuildOptions(ann_factory="OPQ64,IVF16384_HNSW32,PQ64", quantizer_train_seed=1234)
    assert opts.train_seed == 1234  # one declared seed, named for what it serves
    index = _FakeTrainableIndex()
    applied = index_mod._apply_faiss_train_seed(index, opts)
    assert set(applied) == {"index.cp", "index.pq.cp"}
    assert index.cp.seed == 1234 and index.pq.cp.seed == 1234


def test_a_trainable_index_with_no_seedable_knob_refuses_to_train() -> None:
    """A clustering that cannot be pinned fails loudly rather than starting at random."""
    opts = IndexBuildOptions(ann_factory="SomeUnseedableRecipe", quantizer_train_seed=7)
    untrainable = types.SimpleNamespace(is_trained=False)
    with pytest.raises(IndexIntegrityError, match="unseeded"):
        index_mod._apply_faiss_train_seed(untrainable, opts)
    # An index that needs no training is left alone, so this raises no false alarm.
    assert index_mod._apply_faiss_train_seed(types.SimpleNamespace(is_trained=True), opts) == []


def test_the_seed_is_applied_before_train_on_the_non_exact_dense_path() -> None:
    """A seed set after ``train`` pins nothing, so the order is what is checked."""
    order: list[str] = []
    index = _FakeTrainableIndex()
    opts = IndexBuildOptions(ann_factory="OPQ64,IVF16384_HNSW32,PQ64", quantizer_train_seed=99)

    class _Recording(_FakeTrainableIndex):
        def train(self, arr: Any) -> None:
            order.append(f"train(seed={self.cp.seed})")

        def add(self, arr: Any) -> None:
            order.append("add")

    recording = _Recording()
    source = inspect.getsource(index_mod._FaissWriter.finish)
    assert source.index("_apply_faiss_train_seed") < source.index(".train(")
    index_mod._apply_faiss_train_seed(recording, opts)
    recording.train(None)
    assert order == ["train(seed=99)"]
    assert index.cp.seed == -1  # the untouched control


def test_the_plaid_leg_is_opened_with_the_declared_seed_not_the_vendor_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PyLate defaults to ``seed=42`` and hands it to fast-plaid's k-means, which leaves a
    run choice living inside a dependency. It comes from the hashed block instead."""
    seen: list[dict] = []

    def _plaid(**kwargs):
        seen.append(kwargs)
        return object()

    stub = types.ModuleType("pylate")
    stub.indexes = types.SimpleNamespace(PLAID=_plaid)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pylate", stub)
    opts = IndexBuildOptions(quantizer_train_seed=1234, plaid_nbits=2)
    index_mod._open_plaid(tmp_path / "plaidleg", opts, override=True, part=0)
    # The declared seed is what the engine received, and both sides are real values.
    assert seen[0]["seed"] == opts.train_seed
    # It is read per build rather than being a constant that happens to match: a second
    # declared seed moves the keyword with it, which no vendor default could do.
    other = replace(opts, quantizer_train_seed=4321)
    index_mod._open_plaid(tmp_path / "plaidleg2", other, override=True, part=0)
    assert seen[1]["seed"] == other.train_seed
    assert seen[0]["seed"] != seen[1]["seed"]
    assert seen[0]["nbits"] == 2 and seen[0]["override"] is True
    # The Triton k-means kernel is pinned off, and that is the pin that decided the leg's
    # bit-reproducibility. pylate defaults to ``use_triton=None``, fastkmeans reads that as
    # "use Triton if it imports", and that kernel gave 4 distinct centroid sets in 4 runs
    # under one seed, leaving 11 of the leg's 23 files different between two builds. Passing
    # it explicitly also stops a change of vendor default from changing what this stage
    # builds.
    assert seen[0]["use_triton"] is False
    # The seed is part of the leg's identity: two seeds are two different indexes.
    assert leg_config_hash(opts, LATE_INTERACTION_LEG) != leg_config_hash(
        replace(opts, quantizer_train_seed=7), LATE_INTERACTION_LEG
    )


def test_every_plaid_part_is_added_under_the_rng_and_kernel_pins() -> None:
    """The three pins sit at the call site, in the order that makes them effective.

    They only work if they are in force when ``add_documents`` runs, which is an ordering
    property of one function, and no fixture can observe it without the multi-gigabyte
    checkpoints. So it is read off the source. The measurements behind each pin are with
    :func:`index._pin_torch_rng`, :func:`index._deterministic_torch_ops` and
    :data:`index._PLAID_USE_TRITON_KMEANS`.
    """
    source = inspect.getsource(index_mod._PlaidWriter._flush_part)
    assert source.index("_pin_torch_rng") < source.index("add_documents")
    assert "with _deterministic_torch_ops():" in source
    assert source.index("with _deterministic_torch_ops():") < source.index("add_documents(")
    # The Triton pin travels with the index construction, not with the add.
    assert "use_triton" in inspect.getsource(index_mod._open_plaid)


def test_the_sparse_legs_unseedable_clustering_is_declared_with_its_vendor_signature() -> None:
    """The sparse leg's clustering cannot be seeded, and the artefact says so.

    pyseismic-lsr 0.5.1's build call takes no seed argument at all, because its Rust
    ``RandomKmeans`` draws from the OS. Recording which leg is unpinned, and why, is what
    keeps the determinism claim from being unbacked for a third of the index.
    """
    signature = index_mod.UNSEEDABLE_LEGS[SPARSE_LEG]
    assert "build_from_dataset(" in signature
    assert "seed" not in signature  # the absence is the evidence
    assert DENSE_LEG not in index_mod.UNSEEDABLE_LEGS
    assert LATE_INTERACTION_LEG not in index_mod.UNSEEDABLE_LEGS
    # The sparse leg's recipe hash claims no pin it does not have.
    body_source = inspect.getsource(index_mod.leg_config_hash)
    assert body_source.count('"train_seed"') == 1  # late-interaction only


def test_the_dense_recipe_carries_no_knob_that_reaches_no_call() -> None:
    """The dense recipe names no knob the build cannot honour: no ``dense_mrl_dim``, no ``refine``.

    A key in ``leg_config_hash`` advertises that the build honours it. BGE-M3's dense head
    has no Matryoshka truncation knob and this module builds no refine wrapper, so either key
    would be a false pin: two indexes differing only in ``refine`` come out byte-identical while
    their recipe hashes say otherwise. Neither may appear without the call site that reads it.
    """
    fields = set(IndexBuildOptions.__dataclass_fields__)
    assert "dense_mrl_dim" not in fields and "refine" not in fields
    body_source = inspect.getsource(index_mod.leg_config_hash)
    for dead in ('"mrl_dim"', '"refine"'):
        assert dead not in body_source, dead
    # What remains in the dense recipe is the set that reaches a call.
    tree = ast.parse(body_source)
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.comparators[0], ast.Name)
        and node.test.comparators[0].id == "DENSE_LEG"
    )
    dense_dict = next(n for n in ast.walk(branch) if isinstance(n, ast.Dict))
    dense_keys = {k.value for k in dense_dict.keys if isinstance(k, ast.Constant)}
    assert dense_keys == {
        "model",
        "revision",
        "tokenizer_id",
        "ann_factory",
        "nprobe",
        "quantizer_train_seed",
    }


def _with_index_build(cfg: types.SimpleNamespace, **keys: object) -> types.SimpleNamespace:
    """``cfg`` with ``index_build.config`` overridden by ``keys``, and nothing else moved."""
    edited = types.SimpleNamespace(**vars(cfg))
    edited.blocks = {
        **cfg.blocks,
        "index_build": {"config": {cfg.blocks["index_build"]["config"], keys}},
    }
    return edited


def test_a_structural_key_the_code_cannot_honour_fails_at_load(tmp_path: Path) -> None:
    """``index_build.config`` is a free-form leaf, so a key naming a structural fact has to
    be validated or a config could quietly lie.

    ``english_once: false`` and ``dense_engine: qdrant`` describe builds this code cannot
    produce. Ignoring them silently would break the point of a config file that claims to be
    the complete record of a run.
    """
    cfg = _cfg(tmp_path)
    assert index_build_options(cfg)  # the shipped shape is accepted
    for keys, needle in (
        ({"english_once": False}, "english_once"),
        ({"dense_engine": "qdrant"}, "dense_engine"),
        ({"sparse_engine": "lucene"}, "sparse_engine"),
        ({"spine_engine": "stanford_plaid"}, "spine_engine"),
        ({"shard_by": "document_id"}, "shard_by"),
        ({"variants": ["original", "omt"]}, "variants"),
    ):
        with pytest.raises(ConfigError, match=needle):
            index_build_options(_with_index_build(cfg, **keys))


def test_the_structural_expectations_are_read_off_the_code_not_re_typed(
    tmp_path: Path,
) -> None:
    """The validator is not a second, drifting copy of the structure it guards."""
    cfg = _cfg(tmp_path)
    # The engines come from ENGINE_OF, which is derived from the artefact file names.
    assert set(index_mod.ENGINE_OF) == set(LEGS)
    assert index_mod.ENGINE_OF[DENSE_LEG] == index_mod._DENSE_FILENAME.rsplit(".", 1)[-1]
    assert index_mod.ENGINE_OF[SPARSE_LEG] == index_mod._SPARSE_SAVE_STEM.rsplit(".", 1)[-1]
    # The shard column comes from IndexShardSpec itself.
    assert index_mod._shard_by_column() in IndexShardSpec.__dataclass_fields__
    # A declared engine that agrees is accepted and recorded verbatim.
    ok = _with_index_build(cfg, spine_engine=index_mod.ENGINE_OF[LATE_INTERACTION_LEG])
    assert index_build_options(ok).spine_engine == index_mod.ENGINE_OF[LATE_INTERACTION_LEG]
    # An absent key is not a violation; the structure holds whether the file says so or not.
    bare = types.SimpleNamespace(**vars(cfg))
    bare.blocks = {**cfg.blocks, "index_build": {"config": {}}}
    assert index_build_options(bare).spine_engine == index_mod.ENGINE_OF[LATE_INTERACTION_LEG]


def test_the_manifest_records_the_seed_and_the_one_leg_that_cannot_take_it(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    provenance = index_mod._provenance(cfg, [])["train_seed"]
    assert provenance["seed"] == index_build_options(cfg).train_seed
    assert provenance["seeded_legs"] == sorted({DENSE_LEG, LATE_INTERACTION_LEG})
    assert provenance["unseeded_legs"] == {SPARSE_LEG: index_mod.SEISMIC_BUILD_SIGNATURE}
    # A seed alone did not make the late-interaction leg reproducible: its k-means ran a
    # Triton assignment kernel that moves from run to run whatever the seed, giving 4
    # distinct centroid sets in 4 runs. The manifest records that pin beside the seed, so it
    # cannot claim a seeded build while describing a non-reproducible artefact.
    assert provenance["plaid_use_triton_kmeans"] is index_mod._PLAID_USE_TRITON_KMEANS
    assert index_mod._PLAID_USE_TRITON_KMEANS is False


# --------------------------------------------------------------------------- #
# SM22: no filtered retrieval is built anywhere
# --------------------------------------------------------------------------- #
def test_no_filter_is_built_anywhere(tmp_path: Path) -> None:
    """SM22: no filter config key, no filter build option, no IDSelector wiring."""
    from ragtime.config.schema import _ALLOWED

    assert not any(
        "filter" in key or "predicate" in key for key in _ALLOWED["retrieval"]
    )
    for field_name in IndexBuildOptions.__dataclass_fields__:
        assert "filter" not in field_name and "predicate" not in field_name
    source = Path(index_mod.__file__).read_text(encoding="utf-8")
    assert "IDSelector" not in source


# --------------------------------------------------------------------------- #
# The recipe-to-encoder wiring: what the manifest records is what runs.
#
# These run ``bringup`` against the shipped ``config/*.yml``. Every other test in this file
# constructs ``IndexCtx`` and its clients directly, so none of them exercises the
# config to registry to context path, and that is how the dense leg came to take
# ``clients.embedder``, built from the query-time ``retrieval.dense`` key, while the
# manifest and ``leg_config_hash`` recorded ``BAAI/bge-m3``. Nothing here loads a
# checkpoint, because every client is lazy.
# --------------------------------------------------------------------------- #
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@pytest.mark.parametrize(
    "config_path", sorted(_CONFIG_DIR.glob("*.yml")), ids=lambda p: p.stem
)
def test_bringup_encoders_are_the_ones_the_hashed_recipe_records(
    config_path: Path, tmp_path: Path
) -> None:
    """For every shipped config, the encoders ``bringup`` hands the legs are the recipe."""
    from ragtime.config import load

    cfg = load(config_path)
    adapter = IndexAdapter.for_config(cfg, base=tmp_path)
    ctx = adapter.bringup(cfg)
    opts = index_build_options(cfg)

    assert ctx.dense.model == opts.dense_model
    assert ctx.dense.revision == opts.dense_revision
    assert ctx.milco.model == opts.sparse_model
    assert ctx.mtd.checkpoint == opts.spine_model
    # Constructing the bundle loads no weights, so a build worker pays for a model only
    # once it has claimed a shard.
    assert ctx.dense._backend is None
    assert ctx.milco._backend is None and ctx.mtd._backend is None
    # The query-time key is not what the index encodes with.
    retrieval_dense = str(dict(cfg.blocks.get("retrieval", {})).get("dense", ""))
    assert ctx.dense.model != retrieval_dense or retrieval_dense == opts.dense_model
    assert ctx.dense.model, "an empty model id crashes on the first dense batch of the build"


def test_the_manifests_dense_identity_is_the_client_that_encoded(tmp_path: Path) -> None:
    """Recipe, client and manifest agree, on a shipped config."""
    from ragtime.config import load

    cfg = load(_CONFIG_DIR / "e2e-original.yml")
    adapter = IndexAdapter.for_config(cfg, base=tmp_path)
    ctx = adapter.bringup(cfg)
    provenance = index_mod._provenance(cfg, [])
    assert provenance["dense_model"] == ctx.dense.model
    assert provenance["sparse_model"] == ctx.milco.model
    assert provenance["spine_model"] == ctx.mtd.checkpoint


def test_a_client_that_is_not_the_recipe_aborts_bringup(tmp_path: Path) -> None:
    """A client that is declared but not wired is refused rather than left to review."""
    opts = index_build_options(_cfg(tmp_path))

    class _Clients:
        def __init__(self, dense_model: str, sparse_window: int | None = None) -> None:
            self.index_dense = types.SimpleNamespace(model=dense_model)
            self.milco = types.SimpleNamespace(
                model=opts.sparse_model,
                max_length=(
                    opts.sparse_max_length if sparse_window is None else sparse_window
                ),
            )
            self.mtd_colbert = types.SimpleNamespace(
                checkpoint=opts.spine_model,
                document_length=opts.document_length,
            )

    index_mod._assert_clients_match_recipe(_Clients(opts.dense_model), opts)  # a match passes
    with pytest.raises(ValueError, match="dense: recipe says"):
        index_mod._assert_clients_match_recipe(_Clients("Qwen/Qwen3-Embedding-4B"), opts)
    with pytest.raises(ValueError, match="dense: recipe says"):
        index_mod._assert_clients_match_recipe(_Clients(""), opts)
    # The sparse encode window is guarded the same way the identities are. A manifest saying
    # 8192 over a client capped at 512 would describe an index in which 2.36 % of Chinese
    # passages are silently prefixes of themselves.
    with pytest.raises(ValueError, match="sparse_max_length: recipe says"):
        index_mod._assert_clients_match_recipe(_Clients(opts.dense_model, 512), opts)


# --------------------------------------------------------------------------- #
# SM28, the sparse leg's encode window: config-driven, hashed, and measured.
#
# The window is the model's own ceiling, not a library default of 512: over all 9,405,925
# passages in three renderings, 174,913 passage-renderings exceed 512 tokens - zh 2.36 %,
# ru 0.214 %, en 0.030 %, es 0.024 %, maximum 5,277 tokens. That is a roughly 97-fold
# asymmetry between languages, manufactured by a default, in the one leg whose job is to
# fire evenly across languages. The model's own model_max_length is 8192.
# --------------------------------------------------------------------------- #
def test_the_sparse_encode_window_is_the_models_own_ceiling_not_a_library_default() -> None:
    """SM28: the window is the model's own 8192, not 512 and not the corpus maximum."""
    from ragtime.serving import sparse_milco

    assert sparse_milco.MODEL_MAX_LENGTH == 8192
    # The model's ceiling rather than the observed corpus maximum: sizing to today's longest
    # passage brings truncation back the first time a longer one arrives.
    assert sparse_milco.DEFAULT_MAX_LENGTH == sparse_milco.MODEL_MAX_LENGTH
    assert sparse_milco.DEFAULT_MAX_LENGTH > 5_277
    assert sparse_milco.MilcoEncoder("m").max_length == sparse_milco.MODEL_MAX_LENGTH
    # One literal: the stage's recipe default is the client's, imported rather than re-typed.
    assert index_mod._DEFAULT_SPARSE_MAX_LENGTH == sparse_milco.DEFAULT_MAX_LENGTH


def test_the_sparse_window_is_read_from_the_hashed_block_and_joins_the_leg_hash(
    tmp_path: Path,
) -> None:
    """SM28: ``index_build.config.sparse_max_length`` is the record, not a code constant."""
    cfg = _cfg(tmp_path)
    assert "sparse_max_length" not in cfg.blocks["index_build"]["config"]
    assert index_build_options(cfg).sparse_max_length == index_mod._DEFAULT_SPARSE_MAX_LENGTH

    cfg.blocks["index_build"]["config"]["sparse_max_length"] = 4096
    opts = index_build_options(cfg)
    assert opts.sparse_max_length == 4096

    # It changes the leg's bytes, so it changes that leg's identity and no other's, just as
    # ``document_length`` does for the late-interaction leg.
    base = replace(opts, sparse_max_length=index_mod._DEFAULT_SPARSE_MAX_LENGTH)
    assert leg_config_hash(opts, SPARSE_LEG) != leg_config_hash(base, SPARSE_LEG)
    assert leg_config_hash(opts, DENSE_LEG) == leg_config_hash(base, DENSE_LEG)
    assert leg_config_hash(opts, LATE_INTERACTION_LEG) == (
        leg_config_hash(base, LATE_INTERACTION_LEG)
    )
    assert index_mod._provenance(cfg, [])["sparse_max_length"] == 4096


def test_the_registry_hands_the_sparse_client_the_windows_the_recipe_records(
    tmp_path: Path,
) -> None:
    """SM28: the window travels from config to registry to client, so the recipe and the
    engine cannot state two different ones."""
    from ragtime.serving import registry

    cfg = _cfg(tmp_path)
    cfg.blocks["index_build"]["config"]["sparse_max_length"] = 4096
    assert registry._milco_client(cfg).max_length == 4096
    del cfg.blocks["index_build"]["config"]["sparse_max_length"]
    assert registry._milco_client(cfg).max_length == index_mod._DEFAULT_SPARSE_MAX_LENGTH


def test_sparse_truncation_is_counted_the_way_late_interaction_truncation_is(
    tmp_path: Path,
) -> None:
    """SM28: both legs now have the counter whose absence hid this defect.

    Zero truncation can only be asserted on a leg that is measured. Only the
    late-interaction client exposed ``truncated_tokens``, and ``_emit_truncation`` looked at
    that client alone, so the sparse leg's 512-token cap was unobservable by construction,
    whatever the corpus did.
    """
    cfg = _cfg(tmp_path)
    layout = _write(tmp_path, _tables(_fixture_docs()), cfg)
    adapter = IndexAdapter(pack_hash=None, 
        recon_hash=_RECON, idx_hash=index_hash(cfg), base=str(tmp_path), legs=_fake_legs()
    )
    ctx = _ctx(layout, cfg, adapter.legs)
    ctx.milco = types.SimpleNamespace(truncated_tokens=lambda texts: [7 for _ in texts])
    spec = IndexShardSpec(variant="omt", source_lang="zh")

    adapter._emit_truncation(ctx, spec, SPARSE_LEG, ["a", "b"])
    assert adapter.stats.value(
        index_mod.STAT_SPARSE_TRUNCATED, leg=SPARSE_LEG, lang="zh", variant="omt"
    ) == 14.0
    # The two legs keep separate metric ids; a counter's name is part of its meaning.
    assert index_mod.STAT_SPARSE_TRUNCATED != index_mod.STAT_TRUNCATED
    assert adapter.stats.value(
        index_mod.STAT_TRUNCATED, leg=SPARSE_LEG, lang="zh", variant="omt"
    ) == 0.0

    # The dense leg is absent from the truncation sources on purpose: ``Encoder.embed``
    # passes no max_length at all, so BGE-M3 runs at its own 8192 max_seq_length. Asserted
    # on the source, so a ``max_length=`` added there later fails here rather than shipping.
    assert DENSE_LEG not in index_mod._TRUNCATION_SOURCE
    from ragtime.serving import encoders as enc_mod

    assert "max_length" not in inspect.getsource(enc_mod.Encoder)


# --------------------------------------------------------------------------- #
# SM33: the sparse leg's assemble-time top-k cut, and the four declared vendor parameters.
#
# ``seismic_top_k`` cuts each passage to its 300 heaviest components as the index is built,
# over vectors that stay stored unpruned. Measured at 7.35 GiB down to 2.44 GiB per part,
# and about 505 GiB down to 167 GiB per rendering, against the model's own ablation
# (arXiv:2510.00671 Table 11: 72.7 at k=1000, 72.1 at k=300, 68.4 at k=100).
#
# ``n_postings``, ``summary_energy``, ``max_fraction`` and ``doc_cut`` used to come from
# pyseismic-lsr's own defaults, so those run choices lived in a dependency, in no config and
# in no hash. They are declared at their current effective values, so behaviour is unchanged.
#
# The property tested first is that the cut is an assemble key. If it moved
# ``leg_encode_hash``, 1.6 TB of stored sparse vectors would be re-encoded on GPU for a
# change that touches no forward pass.
# --------------------------------------------------------------------------- #
_SEISMIC_DECLARED = (
    "seismic_top_k",
    "seismic_n_postings",
    "seismic_summary_energy",
    "seismic_max_fraction",
    "seismic_doc_cut",
)


def test_the_sparse_cut_re_assembles_but_never_re_encodes(tmp_path: Path) -> None:
    """SM33: every one of these keys is assemble-only, so a change of k keeps the vectors."""
    base = index_build_options(_cfg(tmp_path))
    for key in _SEISMIC_DECLARED:
        assert key in index_mod.ASSEMBLE_KEYS[SPARSE_LEG], key
        for leg in LEGS:
            assert key not in index_mod.ENCODE_KEYS[leg], (key, leg)

    # That filing is what the two hashes read, not a description of them: a different k
    # gives the same vectors and a different index.
    pruned = replace(base, seismic_top_k=300)
    unpruned = replace(base, seismic_top_k=0)
    for leg in LEGS:
        assert leg_encode_hash(pruned, leg) == leg_encode_hash(unpruned, leg), leg
    assert leg_config_hash(pruned, SPARSE_LEG) != leg_config_hash(unpruned, SPARSE_LEG)
    assert leg_config_hash(pruned, DENSE_LEG) == leg_config_hash(unpruned, DENSE_LEG)
    assert leg_config_hash(pruned, LATE_INTERACTION_LEG) == (
        leg_config_hash(unpruned, LATE_INTERACTION_LEG)
    )
    # The four declared parameters behave the same way: index identity, not vector identity.
    for key in _SEISMIC_DECLARED[1:]:
        moved = replace(base, **{key: _bump(getattr(base, key))})
        for leg in LEGS:
            assert leg_encode_hash(moved, leg) == leg_encode_hash(base, leg), (key, leg)
        assert leg_config_hash(moved, SPARSE_LEG) != leg_config_hash(base, SPARSE_LEG), key


def test_the_sparse_cut_is_read_from_the_hashed_block_and_zero_means_no_cut(
    tmp_path: Path,
) -> None:
    """SM33: the config is the record, including the value that means "do not cut".

    Here ``0`` means keep every component, which is why this leaf cannot use the
    ``value or default`` idiom the block's other leaves use. That idiom would read a
    declared "unpruned" as the default 300-component cut, giving a config that says one
    thing while the build does another.
    """
    cfg = _cfg(tmp_path)
    assert "seismic_top_k" not in cfg.blocks["index_build"]["config"]
    assert index_build_options(cfg).seismic_top_k == index_mod._DEFAULT_SEISMIC_TOP_K == 300

    cfg.blocks["index_build"]["config"]["seismic_top_k"] = 150
    assert index_build_options(cfg).seismic_top_k == 150
    cfg.blocks["index_build"]["config"]["seismic_top_k"] = 0
    assert index_build_options(cfg).seismic_top_k == 0  # not silently the default
    from ragtime.serving.sparse_milco import NO_TOP_K

    assert NO_TOP_K == 0
    # A negative k describes no build at all, so it fails at load.
    cfg.blocks["index_build"]["config"]["seismic_top_k"] = -1
    with pytest.raises(ConfigError, match="seismic_top_k"):
        index_build_options(cfg)
    del cfg.blocks["index_build"]["config"]["seismic_top_k"]

    # It is recorded in the manifest's provenance beside the parameters it belongs with.
    seismic = index_mod._provenance(cfg, [])["seismic"]
    assert seismic["top_k"] == 300
    assert seismic["max_fraction"] == index_mod._DEFAULT_SEISMIC_MAX_FRACTION


def test_the_declared_vendor_defaults_are_the_vendors_own_current_defaults() -> None:
    """SM33: declaring the defaults changes no byte, checked against the vendor's signature.

    ``SEISMIC_BUILD_SIGNATURE`` is recorded verbatim from pyseismic-lsr 0.5.1's extension
    module. Reading the expectation off it, rather than restating four literals, is what
    makes a version bump that moves a default fail here instead of quietly redefining what
    unchanged behaviour meant.
    """
    signature = index_mod.SEISMIC_BUILD_SIGNATURE
    vendor = dict(
        re.findall(r"(\w+)=([0-9.]+)", signature)  # name=value pairs, numeric ones only
    )
    for name, declared in (
        ("n_postings", index_mod._DEFAULT_SEISMIC_N_POSTINGS),
        ("summary_energy", index_mod._DEFAULT_SEISMIC_SUMMARY_ENERGY),
        ("max_fraction", index_mod._DEFAULT_SEISMIC_MAX_FRACTION),
        ("doc_cut", index_mod._DEFAULT_SEISMIC_DOC_CUT),
        ("min_cluster_size", index_mod._DEFAULT_SEISMIC_MIN_CLUSTER),
    ):
        assert name in vendor, f"{name} is not in the recorded vendor signature"
        assert float(vendor[name]) == float(declared), name


class _FakeSeismicDataset:
    """Records what the writer added, in order. No engine, no build."""

    def __init__(self) -> None:
        self.documents: list[tuple[str, list[str], list[float]]] = []

    def add_document(self, key: Any, components: Any, values: Any) -> None:
        self.documents.append((str(key), list(components), [float(v) for v in values]))


def _fake_seismic(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a stub ``seismic`` module; return the dict the build call is recorded into."""
    seen: dict[str, Any] = {}

    class _Index:
        def save(self, path: str) -> None:
            seen["saved"] = path

    def _build(dataset: Any, **kwargs: Any) -> Any:
        seen["dataset"] = dataset
        seen["kwargs"] = kwargs
        return _Index()

    stub = types.ModuleType("seismic")
    stub.SeismicDataset = _FakeSeismicDataset  # type: ignore[attr-defined]
    stub.SeismicIndex = types.SimpleNamespace(build_from_dataset=_build)  # type: ignore[attr-defined]
    stub.get_seismic_string = lambda: "U30"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "seismic", stub)
    return seen


def test_the_writer_applies_the_declared_cut_to_what_seismic_actually_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM33: the cut reaches the engine, not only the hash, at the shipped call site.

    ``_SeismicWriter`` is the writer both build paths use, since ``assemble._WRITER_OF``
    imports this class, so pruning here is pruning on the shipped assemble path. The stored
    vectors are untouched by construction, because this function never sees them.
    """
    _fake_seismic(monkeypatch)
    opts = IndexBuildOptions(seismic_top_k=2)
    writer = index_mod._SeismicWriter(tmp_path / "sparse", opts)
    writer.add([0, 1], [{5: 0.9, 3: 0.1, 40: 0.5}, {7: 1.0}])

    # The values are compared with a tolerance because the writer hands Seismic float32,
    # the vendor's own value type, so the recorded floats are a float32 round trip.
    docs = writer._dataset.documents
    assert [(key, list(components)) for key, components, _ in docs] == [
        ("0", ["40", "5"]),  # the two heaviest, in component-string order
        ("1", ["7"]),  # below k, so untouched
    ]
    assert docs[0][2] == pytest.approx([0.5, 0.9], rel=1e-6)
    assert docs[1][2] == pytest.approx([1.0])
    # The loss is counted per component rather than assumed.
    assert writer.pruned == 1

    # An unpruned writer over the same input keeps everything and counts zero.
    plain = index_mod._SeismicWriter(tmp_path / "sparse2", IndexBuildOptions(seismic_top_k=0))
    plain.add([0], [{5: 0.9, 3: 0.1, 40: 0.5}])
    key, components, values = plain._dataset.documents[0]
    assert (key, list(components)) == ("0", ["3", "40", "5"])
    assert values == pytest.approx([0.1, 0.5, 0.9], rel=1e-6)
    assert plain.pruned == 0


def test_every_seismic_build_parameter_is_passed_explicitly_to_the_vendor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM33: no build parameter is left to the library; the config is the whole record.

    All six parameters come from the hashed block. Four of them, ``n_postings``,
    ``summary_energy``, ``max_fraction`` and ``doc_cut``, used to come from pyseismic-lsr's
    signature defaults, so the run choice lived in a dependency and a version bump could
    have moved it with nothing in the artefact tree noticing.
    """
    seen = _fake_seismic(monkeypatch)
    opts = IndexBuildOptions(
        seismic_min_cluster_size=3,
        seismic_centroid_fraction=0.05,
        seismic_n_postings=1234,
        seismic_summary_energy=0.25,
        seismic_max_fraction=6.0,
        seismic_doc_cut=11,
    )
    writer = index_mod._SeismicWriter(tmp_path / "sparse", opts)
    writer.finish()
    assert seen["kwargs"] == {
        "n_postings": 1234,
        "centroid_fraction": 0.05,
        "min_cluster_size": 3,
        "summary_energy": 0.25,
        "max_fraction": 6.0,
        "doc_cut": 11,
    }
    # Each is a real value read off the options object, so a config change moves the vendor
    # call. That is the difference between a parameter applied and one merely accepted.
    assert set(seen["kwargs"]) <= {
        name.removeprefix("seismic_") for name in IndexBuildOptions.__dataclass_fields__
    }
    assert seen["saved"].endswith(index_mod._SPARSE_SAVE_STEM)


# --------------------------------------------------------------------------- #
# SM24: the rank-agreement helpers the corpus-scale checks depend on
# --------------------------------------------------------------------------- #
def test_rank_agreement_helpers_return_known_values() -> None:
    """SM24: the metrics, on lists whose agreement can be worked out by hand."""
    a = ["p1", "p2", "p3", "p4"]
    assert kendall_tau_b(a, a) == pytest.approx(1.0)
    assert kendall_tau_b(a, list(reversed(a))) == pytest.approx(-1.0)
    # one adjacent swap out of 6 comparable pairs -> (5 - 1) / 6
    assert kendall_tau_b(a, ["p2", "p1", "p3", "p4"]) == pytest.approx(4 / 6)
    assert kendall_tau_b(["p1"], ["p1"]) == 0.0  # one item: nothing to measure

    assert rbo(a, a) == pytest.approx(1.0)
    assert rbo(["p1", "p2"], ["p3", "p4"]) == pytest.approx(0.0)
    # p=1.0 is the unweighted average overlap: 0/1, 2/2 -> 0.5
    assert rbo(["p1", "p2"], ["p2", "p1"], p=1.0) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# SM27: the instruction-prefix bias channel does not come back
# --------------------------------------------------------------------------- #
def _model_facing_literals(source: str) -> list[str]:
    """Every string literal in ``source`` that could reach a model.

    Comments and docstrings are excluded, because neither can be prepended to a text on its
    way to an encoder, and a plain text search over them raises false positives: it fired
    once on a comment beginning ``# Query:``. Fragments of f-strings are included, since an
    ``f"Instruct: {task}"`` template is the thing this guard exists to catch.
    """
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_the_literal_scanner_sees_templates_and_ignores_prose() -> None:
    """The SM27 guard is worth having only if its scanner cannot be fooled either way."""
    fooled = _model_facing_literals(
        '"""Module doc mentioning Query: and Instruct: in prose."""\n'
        "# a comment mentioning Query: and Instruct:\n"
        "def f(task, q):\n"
        '    """Local doc: Query: prose."""\n'
        "    return q\n"
    )
    assert not [s for s in fooled if "Query:" in s or "Instruct:" in s], fooled
    caught = _model_facing_literals(
        "def f(task, q):\n"
        '    return f"Instruct: {task}\\nQuery: {q}"\n'
        '    PREFIX = "Query: "\n'
    )
    assert [s for s in caught if "Instruct:" in s] and [s for s in caught if "Query:" in s]


def test_no_instruction_prefix_is_prepended_before_the_dense_encode(tmp_path: Path) -> None:
    """SM27: BGE-M3 needs none, and an English-authored template would be a bias channel."""
    build = _build(tmp_path)
    dense = _FakeDense()
    ctx = IndexCtx(
        dense=dense,
        milco=_FakeMilco(),
        mtd=_FakeMtd(),
        opts=build.ctx.opts,
        legs=default_legs(),
        layout=build.layout,
        recon_hash=_RECON,
        pack_hash=None,
        idx_hash=build.ctx.idx_hash,
        tier=build.ctx.tier,
    )
    from ragtime.preprocess.index import FaissDenseLeg

    class _Capture(_FakeDense):
        def __init__(self) -> None:
            super().__init__()
            self.texts: list[str] = []

        def embed(self, texts, mode="dense"):
            self.texts.extend(texts)
            return super().embed(texts, mode=mode)

    capture = _Capture()
    ctx.dense = capture
    FaissDenseLeg().encode_docs(ctx, ["El café contiene 2 g de cafeína."])
    assert capture.texts == ["El café contiene 2 g de cafeína."]
    # No string literal in the module could become such a template either. The shape being
    # kept out is Qwen3-Embedding's `"Instruct: {task}\nQuery: {q}"`: an English-authored
    # prompt nobody can validate is a bias channel in a study about translation into
    # English, which is why the dense leg is BGE-M3, which needs no prefix at all.
    literals = _model_facing_literals(
        Path(index_mod.__file__).read_text(encoding="utf-8")
    )
    assert literals, "scanner found no literals at all: the guard would be vacuous"
    assert not [s for s in literals if "Instruct:" in s or "Query:" in s]


# --------------------------------------------------------------------------- #
# SM26: every leg's artifact is readable by its own native reader
# --------------------------------------------------------------------------- #
def test_every_leg_artifact_is_readable_by_its_own_native_engine(tmp_path: Path) -> None:
    """SM26: each leg serialises through its own engine, not a hand-rolled Arrow table.

    The three real legs run against fake clients, so the engines are exercised and the
    multi-gigabyte checkpoints are not. Skipped where the ``index`` extra is absent.
    """
    pytest.importorskip("faiss")
    pytest.importorskip("seismic")
    pytest.importorskip("pylate")
    build = _build(tmp_path / "real", legs=default_legs())
    build.adapter.merge(build.cfg, build.receipts)
    shard_dir = build.layout.index_shard_dir(_RECON, build.ctx.idx_hash, "omt", "zh", part=0)

    import faiss

    dense = faiss.read_index(str(shard_dir / DENSE_LEG / "dense.faiss"))
    assert dense.ntotal == _PASSAGES_PER_LANG

    from seismic import SeismicIndex

    # Import the name rather than hardcoding it. Seismic's save and load are asymmetric:
    # save takes a stem and appends ".index.seismic", load needs the full filename. A
    # literal here would drift from what the module wrote. That asymmetry once shipped: the
    # leg published with its _SUCCESS marker and could never be reopened.
    sparse = SeismicIndex.load(str(shard_dir / SPARSE_LEG / index_mod._SPARSE_FILENAME))
    assert (sparse.len() if callable(sparse.len) else sparse.len) == _PASSAGES_PER_LANG

    # Query with a passage's own composed text rather than an ad-hoc English phrase. The
    # fake clients derive a text's vocabulary from its sha256, so an arbitrary query shares
    # no dimensions at all with any document, and the sparse leg correctly returns nothing:
    # 0 of the shard's 55 dimensions. That is a property of the stand-in, not of seismic,
    # whose real encoder puts queries and documents in one SPLADE space. Using a real
    # passage keeps the probe meaningful for all three engines and allows the stronger
    # assertion, that the published artefact round-trips to the right id.
    pid = passage_id(_DOC_ZH, 0)
    text = next(
        r["omt"] for r in iter_final_passages(build.layout, _RECON, pack_hash=None) if r["passage_id"] == pid
    )
    handle = open_shard(shard_dir, build.ctx)
    for leg in LEGS:
        hits = query_leg(handle, leg, text, top_k=2)
        assert hits, leg
        assert all(isinstance(p, str) for p, _ in hits), (leg, hits)
        assert hits[0][0] == pid, (leg, hits)  # through the real engine, not a stand-in


# --------------------------------------------------------------------------- #
# SM27: PLAID's add_documents is called exactly once per part.
#
# The hazard: calling `add_documents` more than once on one index -- once per `Batcher`
# bucket, say. The first call on an index takes fast-plaid's `create` path; every later one
# takes `update`, then `compute_kmeans`, then `index_add_`, which trips a device-side
# `dstIndex < dstAddDimSize` assertion and aborts the worker on real embeddings once enough
# units have accumulated. It reproduces on two GPU architectures. A counting fake is enough
# here, because the property is a call count rather than an engine behaviour.
#
# SM28 adds the part axis: a part is a fresh index, so N parts are N `create` calls and no
# `update`. The counting fake keys on the part directory, so "one call per part" and "one
# call per index" are the same assertion.
# --------------------------------------------------------------------------- #
class _CountingPlaid:
    """A PLAID stand-in that records every ``add_documents`` call it receives."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def add_documents(self, documents_ids, documents_embeddings) -> None:
        assert len(documents_ids) == len(documents_embeddings)
        self.calls.append((tuple(documents_ids), len(documents_embeddings)))

    def __call__(self, queries_embeddings, k):  # pragma: no cover - never queried here
        return [[]]


def _patch_plaid(monkeypatch: pytest.MonkeyPatch) -> dict[Path, _CountingPlaid]:
    """Route every ``_open_plaid`` to one counting stand-in per part directory.

    Keyed by ``(leg_dir, part)``, because a part is its own index: two parts of one leg are
    two stand-ins. Sharing one would hide the second ``add_documents`` call on a single
    index, which is the call that aborts the worker.
    """
    made: dict[Path, _CountingPlaid] = {}

    def _fake_open(index_dir: Path, opts, *, override: bool, part: int):
        path = Path(index_dir) / index_mod._plaid_part_name(part)
        path.mkdir(parents=True, exist_ok=True)
        return made.setdefault(path, _CountingPlaid(path))

    monkeypatch.setattr(index_mod, "_open_plaid", _fake_open)
    return made


def test_plaid_adds_every_window_in_exactly_one_call_per_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM27: many buckets in, one ``add_documents`` out, through the real driver.

    The real ``PlaidLateInteractionLeg`` runs off the real ``IndexAdapter`` with a fake
    client, so the bucket count is whatever ``Batcher`` produced. The check has two sides:
    more than one bucket existed, and exactly one call was made. Without the first, the
    test could pass for the wrong reason.
    """
    made = _patch_plaid(monkeypatch)
    legs = (
        _FakeLeg(name=DENSE_LEG),
        _FakeLeg(name=SPARSE_LEG),
        index_mod.PlaidLateInteractionLeg(),
    )
    build = _build(tmp_path, legs=legs)

    spec = IndexShardSpec(variant="omt", source_lang="zh")
    assert len(build.adapter.buckets(build.ctx, build.adapter.load_shard(build.ctx, spec))) > 1

    assert made, "the late-interaction leg never opened a PLAID index"
    for index_dir, plaid in made.items():
        assert len(plaid.calls) == 1, (index_dir, len(plaid.calls))
        ids, count = plaid.calls[0]
        # Every passage of the shard, in ordinal order, in that one call. There is no
        # windowing, so the PLAID document id is the passage ordinal: no second id space and
        # no window map to publish: ``idmap.parquet`` alone resolves a hit, as it does on the
        # other two legs.
        assert list(ids) == [str(i) for i in range(count)]
        assert count == _PASSAGES_PER_LANG


def test_plaid_writer_buffers_across_buckets_and_never_splits_the_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM27: the writer's own contract. N ``add`` calls become one engine call."""
    import numpy as np

    made = _patch_plaid(monkeypatch)
    out_dir = tmp_path / "leg"
    writer = index_mod._PlaidWriter(out_dir, IndexBuildOptions())
    for ordinal in range(6):  # six buckets of one passage each
        writer.add([ordinal], [np.asarray([_vec(f"w{ordinal}")], dtype="float32")])
    assert not made, "add() must not open the index, let alone add to it"
    writer.finish()

    plaid = made[out_dir / index_mod._plaid_part_name(0)]
    assert len(plaid.calls) == 1
    ids, count = plaid.calls[0]
    assert count == 6
    # The document id is the passage ordinal: one unit per passage, with no windowing.
    assert list(ids) == [str(i) for i in range(6)]
    # One part at the shipped size, and the writer records it; nothing downstream infers it.
    assert writer.parts == 1
    assert common_io.read_jsonl(out_dir / index_mod._PARTS_FILENAME)[0]["parts"] == 1


def test_a_shard_that_cannot_be_buffered_raises_instead_of_dropping_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM27: a shard too big to add in one call raises rather than producing a short index.

    A silently capped PLAID leg would publish fewer passages under a full ``idmap.parquet``,
    which is the id-set violation ``validate`` exists to catch, and it would be caught a
    build later rather than here. So the writer aborts while the failure is still a Python
    exception and the leg's temporary directory is discarded unpublished.

    Part-sharding does not retire this. ``plaid_part_passages`` bounds the buffer in
    passages, so a part that does not fit the node still has to raise here. The part size
    makes the guard satisfiable, not unnecessary.
    """
    import numpy as np

    made = _patch_plaid(monkeypatch)
    monkeypatch.setattr(index_mod, "_host_memory_headroom_bytes", lambda: 1)
    writer = index_mod._PlaidWriter(tmp_path / "leg", IndexBuildOptions())
    with pytest.raises(index_mod.PlaidBufferTooLargeError, match="add_documents"):
        writer.add([0], [[np.asarray([_vec(s)], dtype="float32") for s in "ab"]])
    assert not made, "no PLAID index may be created for a shard that cannot be added"


# --------------------------------------------------------------------------- #
# SM28, part-sharding: the build side, the census, and the fan-merge.
#
# Without this there is no corpus-scale late-interaction build at all. A whole English
# shard's token-embedding buffer is a measured 376 GiB in fp32 on the host, which no node
# has spare, so the index is cut into parts of `plaid_part_passages` passages, one
# `add_documents` per part, fanned and merged at query time. The two ways that can go wrong
# silently are a reader fanning over fewer parts than exist, and a merge that is not a total
# order, which would leave the cross-rendering agreement diagnostics measuring tie-breaking.
# --------------------------------------------------------------------------- #
def test_sm28_the_writer_cuts_parts_on_the_ordinal_not_on_the_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM28: part ``p`` holds ordinals ``[p*N, (p+1)*N)`` whatever the buckets were.

    Part membership is a function of the shard's pinned order and ``N`` alone, which is what
    makes it identical across the three renderings. Their ``Batcher`` buckets are equal by
    construction, but the alignment between bucket and part would otherwise be an accident.
    The buckets here are deliberately ragged, 1, 4, 2 and 3 passages against a part size
    of 5.
    """
    import numpy as np

    made = _patch_plaid(monkeypatch)
    out_dir = tmp_path / "leg"
    writer = index_mod._PlaidWriter(out_dir, IndexBuildOptions(plaid_part_passages=5))
    ordinal = 0
    for size in (1, 4, 2, 3):
        writer.add(
            list(range(ordinal, ordinal + size)),
            [np.asarray([_vec(f"w{ordinal + i}")], dtype="float32") for i in range(size)],
        )
        ordinal += size
    writer.finish()

    assert writer.parts == 2  # 10 passages / 5 per part
    for part in range(2):
        plaid = made[out_dir / index_mod._plaid_part_name(part)]
        assert len(plaid.calls) == 1, f"part {part} took {len(plaid.calls)} add_documents"
        ids, count = plaid.calls[0]
        assert count == 5
        assert list(ids) == [str(i) for i in range(part * 5, part * 5 + 5)]

    record = common_io.read_jsonl(out_dir / index_mod._PARTS_FILENAME)[0]
    assert record["parts"] == 2
    assert record["units"] == 10
    assert record["part_passages"] == 5
    assert record["dirs"] == [index_mod._plaid_part_name(i) for i in range(2)]
    assert index_mod.plaid_part_count(out_dir) == 2


def test_sm28_every_parts_clustering_rng_is_pinned_before_the_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM28: the declared seed reaches the RNG that decides the artefact.

    fast-plaid's ``seed`` keyword does not cover the ``torch.randperm`` sample selection in
    ``compute_kmeans``, so without this pin two builds of one part cluster on different
    samples: 6 of 21 files differed, measured by SM29's control. The pin is asserted per part
    and before the add, because a pin applied after the call, or once for the whole shard,
    would leave every part but the first unpinned.
    """
    import numpy as np

    made = _patch_plaid(monkeypatch)
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        index_mod, "_pin_torch_rng", lambda seed: calls.append(("pin", seed)) or True
    )
    original_open = index_mod._open_plaid

    def _tracking_open(index_dir, opts, *, override, part):
        obj = original_open(index_dir, opts, override=override, part=part)
        real_add = obj.add_documents

        def _add(documents_ids, documents_embeddings):
            calls.append(("add", len(documents_ids)))
            return real_add(documents_ids, documents_embeddings)

        obj.add_documents = _add  # type: ignore[method-assign]
        return obj

    monkeypatch.setattr(index_mod, "_open_plaid", _tracking_open)
    opts = IndexBuildOptions(plaid_part_passages=2, quantizer_train_seed=4242)
    writer = index_mod._PlaidWriter(tmp_path / "leg", opts)
    for ordinal in range(4):
        writer.add([ordinal], [np.asarray([_vec(f"w{ordinal}")], dtype="float32")])
    writer.finish()

    assert len(made) == 2
    assert calls == [("pin", 4242), ("add", 2), ("pin", 4242), ("add", 2)]


def test_sm28_an_empty_shard_still_publishes_one_readable_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM28: zero passages is a legal artefact: one empty part, and a census that says so."""
    made = _patch_plaid(monkeypatch)
    out_dir = tmp_path / "leg"
    writer = index_mod._PlaidWriter(out_dir, IndexBuildOptions())
    writer.finish()
    assert writer.parts == 1
    plaid = made[out_dir / index_mod._plaid_part_name(0)]
    assert plaid.calls == []  # created, never added to
    assert index_mod.plaid_part_count(out_dir) == 1


def test_sm28_the_part_size_is_part_of_the_legs_identity(tmp_path: Path) -> None:
    """SM28: two part sizes are two stored indexes, so two ``config_hash`` values.

    Each part is clustered and residual-coded on its own, so the same passages at 262,144
    and at 131,072 per part are different bytes. A recipe hash that ignored the part size
    would advertise them as one leg.
    """
    opts = IndexBuildOptions(plaid_part_passages=262144)
    other = replace(opts, plaid_part_passages=131072)
    assert leg_config_hash(opts, LATE_INTERACTION_LEG) != leg_config_hash(
        other, LATE_INTERACTION_LEG
    )
    # It reaches only this leg; the other two recipes are untouched by it.
    for leg in (DENSE_LEG, SPARSE_LEG):
        assert leg_config_hash(opts, leg) == leg_config_hash(other, leg)


def test_sm28_the_part_size_comes_from_the_hashed_block_and_must_be_positive(
    tmp_path: Path,
) -> None:
    """SM28: the part size comes from the config, and a nonsensical value fails at load."""
    cfg = _cfg(tmp_path)
    cfg.blocks["index_build"]["config"]["plaid_part_passages"] = 4096
    assert index_build_options(cfg).plaid_part_passages == 4096
    # An absent key gives the shipped default, not a silent 0 meaning "do not shard".
    del cfg.blocks["index_build"]["config"]["plaid_part_passages"]
    assert index_build_options(cfg).plaid_part_passages == index_mod._DEFAULT_PLAID_PART_PASSAGES
    cfg.blocks["index_build"]["config"]["plaid_part_passages"] = 0
    with pytest.raises(ConfigError, match="plaid_part_passages"):
        index_build_options(cfg)


def test_sm28_a_reader_never_infers_the_part_count(tmp_path: Path) -> None:
    """SM28: the census is authoritative and cross-checked, and disagreement raises.

    Both directions fail. A leg whose census is missing cannot be opened at all, rather than
    being guessed at from whatever directories happen to be there. A census that names parts
    the disk does not have, or misses parts it does have, raises instead of quietly
    searching a subset. A partial index that looks complete is worse than a failed read,
    because every downstream check would still pass.
    """
    leg_dir = tmp_path / "late_interaction"
    leg_dir.mkdir(parents=True)
    with pytest.raises(IndexIntegrityError, match="parts.json"):
        index_mod._read_parts(leg_dir)
    assert index_mod.plaid_part_count(leg_dir) is None

    for part in range(3):
        (leg_dir / index_mod._plaid_part_name(part)).mkdir()
    common_io.write_jsonl(
        leg_dir / index_mod._PARTS_FILENAME,
        [{"parts": 3, "dirs": [index_mod._plaid_part_name(i) for i in range(3)]}],
        skip_if_done=False,
    )
    assert index_mod.plaid_part_count(leg_dir) == 3

    # A part has vanished from disk while the count still says 3.
    shutil.rmtree(leg_dir / index_mod._plaid_part_name(2))
    with pytest.raises(IndexIntegrityError, match="not the parts on disk"):
        index_mod._read_parts(leg_dir)


def test_sm28_the_fan_merge_is_a_total_deterministic_order(tmp_path: Path) -> None:
    """SM28: merge by score across parts, breaking ties on ordinal rather than part order.

    Two parts returning the same score for different ordinals is the case that decides
    whether two runs of one query return the same list. Sorting on ``(-score, ordinal)``
    makes the answer independent of which part answered first, so the parts are fed in both
    orders and the result has to be identical.
    """

    class _Part:
        def __init__(self, hits):
            self.hits = hits

        def __call__(self, queries_embeddings, k):
            return [[{"id": str(o), "score": s} for o, s in self.hits][: int(k)]]

    # Equal scores across parts (7.0) plus a clear winner and a clear loser.
    left = _Part([(4, 9.0), (1, 7.0), (7, 2.0)])
    right = _Part([(2, 7.0), (9, 5.0), (0, 1.0)])
    leg = index_mod.PlaidLateInteractionLeg()
    ctx = object()

    forward = leg.search((left, right), ctx, object(), top_k=4)
    backward = leg.search((right, left), ctx, object(), top_k=4)
    assert forward == backward
    assert forward == [(4, 9.0), (1, 7.0), (2, 7.0), (9, 5.0)]
    # Hits from both parts survive into the merged top-k.
    assert {o for o, _ in forward} & {1, 7} and {o for o, _ in forward} & {2, 9}
    # A single-part fan is the same code path, not a special case.
    assert leg.search((left,), ctx, object(), top_k=2) == [(4, 9.0), (1, 7.0)]


def _fake_open_plaid_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route ``_open_plaid`` to a stand-in that replays what that part was given.

    The replay goes through a file inside the part directory rather than process memory. The
    build writes into a temporary leg directory that is then renamed, so a memory-backed
    stand-in would answer from a path nobody reads. Persisting it also matches a real
    engine, where a part answers from what is on disk under it.
    """

    class _ReplayPlaid:
        def __init__(self, path: Path) -> None:
            self.path = path / "ids.json"

        def add_documents(self, documents_ids, documents_embeddings) -> None:
            assert len(documents_ids) == len(documents_embeddings)
            self.path.write_text(json.dumps(list(documents_ids)), encoding="utf-8")

        def __call__(self, queries_embeddings, k):
            ids = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
            return [[{"id": i, "score": 1.0 / (1 + int(i))} for i in ids][: int(k)]]

    def _fake_open(index_dir: Path, opts, *, override: bool, part: int):
        path = Path(index_dir) / index_mod._plaid_part_name(part)
        path.mkdir(parents=True, exist_ok=True)
        return _ReplayPlaid(path)

    monkeypatch.setattr(index_mod, "_open_plaid", _fake_open)


def test_sm28_the_fan_reaches_every_part_of_a_multi_part_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM28: through the real adapter and the real fan, no part is left unsearched.

    There is one part size, ``plaid_part_passages``, so a part directory holds exactly one
    PLAID part and the fan that matters is the cell-level one, :func:`query_lang_leg` over
    :func:`read_shard_parts`. The intra-leg merge is covered directly by the total-order
    test above, and :class:`_PlaidWriter`'s multi-part path is kept as the buffer-bound
    mechanism even though config can no longer reach it.

    The cell is 16 passages at 4 per part, so four parts exist. The stand-in engine returns
    every id it was given, which makes "did the fan reach the last part?" answerable by an
    id that could only have come from it. A reader that assumed one part would return four
    ids and look healthy.
    """
    _fake_open_plaid_parts(monkeypatch)
    legs = (
        _FakeLeg(name=DENSE_LEG),
        _FakeLeg(name=SPARSE_LEG),
        index_mod.PlaidLateInteractionLeg(),
    )
    build = _build(tmp_path, legs=legs, part_passages=4)
    parts = _PASSAGES_PER_LANG // 4
    cell_dir = _lang_dir(build, "omt", "zh")
    census = index_mod.read_shard_parts(cell_dir)
    assert census["parts"] == parts == 4
    # One PLAID part per part directory: a shard part is a PLAID part.
    for part_dir in sorted(cell_dir.glob("part-*")):
        assert index_mod.plaid_part_count(part_dir / LATE_INTERACTION_LEG) == 1

    handle = index_mod.open_lang(cell_dir, build.ctx)
    hits = index_mod.query_lang_leg(
        handle, LATE_INTERACTION_LEG, "anything", top_k=_PASSAGES_PER_LANG
    )
    assert len(hits) == _PASSAGES_PER_LANG, "the fan missed a part"
    expected = {
        row["passage_id"]
        for part_dir in sorted(cell_dir.glob("part-*"))
        for row in common_io.read_parquet(
            part_dir / LATE_INTERACTION_LEG / index_mod.IDMAP_FILENAME
        )
    }
    assert {pid for pid, _ in hits} == expected

    # The published manifest carries the count, so a reader is never left to guess it.
    manifest = common_io.read_jsonl(build.adapter.merge(build.cfg, build.receipts))[0]
    cell = manifest["variants"]["omt"]["shards"]["zh"]
    legs = cell["shard_parts"][0]["legs"]
    # ``plaid_parts`` is the leg's own subdivision and ``parts`` on the cell is the part
    # axis. They keep separate names, because one ambiguous "parts" is how a fan over the
    # wrong axis would hide.
    assert legs[LATE_INTERACTION_LEG]["plaid_parts"] == 1
    assert legs[DENSE_LEG]["plaid_parts"] is None
    assert cell["parts"] == len(cell["shard_parts"]) == parts


# --------------------------------------------------------------------------- #
# SM29: part-sharding against the real PLAID engine.
#
# SM28 pins the plumbing with stand-ins. These two run fast-plaid itself, because the claims
# are about the engine: that a fan-merge over independently clustered parts still puts a
# passage's own text at rank 1, and that a one-part build is byte-for-byte the artefact the
# single-call code produced. Skipped where the `index` extra is absent.
# --------------------------------------------------------------------------- #
def test_sm29_a_multi_part_shard_self_retrieves_at_rank_1_through_the_fan_merge(
    tmp_path: Path,
) -> None:
    """SM29: 16 passages at 8 per part gives two real PLAID indexes, merged at query time.

    The part count is asserted first, because the rest is only meaningful if more than one
    part exists. Each part is clustered and residual-coded on its own, so this also
    exercises the claim the merge rests on: that two parts' scores live in one space and can
    be compared rather than only fused by rank.

    With one part size, a part directory holds exactly one PLAID index and the fan that
    spans them is :func:`query_lang_leg`. What the test says about the engine is the same
    either way: two independently clustered indexes, one merged answer, rank 1 from either
    part.
    """
    pytest.importorskip("pylate")
    build = _build(tmp_path / "parts", legs=default_legs(), part_passages=8)
    cell_dir = _lang_dir(build, "omt", "zh")
    census = index_mod.read_shard_parts(cell_dir)
    assert census["parts"] == 2, "the fan-merge would be trivially 1-part"
    part_dirs = sorted(cell_dir.glob("part-*"))
    assert [p.name for p in part_dirs] == census["dirs"]
    for part_dir in part_dirs:
        assert index_mod.plaid_part_count(part_dir / LATE_INTERACTION_LEG) == 1

    records = {
        r["passage_id"]: r
        for r in iter_final_passages(build.layout, _RECON, pack_hash=None)
        if r["lang"] == "zh"
    }
    handle = index_mod.open_lang(cell_dir, build.ctx)
    def ids_of(part_dir: Path) -> list[str]:
        return [
            row["passage_id"]
            for row in common_io.read_parquet(
                part_dir / LATE_INTERACTION_LEG / index_mod.IDMAP_FILENAME
            )
        ]

    # Probe a passage from each part. A merge that dropped the second part would still put
    # part 0's passages at rank 1, so probing only one would show nothing.
    for probe in (ids_of(part_dirs[0])[0], ids_of(part_dirs[-1])[-1]):
        hits = index_mod.query_lang_leg(
            handle, LATE_INTERACTION_LEG, records[probe]["omt"], top_k=3
        )
        assert hits and hits[0][0] == probe, (probe, hits)
    # Two runs of one query return the identical list, ties included.
    text = records[ids_of(part_dirs[-1])[-1]]["omt"]
    assert index_mod.query_lang_leg(
        handle, LATE_INTERACTION_LEG, text, top_k=5
    ) == index_mod.query_lang_leg(handle, LATE_INTERACTION_LEG, text, top_k=5)


def test_sm29_a_one_part_build_is_byte_identical_to_the_single_call_path(
    tmp_path: Path,
) -> None:
    """SM29: at one part, ``_PlaidWriter`` emits exactly the single-call artefact.

    Part-sharding is meant to generalise the old path, not replace it. The reference is the
    vendor call sequence the writer uses when there is one part: construct the PLAID index
    and hand it every unit in one ``add_documents``. The comparison is a map of shas over
    the engine's own files.

    The reference pins the ambient torch RNG, because the path before part-sharding did not
    and was therefore not reproducible at all. Measured on this fixture, two runs of the
    unpinned single-call path differ in 6 of 21 files: ``centroids.npy``, ``ivf.npy``,
    ``0.codes.npy``, ``merged_codes*`` and ``merged_residuals.manifest.json``. The cause is
    ``compute_kmeans`` selecting its training sample with a bare ``torch.randperm`` that
    fast-plaid's ``seed`` keyword never reaches. So "byte-identical to what the code
    produced before" was not a well-defined target until :func:`_pin_torch_rng` made the leg
    reproducible. What is well defined, and what this asserts, is that the writer at one
    part is byte-for-byte the single-call path under the same pinned RNG.

    Both builds land at the same absolute path, the first being moved aside afterwards, so
    any path the engine embeds in its own metadata is identical in both and the comparison
    measures the artefact rather than the directory.
    """
    import numpy as np

    pylate_indexes = pytest.importorskip("pylate.indexes")
    opts = IndexBuildOptions(plaid_nbits=2, quantizer_train_seed=1234)
    ids = [str(i) for i in range(12)]
    embs = [np.asarray([_vec(f"p{i}"), _vec(f"p{i}#2")], dtype="float32") for i in range(12)]
    leg = tmp_path / "leg"

    def _reference(seed: int) -> None:
        """The single-call path, verbatim: one PLAID index, one ``add_documents``."""
        index = pylate_indexes.PLAID(
            index_folder=str(leg),
            index_name=index_mod._plaid_part_name(0),
            nbits=opts.plaid_nbits,
            seed=seed,
            override=True,
        )
        index._pinned = index_mod._pin_torch_rng(seed)  # the writer's own pre-add pin
        index.add_documents(documents_ids=ids, documents_embeddings=embs)

    def _aside(tag: str) -> dict[str, str]:
        moved = tmp_path / tag
        shutil.move(str(leg), str(moved))
        return _sha_tree(moved / index_mod._plaid_part_name(0))

    leg.mkdir(parents=True)
    _reference(opts.train_seed)
    first = _aside("ref1")
    leg.mkdir(parents=True)
    _reference(opts.train_seed)
    second = _aside("ref2")
    assert first == second, (
        "two runs of the pinned single-call path still differ "
        f"({sorted(k for k in first if first.get(k) != second.get(k))}): the leg is not "
        "reproducible, so the artifact tree is not a checkpoint for it"
    )

    leg.mkdir(parents=True)
    writer = index_mod._PlaidWriter(leg, replace(opts, plaid_part_passages=len(ids)))
    for i, emb in enumerate(embs):  # ragged buckets, one part
        writer.add([i], [emb])
    writer.finish()
    assert writer.parts == 1
    sharded = _sha_tree(leg / index_mod._plaid_part_name(0))
    assert sharded == first, (
        "a 1-part build differs from the single-call build it must generalize: "
        f"{sorted(k for k in set(sharded) | set(first) if sharded.get(k) != first.get(k))}"
    )

    # The pinned seed reaches the stored bytes: a second declared seed gives a different
    # artefact. Without that, "pinned" could mean pinned to something constant the recipe
    # does not name, which is the failure the pin exists to remove.
    shutil.rmtree(leg)
    leg.mkdir(parents=True)
    _reference(opts.train_seed + 1)
    assert _aside("ref3") != first


def _sha_tree(root: Path) -> dict[str, str]:
    """``relative path -> sha256`` for every file under ``root``, with mtimes removed.

    fast-plaid's ``*.manifest.json`` sidecars record ``{"<file>": {"rows": N, "mtime": T}}``,
    where the mtime belongs to files whose content is hashed here anyway. Two builds
    straddling a second boundary would differ in those two files and nowhere else, reporting
    a wall clock as a build difference. ``rows`` is kept, so nothing about what was written
    is waved away; only the timestamp goes, and only in files that are nothing but an index
    of timestamps.
    """

    def _stable(path: Path) -> bytes:
        raw = path.read_bytes()
        if not path.name.endswith(".manifest.json"):
            return raw
        record = json.loads(raw)
        for entry in record.values():
            if isinstance(entry, dict):
                entry.pop("mtime", None)
        return json.dumps(record, sort_keys=True).encode()

    return {
        str(path.relative_to(root)): hashlib.sha256(_stable(path)).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_the_headroom_probe_never_invents_a_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """SM27: an unreadable environment means no guard, not a guessed ceiling."""

    def _unreadable(self, *a, **kw):
        raise OSError("no /proc here")

    monkeypatch.setattr(Path, "read_text", _unreadable)
    assert index_mod._host_memory_headroom_bytes() == 0
    writer = index_mod._PlaidWriter(Path("/nonexistent"), IndexBuildOptions())
    writer._bytes = 2**60
    writer._assert_buffer_fits("probe")  # headroom unknown, so no ceiling is invented


# --------------------------------------------------------------------------- #
# SM30, the shard part axis: plan, census, fan-in merge, cross-rendering identity.
#
# One level up from SM28 and SM29's PLAID parts, with the same design: cut on the pinned
# ordinal, write the count down, cross-check it against the disk, then fan and merge on
# score. What is different here is that a shard part is a claimable work unit with its own
# legs and its own ordinal space, so these tests drive the real work queue, not a writer.
# --------------------------------------------------------------------------- #
_SHARD_PART = 5  # 16 passages/lang -> parts of 5, 5, 5, 1


def _lang_dir(build: _Build, variant: str | None, lang: str) -> Path:
    return build.layout.index_lang_dir(_RECON, build.ctx.idx_hash, variant, lang)


def test_sm30_the_census_names_every_part_and_the_manifest_carries_it(
    tmp_path: Path,
) -> None:
    """SM30: every claimable part is built, self-describes, and is counted in one place.

    The census is what makes a short fan detectable, so it is asserted three ways: the
    directories on disk, ``read_shard_parts``'s reconstruction of them, and the published
    manifest. A reader that inferred the count from any one of these alone would be one lost
    directory away from silently searching a fraction of a language.
    """
    build = _build(tmp_path, part_passages=_SHARD_PART)
    expected_parts = -(-_PASSAGES_PER_LANG // _SHARD_PART)  # 4
    assert len(build.receipts) == (len(RENDERINGS) * 3 + 1) * expected_parts

    cell = _lang_dir(build, "omt", "zh")
    on_disk = sorted(p.name for p in cell.glob("part-*"))
    assert on_disk == [f"part-{i:05d}" for i in range(expected_parts)]
    for part_dir in cell.iterdir():
        record = common_io.read_jsonl(part_dir / "part.json")[0]
        assert record["parts"] == expected_parts
        assert record["part_passages"] == _SHARD_PART
        assert record["source_lang"] == "zh" and record["variant"] == "omt"

    census = index_mod.read_shard_parts(cell)
    assert census["parts"] == expected_parts
    assert census["dirs"] == on_disk  # part order, which is on-disk order by construction
    assert sum(census["passages"]) == _PASSAGES_PER_LANG
    # The witness carries no timestamp. ``work`` rewrites it on every re-entry, and a shard
    # built twice has to be byte-identical; FL03 and FL11 caught exactly this at full scale.
    raw = (cell / "part-00000" / "part.json").read_bytes()
    spec = IndexShardSpec(variant="omt", source_lang="zh", part=0, parts=expected_parts)
    replay = tmp_path / "replay" / spec.name
    replay.parent.mkdir(parents=True, exist_ok=True)
    replay.write_text(json.dumps(spec.payload()), encoding="utf-8")
    build.adapter.work(build.ctx, replay)
    assert (cell / "part-00000" / "part.json").read_bytes() == raw

    manifest = common_io.read_jsonl(build.adapter.merge(build.cfg, build.receipts))[0]
    entry = manifest["variants"]["omt"]["shards"]["zh"]
    assert entry["parts"] == expected_parts
    assert entry["part_passages"] == _SHARD_PART
    assert entry["part_dirs"] == on_disk
    assert [p["part"] for p in entry["shard_parts"]] == list(range(expected_parts))
    assert entry["passages"] == _PASSAGES_PER_LANG
    assert manifest["provenance"]["plaid_part_passages"] == _SHARD_PART


def test_sm30_a_lost_part_raises_instead_of_being_fanned_over_short(
    tmp_path: Path,
) -> None:
    """SM30: the hazard the census exists for. Three parts of four must not look healthy."""
    build = _build(tmp_path, part_passages=_SHARD_PART)
    cell = _lang_dir(build, "omt", "zh")

    # A part directory that has gone.
    shutil.rmtree(cell / "part-00002")
    with pytest.raises(IndexIntegrityError, match="searched short"):
        index_mod.read_shard_parts(cell)
    with pytest.raises(IndexIntegrityError):
        index_mod.open_lang(cell, build.ctx)

    # An abandoned attempt: the directory exists but never finished, so it has no witness.
    build2 = _build(tmp_path / "b", part_passages=_SHARD_PART)
    cell2 = _lang_dir(build2, "omt", "zh")
    (cell2 / "part-00003" / "part.json").unlink()
    common_io.success_marker(cell2 / "part-00003" / "part.json").unlink(missing_ok=True)
    with pytest.raises(IndexIntegrityError, match="abandoned attempt"):
        index_mod.read_shard_parts(cell2)

    # Witnesses that disagree about the plan.
    build3 = _build(tmp_path / "c", part_passages=_SHARD_PART)
    cell3 = _lang_dir(build3, "omt", "zh")
    common_io.write_jsonl(
        cell3 / "part-00001" / "part.json",
        [{"part": 1, "parts": 9, "part_passages": _SHARD_PART, "passages": 5}],
        skip_if_done=False,
    )
    with pytest.raises(IndexIntegrityError, match="disagree about the cell's plan"):
        index_mod.read_shard_parts(cell3)


def test_sm30_query_lang_leg_fans_over_every_part_and_merges_by_SCORE(
    tmp_path: Path,
) -> None:
    """SM30: the fan is total, and the merge is a score sort rather than RRF over parts.

    What is asserted is the property that distinguishes them: the global rank-1 passage has
    the highest score anywhere in the cell, even when it sits in a later part. Under RRF
    every part's own best hit would tie at rank 1, and the answer would depend on how the
    corpus happened to be cut.
    """
    build = _build(tmp_path, part_passages=_SHARD_PART)
    cell = _lang_dir(build, "omt", "zh")
    handle = index_mod.open_lang(cell, build.ctx)
    assert len(handle.parts) == 4

    records = {
        r["passage_id"]: r
        for r in iter_final_passages(build.layout, _RECON, pack_hash=None)
        if r["lang"] == "zh"
    }
    query = records[passage_id(_DOC_ZH, 0)]["omt"]
    for leg in LEGS:
        # Every part's own hits, resolved through its own id map.
        per_part = [
            query_leg(part, leg, query, top_k=_PASSAGES_PER_LANG) for part in handle.parts
        ]
        assert sum(len(hits) for hits in per_part) == _PASSAGES_PER_LANG
        flat = [hit for hits in per_part for hit in hits]
        best = max(flat, key=lambda kv: (kv[1], kv[0]))

        merged = index_mod.query_lang_leg(handle, leg, query, top_k=_PASSAGES_PER_LANG)
        assert len(merged) == _PASSAGES_PER_LANG, "the fan missed a part"
        assert {pid for pid, _ in merged} == set(records)
        assert merged == sorted(flat, key=lambda kv: (-kv[1], kv[0]))
        assert merged[0][1] == best[1], "the merge is not ordering by score"
        # The order is total: two runs of one query return the identical list.
        assert merged == index_mod.query_lang_leg(
            handle, leg, query, top_k=_PASSAGES_PER_LANG
        )
        # A truncated top-k is a prefix of the full one, not a per-part quota.
        assert index_mod.query_lang_leg(handle, leg, query, top_k=3) == merged[:3]


def test_sm30_part_membership_is_identical_across_the_three_renderings(
    tmp_path: Path,
) -> None:
    """SM30: part ``p`` of a language holds the same passages in every rendering.

    That holds by construction, since the cut keys come from the text-free passage table,
    and it is asserted anyway: a rank difference between renderings over differently cut
    parts would be measuring the cut instead of the translation.
    """
    build = _build(tmp_path, part_passages=_SHARD_PART)
    manifest = common_io.read_jsonl(build.adapter.merge(build.cfg, build.receipts))[0]
    for lang in ("es", "ru", "zh"):
        for part in range(4):
            digests = {
                manifest["variants"][variant]["shards"][lang]["shard_parts"][part][
                    "id_digest"
                ]
                for variant in RENDERINGS
            }
            assert len(digests) == 1, (lang, part, digests)
    # The guard refuses a build where they differ.
    with pytest.raises(IndexIntegrityError, match="part membership differs"):
        index_mod._assert_parts_identical_across_renderings(
            {("zh", 1): {"original": "a", "omt": "b", "omt_opus": "a"}}
        )


def test_sm30_validate_rejects_a_part_built_under_the_wrong_boundary(
    tmp_path: Path,
) -> None:
    """SM30: a part's legs cover that part's ids, not the cell's and not another part's.

    Before the part axis, ``validate`` compared a shard against a whole language, so a
    shifted boundary could not even be expressed. It is caught by re-deriving the part from
    the table rather than by trusting the artefact.
    """
    build = _build(tmp_path, part_passages=_SHARD_PART)
    receipts = {
        json.loads(p.read_text(encoding="utf-8"))["shard_dir"]: p for p in build.receipts
    }
    cell = _lang_dir(build, "omt", "zh")
    report = json.loads(receipts[str(cell / "part-00002")].read_text(encoding="utf-8"))
    # Claim part 2's artefacts as part 0's.
    report["part"] = 0
    doctored = tmp_path / "doctored.jsonl"
    common_io.write_jsonl(doctored, [report], skip_if_done=False)
    assert build.adapter.validate(doctored) is False


def test_sm30_a_payload_without_the_part_axis_is_refused(tmp_path: Path) -> None:
    """SM30: a queue entry minted before the part axis fails loudly rather than building
    part 0."""
    with pytest.raises(IndexIntegrityError, match="carries no"):
        IndexShardSpec.from_payload({"variant": "omt", "source_lang": "zh"})
    spec = IndexShardSpec(variant="omt", source_lang="zh", part=3, parts=4)
    assert IndexShardSpec.from_payload(spec.payload()) == spec
    assert spec.name == "index_omt_zh_p00003"


def test_sm30_there_is_exactly_one_part_size_and_shard_part_passages_is_gone(
    tmp_path: Path,
) -> None:
    """SM30: exactly one part size, and ``shard_part_passages`` is a key nowhere.

    Vectorize cuts on ``encode_block_passages`` and assemble on ``plaid_part_passages``, so a
    third part-size key would be parsed and hashed into the recipe while deciding nothing, and a
    key that moves a hash without changing a byte is a false provenance record. The schema, the
    options, the assemble key set and the source are all checked together, so a reader cannot be
    left behind on its own. Part directories are PLAID-sized, which makes an English cell 23
    parts rather than 3.
    """
    assert "shard_part_passages" not in IndexBuildOptions.__dataclass_fields__
    assert "shard_part_passages" not in INDEX_BUILD_CONFIG
    for leg in LEGS:
        assert "shard_part_passages" not in index_mod.ASSEMBLE_KEYS[leg]
        assert "shard_part_passages" not in index_mod.ALL_LEG_KEYS[leg]
    source = Path(index_mod.__file__).read_text(encoding="utf-8")
    assert "opts.shard_part_passages" not in source  # no reader was left behind

    # The one remaining part size reaches the leg it decides and no other, because a shard
    # cut at 131,072 is clustered and residual-coded independently per part.
    cfg = _cfg(tmp_path)
    base = index_build_options(cfg)
    cfg.blocks["index_build"]["config"]["plaid_part_passages"] = 4096
    moved = index_build_options(cfg)
    assert moved.plaid_part_passages == 4096 != base.plaid_part_passages
    assert leg_config_hash(base, LATE_INTERACTION_LEG) != (
        leg_config_hash(moved, LATE_INTERACTION_LEG)
    )
    assert leg_config_hash(base, DENSE_LEG) == leg_config_hash(moved, DENSE_LEG)
    assert leg_config_hash(base, SPARSE_LEG) == leg_config_hash(moved, SPARSE_LEG)
    # It is still not a rendering input: one recipe, one hash per leg.
    assert len({leg_config_hash(moved, leg) for leg in LEGS}) == len(LEGS)
    with pytest.raises(ConfigError, match="at least one passage"):
        cfg.blocks["index_build"]["config"]["plaid_part_passages"] = 0
        index_build_options(cfg)


# --------------------------------------------------------------------------- #
# The encode and assemble key partition, which is what keeps the vectorize and assemble
# split safe.
#
# Get the partition wrong and the build reuses stale vectors under a moved recipe hash,
# which is a false provenance record. The danger is structural rather than clerical: if
# "encode-relevant" were an allowlist, a newly added hashed key would default to "not
# encode-relevant" and nothing would notice. So the expectation below is derived from the
# live `INDEX_BUILD_CONFIG` through `leg_recipe_keys`, the same way
# `test_every_key_the_real_configs_use_is_declared` derives its expectation from the shipped
# configs rather than from a hand-written list.
# --------------------------------------------------------------------------- #
def _partition_problems(
    encode: dict[str, frozenset[str]],
    assemble: dict[str, frozenset[str]],
    universe: set[str] | None = None,
) -> list[str]:
    """The two partition properties, per leg, as a list of failures. Empty means it holds."""
    all_leg_keys = leg_recipe_keys(universe)
    problems: list[str] = []
    for leg in LEGS:
        filed = encode[leg] | assemble[leg]
        unfiled = all_leg_keys[leg] - filed
        phantom = filed - all_leg_keys[leg]
        both = encode[leg] & assemble[leg]
        if unfiled:
            problems.append(f"{leg}: filed in NEITHER set: {sorted(unfiled)}")
        if phantom:
            problems.append(f"{leg}: filed but not a recipe key: {sorted(phantom)}")
        if both:
            problems.append(f"{leg}: filed in BOTH sets: {sorted(both)}")
    return problems


def test_the_encode_assemble_key_sets_partition_every_leg_recipe() -> None:
    """``ENCODE | ASSEMBLE == ALL`` and ``ENCODE & ASSEMBLE == {}``, for every leg.

    Being exhaustive is what stops stale vectors being reused. Being disjoint is what stops
    a key counting as both, which would make every assemble-only change re-encode and buy
    nothing.
    """
    assert _partition_problems(index_mod.ENCODE_KEYS, index_mod.ASSEMBLE_KEYS) == []
    # The universe itself is complete: every declared schema key lands in exactly one of the
    # four buckets, so none can hide in the gap between them.
    filed = set().union(*index_mod.ENCODE_KEYS.values(), *index_mod.ASSEMBLE_KEYS.values())
    declared = set(INDEX_BUILD_CONFIG)
    # Nothing hides. Every key the schema declares reaches a bucket, and a new one filed
    # nowhere fails here. This is the direction that stops stale vectors being reused.
    assert declared - (filed | index_mod.STRUCTURAL_KEYS | index_mod.QUERY_TIME_KEYS) == set()
    # No bucket invents a recipe key. A filed name with no schema key behind it would be
    # hashed into a leg recipe that no config can move. ``filed`` is also checked against
    # ``ALL_LEG_KEYS`` by ``_partition_problems``; ``STRUCTURAL_KEYS`` has no other check, so
    # it is pinned here.
    assert (filed | index_mod.STRUCTURAL_KEYS) <= declared
    # Query-time keys are declarable, and ``index_hash`` is invariant to them:
    # ``_without_query_time_keys`` excludes them from the block it hashes, so declaring one
    # re-keys nothing. Every assertion below runs unconditionally on real values, and each
    # fails if the exclusion is removed or a new query-time key is added without excluding
    # it.
    every_leg_key = set().union(*index_mod.ALL_LEG_KEYS.values())
    assert index_mod.QUERY_TIME_KEYS, "QUERY_TIME_KEYS is empty: this test would prove nothing"
    for key in index_mod.QUERY_TIME_KEYS:
        # Not asserted: that every query-time key is an ``IndexBuildOptions`` field.
        # ``query_plaid_device`` is one; ``query_length`` is consumed elsewhere and is not.
        assert key not in every_leg_key, key
        assert key in declared, f"{key} must be declarable in index_build.config"

    # The property itself: moving a query-time key does not move index_hash, while moving a
    # build key does. The second half is the control, since a hasher that returned a
    # constant would pass without it.
    from types import SimpleNamespace

    from ragtime.preprocess.index import index_hash as _index_hash

    # A minimal synthetic blocks dict rather than a round-tripped real config: `cfg.blocks`
    # holds non-mapping values and a MappingProxyType, so neither `deepcopy` nor a JSON round
    # trip survives it. `index_hash` reads only `blocks["index_build"]` and
    # `blocks["packing"]`, so this exercises the hashing property and nothing else.
    def _blocks(**overrides):
        cfg = {"dense_model": "BAAI/bge-m3", "plaid_nbits": 2,
               "query_plaid_device": "cuda:0", "query_length": 32}
        cfg.update(overrides)
        return SimpleNamespace(blocks={"index_build": {"config": cfg},
                                       "packing": {"pack_budget": 511}})

    _base = _index_hash(_blocks())
    for key, moved in (("query_plaid_device", "cpu"), ("query_length", 64)):
        assert _index_hash(_blocks(**{key: moved})) == _base, (
            f"index_hash MOVED when {key} changed. Query-time keys decide how a built index is "
            "READ, never what was written; hashing them re-keys the published tree and orphans "
            "1.1 TB of artifacts for a knob that wrote none of it."
        )
    assert _index_hash(_blocks(dense_model="some/other-encoder")) != _base, (
        "index_hash did not move when dense_model changed: the exclusion is over-broad and the "
        "hash no longer identifies the build."
    )


def test_a_key_filed_in_neither_set_fails_the_partition() -> None:
    """Mis-file one key and the partition check fails.

    Two mutations, because they are two different mistakes. A key dropped from the encode
    set is the stale-vector failure itself: the build would decide ``sparse_max_length`` does
    not affect a vector and reuse vectors encoded at the old window. A key added to both sets
    is the opposite error, and is caught too.
    """
    encode = dict(index_mod.ENCODE_KEYS)
    encode[SPARSE_LEG] = encode[SPARSE_LEG] - {"sparse_max_length"}
    problems = _partition_problems(encode, index_mod.ASSEMBLE_KEYS)
    assert problems and "sparse_max_length" in problems[0], problems

    both = dict(index_mod.ASSEMBLE_KEYS)
    both[DENSE_LEG] = both[DENSE_LEG] | {"dense_model"}
    problems = _partition_problems(index_mod.ENCODE_KEYS, both)
    assert any("BOTH" in p and "dense_model" in p for p in problems), problems


def test_a_new_schema_key_filed_nowhere_fails_at_commit_time() -> None:
    """A key added to ``INDEX_BUILD_CONFIG`` and filed in neither set fails the partition.

    This is the whole safety argument, so it is simulated rather than waited for:
    ``leg_recipe_keys`` is asked what the universe would be with one more declared key, and
    the partition has to break. A leg-scoped name breaks that leg only. A name with no leg
    marker is shared and breaks all three, which is the conservative direction, since an
    unclassified key is treated as everyone's rather than as nobody's.
    """
    universe = set(INDEX_BUILD_CONFIG)
    scoped = _partition_problems(
        index_mod.ENCODE_KEYS, index_mod.ASSEMBLE_KEYS, universe | {"dense_new_knob"}
    )
    assert [p for p in scoped if "dense_new_knob" in p]
    assert not [p for p in scoped if p.startswith(SPARSE_LEG)]

    shared = _partition_problems(
        index_mod.ENCODE_KEYS, index_mod.ASSEMBLE_KEYS, universe | {"a_knob_with_no_leg"}
    )
    assert len([p for p in shared if "a_knob_with_no_leg" in p]) == len(LEGS)


def test_the_keys_that_decide_a_vector_are_on_the_encode_side() -> None:
    """The check the partition alone cannot make: a mis-filed key is still a partition.

    Exhaustive and disjoint catches an unfiled key. It cannot catch ``sparse_max_length``
    filed as assemble, because that is still a partition. So the keys whose encode-relevance
    is a fact rather than a choice, an encoder identity, a tokenizer, an encode window, a
    store form and the ``encode_*`` composition leaves, are recognised by name and required
    to be on the encode side. Deriving that from the name rather than restating it means a
    new ``*_model`` or ``*_length`` key is covered the day it is declared.
    """
    encode_markers = ("model", "revision", "tokenizer", "length", "_store_", "encode_")
    for leg in LEGS:
        for key in sorted(index_mod.ALL_LEG_KEYS[leg]):
            if key.startswith("assemble_"):
                assert key in index_mod.ASSEMBLE_KEYS[leg], key
            elif any(marker in key for marker in encode_markers):
                assert key in index_mod.ENCODE_KEYS[leg], (
                    f"{key} names an encoder identity/window/store form but is filed as "
                    "assemble: an assemble-only change would then reuse vectors it invalidates"
                )
    # Move the sparse window to the assemble side and the partition check fails.
    moved_encode = dict(index_mod.ENCODE_KEYS)
    moved_encode[SPARSE_LEG] = moved_encode[SPARSE_LEG] - {"sparse_max_length"}
    assert _partition_problems(moved_encode, index_mod.ASSEMBLE_KEYS)


def test_every_filed_key_is_a_real_option_field_a_hash_can_read() -> None:
    """``leg_encode_hash`` reads its body off ``IndexBuildOptions``, so a typo is a crash.

    A filed name that is not a field would be silently absent from the hash body if the hash
    were hand-written. It is not, so this pins the property from the other side. The
    assemble set is covered too, because it is built the same way and would otherwise crash
    on a name with no field behind it.
    """
    fields = set(IndexBuildOptions.__dataclass_fields__)
    for leg in LEGS:
        for filed in (index_mod.ENCODE_KEYS[leg], index_mod.ASSEMBLE_KEYS[leg]):
            assert filed <= fields, sorted(filed - fields)
    # The hash is built from the declared set, not from a parallel literal list.
    source = inspect.getsource(leg_encode_hash)
    assert "ENCODE_KEYS[leg]" in source


def test_structural_keys_are_the_ones_the_validator_actually_checks() -> None:
    """``STRUCTURAL_KEYS`` is the only way out of the partition, so it is closed too.

    A key parked here reaches no leg hash, which is the "declared but not applied" failure
    this stage has shipped twice. It is acceptable only when the value is checked instead,
    so every name in the set has to appear in ``_assert_structural_keys``' own source.
    """
    validator = inspect.getsource(index_mod._assert_structural_keys)
    for key in index_mod.STRUCTURAL_KEYS:
        assert f'"{key}"' in validator, key
    # The query-time set is the other way out, and it is closed the same way: every member
    # is named here together with the function that reads it, and the set has to equal
    # exactly the names below, so a new member cannot be added without someone writing down
    # where it is consumed.
    from ragtime.serving import registry

    # Both device keys are query-time rather than recipe keys because neither writes a byte
    # of the index: `_open_plaid` picks the device a search runs on, and
    # `_query_dense_client` picks the device a query string is embedded on for decompose's
    # dedup pre-filter.
    readers = {
        "query_length": registry._mtd_client,
        "query_plaid_device": index_mod._open_plaid,
        "query_encode_device": registry._query_dense_client,
    }
    assert set(readers) == index_mod.QUERY_TIME_KEYS
    for key, reader in readers.items():
        assert key in inspect.getsource(reader), key


def _bump(value: Any) -> Any:
    """A different value of the same type: the smallest change that can be made."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    return f"{value}-moved"


def test_leg_encode_hash_moves_for_encode_keys_and_only_for_them(tmp_path: Path) -> None:
    """An assemble-only change re-encodes nothing.

    Raising ``sparse_max_length`` re-encodes only the sparse leg, 13 GPU-hours of 82.
    Changing ``plaid_nbits`` re-encodes nothing. Without the split either costs all 82.
    """
    base = index_build_options(_cfg(tmp_path))
    baseline = {leg: leg_encode_hash(base, leg) for leg in LEGS}

    for leg in LEGS:
        for key in sorted(index_mod.ENCODE_KEYS[leg]):
            moved = replace(base, **{key: _bump(getattr(base, key))})
            assert leg_encode_hash(moved, leg) != baseline[leg], f"{leg}/{key}"
        for key in sorted(index_mod.ASSEMBLE_KEYS[leg]):
            moved = replace(base, **{key: _bump(getattr(base, key))})
            # An assemble key moves no leg's vectors, not just this leg's.
            for other in LEGS:
                assert leg_encode_hash(moved, other) == baseline[other], f"{leg}/{key}"

    # The two cases named in the docstring.
    sparse_window = replace(base, sparse_max_length=4096)
    assert leg_encode_hash(sparse_window, SPARSE_LEG) != baseline[SPARSE_LEG]
    assert leg_encode_hash(sparse_window, DENSE_LEG) == baseline[DENSE_LEG]
    assert leg_encode_hash(sparse_window, LATE_INTERACTION_LEG) == (
        baseline[LATE_INTERACTION_LEG]
    )
    nbits = replace(base, plaid_nbits=4)
    assert {leg: leg_encode_hash(nbits, leg) for leg in LEGS} == baseline


def test_the_encode_hash_is_rendering_free_and_per_leg_distinct(tmp_path: Path) -> None:
    """The same property ``leg_config_hash`` has: the rendering is not an input."""
    opts = index_build_options(_cfg(tmp_path))
    assert len({leg_encode_hash(opts, leg) for leg in LEGS}) == len(LEGS)
    assert "rendering" not in IndexBuildOptions.__dataclass_fields__
    with pytest.raises(ValueError, match="unknown leg"):
        leg_encode_hash(opts, "bm25")


def test_the_passages_are_an_input_to_a_vector_not_only_to_the_index(tmp_path: Path) -> None:
    """``pack12`` is in the encode hash, because a repacking is different text under a
    different id.

    Vectors keyed by the encoder alone would be reused across two different passage sets
    living side by side under one ``recon12``. It is the same stale-artefact hazard
    ``index_hash`` folds ``packing`` in to close, one level down.
    """
    cfg = _cfg(tmp_path)
    base = index_build_options(cfg)
    assert base.packing_hash  # resolved from the packing block rather than left empty
    repacked = replace(base, packing_hash="0" * 12)
    for leg in LEGS:
        assert leg_encode_hash(repacked, leg) != leg_encode_hash(base, leg)


def test_the_new_recipe_leaves_are_validated_not_merely_accepted(tmp_path: Path) -> None:
    """Every key in a hashed block is applied or validated; none is merely accepted."""
    cfg = _cfg(tmp_path)
    for keys, needle in (
        ({"encode_block_passages": 0}, "at least one passage"),
        ({"encode_device": "auto"}, "encode_device"),
        ({"assemble_device": "auto"}, "assemble_device"),
        ({"assemble_device": "mps"}, "assemble_device"),
        ({"dense_store_dtype": "bfloat16"}, "dense_store_dtype"),
        ({"late_interaction_store_dtype": "int8"}, "late_interaction_store_dtype"),
        ({"sparse_store_format": "csr"}, "sparse_store_format"),
    ):
        with pytest.raises(ConfigError, match=needle):
            index_build_options(_with_index_build(cfg, **keys))
    ok = _with_index_build(
        cfg,
        encode_block_passages=1024,
        encode_device="cuda",
        assemble_device="cpu",
        dense_store_dtype="float16",
        late_interaction_store_dtype="float16",
        sparse_store_format="seismic_u30_f16",
    )
    opts = index_build_options(ok)
    assert opts.encode_block_passages == 1024
    assert opts.late_interaction_store_dtype == "float16"


def test_assemble_device_defaults_to_cpu_and_is_passed_to_the_vendor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU by measurement rather than as a fallback, and handed to fast-plaid.

    Both expensive assemble steps are CPU-bound: Seismic at 268.1 s per 131,072 passages,
    running at 19.3 times parallelism on 32 cores; PLAID's ``add_documents`` at 76.8 s and
    bound by host RAM; FAISS at about 2 s. CPU k-means is also the reproducible one,
    identical across repeats, against three distinct results on cuda. Passing no device at
    all would let the assembled bytes depend on whether the scheduler handed the worker a
    GPU.
    """
    opts = index_build_options(_cfg(tmp_path))
    assert opts.assemble_device == "cpu"

    # It also reaches the vendor. This spies on the PLAID constructor rather than searching
    # the source, because a source search binds to a string that can stop appearing while the
    # property it stands for still holds. Both states are constructed here: the build path,
    # ``override=True``, takes ``assemble_device``; the query path, ``override=False``, takes
    # ``query_plaid_device``; and each is shown to be read per call rather than a constant
    # matching the default.
    seen: list[dict] = []
    stub = types.ModuleType("pylate")
    stub.indexes = types.SimpleNamespace(  # type: ignore[attr-defined]
        PLAID=lambda **kwargs: seen.append(kwargs)
    )
    monkeypatch.setitem(sys.modules, "pylate", stub)
    monkeypatch.delenv("RAGTIME_QUERY_PLAID_DEVICE", raising=False)

    index_mod._open_plaid(tmp_path / "plaid_build_cpu", opts, override=True, part=0)
    assert seen[-1]["device"] == opts.assemble_device == "cpu"
    on_gpu = replace(opts, assemble_device="cuda")
    index_mod._open_plaid(tmp_path / "plaid_build_cuda", on_gpu, override=True, part=0)
    assert seen[-1]["device"] == "cuda"  # genuinely read, not a hardcoded "cpu"
    # The query path reads the other key, so a cuda assemble does not move it.
    index_mod._open_plaid(tmp_path / "plaid_query", on_gpu, override=False, part=0)
    assert seen[-1]["device"] == opts.query_plaid_device

    # A declared cuda assemble on a worker with no cuda fails rather than falling back.
    cuda = replace(opts, assemble_device="cuda")
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is always installed in a real run
        available = False
    if available:  # pragma: no cover - depends on the node this runs on
        assert index_mod._assert_assemble_device(cuda) == "cuda"
    else:
        with pytest.raises(ConfigError, match="assemble_device"):
            index_mod._assert_assemble_device(cuda)
    assert index_mod._assert_assemble_device(opts) == "cpu"


def test_the_plaid_buffer_guard_bounds_the_call_not_our_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``add_documents`` peaks at a measured 2.17 times the buffer handed to it in fp32, and
    3.23 times in fp16.

    A guard comparing headroom against the buffer alone passes at 1.5 times headroom and
    then runs out of memory inside the vendor call, losing the whole leg for that part,
    which is the failure it exists to prevent. It also grew less protective as fp16 shrank
    the buffer, which is backwards: fast-plaid materialises an fp32 tensor whatever the
    input dtype, so the vendor's working set is fp32-sized either way.
    """
    assert index_mod.PLAID_ADD_PEAK_MULTIPLIER >= 3.23  # the fp16 ratio, the larger of the two

    writer = index_mod._PlaidWriter(Path("/nonexistent/plaid"), IndexBuildOptions())
    writer._bytes = 10 * 2**30
    writer._ids = ["0"]
    assert writer._required_bytes() == int(10 * 2**30 * index_mod.PLAID_ADD_PEAK_MULTIPLIER)

    # Headroom above the buffer but below what the call needs: once a silent pass.
    monkeypatch.setattr(
        index_mod, "_host_memory_headroom_bytes", lambda: int(writer._bytes * 1.5)
    )
    with pytest.raises(index_mod.PlaidBufferTooLargeError, match="measured peak RSS"):
        writer._assert_buffer_fits("test")
    # Headroom above the measured requirement still passes.
    monkeypatch.setattr(
        index_mod,
        "_host_memory_headroom_bytes",
        lambda: writer._required_bytes() + 1,
    )
    writer._assert_buffer_fits("test")


# --------------------------------------------------------------------------- #
# The per-part query fan runs concurrently and leaves no trace in the result.
# --------------------------------------------------------------------------- #
# The risk of parallelising `query_lang_leg` is that it changes an answer, so these tests
# vary the width and require the output to be identical, tie-breaks included, and identical
# to a hand-rolled copy of the sequential algorithm the fan replaced.
# --------------------------------------------------------------------------- #
def test_the_part_fan_returns_the_identical_list_at_every_width(tmp_path: Path) -> None:
    """The concurrency changes throughput only: same hits, same scores, same order.

    Four widths over a real four-part cell and all three legs: 1, the sequential path with
    no pool created at all; 2, fewer workers than parts, so they finish in waves; 4, one
    worker per part; and 16, more workers than parts, which is clamped rather than honoured.
    """
    build = _build(tmp_path, part_passages=_SHARD_PART)
    cell = _lang_dir(build, "omt", "zh")
    handle = index_mod.open_lang(cell, build.ctx)
    assert len(handle.parts) == 4, "the fixture must be multi-part or this proves nothing"

    query = next(
        r["omt"]
        for r in iter_final_passages(build.layout, _RECON, pack_hash=None)
        if r["lang"] == "zh"
    )
    for leg in LEGS:
        build.ctx.query_workers = 1
        sequential = index_mod.query_lang_leg(handle, leg, query, top_k=_PASSAGES_PER_LANG)
        assert sequential, "an empty result would make the equalities below vacuous"
        for width in (2, 4, 16):
            build.ctx.query_workers = width
            assert (
                index_mod.query_lang_leg(handle, leg, query, top_k=_PASSAGES_PER_LANG)
                == sequential
            ), f"leg {leg} changed its answer at width {width}"
        # The width-1 path is still the original algorithm: encode once, concatenate the
        # per-part hits in part order, sort by (-score, passage_id), truncate.
        build.ctx.query_workers = 4
        impl = handle.parts[0].leg_impl(leg)
        rep = impl.encode_query(handle.ctx, query)
        by_hand = [
            hit
            for part in handle.parts
            for hit in index_mod.search_with_rep(part, leg, rep, _PASSAGES_PER_LANG)
        ]
        by_hand.sort(key=lambda kv: (-kv[1], kv[0]))
        assert sequential == by_hand[:_PASSAGES_PER_LANG]


def test_the_fan_collects_in_part_order_never_in_completion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tie-break does not depend on which engine answered first.

    Part 0 is made the slowest, so completion order is the reverse of part order, and every
    hit carries the same score. That is the case where the stable sort one level up would
    otherwise resolve the tie by arrival. ``Executor.map`` yields in submission order, so
    the concatenation is part-ordered whatever the engines do.
    """
    import time as _time

    parts = [0, 1, 2, 3]

    def _search(part: int) -> list[tuple[str, float]]:
        _time.sleep(0.05 * (len(parts) - part))
        return [(f"p{part}", 1.0)]

    ordered = [("p0", 1.0), ("p1", 1.0), ("p2", 1.0), ("p3", 1.0)]
    assert index_mod._fan_parts(parts, _search, workers=4) == ordered
    assert index_mod._fan_parts(parts, _search, workers=1) == ordered

    # The pool is never wider than there is work for it, and is not created at all when
    # there is nothing to overlap.
    widths: list[int] = []
    real = index_mod.ThreadPoolExecutor

    def _spy(*args: Any, **kwargs: Any) -> Any:
        widths.append(int(kwargs["max_workers"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(index_mod, "ThreadPoolExecutor", _spy)
    index_mod._fan_parts(parts, _search, workers=32)
    index_mod._fan_parts(parts, _search, workers=1)
    index_mod._fan_parts(parts[:1], _search, workers=32)
    assert widths == [4], "a pool was created for a one-part fan, or was wider than the fan"


def test_the_fan_width_is_config_driven_with_a_bounded_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``execution.query_part_workers`` is read from the config, not a literal in the code."""
    from ragtime.config.schema import _ALLOWED

    cfg = _cfg(tmp_path)
    assert "query_part_workers" in _ALLOWED["execution"]

    # Absent: derived from the allocation rather than the node, and bounded on both sides.
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    assert index_mod.default_query_workers() == index_mod.QUERY_WORKERS_CAP
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert index_mod.default_query_workers() == 2
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert index_mod.default_query_workers() == 1
    assert index_mod.query_part_workers(cfg) == index_mod.default_query_workers()

    # Present: honoured verbatim, including above the cap, which bounds only the derivation.
    cfg.blocks["execution"]["query_part_workers"] = 12
    assert index_mod.query_part_workers(cfg) == 12
    assert index_mod.resolve_query_workers(0) == 1  # falls back to sequential, not to nothing
    with pytest.raises(ConfigError, match="query_part_workers"):
        index_mod.resolve_query_workers("wide")


def test_the_fan_width_is_a_width_and_therefore_re_keys_nothing(tmp_path: Path) -> None:
    """Retuning latency does not orphan an index.

    This is the same argument ``vectorize_blocks_per_task`` records, on the query side. The
    knob lives in the unshared ``execution`` block, so it reaches neither the index recipe
    hash nor any leg's identity, and two members of a run family may disagree about it.
    """
    cfg = _cfg(tmp_path)

    def _identity(c: Any) -> tuple:
        opts = index_build_options(c)
        return (index_hash(c), tuple(leg_config_hash(opts, leg) for leg in LEGS))

    before = _identity(cfg)
    cfg.blocks["execution"]["query_part_workers"] = 31
    assert _identity(cfg) == before
    assert "query_part_workers" not in INDEX_BUILD_CONFIG


# --------------------------------------------------------------------------- #
# A fast twin of the harness in tests/preprocess/test_index_full.py.
#
# The corpus-scale build runs through `_TimedLeg`, a wrapper whose contract is that every
# call delegates: it measures the real leg and fakes nothing. A wrapper that stopped
# forwarding a method does not fail at import. It fails with an AttributeError partway
# through a ten-shard build with the real checkpoints resident, after all the expensive
# part. The leg protocol is six names long and can be read in milliseconds, so it is read
# here instead.
# --------------------------------------------------------------------------- #
def test_the_full_gates_timing_wrapper_forwards_the_whole_leg_protocol() -> None:
    """Every public member of every real leg exists on ``_TimedLeg``.

    ``default_legs()`` is the shipped trio and costs about a second to construct, since no
    model loads until ``bringup``, so the comparison is against the real classes rather than
    a hand-copied list of names that would drift from them.
    """
    from ragtime.preprocess.index import default_legs
    from tests.preprocess.test_index_full import _TimedLeg, _TimedWriter

    wrapper = set(dir(_TimedLeg))
    for leg in default_legs():
        expected = {name for name in dir(leg) if not name.startswith("_")}
        assert expected, type(leg).__name__
        missing = sorted(expected - wrapper)
        assert not missing, (
            f"_TimedLeg does not forward {missing} of {type(leg).__name__}: the index build gate would "
            "raise AttributeError partway through a real 10-shard build"
        )
    # `_TimedWriter` forwards the two timed calls explicitly and everything else through
    # __getattr__, so only that pair can go missing.
    assert {"add", "finish"} <= set(dir(_TimedWriter))
