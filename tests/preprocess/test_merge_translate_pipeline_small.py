"""Spine, merge and translate, end to end on one fixture corpus, with no model.

What ``merge`` writes is what ``translate_omt`` reads. Both stages run through ``Layout``'s
own paths and both adapters' real ``shards``, ``work``, ``validate`` and ``merge`` hooks,
with a stub MT client standing in for NLLB, so the file is about the seam between them:

- the merge map is consumable by ``translate_omt`` unchanged;
- the two arms together cover the sentence inventory exactly once, so every non-English
  sentence is either stored under its own id or is a constituent of a unit that is, and
  every English sentence has its own identity row;
- the shard count is a throughput knob and moves no byte of the merged artefacts.
"""

from __future__ import annotations

import json
import types

import pytest

from ragtime.common import Layout
from ragtime.common.io import iter_parquet
from ragtime.preprocess import merge as merge_mod
from ragtime.preprocess import spine
from ragtime.preprocess import translate_omt as tr

pytestmark = pytest.mark.small

_MIN = 4
_CAP = 12.0
_CHUNKER_HASH = "c" * 64
_MERGE_HASH = "m" * 64
_TRANSLATION_HASH = "t" * 64


def _count(text: str) -> int:
    return len(text.split())


class StubMt:
    """Echoes each unit's source, keeping every boundary marker it was given."""

    model = "stub-nllb"

    def assert_marker_atomic(self, marker: str, probe_lang: str = "spa_Latn") -> None:
        assert marker and probe_lang

    def tokenize(self, text: str, src_lang: str, **_kw) -> list[str]:
        assert src_lang
        return text.split()

    def translate_batch(self, items, **_kw) -> dict[str, str]:
        return {it.sentence_id: f"EN[{it.text}]" for it in items}


def _corpus() -> list[dict]:
    return [
        {
            "document_id": "spa-docs/0000001",
            "lang": "es",
            "paragraphs": [
                ["Titulo", "el cuerpo de la primera frase"],
                ["otra frase de cuerpo bastante", "corta"],
            ],
        },
        {
            "document_id": "eng-docs/0000002",
            "lang": "en",
            "paragraphs": [["Headline", "the body of this english sentence"]],
        },
        {
            "document_id": "zho-docs/0000003",
            "lang": "zh",
            "paragraphs": [["ab", "cd ef gh ij kl"], ["mn op qr st"]],
        },
        {
            "document_id": "rus-docs/0000004",
            "lang": "ru",
            "paragraphs": [["odna", "dve tri chetyre pyat shest"]],
        },
    ]


def _drive(adapter, ctx, table, documents, wq_root, *, n_shards: int):
    """The driver lifecycle in process: seed, then work and validate each shard, then merge."""
    ranges, boundary_docs = spine.plan_shards(table, n_shards=n_shards)
    aligned = spine.align_documents(documents, ranges, boundary_docs)
    outs = []
    for r in aligned:
        shard = wq_root / "running" / r.name
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps(r.payload()), encoding="utf-8")
        out = adapter.work(ctx, shard)
        assert adapter.validate(out), r.name
        outs.append(out)
    return outs


def _run_build(tmp_path, build_corpus_tables, *, n_shards: int, name: str):
    paths = build_corpus_tables(_corpus(), name=name)
    base = tmp_path / f"base-{name}"
    layout = Layout(run_dir=base, base=base, family="e2e", chunker_hash=_CHUNKER_HASH)

    # -- merge ------------------------------------------------------------- #
    merge_adapter = merge_mod.MergeAdapter(merge_hash=_MERGE_HASH, base=str(base))
    merge_ctx = merge_mod._MergeCtx(
        tokenizer=types.SimpleNamespace(count=_count),
        documents=paths["documents"],
        sentences=paths["sentences"],
        min_tokens=_MIN,
        cap=_CAP,
        cross_paragraph="singleton_only",
        languages=("zh", "ru", "es"),
        enabled=True,
    )
    shard_outs = _drive(
        merge_adapter,
        merge_ctx,
        paths["sentences"],
        paths["documents"],
        base / "wq" / "merge",
        n_shards=n_shards,
    )
    from ragtime.common.io import concat_parquet
    from ragtime.common.schemas import merge_map_arrow_schema

    map_path = layout.merge_map_path(_MERGE_HASH)
    concat_parquet(map_path, sorted(shard_outs), schema=merge_map_arrow_schema())

    # -- translate: the MT arm reads the map merge just wrote -------------- #
    mt_adapter = _adapter(tr.OmtTranslateAdapter, base)
    mt_ctx = tr._TranslateCtx(
        mt=StubMt(),
        documents=paths["documents"],
        merge_map=map_path,
        model_id="stub-nllb",
        model_config_hash=_TRANSLATION_HASH,
        merge_hash=_MERGE_HASH,
    )
    mt_outs = _drive(
        mt_adapter,
        mt_ctx,
        map_path,
        paths["documents"],
        base / "wq" / "mt",
        n_shards=n_shards,
    )

    # -- translate: the English identity arm reads the sentence table ------ #
    id_adapter = _adapter(tr.OmtIdentityAdapter, base)
    id_ctx = tr._IdentityCtx(
        documents=paths["documents"],
        sentences=paths["sentences"],
        model_config_hash=_TRANSLATION_HASH,
        merge_hash=_MERGE_HASH,
    )
    id_outs = _drive(
        id_adapter,
        id_ctx,
        paths["sentences"],
        paths["documents"],
        base / "wq" / "identity",
        n_shards=n_shards,
    )

    from ragtime.common.schemas import translation_raw_arrow_schema

    mt_path = layout.translations_raw_path("omt", _TRANSLATION_HASH, _MERGE_HASH)
    id_path = layout.translations_raw_path(
        "omt", _TRANSLATION_HASH, _MERGE_HASH, part="identity"
    )
    concat_parquet(mt_path, sorted(mt_outs), schema=translation_raw_arrow_schema())
    concat_parquet(id_path, sorted(id_outs), schema=translation_raw_arrow_schema())
    return {
        "sentences": paths["sentences"],
        "merge_map": map_path,
        "mt": mt_path,
        "identity": id_path,
    }


def _adapter(cls, base):
    cfg = types.SimpleNamespace(
        run_id="e2e-omt",
        blocks={
            "chunker": {"config": {}},
            "translation": {"config": {}},
            "merge": {},
            "execution": {},
        },
    )
    return cls(cfg, base=str(base))


def test_merge_output_is_consumed_by_translate_unchanged(tmp_path, build_corpus_tables):
    out = _run_build(tmp_path, build_corpus_tables, n_shards=3, name="e2e")

    sentences = list(iter_parquet(out["sentences"]))
    map_rows = list(iter_parquet(out["merge_map"]))
    mt_rows = list(iter_parquet(out["mt"]))
    id_rows = list(iter_parquet(out["identity"]))

    non_en = {r["sentence_id"] for r in sentences if r["lang"] != "en"}
    en = {r["sentence_id"] for r in sentences if r["lang"] == "en"}

    # The map is total over the non-English sentences and carries no English row.
    assert {r["sentence_id"] for r in map_rows} == non_en
    assert merge_mod.merge_map_violation(map_rows) is None

    # Every non-English sentence is reachable: either it keys a row, or it is a constituent
    # of a unit whose first constituent does.
    stored = {r["sentence_id"] for r in mt_rows}
    unit_of = {r["sentence_id"]: r["merge_unit_id"] for r in map_rows}
    assert all(sid in stored or unit_of[sid] in stored for sid in non_en)
    assert stored <= non_en

    # English passes through unchanged: one row per sentence, and no merge provenance.
    assert {r["sentence_id"] for r in id_rows} == en
    assert all(r["merge_unit_id"] is None for r in id_rows)
    assert all(r["model_id"] == "identity" for r in id_rows)

    # Both tables carry both hashes, so a row traces back to the rule and to the model.
    for r in mt_rows + id_rows:
        assert r["model_config_hash"] == _TRANSLATION_HASH
        assert r["merge_hash"] == _MERGE_HASH
        assert r["variant"] == "omt"


def test_shard_count_changes_no_byte_of_either_artifact(tmp_path, build_corpus_tables):
    """Shard count is a throughput knob and does not change an artefact's contents."""
    few = _run_build(tmp_path, build_corpus_tables, n_shards=1, name="few")
    many = _run_build(tmp_path, build_corpus_tables, n_shards=7, name="many")
    for key in ("merge_map", "mt", "identity"):
        a = [dict(r) for r in iter_parquet(few[key])]
        b = [dict(r) for r in iter_parquet(many[key])]
        for row in a + b:
            row.pop("created_at", None)  # a wall-clock stamp, not a content field
        assert a == b, key
