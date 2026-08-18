"""Spine, merge, translate, reconcile and pack, end to end on one fixture with no model.

What the two translate arms write is what reconciliation reads. Both run through
``Layout``'s own paths and the real ``shards``, ``work``, ``validate`` and ``merge`` hooks,
with a stub MT client whose behaviour is the whole point: ``LossyMt`` returns the boundary
marker for some units and drops it for others, so one run exercises the split path and the
fuse path together, in roughly the proportions the corpus shows.

What it pins:

- the three ``final/<recon12>/`` tables reconciliation owns exist, resolve through
  ``Layout``, and carry the pinned schemas' column order and types. Passages are not among
  them; they are ``preprocess.packing``'s single output under its own ``pack12`` level;
- the final inventory is dense per document and every span slices ``documents.text``;
- ``translations`` has exactly one row per final sentence, no ``§`` anywhere, and English
  identity rows are byte-identical to their source spans;
- ``remap`` is total over the inventory as it was before reconciliation, and lands only on
  real final ids;
- every passage member resolves in the final inventory and no final sentence is orphaned;
- the shard count is a throughput knob and moves no byte of an artefact;
- the sub-table alignment survives documents that fall in neither raw part's range, English
  in the MT part and non-English in the identity part. That case is what made
  ``spine.align_documents`` insufficient on its own.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from ragtime.common import Layout
from ragtime.common.io import concat_parquet, iter_parquet, parquet_row_group_sizes
from ragtime.common.schemas import merge_map_arrow_schema, translation_raw_arrow_schema
from ragtime.preprocess import merge as merge_mod
from ragtime.preprocess import packing as pk
from ragtime.preprocess import reconcile as rc
from ragtime.preprocess import spine
from ragtime.preprocess import translate_omt as tr
from ragtime.preprocess.merge_join import MARKER

pytestmark = pytest.mark.small

_CHUNKER_HASH = "c" * 64
_MERGE_HASH = "m" * 64
_TRANSLATION_HASH = "t" * 64
_RECON_HASH = "r" * 64
_MIN = 4
_CAP = 12.0


def _count(text: str) -> int:
    return len(text.split())


class NoCountTokenizer:
    """The packer needs ``num_special()`` and nothing else, so ``count`` never runs."""

    def num_special(self) -> int:
        return 0

    def count(self, text: str) -> int:  # pragma: no cover - must never run
        raise AssertionError(f"reconciliation re-tokenized {text!r}")


class LossyMt:
    """Echoes each unit's source, but drops the marker for every second merged unit.

    On the real corpus about 35 % of merged units come back unsplittable. A stub that always
    returned the marker would leave the fuse path, the branch that renumbers the inventory
    and re-packs the document, untested.
    """

    model = "stub-nllb"

    def __init__(self) -> None:
        self._merged_seen = 0

    def assert_marker_atomic(self, marker: str, probe_lang: str = "spa_Latn") -> None:
        assert marker and probe_lang

    def tokenize(self, text: str, src_lang: str, **_kw) -> list[str]:
        assert src_lang
        return text.split()

    def translate_batch(self, items, **_kw) -> dict[str, str]:
        out: dict[str, str] = {}
        for it in sorted(items, key=lambda i: i.sentence_id):
            if MARKER in it.text:
                self._merged_seen += 1
                if self._merged_seen % 2 == 0:
                    out[it.sentence_id] = f"EN[{it.text.replace(MARKER, '')}]"
                    continue
            out[it.sentence_id] = f"EN[{it.text}]"
        return out


def _corpus() -> list[dict]:
    """Four documents: two entirely English, two non-English with short mergeable sentences."""
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
            "paragraphs": [["See § 1983 rules", "the body of this english sentence"]],
        },
        {
            "document_id": "zho-docs/0000003",
            "lang": "zh",
            "paragraphs": [["ab", "cd ef gh ij kl"], ["mn", "op qr st uv wx"]],
        },
        {
            "document_id": "eng-docs/0000004",
            "lang": "en",
            "paragraphs": [["another english headline here", "and a second body sentence"]],
        },
    ]


def _drive(adapter, ctx, table, documents, wq_root, *, n_shards: int, planner=None):
    """The driver lifecycle in process: plan, then work and validate each shard, then collect."""
    ranges, boundary_docs = spine.plan_shards(table, n_shards=n_shards)
    aligned = spine.align_documents(documents, ranges, boundary_docs)
    outs = []
    for i, r in enumerate(aligned):
        shard = wq_root / "running" / r.name
        shard.parent.mkdir(parents=True, exist_ok=True)
        payload = planner(aligned)[i] if planner else r.payload()
        shard.write_text(json.dumps(payload), encoding="utf-8")
        out = adapter.work(ctx, shard)
        assert adapter.validate(out), r.name
        outs.append(out)
    return outs


def _drive_pack(adapter, ctx, table, wq_root, *, n_shards: int):
    """The packing stage shards its own work table, with no ``documents.parquet``.

    That absence is the point. Packing branches on token counts and paragraph indices, so it
    never opens the corpus text.
    """
    ranges, _ = spine.plan_shards(table, n_shards=n_shards)
    outs = []
    for r in ranges:
        shard = wq_root / "running" / r.name
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps(r.payload()), encoding="utf-8")
        out = adapter.work(ctx, shard)
        assert adapter.validate(out), r.name
        outs.append(out)
    cfg = types.SimpleNamespace(
        run_id="e2e-omt",
        blocks={
            "chunker": {"config": {}},
            "translation": {"config": {}},
            "merge": {},
            "reconcile": {},
            "packing": {"pack_length": "native", "pack_budget": 12},
            "execution": {},
        },
    )
    adapter.merge(cfg, outs)
    return outs


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


def _run_build(
    tmp_path, build_corpus_tables, *, n_shards: int, name: str, store_identity: bool = True
) -> dict[str, Path]:
    paths = build_corpus_tables(_corpus(), name=name)
    base = tmp_path / f"base-{name}"
    layout = Layout(run_dir=base, base=base, family="e2e", chunker_hash=_CHUNKER_HASH)
    docs, sents = paths["documents"], paths["sentences"]

    # -- merge ------------------------------------------------------------- #
    merge_adapter = merge_mod.MergeAdapter(merge_hash=_MERGE_HASH, base=str(base))
    merge_ctx = merge_mod._MergeCtx(
        tokenizer=types.SimpleNamespace(count=_count),
        documents=docs,
        sentences=sents,
        min_tokens=_MIN,
        cap=_CAP,
        cross_paragraph="singleton_only",
        languages=("zh", "ru", "es"),
        enabled=True,
    )
    map_path = layout.merge_map_path(_MERGE_HASH)
    concat_parquet(
        map_path,
        sorted(_drive(merge_adapter, merge_ctx, sents, docs, base / "wq/merge", n_shards=n_shards)),
        schema=merge_map_arrow_schema(),
    )

    # -- translate (both arms) --------------------------------------------- #
    mt_ctx = tr._TranslateCtx(
        mt=LossyMt(),
        documents=docs,
        merge_map=map_path,
        model_id="stub-nllb",
        model_config_hash=_TRANSLATION_HASH,
        merge_hash=_MERGE_HASH,
    )
    mt_path = layout.translations_raw_path("omt", _TRANSLATION_HASH, _MERGE_HASH)
    concat_parquet(
        mt_path,
        sorted(
            _drive(
                _adapter(tr.OmtTranslateAdapter, base),
                mt_ctx,
                map_path,
                docs,
                base / "wq/mt",
                n_shards=n_shards,
            )
        ),
        schema=translation_raw_arrow_schema(),
    )
    id_ctx = tr._IdentityCtx(
        documents=docs,
        sentences=sents,
        model_config_hash=_TRANSLATION_HASH,
        merge_hash=_MERGE_HASH,
    )
    id_path = layout.translations_raw_path(
        "omt", _TRANSLATION_HASH, _MERGE_HASH, part="identity"
    )
    concat_parquet(
        id_path,
        sorted(
            _drive(
                _adapter(tr.OmtIdentityAdapter, base),
                id_ctx,
                sents,
                docs,
                base / "wq/identity",
                n_shards=n_shards,
            )
        ),
        schema=translation_raw_arrow_schema(),
    )

    # -- reconcile: the real adapter, over the real sub-table alignment ----- #
    adapter = rc.ReconcileAdapter(recon_hash=_RECON_HASH, base=str(base))
    ctx = rc._ReconcileCtx(
        documents=docs,
        sentences=sents,
        merge_map=map_path,
        raw_data=mt_path,
        raw_identity=id_path,
        variant="omt",
        split_back=True,
        min_segment_chars=1,
        marker=MARKER,
        store_identity=store_identity,
    )

    def _plan(aligned):
        ordinals = [r.doc_row_start for r in aligned]
        m = rc.align_subtable(docs, map_path, ordinals)
        d = rc.align_subtable(docs, mt_path, ordinals)
        i = rc.align_subtable(docs, id_path, ordinals)
        return [
            rc.ReconcileShard(rows=r, map_rows=mm, raw_rows=dd, identity_rows=ii).payload()
            for r, mm, dd, ii in zip(aligned, m, d, i, strict=True)
        ]

    _drive(adapter, ctx, sents, docs, base / "wq/reconcile", n_shards=n_shards, planner=_plan)
    # `merge` is handed the whole `out/` listing, as the driver does. This stage writes three
    # files per shard and `work` returns only the primary one, so passing the return values
    # alone would merge one table and leave the other two empty.
    out_dir = base / "wq/reconcile" / "out"
    outs = [
        f
        for f in sorted(out_dir.glob("*"))
        if f.is_file() and not f.name.endswith("._SUCCESS")
    ]
    cfg = types.SimpleNamespace(
        run_id="e2e-omt",
        blocks={
            "chunker": {"config": {}},
            "translation": {"config": {}},
            "merge": {},
            "reconcile": {
                "split_back": True,
                "min_segment_chars": 1,
                "store_identity_translations": store_identity,
            },
            "execution": {},
        },
    )
    adapter.merge(cfg, outs)
    final = adapter.final_paths(cfg)
    # Packing is its own stage. It reads the inventory reconciliation just published and
    # writes one artefact under its own `pack12`. Running it here keeps the chain whole:
    # what reconcile writes is what packing reads, through `Layout`.
    pack_cfg = types.SimpleNamespace(
        run_id=cfg.run_id,
        blocks={**cfg.blocks, "packing": {"pack_length": "native", "pack_budget": 12}},
    )
    pack = pk.PackAdapter(
        recon_hash=_RECON_HASH, pack_hash=pk.packing_hash(pack_cfg), base=str(base)
    )
    pctx = pk._PackCtx(
        tokenizer=NoCountTokenizer(),
        sentences=final["sentences"],
        len_max=None,
        token_budget=12,
        overlap_frac=0.0,
        prefer_paragraph_break=False,
        prefer_min_fill=0.6,
    )
    _drive_pack(pack, pctx, final["sentences"], base / "wq/packing", n_shards=n_shards)
    passages = pack.out_path(pack_cfg)
    # `spine_sentences` is the inventory before reconciliation and `sentences` is the final
    # one. They are different tables with the same column set, which is why they get
    # separate keys here.
    return {
        "documents": docs,
        "spine_sentences": sents,
        "manifest": adapter.manifest_path(cfg),
        "passages": passages,
        **dict(final),
    }


def _built(tmp_path, build_corpus_tables, n_shards=3, name="e2e"):
    return _run_build(tmp_path, build_corpus_tables, n_shards=n_shards, name=name)


# --------------------------------------------------------------------------- #
# Artefact shape.
# --------------------------------------------------------------------------- #
def test_reconcile_emits_the_three_final_tables_plus_a_manifest(tmp_path, build_corpus_tables):
    out = _built(tmp_path, build_corpus_tables)
    for key in ("sentences", "translations", "remap", "manifest"):
        assert out[key].exists(), key
    # All of them live under the one composite-keyed node.
    node = out["sentences"].parent
    assert node.name == _RECON_HASH[:12]
    assert node.parent.name == "final"
    assert out["translations"].parent == node / "translations"

    manifest = json.loads(out["manifest"].read_text(encoding="utf-8").strip())
    assert manifest["reconcile_hash"] == _RECON_HASH
    assert set(manifest["parents"]) == {"chunker_hash", "merge_hash", "translation_hash"}
    assert set(manifest["tables"]) == {"sentences", "translations", "remap"}
    for entry in manifest["tables"].values():
        assert len(entry["sha256"]) == 64
        assert entry["rows"] >= 0


def test_final_inventory_is_dense_per_document_and_slices_its_own_text(
    tmp_path, build_corpus_tables
):
    out = _built(tmp_path, build_corpus_tables)
    texts = {r["document_id"]: r["text"] for r in iter_parquet(out["documents"])}
    per_doc: dict[str, list[dict]] = {}
    for r in iter_parquet(out["sentences"]):
        per_doc.setdefault(r["document_id"], []).append(r)
    assert per_doc, "no final sentences at all"
    for doc_id, rows in per_doc.items():
        assert [r["sentence_index"] for r in rows] == list(range(len(rows))), doc_id
        assert [r["sentence_id"] for r in rows] == [
            f"{doc_id}#s{k}" for k in range(len(rows))
        ], doc_id
        text = texts[doc_id]
        prev_end = -1
        for r in rows:
            assert 0 <= r["start"] < r["end"] <= len(text)
            assert text[r["start"] : r["end"]].strip(), r["sentence_id"]
            assert r["start"] >= prev_end
            prev_end = r["end"]


def test_fusion_actually_happened_so_the_fuse_path_is_really_covered(
    tmp_path, build_corpus_tables
):
    """A run in which nothing ever fused would say nothing about the harder path."""
    out = _built(tmp_path, build_corpus_tables)
    remap = list(iter_parquet(out["remap"]))
    assert any(r["fused"] for r in remap), "the stub never dropped a marker: test is vacuous"
    n_final = sum(1 for _ in iter_parquet(out["sentences"]))
    assert n_final < len(remap), "fusion must shrink the inventory"


# --------------------------------------------------------------------------- #
# The same properties again, this time on the merged artefacts.
# --------------------------------------------------------------------------- #
def test_exactly_one_translation_row_per_final_sentence_and_no_marker_survives(
    tmp_path, build_corpus_tables
):
    out = _built(tmp_path, build_corpus_tables)
    final_ids = [r["sentence_id"] for r in iter_parquet(out["sentences"])]
    langs = {r["sentence_id"]: r["lang"] for r in iter_parquet(out["sentences"])}
    rows = list(iter_parquet(out["translations"]))

    assert [r["sentence_id"] for r in rows] == final_ids
    assert len(rows) == len(final_ids)
    for r in rows:
        if langs[r["sentence_id"]] != "en":
            assert MARKER not in r["text"], r["sentence_id"]
        assert r["variant"] == "omt"


def test_english_identity_text_is_byte_identical_including_a_natural_marker(
    tmp_path, build_corpus_tables
):
    out = _built(tmp_path, build_corpus_tables)
    texts = {r["document_id"]: r["text"] for r in iter_parquet(out["documents"])}
    spans = {r["sentence_id"]: r for r in iter_parquet(out["sentences"])}
    seen_marker = False
    for r in iter_parquet(out["translations"]):
        s = spans[r["sentence_id"]]
        if s["lang"] != "en":
            continue
        source = texts[s["document_id"]][s["start"] : s["end"]]
        assert r["text"] == source, r["sentence_id"]
        seen_marker = seen_marker or MARKER in source
    assert seen_marker, "the fixture no longer covers a natural § in English"


def test_every_passage_member_resolves_and_no_final_sentence_is_orphaned(
    tmp_path, build_corpus_tables
):
    out = _built(tmp_path, build_corpus_tables)
    final = {r["sentence_id"]: r for r in iter_parquet(out["sentences"])}
    covered: set[str] = set()
    for p in iter_parquet(out["passages"]):
        members = list(p["sentence_ids"])
        assert members, p["passage_id"]
        assert all(m in final for m in members), p["passage_id"]
        assert p["token_count"] == sum(final[m]["token_count"] for m in members)
        covered.update(members)
    assert covered == set(final)


def test_remap_is_a_total_function_from_the_spine_onto_the_final_inventory(
    tmp_path, build_corpus_tables
):
    """Every original sentence id appears exactly once, and every target is a real final id.

    Renumbering is otherwise irreversible. After this stage the old ids appear in no table,
    so a remap that lost a row would leave an id from an earlier experiment permanently
    unresolvable, and say nothing about it.
    """
    out = _built(tmp_path, build_corpus_tables)
    remap = list(iter_parquet(out["remap"]))
    spine_ids = [r["sentence_id"] for r in iter_parquet(out["spine_sentences"])]
    final_ids = {r["sentence_id"] for r in iter_parquet(out["sentences"])}

    assert [r["sentence_id"] for r in remap] == spine_ids  # total, in order, no duplicates
    assert all(r["final_sentence_id"] in final_ids for r in remap)
    # A fused unit's constituents share one final id; an unfused sentence keeps its own.
    fused_targets = {r["final_sentence_id"] for r in remap if r["fused"]}
    assert fused_targets, "no fused rows: the stub never dropped a marker"
    for target in fused_targets:
        assert sum(1 for r in remap if r["final_sentence_id"] == target) > 1


# --------------------------------------------------------------------------- #
# The shipped storage policy, end to end over the real merged tables.
#
# `store_identity_translations: false` makes `translations/<variant>.parquet` a subsequence
# of the inventory. Everything downstream has to walk the two together rather than read the
# same row index, and that contract is checked here against real Parquet rather than
# hand-built rows.
# --------------------------------------------------------------------------- #
def test_no_english_row_is_stored_and_the_inventory_is_unchanged(
    tmp_path, build_corpus_tables
):
    stored = _run_build(tmp_path, build_corpus_tables, n_shards=3, name="stored")
    omitted = _run_build(
        tmp_path, build_corpus_tables, n_shards=3, name="omitted", store_identity=False
    )
    # The inventory is byte-identical, because the knob changes what is stored, not what
    # exists. That equality is why the Opus-MT arm's raw output can be keyed by an
    # `inventory_hash` this knob does not move.
    for key in ("sentences", "remap", "passages"):
        assert [dict(r) for r in iter_parquet(stored[key])] == [
            dict(r) for r in iter_parquet(omitted[key])
        ], key

    langs = {r["sentence_id"]: r["lang"] for r in iter_parquet(omitted["sentences"])}
    assert "en" in set(langs.values()), "the fixture no longer covers English"
    rows = list(iter_parquet(omitted["translations"]))
    assert rows, "the non-English sentences still have their rows"
    assert not [r for r in rows if langs[r["sentence_id"]] == "en"]
    assert [r["sentence_id"] for r in rows] == [
        sid for sid, lang in langs.items() if lang != "en"
    ]
    # The table shrank by exactly the English rows.
    assert len(list(iter_parquet(stored["translations"]))) - len(rows) == sum(
        1 for lang in langs.values() if lang == "en"
    )


def test_the_passage_store_resolves_english_from_the_native_span(
    tmp_path, build_corpus_tables
):
    """Composed passages read back over the subsequence tables."""
    import shutil

    from ragtime.common.passage_store import iter_final_passages

    out = _run_build(
        tmp_path, build_corpus_tables, n_shards=3, name="compose", store_identity=False
    )
    # The final node's own corpus directory. The fixture writes `documents.parquet` outside
    # it, so the one table `iter_final_passages` resolves through `Layout` is put there.
    corpus_dir = out["sentences"].parent.parent.parent
    base = corpus_dir.parent.parent.parent.parent
    layout = Layout(
        run_dir=base, base=base, family="e2e", chunker_hash=corpus_dir.parent.name
    )
    assert layout.final_sentences_path(_RECON_HASH) == out["sentences"]
    shutil.copy(out["documents"], layout.documents_path())
    shutil.copy(
        out["documents"].with_name(out["documents"].name + "._SUCCESS"),
        layout.documents_path().with_name(layout.documents_path().name + "._SUCCESS"),
    )
    pack_hash = out["passages"].parent.name
    assert layout.final_passages_path(_RECON_HASH, pack_hash) == out["passages"]
    records = list(
        iter_final_passages(
            layout, _RECON_HASH, pack_hash=pack_hash, renderings=("original", "omt")
        )
    )
    assert records
    english = [r for r in records if r["lang"] == "en"]
    assert english, "the fixture no longer covers English"
    for r in english:
        assert r["omt"] == r["original"]
    for r in records:
        if r["lang"] != "en":
            assert r["omt"] != r["original"], r["passage_id"]


def test_len_max_co_walks_the_subsequence_over_real_parquet(tmp_path, build_corpus_tables):
    """The alignment that used to rely on row identity, done as a walk over both tables.

    ``align_subtable`` is handed the final sentence table as its parent rather than
    ``documents.parquet``, and the per-row checks inside ``rows_for_shard`` then fail loudly
    if a range is off by one.
    """
    from ragtime.preprocess import len_max as lm

    out = _run_build(
        tmp_path, build_corpus_tables, n_shards=3, name="lenmax", store_identity=False
    )
    sentences, translations = out["sentences"], out["translations"]
    total = sum(parquet_row_group_sizes(sentences))
    ranges, _ = spine.plan_shards(sentences, n_shards=3)
    aligned = rc.align_subtable(sentences, translations, [r.row_start for r in ranges])

    adapter = lm.LenMaxAdapter(recon_hash=_RECON_HASH, lm_hash="l" * 64, renderings=("original", "omt"))
    ctx = lm._LenMaxCtx(
        tokenizer=types.SimpleNamespace(count_batch=lambda ts: [len(str(t).split()) for t in ts]),
        sentences=sentences,
        translations={"omt": translations},
        renderings=("original", "omt"),
        content_budget=None,
    )
    rows = [
        row
        for r, tr_range in zip(ranges, aligned, strict=True)
        for row in adapter.rows_for_shard(
            ctx, lm.LenMaxShard(rows=r, translation_rows={"omt": tr_range})
        )
    ]
    final = list(iter_parquet(sentences))
    assert len(rows) == total == len(final)  # total over the inventory, English included
    assert [r["sentence_id"] for r in rows] == [s["sentence_id"] for s in final]
    for row, sentence in zip(rows, final, strict=True):
        if sentence["lang"] == "en":
            # No row to read, so all three lengths are the same stored integer.
            assert row["len_omt"] == row["len_original"] == sentence["token_count"]


# --------------------------------------------------------------------------- #
# Throughput knobs move no byte of an artefact.
# --------------------------------------------------------------------------- #
def test_shard_count_changes_no_byte_of_any_final_table(tmp_path, build_corpus_tables):
    few = _run_build(tmp_path, build_corpus_tables, n_shards=1, name="few")
    many = _run_build(tmp_path, build_corpus_tables, n_shards=9, name="many")
    for key in ("sentences", "passages", "translations", "remap"):
        a = [dict(r) for r in iter_parquet(few[key])]
        b = [dict(r) for r in iter_parquet(many[key])]
        assert a == b, key


def test_subtable_alignment_handles_documents_absent_from_a_part(
    tmp_path, build_corpus_tables
):
    """An English document has no merge map and no MT row, and a Spanish one has no identity
    row.

    That is why ``spine.align_documents``, which raises on a boundary document it cannot
    find, could not be reused for the three sub-tables. It is also the condition under which
    a naive alignment hands a worker another document's rows.
    """
    paths = build_corpus_tables(_corpus(), name="align")
    docs, sents = paths["documents"], paths["sentences"]
    ranges, boundary_docs = spine.plan_shards(sents, n_shards=4)
    aligned = spine.align_documents(docs, ranges, boundary_docs)
    ordinals = [r.doc_row_start for r in aligned]

    # A sub-table containing only the two English documents' rows.
    english_only = tmp_path / "english_only.parquet"
    from ragtime.common.io import write_parquet_stream
    from ragtime.common.schemas import sentence_arrow_schema

    rows = [r for r in iter_parquet(sents) if r["lang"] == "en"]
    write_parquet_stream(english_only, rows, schema=sentence_arrow_schema())

    got = rc.align_subtable(docs, english_only, ordinals)
    assert len(got) == len(aligned)
    # The ranges tile the sub-table exactly, in order, with empties where they belong.
    assert got[0][0] == 0
    assert got[-1][1] == len(rows)
    assert all(a <= b for a, b in got)
    assert [b for _, b in got[:-1]] == [a for a, _ in got[1:]]
    assert sum(b - a for a, b in got) == len(rows)
