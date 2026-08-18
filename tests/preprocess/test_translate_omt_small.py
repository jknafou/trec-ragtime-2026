"""``preprocess.translate_omt``: units, the ``§`` marker, and raw storage.

The contract this stage owes its consumers:

- exactly one row per ``sentence_id``, and every emitted id is a real sentence id, since a
  unit is stored under its first constituent's id rather than a synthesised key;
- a unit whose source already carries ``§`` declines the merge rather than escaping it;
- a marker that comes back in the wrong count changes nothing about what is stored: the
  raw string is kept and the decision is left to reconciliation;
- markers reach ``translations_raw`` intact, because this stage never scrubs; the one
  ``scrub_markers`` call belongs to reconciliation;
- nothing passage-scoped survives, including the ``_is_home`` predicate.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from ragtime.common.io import iter_parquet
from ragtime.common.schemas import translation_raw_arrow_schema
from ragtime.preprocess import merge as merge_mod
from ragtime.preprocess import spine
from ragtime.preprocess import translate_omt as tr
from ragtime.preprocess.merge_join import MARKER

pytestmark = pytest.mark.small

_MIN = 4
_CAP = 12.0


def _count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# A deterministic stand-in for serving.mt.MtClient: no model, no GPU and no network.
# --------------------------------------------------------------------------- #
class FakeMt:
    """Echoes its input, with a per-unit policy for how many markers come back.

    ``policy`` maps a unit id to the number of markers to emit; the default is as many as it
    was given, which is the healthy case. That is the axis ``split_translation`` keys on, so
    a test can force the marker-lost path without a model.
    """

    model = "fake-nllb"

    def __init__(self, policy: dict[str, int] | None = None) -> None:
        self.policy = policy or {}
        self.calls: list[dict] = []

    def assert_marker_atomic(self, marker: str, probe_lang: str = "spa_Latn") -> None:
        assert marker and probe_lang

    def tokenize(self, text: str, src_lang: str, **_kw) -> list[str]:
        assert src_lang
        return text.split()

    def translate_batch(self, items, **kw) -> dict[str, str]:
        self.calls.append(kw)
        out: dict[str, str] = {}
        for it in items:
            parts = [p.strip() for p in it.text.split(MARKER)]
            want = self.policy.get(it.sentence_id, len(parts) - 1)
            english = [f"EN({p})" for p in parts]
            if want == len(parts) - 1:
                out[it.sentence_id] = f" {MARKER} ".join(english)
            else:  # the model dropped or invented boundaries
                joined = " ".join(english)
                out[it.sentence_id] = joined if want == 0 else f" {MARKER} ".join([joined] * 1)
        return out


def _map_rows(paths, *, langs=("es", "ru", "zh")) -> list[dict]:
    """Build the merge map the way ``preprocess.merge`` does, in sentence order."""
    rows: list[dict] = []
    texts = {r["document_id"]: r["text"] for r in iter_parquet(paths["documents"])}
    for doc_id, doc_rows in spine.group_by_document(iter_parquet(paths["sentences"])):
        lang = doc_rows[0]["lang"]
        if lang not in langs:
            continue
        rows.extend(
            merge_mod.merge_document(
                merge_mod.document_sentences(doc_rows),
                texts[doc_id],
                doc_id,
                lang,
                _count,
                min_tokens=_MIN,
                cap=_CAP,
                cross_paragraph="singleton_only",
            )
        )
    return rows


def _write_map(paths, rows: list[dict]) -> pathlib.Path:
    from ragtime.common.io import write_parquet_stream
    from ragtime.common.schemas import merge_map_arrow_schema

    out = paths["sentences"].parent / "merge_map.parquet"
    write_parquet_stream(out, rows, schema=merge_map_arrow_schema())
    return out


def _translate_ctx(paths, map_path, mt) -> tr._TranslateCtx:
    return tr._TranslateCtx(
        mt=mt,
        documents=paths["documents"],
        merge_map=map_path,
        model_id="fake-nllb",
        model_config_hash="t" * 64,
        merge_hash="m" * 64,
    )


def _run_arm(adapter, ctx, table, documents, *, n_shards=1) -> list[dict]:
    ranges, boundary_docs = spine.plan_shards(table, n_shards=n_shards)
    aligned = spine.align_documents(documents, ranges, boundary_docs)
    rows: list[dict] = []
    for r in aligned:
        # One queue per arm. The two arms shard different tables and their shard names are
        # both `rows_<start>_<end>`, so a shared `out/` directory would collide.
        shard = table.parent / f"wq_{adapter.part}" / "running" / r.name
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps(r.payload()), encoding="utf-8")
        out = adapter.work(ctx, shard)
        assert adapter.validate(out)
        rows.extend(iter_parquet(out))
    return rows


def _adapter(cls, cfg_blocks: dict | None = None):
    """Build an adapter without config loading; the hashes are injected directly."""
    cfg = types.SimpleNamespace(
        run_id="e2e-omt",
        blocks={
            "chunker": {"config": {}},
            "translation": {"config": {}},
            "merge": {},
            "execution": cfg_blocks or {},
        },
    )
    return cls(cfg)


# --------------------------------------------------------------------------- #
# Unit construction off the merge map.
# --------------------------------------------------------------------------- #
def test_one_row_per_sentence_id_and_every_id_is_a_real_spine_id(
    build_corpus_tables,
) -> None:
    paths = build_corpus_tables(
        [
            {
                "document_id": "spa-docs/0000001",
                "lang": "es",
                "paragraphs": [["Uno", "dos tres cuatro cinco"], ["seis siete ocho nueve"]],
            },
            {
                "document_id": "eng-docs/0000002",
                "lang": "en",
                "paragraphs": [["Headline", "the body of this english sentence"]],
            },
        ]
    )
    map_path = _write_map(paths, _map_rows(paths))
    spine_ids = {r["sentence_id"] for r in iter_parquet(paths["sentences"])}

    mt_rows = _run_arm(
        _adapter(tr.OmtTranslateAdapter),
        _translate_ctx(paths, map_path, FakeMt()),
        map_path,
        paths["documents"],
    )
    id_rows = _run_arm(
        _adapter(tr.OmtIdentityAdapter),
        tr._IdentityCtx(
            documents=paths["documents"],
            sentences=paths["sentences"],
            model_config_hash="t" * 64,
            merge_hash="m" * 64,
        ),
        paths["sentences"],
        paths["documents"],
    )

    ids = [r["sentence_id"] for r in mt_rows + id_rows]
    assert len(ids) == len(set(ids)), "a sentence_id must key at most one row"
    assert set(ids) <= spine_ids, "every stored id is a real sentence id, never synthesized"
    assert set(mt_rows[0]) == set(translation_raw_arrow_schema().names)
    # English is never merged, so the English arm carries no merge provenance.
    assert all(r["merge_unit_id"] is None for r in id_rows)
    assert {r["document_id"] for r in id_rows} == {"eng-docs/0000002"}


def test_unit_declines_when_the_source_already_carries_the_marker(
    build_corpus_tables,
) -> None:
    """A ``§`` already in the source makes the returned marker count ambiguous."""
    paths = build_corpus_tables(
        [
            {
                "document_id": "spa-docs/0000001",
                "lang": "es",
                "paragraphs": [[f"Art {MARKER} 3", "el cuerpo de la frase larga"]],
            }
        ]
    )
    rows = _map_rows(paths)
    assert {int(r["merge_constituent_count"]) for r in rows} == {2}, "the rule DID merge them"

    text = next(iter_parquet(paths["documents"]))["text"]
    stats = types.SimpleNamespace(emitted=[])
    stats.emit = lambda mid, *a, **k: stats.emitted.append(mid)
    units = tr.build_units(rows, text, stats=stats)

    assert [u.unit_id for u in units] == [
        "spa-docs/0000001#s0",
        "spa-docs/0000001#s1",
    ], "the merged run broke into lone sentences"
    assert all(not u.merged for u in units)
    assert "translate.merge_declined_marker_in_source" in stats.emitted


def test_a_marker_returned_in_the_wrong_count_still_stores_the_raw_string(
    build_corpus_tables,
) -> None:
    """Whether to fuse or split is reconciliation's decision. This stage records what came
    back, verbatim."""
    paths = build_corpus_tables(
        [
            {
                "document_id": "spa-docs/0000001",
                "lang": "es",
                "paragraphs": [["Uno", "dos tres cuatro cinco"]],
            }
        ]
    )
    map_path = _write_map(paths, _map_rows(paths))
    lost = FakeMt(policy={"spa-docs/0000001#s0": 0})  # the model returned no boundary
    rows = _run_arm(
        _adapter(tr.OmtTranslateAdapter),
        _translate_ctx(paths, map_path, lost),
        map_path,
        paths["documents"],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["sentence_id"] == "spa-docs/0000001#s0"  # the unit keeps its first id
    assert row["merge_unit_id"] == "spa-docs/0000001#s0"
    assert int(row["merge_constituent_count"]) == 2  # reconciliation can still try to split it
    assert MARKER not in row["text_raw"]  # nothing to scrub; the model dropped it
    assert row["text_raw"] == "EN(Uno) EN(dos tres cuatro cinco)"


def test_markers_survive_into_the_artifact_verbatim(build_corpus_tables) -> None:
    """``translations_raw`` is raw by contract: the stored bytes are the model's output.

    Scrubbing at translation time would destroy the only evidence of where the boundaries
    were, so a later change of fusion policy would cost a re-translate rather than a cheap
    second pass. Reconciliation owns the one scrub.
    """
    paths = build_corpus_tables(
        [
            {
                "document_id": "spa-docs/0000001",
                "lang": "es",
                "paragraphs": [["Uno", "dos tres cuatro cinco"]],
            }
        ]
    )
    map_path = _write_map(paths, _map_rows(paths))
    mt = FakeMt()
    rows = _run_arm(
        _adapter(tr.OmtTranslateAdapter),
        _translate_ctx(paths, map_path, mt),
        map_path,
        paths["documents"],
    )
    expected = f"EN(Uno) {MARKER} EN(dos tres cuatro cinco)"
    assert rows[0]["text_raw"] == expected, "stored text must be the model output, verbatim"
    assert MARKER in rows[0]["text_raw"]
    # The marker-free view reconciliation will produce is one call away.
    from ragtime.preprocess.merge_join import scrub_markers, split_translation

    assert split_translation(rows[0]["text_raw"], 2) == [
        "EN(Uno)",
        "EN(dos tres cuatro cinco)",
    ]
    assert MARKER not in scrub_markers(rows[0]["text_raw"])


def test_no_is_home_or_passage_concept_survives() -> None:
    """The home-occurrence predicate is deleted rather than bypassed, checked in the source."""
    root = pathlib.Path(tr.__file__).parent
    for name in ("translate_omt.py", "merge.py", "merge_join.py", "spine.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "_is_home" not in source, name


def test_no_passage_id_participates_in_any_M05f_key_or_schema() -> None:
    """No passage exists at this stage, so nothing here keys a row, a column or a read on one."""
    from ragtime.common.schemas import merge_map_arrow_schema, translation_raw_arrow_schema

    assert "passage_id" not in merge_map_arrow_schema().names
    assert "passage_id" not in translation_raw_arrow_schema().names
    assert "passage_id" not in tr._MAP_COLUMNS
    assert "passage_id" not in tr._SENTENCE_COLUMNS
    assert "passage_id" not in merge_mod._SENTENCE_COLUMNS


# --------------------------------------------------------------------------- #
# Hashed decoding knobs and artefact keying.
# --------------------------------------------------------------------------- #
def test_no_repeat_ngram_size_is_passed_from_config_not_a_client_default(
    build_corpus_tables,
) -> None:
    """The knob changes the output text, so it is hashed config and an explicit argument."""
    paths = build_corpus_tables(
        [
            {
                "document_id": "spa-docs/0000001",
                "lang": "es",
                "paragraphs": [["uno dos tres cuatro cinco"]],
            }
        ]
    )
    map_path = _write_map(paths, _map_rows(paths))
    mt = FakeMt()
    ctx = _translate_ctx(paths, map_path, mt)
    ctx.no_repeat_ngram_size = 7
    _run_arm(_adapter(tr.OmtTranslateAdapter), ctx, map_path, paths["documents"])
    assert mt.calls and all(c["no_repeat_ngram_size"] == 7 for c in mt.calls)


def test_artifact_and_queue_are_keyed_by_BOTH_semantic_hashes() -> None:
    """A new merge map does not resolve to the previous map's translations."""
    from ragtime.common import Layout

    layout = Layout(run_dir="runs", base="runs", family="e2e", chunker_hash="c" * 64)
    a = layout.translations_raw_path("omt", "t" * 64, "m" * 64)
    b = layout.translations_raw_path("omt", "t" * 64, "n" * 64)
    assert a != b
    assert "tttttttttttt-mmmmmmmmmmmm" in str(a)


def test_the_low_tier_raw_output_lands_OUTSIDE_the_final_node(tmp_path, monkeypatch) -> None:
    """``translations_raw_opus`` is keyed the same way as its NLLB counterpart.

    It sits under ``corpus_dir/translations_raw/omt_opus/<opus12>-<inv12>/``, outside
    ``final/<recon12>/``, keyed by the Opus recipe and by the inventory it translated, which the
    node declares in its own manifest. Same directory, same two-hash scheme and same guarantee as
    ``omt``. Inside the final node, any edit to the ``reconcile`` block would orphan GPU-hours of
    translation by path, even an edit to the storage policy that changed not a single span.
    """
    from ragtime.common import Layout
    from ragtime.config import all_hashes

    inv_hash = "i" * 64
    node = tmp_path / "final" / inv_hash[:12]
    node.mkdir(parents=True)
    (node / "manifest.json").write_text(
        json.dumps({"reconcile_hash": "r" * 64, "inventory_hash": inv_hash}) + "\n",
        encoding="utf-8",
    )
    assert tr.inventory_hash_of_node(node) == inv_hash

    monkeypatch.setenv("RAGTIME_INVENTORY_DIR", str(node))
    cfg = types.SimpleNamespace(
        run_id="e2e-omt",
        blocks={
            "chunker": {"config": {}},
            "translation": {"config": {}},
            "merge": {},
            "execution": {},
        },
    )
    adapter = tr.OmtIdentityAdapter(cfg, base="runs")
    out = adapter.out_path(cfg)
    assert str(node) not in str(out), "the raw table must not live inside the final node"
    assert "translations_raw" in str(out) and "/final/" not in str(out)
    assert out.parent.name.endswith(f"-{inv_hash[:12]}")
    assert out.parent.parent.name == tr.OPUS_VARIANT != tr.VARIANT
    # Same scheme and same parent directory as the NLLB arm's raw output; only the
    # rendering differs.
    nllb = Layout(
        run_dir="runs", base="runs", family="e2e", chunker_hash=all_hashes(cfg)["chunker"]
    ).translations_raw_path(tr.VARIANT, "t" * 64, "m" * 64)
    assert out.parent.parent.parent == nllb.parent.parent.parent


def test_a_node_that_does_not_declare_its_inventory_hash_raises(tmp_path, monkeypatch) -> None:
    """Guessing the key of a 3.8 GB artefact is not a fallback worth having.

    The directory name is ``recon12``, which hashes more than this arm's output depends on.
    Using it would be right by coincidence today and quietly wrong after any edit to the
    storage policy. An older node therefore fails loudly and says what to do about it.
    """
    node = tmp_path / "final" / "f8f20fe2cf17"
    node.mkdir(parents=True)
    (node / "manifest.json").write_text(
        json.dumps({"reconcile_hash": "r" * 64}) + "\n", encoding="utf-8"
    )
    with pytest.raises(KeyError, match="predates the inventory/node key split"):
        tr.inventory_hash_of_node(node)
    with pytest.raises(FileNotFoundError, match="that key is unknowable"):
        tr.inventory_hash_of_node(tmp_path / "nope")


def test_buckets_are_reproducible_under_a_shuffled_input() -> None:
    """Bucket composition is a function of the items alone, not of arrival order."""
    import random

    from ragtime.serving.batching import Tier

    items = [
        tr.MtInput(sentence_id=f"d#s{i}", src_lang="spa_Latn", text="x", tokens=["t"] * (i % 7))
        for i in range(40)
    ]
    tier = Tier(token_budget=10, max_items=4)
    ordered = [[it.sentence_id for it in b] for b in tr.buckets(items, tier)]
    shuffled = list(items)
    random.Random(3).shuffle(shuffled)
    assert [[it.sentence_id for it in b] for b in tr.buckets(shuffled, tier)] == ordered


def test_flores_code_resolves_zh_script_on_the_translated_text() -> None:
    assert tr.flores_code_for("es", "hola") == "spa_Latn"
    assert tr.flores_code_for("zh", "国说这个") == "zho_Hans"
    assert tr.flores_code_for("zh", "國說這個") == "zho_Hant"
    assert tr.flores_code_for("zh", "12345") == "zho_Hans"  # no script evidence: Hans
