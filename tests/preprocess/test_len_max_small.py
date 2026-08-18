"""``len_max``: the rendering-invariant sentence length and the packing it drives.

The mechanism exists for one property, and these tests try to falsify it rather than
illustrate it:

    pack against ``max_r(len_r)``  =>  every non-oversized passage fits the content budget
                                       in every rendering, not just the native one.

Two groups. The measurement is pure and does no IO: the maximum is over all renderings, the
native length is read rather than re-tokenised, and the translation tables are walked as
subsequences, so English has no row at all and its three lengths are one integer read three
times. A misaligned, short or over-long table raises rather than pair one sentence's length
with another's, or worse, take a non-English sentence's native length as its maximum.

Then the keys. A packing edit must not move ``recon12``, which keys the node holding both
renderings' translations, and must move ``pack12``. That asymmetry is what lets a re-pack
read a sidecar already paid for, with no re-measurement.

The packing side of this is in ``test_packing_small.py``. Everything here runs on
hand-built rows and a whitespace tokenizer, so there is no corpus, no model and no network.
"""

from __future__ import annotations

import types
from itertools import pairwise

import pytest

from ragtime.preprocess import len_max as lm
from ragtime.preprocess import packing as pk
from ragtime.preprocess import reconcile as rc

pytestmark = pytest.mark.small

_DOC = "spa-docs/0000001"
_TOKENIZER_ID = "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181"


class WordTokenizer:
    """A whitespace counter with the two methods this stage uses.

    ``count_batch`` is the only path ``len_max`` takes. Leaving ``count`` as a raise shows
    the stage never falls back to a per-text loop that would then have to be kept in step.
    """

    def count_batch(self, texts) -> list[int]:
        return [len(str(t).split()) for t in texts]

    def num_special(self) -> int:
        return 0

    def count(self, text: str) -> int:  # pragma: no cover - must never run in len_max
        raise AssertionError(f"len_max tokenized one text at a time: {text!r}")


def _sent(idx: int, *, lang: str = "es", tokens: int = 3, doc: str = _DOC) -> dict:
    return {
        "sentence_id": f"{doc}#s{idx}",
        "document_id": doc,
        "lang": lang,
        "token_count": tokens,
    }


def _tr(idx: int, text: str, *, doc: str = _DOC) -> dict:
    return {"sentence_id": f"{doc}#s{idx}", "text": text}


# --------------------------------------------------------------------------- #
# The measurement.
# --------------------------------------------------------------------------- #
def test_len_max_is_the_max_over_every_rendering() -> None:
    rows = lm.len_max_rows(
        [_sent(0, tokens=4)],
        {"omt": [_tr(0, "one two three four five six seven")], "omt_opus": [_tr(0, "short one")]},
        WordTokenizer(),
    )
    (row,) = rows
    assert row["len_original"] == 4  # the stored native count
    assert row["len_omt"] == 7
    assert row["len_omt_opus"] == 2
    assert row["len_max"] == 7


def test_the_native_length_is_read_from_the_column_and_never_re_tokenized() -> None:
    """``token_count`` is the only place a fused sentence's count exists.

    The fake's ``count`` raises, so a stage that re-derived the native length from text
    fails here rather than disagree quietly with ``sentences.parquet``, which is what
    ``verify_document`` checks passages against.
    """
    (row,) = lm.len_max_rows(
        [_sent(0, tokens=99)],
        {"omt": [_tr(0, "a b")], "omt_opus": [_tr(0, "a b")]},
        WordTokenizer(),
    )
    assert row["len_original"] == 99 and row["len_max"] == 99


def test_english_has_no_translation_row_and_its_three_lengths_are_one_integer() -> None:
    """English is skipped by the walk rather than compared against a copy of itself.

    Under ``store_identity_translations: false`` no English row exists to disagree with the
    span, so ``len_original == len_omt == len_omt_opus`` is the same ``token_count`` read
    three times, and a disagreement cannot be constructed. The sentences either side still
    consume their own rows, which is what shows the skip did not shift the walk.
    """
    rows = lm.len_max_rows(
        [_sent(0), _sent(1, lang="en", tokens=3), _sent(2)],
        {
            "omt": [_tr(0, "a b c d e"), _tr(2, "a")],
            "omt_opus": [_tr(0, "a"), _tr(2, "a b c d e f g")],
        },
        WordTokenizer(),
    )
    assert [r["len_max"] for r in rows] == [5, 3, 7]
    english = rows[1]
    assert english["len_original"] == english["len_omt"] == english["len_omt_opus"] == 3


def test_a_stored_english_identity_row_is_still_length_checked() -> None:
    """Under ``store_identity_translations: true`` nothing is lost.

    An identity row that is present is tokenised like any other and has to agree with the
    span. The newer policy makes the row absent; it does not make the check optional.
    """
    ok = lm.len_max_rows(
        [_sent(0, lang="en", tokens=3)],
        {"omt": [_tr(0, "a b c")], "omt_opus": [_tr(0, "a b c")]},
        WordTokenizer(),
    )
    assert ok[0]["len_max"] == 3
    with pytest.raises(lm.EnglishIdentityLengthError, match="identity pass-through"):
        lm.len_max_rows(
            [_sent(0, lang="en", tokens=3)],
            {"omt": [_tr(0, "a b c d")], "omt_opus": [_tr(0, "a b c")]},
            WordTokenizer(),
        )


def test_a_non_english_sentence_with_no_translation_row_raises() -> None:
    """A missing row is never read as "untranslated, so use the native length".

    That reading is how a truncated or misaligned table would restore the per-language
    overflow this stage exists to remove, with every downstream check still passing.
    """
    with pytest.raises(lm.LenMaxMismatchError, match="Only 'en' sentences are untranslated"):
        lm.len_max_rows(
            [_sent(0), _sent(1)],
            {"omt": [_tr(0, "a")], "omt_opus": [_tr(0, "a"), _tr(1, "b")]},
            WordTokenizer(),
        )


def test_a_misordered_translation_table_raises_rather_than_mis_pairing() -> None:
    with pytest.raises(lm.LenMaxMismatchError, match="has no row for"):
        lm.len_max_rows(
            [_sent(0), _sent(1)],
            {"omt": [_tr(1, "a"), _tr(0, "b")], "omt_opus": [_tr(0, "a"), _tr(1, "b")]},
            WordTokenizer(),
        )


def test_leftover_translation_rows_raise() -> None:
    """A table longer than its inventory is a mis-cut range, not a harmless surplus."""
    with pytest.raises(lm.LenMaxMismatchError, match="left over"):
        lm.len_max_rows(
            [_sent(0)],
            {"omt": [_tr(0, "a"), _tr(1, "b")], "omt_opus": [_tr(0, "a")]},
            WordTokenizer(),
        )


def test_the_pinned_schema_matches_the_emitted_record_exactly() -> None:
    """Column order is part of the contract; ``common.io``'s byte determinism rests on it."""
    (row,) = lm.len_max_rows(
        [_sent(0)], {"omt": [_tr(0, "a")], "omt_opus": [_tr(0, "a")]}, WordTokenizer()
    )
    assert [f.name for f in lm.len_max_arrow_schema()] == list(row)


def test_counters_name_the_tail_the_packer_cannot_remove() -> None:
    """The tail the packer cannot remove is counted per language, not special-cased away."""
    from ragtime.common import Statistics

    stats = Statistics()
    lm.len_max_rows(
        [_sent(0, tokens=1), _sent(1, tokens=1)],
        {
            "omt": [_tr(0, "a b c d e f"), _tr(1, "a")],
            "omt_opus": [_tr(0, "a"), _tr(1, "a")],
        },
        WordTokenizer(),
        content_budget=3,
        stats=stats,
    )
    assert stats.value(lm.STAT_ROWS, lang="es") == 2
    assert stats.value(lm.STAT_GREW, lang="es") == 1  # sentence 0 grew 1 -> 6
    assert stats.value(lm.STAT_OVER_BUDGET, lang="es") == 1  # and 6 is over the budget of 3
    assert stats.total(lm.STAT_ENGLISH_CHECKED) == 0  # nothing English in this batch


# --------------------------------------------------------------------------- #
# The keys.
# --------------------------------------------------------------------------- #
def _cfg(**packing) -> types.SimpleNamespace:
    """A config in the shipped shape: `reconcile` holds fusion only, `packing` its own block."""
    block = {"pack_length": "native", "pack_budget": 512}
    block.update(packing)
    return types.SimpleNamespace(
        run_id="e2e-original",
        blocks={
            "chunker": {
                "config": {
                    "token_budget": 512,
                    "overlap_frac": 0.0,
                    "tokenizer_id": _TOKENIZER_ID,
                    "prefer_paragraph_break": False,
                }
            },
            "merge": {"min_sentence_tokens": 16},
            "translation": {"config": {"omt_model": "facebook/nllb-200-3.3B"}},
            "reconcile": {"split_back": True, "min_segment_chars": 1},
            "packing": block,
            "execution": {},
        },
    )


def test_a_packing_edit_cannot_move_recon12() -> None:
    """``final/<recon12>/`` holds both renderings' translations: 106 GPU-hours of NLLB plus
    the Opus-MT pass.

    If a packing knob could move ``recon12``, a re-pack would land in a fresh node and
    orphan all of that by path, and the only way back would be a checksum-guarded copy. So
    every packing knob changes here, ``recon12`` stays put and ``pack12`` moves.
    """
    before = _cfg()
    after = _cfg(pack_length="len_max", pack_budget=384, len_max_tokenizer_id=_TOKENIZER_ID)
    assert rc.reconcile_hash(before) == rc.reconcile_hash(after)
    assert pk.packing_hash(before) != pk.packing_hash(after)


def test_a_fusion_knob_moves_recon12_because_it_changes_which_sentences_exist() -> None:
    other = _cfg()
    other.blocks["reconcile"] = {"split_back": False, "min_segment_chars": 1}
    assert rc.reconcile_hash(_cfg()) != rc.reconcile_hash(other)


def test_the_measurement_recipe_keys_the_sidecar_separately() -> None:
    """A re-measure gets a fresh sidecar. A new budget does not, since it measures nothing."""
    a = _cfg()
    b = _cfg(len_max_tokenizer_id="BAAI/bge-m3@0000000000000000000000000000000000000000")
    assert lm.len_max_hash(a) != lm.len_max_hash(b)
    assert lm.len_max_hash(a) == lm.len_max_hash(_cfg(pack_budget=384))
    assert rc.reconcile_hash(a) == rc.reconcile_hash(b)  # same sentences, re-measured


def test_packing_is_a_fairness_shared_block() -> None:
    """Passages are identical across renderings, so packing is never an `execution` knob."""
    from ragtime.config.schema import SHARED_BLOCKS

    assert "packing" in SHARED_BLOCKS


def test_the_tokenizer_defaults_to_the_chunkers_pinned_identity() -> None:
    assert lm.len_max_options(_cfg())["tokenizer_id"] == _TOKENIZER_ID


def test_the_sidecar_hangs_off_the_corpus_anchor_through_layout(tmp_path) -> None:
    from ragtime.common import Layout

    layout = Layout(run_dir=tmp_path, base=tmp_path, family="e2e", chunker_hash="c" * 64)
    path = layout.sentence_len_max_path("r" * 64, "m" * 64)
    assert path.parent.parent == layout.corpus_dir("e2e", "c" * 64) / "sentence_len_max"
    assert path.parent.name == f"{'r' * 12}-{'m' * 12}"
    assert path.name == "len_max.parquet"


def test_the_adapter_builds_the_sidecar_from_the_real_final_tables(tmp_path) -> None:
    """Real Parquet in and out: the final tables in, the sidecar out.

    It reads exactly what reconciliation publishes, ``final/<inv12>/sentences.parquet`` and
    ``translations/<variant>.parquet`` through ``Layout``, and writes one row per final
    sentence under the corpus anchor. A schema or column-name drift on either side fails
    here rather than 88.7 M rows later.
    """
    from ragtime.common import Layout
    from ragtime.common.io import iter_parquet, write_parquet_stream
    from ragtime.common.schemas import sentence_arrow_schema, translation_final_arrow_schema
    from ragtime.config import all_hashes
    from ragtime.orchestration.run_identity import run_family

    cfg = _cfg()
    layout = Layout(
        run_dir=tmp_path,
        base=tmp_path,
        family=run_family(cfg),
        chunker_hash=all_hashes(cfg)["chunker"],
    )
    inv = rc.reconcile_hash(cfg)
    sents = [
        {
            "sentence_id": f"{_DOC}#s{i}",
            "document_id": _DOC,
            "sentence_index": i,
            "lang": lang,
            "start": i * 4,
            "end": i * 4 + 3,
            "paragraph_index": 0,
            "token_count": n,
        }
        for i, (lang, n) in enumerate([("es", 2), ("es", 3), ("en", 4)])
    ]
    # The tables are subsequences: two Spanish rows each, no row for the English sentence.
    texts = {
        "omt": ["a b c d e", "a b"],
        "omt_opus": ["a b c", "a b c d e f"],
    }
    translated = [s for s in sents if s["lang"] != "en"]
    write_parquet_stream(layout.final_sentences_path(inv), sents, schema=sentence_arrow_schema())
    for variant, rows in texts.items():
        write_parquet_stream(
            layout.final_translations_path(inv, variant),
            [
                {
                    "sentence_id": s["sentence_id"],
                    "document_id": _DOC,
                    "variant": variant,
                    "text": t,
                    "source_lang": "spa_Latn",
                }
                for s, t in zip(translated, rows, strict=True)
            ],
            schema=translation_final_arrow_schema(),
        )

    out = lm.build(cfg, base=tmp_path)
    assert out == layout.sentence_len_max_path(inv, lm.len_max_hash(cfg))
    got = list(iter_parquet(out))
    # One row per final sentence, English included. The sidecar is total over the inventory
    # even though the tables it reads are not.
    assert [r["sentence_id"] for r in got] == [s["sentence_id"] for s in sents]
    assert [r["len_max"] for r in got] == [5, 6, 4]
    assert [r["len_original"] for r in got] == [2, 3, 4]
    assert got[2]["len_omt"] == got[2]["len_omt_opus"] == 4  # English, from the span
    # A second call sees the `_SUCCESS` and rewrites nothing.
    assert lm.build(cfg, base=tmp_path) == out


def test_the_adapter_shards_cover_the_final_table_exactly_once(tmp_path) -> None:
    from ragtime.common import Layout
    from ragtime.common.io import write_parquet_stream
    from ragtime.common.schemas import sentence_arrow_schema, translation_final_arrow_schema
    from ragtime.config import all_hashes
    from ragtime.orchestration.run_identity import run_family

    cfg = _cfg()
    cfg.blocks["execution"]["len_max_shards"] = 3
    layout = Layout(
        run_dir=tmp_path,
        base=tmp_path,
        family=run_family(cfg),
        chunker_hash=all_hashes(cfg)["chunker"],
    )
    rows = [
        {
            "sentence_id": f"doc{d}#s{i}",
            "document_id": f"doc{d}",
            "sentence_index": i,
            "lang": "es",
            "start": i,
            "end": i + 1,
            "paragraph_index": 0,
            "token_count": 1,
        }
        for d in range(6)
        for i in range(4)
    ]
    inv = rc.reconcile_hash(cfg)
    # doc2 is English, so the translation tables are shorter than the inventory and the
    # aligned ranges have to skip it. A same-length table would show nothing.
    for r in rows:
        if r["document_id"] == "doc2":
            r["lang"] = "en"
    write_parquet_stream(
        layout.final_sentences_path(inv), rows, schema=sentence_arrow_schema()
    )
    translated = [r for r in rows if r["lang"] != "en"]
    for variant in ("omt", "omt_opus"):
        write_parquet_stream(
            layout.final_translations_path(inv, variant),
            [
                {
                    "sentence_id": r["sentence_id"],
                    "document_id": r["document_id"],
                    "variant": variant,
                    "text": "x",
                    "source_lang": "spa_Latn",
                }
                for r in translated
            ],
            schema=translation_final_arrow_schema(),
        )
    adapter = lm.LenMaxAdapter.for_config(cfg, base=str(tmp_path))
    shards = list(adapter.shards(cfg))
    ranges = [s.payload for s in shards]
    assert ranges[0]["row_start"] == 0
    assert ranges[-1]["row_end"] == len(rows)
    for a, b in pairwise(ranges):
        assert a["row_end"] == b["row_start"]  # contiguous, no gap and no overlap
    # No shard boundary falls inside a document; each has four rows.
    assert all(r["row_start"] % 4 == 0 for r in ranges)
    # Each rendering's ranges tile its own table exactly once. That tiling differs from the
    # sentence one, because the English document contributes no translation rows.
    for variant in ("omt", "omt_opus"):
        tr = [lm.LenMaxShard.from_payload(p).translation_rows[variant] for p in ranges]
        assert tr[0][0] == 0
        assert tr[-1][1] == len(translated)
        for a, b in pairwise(tr):
            assert a[1] == b[0]
        assert sum(b - a for a, b in tr) == len(translated) < len(rows)
