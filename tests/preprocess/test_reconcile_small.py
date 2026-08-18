"""The reconciliation core: fuse-vs-split, renumbering, re-packing, and the scrub boundary.

These are the pure per-document functions: no Parquet, no work queue and no model, so each
invariant has a test that can fail for one reason only. The table rows are hand-built dicts
in the shape of the pinned schemas.

The properties under test:

- a surviving marker splits back, so constituents keep their spans and get their own English;
- a lost marker fuses the unit into one sentence spanning ``(first.start, last.end)``, with
  the token count taken from the merge map rather than re-counted, and every constituent
  remapped onto it;
- final ids are dense ``0..n-1`` per document and every span slices its own text;
- there is exactly one English row per final sentence and it is marker-free, except English
  identity, which is byte-identical to its source span including any ``§`` already there;
- ``remap`` is total over the input inventory;
- raw rows that do not partition the document raise rather than guess;
- the stage loads no tokenizer, and its signature has no place for one.
"""

from __future__ import annotations

import pytest

from ragtime.preprocess import reconcile as rc
from ragtime.preprocess.merge_join import MARKER

pytestmark = pytest.mark.small

_DOC = "spa-docs/0000001"


def _sent(idx: int, start: int, end: int, *, para: int = 0, tokens: int = 3) -> dict:
    return {
        "sentence_id": f"{_DOC}#s{idx}",
        "document_id": _DOC,
        "sentence_index": idx,
        "lang": "es",
        "start": start,
        "end": end,
        "paragraph_index": para,
        "token_count": tokens,
    }


def _raw(first_idx: int, text_raw: str, n: int = 1, *, doc: str = _DOC) -> dict:
    return {
        "sentence_id": f"{doc}#s{first_idx}",
        "document_id": doc,
        "text_raw": text_raw,
        "source_lang": "spa_Latn",
        "merge_constituent_count": n,
    }


def _map(first_idx: int, n: int, unit_tokens: int) -> list[dict]:
    unit = f"{_DOC}#s{first_idx}"
    return [
        {
            "sentence_id": f"{_DOC}#s{first_idx + k}",
            "document_id": _DOC,
            "merge_unit_id": unit,
            "merge_constituent_count": n,
            "merge_unit_token_count": unit_tokens,
        }
        for k in range(n)
    ]


# One document: "aaa bbb ccc", three sentences, sentences 0+1 are a merge unit.
_TEXT = "aaa bbb ccc"
_SENTS = [_sent(0, 0, 3), _sent(1, 4, 7), _sent(2, 8, 11)]
_MAP2 = _map(0, 2, 9) + [
    {
        "sentence_id": f"{_DOC}#s2",
        "document_id": _DOC,
        "merge_unit_id": f"{_DOC}#s2",
        "merge_constituent_count": 1,
        "merge_unit_token_count": 3,
    }
]


def _reconcile(raw_rows, *, sents=None, text=_TEXT, map_rows=None, lang="es"):
    return rc.reconcile_document(
        _DOC,
        lang,
        text,
        sents if sents is not None else _SENTS,
        raw_rows,
        _MAP2 if map_rows is None else map_rows,
    )


def _remap_of(finals):
    return tuple(
        {
            "sentence_id": old,
            "final_sentence_id": f.sentence_id,
            "document_id": f.document_id,
            "fused": f.fused,
        }
        for f in finals
        for old in f.members
    )


def _build(raws, *, sents=None, text=_TEXT, map_rows=None, lang="es", **kwargs):
    return rc.build_document(
        _DOC,
        lang,
        text,
        sents if sents is not None else _SENTS,
        raws,
        _MAP2 if map_rows is None else map_rows,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# The marker survives, so the unit splits back.
# --------------------------------------------------------------------------- #
def test_surviving_marker_splits_back_and_keeps_every_constituent():
    finals = _reconcile([_raw(0, f"AAA {MARKER} BBB", 2), _raw(2, "CCC")])

    assert [f.sentence_id for f in finals] == [f"{_DOC}#s{k}" for k in range(3)]
    assert [f.sentence_index for f in finals] == [0, 1, 2]
    # Each constituent keeps its own span, so the inventory is unchanged for this unit.
    assert [(f.start, f.end) for f in finals] == [(0, 3), (4, 7), (8, 11)]
    assert [f.text_en for f in finals] == ["AAA", "BBB", "CCC"]
    assert not any(f.fused for f in finals)
    # The stored per-sentence token counts, not the unit's.
    assert [f.token_count for f in finals] == [3, 3, 3]


def test_split_back_is_declinable_by_config_so_a_policy_change_costs_no_gpu_hour():
    """``split_back: false`` fuses everything, which is why the raw table is stored raw."""
    finals = _reconcile([_raw(0, f"AAA {MARKER} BBB", 2), _raw(2, "CCC")])
    fused = rc.reconcile_document(
        _DOC, "es", _TEXT, _SENTS, [_raw(0, f"AAA {MARKER} BBB", 2), _raw(2, "CCC")], _MAP2,
        split_back=False,
    )
    assert len(finals) == 3
    assert len(fused) == 2
    assert fused[0].fused and (fused[0].start, fused[0].end) == (0, 7)


# --------------------------------------------------------------------------- #
# The marker is lost or miscounted, so the unit fuses.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text_raw", "why"),
    [
        ("AAA BBB", "the model never emitted the marker"),
        (f"A {MARKER} B {MARKER} C", "too many markers to attribute segments"),
        (f"{MARKER} BBB", "a boundary came back with an empty segment"),
    ],
)
def test_lost_or_miscounted_marker_fuses_the_unit_into_one_sentence(text_raw, why):
    finals = _reconcile([_raw(0, text_raw, 2), _raw(2, "CCC")])

    assert len(finals) == 2, why
    fused = finals[0]
    assert fused.fused
    # The union of two adjacent spans is itself a contiguous span of the document.
    assert (fused.start, fused.end) == (0, 7)
    assert _TEXT[fused.start : fused.end] == "aaa bbb"
    # The token count comes from the map's `merge_unit_token_count`, not from a re-count.
    assert fused.token_count == 9
    assert fused.members == (f"{_DOC}#s0", f"{_DOC}#s1")
    # Renumbering is dense, so the sentence behind it becomes #s1 rather than #s2.
    assert [f.sentence_id for f in finals] == [f"{_DOC}#s0", f"{_DOC}#s1"]
    assert finals[1].members == (f"{_DOC}#s2",)


def test_fusing_without_a_matching_merge_map_unit_raises_rather_than_re_tokenizing():
    """The fused token count exists only in the map, and inventing one is not an option."""
    with pytest.raises(rc.ReconciliationError, match="merge-map unit"):
        _reconcile([_raw(0, "AAA BBB", 2), _raw(2, "CCC")], map_rows=[])


def test_a_non_adjacent_fused_unit_is_refused():
    """A gap between constituents would fold untranslated text into the sentence."""
    sents = [_sent(0, 0, 3), _sent(1, 8, 11), _sent(2, 4, 7)]
    with pytest.raises(rc.ReconciliationError, match="not adjacent"):
        _reconcile([_raw(0, "AAA BBB", 2), _raw(2, "CCC")], sents=sents)


# --------------------------------------------------------------------------- #
# The scrub boundary, and the English case that looks like an exemption and is not.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text_raw", [f"AAA {MARKER} BBB", f"AAA{MARKER}", f"{MARKER}{MARKER} X", "AAA BBB"]
)
def test_no_marker_ever_reaches_a_stored_non_english_translation(text_raw):
    """Every non-English path scrubs, whether or not the unit was split."""
    finals = _reconcile([_raw(0, text_raw, 2), _raw(2, "CCC")])
    for f in finals:
        assert MARKER not in f.text_en


_EN_TEXT = "See § 1983 rules. Next one."
_EN_SENTS = [
    {**_sent(0, 0, 17), "lang": "en"},
    {**_sent(1, 18, 27), "lang": "en"},
]


def test_english_identity_keeps_a_natural_marker_byte_identically():
    """About 1,170 English rows in the corpus carry a real ``§``, mostly US legal citations,
    and scrubbing would corrupt them.

    This runs the whole per-document path, which ends in the verifier, so the byte-identity
    is asserted by the shipped check and not only by this test.
    """
    raws = [_raw(0, _EN_TEXT[0:17]), _raw(1, _EN_TEXT[18:27])]
    doc = _build(raws, sents=_EN_SENTS, text=_EN_TEXT, map_rows=[], lang="en")
    assert doc.sentences[0].text_en == "See § 1983 rules."
    assert MARKER in doc.sentences[0].text_en


def test_english_text_that_is_not_its_source_span_is_caught():
    """The identity arm copies a span, so anything else means the tables disagree."""
    raws = [_raw(0, "SOMETHING ELSE"), _raw(1, _EN_TEXT[18:27])]
    with pytest.raises(rc.ReconciliationError, match="byte-identical"):
        _build(raws, sents=_EN_SENTS, text=_EN_TEXT, map_rows=[], lang="en")


# --------------------------------------------------------------------------- #
# The raw rows have to partition the document: no gap, no overlap, no reordering.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raws", "match"),
    [
        ([_raw(0, "AAA"), _raw(2, "CCC")], "starts at sentence 2"),  # gap at #s1
        ([_raw(0, "AAA", 2), _raw(1, "BBB"), _raw(2, "CCC")], "starts at sentence 1"),
        ([_raw(2, "CCC"), _raw(0, "AAA", 2)], "starts at sentence 2"),  # wrong order
        ([_raw(0, "AAA", 2)], "cover 2 of 3"),  # truncated
        ([_raw(0, "AAA", 9)], "over-reaches"),
    ],
)
def test_raw_rows_that_do_not_partition_the_document_raise(raws, match):
    with pytest.raises(rc.ReconciliationError, match=match):
        _reconcile(raws)


def test_a_translation_row_for_a_foreign_sentence_raises():
    with pytest.raises(rc.ReconciliationError, match="not a sentence of this document"):
        _reconcile([_raw(0, "AAA", 2), _raw(0, "CCC", doc="eng-docs/9")])


# --------------------------------------------------------------------------- #
# Remap totality and the stored projections, through the whole per-document path.
#
# There is no packing here. Grouping sentences into passages belongs to
# `preprocess.packing` under its own `pack12` key, and its tests live in
# `test_packing_small.py`. This stage produces no passage at all, which the tokenizer-free
# `build_document` signature makes structural.
# --------------------------------------------------------------------------- #
def test_remap_is_a_total_function_over_the_input_inventory():
    doc = _build([_raw(0, "AAA BBB", 2), _raw(2, "CCC")])
    assert [r["sentence_id"] for r in doc.remap] == [f"{_DOC}#s{k}" for k in range(3)]
    # Both constituents of the fused unit land on the same final id.
    assert doc.remap[0]["final_sentence_id"] == doc.remap[1]["final_sentence_id"]
    assert doc.remap[0]["fused"] and doc.remap[1]["fused"]
    assert doc.remap[2]["final_sentence_id"] == f"{_DOC}#s1"
    assert not doc.remap[2]["fused"]


def test_exactly_one_translation_row_per_final_sentence_unconditionally():
    for raws in (
        [_raw(0, f"AAA {MARKER} BBB", 2), _raw(2, "CCC")],  # split
        [_raw(0, "AAA BBB", 2), _raw(2, "CCC")],  # fused
        [_raw(0, "AAA"), _raw(1, "BBB"), _raw(2, "CCC")],  # unmerged
    ):
        doc = _build(raws, map_rows=_MAP2 if raws[0]["merge_constituent_count"] > 1 else [])
        rows = list(doc.translation_rows("omt"))
        assert [r["sentence_id"] for r in rows] == [s.sentence_id for s in doc.sentences]
        assert len(rows) == len(doc.sentences)
        assert all(r["variant"] == "omt" and r["text"] for r in rows)


def test_the_final_sentence_rows_are_dense_and_slice_their_own_text():
    doc = _build([_raw(0, "AAA BBB", 2), _raw(2, "CCC")])
    rows = list(doc.sentence_rows())
    assert [r["sentence_index"] for r in rows] == list(range(len(rows)))
    assert [r["sentence_id"] for r in rows] == [f"{_DOC}#s{k}" for k in range(len(rows))]
    assert [_TEXT[r["start"] : r["end"]] for r in rows] == ["aaa bbb", "ccc"]


def test_verify_rejects_a_partial_remap():
    doc = _build([_raw(0, "AAA BBB", 2), _raw(2, "CCC")])
    partial = rc.ReconciledDocument(
        doc.document_id, doc.lang, doc.sentences, doc.remap[:1]
    )
    with pytest.raises(rc.ReconciliationError, match="must be total"):
        rc.verify_document(partial, _TEXT, 3)


# --------------------------------------------------------------------------- #
# The storage policy: English is verified here and stored nowhere.
#
# `store_identity_translations: false` is the shipped value. The code default is True so a
# config that does not mention the key hashes and behaves as it always did, and the tests
# above deliberately run on that default.
# --------------------------------------------------------------------------- #
def test_english_emits_no_translation_row_when_identity_is_not_stored():
    """English has no translation, so the table carries no row for it."""
    raws = [_raw(0, _EN_TEXT[0:17]), _raw(1, _EN_TEXT[18:27])]
    doc = _build(
        raws, sents=_EN_SENTS, text=_EN_TEXT, map_rows=[], lang="en", store_identity=False
    )
    assert len(doc.sentences) == 2  # the inventory is untouched
    assert list(doc.translation_rows("omt", store_identity=False)) == []  # the table is not
    assert doc.n_translation_rows(store_identity=False) == 0
    # The identity text is still carried, and still verified on the way past.
    assert doc.sentences[0].text_en == _EN_TEXT[0:17]


def test_a_non_english_document_still_emits_a_row_per_sentence():
    doc = _build([_raw(0, "AAA BBB", 2), _raw(2, "CCC")], store_identity=False)
    rows = list(doc.translation_rows("omt", store_identity=False))
    assert [r["sentence_id"] for r in rows] == [s.sentence_id for s in doc.sentences]
    assert doc.n_translation_rows(store_identity=False) == len(doc.sentences)


def test_the_english_span_check_survives_the_policy_change():
    """The check is on the raw row rather than the stored one, so it still fires.

    The identity arm's output is compared byte for byte against
    ``documents.text[start:end]`` on its way past, whether or not the row is written down.
    """
    raws = [_raw(0, "SOMETHING ELSE"), _raw(1, _EN_TEXT[18:27])]
    with pytest.raises(rc.ReconciliationError, match="byte-identical"):
        _build(
            raws, sents=_EN_SENTS, text=_EN_TEXT, map_rows=[], lang="en", store_identity=False
        )


def test_validates_row_count_agrees_with_the_projection_under_both_policies():
    """``validate`` compares the file against a projected count, so the two have to agree.

    ``n_translation_rows`` is what ``validate`` compares against and ``translation_rows`` is
    what ``work`` wrote. Drift between them would make the row-count check test the wrong
    number and let a truncated write through, so both are exercised for both languages and
    both storage policies.
    """
    english = _build(
        [_raw(0, _EN_TEXT[0:17]), _raw(1, _EN_TEXT[18:27])],
        sents=_EN_SENTS,
        text=_EN_TEXT,
        map_rows=[],
        lang="en",
        store_identity=False,
    )
    spanish = _build([_raw(0, "AAA BBB", 2), _raw(2, "CCC")], store_identity=False)
    for doc in (english, spanish):
        for store in (True, False):
            emitted = list(doc.translation_rows("omt", store_identity=store))
            assert doc.n_translation_rows(store_identity=store) == len(emitted)
    assert english.n_translation_rows(store_identity=False) == 0
    assert english.n_translation_rows(store_identity=True) == 2
    assert spanish.n_translation_rows(store_identity=False) == len(spanish.sentences)


def test_the_storage_knob_moves_recon12_but_not_the_inventory_hash():
    """``recon12`` names the node, whose contents changed. ``inventory_hash`` names which
    sentences exist, which did not, and it is what the Opus-MT arm's raw output is keyed by
    outside ``final/``."""
    import types

    def cfg(**reconcile):
        block = {"split_back": True, "min_segment_chars": 1}
        block.update(reconcile)
        return types.SimpleNamespace(
            run_id="e2e-omt",
            blocks={
                "chunker": {"config": {"token_budget": 512}},
                "merge": {"min_sentence_tokens": 16},
                "translation": {"config": {"omt_model": "facebook/nllb-200-3.3B"}},
                "reconcile": block,
            },
        )

    before, after = cfg(), cfg(store_identity_translations=False)
    assert rc.reconcile_hash(before) != rc.reconcile_hash(after)
    assert rc.inventory_hash(before) == rc.inventory_hash(after)
    # A fusion knob moves both, because the inventory really did change.
    fused = cfg(split_back=False)
    assert rc.inventory_hash(before) != rc.inventory_hash(fused)
    # For an unedited config the two hashes coincide, which happens exactly when the block
    # holds nothing but fusion keys. That is why the shipped corpus relocates rather than
    # re-translates.
    assert rc.inventory_hash(before) == rc.reconcile_hash(before)


def test_this_stage_emits_no_passage_part_at_all():
    """One producer per artefact: three parts here, and no packing surface at all."""
    assert rc.PARTS == ("sentences", "translations", "remap")
    assert not hasattr(rc, "pack_document")
    assert "passages" not in rc.ReconciledDocument.__slots__
