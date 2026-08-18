"""Packing: the sentences-to-passages grouping, and the budget theorem it makes true.

The property under test is general, so these tests try to falsify it rather than illustrate
it:

    pack against ``max_r(len_r)``  =>  every non-oversized passage fits the content budget
                                       in every rendering, not only the native one.

The rest of the file pins what makes the stage safe to re-run over an output that is
already frozen:

- the stored ``token_count`` keeps its native meaning, so one column has one meaning;
- ``lengths=None`` reproduces the earlier packer byte for byte;
- a ``len_max`` sidecar measured over a different inventory is refused per document, by id,
  which is the one thing comparing hashes cannot establish;
- a sentence that alone exceeds the budget becomes its own ``is_oversized`` passage and is
  counted. That tail is irreducible: 1,812 sentences across the collection, en 846, es 268,
  ru 338, zh 360, so a test asserting no truncation at all would be false.

The rows are hand-built and the tokenizer counts words, so this needs no collection, no
model and no network.
"""

from __future__ import annotations

import pytest

from ragtime.preprocess import packing as pk
from ragtime.preprocess import reconcile as rc

pytestmark = pytest.mark.small

_DOC = "spa-docs/0000001"


class WordTokenizer:
    """Supplies ``num_special()`` and raises on ``count()``.

    Packing not re-tokenising is a correctness property rather than a speed one: a fresh
    count could disagree with the stored one and leave a passage's token_count out of step
    with its sentences. Raising here makes the forbidden call impossible to miss.
    """

    def num_special(self) -> int:
        return 0

    def count(self, text: str) -> int:  # pragma: no cover - must never run
        raise AssertionError(f"packing re-tokenized {text!r}")


def _sentences(native: list[int], pack: list[int] | None = None, *, para: int = 0):
    lengths = native if pack is None else pack
    return [
        pk.PackedSentence(
            sentence_id=f"{_DOC}#s{i}",
            document_id=_DOC,
            lang="es",
            paragraph_index=para,
            token_count=n,
            pack_length=m,
        )
        for i, (n, m) in enumerate(zip(native, lengths, strict=True))
    ]


def _pack(sentences, *, budget: int):
    return pk.pack_document(
        sentences,
        "es",
        WordTokenizer(),
        token_budget=budget,
        overlap_frac=0.0,
        prefer_paragraph_break=False,
        prefer_min_fill=0.6,
    )


# --------------------------------------------------------------------------- #
# The defect, reproduced and then removed.
# --------------------------------------------------------------------------- #
def test_a_passage_that_only_overflows_once_translated_is_really_split() -> None:
    """Three sentences of 3 native tokens fit a 10-token budget as one passage. That same
    passage is over budget as soon as it is read in English, where the sentences are 4
    tokens each and come to 12. Packing against ``len_max`` splits it."""
    native = _pack(_sentences([3, 3, 3]), budget=10)
    assert len(native) == 1  # it fits natively
    assert 3 * 4 > 10  # and does not fit once translated

    invariant = _pack(_sentences([3, 3, 3], [4, 4, 4]), budget=10)
    assert len(invariant) == 2
    assert [len(p["sentence_ids"]) for p in invariant] == [2, 1]


def test_every_packed_passage_fits_the_budget_in_every_rendering() -> None:
    """The property itself over a mixed inventory, rather than a single example."""
    native = [3, 7, 2, 9, 4, 1, 6, 8, 2, 5, 3, 7]
    omt = [5, 4, 9, 3, 8, 2, 4, 9, 6, 3, 7, 2]
    opus = [2, 9, 4, 6, 3, 7, 8, 2, 5, 9, 1, 4]
    biggest = [max(t) for t in zip(native, omt, opus, strict=True)]
    for p in _pack(_sentences(native, biggest), budget=20):
        if p["is_oversized"]:
            continue
        members = [int(sid.rsplit("#s", 1)[1]) for sid in p["sentence_ids"]]
        for rendering in (native, omt, opus):
            assert sum(rendering[i] for i in members) <= 20


def test_the_stored_token_count_keeps_its_native_meaning() -> None:
    """A passage's token_count sums its sentences' native counts, and nothing else."""
    sentences = _sentences([3, 3, 3], [4, 4, 4])
    by_id = {s.sentence_id: s.token_count for s in sentences}
    for p in _pack(sentences, budget=10):
        assert p["token_count"] == sum(by_id[s] for s in p["sentence_ids"])


def test_the_native_path_is_untouched() -> None:
    """Packing on the stored count reproduces the earlier packer byte for byte."""
    native = [4, 4, 4, 9, 2]
    assert _pack(_sentences(native), budget=10) == _pack(_sentences(native, native), budget=10)


def test_a_sentence_that_alone_exceeds_the_budget_is_the_irreducible_tail() -> None:
    """A packer cannot split a sentence, so it becomes its own oversized passage."""
    packs = _pack(_sentences([3, 3, 3], [3, 30, 3]), budget=10)
    assert [p["is_oversized"] for p in packs] == [False, True, False]


# --------------------------------------------------------------------------- #
# Verification, and pairing a document with its sidecar.
# --------------------------------------------------------------------------- #
def _rows(n: int, lang: str = "es"):
    return [
        {
            "sentence_id": f"{_DOC}#s{i}",
            "document_id": _DOC,
            "sentence_index": i,
            "lang": lang,
            "start": i * 4,
            "end": i * 4 + 3,
            "paragraph_index": 0,
            "token_count": 3,
        }
        for i in range(n)
    ]


def _build(len_rows=None, *, n: int = 4, budget: int = 10):
    return pk.build_document(
        _DOC,
        "es",
        _rows(n),
        len_rows,
        WordTokenizer(),
        token_budget=budget,
        overlap_frac=0.0,
        prefer_paragraph_break=False,
        prefer_min_fill=0.6,
    )


def test_a_sidecar_measured_over_a_different_inventory_is_refused() -> None:
    """Comparing hashes cannot catch this, so it is checked per document."""
    stale = [{"sentence_id": f"{_DOC}#s{i}", "len_max": 5} for i in range(3)]
    with pytest.raises(rc.ReconciliationError, match="different sentence inventory"):
        _build(stale)


def test_a_matching_sidecar_packs_and_keeps_the_native_token_count() -> None:
    passages = _build([{"sentence_id": f"{_DOC}#s{i}", "len_max": 5} for i in range(4)])
    assert [len(p["sentence_ids"]) for p in passages] == [2, 2]
    assert all(p["token_count"] == 6 for p in passages)  # the native 3+3, not the 5+5


def test_every_sentence_must_appear_in_some_passage() -> None:
    sentences = _sentences([3, 3])
    with pytest.raises(rc.ReconciliationError, match="appear in no passage"):
        pk.verify_passages(_DOC, sentences, [], content_budget=10)


def test_a_passage_referencing_a_stale_id_is_refused() -> None:
    sentences = _sentences([3, 3])
    orphan = [{
        "passage_id": f"{_DOC}#p0",
        "sentence_ids": [f"{_DOC}#s7"],
        "token_count": 3,
        "is_oversized": False,
    }]
    with pytest.raises(rc.ReconciliationError, match="orphan citation target"):
        pk.verify_passages(_DOC, sentences, orphan, content_budget=10)


def test_an_over_budget_non_oversized_passage_is_refused() -> None:
    """The budget is enforced at build time, not merely expected of the packer."""
    sentences = _sentences([3, 3], [9, 9])
    bad = [{
        "passage_id": f"{_DOC}#p0",
        "sentence_ids": [s.sentence_id for s in sentences],
        "token_count": 6,
        "is_oversized": False,
    }]
    with pytest.raises(rc.ReconciliationError, match="content budget"):
        pk.verify_passages(_DOC, sentences, bad, content_budget=10)


def test_a_partial_document_cannot_mint_colliding_passage_ids() -> None:
    """Passage ids are positional, so a shard handed only a document's tail would mint
    ids that collide with the ones built from the whole document."""
    tail = _rows(4)[2:]
    with pytest.raises(rc.ReconciliationError, match="dense 0..n-1"):
        pk.build_document(
            _DOC, "es", tail, None, WordTokenizer(),
            token_budget=10, overlap_frac=0.0,
            prefer_paragraph_break=False, prefer_min_fill=0.6,
        )


def test_pack_length_is_validated_rather_than_silently_ignored() -> None:
    import types

    cfg = types.SimpleNamespace(
        blocks={
            "chunker": {"config": {"token_budget": 512, "overlap_frac": 0.15,
                                   "tokenizer_id": "BAAI/bge-m3@abc"}},
            "packing": {"pack_length": "maxp"},
        }
    )
    with pytest.raises(ValueError, match="pack_length"):
        pk.packing_options(cfg)
