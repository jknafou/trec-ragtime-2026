"""``preprocess.merge``: the translation-unit rule over the stored ``paragraph_index``.

The rule is unchanged; what moved is where its inputs come from. These tests pin the
properties that could regress quietly under the corpus data model:

- direction comes from the stored ``paragraph_index`` column, with no source file read;
- English produces zero rows;
- the output is a function of the document alone, not of input arrival order;
- the cap is never exceeded, and an unmergeable sentence is emitted as its own unit;
- ``merge_unit_token_count`` equals a direct re-count of the fused source text, so
  reconciliation can trust it without loading a tokenizer.
"""

from __future__ import annotations

import json
import random
import types

import pytest

from ragtime.common.io import iter_parquet
from ragtime.common.schemas import merge_map_arrow_schema
from ragtime.preprocess import merge as merge_mod
from ragtime.preprocess import spine

pytestmark = pytest.mark.small

# The rule's inputs, small and explicit. A whitespace tokenizer makes tokens into words.
_MIN = 4
_CAP = 12.0


def _count(text: str) -> int:
    return len(text.split())


def _doc(paragraphs: list[list[str]], *, lang: str = "es", doc_id: str = "spa-docs/0000001"):
    return {"document_id": doc_id, "lang": lang, "paragraphs": paragraphs}


def _sentences_of(paths, document_id: str) -> tuple[list, str]:
    rows = [r for r in iter_parquet(paths["sentences"]) if r["document_id"] == document_id]
    text = next(
        r["text"] for r in iter_parquet(paths["documents"]) if r["document_id"] == document_id
    )
    return merge_mod.document_sentences(rows), text


def _run(paths, document_id: str, *, lang: str = "es", cross: str = "singleton_only"):
    sents, text = _sentences_of(paths, document_id)
    return merge_mod.merge_document(
        sents,
        text,
        document_id,
        lang,
        _count,
        min_tokens=_MIN,
        cap=_CAP,
        cross_paragraph=cross,
    )


def _units(rows: list[dict]) -> list[list[str]]:
    """Rows to a list of units, each given as its constituent sentence ids."""
    out: list[list[str]] = []
    for r in rows:
        if int(r["merge_constituent_index"]) == 0:
            out.append([])
        out[-1].append(r["sentence_id"])
    return out


# --------------------------------------------------------------------------- #
# The paragraph-directional rule, off the stored column.
# --------------------------------------------------------------------------- #
def test_paragraph_opener_merges_next_and_non_opener_merges_previous(
    build_corpus_tables,
) -> None:
    # Paragraph 0 is [short opener][long body], so the opener merges forwards.
    # Paragraph 1 is [long body][short trailer], so the trailer merges backwards.
    paths = build_corpus_tables(
        [
            _doc(
                [
                    ["Titulo corto", "cuerpo de la primera frase larga"],
                    ["cuerpo de la segunda frase larga", "muy corta"],
                ]
            )
        ]
    )
    units = _units(_run(paths, "spa-docs/0000001"))
    assert units == [
        ["spa-docs/0000001#s0", "spa-docs/0000001#s1"],
        ["spa-docs/0000001#s2", "spa-docs/0000001#s3"],
    ]


def test_paragraph_boundary_blocks_a_backward_merge(build_corpus_tables) -> None:
    """A short sentence that opens a paragraph does not reach back across the boundary.

    With ``merge_cross_paragraph="never"`` the whole-paragraph exception is off, so a short
    opener whose own paragraph continues can only merge forwards. The boundary is
    structural, not a preference.
    """
    paths = build_corpus_tables(
        [_doc([["cuerpo de la primera frase larga"], ["Titulo", "y su cuerpo bastante largo"]])]
    )
    units = _units(_run(paths, "spa-docs/0000001", cross="never"))
    assert units == [
        ["spa-docs/0000001#s0"],
        ["spa-docs/0000001#s1", "spa-docs/0000001#s2"],
    ]


def test_singleton_paragraph_takes_next_then_falls_back_to_previous(
    build_corpus_tables,
) -> None:
    # A one-sentence paragraph followed by another paragraph merges with the next one.
    forward = build_corpus_tables(
        [_doc([["Titulo"], ["cuerpo de la frase siguiente"]])], name="fwd"
    )
    assert _units(_run(forward, "spa-docs/0000001")) == [
        ["spa-docs/0000001#s0", "spa-docs/0000001#s1"]
    ]
    # The same heading with nothing after it falls back to the previous paragraph.
    backward = build_corpus_tables(
        [_doc([["cuerpo de la frase anterior"], ["Titulo"]])], name="bwd"
    )
    assert _units(_run(backward, "spa-docs/0000001")) == [
        ["spa-docs/0000001#s0", "spa-docs/0000001#s1"]
    ]


def test_merge_reads_no_source_file(build_corpus_tables, monkeypatch) -> None:
    """The rule works off artefacts, so opening a raw corpus file is a defect.

    An earlier ``assign_paragraphs`` re-derived paragraph membership from the raw
    ``.jsonl.gz`` and dropped any document whose two streams disagreed, which produced zero
    rows for that document and said nothing. ``paragraph_index`` is a stored column now, so
    the stage has no reason to reach for a source file, and ``gzip.open`` raises here if it
    does.
    """
    import gzip

    paths = build_corpus_tables([_doc([["Titulo corto", "cuerpo de la frase larga"]])])

    def _boom(*_a, **_k):  # pragma: no cover - only runs if the stage regresses
        raise AssertionError("merge opened a gzip source file")

    monkeypatch.setattr(gzip, "open", _boom)
    assert len(_run(paths, "spa-docs/0000001")) == 2


# --------------------------------------------------------------------------- #
# Totality, English exclusion, determinism, cap.
# --------------------------------------------------------------------------- #
def test_every_sentence_gets_exactly_one_row(build_corpus_tables) -> None:
    paths = build_corpus_tables(
        [_doc([["Uno", "dos tres cuatro cinco seis"], ["siete ocho nueve diez", "once"]])]
    )
    rows = _run(paths, "spa-docs/0000001")
    ids = [r["sentence_id"] for r in rows]
    assert sorted(ids) == sorted(f"spa-docs/0000001#s{j}" for j in range(4))
    assert len(set(ids)) == len(ids)
    assert merge_mod.merge_map_violation(rows) is None


def test_english_produces_zero_rows(build_corpus_tables, make_cfg, monkeypatch) -> None:
    """English passes through unchanged, so merging it would cost granularity for nothing."""
    paths = build_corpus_tables(
        [
            {
                "document_id": "eng-docs/0000001",
                "lang": "en",
                "paragraphs": [["Headline", "the body of the first sentence"]],
            },
            _doc([["Titulo", "el cuerpo de la primera frase"]]),
        ]
    )
    rows = _adapter_rows(paths, make_cfg, monkeypatch)
    assert rows, "the non-English document must still produce rows"
    assert {r["document_id"] for r in rows} == {"spa-docs/0000001"}


def test_output_is_independent_of_document_arrival_order(build_corpus_tables) -> None:
    """The rule is per document and deterministic, so shuffling the corpus changes nothing."""
    docs = [
        _doc([["Uno", "dos tres cuatro cinco"]], doc_id=f"spa-docs/{i:07d}") for i in range(6)
    ]
    ordered = build_corpus_tables(docs, name="ordered")
    shuffled_docs = list(docs)
    random.Random(7).shuffle(shuffled_docs)
    shuffled = build_corpus_tables(shuffled_docs, name="shuffled")

    def per_doc(paths):
        return {
            d["document_id"]: _units(_run(paths, d["document_id"]))
            for d in docs
        }

    assert per_doc(ordered) == per_doc(shuffled)


def test_cap_is_never_exceeded_and_unmergeable_is_emitted_honestly(
    build_corpus_tables,
) -> None:
    """A short sentence with no room under the cap stays its own unit."""
    # 12 tokens: fusing the one-token trailer would make 13, one past the cap of 12.
    long_body = " ".join(f"w{i}" for i in range(12))
    paths = build_corpus_tables([_doc([[long_body, "eh"]])])
    rows = _run(paths, "spa-docs/0000001")
    assert _units(rows) == [["spa-docs/0000001#s0"], ["spa-docs/0000001#s1"]]
    assert all(int(r["merge_unit_token_count"]) <= _CAP for r in rows)

    # The boundary itself is inclusive: a fused unit of exactly the cap is allowed.
    at_cap = build_corpus_tables(
        [_doc([[" ".join(f"w{i}" for i in range(11)), "eh"]])], name="atcap"
    )
    assert _units(_run(at_cap, "spa-docs/0000001")) == [
        ["spa-docs/0000001#s0", "spa-docs/0000001#s1"]
    ]


def test_merge_unit_token_count_matches_a_direct_recount(build_corpus_tables) -> None:
    """Reconciliation reads this number instead of loading a tokenizer.

    A unit's source text is the document slice spanning its constituents, which are adjacent
    by construction, and that slice is also what is handed to MT.
    """
    paths = build_corpus_tables(
        [_doc([["Uno", "dos tres cuatro", "cinco seis siete ocho"], ["nueve", "diez once doce"]])]
    )
    rows = _run(paths, "spa-docs/0000001")
    text = next(
        r["text"]
        for r in iter_parquet(paths["documents"])
        if r["document_id"] == "spa-docs/0000001"
    )
    by_unit: dict[str, list[dict]] = {}
    for r in rows:
        by_unit.setdefault(r["merge_unit_id"], []).append(r)
    for members in by_unit.values():
        fused = text[int(members[0]["start"]) : int(members[-1]["end"])]
        assert int(members[0]["merge_unit_token_count"]) == _count(fused)


# --------------------------------------------------------------------------- #
# The adapter: artefact shape and the structural check made at write time.
# --------------------------------------------------------------------------- #
def _adapter_rows(paths, make_cfg, monkeypatch, *, n_shards: int = 2) -> list[dict]:
    """Run ``MergeAdapter.work`` over every shard and return the concatenated rows."""
    del make_cfg, monkeypatch  # the context is built here, so nothing has to be brought up
    adapter = merge_mod.MergeAdapter(merge_hash="deadbeefcafe0000")
    ctx = merge_mod._MergeCtx(
        tokenizer=types.SimpleNamespace(count=_count),
        documents=paths["documents"],
        sentences=paths["sentences"],
        min_tokens=_MIN,
        cap=_CAP,
        cross_paragraph="singleton_only",
        languages=("zh", "ru", "es"),
        enabled=True,
    )

    ranges, boundary_docs = spine.plan_shards(paths["sentences"], n_shards=n_shards)
    aligned = spine.align_documents(paths["documents"], ranges, boundary_docs)
    rows: list[dict] = []
    for r in aligned:
        base = paths["sentences"].parent / "wq" / r.name
        shard = base.parent / "running" / r.name
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps(r.payload()), encoding="utf-8")
        out = adapter.work(ctx, shard)
        assert adapter.validate(out)
        rows.extend(iter_parquet(out))
    return rows


def test_adapter_writes_the_pinned_schema_and_passes_its_own_gate(
    build_corpus_tables, make_cfg, monkeypatch
) -> None:
    paths = build_corpus_tables(
        [
            _doc([["Uno", "dos tres cuatro cinco"]], doc_id="spa-docs/0000001"),
            _doc([["Seis", "siete ocho nueve diez"]], doc_id="spa-docs/0000002"),
            {
                "document_id": "zho-docs/0000003",
                "lang": "zh",
                "paragraphs": [["ab cd", "ef gh ij kl mn"]],
            },
        ]
    )
    rows = _adapter_rows(paths, make_cfg, monkeypatch)
    assert {r["document_id"] for r in rows} == {
        "spa-docs/0000001",
        "spa-docs/0000002",
        "zho-docs/0000003",
    }
    assert set(rows[0]) == set(merge_map_arrow_schema().names)
    assert merge_mod.merge_map_violation(rows) is None


def test_validate_rejects_a_structurally_broken_map(tmp_path) -> None:
    """The conditions that raise ``CorruptMergeMapError`` downstream are caught on write."""
    from ragtime.common.io import write_parquet_stream

    adapter = merge_mod.MergeAdapter(merge_hash="deadbeefcafe0000")
    row = {
        "sentence_id": "spa-docs/0000001#s0",
        "document_id": "spa-docs/0000001",
        "sentence_index": 0,
        "lang": "es",
        "start": 0,
        "end": 3,
        "merge_unit_id": "spa-docs/0000001#s0",
        "merge_constituent_index": 0,
        "merge_constituent_count": 2,  # declares 2 members, only 1 present
        "merge_unit_token_count": 1,
    }
    out = tmp_path / "broken.parquet"
    write_parquet_stream(out, [row], schema=merge_map_arrow_schema())
    assert adapter.validate(out) is False
