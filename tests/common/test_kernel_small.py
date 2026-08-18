"""Small-fixture smoke tests for the common kernel."""

from __future__ import annotations

import pytest

from ragtime.common import (
    Answer,
    LmdbPassageStore,
    Nugget,
    Statistics,
    T1ReportRow,
    T1Response,
    T2RunRow,
    accumulate_answers,
    doc_id_of,
    maxp_collapse,
    nfc,
    nfkc_len,
    nugget_id,
    passage_id,
    rrf,
    sentence_id,
    write_jsonl,
)
from ragtime.common.io import is_done

pytestmark = pytest.mark.small


# --------------------------------------------------------------------------- #
# normalize: nfc symmetry vs nfkc collapse
# --------------------------------------------------------------------------- #
def test_nfc_symmetry_precomposed_vs_decomposed(cafe_forms):
    precomposed, decomposed, _ = cafe_forms
    # The two forms are not byte-equal...
    assert precomposed != decomposed
    # ...but NFC makes them identical, which is what the symmetric span commit rests on.
    assert nfc(precomposed) == nfc(decomposed)


def test_nfkc_collapses_compatibility_but_nfc_does_not(cafe_forms):
    import unicodedata

    _, _, fullwidth = cafe_forms  # "２ g" with a full-width 2
    # nfc is canonical-only: the full-width digit survives unchanged.
    assert nfc(fullwidth) == fullwidth
    assert "２" in nfc(fullwidth)
    # nfkc folds the full-width '２' to ASCII '2', a genuinely different result.
    assert unicodedata.normalize("NFKC", fullwidth) != nfc(fullwidth)
    assert nfkc_len(fullwidth) == len("2 g")


def test_nfc_and_nfkc_len_are_distinct_functions():
    ligature = "ﬁ"  # U+FB01 LATIN SMALL LIGATURE FI
    assert nfc(ligature) == ligature  # canonical: unchanged
    assert nfkc_len(ligature) == 2  # compatibility: folds to "fi"


# --------------------------------------------------------------------------- #
# ids: round-trips on real doc-ids (no '#' in them)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("doc", ["eng-docs/0007421", "0e4ae416-b3d1_1793347", "2061"])
def test_doc_id_round_trip(doc):
    """Every suffix form strips back to the same doc-id.

    ``doc_id_of`` splits on the first ``#`` only and is a pure function of the stored
    string, so both the current document-scoped sentence id (``#s{j}``) and the legacy
    passage-scoped form (``#p{k}#s{j}``) resolve to the same document.
    """
    assert doc_id_of(passage_id(doc, 2)) == doc
    assert doc_id_of(sentence_id(doc, 5)) == doc  # document-scoped sentence id
    assert doc_id_of(f"{passage_id(doc, 2)}#s5") == doc  # the legacy nested form too
    assert doc_id_of(doc) == doc  # a bare doc-id is its own doc-id


def test_id_shapes():
    assert passage_id("eng-docs/0007421", 0) == "eng-docs/0007421#p0"
    # A sentence is a span of its document, so its id is a sibling of the passage
    # id rather than nested under one.
    assert sentence_id("eng-docs/0007421", 3) == "eng-docs/0007421#s3"
    assert nugget_id("2061", 3) == "2061#n3"


# --------------------------------------------------------------------------- #
# fusion: rrf ordering and maxp collapse
# --------------------------------------------------------------------------- #
def test_rrf_ordering_and_default_k():
    # id "a" appears rank-1 in both pools -> highest fused score.
    pools = [["a", "b", "c"], ["a", "c", "d"]]
    fused = rrf(pools)  # k=60 default, not hardcoded per call
    order = [pid for pid, _ in fused]
    assert order[0] == "a"
    # a: 1/61 + 1/61 ; c: 1/63 + 1/62 ; both beat b (1/62) and d (1/63).
    assert order.index("c") < order.index("d")
    # scores are non-increasing
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_tie_break_is_lexicographic_by_id():
    # Two ids each rank-1 in one pool -> equal score -> (-score, id) tie-break.
    fused = rrf([["z"], ["a"]])
    assert [pid for pid, _ in fused] == ["a", "z"]


def test_rrf_accepts_id_score_pairs_and_records():
    from ragtime.common import Retrieved

    pools = [[("a", 9.9), ("b", 1.0)], [Retrieved("a", 5.0), Retrieved("c", 2.0)]]
    order = [pid for pid, _ in rrf(pools)]
    assert order[0] == "a"


def test_maxp_collapse_keeps_best_score_per_doc():
    ranked = [
        ("eng-docs/1#p0", 5.0),
        ("eng-docs/1#p3", 9.0),  # same doc, higher score wins
        ("eng-docs/2#p1", 7.0),
    ]
    collapsed = maxp_collapse(ranked, doc_id_of)
    assert collapsed == [("eng-docs/1", 9.0), ("eng-docs/2", 7.0)]


# --------------------------------------------------------------------------- #
# stats: counter-only, unknown slice keys rejected
# --------------------------------------------------------------------------- #
def test_stats_accumulates_on_canonical_slices():
    stats = Statistics()
    stats.emit("nuggets_new", variant="original", seed=1, round=1)
    stats.emit("nuggets_new", value=2.0, variant="original", seed=1, round=1)
    assert stats.value("nuggets_new", variant="original", seed=1, round=1) == 3.0
    assert stats.total("nuggets_new") == 3.0


def test_stats_rejects_unknown_slice_key():
    stats = Statistics()
    with pytest.raises(ValueError, match="unknown slice key"):
        stats.emit("bad", topic="2061")  # 'topic' is not canonical


def test_stats_is_instance_isolated():
    a, b = Statistics(), Statistics()
    a.emit("x", lang="en")
    assert a.total("x") == 1.0
    assert b.total("x") == 0.0


# --------------------------------------------------------------------------- #
# passage store: render totality over the three renderings
# --------------------------------------------------------------------------- #
def test_render_is_total_over_three_renderings(passage_store, passage_records):
    for rec in passage_records:
        pid = rec["passage_id"]
        for lang in ("original", "omt", "omt_opus"):
            text = passage_store.render(pid, lang)
            assert isinstance(text, str) and text  # never raises, always non-empty
        # original is always the native text
        assert passage_store.render(pid, "original") == rec["original"]


def test_render_english_passthrough_falls_back_to_original(passage_store):
    # The English passage supplied no omt/omt_opus -> identity passthrough.
    pid = "eng-docs/0007421#p0"
    original = passage_store.render(pid, "original")
    assert passage_store.render(pid, "omt_opus") == original
    assert passage_store.render(pid, "omt") == original


def test_render_rejects_unknown_rendering(passage_store):
    with pytest.raises(ValueError, match="unknown rendering"):
        passage_store.render("eng-docs/0007421#p0", "sockeye")


def test_passage_returns_schema_and_doc_id(passage_store):
    p = passage_store.passage("spa-docs/0451820#p2")
    assert p.lang == "es"
    assert p.document_id == "spa-docs/0451820"
    assert doc_id_of(p.passage_id) == p.document_id


def test_in_memory_store_round_trips_sentence_char_spans(passage_store, passage_records):
    """The sentence-grain spans survive ``from_records`` and slice the exact sentence.

    Sentence-level MT and the translated renderings the passage store composes consume
    them: one ``(start, end)`` per ``sentence_id``, in the same order, as offsets into
    the passage's own text.
    """
    for rec in passage_records:
        p = passage_store.passage(rec["passage_id"])
        assert p.sentence_char_spans == tuple(
            tuple(s) for s in rec["sentence_char_spans"]
        )
        assert len(p.sentence_char_spans) == len(p.sentence_ids)
        # slicing the passage text with a span yields that sentence, verbatim.
        assert " ".join(p.text[s:e] for s, e in p.sentence_char_spans) == p.text
    multi = passage_store.passage("eng-docs/0007421#p1")
    assert [multi.text[s:e] for s, e in multi.sentence_char_spans] == [
        "Poles engage the upper body.",
        "The technique started in Finland.",
    ]


def test_render_carries_precomposed_accent(passage_store, cafe_forms):
    precomposed, decomposed, _ = cafe_forms
    native = passage_store.render("spa-docs/0451820#p2", "original")
    # native text holds the precomposed form; NFC-equal to the decomposed spelling.
    assert precomposed in native
    assert nfc(native) == nfc(native.replace(precomposed, decomposed))


# --------------------------------------------------------------------------- #
# LmdbPassageStore: build, then reopen read-only with lock=False
# --------------------------------------------------------------------------- #
def test_lmdb_passage_store_round_trip(tmp_path, passage_records):
    db = tmp_path / "passages.lmdb"
    LmdbPassageStore.build(db, passage_records)
    # Reopen read-only (readonly=True, lock=False under the hood).
    store = LmdbPassageStore(db)
    try:
        for rec in passage_records:
            pid = rec["passage_id"]
            assert store.render(pid, "original") == rec["original"]
            assert store.render(pid, "omt")  # totality
            p = store.passage(pid)
            assert p.document_id == rec["document_id"]
            assert p.lang == rec["lang"]
            # sentence-grain spans survive the JSON blob as int TUPLES, one per
            # sentence_id, and still slice the exact sentence out of the passage text.
            assert p.sentence_char_spans == tuple(
                tuple(s) for s in rec["sentence_char_spans"]
            )
            assert len(p.sentence_char_spans) == len(p.sentence_ids)
            assert " ".join(p.text[s:e] for s, e in p.sentence_char_spans) == p.text
        with pytest.raises(KeyError):
            store.passage("nope#p0")
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# io: atomic write, _SUCCESS marker, skip-if-done resume no-op
# --------------------------------------------------------------------------- #
def test_write_jsonl_atomic_success_and_resume_noop(tmp_path):
    path = tmp_path / "sub" / "out.jsonl"
    write_jsonl(path, [{"a": 1}, {"b": "café"}])
    assert path.exists()
    assert is_done(path)  # _SUCCESS marker written after payload

    first_bytes = path.read_bytes()
    first_mtime = path.stat().st_mtime_ns
    # native non-ASCII preserved, byte-stable separators
    assert "café" in first_bytes.decode("utf-8")
    assert b", " not in first_bytes  # separators=(",",":") -> no spaces

    # Resume: a re-launch over a completed artifact is a no-op (stale content and
    # mtime are left untouched, even with different records).
    write_jsonl(path, [{"different": True}])
    assert path.read_bytes() == first_bytes
    assert path.stat().st_mtime_ns == first_mtime


def test_write_jsonl_rejects_non_finite_float(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        write_jsonl(tmp_path / "bad.jsonl", [{"score": float("nan")}])


def test_parquet_round_trip_and_resume_noop(tmp_path):
    from ragtime.common import read_parquet, write_parquet
    from ragtime.common.io import is_done

    path = tmp_path / "p" / "rows.parquet"
    rows = [
        {"passage_id": "eng-docs/1#p0", "score": 1.5},
        {"passage_id": "spa-docs/2#p1", "score": 0.25},
    ]
    write_parquet(path, rows)
    assert is_done(path)
    assert read_parquet(path) == rows
    # skip-if-done: a re-write over a completed artifact is a no-op.
    mtime = path.stat().st_mtime_ns
    write_parquet(path, [{"passage_id": "x#p0", "score": 9.0}])
    assert path.stat().st_mtime_ns == mtime
    assert read_parquet(path) == rows


def test_parquet_rejects_non_finite(tmp_path):
    from ragtime.common import write_parquet

    with pytest.raises(ValueError, match="non-finite"):
        write_parquet(tmp_path / "bad.parquet", [{"score": float("inf")}])


# --------------------------------------------------------------------------- #
# schemas: frozen accumulation and track-row serialisation
# --------------------------------------------------------------------------- #
def test_nugget_accumulation_is_frozen_safe():
    n = Nugget(nugget_id="2061#n0", question="What reactions?")
    assert n.status == "unanswered"
    grown = accumulate_answers(n, (Answer(answer="headache", sentence="Headache."),))
    # the original is untouched (frozen); a new nugget carries the answer and status.
    assert n.answers == ()
    assert grown.answers[0].answer == "headache"
    assert grown.status == "answered"


def test_t2_run_row_fixed_non_scientific_score():
    line = T2RunRow(request_id="2061", doc_id="eng-docs/1", rank=1, score=1e-7, run_id="r").to_line()
    parts = line.split()
    # Nine decimals: `.6f` manufactures ties the spec forbids, because RRF gaps fall
    # below 1e-6 from rank 963, inside the shipped `top_docs: 1000`.
    assert parts == ["2061", "Q0", "eng-docs/1", "1", "0.000000100", "r"]
    assert "e" not in parts[4]  # never scientific notation


def test_t2_run_row_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        T2RunRow("2061", "d", 1, float("inf"), "r").to_line()


def test_t1_report_row_serializes_byte_stable():
    row = T1ReportRow(
        topic_id="2061",
        team_id="TEAM",
        run_id="e2e-original",
        run_desc="d",
        responses=(T1Response(text="Headache.", citations={"eng-docs/1": 0.91}),),
        references=("eng-docs/1",),
    )
    line = row.to_line()
    assert '"topic_id":"2061"' in line  # compact separators
    assert '"citations":{"eng-docs/1":0.91}' in line
    d = row.to_dict()
    assert d["references"] == ["eng-docs/1"]


def test_t1_report_row_rejects_non_finite_citation():
    row = T1ReportRow(
        topic_id="2061",
        team_id="T",
        run_id="r",
        run_desc="d",
        responses=(T1Response(text="x", citations={"d": float("nan")}),),
    )
    with pytest.raises(ValueError, match="non-finite"):
        row.to_line()


# --------------------------------------------------------------------------- #
# layout: submission paths verbatim from config.outputs
# --------------------------------------------------------------------------- #
def test_layout_submission_is_verbatim_from_outputs(tmp_path):
    from ragtime.common import Layout

    outputs = [
        {"task": 1, "track": "report_generation", "path": "submissions/report_generation/run_4.jsonl"},
        {"task": 2, "track": "retrieval", "path": "submissions/retrieval/run_4.txt"},
    ]
    layout = Layout(tmp_path / "run", outputs)
    assert str(layout.submission("report_generation")) == "submissions/report_generation/run_4.jsonl"
    assert str(layout.submission("retrieval")) == "submissions/retrieval/run_4.txt"
    with pytest.raises(KeyError):
        layout.submission("nuggetization")


def test_layout_pipeline_paths(tmp_path):
    from ragtime.common import Layout

    layout = Layout(tmp_path / "run")
    assert layout.decompose_round(1).name == "round_1.jsonl"
    assert layout.rag_loop("2061#n0").name == "2061#n0.jsonl"
    assert layout.citation_scores().name == "citation_scores"
    assert layout.manifest().name == "manifest.json"


# --------------------------------------------------------------------------- #
# topics: the small slice parses from the concatenated single line
# --------------------------------------------------------------------------- #
def test_small_topics_parse(small_topics):
    assert len(small_topics) == 3
    t = small_topics[0]
    assert t.topic_id == "2000"
    assert t.collection_id == "ragtime/2/all"
    assert t.limit == 5000
    assert t.problem_statement and t.background


def test_small_topics_text_is_verbatim_no_nfc(small_topics_path, small_topics):
    # The loader takes topic text byte-exact: field values must equal an
    # independent raw JSON parse of the same bytes (no NFC, no mutation).
    import json

    text = small_topics_path.read_text(encoding="utf-8")
    dec, i, raw_objs = json.JSONDecoder(), 0, []
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        obj, i = dec.raw_decode(text, i)
        raw_objs.append(obj)
    for topic, obj in zip(small_topics, raw_objs, strict=True):
        assert topic.problem_statement == obj["problem_statement"]
        assert topic.background == obj["background"]


# --------------------------------------------------------------------------- #
# topics: `title` is a required request component (fast twin of the full-data tier)
# --------------------------------------------------------------------------- #
def test_small_topics_carry_title(small_topics):
    """The fixture is a slice of the canonical file, so every topic has a title."""
    assert [t.title for t in small_topics] == [
        "Anti-vaping Legislation",
        "Advantages of Nordic walking",
        '"Lying Flat" movement in China',
    ]


def test_small_topics_fixture_is_still_the_concatenated_single_line(small_topics_path):
    """The canonical file is strict JSONL now; the TREC quirk survives only here."""
    assert small_topics_path.read_bytes().count(b"\n") == 0
    assert len(small_topics_path.read_text(encoding="utf-8").splitlines()) == 1


def test_parse_topics_refuses_a_topic_with_no_title():
    """A missing `title` is a hard error, never a silent ``""``.

    Fast twin of ``test_superseded_topics_file_is_refused_not_defaulted`` in the full
    tier, which asserts the same thing against the real superseded file. `title`
    reaches the decompose prompt, so a default would truncate the request invisibly.
    """
    from ragtime.common.topics import CANONICAL_TOPICS_REL, TopicsError, parse_topics

    complete = (
        '{"topic_id": "9000", "collection_id": "ragtime/2/all", "title": "T",'
        ' "problem_statement": "p", "background": "b", "limit": 5000}'
    )
    assert parse_topics(complete)[0].title == "T"

    titleless = complete.replace('"title": "T", ', "")
    with pytest.raises(TopicsError) as exc:
        parse_topics(titleless)
    # The message names the offending topic and the file to use instead; a bare
    # KeyError('title') points at nothing.
    assert "9000" in str(exc.value)
    assert CANONICAL_TOPICS_REL in str(exc.value)


def test_parse_topics_keeps_an_empty_title_it_was_actually_given():
    """A present-but-empty title is what the file says, so it is read verbatim.

    The rule is never to default silently; inventing a second policy for ``""`` would
    make the loader edit its source.
    """
    from ragtime.common.topics import parse_topics

    obj = (
        '{"topic_id": "9001", "collection_id": "c", "title": "",'
        ' "problem_statement": "p", "background": "b", "limit": 1}'
    )
    assert parse_topics(obj)[0].title == ""
