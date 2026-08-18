"""Corpus reader: native and translated record shapes, and the join key between them.

``corpus.read_native`` and ``read_trans`` parse the record shape unmodified, in file order,
tag the language from the filename, and preserve the original ``id`` that both files are
keyed by. The organiser's own translations are not aligned onto the sentence spine, so
the join is not an input to any rendering, though the reader and its key are. Corpus JSONL is
ordinary multi-line, unlike the concatenated single line the topics file ships.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from ragtime.preprocess import corpus

pytestmark = pytest.mark.small


def test_read_native_preserves_fields_and_order(gzip_jsonl_fixture) -> None:
    recs = list(corpus.read_native(gzip_jsonl_fixture["docs"]))
    assert [r["id"] for r in recs] == [
        "spa-docs/0000010",
        "spa-docs/0000011",
        "spa-docs/0000012",
    ]  # JSONL order preserved, not resorted
    first = recs[0]
    # id/text/url/date exactly as written; lang added from the filename (native has none).
    assert first["id"] == "spa-docs/0000010"
    assert first["text"] == "Hola mundo."
    assert first["url"] == "http://a"
    assert first["date"] == "2026-01-01"
    assert first["lang"] == "es"


def test_lang_of_maps_every_docs_and_trans_form() -> None:
    for stem, lang in (("eng", "en"), ("spa", "es"), ("rus", "ru"), ("zho", "zh")):
        assert corpus.lang_of(f"{stem}-docs.jsonl.gz") == lang
    for stem, lang in (("spa", "es"), ("rus", "ru"), ("zho", "zh")):
        assert corpus.lang_of(f"{stem}-trans.jsonl.gz") == lang
    # eng has no -trans counterpart (English identity): nothing to assert beyond docs.
    with pytest.raises(ValueError):
        corpus.lang_of("bogus-docs.jsonl.gz")


def test_read_trans_ids_are_a_subset_of_native_ids(gzip_jsonl_fixture) -> None:
    native_ids = {r["id"] for r in corpus.read_native(gzip_jsonl_fixture["docs"])}
    trans_ids = {r["id"] for r in corpus.read_trans(gzip_jsonl_fixture["trans"])}
    assert trans_ids <= native_ids  # the native<->trans join key
    assert trans_ids  # non-empty (a real subset, not vacuous)


def test_reader_does_not_special_case_a_single_giant_line(tmp_path: Path) -> None:
    """The corpus is clean multi-line JSONL; a concatenated single line is not the shape."""
    p = tmp_path / "rus-docs.jsonl.gz"
    rows = [{"id": f"rus-docs/{i}", "text": f"t{i}", "url": "u", "date": "d"} for i in range(4)]
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    got = list(corpus.read_native(p))
    assert len(got) == 4  # 4 lines -> 4 records (not one giant concatenated line)
