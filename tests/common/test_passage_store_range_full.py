"""The composition pushdown, checked byte for byte against the real corpus.

``iter_final_passages`` takes a ``document_range``: a caller composes one contiguous slice
of ``documents.parquet`` and reads only the row groups that slice touches in all five
tables, instead of composing every passage of every document and filtering afterwards.
Without it the fixed re-read is the dominant cost of a sharded index build, and it grows
as a share of the total the finer the sharding becomes.

The bar is byte-identity and the oracle is the un-ranged behaviour. This file runs the
un-ranged stream over the real corpus once, captures the records it yields for the target
documents, and requires each ranged stream to reproduce them field for field and in order.
A subtly wrong slice would produce a silently incomplete index, and no sanity check on
counts would catch it.

The five shapes that can break alignment, each on real data:

1. a range starting mid-corpus rather than at row 0,
2. a range whose boundary document has no passages, so nothing anchors the passage cursor.
   It is located at runtime; if the shipped corpus has no such document the case is
   reported and skipped rather than faked, and the synthetic proof lives in the small
   sibling,
3. a range of English documents, which has zero rows in either translation table, the
   normal state since the identity-row dedup (``store_identity_translations: false``),
4. a range of non-English documents, with rows in both translation tables,
5. the degenerate single-document range.

Plus: a plan's ranges partition the corpus, so their passage counts sum to the manifest's,
and the measured cost of a ranged read against a full scan.

Read-only: the corpus node under ``runs/`` is never written to, and every test here skips
loudly when it is absent. CPU-only; no model, no GPU.
"""

from __future__ import annotations

import json
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout
from ragtime.common.io import iter_parquet_batches, parquet_row_group_sizes
from ragtime.common.passage_store import (
    IDENTITY_LANG,
    PASSAGES_TABLE,
    RENDERINGS,
    SENTENCES_TABLE,
    iter_final_passages,
    plan_final_ranges,
    translations_table,
)
from ragtime.config import all_hashes, load
from ragtime.orchestration.cli import artifact_root
from ragtime.orchestration.run_identity import run_family
from ragtime.preprocess.packing import packing_hash
from ragtime.preprocess.reconcile import reconcile_hash

pytestmark = pytest.mark.full

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "config" / "e2e-original.yml"
#: The artifact root. Every path below is Layout-resolved from the config's own hashes,
#: never a hardcoded ``recon12`` or ``pack12``, which would silently pass against a stale
#: node. It goes through the one resolver (``orchestration.cli.artifact_root``), and since
#: ``_REPO_ROOT / <absolute>`` is the absolute path, the same expression covers both the
#: repo-relative default and an off-repo scratch root. A test that looked only in the repo
#: would skip itself green the day the store moved.
_SOURCE_BASE = _REPO_ROOT / artifact_root(load(_CONFIG))

#: Documents per probed range. Small enough that five ranges cost seconds, large enough that
#: a range spans many row groups in every table (so the footer pruning is genuinely exercised
#: and an off-by-one in a boundary group would show up).
_RANGE_DOCS = 1500
#: How far into the corpus the mid-corpus case sits, as a fraction of the document count.
#: The oracle pass has to compose everything before it, so this trades wall-clock
#: against being genuinely mid-corpus; 0.5 costs about half a full scan.
_MIDPOINT = 0.5
#: Measured facts, printed by the last test so the numbers land in the job's own log
#: rather than in a file nobody reads.
_MEASURED: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# The real corpus node.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def corpus() -> dict[str, Any]:
    """Layout + hashes for the shipped corpus, or a loud skip."""
    if not _CONFIG.exists():
        pytest.skip(f"{_CONFIG} is absent")
    cfg = load(_CONFIG)
    layout = Layout(
        run_dir=_SOURCE_BASE,
        base=_SOURCE_BASE,
        family=run_family(cfg),
        chunker_hash=all_hashes(cfg)["chunker"],
    )
    recon = reconcile_hash(cfg)
    pack = packing_hash(cfg)
    paths = {
        "documents": layout.documents_path(),
        SENTENCES_TABLE: layout.final_sentences_path(recon),
        PASSAGES_TABLE: layout.final_passages_path(recon, pack),
        **{
            translations_table(v): layout.final_translations_path(recon, v)
            for v in RENDERINGS
            if v != "original"
        },
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"the real corpus node is not built here; missing: {missing[0]}")
    return {
        "layout": layout,
        "recon": recon,
        "pack": pack,
        "paths": paths,
        "total_documents": sum(parquet_row_group_sizes(paths["documents"])),
        "total_passages": sum(parquet_row_group_sizes(paths[PASSAGES_TABLE])),
    }


@pytest.fixture(scope="module")
def lang_blocks(corpus: dict[str, Any]) -> list[tuple[str, int, int]]:
    """The corpus's contiguous per-language document blocks, as ``(lang, start, end)``.

    One narrow-column pass over ``documents.lang``. The blocks are what make "an English
    range" and "a non-English range" addressable as row ranges at all; they are read from the
    table rather than assumed, because a block order baked into a test is a fact that rots.
    """
    blocks: list[tuple[str, int, int]] = []
    cur: str | None = None
    start = 0
    i = 0
    for batch in iter_parquet_batches(
        corpus["paths"]["documents"], columns=["lang"], batch_size=200_000
    ):
        for row in batch:
            lang = row["lang"]
            if lang != cur:
                if cur is not None:
                    blocks.append((cur, start, i))
                cur, start = lang, i
            i += 1
    if cur is not None:
        blocks.append((cur, start, i))
    _MEASURED["lang_blocks"] = blocks
    return blocks


def _block(blocks: list[tuple[str, int, int]], lang: str) -> tuple[int, int]:
    for name, start, end in blocks:
        if name == lang:
            return start, end
    pytest.skip(f"the corpus has no {lang!r} document block")
    raise AssertionError  # unreachable, keeps the type checker honest


@pytest.fixture(scope="module")
def zero_passage_ordinals(corpus: dict[str, Any]) -> list[int]:
    """Every document ordinal that owns no passage rows, from a whole-corpus co-walk.

    Two ``document_id`` columns, forward only, so this is a measured corpus fact rather
    than a sampled guess. It does two jobs: it supplies the
    'boundary document with no passages' case if such a document exists, and it licenses the
    oracle's ordinal arithmetic if none does (the k-th distinct document in the passage
    stream is document ordinal k only when every document has at least one passage).
    """
    passages = (
        row["document_id"]
        for batch in iter_parquet_batches(
            corpus["paths"][PASSAGES_TABLE], columns=["document_id"], batch_size=200_000
        )
        for row in batch
    )
    head = next(passages, None)
    found: list[int] = []
    ordinal = 0
    for batch in iter_parquet_batches(
        corpus["paths"]["documents"], columns=["document_id"], batch_size=200_000
    ):
        for row in batch:
            did = row["document_id"]
            if head == did:
                while head == did:
                    head = next(passages, None)
            else:
                found.append(ordinal)
            ordinal += 1
    assert head is None, "the passage table names a document documents.parquet does not"
    _MEASURED["documents_with_no_passages"] = len(found)
    _MEASURED["first_zero_passage_ordinals"] = found[:10]
    return found


# --------------------------------------------------------------------------- #
# The ranges under test, and the oracle: one un-ranged pass over the real corpus.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cases(
    corpus: dict[str, Any],
    lang_blocks: list[tuple[str, int, int]],
    zero_passage_ordinals: list[int],
) -> dict[str, tuple[int, int]]:
    """``name -> (doc_row_start, doc_row_end)``: the five shapes, on real ordinals."""
    total = corpus["total_documents"]
    en_start, en_end = _block(lang_blocks, IDENTITY_LANG)
    non_english = [b for b in lang_blocks if b[0] != IDENTITY_LANG]
    assert non_english, "the corpus must have a non-English block"

    # Mid-corpus: the block containing the corpus midpoint, offset into it so the range can
    # never coincide with a block edge (an edge is the easy case for an alignment bug).
    midpoint = int(total * _MIDPOINT)
    mid_block = next(
        (b for b in lang_blocks if b[1] <= midpoint < b[2]), non_english[-1]
    )
    mid_start = min(max(midpoint, mid_block[1] + 1), mid_block[2] - _RANGE_DOCS - 1)

    out = {
        "mid_corpus": (mid_start, mid_start + _RANGE_DOCS),
        "english": (en_start + 1, en_start + 1 + _RANGE_DOCS),
        "non_english": (
            non_english[0][1] + 1,
            non_english[0][1] + 1 + _RANGE_DOCS,
        ),
        # The seam between two language blocks: English documents (no translation rows) and
        # non-English ones (rows in both tables) inside one range, which is where a cursor
        # that mistook "no rows" for "end of table" would break.
        "language_seam": (en_end - _RANGE_DOCS // 2, en_end + _RANGE_DOCS // 2),
        "single_document": (mid_start + 7, mid_start + 8),
    }
    if zero_passage_ordinals:
        first = zero_passage_ordinals[0]
        out["no_passage_boundary"] = (first, min(first + _RANGE_DOCS, total))
    _MEASURED["cases"] = out
    return out


@pytest.fixture(scope="module")
def doc_of(corpus: dict[str, Any], cases: dict[str, tuple[int, int]]) -> dict[int, str]:
    """``document ordinal -> document_id`` for the cases' documents.

    A plain forward scan of ``documents.parquet``'s id column, stopped at the deepest case.
    Not a footer-indexed range read: the oracle must not be built out of the
    mechanism it exists to check, or a bug in that mechanism would make the whole file
    agree with itself.
    """
    deepest = max(end for _, end in cases.values())
    wanted = {o for start, end in cases.values() for o in range(start, end)}
    out: dict[int, str] = {}
    ordinal = 0
    for batch in iter_parquet_batches(
        corpus["paths"]["documents"], columns=["document_id"], batch_size=200_000
    ):
        for row in batch:
            if ordinal in wanted:
                out[ordinal] = row["document_id"]
            ordinal += 1
            if ordinal >= deepest:
                break
        if ordinal >= deepest:
            break
    assert len(out) == len(wanted)
    return out


@pytest.fixture(scope="module")
def oracle(
    corpus: dict[str, Any], doc_of: dict[int, str]
) -> dict[str, list[dict[str, Any]]]:
    """What the un-ranged stream yields for every case's documents: the reference.

    This is the expensive fixture: the only trustworthy oracle for "the range
    composes what the corpus composes" is the corpus stream itself, run exactly as every
    shipped caller runs it (``document_range=None``). It is bounded by breaking out once every
    wanted document has gone by and a later one appears; the tables are in document order, so
    nothing after that can be relevant.
    """
    layout, recon, pack = corpus["layout"], corpus["recon"], corpus["pack"]
    by_document: dict[str, list[dict[str, Any]]] = {d: [] for d in doc_of.values()}
    remaining = set(by_document)

    t0 = time.time()
    documents_streamed = 0
    last_document: str | None = None
    stream = iter_final_passages(layout, recon, pack_hash=pack)
    for record in stream:
        document_id = record["document_id"]
        if document_id != last_document:
            last_document = document_id
            documents_streamed += 1
            if not remaining and document_id not in by_document:
                break
        rows = by_document.get(document_id)
        if rows is not None:
            rows.append(record)
            remaining.discard(document_id)
    stream.close()
    _MEASURED["oracle_seconds"] = round(time.time() - t0, 1)
    _MEASURED["oracle_documents_streamed"] = documents_streamed
    _MEASURED["oracle_records_captured"] = sum(len(v) for v in by_document.values())
    assert not remaining, f"{len(remaining)} wanted document(s) never appeared in the stream"
    return by_document


def _expected(
    oracle: dict[str, list[dict[str, Any]]],
    doc_of: dict[int, str],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    """The oracle's records for documents ``[start, end)``, in corpus order."""
    out: list[dict[str, Any]] = []
    for ordinal in range(start, end):
        out.extend(oracle[doc_of[ordinal]])
    return out


# --------------------------------------------------------------------------- #
# Byte identity against the un-ranged stream.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "case",
    [
        "mid_corpus",
        "english",
        "non_english",
        "language_seam",
        "single_document",
        "no_passage_boundary",
    ],
)
def test_a_ranged_composition_is_byte_identical_to_the_full_stream(
    corpus: dict[str, Any],
    cases: dict[str, tuple[int, int]],
    doc_of: dict[int, str],
    oracle: dict[str, list[dict[str, Any]]],
    case: str,
) -> None:
    """Field for field, in order, on the real corpus."""
    if case not in cases:
        pytest.skip(
            "this corpus has "
            f"{_MEASURED.get('documents_with_no_passages')} documents with no passage rows, "
            "so the case is unconstructible on real data; the synthetic proof is in "
            "tests/common/test_passage_store_range_small.py"
        )
    start, end = cases[case]
    (rng,) = plan_final_ranges(
        corpus["layout"],
        corpus["recon"],
        pack_hash=corpus["pack"],
        document_ranges=[(start, end)],
    )
    t0 = time.time()
    got = list(
        iter_final_passages(
            corpus["layout"],
            corpus["recon"],
            pack_hash=corpus["pack"],
            document_range=rng,
        )
    )
    seconds = time.time() - t0
    want = _expected(oracle, doc_of, start, end)
    _MEASURED.setdefault("ranged_reads", {})[case] = {
        "documents": end - start,
        "passages": len(got),
        "seconds": round(seconds, 2),
        "sentence_rows": rng.rows_for(SENTENCES_TABLE),
        "passage_rows": rng.rows_for(PASSAGES_TABLE),
        "omt_rows": rng.rows_for(translations_table("omt")),
    }
    assert len(got) == len(want), f"{case}: {len(got)} passages vs the corpus stream's {len(want)}"
    assert got == want, f"{case}: the ranged composition differs from the corpus stream"


def test_the_english_range_owns_no_translation_rows_and_still_renders(
    corpus: dict[str, Any],
    cases: dict[str, tuple[int, int]],
    doc_of: dict[int, str],
    oracle: dict[str, list[dict[str, Any]]],
) -> None:
    """The dedup's shape, on real data: absence of rows is the English rendering.

    Not merely "no rows were read": the planned range is empty, which is what makes the read
    free rather than filtered, and every rendering still resolves to the native slice.
    """
    start, end = cases["english"]
    (rng,) = plan_final_ranges(
        corpus["layout"],
        corpus["recon"],
        pack_hash=corpus["pack"],
        document_ranges=[(start, end)],
    )
    for variant in RENDERINGS:
        if variant == "original":
            continue
        lo, hi = rng.rows_for(translations_table(variant))
        assert lo == hi, f"an all-English range must own zero {variant} rows, got {hi - lo}"
    records = _expected(oracle, doc_of, start, end)
    assert records, "the English range is not empty"
    for record in records:
        assert record["lang"] == IDENTITY_LANG
        assert record["original"] == record["omt"] == record["omt_opus"]


def test_a_non_english_range_owns_rows_in_both_translation_tables(
    corpus: dict[str, Any],
    cases: dict[str, tuple[int, int]],
    doc_of: dict[int, str],
    oracle: dict[str, list[dict[str, Any]]],
) -> None:
    start, end = cases["non_english"]
    (rng,) = plan_final_ranges(
        corpus["layout"],
        corpus["recon"],
        pack_hash=corpus["pack"],
        document_ranges=[(start, end)],
    )
    for variant in RENDERINGS:
        if variant == "original":
            continue
        lo, hi = rng.rows_for(translations_table(variant))
        assert hi > lo, f"a non-English range must own {variant} rows"
    records = _expected(oracle, doc_of, start, end)
    assert records
    assert all(r["lang"] != IDENTITY_LANG for r in records)
    # A translated rendering that silently fell back to the source would be invisible to a
    # count; it is not invisible to this.
    differing = [r for r in records if r["omt"] != r["original"]]
    assert differing, "every non-English passage must have a real translation"


def test_the_language_seam_range_spans_both_regimes(
    cases: dict[str, tuple[int, int]],
    doc_of: dict[int, str],
    oracle: dict[str, list[dict[str, Any]]],
) -> None:
    """One range holding documents with and without translation rows.

    This is the case a cursor bug hides in: the English half legitimately consumes nothing
    from the translation tables, so a cursor that treated "no rows for this document" as "end
    of stream" would starve every later document in the same range.
    """
    start, end = cases["language_seam"]
    records = _expected(oracle, doc_of, start, end)
    langs = {r["lang"] for r in records}
    assert IDENTITY_LANG in langs and len(langs) > 1, f"the seam range is monolingual: {langs}"


def test_a_plan_partitions_the_whole_corpus(corpus: dict[str, Any]) -> None:
    """The ranges of a shard plan tile every table with no gap and no overlap.

    "Each shard is right" does not imply "the shards together are the corpus". The strongest
    cheap statement of the latter is that the per-table sub-ranges chain end to start across
    the plan, the first starting at 0 and the last ending at the table's row count, checked
    against the real row counts rather than against each other.
    """
    total = corpus["total_documents"]
    n = 51  # the shard count that motivated the pushdown
    step = total // n
    cuts = [(i * step, (i + 1) * step if i < n - 1 else total) for i in range(n)]
    t0 = time.time()
    ranges = plan_final_ranges(
        corpus["layout"], corpus["recon"], pack_hash=corpus["pack"], document_ranges=cuts
    )
    _MEASURED["plan_seconds_51_shards"] = round(time.time() - t0, 1)
    assert [(r.doc_row_start, r.doc_row_end) for r in ranges] == cuts

    for table, path in corpus["paths"].items():
        if table == "documents":
            continue
        rows = sum(parquet_row_group_sizes(path))
        spans = [r.rows_for(table) for r in ranges]
        assert spans[0][0] == 0, f"{table}: the plan does not start at row 0"
        assert spans[-1][1] == rows, f"{table}: the plan ends at {spans[-1][1]} of {rows}"
        for (_, prev_end), (start, _) in pairwise(spans):
            assert start == prev_end, f"{table}: a gap or overlap at row {prev_end}"
        assert sum(hi - lo for lo, hi in spans) == rows


def test_the_measured_cost_of_a_range_against_the_full_scan(
    corpus: dict[str, Any], oracle: dict[str, list[dict[str, Any]]]
) -> None:
    """Print the numbers, and assert the one inequality the pushdown exists to create.

    The claim is not that ranged reads are fast; it is that a range's cost tracks the range
    and not the corpus. So the assertion is per-passage: a ranged read must compose a passage no
    slower than the un-ranged stream does, which is only possible if the fixed corpus re-read
    is gone. Absolute wall-clock on a shared filesystem is reported, never asserted.
    """
    reads = _MEASURED.get("ranged_reads", {})
    assert reads, "no ranged read was measured"
    oracle_seconds = _MEASURED["oracle_seconds"]
    oracle_records = _MEASURED["oracle_records_captured"]
    oracle_documents = _MEASURED["oracle_documents_streamed"]

    full_scan_estimate = oracle_seconds * corpus["total_documents"] / max(1, oracle_documents)
    _MEASURED["full_scan_seconds_estimate"] = round(full_scan_estimate, 1)
    _MEASURED["total_documents"] = corpus["total_documents"]
    _MEASURED["total_passages"] = corpus["total_passages"]

    slowest = max(reads.values(), key=lambda r: r["seconds"])
    ranged_total = sum(r["seconds"] for r in reads.values())
    ranged_passages = sum(r["passages"] for r in reads.values())
    _MEASURED["ranged_total_seconds"] = round(ranged_total, 2)
    _MEASURED["ranged_total_passages"] = ranged_passages
    _MEASURED["speedup_vs_full_scan_per_range"] = round(
        full_scan_estimate / max(0.01, slowest["seconds"]), 1
    )
    print("\n=== ranged read, measured ===")
    print(json.dumps(_MEASURED, indent=2, default=str))

    assert ranged_passages > 0
    # Every ranged read must be cheaper than one full corpus scan by a wide margin; the
    # defect this replaces was that each shard paid a full scan.
    assert slowest["seconds"] < full_scan_estimate / 10, (
        f"the slowest ranged read ({slowest['seconds']:.2f}s) is not an order of magnitude "
        f"below one full corpus scan (~{full_scan_estimate:.0f}s), so the pushdown is not "
        "actually pushing down"
    )
    # ... and the oracle, which is the un-ranged behaviour, is measurably what is avoided.
    assert oracle_records > 0
    assert oracle_seconds > slowest["seconds"]
