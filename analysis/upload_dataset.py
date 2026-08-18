#!/usr/bin/env python3
"""Build and publish the four released configurations of the sentence collection.

The release is four subsets of the same corpus, and none of them ships the document text:
that belongs to the parent collection and is not reshipped here.

    sentences           eng spa rus zho    the segmentation, with the span each sentence claims
    translations_nllb       spa rus zho    the NLLB-200-3.3B English translation, per sentence
    translations_opus       spa rus zho    the OPUS-MT English translation, per sentence
    passages            eng spa rus zho    the passage inventory, as ordered sentence ids

Publishing splits into two halves that want opposite machines, and the split is not a
preference:

    build     CPU and disk, no network       -> SLURM   (slurm/upload_dataset.sbatch)
    transfer  network, almost no CPU         -> the login node, because compute nodes on
                                                this cluster have no route to the internet

Hence the subcommands, in the order they are meant to be run:

    inspect                      what the sources weigh and what the output should weigh
    plan                         resolve every shard's row range, once          (SLURM)
    convert --shard N            write exactly one shard                        (SLURM)
    card                         assemble README.md from the two settled sources
    manifest                     check what is staged against the plan
    push [--dry-run]             transfer the staged folder              (login node)

`convert` never opens a socket and `push` refuses to run inside a SLURM allocation. The
access token is read from HF_TOKEN or HUGGINGFACE_HUB_TOKEN and from nowhere else: never
from a file in this repository, never written to one, never printed.

Layout produced under the staging directory:

    sentences/eng-00000-of-00005.parquet   ...
    translations_nllb/spa-00000-of-00003.parquet   ...
    translations_opus/...
    passages/eng-00000-of-00002.parquet    ...
    README.md                              the dataset card, written by `card`
    _state/<config>/<file>.json            build receipts; never uploaded

Nothing under `_state/` can reach the Hub: `push` uploads an explicit allow-list built
from the plan, not a directory walk.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# What the release is
# --------------------------------------------------------------------------- #

# The parent collection names its splits with three letters, so the release does too, and
# the card's `path: <config>/<split3>-*.parquet` globs are written against these names.
# The corpus tables carry the two-letter code in their own `lang` column and keep it: the
# split name is the file's address, the column is the row's data.
SPLIT_OF_LANG = {"en": "eng", "es": "spa", "ru": "rus", "zh": "zho"}

# The translation tables record the source language in the translator's notation, and
# Chinese arrives under two script tags that are one published split.
SPLIT_OF_SOURCE_LANG = {
    "eng_Latn": "eng",
    "spa_Latn": "spa",
    "rus_Cyrl": "rus",
    "zho_Hans": "zho",
    "zho_Hant": "zho",
}

# One compression setting for the whole release.
#
# Level 9, measured on the real `sentences` columns: level 3 gives 4.69 GB, level 6 gives
# 4.39, level 9 gives 4.30 and level 12 gives 4.24. Decompression speed does not depend on
# the level, so the only thing a higher level costs is time in a build that runs once, and
# level 12 spends about sixty per cent more of it to gain another 1.2 points. Level 9 takes
# what is on offer. Snappy is roughly twice the size and is not a candidate for a table
# that will be read far more often than it is written.
#
# Dictionary encoding is off everywhere, and this is measured rather than assumed: in every
# one of these tables the string columns are either unique per row by construction (ids,
# text) or constant within a split (`lang`), so a dictionary has nothing to reuse and its
# page overhead is pure cost — 1.3 per cent added on `sentences` at the row-group size used
# here, and between 0.3 and 0.5 per cent on the translation tables.
WRITER = {
    "compression": "zstd",
    "compression_level": 9,
    "use_dictionary": False,
    "version": "2.6",
    "data_page_size": 1 << 20,
    "write_statistics": True,
}

# Arrow's `string` type carries 32-bit offsets, so one contiguous string array cannot hold
# more than 2,147,483,647 bytes. A reader who loads a whole split and calls
# `combine_chunks()` on its widest string column is doing exactly that, and the text of a
# split exceeds the limit in every language. The shard cap is what keeps that impossible.
#
# The cap is expressed in UNCOMPRESSED bytes of the widest column, which is the quantity
# the limit is actually about. Capping the compressed size instead — the obvious thing —
# only bounds it through a compression ratio that has to be guessed before the file exists,
# and the ratio differs by a factor of three between `text` and `sentence_ids`.
#
# 1.2 GB leaves a factor of 1.79 under the limit. At zstd 9 that lands a text shard around
# 300 MB on disk, which is also what the Hub asks for.
SHARD_CAP_RAW_BYTES = 1_200_000_000
STRING_OFFSET_LIMIT = 2_147_483_647

# How often a conversion task reports progress, in rows.
PROGRESS_EVERY_ROWS = 5_000_000

TOKEN_ENV = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")

# Ratio between zstd level 9 and the level 3 the internal tables were written at, measured
# on the `sentences` columns (4.30 GB against 4.69 GB). Used only to project a size before
# a config has been built; every built file reports its own bytes.
LEVEL_9_OVER_LEVEL_3 = 4_300_761_364 / 4_692_000_000


@dataclass(frozen=True)
class ConfigSpec:
    """One published configuration: where its rows come from and how they are written."""

    name: str
    #: relative path of the source table under the reconciled build
    source: str
    #: source column naming the language, and how its values map to a split
    lang_column: str
    split_of: dict[str, str]
    #: (source column, published column) in published order
    columns: tuple[tuple[str, str], ...]
    #: the column whose uncompressed size the shard cap is measured on
    widest: str
    #: rows per row group in the published file
    row_group_rows: int
    #: whether a row's text has to be sliced out of the document table
    needs_documents: bool = False
    #: splits that legitimately do not exist in the source
    absent_splits: tuple[str, ...] = ()


# Row-group sizes. The Hub asks for row groups between 100 and 300 MB uncompressed, and the
# published row is small in three of these four configs — about 260 bytes in `sentences`,
# 183 to 269 in the translations — so a million rows lands each of them inside the band. A
# `passages` row is wider, 658 uncompressed bytes measured over the whole table, so a
# million rows would be 658 MB, well above it; 300,000 lands at 197 MB, inside.
#
# Nothing else depends on that figure: it is the one line to change if `passages` is ever
# wanted at a different row-group size.
CONFIGS: tuple[ConfigSpec, ...] = (
    ConfigSpec(
        name="sentences",
        source="sentences.parquet",
        lang_column="lang",
        split_of=SPLIT_OF_LANG,
        columns=(
            ("document_id", "document_id"),
            ("sentence_id", "sentence_id"),
            ("sentence_index", "sentence_index"),
            ("lang", "lang"),
            ("start", "char_start"),
            ("end", "char_end"),
            (None, "text"),
        ),
        widest="text",
        row_group_rows=1_000_000,
        needs_documents=True,
    ),
    ConfigSpec(
        name="translations_nllb",
        source="translations/omt.parquet",
        lang_column="source_lang",
        split_of=SPLIT_OF_SOURCE_LANG,
        columns=(
            ("sentence_id", "sentence_id"),
            ("document_id", "document_id"),
            ("text", "text"),
        ),
        widest="text",
        row_group_rows=1_000_000,
        absent_splits=("eng",),
    ),
    ConfigSpec(
        name="translations_opus",
        source="translations/omt_opus.parquet",
        lang_column="source_lang",
        split_of=SPLIT_OF_SOURCE_LANG,
        columns=(
            ("sentence_id", "sentence_id"),
            ("document_id", "document_id"),
            ("text", "text"),
        ),
        widest="text",
        row_group_rows=1_000_000,
        absent_splits=("eng",),
    ),
    ConfigSpec(
        name="passages",
        source="passages/passages.parquet",
        lang_column="lang",
        split_of=SPLIT_OF_LANG,
        columns=(
            ("passage_id", "passage_id"),
            ("document_id", "document_id"),
            ("lang", "lang"),
            ("sentence_ids", "sentence_ids"),
            ("token_count", "token_count"),
            ("is_oversized", "is_oversized"),
        ),
        widest="sentence_ids",
        row_group_rows=300_000,
    ),
)

BY_NAME = {spec.name: spec for spec in CONFIGS}


def arrow_schema(spec: ConfigSpec):
    """The published Arrow schema, in file order.

    The order is not decorative: the card declares `features` in this order, and a card
    whose order matches the file is one less thing for a reader to reconcile.
    """
    import pyarrow as pa

    types = {
        "document_id": pa.string(),
        "sentence_id": pa.string(),
        "sentence_index": pa.int32(),
        "lang": pa.string(),
        "char_start": pa.int32(),
        "char_end": pa.int32(),
        "text": pa.string(),
        "passage_id": pa.string(),
        "sentence_ids": pa.list_(pa.string()),
        "token_count": pa.int32(),
        "is_oversized": pa.bool_(),
    }
    return pa.schema([pa.field(out, types[out]) for _, out in spec.columns])


# --------------------------------------------------------------------------- #
# Locating the corpus and the staging directory
# --------------------------------------------------------------------------- #


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{int(n):,} B" if unit == "B" else f"{n:,.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


@dataclass
class Corpus:
    corpus_dir: Path
    final_dir: Path
    pack_dir: Path
    tables: dict[str, Path]
    documents: Path


def resolve_corpus(args: argparse.Namespace) -> Corpus:
    """Find the corpus build to publish and the tables the four configs read.

    A reconciled build can legitimately exist more than once under `final/`: a rebuild with
    a different reconciliation policy sits beside the first and both are valid. The one to
    publish is the one the runs used, which is the one the retrieval index was built under,
    so a node carrying an `index/` subtree wins. If that does not pick exactly one, the
    caller says which.
    """
    if args.corpus_dir:
        corpus = Path(args.corpus_dir)
    else:
        root = os.environ.get("RAGTIME_ARTIFACT_ROOT")
        chunker = args.chunker or os.environ.get("DS_CHUNKER")
        if not root or not chunker:
            raise SystemExit("pass --corpus-dir, or set RAGTIME_ARTIFACT_ROOT and DS_CHUNKER")
        corpus = Path(root) / "corpus" / args.family / chunker[:12] / "corpus-preprocess"
    if not corpus.is_dir():
        raise SystemExit(f"no corpus build at {corpus}")

    final_root = corpus / "final"
    named = args.final or os.environ.get("DS_FINAL")
    if named:
        final = final_root / named[:12]
    else:
        candidates = sorted(p for p in final_root.iterdir() if p.is_dir())
        indexed = [p for p in candidates if (p / "index").is_dir()]
        picked = indexed or candidates
        if len(picked) != 1:
            names = ", ".join(p.name for p in candidates)
            raise SystemExit(
                f"{final_root} holds {len(candidates)} reconciled builds ({names}) and none "
                "of them is the unique one carrying a retrieval index; name one with --final"
            )
        final = picked[0]
    if not final.is_dir():
        raise SystemExit(f"no reconciled build at {final}")

    packs = sorted(p for p in (final / "passages").iterdir() if p.is_dir())
    named_pack = args.pack or os.environ.get("DS_PACK")
    pack = final / "passages" / named_pack[:12] if named_pack else (packs[0] if packs else None)
    if pack is None or not (pack / "passages.parquet").is_file():
        raise SystemExit(f"no passage table under {final / 'passages'}; pass --pack")
    if not named_pack and len(packs) != 1:
        names = ", ".join(p.name for p in packs)
        raise SystemExit(f"{final} holds several passage groupings ({names}); pass --pack")

    tables: dict[str, Path] = {}
    for spec in CONFIGS:
        tables[spec.name] = (
            pack / "passages.parquet"
            if spec.source.startswith("passages/")
            else final / spec.source
        )
    documents = corpus / "documents.parquet"
    missing = [str(p) for p in [*tables.values(), documents] if not p.is_file()]
    if missing:
        raise SystemExit("missing source tables:\n  " + "\n  ".join(missing))

    return Corpus(corpus, final, pack, tables, documents)


def staging_dir(args: argparse.Namespace) -> Path:
    staging = args.staging or os.environ.get("DS_STAGING")
    if not staging:
        raise SystemExit("set DS_STAGING to the staging directory, or pass --staging")
    return Path(staging)


# --------------------------------------------------------------------------- #
# inspect: what the sources weigh, from the footers alone
# --------------------------------------------------------------------------- #


def footer(path: Path) -> dict[str, Any]:
    """Row counts and per-column bytes, per row group, without decompressing a page.

    Parquet keeps a per-row-group, per-column index in its footer, so both the stored and
    the uncompressed size of every column are readable from metadata alone. The same footer
    carries per-row-group minima and maxima, and these tables are written in document order
    with the languages in contiguous blocks, so a row group whose language statistic has a
    single value can be attributed to that language. The handful that straddle a boundary
    are reported apart rather than split by a guess.
    """
    import pyarrow.parquet as pq

    with pq.ParquetFile(str(path)) as handle:
        meta = handle.metadata
        first = meta.row_group(0)
        # Column chunks are per leaf, not per Arrow field: a list column contributes several
        # leaves under a dotted path, so index by the leaf path or a list column is
        # mis-attributed silently.
        leaves = [first.column(j).path_in_schema for j in range(first.num_columns)]
        groups = []
        columns = {leaf: {"stored": 0, "raw": 0} for leaf in leaves}
        for i in range(meta.num_row_groups):
            group = meta.row_group(i)
            entry = {"rows": group.num_rows, "stored": 0, "raw": 0, "columns": {}}
            for j, leaf in enumerate(leaves):
                chunk = group.column(j)
                entry["columns"][leaf] = {
                    "stored": chunk.total_compressed_size,
                    "raw": chunk.total_uncompressed_size,
                }
                entry["stored"] += chunk.total_compressed_size
                entry["raw"] += chunk.total_uncompressed_size
                columns[leaf]["stored"] += chunk.total_compressed_size
                columns[leaf]["raw"] += chunk.total_uncompressed_size
            groups.append(entry)
        return {
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "rows": meta.num_rows,
            "row_groups": meta.num_row_groups,
            "leaves": leaves,
            "columns": columns,
            "groups": groups,
        }


def leaf_for(report: dict[str, Any], column: str) -> str:
    """The footer leaf carrying a published column, list children included."""
    if column in report["leaves"]:
        return column
    child = f"{column}.list.element"
    if child in report["leaves"]:
        return child
    raise SystemExit(f"{column} is not a column of {report['path']}")


def published_stored(report: dict[str, Any], spec: ConfigSpec) -> int:
    """Bytes the published columns already occupy inside the internal table."""
    total = 0
    for source, _ in spec.columns:
        if source is None:
            continue
        total += report["columns"][leaf_for(report, source)]["stored"]
    return total


def cmd_inspect(args: argparse.Namespace) -> int:
    corpus = resolve_corpus(args)
    reports = {name: footer(path) for name, path in corpus.tables.items()}
    reports["documents"] = footer(corpus.documents)
    staging = Path(args.staging or os.environ.get("DS_STAGING") or ".")

    print(f"corpus build   {corpus.corpus_dir}")
    print(f"reconciled as  {corpus.final_dir.name}   passages grouped as {corpus.pack_dir.name}")
    print()
    print("SOURCE TABLES, as they sit on scratch")
    print(f"{'table':<20}{'rows':>14}{'on disk':>16}{'uncompressed':>16}{'row groups':>12}")
    read_total = 0
    for name in ("documents", *[s.name for s in CONFIGS]):
        entry = reports[name]
        raw = sum(c["raw"] for c in entry["columns"].values())
        read_total += entry["file_bytes"]
        print(
            f"{name:<20}{entry['rows']:>14,}{human(entry['file_bytes']):>16}"
            f"{human(raw):>16}{entry['row_groups']:>12,}"
        )
    print(f"{'read by the build':<20}{'':>14}{human(read_total):>16}")
    print()

    print("PUBLISHED COLUMNS, and what they should weigh once rewritten")
    print(f"{'config':<20}{'rows':>14}{'in the source':>16}{'projected':>16}{'built':>16}")
    projected_total = 0
    built_total = 0
    for spec in CONFIGS:
        entry = reports[spec.name]
        stored = published_stored(entry, spec)
        if spec.needs_documents:
            # The sentence text is not a column of the sentence table: it is sliced out of
            # the documents, which the sentences tile. The document text column is the
            # right term to add, and it is a slight overestimate because it also carries
            # the separators between sentences, which no published row holds.
            stored += reports["documents"]["columns"]["text"]["stored"]
        projected = int(stored * LEVEL_9_OVER_LEVEL_3)
        built = sum(p.stat().st_size for p in sorted((staging / spec.name).glob("*.parquet")))
        projected_total += projected
        built_total += built
        print(
            f"{spec.name:<20}{entry['rows']:>14,}{human(stored):>16}{human(projected):>16}"
            f"{(human(built) if built else '-'):>16}"
        )
    print(
        f"{'all four':<20}{'':>14}{'':>16}{human(projected_total):>16}"
        f"{(human(built_total) if built_total else '-'):>16}"
    )
    print()
    print(
        "The projection rescales what the published columns already occupy inside the\n"
        "internal tables, which are written at zstd level 3 with 20,000-row groups, by the\n"
        f"{LEVEL_9_OVER_LEVEL_3:.3f} that level 9 measured on the sentences columns. It is a\n"
        "projection across tables with different column shapes; the built column is the\n"
        "number to trust, and it appears as soon as a config is converted."
    )
    print()

    print("SHARDS the cap implies, per split")
    print(f"{'config':<20}{'split':>8}{'rows':>14}{'widest column':>18}{'shards':>9}")
    for spec in CONFIGS:
        for split, block in language_blocks(reports, spec).items():
            raw = block["widest_raw"]
            print(
                f"{spec.name:<20}{split:>8}{block['rows']:>14,}{human(raw):>18}"
                f"{max(1, math.ceil(raw / SHARD_CAP_RAW_BYTES)):>9}"
            )
    print()
    print(
        f"cap {human(SHARD_CAP_RAW_BYTES)} uncompressed on the widest column, a factor of "
        f"{STRING_OFFSET_LIMIT / SHARD_CAP_RAW_BYTES:.2f} under Arrow's 32-bit string offset "
        "limit.\nRow-group boundaries are what the footer resolves, so these counts are one "
        "off at worst;\nthe plan cuts on document boundaries and is exact."
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {k: {kk: vv for kk, vv in v.items() if kk != "groups"} for k, v in reports.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten to {args.out}")
    return 0


def _decode(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value if isinstance(value, str) else None


def language_blocks(
    reports: dict[str, dict[str, Any]], spec: ConfigSpec
) -> dict[str, dict[str, int]]:
    """Per-split rows and widest-column uncompressed bytes, at row-group resolution.

    `sentences` is the one config whose widest published column is not in its own source
    table: the text is sliced out of the document table at build time. Its budget is
    therefore the document text of the same language, which the sentences tile.
    """
    import pyarrow.parquet as pq

    report = reports[spec.name]
    budget = reports["documents"] if spec.needs_documents else report
    budget_leaf = "text" if spec.needs_documents else leaf_for(report, spec.widest)
    budget_lang = "lang" if spec.needs_documents else spec.lang_column
    budget_split_of = SPLIT_OF_LANG if spec.needs_documents else spec.split_of

    def per_split(
        entry: dict[str, Any], lang_column: str, split_of: dict[str, str], leaf: str | None
    ):
        out: dict[str, dict[str, int]] = {}
        with pq.ParquetFile(entry["path"]) as handle:
            meta = handle.metadata
            key = entry["leaves"].index(lang_column)
            for i, group in enumerate(entry["groups"]):
                stats = meta.row_group(i).column(key).statistics
                lo = _decode(stats.min) if stats is not None and stats.has_min_max else None
                hi = _decode(stats.max) if stats is not None and stats.has_min_max else None
                split = None
                if lo is not None and split_of.get(lo) == split_of.get(hi):
                    split = split_of.get(lo)
                bucket = out.setdefault(split or "(spans two)", {"rows": 0, "widest_raw": 0})
                bucket["rows"] += group["rows"]
                if leaf:
                    bucket["widest_raw"] += group["columns"][leaf]["raw"]
        return out

    rows = per_split(
        report, spec.lang_column, spec.split_of, None if spec.needs_documents else budget_leaf
    )
    if not spec.needs_documents:
        return rows
    bytes_by_split = per_split(budget, budget_lang, budget_split_of, budget_leaf)
    for split, bucket in rows.items():
        bucket["widest_raw"] = bytes_by_split.get(split, {}).get("widest_raw", 0)
    return rows


# --------------------------------------------------------------------------- #
# plan: every shard's row range, resolved once
# --------------------------------------------------------------------------- #
#
# Planning is a separate step for one reason: it is the only part of the build that has to
# read a whole column of every source table, and it would otherwise be repeated by every
# task in the array. It resolves, for each config and split, a list of shards, and each
# shard is a contiguous row range that
#
#   * lies inside one language block,
#   * begins and ends on a document boundary, so a document's rows are never split across
#     two files, and
#   * carries at most SHARD_CAP_RAW_BYTES of its widest column.
#
# Document boundaries come from a forward scan of the source's `document_id` column, done
# with Arrow's own comparison kernels rather than in Python: a batch is compared against
# itself shifted by one row, and only the positions that differ are materialised. The byte
# budget comes from the footer, which gives the widest column's uncompressed size per row
# group; a row index is converted to a byte offset by interpolating inside its group, which
# is exact to 20,000 rows and is being compared against a cap of over a gigabyte.


def document_starts(path: Path, batch_rows: int = 1 << 20) -> list[int]:
    """Row indices at which a new document begins, over the whole table.

    Four million boundaries is thirty-two megabytes of Python integers, so the list is kept
    whole; what is never kept whole is the column it came from.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    starts = [0]
    previous: Any = None
    offset = 0
    with pq.ParquetFile(str(path)) as handle:
        for batch in handle.iter_batches(batch_size=batch_rows, columns=["document_id"]):
            column = batch.column(0)
            if len(column) == 0:
                continue
            if previous is not None and column[0].as_py() != previous:
                starts.append(offset)
            changed = pc.not_equal(column.slice(1), column.slice(0, len(column) - 1))
            for index in pc.indices_nonzero(changed).to_pylist():
                starts.append(offset + index + 1)
            previous = column[len(column) - 1].as_py()
            offset += len(column)
    return starts


def cumulative(values: list[int]) -> list[int]:
    total = 0
    out = [0]
    for value in values:
        total += value
        out.append(total)
    return out


def byte_at(row: int, cum_rows: list[int], cum_bytes: list[int]) -> float:
    """Uncompressed bytes of the widest column before `row`, interpolated in its group."""

    index = max(0, bisect.bisect_right(cum_rows, row) - 1)
    index = min(index, len(cum_rows) - 2)
    span_rows = cum_rows[index + 1] - cum_rows[index]
    span_bytes = cum_bytes[index + 1] - cum_bytes[index]
    if span_rows <= 0:
        return float(cum_bytes[index])
    return cum_bytes[index] + span_bytes * (row - cum_rows[index]) / span_rows


def split_runs(
    path: Path, spec: ConfigSpec, batch_rows: int = 1 << 20
) -> list[tuple[str, int, int]]:
    """Contiguous [start, end) row ranges, one per language block, in table order.

    Two blocks of the same language would be a defect rather than a shape to accommodate,
    so they are reported and refused rather than merged: every downstream number here
    assumes one contiguous block per split.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    runs: list[tuple[str, int, int]] = []
    offset = 0
    with pq.ParquetFile(str(path)) as handle:
        for batch in handle.iter_batches(batch_size=batch_rows, columns=[spec.lang_column]):
            column = batch.column(0)
            if len(column) == 0:
                continue
            boundaries = [0]
            changed = pc.not_equal(column.slice(1), column.slice(0, len(column) - 1))
            boundaries.extend(i + 1 for i in pc.indices_nonzero(changed).to_pylist())
            boundaries.append(len(column))
            for start, end in itertools.pairwise(boundaries):
                value = column[start].as_py()
                split = spec.split_of.get(value)
                if split is None:
                    raise SystemExit(
                        f"{path} row {offset + start} has language {value!r}, which is not "
                        "one of the published splits"
                    )
                if runs and runs[-1][0] == split and runs[-1][2] == offset + start:
                    runs[-1] = (split, runs[-1][1], offset + end)
                else:
                    runs.append((split, offset + start, offset + end))
            offset += len(column)
    seen = [run[0] for run in runs]
    if len(set(seen)) != len(seen):
        raise SystemExit(
            f"{path} interleaves languages ({' '.join(seen)}); the shard plan assumes one "
            "contiguous block per split"
        )
    return runs


def plan_config(
    spec: ConfigSpec, path: Path, report: dict[str, Any], budget_report: dict[str, Any]
) -> list[dict[str, Any]]:
    """Cut one source table into shards, honouring the cap and the document boundaries."""
    runs = split_runs(path, spec)
    starts = document_starts(path)
    cum_rows = cumulative([g["rows"] for g in report["groups"]])
    if spec.needs_documents:
        # The budget is the document table's text, which is indexed by document, not by
        # sentence, so the interpolation runs over documents and the boundary list is what
        # converts between the two.
        cum_budget_rows = cumulative([g["rows"] for g in budget_report["groups"]])
        cum_budget_bytes = cumulative(
            [g["columns"]["text"]["raw"] for g in budget_report["groups"]]
        )

        def budget_at(row: int) -> float:
            document = bisect.bisect_right(starts, row) - 1
            return byte_at(document, cum_budget_rows, cum_budget_bytes)
    else:
        leaf = leaf_for(report, spec.widest)
        cum_bytes = cumulative([g["columns"][leaf]["raw"] for g in report["groups"]])

        def budget_at(row: int) -> float:
            return byte_at(row, cum_rows, cum_bytes)

    shards: list[dict[str, Any]] = []
    for split, run_start, run_end in runs:
        first = bisect.bisect_left(starts, run_start)
        last = bisect.bisect_left(starts, run_end)
        cuts = [run_start]
        anchor = budget_at(run_start)
        for index in range(first, last):
            row = starts[index]
            if row <= cuts[-1]:
                continue
            if budget_at(row) - anchor >= SHARD_CAP_RAW_BYTES:
                cuts.append(row)
                anchor = budget_at(row)
        cuts.append(run_end)
        for i, (start, end) in enumerate(itertools.pairwise(cuts)):
            shards.append(
                {
                    "config": spec.name,
                    "split": split,
                    "ordinal": i,
                    "row_start": start,
                    "row_end": end,
                    "rows": end - start,
                    "widest_raw_bytes": int(budget_at(end) - budget_at(start)),
                    **(
                        {
                            "document_start": bisect.bisect_right(starts, start) - 1,
                            "document_end": bisect.bisect_right(starts, end - 1),
                        }
                        if spec.needs_documents
                        else {}
                    ),
                }
            )
    # The name carries the total, so it can only be written once every shard of a split is
    # known. `-of-` counts the split's files, not the config's, which is what the Hub's own
    # naming and the card's globs expect.
    totals: dict[tuple[str, str], int] = {}
    for shard in shards:
        key = (shard["config"], shard["split"])
        totals[key] = totals.get(key, 0) + 1
    for shard in shards:
        total = totals[(shard["config"], shard["split"])]
        shard["shards_in_split"] = total
        shard["relative_path"] = (
            f"{shard['config']}/{shard['split']}-{shard['ordinal']:05d}-of-{total:05d}.parquet"
        )
    return shards


def adopt_built(staging: Path, spec: ConfigSpec, source_rows: int) -> list[dict[str, Any]] | None:
    """Take an already-complete config in the staging directory as it stands.

    A config can arrive here built by something other than this script — `sentences` did,
    and its files have already been checked against the card. Re-cutting them would change
    nothing a reader can see and would invalidate work that is verified, so a complete set
    is adopted rather than replanned.

    Complete means three things, all checkable from the footers alone: every file is named
    `<split>-<ordinal>-of-<total>.parquet`, each split's file count matches the total its
    own names declare with no ordinal missing or repeated, and the row counts sum to the
    source table's. The third is what makes this more than a naming check — a truncated
    build has the right names and the wrong rows.
    """
    import pyarrow.parquet as pq

    directory = staging / spec.name
    files = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
    if not files:
        return None

    found: dict[str, dict[int, tuple[Path, int]]] = {}
    declared: dict[str, set[int]] = {}
    for path in files:
        stem = path.stem
        try:
            split, ordinal, _of, total = stem.split("-")
            index, count = int(ordinal), int(total)
        except ValueError:
            print(f"  {path.name} is not named <split>-<n>-of-<n>; not adopting {spec.name}")
            return None
        with pq.ParquetFile(str(path)) as handle:
            rows = handle.metadata.num_rows
        found.setdefault(split, {})[index] = (path, rows)
        declared.setdefault(split, set()).add(count)

    total_rows = 0
    shards: list[dict[str, Any]] = []
    for split, members in sorted(found.items()):
        if len(declared[split]) != 1:
            print(f"  {spec.name}/{split} files disagree on how many there are; not adopting")
            return None
        count = next(iter(declared[split]))
        if sorted(members) != list(range(count)):
            print(f"  {spec.name}/{split} is missing or repeating an ordinal; not adopting")
            return None
        for index in range(count):
            path, rows = members[index]
            total_rows += rows
            shards.append(
                {
                    "config": spec.name,
                    "split": split,
                    "ordinal": index,
                    "shards_in_split": count,
                    "relative_path": f"{spec.name}/{path.name}",
                    "rows": rows,
                    "adopted": True,
                }
            )
    if total_rows != source_rows:
        print(
            f"  {spec.name} holds {total_rows:,} rows against the source's {source_rows:,}; "
            "not adopting"
        )
        return None
    return shards


def cmd_plan(args: argparse.Namespace) -> int:
    corpus = resolve_corpus(args)
    staging = staging_dir(args)
    staging.mkdir(parents=True, exist_ok=True)

    documents = footer(corpus.documents)

    # Replanning one config must not silently drop the others: a shard index is the array
    # index, so a plan that loses a config renumbers every task after it and the receipts
    # under _state/ stop matching what the array now builds. Anything not replanned is
    # carried over from the plan already on disk.
    carried: list[dict[str, Any]] = []
    if args.only and (staging / "plan.json").is_file():
        carried = [
            shard for shard in read_plan(staging)["shards"] if shard["config"] not in set(args.only)
        ]
        if carried:
            print(f"carrying {len(carried)} shard(s) from the plan already on disk")

    shards: list[dict[str, Any]] = []
    for spec in CONFIGS:
        if args.only and spec.name not in args.only:
            shards.extend(s for s in carried if s["config"] == spec.name)
            continue
        path = corpus.tables[spec.name]
        print(f"planning {spec.name} from {path.name}", flush=True)
        started = time.time()
        report = footer(path)
        produced = None if args.no_adopt else adopt_built(staging, spec, report["rows"])
        if produced is not None:
            print(f"  adopting {len(produced)} file(s) already in the staging directory")
        else:
            produced = plan_config(spec, path, report, documents)
        shards.extend(produced)
        by_split: dict[str, int] = {}
        for shard in produced:
            by_split[shard["split"]] = by_split.get(shard["split"], 0) + 1
        summary = " ".join(f"{split} {count}" for split, count in sorted(by_split.items()))
        print(f"  {len(produced)} shards ({summary})  {time.time() - started:.0f}s", flush=True)

    for index, shard in enumerate(shards):
        shard["index"] = index

    plan = {
        "corpus_dir": str(corpus.corpus_dir),
        "final": corpus.final_dir.name,
        "pack": corpus.pack_dir.name,
        "documents": str(corpus.documents),
        "sources": {spec.name: str(corpus.tables[spec.name]) for spec in CONFIGS},
        "writer": WRITER,
        "row_group_rows": {spec.name: spec.row_group_rows for spec in CONFIGS},
        "shard_cap_raw_bytes": SHARD_CAP_RAW_BYTES,
        "shards": shards,
    }
    out = staging / "plan.json"
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print()
    print(f"{len(shards)} shards planned, written to {out}")
    print(
        f"convert them with  sbatch --array=0-{len(shards) - 1}%{args.throttle} slurm/upload_dataset.sbatch"
    )
    return 0


def read_plan(staging: Path) -> dict[str, Any]:
    path = staging / "plan.json"
    if not path.is_file():
        raise SystemExit(f"no plan at {path}; run the plan stage first")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# convert: one task writes one file
# --------------------------------------------------------------------------- #
#
# The resume unit is one output file, which is what makes the array free to re-run. A task
# writes into a temporary beside its destination, renames it into place, and only then
# writes a receipt under `_state/`. So the three states are unambiguous: no receipt means
# not built, a receipt whose size matches the file on disk means built, and a leftover
# temporary means a task died and its successor starts that file again. A rerun of the
# whole array costs one stat per finished shard.


def state_path(staging: Path, shard: dict[str, Any]) -> Path:
    return staging / "_state" / f"{shard['relative_path'].replace('/', '__')}.json"


def already_built(staging: Path, shard: dict[str, Any]) -> dict[str, Any] | None:
    receipt = state_path(staging, shard)
    target = staging / shard["relative_path"]
    if not receipt.is_file() or not target.is_file():
        return None
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    if recorded.get("bytes") != target.stat().st_size:
        return None
    return recorded


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ShardWriter:
    """Write one parquet file with row groups of an exact size.

    Handing the writer each incoming table straight through would make the row groups as
    ragged as the input, and the input here is cut on document boundaries, so groups would
    range from one row to a million. Buffering and emitting fixed slices is what turns the
    row-group size from an intention into a property of the file.
    """

    def __init__(self, path: Path, schema, row_group_rows: int) -> None:
        import pyarrow.parquet as pq

        self.path = path
        self.row_group_rows = row_group_rows
        self.rows = 0
        self._buffer: list[Any] = []
        self._buffered = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_name(path.name + f".partial-{os.getpid()}")
        self._writer = pq.ParquetWriter(str(self.temporary), schema, **WRITER)

    def write(self, table) -> None:
        import pyarrow as pa

        self._buffer.append(table)
        self._buffered += table.num_rows
        while self._buffered >= self.row_group_rows:
            merged = pa.concat_tables(self._buffer)
            self._writer.write_table(
                merged.slice(0, self.row_group_rows), row_group_size=self.row_group_rows
            )
            self.rows += self.row_group_rows
            rest = merged.slice(self.row_group_rows)
            self._buffer = [rest] if rest.num_rows else []
            self._buffered = rest.num_rows

    def close(self) -> None:
        import pyarrow as pa

        if self._buffered:
            self._writer.write_table(
                pa.concat_tables(self._buffer), row_group_size=self.row_group_rows
            )
            self.rows += self._buffered
            self._buffer, self._buffered = [], 0
        self._writer.close()
        os.replace(self.temporary, self.path)


def row_range_batches(
    path: Path, columns: list[str], start: int, end: int | None, batch_rows: int
) -> Iterator[Any]:
    """Record batches covering exactly rows [start, end) of a table, or to its end.

    Parquet's footer indexes row groups, so the groups outside the range are never read;
    the two at the edges are read whole and trimmed. `iter_batches` has no offset, which is
    why this walks groups rather than batches.
    """
    import pyarrow.parquet as pq

    with pq.ParquetFile(str(path)) as handle:
        if end is None:
            end = handle.metadata.num_rows
        offset = 0
        for index in range(handle.metadata.num_row_groups):
            rows = handle.metadata.row_group(index).num_rows
            group_start, group_end = offset, offset + rows
            offset = group_end
            if group_end <= start:
                continue
            if group_start >= end:
                break
            table = handle.read_row_group(index, columns=columns)
            lo = max(0, start - group_start)
            hi = min(rows, end - group_start)
            table = table.slice(lo, hi - lo)
            yield from table.to_batches(max_chunksize=batch_rows)


def documents_in_order(path: Path, start: int, end: int | None) -> Iterator[tuple[str, str]]:
    for batch in row_range_batches(path, ["document_id", "text"], start, end, 4096):
        ids = batch.column(0).to_pylist()
        texts = batch.column(1).to_pylist()
        yield from zip(ids, texts)


def convert_sentences(
    corpus: Corpus, spec: ConfigSpec, shard: dict[str, Any], writer: ShardWriter
) -> None:
    """Slice each sentence's text out of its document and write the published columns.

    The sentence table stores a span, not a string: the text lives in the document table
    and is recovered by slicing. Both tables are in document order, and a document's
    sentences are contiguous, so the join needs no index and no memory beyond the document
    currently being read — the document reader is advanced until its id matches the one the
    sentence claims, and a mismatch is an error rather than something to search around.
    """
    import pyarrow as pa

    schema = arrow_schema(spec)
    # The end of the range is open. `document_start` is derived from how many
    # documents have sentences, which can only be at most the document row it names, so it
    # is safe as a place to start reading and unsafe as a place to stop; the generator is
    # lazy, so reading to the end of the table costs nothing beyond the rows actually
    # pulled. What guarantees correctness is the id match below, not the bound.
    documents = documents_in_order(corpus.documents, shard["document_start"], None)
    current_id: str | None = None
    current_text: str = ""
    columns = ["sentence_id", "document_id", "sentence_index", "lang", "start", "end"]
    seen = 0
    reported = PROGRESS_EVERY_ROWS
    for batch in row_range_batches(
        corpus.tables[spec.name], columns, shard["row_start"], shard["row_end"], 65_536
    ):
        block = {name: batch.column(i).to_pylist() for i, name in enumerate(columns)}
        texts: list[str] = []
        for document_id, lo, hi in zip(block["document_id"], block["start"], block["end"]):
            while current_id != document_id:
                try:
                    current_id, current_text = next(documents)
                except StopIteration:
                    raise SystemExit(
                        f"document {document_id} is not inside rows "
                        f"[{shard['document_start']}, {shard['document_end']}) of the "
                        "document table; the two tables are not in the same order"
                    )
            texts.append(current_text[lo:hi])
        writer.write(
            pa.table(
                {
                    "document_id": pa.array(block["document_id"], pa.string()),
                    "sentence_id": pa.array(block["sentence_id"], pa.string()),
                    "sentence_index": pa.array(block["sentence_index"], pa.int32()),
                    "lang": pa.array(block["lang"], pa.string()),
                    "char_start": pa.array(block["start"], pa.int32()),
                    "char_end": pa.array(block["end"], pa.int32()),
                    "text": pa.array(texts, pa.string()),
                },
                schema=schema,
            )
        )
        seen += batch.num_rows
        if seen >= reported:
            print(f"  {seen:,} rows", flush=True)
            reported += PROGRESS_EVERY_ROWS


def convert_projection(
    corpus: Corpus, spec: ConfigSpec, shard: dict[str, Any], writer: ShardWriter
) -> None:
    """Rewrite a row range as the published columns, renamed and reordered.

    Three of the four configs need nothing else: their rows are already in the source
    table, and publishing them is a projection onto a subset of the columns under the
    names the card declares. The rewrite is not a copy — the row-group size and the
    compression level both change — so the file is written rather than sliced out.
    """
    import pyarrow as pa

    schema = arrow_schema(spec)
    sources = [source for source, _ in spec.columns]
    seen = 0
    reported = PROGRESS_EVERY_ROWS
    for batch in row_range_batches(
        corpus.tables[spec.name], sources, shard["row_start"], shard["row_end"], 65_536
    ):
        table = pa.Table.from_batches([batch])
        writer.write(
            pa.table(
                {out: table.column(source) for source, out in spec.columns},
                schema=schema,
            )
        )
        seen += batch.num_rows
        if seen >= reported:
            print(f"  {seen:,} rows", flush=True)
            reported += PROGRESS_EVERY_ROWS


def write_receipt(
    staging: Path, shard: dict[str, Any], spec: ConfigSpec, target: Path, elapsed: float
) -> dict[str, Any]:
    """Record what is on disk, read back off the file rather than off the intention.

    The receipt is the resume marker and the input to the transfer's skip decision, so it
    describes the file as it exists: its rows and its row-group size come from its own
    footer, not from the spec that asked for them. An adopted file written by something
    else therefore gets an honest receipt, and a disagreement with the release's settings
    is visible rather than assumed away.
    """
    import pyarrow.parquet as pq

    with pq.ParquetFile(str(target)) as handle:
        meta = handle.metadata
        rows = meta.num_rows
        groups = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
    receipt = {
        "relative_path": shard["relative_path"],
        "config": shard["config"],
        "split": shard["split"],
        "rows": rows,
        "bytes": target.stat().st_size,
        "sha256": sha256_of(target),
        "row_groups": len(groups),
        "row_group_rows": max(groups) if groups else 0,
        "row_group_rows_intended": spec.row_group_rows,
        "writer": WRITER,
        "adopted": bool(shard.get("adopted")),
        "elapsed_s": round(elapsed, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = state_path(staging, shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def cmd_convert(args: argparse.Namespace) -> int:
    corpus = resolve_corpus(args)
    staging = staging_dir(args)
    plan = read_plan(staging)
    try:
        shard = next(s for s in plan["shards"] if s["index"] == args.shard)
    except StopIteration:
        raise SystemExit(f"no shard {args.shard} in the plan ({len(plan['shards'])} shards)")

    spec = BY_NAME[shard["config"]]
    target = staging / shard["relative_path"]
    if not args.force:
        recorded = already_built(staging, shard)
        if recorded is not None:
            print(
                f"shard {args.shard} ({shard['relative_path']}) is already built: "
                f"{recorded['rows']:,} rows, {human(recorded['bytes'])}"
            )
            return 0

    if shard.get("adopted"):
        # An adopted shard was written before this plan existed, so there is nothing to
        # build; what is missing is only its receipt, which the transfer needs in order to
        # skip a file the Hub already holds. Writing one costs a checksum of a file that is
        # already on disk.
        if not target.is_file():
            raise SystemExit(
                f"{target} is adopted by the plan but is not on disk; replan with --no-adopt "
                "to cut this config from the source instead"
            )
        write_receipt(staging, shard, spec, target, elapsed=0.0)
        print(f"{target}  adopted, {shard['rows']:,} rows, {human(target.stat().st_size)}")
        return 0

    print(
        f"shard {args.shard}  {shard['relative_path']}  "
        f"rows [{shard['row_start']:,}, {shard['row_end']:,})",
        flush=True,
    )
    started = time.time()
    writer = ShardWriter(target, arrow_schema(spec), spec.row_group_rows)
    try:
        if spec.needs_documents:
            convert_sentences(corpus, spec, shard, writer)
        else:
            convert_projection(corpus, spec, shard, writer)
        writer.close()
    except BaseException:
        writer.temporary.unlink(missing_ok=True)
        raise

    if writer.rows != shard["rows"]:
        target.unlink(missing_ok=True)
        raise SystemExit(
            f"wrote {writer.rows:,} rows where the plan says {shard['rows']:,}; the plan and "
            "the corpus disagree, and a shard that does not match its plan is not publishable"
        )

    receipt = write_receipt(staging, shard, spec, target, elapsed=time.time() - started)
    print(f"{target}  {writer.rows:,} rows  {human(receipt['bytes'])}  {receipt['elapsed_s']:.0f}s")
    return 0


# --------------------------------------------------------------------------- #
# manifest: what is staged, against what was planned
# --------------------------------------------------------------------------- #


def staged(staging: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per planned file, with whatever is known about the local copy."""
    entries = []
    for shard in plan["shards"]:
        path = staging / shard["relative_path"]
        receipt = state_path(staging, shard)
        recorded = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
        entries.append(
            {
                "path": shard["relative_path"],
                "local": path,
                "config": shard["config"],
                "split": shard["split"],
                "planned_rows": shard["rows"],
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "rows": recorded.get("rows"),
                "sha256": recorded.get("sha256"),
            }
        )
    return entries


def cmd_manifest(args: argparse.Namespace) -> int:
    staging = staging_dir(args)
    plan = read_plan(staging)
    entries = staged(staging, plan)

    print(f"{'file':<46}{'rows':>14}{'size':>14}  state")
    per_config: dict[str, list[int]] = {}
    for entry in entries:
        totals = per_config.setdefault(entry["config"], [0, 0, 0])
        if not entry["present"]:
            print(f"{entry['path']:<46}{entry['planned_rows']:>14,}{'-':>14}  not built")
            continue
        state = "built"
        if entry["rows"] is None:
            state = "no receipt; rebuild it or it cannot be verified"
        elif entry["rows"] != entry["planned_rows"]:
            state = f"ROWS DISAGREE with the plan ({entry['rows']:,})"
        totals[0] += 1
        totals[1] += entry["rows"] or 0
        totals[2] += entry["bytes"]
        print(f"{entry['path']:<46}{entry['rows'] or 0:>14,}{human(entry['bytes']):>14}  {state}")
    print()
    print(f"{'config':<20}{'files':>8}{'rows':>16}{'size':>16}")
    for name, (files, rows, size) in per_config.items():
        print(f"{name:<20}{files:>8}{rows:>16,}{human(size):>16}")
    total = sum(t[2] for t in per_config.values())
    print(
        f"{'all':<20}{sum(t[0] for t in per_config.values()):>8}"
        f"{sum(t[1] for t in per_config.values()):>16,}{human(total):>16}"
    )

    missing = [e["path"] for e in entries if not e["present"]]
    card = staging / "README.md"
    print()
    print(f"dataset card: {'present' if card.is_file() else 'MISSING'} ({card})")
    if missing:
        indices = [s["index"] for s in plan["shards"] if s["relative_path"] in set(missing)]
        print(f"{len(missing)} file(s) not built; convert them with")
        print(f"  sbatch --array={','.join(str(i) for i in indices)} slurm/upload_dataset.sbatch")
        return 1
    return 0

# --------------------------------------------------------------------------- #
# card: assemble README.md out of the two settled sources
# --------------------------------------------------------------------------- #
#
# The card is front-matter plus body, and neither is written here: the front-matter is the
# settled block in `docs/release/card-front-matter.md` and the body is the block under
# "The body, verbatim" in `docs/HUGGINGFACE.md`. This step extracts both, checks the
# front-matter against what the release actually is, and writes the file. Keeping it a
# step rather than a paste is the point — a hand-assembled card drifts from the documents
# it was copied out of, and the one field where drift is not survivable is `num_examples`:
# `datasets` verifies it at the default verification level, so a card that is wrong by one
# row raises `NonMatchingSplitsSizesError` for every user until it is fixed.

CARD_FRONT_MATTER = Path(__file__).resolve().parents[1] / "docs" / "release" / "card-front-matter.md"
CARD_BODY = Path(__file__).resolve().parents[1] / "docs" / "HUGGINGFACE.md"
CARD_FRONT_MATTER_HEADING = "## The block"
CARD_BODY_HEADING = "## The body, verbatim"

#: The repository id the body's examples are written against; `--repo-id` rewrites it.
CARD_REPO_PLACEHOLDER = "jknafou/trec-ragtime-2026"

#: Splits per config, in the order the card declares them.
RELEASE_SPLITS: dict[str, tuple[str, ...]] = {
    "sentences": ("eng", "spa", "rus", "zho"),
    "translations_nllb": ("spa", "rus", "zho"),
    "translations_opus": ("spa", "rus", "zho"),
    "passages": ("eng", "spa", "rus", "zho"),
}

#: Measured rows per config and split — the counts the front-matter has to declare. They
#: come from each element's own full-table measurement and are re-checked here against the
#: footers of whatever is already staged, which is the only check that can catch drift
#: before a reader does.
RELEASE_ROWS: dict[str, dict[str, int]] = {
    "sentences": {"eng": 32_799_412, "spa": 21_435_940, "rus": 18_614_576, "zho": 15_869_272},
    "translations_nllb": {"spa": 21_435_940, "rus": 18_614_576, "zho": 15_869_272},
    "translations_opus": {"spa": 21_435_940, "rus": 18_614_576, "zho": 15_869_272},
    "passages": {"eng": 2_906_906, "spa": 2_545_034, "rus": 1_832_768, "zho": 2_657_132},
}

RELEASE_TOTAL_ROWS = sum(sum(rows.values()) for rows in RELEASE_ROWS.values())
RELEASE_SIZE_CATEGORY = "100M<n<1B"
RELEASE_LICENSE = "cc-by-sa-4.0"
DEFAULT_CONFIG = "sentences"


def fenced_block(path: Path, heading: str, language: str = "") -> str:
    """The first fenced code block under a heading, without its fences.

    Located by heading rather than by line number: both source documents are
    edited by hand and by other people, and a line offset would go silently wrong where a
    missing heading goes loudly wrong.
    """
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; the card cannot be assembled without it")
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        raise SystemExit(f"{path}: no heading {heading!r}; the card source has moved") from None
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.startswith("```"):
            if line.startswith("## "):
                break
            continue
        ticks = line[: len(line) - len(line.lstrip("`"))]
        info = line[len(ticks) :].strip()
        if language and info != language:
            raise SystemExit(
                f"{path}: the first fence under {heading!r} is "
                f"{info or 'unlabelled'}, expected {language}"
            )
        for j in range(i + 1, len(lines)):
            if lines[j].rstrip() == ticks:
                return "\n".join(lines[i + 1 : j]).strip("\n")
        raise SystemExit(f"{path}: the fence opened at line {i + 1} is never closed")
    raise SystemExit(f"{path}: no fenced block under {heading!r}")


def check_front_matter(block: str) -> list[str]:
    """Everything about the front-matter that can be wrong and would not be noticed.

    Returns the failures rather than raising on the first, so one run reports the whole
    list. The four declarations checked here are the four the release was reshaped around:
    four configs and no fifth, a default so the quick start works, and counts that match
    what the files hold.
    """
    import yaml

    failures: list[str] = []
    try:
        card = yaml.safe_load(block)
    except yaml.YAMLError as error:
        return [f"the front-matter does not parse as YAML: {error}"]
    if not isinstance(card, dict):
        return ["the front-matter does not parse to a mapping"]

    expected = list(RELEASE_SPLITS)
    configs = card.get("configs")
    if not isinstance(configs, list):
        failures.append("no `configs` block; without it the dataset viewer cannot work")
        configs = []
    names = [c.get("config_name") for c in configs if isinstance(c, dict)]
    if names != expected:
        failures.append(f"configs are {names}, expected exactly {expected}")
    if "documents" in names:
        failures.append(
            "a `documents` config is declared; the document text belongs to the parent "
            "and is not reshipped, and its data_files would match nothing"
        )
    defaults = [c.get("config_name") for c in configs if isinstance(c, dict) and c.get("default")]
    if defaults != [DEFAULT_CONFIG]:
        failures.append(
            f"`default: true` is on {defaults or 'no config'}, expected [{DEFAULT_CONFIG!r}]; "
            "without it the card's quick start fails and the reader lands on `passages`"
        )
    for entry in configs:
        if not isinstance(entry, dict):
            continue
        name = entry.get("config_name")
        want = RELEASE_SPLITS.get(name)
        if want is None:
            continue
        files = entry.get("data_files") or []
        got = tuple(f.get("split") for f in files if isinstance(f, dict))
        if got != want:
            failures.append(f"{name}: data_files splits are {list(got)}, expected {list(want)}")
        for spec in files:
            if not isinstance(spec, dict):
                continue
            path = str(spec.get("path", ""))
            if not path.startswith(f"{name}/{spec.get('split')}-"):
                failures.append(
                    f"{name}/{spec.get('split')}: path {path!r} does not match the staged "
                    f"layout <config>/<split>-<NNNNN>-of-<NNNNN>.parquet"
                )

    info = card.get("dataset_info")
    if not isinstance(info, list):
        failures.append("no `dataset_info`; the measured counts and the schema are not declared")
        info = []
    if [i.get("config_name") for i in info if isinstance(i, dict)] != expected:
        failures.append(
            f"dataset_info names {[i.get('config_name') for i in info]}, expected {expected}"
        )
    declared_total = 0
    for entry in info:
        if not isinstance(entry, dict):
            continue
        name = entry.get("config_name")
        want_rows = RELEASE_ROWS.get(name)
        if want_rows is None:
            continue
        if not entry.get("features"):
            failures.append(f"{name}: dataset_info declares no features")
        got_rows = {
            s.get("name"): s.get("num_examples")
            for s in entry.get("splits") or []
            if isinstance(s, dict)
        }
        declared_total += sum(v for v in got_rows.values() if isinstance(v, int))
        if got_rows != want_rows:
            failures.append(
                f"{name}: declared num_examples {got_rows} do not match the measured "
                f"{want_rows}; `datasets` verifies this at BASIC_CHECKS, so a difference of "
                f"one row raises NonMatchingSplitsSizesError for every user"
            )
    if info and declared_total != RELEASE_TOTAL_ROWS:
        failures.append(
            f"the declared counts sum to {declared_total:,}, not the measured "
            f"{RELEASE_TOTAL_ROWS:,} that `size_categories` is derived from"
        )

    if card.get("license") != RELEASE_LICENSE:
        failures.append(
            f"license is {card.get('license')!r}, expected {RELEASE_LICENSE!r}: share-alike on "
            "the parent fixes this value rather than leaving it a preference"
        )
    if card.get("size_categories") != [RELEASE_SIZE_CATEGORY]:
        failures.append(
            f"size_categories is {card.get('size_categories')}, expected "
            f"[{RELEASE_SIZE_CATEGORY!r}] for {RELEASE_TOTAL_ROWS:,} rows"
        )
    return failures


def check_body(body: str) -> list[str]:
    """The card body has to be complete on its own, and one snippet has to be the right one.

    A reader who copies the rebuild snippet out of the card and gets it wrong gets text
    that is nearly right, which is the failure mode this release most has to avoid. The
    dict-access form is the tested one; attribute access raises on a `datasets` row.
    """
    failures: list[str] = []
    required = {
        "the parent collection": "trec-ragtime/ragtime2",
        "a load_dataset example": "load_dataset(",
        "the rebuild rule, first member": 'parts = [members[0]["text"]]',
        "the rebuild rule, the adjacency test": 'if cur["char_start"] > prev["char_end"]:',
        "the rebuild rule, the join": 'text = "".join(parts)',
        "the English rendering rule": '" ".join(',
        "share-alike on the parent": "CC-BY-SA-4.0",
        "the NLLB licence disclosure": "CC-BY-NC-4.0",
        "the OPUS-MT licences": "Apache-2.0",
    }
    for what, needle in required.items():
        if needle not in body:
            failures.append(f"the body does not carry {what} ({needle!r})")
    for name in RELEASE_SPLITS:
        if f"`{name}`" not in body:
            failures.append(f"the body never names the `{name}` config")
    for wrong in ("prev.char_end", "cur.char_start", "members[0].text", "passage.sentence_ids"):
        if wrong in body:
            failures.append(
                f"the body uses attribute access ({wrong!r}); a `datasets` row is a dict "
                "and attribute access raises AttributeError"
            )
    if "`documents` config" in body:
        failures.append("the body still refers to a `documents` config; the release ships four")
    return failures


def staged_rows(staging: Path) -> dict[tuple[str, str], tuple[int, int, int]]:
    """Rows per (config, split) read from the footers of whatever is already staged.

    Returns (files present, files the names say there are, rows). Completeness comes from
    the `-of-NNNNN` in the shard name rather than from the plan, so this works before a
    plan exists and after one has been superseded.
    """
    found: dict[tuple[str, str], tuple[int, int, int]] = {}
    for config, splits in RELEASE_SPLITS.items():
        for split in splits:
            paths = sorted((staging / config).glob(f"{split}-*.parquet"))
            if not paths:
                continue
            rows = sum(footer(path)["rows"] for path in paths)
            totals = set()
            for path in paths:
                parts = path.stem.split("-of-")
                totals.add(int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0)
            expected = totals.pop() if len(totals) == 1 else 0
            found[(config, split)] = (len(paths), expected, rows)
    return found


def assemble_card(front_matter: str, body: str) -> str:
    return f"---\n{front_matter}\n---\n\n{body}\n"


def cmd_card(args: argparse.Namespace) -> int:
    staging = staging_dir(args)
    front_matter = fenced_block(args.front_matter, CARD_FRONT_MATTER_HEADING, "yaml")
    body = fenced_block(args.body, CARD_BODY_HEADING, "markdown")
    if args.repo_id and args.repo_id != CARD_REPO_PLACEHOLDER:
        if CARD_REPO_PLACEHOLDER not in body:
            raise SystemExit(
                f"--repo-id was given but the body never names {CARD_REPO_PLACEHOLDER!r}, "
                "so there is nothing to rewrite; check the body source before shipping"
            )
        body = body.replace(CARD_REPO_PLACEHOLDER, args.repo_id)

    print(f"front-matter  {args.front_matter}")
    print(f"body          {args.body}")
    print(f"staging       {staging}")
    print()
    print("--- front-matter, as it will be written ---")
    print(front_matter)
    print("--- end of front-matter ---")
    print()

    failures = check_front_matter(front_matter) + check_body(body)

    present = staged_rows(staging)
    print(f"{'config/split':<30}{'declared':>14}{'staged':>16}  files")
    for config, splits in RELEASE_SPLITS.items():
        for split in splits:
            declared = RELEASE_ROWS[config][split]
            entry = present.get((config, split))
            if entry is None:
                print(f"{config + '/' + split:<30}{declared:>14,}{'not staged':>16}  -")
                continue
            files, expected, rows = entry
            complete = expected and files == expected
            state = f"{files}/{expected or '?'}" + ("" if complete else "  partial")
            print(f"{config + '/' + split:<30}{declared:>14,}{rows:>16,}  {state}")
            if complete and rows != declared:
                failures.append(
                    f"{config}/{split}: the staged files hold {rows:,} rows against the "
                    f"{declared:,} the card declares"
                )
            elif not complete and rows > declared:
                failures.append(
                    f"{config}/{split}: {files} staged file(s) already hold {rows:,} rows, "
                    f"more than the {declared:,} the card declares"
                )
    print()

    if failures:
        print(f"{len(failures)} problem(s) with the card; nothing was written:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("front-matter and body check out")

    card = assemble_card(front_matter, body)
    if args.check_only:
        print(f"check only: {staging / 'README.md'} was not written")
        return 0
    staging.mkdir(parents=True, exist_ok=True)
    target = staging / "README.md"
    temp = target.with_suffix(".md.tmp")
    temp.write_text(card, encoding="utf-8")
    os.replace(temp, target)
    print(f"wrote {target}  ({len(card.encode('utf-8')):,} bytes, {card.count(chr(10)) + 1} lines)")
    print(f"first line: {card.splitlines()[0]!r}")
    return 0



# --------------------------------------------------------------------------- #
# push: the transfer, on the login node
# --------------------------------------------------------------------------- #


def token() -> str:
    for name in TOKEN_ENV:
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit(
        f"no access token in the environment; export one of {', '.join(TOKEN_ENV)}. "
        "It is never read from a file in this repository and never written to one."
    )


def cmd_push(args: argparse.Namespace) -> int:
    """Transfer the staged folder, skipping whatever the Hub already holds.

    Two layers of resume, and both are wanted. The coarse one is here: the repository is
    listed, every staged file is compared against the copy already there by checksum where
    the Hub exposes one and by size otherwise, and the matches are excluded. A rerun after
    a completed transfer therefore reads no bytes at all. The fine one belongs to the
    client: what is left goes to a single `upload_folder` call, which with `hf_xet`
    installed streams the files through a worker pool, hashes them in the same read pass,
    commits them in several batches to stay under the server's limits, and deduplicates
    anything that already arrived, so an interrupted transfer resumes inside a file.

    On the choice of call: `upload_large_folder` is the name that sounds right and the
    installed client deprecates it in favour of `upload_folder`, which is the multi-commit,
    interruption-tolerant path now. Parallelism comes from that call's own workers and from
    having sharded the data, never from running several push processes: concurrent commits
    to one branch conflict, so a fan of uploaders would be slower and less safe than one.
    """
    if os.environ.get("SLURM_JOB_ID") and not args.dry_run:
        raise SystemExit(
            "this is the transfer stage and compute nodes have no route to the internet; "
            "run it on the login node. Only the conversion belongs in a batch job."
        )
    if os.environ.get("HF_HUB_OFFLINE") not in (None, "", "0") and not args.dry_run:
        raise SystemExit(
            "HF_HUB_OFFLINE is set in this shell, which would make every request fail with a "
            "misleading error; unset it before transferring."
        )

    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    staging = staging_dir(args)
    plan = read_plan(staging)
    entries = staged(staging, plan)
    card = staging / "README.md"
    if card.is_file():
        entries.append(
            {
                "path": "README.md",
                "local": card,
                "config": "-",
                "split": "-",
                "planned_rows": 0,
                "present": True,
                "bytes": card.stat().st_size,
                "rows": None,
                "sha256": None,
            }
        )

    print(f"staging     {staging}")
    print(f"repository  {args.repo_id}  (dataset, revision {args.revision or 'main'})")
    print()

    missing = [e["path"] for e in entries if not e["present"]]
    if not card.is_file():
        missing.append("README.md")
    print(f"{'file':<46}{'size':>14}  state")
    for entry in entries:
        mark = "staged" if entry["present"] else "NOT BUILT"
        size = human(entry["bytes"]) if entry["present"] else "-"
        print(f"{entry['path']:<46}{size:>14}  {mark}")
    if not card.is_file():
        print(f"{'README.md':<46}{'-':>14}  NOT WRITTEN — run the `card` subcommand")
    print(f"{'total staged':<46}{human(sum(e['bytes'] for e in entries)):>14}")

    if missing:
        print()
        print(f"{len(missing)} file(s) missing: " + ", ".join(missing))
        if not args.dry_run:
            raise SystemExit(
                "refusing to upload a partial release; finish the conversion and write the card"
            )

    # A dry run against a public repository needs no credentials, so it stays usable before
    # anyone has exported a token. Anything that writes requires one.
    have_token = any(os.environ.get(name) for name in TOKEN_ENV)
    api = HfApi(token=token() if (have_token or not args.dry_run) else None)
    remote: dict[str, Any] = {}
    exists = True
    try:
        for item in api.list_repo_tree(
            args.repo_id, repo_type="dataset", recursive=True, revision=args.revision
        ):
            if getattr(item, "size", None) is not None:
                remote[item.path] = item
    except RepositoryNotFoundError:
        exists = False
        print("\nthe repository does not exist yet; a real push would create it")
    except Exception as error:  # noqa: BLE001 — a network failure is not a reason to guess
        print(f"\ncould not read the repository ({type(error).__name__}); assuming it is empty")

    unchanged: list[str] = []
    for entry in entries:
        item = remote.get(entry["path"])
        if item is None:
            continue
        lfs = getattr(item, "lfs", None)
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if digest and entry["sha256"]:
            if digest == entry["sha256"]:
                unchanged.append(entry["path"])
        elif item.size == entry["bytes"]:
            unchanged.append(entry["path"])

    outgoing = [e for e in entries if e["present"] and e["path"] not in unchanged]
    outgoing_bytes = sum(e["bytes"] for e in outgoing)
    print()
    print(f"already on the Hub and unchanged: {len(unchanged)} file(s)")
    print(f"would transfer: {len(outgoing)} file(s), {human(outgoing_bytes)}")
    for entry in outgoing:
        print(f"  {entry['path']:<44}{human(entry['bytes']):>14}")

    try:
        import hf_xet  # noqa: F401

        xet = True
    except ImportError:
        xet = False
    print()
    print(
        "transfer client: hf_xet present, so upload_folder streams and commits in batches"
        if xet
        else "transfer client: hf_xet is NOT installed, so upload_folder falls back to a "
        "single commit and loses resume-inside-a-file; install it before a large upload"
    )
    print("nothing under _state/ is in the upload list; the list comes from the plan")

    if args.dry_run:
        print("\ndry run: no repository was created and nothing was uploaded")
        return 0
    if not outgoing:
        print("\nnothing to do")
        return 0

    if not exists:
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(staging),
        revision=args.revision,
        commit_message=args.commit_message,
        allow_patterns=[e["path"] for e in outgoing],
    )
    print(f"\ntransferred {len(outgoing)} file(s), {human(outgoing_bytes)}")
    return 0


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--staging", help="staging directory (or DS_STAGING)")
    parser.add_argument("--corpus-dir", help="corpus build directory, bypassing discovery")
    parser.add_argument("--family", default=os.environ.get("DS_FAMILY", "e2e"))
    parser.add_argument("--chunker", help="corpus build key (or DS_CHUNKER)")
    parser.add_argument("--final", help="reconciled build to publish (or DS_FINAL)")
    parser.add_argument("--pack", help="passage grouping to publish (or DS_PACK)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="what the sources weigh and what the output should weigh")
    p.add_argument("--out", help="also write the footer report as JSON here")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("plan", help="resolve every shard's row range, once")
    p.add_argument(
        "--only",
        nargs="*",
        choices=[s.name for s in CONFIGS],
        help="plan only these configs (the rest keep whatever the plan already had)",
    )
    p.add_argument(
        "--throttle",
        type=int,
        default=8,
        help="concurrent array tasks to suggest in the printed sbatch line",
    )
    p.add_argument(
        "--no-adopt",
        action="store_true",
        help="re-cut every config from the source, ignoring files already in staging",
    )
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("convert", help="write exactly one shard")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--force", action="store_true", help="rebuild even if a receipt exists")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("manifest", help="check what is staged against the plan")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("card", help="assemble README.md into the staging directory")
    p.add_argument(
        "--front-matter",
        type=Path,
        default=CARD_FRONT_MATTER,
        help="document holding the settled YAML block (default: docs/release/card-front-matter.md)",
    )
    p.add_argument(
        "--body",
        type=Path,
        default=CARD_BODY,
        help="document holding the card body (default: docs/HUGGINGFACE.md)",
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help=f"rewrite the repository id the examples use (default: {CARD_REPO_PLACEHOLDER})",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="run every check and print the front-matter, but write nothing",
    )
    p.set_defaults(func=cmd_card)

    p = sub.add_parser("push", help="transfer the staged folder to the Hub (login node)")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--private", action="store_true")
    p.add_argument("--commit-message", default="Add the sentence, translation and passage tables")
    p.add_argument("--dry-run", action="store_true", help="report what would move, upload nothing")
    p.set_defaults(func=cmd_push)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
