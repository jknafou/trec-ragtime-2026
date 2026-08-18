#!/usr/bin/env python3
"""Language mix of the documents each Task-2 run retrieves, at ranks 1, 3, 5 and 10.

The three Task-2 runs differ only in which rendering of the collection is indexed and
searched: the native text, our low-tier Opus-MT translation, or our high-tier
NLLB-200-3.3B translation. Everything below is ordered that way — native first, then the
two translations in ascending quality. The query is the same English request in all three.
The question is whether searching a translated index pulls the result set toward English
and away from the other three languages.

Inputs:

  * the three submitted run files, ``submission/task2/T2_run_{1,2,3}.txt``, in the
    six-column TREC format ``topic Q0 document rank score run-name``;
  * a document-id -> language lookup, because the document id carries no language of its
    own (the leading UUID is a batch identifier shared across languages).

The lookup comes from the corpus document table produced during preprocessing, which
carries one ``lang`` value per document. Scanning only its ``document_id`` and ``lang``
columns is a single streaming pass. The resolved ids are cached to a small TSV beside this
script, so a second run needs no corpus access at all.

Usage (compute node; the corpus table lives on the shared filesystem):

    sbatch --partition=shared-cpu,public-cpu --cpus-per-task=4 --mem=16G --time=00:30:00 \
           --wrap="uv run python analysis/task2_language_mix.py submission"

Outputs:

    docs/figures/task2_language_mix.svg   grouped bar chart, one panel per rank cut-off
    analysis/task2_language_mix.md        the same numbers as markdown tables
    analysis/doc_languages.tsv            the cached id -> language lookup
    analysis/collection_languages.tsv     the language totals of the whole collection
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# The three Task-2 runs as (file, short name, chart label), in the order used everywhere:
# native text, then the low tier, then the high tier. They are named by what they search
# rather than by their run number: run numbering is not aligned across the three tasks, so
# a run number alone is ambiguous once a figure leaves this directory.
RUNS = [
    ("T2_run_1.txt", "native text", "native text"),
    ("T2_run_3.txt", "Opus-MT (low tier)", "Opus-MT (low tier)"),
    ("T2_run_2.txt", "NLLB (high tier)", "NLLB (high tier)"),
]

# Rank cut-offs. A run holds every document its retrieval loops surfaced, so depth is
# emergent rather than a fixed 1000; not every request reaches every cut-off, and the
# denominators printed with the tables record that.
CUTOFFS = [1, 3, 5, 10]

LANGS = ["en", "es", "ru", "zh"]
LANG_NAMES = {"en": "English", "es": "Spanish", "ru": "Russian", "zh": "Chinese"}

# One colour per index, held to the index rather than to a bar position.
COLOURS = {
    "native text": "#2F6BC0",
    "Opus-MT (low tier)": "#C5701F",
    "NLLB (high tier)": "#1B7A3E",
}

DEFAULT_DOCUMENTS_PARQUET = pathlib.Path(
    "/srv/beegfs/scratch/users/k/knafouj/ragtime-runs/corpus/e2e/8fbe879560a4"
    "/corpus-preprocess/documents.parquet"
)


def read_run(path: pathlib.Path) -> dict[str, list[str]]:
    """Return {topic: [document ids in rank order]} for one six-column TREC run file."""
    by_topic: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        topic, _q0, doc, rank, _score, _tag = line.split()
        by_topic[topic].append((int(rank), doc))
    return {t: [d for _, d in sorted(rows)] for t, rows in by_topic.items()}


def load_language_map(
    wanted: set[str], cache: pathlib.Path, documents_parquet: pathlib.Path
) -> tuple[dict[str, str], collections.Counter]:
    """Resolve every wanted document id to a language.

    Uses the cached TSV when it already covers every wanted id; otherwise streams the
    corpus document table once and rewrites the cache. The second return value is the
    language distribution of the whole collection, counted during the same pass and cached
    beside the lookup so that a cached re-run produces the identical tables.
    """
    totals_cache = cache.with_name("collection_languages.tsv")

    def read_totals() -> collections.Counter:
        if not totals_cache.exists():
            return collections.Counter()
        rows = (ln.split("\t") for ln in totals_cache.read_text().splitlines() if ln.strip())
        return collections.Counter({lg: int(n) for lg, n in rows})

    if cache.exists():
        cached = dict(
            line.split("\t") for line in cache.read_text().splitlines() if line.strip()
        )
        if wanted <= set(cached):
            print(f"language lookup: {len(wanted)} ids served from {cache}")
            return {k: cached[k] for k in wanted}, read_totals()

    import pyarrow.parquet as pq

    print(f"language lookup: scanning {documents_parquet}", flush=True)
    resolved: dict[str, str] = {}
    collection = collections.Counter()
    reader = pq.ParquetFile(documents_parquet)
    for batch in reader.iter_batches(columns=["document_id", "lang"], batch_size=100_000):
        ids = batch.column("document_id").to_pylist()
        langs = batch.column("lang").to_pylist()
        collection.update(langs)
        for doc_id, lang in zip(ids, langs):
            if doc_id in wanted:
                resolved[doc_id] = lang

    missing = wanted - set(resolved)
    if missing:
        raise SystemExit(f"{len(missing)} document ids have no language, e.g. {sorted(missing)[:3]}")

    cache.write_text("".join(f"{k}\t{v}\n" for k, v in sorted(resolved.items())))
    totals_cache.write_text("".join(f"{lg}\t{n}\n" for lg, n in sorted(collection.items())))
    print(f"language lookup: resolved {len(resolved)} ids, cached to {cache}")
    return resolved, collection


def language_mix(
    run: dict[str, list[str]], lang_of: dict[str, str], cutoff: int
) -> tuple[dict[str, float], int, int]:
    """Share of retrieved documents by language down to ``cutoff``.

    Returns the percentages, the number of documents pooled over all topics, and the
    number of topics that actually reach the cut-off.
    """
    counts = collections.Counter()
    topics_reaching = 0
    for docs in run.values():
        head = docs[:cutoff]
        if len(docs) >= cutoff:
            topics_reaching += 1
        counts.update(lang_of[d] for d in head)
    total = sum(counts.values())
    return {lg: 100.0 * counts[lg] / total for lg in LANGS}, total, topics_reaching


def write_tables(results, collection, out: pathlib.Path) -> None:
    lines = [
        "# Language mix of retrieved documents",
        "",
        "Share of the documents each Task-2 run retrieves, by the language the document is",
        "written in, pooled over all 103 requests. The three runs differ only in which",
        "rendering of the collection is indexed and searched; the request is the same English",
        "text in all three, and every ranked list is of original document ids whatever was",
        "searched. Rows run native text, then the low-tier Opus-MT translation, then the",
        "high-tier NLLB translation.",
        "",
        "Retrieval depth is emergent rather than a fixed cut-off — a run holds the documents",
        "its loops actually surfaced — so the deeper rows are computed over slightly fewer",
        "requests. The last column records how many.",
        "",
    ]
    if collection:
        total = sum(collection.values())
        share = ", ".join(
            f"{LANG_NAMES[lg]} {100.0 * collection[lg] / total:.1f}%" for lg in LANGS
        )
        lines += [
            "The collection itself is balanced across the four languages",
            f"({share}), so any departure from an even quarter is",
            "something retrieval did, not something the collection imposed.",
            "",
        ]
    for cutoff in CUTOFFS:
        lines += [
            f"## Down to rank {cutoff}",
            "",
            "| Index searched | " + " | ".join(LANG_NAMES[lg] for lg in LANGS)
            + " | Documents | Requests reaching this rank |",
            "| --- | " + " | ".join("---:" for _ in LANGS) + " | ---: | ---: |",
        ]
        for _, label, _legend in RUNS:
            pct, total, reaching = results[label][cutoff]
            cells = " | ".join(f"{pct[lg]:.1f}%" for lg in LANGS)
            lines.append(f"| {label} | {cells} | {total} | {reaching} of 103 |")
        lines.append("")
    lines += [
        "---",
        "",
        "The same numbers are drawn in `docs/figures/task2_language_mix.svg`. Both are",
        "produced by `analysis/task2_language_mix.py`, which reads the submitted run files",
        "and the two lookups cached beside it, `analysis/doc_languages.tsv` for the language",
        "of each retrieved document and `analysis/collection_languages.tsv` for the totals of",
        "the whole collection. Both are rewritten from the corpus document table whenever a",
        "wanted document id is missing from the cache.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def draw(results, out: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.7,
            "svg.fonttype": "path",  # text as outlines: no font dependency for a reader
        }
    )

    fig, axes = plt.subplots(1, len(CUTOFFS), figsize=(9.2, 3.1), sharey=True)
    x = range(len(LANGS))
    width = 0.26

    for panel, (ax, cutoff) in enumerate(zip(axes, CUTOFFS)):
        for i, (_, label, legend) in enumerate(RUNS):
            pct, _total, _reaching = results[label][cutoff]
            offset = (i - 1) * width
            ax.bar(
                [v + offset for v in x],
                [pct[lg] for lg in LANGS],
                width,
                label=legend,
                color=COLOURS[label],
                edgecolor="white",
                linewidth=0.4,
            )
        # The collection holds the same number of documents in each language, so a quarter
        # is what a language-blind retriever would return. Labelled once, for the legend.
        ax.axhline(
            25.0,
            color="#888888",
            linewidth=0.8,
            linestyle=(0, (3, 3)),
            zorder=0,
            label="an even quarter of the collection" if panel == 0 else None,
        )
        reaching = sorted(results[name][cutoff][2] for _, name, _legend in RUNS)
        span = f"{reaching[0]}" if reaching[0] == reaching[-1] else f"{reaching[0]}-{reaching[-1]}"
        ax.set_title(f"top {cutoff}   ({span} of 103 requests)", fontsize=8.5, pad=6)
        ax.set_xticks(list(x))
        ax.set_xticklabels([LANG_NAMES[lg] for lg in LANGS], rotation=30, ha="right")
        ax.set_ylim(0, 44)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.tick_params(length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    axes[0].set_ylabel("share of retrieved documents (%)")
    # Matplotlib lists lines before bars; put the three indexes first and the reference last.
    handles, labels = axes[0].get_legend_handles_labels()
    order = [i for i, lb in enumerate(labels) if not lb.startswith("an even")]
    order += [i for i, lb in enumerate(labels) if lb.startswith("an even")]
    axes[0].legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.34),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.30, wspace=0.12)
    fig.savefig(out, dpi=200)  # the format follows the suffix; svg for the README, png to eyeball
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission", type=pathlib.Path, help="the submission directory")
    ap.add_argument("--documents-parquet", type=pathlib.Path, default=DEFAULT_DOCUMENTS_PARQUET)
    ap.add_argument("--lang-cache", type=pathlib.Path, default=REPO / "analysis" / "doc_languages.tsv")
    ap.add_argument("--figure-dir", type=pathlib.Path, default=REPO / "docs" / "figures")
    ap.add_argument("--table", type=pathlib.Path, default=REPO / "analysis" / "task2_language_mix.md")
    args = ap.parse_args()

    runs = {
        label: read_run(args.submission / "task2" / file) for file, label, _legend in RUNS
    }
    for label, run in runs.items():
        depths = sorted(len(v) for v in run.values())
        print(
            f"{label}: {len(run)} requests, {sum(depths)} rows, "
            f"depth min {depths[0]} median {depths[len(depths) // 2]} max {depths[-1]}"
        )

    wanted = {d for run in runs.values() for docs in run.values() for d in docs}
    lang_of, collection = load_language_map(wanted, args.lang_cache, args.documents_parquet)
    if collection:
        total = sum(collection.values())
        print("collection: " + " ".join(f"{lg}={collection[lg]}" for lg in sorted(collection)))
        print("collection shares: " + " ".join(
            f"{lg}={100.0 * collection[lg] / total:.2f}%" for lg in LANGS
        ))

    results = {
        label: {c: language_mix(run, lang_of, c) for c in CUTOFFS} for label, run in runs.items()
    }
    for label in results:
        for c in CUTOFFS:
            pct, total, reaching = results[label][c]
            print(
                f"{label:<14} top {c:>2}: "
                + "  ".join(f"{lg} {pct[lg]:5.1f}%" for lg in LANGS)
                + f"   n={total} topics={reaching}"
            )

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    write_tables(results, collection, args.table)
    draw(results, args.figure_dir / "task2_language_mix.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
