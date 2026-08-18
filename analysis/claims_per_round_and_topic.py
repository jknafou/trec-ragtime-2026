"""Claims per round, and citations per report, for the three reading strategies.

Panel (a) comes from the run trace. Every retrieval loop the pipeline ran left one JSON record
under ``topics/<topic>/rag_loop/<nugget>.jsonl``, carrying the round it belonged to in the field
``round`` and the number of claims the loop managed to ground in a passage in the field
``claims_committed``. Summing ``claims_committed`` over the loops that share a topic and a round
gives the unit plotted here: how much grounded evidence one round of work produced.

Panel (b) comes from the submitted Task 1 files. Each line is one report: a list of sentences,
each with a map of the documents it cites. Two quantities are read from it -- how many cited
sentences a report contains, and how many documents a sentence cites.

The three strategies differ only in which rendering of a retrieved passage the generator was
shown: the native text, our low-tier Opus-MT translation of it, or our high-tier NLLB-200-3.3B
translation of it. Everything else, including the index that was searched, was held identical.
Everything below is presented in that order -- native text, then the low tier, then the high
tier -- which is ascending translation quality and matches the pipeline figure.

Run it on a compute node, never on the login node:

    sbatch --partition=shared-cpu,public-cpu --cpus-per-task=4 --mem=16G --time=00:30:00 \
           --wrap="uv run python analysis/claims_per_round_and_topic.py submission"

Outputs:

    docs/figures/claims_per_round.svg        the per-round density of grounded claims
    docs/figures/report_citations.svg        cited sentences, and documents per sentence
    analysis/claims_per_round_and_topic.md   the same numbers as markdown tables
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Default location of the run trace. Six cell trees live here, one per configuration; only the
# three that vary the reading rendering are read below.
DEFAULT_TRACE = pathlib.Path("/srv/beegfs/scratch/users/k/knafouj/ragtime-runs")

# One row per reading strategy, in the order used everywhere: native text, then the low tier,
# then the high tier. Each row is the label a reader sees, the colour, the directory in the run
# trace, and the Task 1 file that strategy produced. The run numbering is not a strategy
# ordering -- run 1 of Task 1 is the NLLB reading, run 3 is the native reading -- so everything
# downstream is keyed by strategy and never by run number.
STRATEGIES = [
    ("native text", "#2F6BC0", "e2e-original__original__seed0", "T1_run_3.jsonl"),
    ("Opus-MT (low tier)", "#C5701F", "e2e-omt-weak__omt_opus__seed0", "T1_run_2.jsonl"),
    ("NLLB (high tier)", "#1B7A3E", "e2e-omt__omt__seed0", "T1_run_1.jsonl"),
]

# The column head shared with the other tables that split by this axis.
AXIS = "Rendering read"

GRID = {"color": "#D8D8D8", "linewidth": 0.6}


def read_rounds(tree: pathlib.Path) -> tuple[list[int], dict]:
    """Claims committed in each (topic, round), plus a few counts worth reporting.

    Returns the per-round totals and a diagnostics dict. A round that ran and grounded nothing
    is a real zero and is kept; dropping it would flatter the distribution.
    """
    per_round: collections.Counter[tuple[str, int]] = collections.Counter()
    diag = collections.Counter()
    for path in sorted(tree.glob("topics/*/rag_loop/*.jsonl")):
        topic = path.parent.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            diag["loops"] += 1
            diag[f"status:{rec.get('status')}"] += 1
            committed = int(rec.get("claims_committed", 0))
            if committed == 0:
                diag["loops_with_no_claim"] += 1
            # A record without a round cannot be placed on the axis; count it rather than
            # silently absorbing it into round 0.
            if "round" not in rec:
                diag["loops_missing_round"] += 1
                continue
            per_round[(topic, int(rec["round"]))] += committed
            diag[f"round:{rec['round']}"] += committed
    diag["topics"] = len({t for t, _ in per_round})
    return sorted(per_round.values()), dict(diag)


def read_reports(path: pathlib.Path) -> tuple[list[int], collections.Counter, list[str]]:
    """Cited sentences per report, the citation-count histogram, and any empty reports."""
    cited_per_report: list[int] = []
    citations_per_sentence: collections.Counter[int] = collections.Counter()
    empty: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        counts = [len(s.get("citations") or {}) for s in rec["responses"]]
        citations_per_sentence.update(counts)
        cited = [c for c in counts if c > 0]
        cited_per_report.append(len(cited))
        if not cited:
            n = len(counts)
            empty.append(f"{rec['metadata']['topic_id']} ({n} sentence{'' if n == 1 else 's'})")
    return sorted(cited_per_report), citations_per_sentence, empty


def kde(sample: list[int], grid: np.ndarray) -> np.ndarray:
    """Gaussian kernel density with Silverman's bandwidth, reflected at zero.

    The quantities plotted are counts, so they cannot be negative. Without the reflection term
    the estimate leaks probability mass below zero and the peak is pulled down; with it, the
    curve integrates to one over the non-negative half-line.
    """
    x = np.asarray(sample, dtype=float)
    spread = min(x.std(ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349)
    if spread <= 0:
        spread = max(x.std(ddof=1), 1e-6)
    h = 0.9 * spread * len(x) ** (-0.2)
    d = grid[:, None]
    main = np.exp(-0.5 * ((d - x[None, :]) / h) ** 2)
    mirrored = np.exp(-0.5 * ((d + x[None, :]) / h) ** 2)
    return (main + mirrored).sum(axis=1) / (len(x) * h * np.sqrt(2 * np.pi))


def summary(sample: list[int]) -> dict:
    a = np.asarray(sample, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return {
        "n": len(a),
        "median": float(np.median(a)),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def save(fig, out: pathlib.Path) -> None:
    """Write the figure, suppressing the timestamp so a re-run reproduces the same bytes."""
    fmt = out.suffix.lstrip(".")
    fig.savefig(out, format=fmt, metadata={"Date": None} if fmt == "svg" else None)
    plt.close(fig)


def style(ax) -> None:
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure_claims_per_round(data: dict, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    grid = np.linspace(0, 45, 601)
    for label, colour, _, _ in STRATEGIES:
        sample = data[label]
        ax.plot(grid, kde(sample, grid), color=colour, linewidth=2.0,
                label=f"{label}  (n = {len(sample)} rounds)")
    ax.set_xlim(0, 45)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Claims grounded in a passage during one round of one request")
    ax.set_ylabel("Share of rounds (density)")
    ax.legend(frameon=False, loc="upper right", title=AXIS)
    style(ax)
    fig.tight_layout()
    save(fig, out)


def figure_report_citations(sentences: dict, hist: dict, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    left = axes[0]
    grid = np.linspace(0, 46, 601)
    for label, colour, _, _ in STRATEGIES:
        left.plot(grid, kde(sentences[label], grid), color=colour, linewidth=2.0, label=label)
    left.set_xlim(0, 46)
    left.set_ylim(bottom=0)
    left.set_xlabel("Cited sentences in one report")
    left.set_ylabel("Share of the 103 requests (density)")
    style(left)

    # The second quantity is all but constant -- almost every sentence cites exactly one
    # document -- so a density curve would be a spike carrying no information. The mark for it
    # is the discrete distribution itself, on a log axis so that the rare multi-citation
    # sentences are visible rather than invisible.
    right = axes[1]
    levels = [1, 2, 3]
    width = 0.26
    for i, (label, colour, _, _) in enumerate(STRATEGIES):
        counts = hist[label]
        total = sum(counts[k] for k in levels)
        shares = [100.0 * counts.get(k, 0) / total for k in levels]
        pos = [k + (i - 1) * width for k in levels]
        right.bar(pos, shares, width=width, color=colour, label=label)
        for p, s, k in zip(pos, shares, levels):
            # A zero count draws no bar, which is easy to misread as a missing measurement, so
            # it is labelled at the baseline instead.
            right.text(p, s * 1.35 if s > 0 else 0.011, f"{counts.get(k, 0)}", ha="center",
                       va="bottom", fontsize=8, color="#444444")
    right.set_yscale("log")
    right.set_ylim(0.01, 400)
    right.set_yticks([0.01, 0.1, 1, 10, 100])
    right.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    right.set_xticks(levels)
    right.set_xlabel("Documents cited by one sentence")
    right.set_ylabel("Share of cited sentences (log scale)")
    style(right)

    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               title=AXIS, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, out)


def md_table(title: str, unit: str, rows: dict) -> str:
    lines = [f"### {title}", "", f"Unit: {unit}.", "",
             f"| {AXIS} | n | Median | IQR | Min | Max | Mean |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, _, _, _ in STRATEGIES:
        s = rows[label]
        lines.append(
            f"| {label} | {s['n']} | {s['median']:.0f} | {s['q1']:.0f}–{s['q3']:.0f} | "
            f"{s['min']:.0f} | {s['max']:.0f} | {s['mean']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission", type=pathlib.Path,
                    help="the submission directory holding task1/, task2/, task3/")
    ap.add_argument("--trace", type=pathlib.Path, default=DEFAULT_TRACE,
                    help="root of the run trace")
    ap.add_argument("--figures", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures")
    ap.add_argument("--table", type=pathlib.Path,
                    default=pathlib.Path(__file__).with_suffix(".md"))
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Glyphs are written as outlines, matching the other figures in docs/figures/, so the
        # SVG carries no font dependency and renders the same wherever the README is read.
        "svg.fonttype": "path",
    })

    rounds, diagnostics = {}, {}
    sentences, hist, empties = {}, {}, {}
    for label, _, tree, report_file in STRATEGIES:
        rounds[label], diagnostics[label] = read_rounds(args.trace / tree)
        sentences[label], hist[label], empties[label] = read_reports(
            args.submission / "task1" / report_file)

    args.figures.mkdir(parents=True, exist_ok=True)
    figure_claims_per_round(rounds, args.figures / "claims_per_round.svg")
    figure_report_citations(sentences, hist, args.figures / "report_citations.svg")

    parts = [
        "# Claims per round, and citations per report",
        "",
        "Read from the run trace and from the submitted Task 1 files. Figures:",
        "`docs/figures/claims_per_round.svg`, `docs/figures/report_citations.svg`.",
        "",
        "All three runs searched the same native index and differ only in which rendering of a",
        "retrieved passage the model was shown. Rows run native text, then the low-tier Opus-MT",
        "translation, then the high-tier NLLB translation.",
        "",
        md_table("Claims grounded per round", "one round of one request",
                 {k: summary(v) for k, v in rounds.items()}),
        md_table("Cited sentences per report", "one of the 103 requests",
                 {k: summary(v) for k, v in sentences.items()}),
        "Requests whose report carries no cited sentence at all: "
        + "; ".join(f"{label} — {', '.join(empties[label]) or 'none'}"
                    for label, _, _, _ in STRATEGIES) + ".",
        "",
        "### Documents cited per sentence", "",
        f"| {AXIS} | Sentences | 1 document | 2 documents | 3 documents | Uncited |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _, _, _ in STRATEGIES:
        h = hist[label]
        cited = sum(v for k, v in h.items() if k > 0)
        parts.append(f"| {label} | {sum(h.values())} | {h.get(1, 0)} | {h.get(2, 0)} | "
                     f"{h.get(3, 0)} | {h.get(0, 0)} |")
        print(f"[{label}] cited sentences={cited} empty reports={empties[label]}")
    parts.append("")

    parts += ["### Retrieval loops behind those rounds", "",
              f"| {AXIS} | Loops | Answered | Unanswered | Errored | Grounding nothing |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, _, _, _ in STRATEGIES:
        d = diagnostics[label]
        parts.append(
            f"| {label} | {d['loops']} | {d.get('status:answered', 0)} | "
            f"{d.get('status:unanswered', 0)} | {d.get('status:error', 0)} | "
            f"{d.get('loops_with_no_claim', 0)} |")
    parts.append("")

    parts += ["### Claims grounded, by round index", "",
              "Round 0 is the initial nugget bank and runs no retrieval, so the loops start at\n"
              "round 1. A request stops when its coverage audit finds nothing left to ask, which is\n"
              "why the later columns rest on fewer requests.", "",
              f"| {AXIS} | " + " | ".join(f"R{r}" for r in range(1, 10)) + " |",
              "| --- | " + " | ".join(["---:"] * 9) + " |"]
    for label, _, _, _ in STRATEGIES:
        d = diagnostics[label]
        parts.append(f"| {label} | "
                     + " | ".join(str(d.get(f"round:{r}", 0)) for r in range(1, 10)) + " |")
    parts.append("")

    args.table.write_text("\n".join(parts), encoding="utf-8")

    for label, _, _, _ in STRATEGIES:
        print(f"[{label}] rounds={summary(rounds[label])}")
        print(f"[{label}] sentences={summary(sentences[label])}")
        print(f"[{label}] diagnostics={diagnostics[label]}")
    print(f"wrote {args.figures / 'claims_per_round.svg'}")
    print(f"wrote {args.figures / 'report_citations.svg'}")
    print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
