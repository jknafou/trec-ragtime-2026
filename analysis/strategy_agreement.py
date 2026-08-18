#!/usr/bin/env python3
"""How much do the three strategies agree with one another?

We submitted three strategies to each task. They differ in exactly one thing, and which
thing depends on the task:

  * In Task 2 the three runs differ in which rendering of the collection is indexed and
    searched — the native text, a low-tier Opus-MT translation, or our high-tier
    NLLB-200-3.3B translation. The request is the same English text in all three, and every
    ranked list is of original document ids whatever was searched.
  * In Task 1 the three runs all search the same native index and differ in which rendering
    of a passage the model reads once it has been retrieved.

So Task 2 measures what changing the search does to what is found, and Task 1 measures what
changing the reading does to what is cited, with the search held fixed. Task 1 is the closest
thing we have to a control, because we have no repeat run at a fixed setting: one seed per
strategy.

Measures, all defined per request and then averaged over requests:

  overlap        shared documents divided by the documents in either list (Jaccard). It is
                 used for the top of the ranking, for the whole ranked list, and for the
                 documents a report cites, so one number means one thing throughout.
  shared in top  the plain count of documents that appear in both top tens.
  Kendall's tau  computed over the documents the two lists have in common, on their ranks
                 within each list: +1 if the shared documents are in the same order, 0 if
                 the orders are unrelated, -1 if reversed. This separates "found the same
                 documents" from "put them in the same order".
  first document whether both runs rank the same document first.

Usage (compute node; reads nothing but the submission directory):

    sbatch --partition=shared-cpu,public-cpu --cpus-per-task=4 --mem=8G --time=00:15:00 \
           --wrap="uv run python analysis/strategy_agreement.py submission"

Outputs:

    docs/figures/strategy_agreement.svg   three panels: found, ordered, cited
    analysis/strategy_agreement.md        the same numbers as markdown tables
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# The three strategies, keyed by what varies. Run numbering is not aligned across tasks —
# run 1 is a translation-reading strategy in Task 1 and the native index in Task 2 — so
# every strategy is carried by name and the file's own self-declared run name is checked
# against it below. A figure labelled by run number would silently invert one of the tasks.
NATIVE, OPUS, NLLB = "native", "opus", "nllb"

# Everything below is presented in this order, everywhere: no translation, then the low
# tier, then the high tier. It is ascending translation quality and it matches the order
# the pipeline figure puts the three arms in.
ORDER = (NATIVE, OPUS, NLLB)

LABEL = {
    NATIVE: "native text",
    OPUS: "Opus-MT (low tier)",
    NLLB: "NLLB (high tier)",
}

# file, strategy, the run name the file declares about itself
TASK2_RUNS = [
    ("T2_run_1.txt", NATIVE, "mlir-original"),
    ("T2_run_3.txt", OPUS, "mlir-omt-weak"),
    ("T2_run_2.txt", NLLB, "mlir-omt"),
]
TASK1_RUNS = [
    ("T1_run_3.jsonl", NATIVE, "T1-e2e-original"),
    ("T1_run_2.jsonl", OPUS, "T1-e2e-omt-weak"),
    ("T1_run_1.jsonl", NLLB, "T1-e2e-omt"),
]

PAIRS = [(NATIVE, OPUS), (NATIVE, NLLB), (OPUS, NLLB)]

PAIR_LABEL = {
    (NATIVE, OPUS): "native vs Opus-MT",
    (NATIVE, NLLB): "native vs NLLB",
    (OPUS, NLLB): "Opus-MT vs NLLB",
}

PAIR_COLOUR = {
    (NATIVE, OPUS): "#2F6BC0",
    (NATIVE, NLLB): "#C5701F",
    (OPUS, NLLB): "#1B7A3E",
}

HEAD = 10  # the top of the ranking, where a reader of the run actually looks
DEEP = 100  # a conventional deeper cut; see the note in the markdown about depth


# --------------------------------------------------------------------------- reading


def read_task2(path: pathlib.Path, expect_run_name: str) -> dict[str, list[str]]:
    """Return {request: [document ids in rank order]} for one six-column TREC run file."""
    by_request: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    names = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        request, _q0, doc, rank, _score, run_name = line.split()
        names.add(run_name)
        by_request[request].append((int(rank), doc))
    if names != {expect_run_name}:
        raise SystemExit(f"{path.name}: expected run name {expect_run_name!r}, found {sorted(names)}")
    return {r: [d for _, d in sorted(rows)] for r, rows in by_request.items()}


def read_task1(path: pathlib.Path, expect_run_id: str) -> dict[str, list[str]]:
    """Return {request: [distinct documents cited, in order of first use]} for one report file."""
    cited: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        run_id = record["metadata"]["run_id"]
        if run_id != expect_run_id:
            raise SystemExit(f"{path.name}: expected run id {expect_run_id!r}, found {run_id!r}")
        seen: list[str] = []
        for sentence in record["responses"]:
            for doc in sentence.get("citations", {}):
                if doc not in seen:
                    seen.append(doc)
        cited[record["metadata"]["topic_id"]] = seen
    return cited


# --------------------------------------------------------------------------- measures


def overlap(a: list[str], b: list[str]) -> float:
    """Shared documents as a share of the documents in either list.

    Symmetric, and well behaved when the two lists have different lengths — which they do
    here, because retrieval depth is emergent rather than a fixed cut-off.
    """
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else float("nan")


def kendall_tau(a: list[str], b: list[str]) -> float | None:
    """Rank agreement over the documents the two lists share.

    Returns None when fewer than two documents are shared, since no pair of documents can
    then be put in a consistent or an inconsistent order. Ranks are distinct within a run
    (verified: no run ties two documents in one request), so there is no tie correction to
    make and the coefficient is the plain concordant-minus-discordant ratio.
    """
    rank_a = {d: i for i, d in enumerate(a)}
    rank_b = {d: i for i, d in enumerate(b)}
    shared = [(rank_a[d], rank_b[d]) for d in rank_a.keys() & rank_b.keys()]
    if len(shared) < 2:
        return None
    concordant = discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            direction = (shared[i][0] - shared[j][0]) * (shared[i][1] - shared[j][1])
            if direction > 0:
                concordant += 1
            elif direction < 0:
                discordant += 1
    return (concordant - discordant) / (len(shared) * (len(shared) - 1) / 2)


def three_way(lists: list[list[str]]) -> float:
    """Share of the pooled documents that all three lists hold."""
    sets = [set(x) for x in lists]
    union = sets[0] | sets[1] | sets[2]
    common = sets[0] & sets[1] & sets[2]
    return len(common) / len(union) if union else float("nan")


def summarise(values: list[float]) -> tuple[float, float, float]:
    """Mean and interquartile range of a per-request measure."""
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) >= 4 else [ordered[0], ordered[0], ordered[-1]]
    return statistics.fmean(ordered), quartiles[0], quartiles[2]


# --------------------------------------------------------------------------- analysis


def analyse_ranked_lists(runs: dict[str, dict[str, list[str]]]) -> dict:
    """Pairwise and three-way agreement between the three Task-2 ranked lists."""
    requests = sorted(set.intersection(*(set(r) for r in runs.values())))
    out: dict = {"requests": requests, "pairs": {}, "three_way": {}, "depth": {}}

    for strategy, run in runs.items():
        depths = sorted(len(run[r]) for r in requests)
        out["depth"][strategy] = {
            "min": depths[0],
            "median": depths[len(depths) // 2],
            "max": depths[-1],
            "total": sum(depths),
            "reaching_head": sum(1 for d in depths if d >= HEAD),
            "reaching_deep": sum(1 for d in depths if d >= DEEP),
        }

    for pair in PAIRS:
        a_run, b_run = runs[pair[0]], runs[pair[1]]
        head_overlap, full_overlap, shared_head, taus = [], [], [], []
        first_agree = 0
        for r in requests:
            a, b = a_run[r], b_run[r]
            head_overlap.append(overlap(a[:HEAD], b[:HEAD]))
            full_overlap.append(overlap(a[:DEEP], b[:DEEP]))
            shared_head.append(len(set(a[:HEAD]) & set(b[:HEAD])))
            tau = kendall_tau(a, b)
            if tau is not None:
                taus.append(tau)
            if a and b and a[0] == b[0]:
                first_agree += 1
        out["pairs"][pair] = {
            "head_overlap": summarise(head_overlap),
            "full_overlap": summarise(full_overlap),
            "shared_head": statistics.fmean(shared_head),
            "tau": summarise(taus),
            "tau_values": taus,
            "tau_requests": len(taus),
            "first_agree": first_agree,
            "head_values": head_overlap,
            "full_values": full_overlap,
        }

    for name, cut in (("head", HEAD), ("full", DEEP)):
        values = [three_way([runs[s][r][:cut] for s in ORDER]) for r in requests]
        out["three_way"][name] = statistics.fmean(values)
    return out


def analyse_citations(runs: dict[str, dict[str, list[str]]]) -> dict:
    """Pairwise and three-way agreement between the documents the three reports cite."""
    requests = sorted(set.intersection(*(set(r) for r in runs.values())))
    out: dict = {"requests": requests, "pairs": {}, "cited": {}}

    for strategy, run in runs.items():
        counts = [len(run[r]) for r in requests]
        # Share of a report's cited documents that no other strategy cites anywhere for
        # that request — the plainest reading of "how much of this evidence is its own".
        others = [s for s in runs if s != strategy]
        unique = []
        for r in requests:
            mine = set(run[r])
            if not mine:
                continue
            theirs = set(runs[others[0]][r]) | set(runs[others[1]][r])
            unique.append(len(mine - theirs) / len(mine))
        out["cited"][strategy] = {
            "mean": statistics.fmean(counts),
            "median": statistics.median(counts),
            "total_distinct": len({d for r in requests for d in run[r]}),
            "unique_share": statistics.fmean(unique),
        }

    for pair in PAIRS:
        values, shared_counts = [], []
        for r in requests:
            a, b = runs[pair[0]][r], runs[pair[1]][r]
            values.append(overlap(a, b))
            shared_counts.append(len(set(a) & set(b)))
        out["pairs"][pair] = {
            "overlap": summarise([v for v in values if v == v]),
            "values": values,
            "shared": statistics.fmean(shared_counts),
        }

    out["three_way"] = statistics.fmean(
        three_way([runs[s][r] for s in ORDER]) for r in requests
    )
    return out


# --------------------------------------------------------------------------- output


def write_tables(ranked: dict, cited: dict, out: pathlib.Path) -> None:
    n = len(ranked["requests"])
    deepest = max(d["max"] for d in ranked["depth"].values())
    lines = [
        "# Agreement between the three strategies",
        "",
        "Each task was answered three times. The three answers differ in one thing, and",
        "which thing depends on the task: in Task 2 the three runs search a different",
        "rendering of the collection (the native text, a low-tier Opus-MT translation, or",
        "a high-tier NLLB translation); in Task 1 all three search the same native index and differ",
        "only in which rendering of a retrieved passage the model reads. Both are measured",
        "over the same 103 requests, and every document id is the original one whatever was",
        "searched or read.",
        "",
        "Throughout, *overlap* means the documents two runs share divided by the documents",
        "in either of them, averaged over requests, with the interquartile range in",
        "brackets. *Kendall's tau* is computed on the documents a pair has in common, from",
        "their positions in each ranking: +1 is the same order, 0 is an unrelated order.",
        "",
        "## Task 2 — does changing the index change what is found, or only the order?",
        "",
        "| Comparison | Of the top 10, shared | Overlap, top 10 | Overlap, whole list | Order of shared documents | Same document ranked first |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in PAIRS:
        p = ranked["pairs"][pair]
        hm, hlo, hhi = p["head_overlap"]
        fm, flo, fhi = p["full_overlap"]
        tm, tlo, thi = p["tau"]
        lines.append(
            f"| {PAIR_LABEL[pair]} | {p['shared_head']:.1f} of 10 | "
            f"{hm:.2f} [{hlo:.2f}–{hhi:.2f}] | {fm:.2f} [{flo:.2f}–{fhi:.2f}] | "
            f"{tm:+.2f} [{tlo:+.2f} to {thi:+.2f}] | {p['first_agree']} of {n} |"
        )
    lines += [
        f"| **all three at once** | | **{ranked['three_way']['head']:.2f}** | "
        f"**{ranked['three_way']['full']:.2f}** | | |",
        "",
        "Retrieval depth here is emergent — a run holds the documents its loops actually",
        "surfaced, not a fixed thousand — and the deepest single ranked list in the whole",
        f"submission is {deepest} documents long. A cut at 100 therefore selects the entire",
        "list every time, so that column is the whole list and not a truncation. The lists",
        "also differ in length between runs, which by itself holds the overlap below 1.",
        "Per-run depth:",
        "",
        "| Index searched | Shortest | Median | Longest | Requests reaching rank 10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy in ORDER:
        d = ranked["depth"][strategy]
        lines.append(
            f"| {LABEL[strategy]} | {d['min']} | {d['median']} | {d['max']} | "
            f"{d['reaching_head']} of {n} |"
        )

    lines += [
        "",
        "## Task 1 — do the three reports rest on the same documents?",
        "",
        "All three of these runs searched the same native index. Any disagreement below is",
        "therefore produced downstream of retrieval: by what the model read, by which",
        "queries it chose to issue, and by its own run-to-run variation.",
        "",
        "| Comparison | Documents cited by both | Overlap of cited documents |",
        "| --- | ---: | ---: |",
    ]
    for pair in PAIRS:
        p = cited["pairs"][pair]
        m, lo, hi = p["overlap"]
        lines.append(f"| {PAIR_LABEL[pair]} | {p['shared']:.1f} | {m:.2f} [{lo:.2f}–{hi:.2f}] |")
    lines += [
        f"| **all three at once** | | **{cited['three_way']:.2f}** |",
        "",
        "| Rendering read | Documents cited per report | Cited by no other strategy |",
        "| --- | ---: | ---: |",
    ]
    for strategy in ORDER:
        c = cited["cited"][strategy]
        lines.append(
            f"| {LABEL[strategy]} | {c['mean']:.1f} | {100 * c['unique_share']:.0f}% |"
        )

    head_mean = statistics.fmean(ranked["pairs"][p]["head_overlap"][0] for p in PAIRS)
    full_mean = statistics.fmean(ranked["pairs"][p]["full_overlap"][0] for p in PAIRS)
    tau_mean = statistics.fmean(ranked["pairs"][p]["tau"][0] for p in PAIRS)
    cite_mean = statistics.fmean(cited["pairs"][p]["overlap"][0] for p in PAIRS)

    shared_head_mean = statistics.fmean(ranked["pairs"][p]["shared_head"] for p in PAIRS)
    first_mean = statistics.fmean(ranked["pairs"][p]["first_agree"] for p in PAIRS)
    unique_mean = statistics.fmean(cited["cited"][s]["unique_share"] for s in ORDER)

    lines += [
        "",
        "## What this shows",
        "",
        "Changing the index changes what is found, and it changes the order as well.",
        f"Two runs share about {full_mean:.0%} of their documents over the whole ranked list,",
        f"only {ranked['three_way']['full']:.0%} of the pooled documents are found by all",
        "three, and a system that searched a single rendering would never see most of what",
        "the other two surface.",
        "",
        "Two details are worth more than the headline overlap. First, agreement is *worse*",
        f"at the top of the ranking than over the whole list ({head_mean:.0%} against",
        f"{full_mean:.0%}); the three runs share on average {shared_head_mean:.1f} of their",
        f"top ten documents and put the same document in first place in {first_mean:.0f} of",
        f"{n} requests. Second, the shared documents are not merely reshuffled slightly: the",
        f"rank correlation over them averages {tau_mean:+.2f}, clearly positive but a long",
        "way from the +1 that a pure reordering of one common pool would give. The",
        "disagreement is therefore both about which documents are found and about where they",
        "are placed, and it is sharpest at the head, where a reader looks.",
        "",
        f"The same overlap measure on the Task 1 reports gives about {cite_mean:.0%} — the",
        "same magnitude — from three runs that all searched the *same* index; about",
        f"{unique_mean:.0%} of the documents a report cites are cited by neither of the other",
        "two. It is a caution, not a result: the Task 2 disagreement cannot be credited to",
        "the index alone, because holding the index fixed and varying only what the model",
        "reads moves the answer about as much. The two are not measuring quite the same",
        "object — Task 2 compares what was retrieved, Task 1 compares what survived into a",
        "report, and that extra selection step carries variation of its own — so this bounds",
        "the index effect loosely rather than isolating it.",
        "",
        "## What this does not show",
        "",
        "* Not quality. This track has no relevance judgements and none are coming, so",
        "  nothing here says which of the three found the better documents, or whether any",
        "  of them found good ones. Three strategies could disagree completely and all be",
        "  right, or all be wrong. Agreement is not accuracy.",
        "* Not an isolated index effect. Each strategy was run once, so there is no",
        "  repeat at a fixed setting to measure run-to-run variation against. The pipeline",
        "  is agentic: it issues its own queries over several rounds, and two runs of one",
        "  setting would not be expected to agree perfectly either. The Task 1 figure above",
        "  bounds how much of the Task 2 disagreement the index can be credited with, but it",
        "  does not separate the two cleanly.",
        "* Not independent runs. The native Task 2 run and the native Task 1 report come",
        "  from a single execution — searching the native index while reading native",
        "  passages is one system under two task descriptions — so those two files are not",
        "  independent observations of one another.",
        "* Not a ceiling. Overlap is measured over the documents each run actually",
        "  returned. Depth is emergent and unequal between runs, which by itself puts an",
        "  upper bound below 1 on any pair whose lists differ in length.",
        "",
        "---",
        "",
        "The same numbers are drawn in `docs/figures/strategy_agreement.svg`. Both are",
        "produced by `analysis/strategy_agreement.py`, which reads only the submitted run",
        "files.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def draw(ranked: dict, cited: dict, out: pathlib.Path) -> None:
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

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    width = 0.26

    def style(ax) -> None:
        ax.tick_params(length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    # Panel 1 — how many of the same documents each pair finds, at the top and overall.
    ax = axes[0]
    groups = [("top 10", "head_overlap", "head"), ("whole list", "full_overlap", "full")]
    for gi, (_name, key, three_key) in enumerate(groups):
        for i, pair in enumerate(PAIRS):
            mean, lo, hi = ranked["pairs"][pair][key]
            x = gi + (i - 1) * width
            ax.bar(
                x, mean, width, color=PAIR_COLOUR[pair], edgecolor="white", linewidth=0.4,
                label=PAIR_LABEL[pair] if gi == 0 else None,
            )
            # The middle half of the requests, so the reader sees that the mean is not a
            # summary of a tight distribution.
            ax.vlines(x, lo, hi, color="#33333366", linewidth=1.0)
            ax.text(x, hi + 0.02, f"{mean:.2f}", ha="center", fontsize=7.5, color="#333333")
        ax.hlines(
            ranked["three_way"][three_key], gi - 0.45, gi + 0.45,
            color="#444444", linestyle=(0, (3, 2)), linewidth=1.0,
            label="all three at once" if gi == 0 else None,
        )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("share of documents found by both")
    ax.set_title("Task 2: the same documents?", fontsize=8.5, pad=6)
    ax.set_ylim(0, 0.75)
    style(ax)

    # Panel 2 — and are the shared documents in the same order?
    ax = axes[1]
    data = [ranked["pairs"][pair]["tau_values"] for pair in PAIRS]
    box = ax.boxplot(
        data, widths=0.5, patch_artist=True, showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#888888", "linewidth": 0.8},
        capprops={"color": "#888888", "linewidth": 0.8},
    )
    for patch, pair in zip(box["boxes"], PAIRS):
        patch.set_facecolor(PAIR_COLOUR[pair])
        patch.set_alpha(0.75)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.4)
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.set_xticks(range(1, len(PAIRS) + 1))
    ax.set_xticklabels(["" for _ in PAIRS])
    ax.set_ylabel("rank agreement (+1 is the same order)")
    ax.set_title("Task 2: in the same order?", fontsize=8.5, pad=6)
    ax.set_ylim(-0.55, 1.15)
    for i, pair in enumerate(PAIRS):
        ax.text(
            i + 1, 1.06, f"{ranked['pairs'][pair]['tau'][0]:+.2f}",
            ha="center", fontsize=7.5, color="#333333",
        )
    style(ax)

    # Panel 3 — and do the reports built on top of it rest on the same documents?
    ax = axes[2]
    for i, pair in enumerate(PAIRS):
        mean, lo, hi = cited["pairs"][pair]["overlap"]
        ax.bar(i, mean, 0.55, color=PAIR_COLOUR[pair], edgecolor="white", linewidth=0.4)
        ax.vlines(i, lo, hi, color="#33333366", linewidth=1.0)
        ax.text(i, hi + 0.02, f"{mean:.2f}", ha="center", fontsize=7.5, color="#333333")
    ax.hlines(
        cited["three_way"], -0.5, len(PAIRS) - 0.5,
        color="#444444", linestyle=(0, (3, 2)), linewidth=1.0,
    )
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels(["" for _ in PAIRS])
    ax.set_xlim(-0.6, len(PAIRS) - 0.4)
    ax.set_ylabel("share of documents cited by both")
    ax.set_title("Task 1: the same evidence cited?", fontsize=8.5, pad=6)
    ax.set_ylim(0, 0.75)
    style(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=4,
        frameon=False, fontsize=8.5, handlelength=1.4, columnspacing=1.6, borderaxespad=0.0,
    )
    fig.text(
        0.5, 0.018,
        "bars are the mean over 103 requests; the vertical line spans the middle half of them, "
        "and the boxes show the same spread",
        ha="center", fontsize=7.5, color="#555555",
    )
    fig.subplots_adjust(left=0.078, right=0.995, top=0.90, bottom=0.25, wspace=0.32)
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Agreement between the three submitted strategies.")
    ap.add_argument("submission", type=pathlib.Path, help="the submission directory")
    ap.add_argument("--figure-dir", type=pathlib.Path, default=REPO / "docs" / "figures")
    ap.add_argument("--table", type=pathlib.Path, default=REPO / "analysis" / "strategy_agreement.md")
    args = ap.parse_args()

    task2 = {
        strategy: read_task2(args.submission / "task2" / file, run_name)
        for file, strategy, run_name in TASK2_RUNS
    }
    task1 = {
        strategy: read_task1(args.submission / "task1" / file, run_id)
        for file, strategy, run_id in TASK1_RUNS
    }

    for strategy, run in task2.items():
        depths = sorted(len(v) for v in run.values())
        print(
            f"task2 {strategy:<7}: {len(run)} requests, {sum(depths)} rows, "
            f"depth {depths[0]}-{depths[-1]} median {depths[len(depths) // 2]}"
        )
    for strategy, run in task1.items():
        counts = sorted(len(v) for v in run.values())
        print(
            f"task1 {strategy:<7}: {len(run)} reports, "
            f"documents cited {counts[0]}-{counts[-1]} median {counts[len(counts) // 2]}"
        )

    ranked = analyse_ranked_lists(task2)
    cite = analyse_citations(task1)

    print()
    for pair in PAIRS:
        p = ranked["pairs"][pair]
        print(
            f"task2 {PAIR_LABEL[pair]:<20} top10 {p['head_overlap'][0]:.3f}  "
            f"whole {p['full_overlap'][0]:.3f}  shared-in-10 {p['shared_head']:.2f}  "
            f"tau {p['tau'][0]:+.3f} (n={p['tau_requests']})  first-doc {p['first_agree']}"
        )
    print(f"task2 all three       top10 {ranked['three_way']['head']:.3f}  "
          f"whole {ranked['three_way']['full']:.3f}")
    print()
    for pair in PAIRS:
        p = cite["pairs"][pair]
        print(
            f"task1 {PAIR_LABEL[pair]:<20} cited-overlap {p['overlap'][0]:.3f}  "
            f"shared {p['shared']:.2f}"
        )
    print(f"task1 all three       {cite['three_way']:.3f}")
    for strategy in ORDER:
        c = cite["cited"][strategy]
        print(
            f"task1 {strategy:<7}: {c['mean']:.2f} documents/report, "
            f"{100 * c['unique_share']:.1f}% cited by no other strategy, "
            f"{c['total_distinct']} distinct overall"
        )

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    write_tables(ranked, cite, args.table)
    draw(ranked, cite, args.figure_dir / "strategy_agreement.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
