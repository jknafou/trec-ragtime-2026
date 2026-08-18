#!/usr/bin/env python3
"""How the nugget bank grows, and how large it ends up.

The pipeline does not decide up front what a report should cover. It writes a small set of
nuggets from the request alone, answers what it can, audits the result for gaps, writes
more nuggets, and repeats until an audit finds nothing new. Each pass leaves the whole
nugget bank on disk as ``topics/<topic>/decompose/round_<r>.jsonl``, so the bank after
round *r* is a direct read of a file, and the highest round a topic wrote is the pass on
which it stopped.

Panel (a), from the run trace: the size of the bank against the round number, and the
distribution of the round each request stopped on.

Panel (b), from the submitted Task 3 files: how many nuggets each submitted bank holds,
how many answers sit under a nugget, and the mix of aggregator types.

The three strategies differ only in which rendering of a retrieved passage the generator
was shown -- the native text, the low-tier Opus-MT translation of it, or the high-tier
NLLB-200-3.3B translation of it. Everything below is ordered that way, by ascending
translation quality. The index that was searched was the same in all three.

Run it on a compute node, never on the login node:

    sbatch --partition=shared-cpu,public-cpu --cpus-per-task=4 --mem=16G --time=00:30:00 \
           --wrap="uv run python analysis/nuggets_per_round_and_topic.py submission"

Outputs:

    docs/figures/nuggets_per_round.svg        growth curves, and where each request stopped
    docs/figures/nuggets_per_topic.svg        submitted bank size, and answers per nugget
    analysis/nuggets_per_round_and_topic.md   the same numbers as markdown tables
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parent.parent

# Where the run trace lives. Six directories sit here, one per configuration; the three
# read below are the ones that vary the rendering the generator reads.
DEFAULT_TRACE = pathlib.Path("/srv/beegfs/scratch/users/k/knafouj/ragtime-runs")

# One row per strategy: the label a reader sees, its colour, the directory in the run trace,
# and the submitted Task 3 file that strategy produced. The order is the one used everywhere
# -- native text, then the low tier, then the high tier. Run numbering is not a strategy
# ordering -- run 1 of Task 3 is the NLLB reading and run 3 is the native reading -- so
# everything downstream is keyed by strategy and never by run number.
STRATEGIES = [
    ("native text", "#2F6BC0", "e2e-original__original__seed0", "T3_run_3.jsonl"),
    ("Opus-MT (low tier)", "#C5701F", "e2e-omt-weak__omt_opus__seed0", "T3_run_2.jsonl"),
    ("NLLB (high tier)", "#1B7A3E", "e2e-omt__omt__seed0", "T3_run_1.jsonl"),
]

# Two further directories in the trace hold complete executions that differ in which index
# was searched rather than in what was read. They produce no nugget bank of their own in
# any submitted file, so they stay out of the figures and are reported below the tables. Same
# ordering, minus the native arm, which is the reading comparison's own baseline.
SEARCH_SIDE = [
    ("Searching the Opus-MT index (low tier)", "mlir-omt-weak__original__seed0"),
    ("Searching the NLLB index (high tier)", "mlir-omt__original__seed0"),
]

# The submission keeps at most this many nuggets per request, the highest-weighted first,
# and at most this many answers under each. Both are our own choices; the format imposes
# neither. The tables read the values back from the artefacts; these constants only mark
# the two limits on the figures.
BANK_CAP = 30
ANSWERS_CAP = 5

# The configured ceiling on rounds. The loop is meant to stop earlier, when an audit finds
# fewer than one new nugget twice running; this is the backstop. The figure marks it,
# because a request that stopped on the backstop had not run out of nuggets.
ROUND_CEILING = 8

GRID = {"color": "#D8D8D8", "linewidth": 0.6}


def read_growth(tree: pathlib.Path) -> tuple[dict[str, list[int]], collections.Counter]:
    """Bank size after each round, per request, plus a few counts worth reporting.

    Returns a mapping from request id to the list of bank sizes indexed by round, and a
    diagnostics counter. A request whose round files skip a number would break the indexing
    silently, so the gap is counted rather than smoothed over.
    """
    growth: dict[str, list[int]] = {}
    seed_questions: dict[str, frozenset[str]] = {}
    diag: collections.Counter[str] = collections.Counter()
    origin: collections.Counter[int] = collections.Counter()
    status: collections.Counter[str] = collections.Counter()

    for topic_dir in sorted((tree / "topics").iterdir()):
        rounds: dict[int, int] = {}
        final_records: list[dict] = []
        for path in topic_dir.glob("decompose/round_*.jsonl"):
            number = int(path.stem.split("_")[1])
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            rounds[number] = len(records)
            if number >= max(rounds):
                final_records = records
            if number == 0:
                seed_questions[topic_dir.name] = frozenset(r["question"] for r in records)
        if not rounds:
            diag["requests_with_no_round_file"] += 1
            continue

        ordered = [rounds[n] for n in range(max(rounds) + 1) if n in rounds]
        if len(ordered) != max(rounds) + 1:
            diag["requests_with_a_missing_round"] += 1
        if any(b < a for a, b in zip(ordered, ordered[1:], strict=False)):
            diag["requests_whose_bank_shrank"] += 1

        growth[topic_dir.name] = ordered
        diag["requests"] += 1
        for record in final_records:
            origin[int(record.get("origin_round", 0))] += 1
            status[str(record.get("status", "unknown"))] += 1

    return growth, (diag, origin, status, seed_questions)


def carried_forward(growth: dict[str, list[int]], horizon: int) -> np.ndarray:
    """Bank size per request at every round out to ``horizon``, held flat after it stops.

    A request that stopped on round 4 had a bank of its final size from then on; it did not
    cease to have one. Averaging only over the requests still running at round *r* would
    instead measure "the requests that were still finding gaps", which climbs by
    construction and says nothing about the collection as a whole.
    """
    rows = []
    for sizes in growth.values():
        row = list(sizes) + [sizes[-1]] * (horizon + 1 - len(sizes))
        rows.append(row[: horizon + 1])
    return np.array(rows, dtype=float)


def read_submitted(path: pathlib.Path) -> dict[str, list[dict]]:
    """The submitted nugget bank of every request in one Task 3 file."""
    banks: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        banks[str(row["metadata"]["topic_id"])] = row["nugget_bank"]
    return banks


def figure_rounds(growth_by_strategy, stop_by_strategy, out: pathlib.Path) -> None:
    """Panel (a): the growth curves, and where each request stopped."""
    horizon = max(max(stops) for stops in stop_by_strategy.values())
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.9))

    for label, colour, _, _ in STRATEGIES:
        table = carried_forward(growth_by_strategy[label], horizon)
        rounds = np.arange(horizon + 1)
        median = np.median(table, axis=0)
        lower = np.percentile(table, 25, axis=0)
        upper = np.percentile(table, 75, axis=0)
        left.fill_between(rounds, lower, upper, color=colour, alpha=0.13, linewidth=0)
        left.plot(rounds, median, color=colour, linewidth=1.9, marker="o", markersize=3.2, label=label)

    left.set_xlabel("Round of nugget-writing and answering")
    left.set_ylabel("Nuggets in the bank")
    left.set_xticks(range(horizon + 1))
    left.set_ylim(bottom=0)
    left.grid(True, axis="y", **GRID)
    left.set_axisbelow(True)
    for side in ("top", "right"):
        left.spines[side].set_visible(False)
    legend = left.legend(loc="upper left", frameon=False, fontsize=8.5, title="Passages read as")
    legend.get_title().set_fontsize(8.5)

    width = 0.27
    positions = np.arange(horizon + 1)
    for offset, (label, colour, _, _) in enumerate(STRATEGIES):
        stops = stop_by_strategy[label]
        counts = [sum(1 for s in stops if s == r) for r in range(horizon + 1)]
        right.bar(positions + (offset - 1) * width, counts, width=width, color=colour, label=label)

    right.set_xlabel("Round on which the request stopped")
    right.set_ylabel("Requests")
    right.set_xticks(range(horizon + 1))
    right.grid(True, axis="y", **GRID)
    right.set_axisbelow(True)
    for side in ("top", "right"):
        right.spines[side].set_visible(False)

    # The last group is not a saturation peak: it is the configured ceiling on rounds. Saying
    # so on the figure keeps a reader from reading "ran out of budget" as "found everything".
    right.annotate(
        f"round {ROUND_CEILING} is the\nconfigured ceiling",
        xy=(positions[ROUND_CEILING], right.get_ylim()[1] * 0.95),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color="#666666",
    )

    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


def figure_topics(banks_by_strategy, out: pathlib.Path) -> None:
    """Panel (b): submitted bank size per request, and answers per nugget."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.9))

    # Bins stop exactly at the ceiling, so no bar can extend past the dashed line and imply a
    # bank the submission could not contain. numpy closes the last bin on the right, so a bank
    # of exactly BANK_CAP lands in it.
    edges = np.arange(0, BANK_CAP + 2, 2)
    for label, colour, _, _ in STRATEGIES:
        sizes = [len(bank) for bank in banks_by_strategy[label].values()]
        left.hist(
            sizes,
            bins=edges,
            histtype="step",
            linewidth=1.9,
            color=colour,
            label=label,
            weights=np.full(len(sizes), 100.0 / len(sizes)),
        )
    left.axvline(BANK_CAP, color="#666666", linestyle="--", linewidth=1.0)
    left.annotate(
        f"submission keeps\nat most {BANK_CAP}",
        xy=(BANK_CAP, left.get_ylim()[1] * 0.55),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color="#666666",
    )
    left.set_xlim(4, BANK_CAP + 1)
    left.set_xlabel("Nuggets in a submitted bank")
    left.set_ylabel("Requests (%)")
    left.grid(True, axis="y", **GRID)
    left.set_axisbelow(True)
    for side in ("top", "right"):
        left.spines[side].set_visible(False)
    legend = left.legend(loc="upper left", frameon=False, fontsize=8.5, title="Passages read as")
    legend.get_title().set_fontsize(8.5)

    width = 0.27
    positions = np.arange(ANSWERS_CAP + 1)
    for offset, (label, colour, _, _) in enumerate(STRATEGIES):
        counts = collections.Counter(
            len(nugget["answers"]) for bank in banks_by_strategy[label].values() for nugget in bank
        )
        total = sum(counts.values())
        shares = [100.0 * counts[k] / total for k in range(ANSWERS_CAP + 1)]
        right.bar(positions + (offset - 1) * width, shares, width=width, color=colour, label=label)

    right.set_xlabel("Answers found for one nugget")
    right.set_ylabel("Share of nuggets (%)")
    right.set_xticks(positions)
    right.grid(True, axis="y", **GRID)
    right.set_axisbelow(True)
    for side in ("top", "right"):
        right.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


def summary(rows: list[str]) -> str:
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=pathlib.Path, help="the submission directory")
    parser.add_argument("--trace", type=pathlib.Path, default=DEFAULT_TRACE, help="the run trace root")
    parser.add_argument("--figures", type=pathlib.Path, default=REPO / "docs" / "figures")
    parser.add_argument("--table", type=pathlib.Path, default=REPO / "analysis" / "nuggets_per_round_and_topic.md")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    growth_by_strategy: dict[str, dict[str, list[int]]] = {}
    stop_by_strategy: dict[str, list[int]] = {}
    banks_by_strategy: dict[str, dict[str, list[dict]]] = {}
    diagnostics: dict[str, tuple] = {}

    for label, _, tree_name, task3_name in STRATEGIES:
        growth, diag = read_growth(args.trace / tree_name)
        growth_by_strategy[label] = growth
        stop_by_strategy[label] = [len(sizes) - 1 for sizes in growth.values()]
        banks_by_strategy[label] = read_submitted(args.submission / "task3" / task3_name)
        diagnostics[label] = diag

    figure_rounds(growth_by_strategy, stop_by_strategy, args.figures / "nuggets_per_round.svg")
    figure_topics(banks_by_strategy, args.figures / "nuggets_per_topic.svg")

    rows: list[str] = []
    rows.append("# How the nugget bank grows, and how large it ends up\n")
    rows.append(
        "The pipeline writes a first set of nuggets from the request alone, answers what it\n"
        "can, audits the result for gaps, writes more nuggets and repeats. Each pass leaves\n"
        "the whole bank on disk, so bank size against round number is a direct read, and the\n"
        "highest round a request wrote is the pass on which it stopped.\n"
    )
    rows.append(
        "The three strategies differ only in the rendering of a retrieved passage the generator\n"
        "was shown. Rows run native text, then the low-tier Opus-MT translation, then the\n"
        "high-tier NLLB translation."
    )

    rows.append("\n## Growth across rounds\n")
    rows.append(
        "| Reads | Requests | First bank | Final bank | Growth | Last round | Rounds (median) |"
    )
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, _, _, _ in STRATEGIES:
        growth = growth_by_strategy[label]
        first = [sizes[0] for sizes in growth.values()]
        final = [sizes[-1] for sizes in growth.values()]
        stops = stop_by_strategy[label]
        rows.append(
            f"| {label} | {len(growth)} "
            f"| {statistics.median(first):.0f} (mean {statistics.mean(first):.1f}) "
            f"| {statistics.median(final):.0f} (mean {statistics.mean(final):.1f}, max {max(final)}) "
            f"| {statistics.mean(final) / statistics.mean(first):.2f}x "
            f"| {min(stops)}-{max(stops)} "
            f"| {statistics.median(stops):.0f} |"
        )

    rows.append("\n## Where each request stopped\n")
    rows.append(
        f"A request stops when an audit round adds fewer than one new nugget twice running.\n"
        f"Round {ROUND_CEILING} is a configured ceiling, not a stopping condition, so the requests\n"
        f"in that last column ran out of budget rather than out of nuggets.\n"
    )
    horizon = max(max(stops) for stops in stop_by_strategy.values())
    rows.append("| Reads | " + " | ".join(f"round {r}" for r in range(horizon + 1)) + " |")
    rows.append("| --- |" + " ---: |" * (horizon + 1))
    for label, _, _, _ in STRATEGIES:
        stops = stop_by_strategy[label]
        counts = [sum(1 for s in stops if s == r) for r in range(horizon + 1)]
        rows.append(f"| {label} | " + " | ".join(str(c) for c in counts) + " |")
    rows.append("")
    for label, _, _, _ in STRATEGIES:
        stops = stop_by_strategy[label]
        hit = sum(1 for s in stops if s >= ROUND_CEILING)
        rows.append(
            f"* Reading {label}: {hit} of {len(stops)} requests reached the ceiling "
            f"({100.0 * hit / len(stops):.0f}%), so they were still finding new nuggets when "
            f"the loop was stopped."
        )

    rows.append("\n## The submitted bank\n")
    rows.append(
        "| Reads | Nuggets | Per request | At the cap | Answers per nugget | With no answer | Aggregator |"
    )
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for label, _, _, _ in STRATEGIES:
        banks = banks_by_strategy[label]
        sizes = [len(bank) for bank in banks.values()]
        nuggets = [nugget for bank in banks.values() for nugget in bank]
        answers = [len(nugget["answers"]) for nugget in nuggets]
        mix = collections.Counter(nugget["aggregator_type"] for nugget in nuggets)
        mix_text = ", ".join(f"{k} {100.0 * v / len(nuggets):.0f}%" for k, v in mix.most_common())
        rows.append(
            f"| {label} | {sum(sizes)} "
            f"| {statistics.median(sizes):.0f} (mean {statistics.mean(sizes):.1f}, "
            f"{min(sizes)}-{max(sizes)}) "
            f"| {sum(1 for s in sizes if s >= BANK_CAP)} of {len(sizes)} "
            f"| {statistics.mean(answers):.2f} (median {statistics.median(answers):.0f}) "
            f"| {100.0 * sum(1 for a in answers if a == 0) / len(answers):.1f}% "
            f"| {mix_text} |"
        )

    rows.append("\n## What the submission keeps of what was grown\n")
    rows.append(
        "Two things stand between the grown bank and the submitted one, and they are separable.\n"
        "A near-duplicate merge runs first and folds together nuggets that ask the same thing in\n"
        f"different words; the {BANK_CAP}-nugget ceiling applies afterwards. So for any request\n"
        f"whose submitted bank is short of {BANK_CAP}, the whole difference is the merge, and the\n"
        "ceiling removed nothing.\n"
    )
    rows.append(
        f"| Reads | Grown (median) | Submitted (median) | Requests at the {BANK_CAP}-nugget "
        "ceiling | Merged away as near-duplicates |"
    )
    rows.append("| --- | ---: | ---: | ---: | ---: |")
    for label, _, _, _ in STRATEGIES:
        growth = growth_by_strategy[label]
        banks = banks_by_strategy[label]
        shared = sorted(set(growth) & set(banks))
        grown = [growth[t][-1] for t in shared]
        kept = [len(banks[t]) for t in shared]
        at_ceiling = [k >= BANK_CAP for k in kept]
        merged = sum(g - k for g, k, c in zip(grown, kept, at_ceiling, strict=True) if not c)
        below = sum(g for g, c in zip(grown, at_ceiling, strict=True) if not c)
        rows.append(
            f"| {label} | {statistics.median(grown):.0f} | {statistics.median(kept):.0f} "
            f"| {sum(at_ceiling)} of {len(shared)} "
            f"| {merged} of {below} ({100.0 * merged / below:.0f}%) |"
        )

    rows.append("\n## Where the nuggets came from\n")
    rows.append("| Reads | " + " | ".join(f"round {r}" for r in range(horizon + 1)) + " |")
    rows.append("| --- |" + " ---: |" * (horizon + 1))
    for label, _, _, _ in STRATEGIES:
        _, origin, _, _ = diagnostics[label]
        total = sum(origin.values())
        rows.append(
            f"| {label} | "
            + " | ".join(f"{100.0 * origin[r] / total:.1f}%" for r in range(horizon + 1))
            + " |"
        )

    rows.append("\n## Also from the trace\n")
    for label, _, _, _ in STRATEGIES:
        diag, _, status, _ = diagnostics[label]
        answered = status.get("answered", 0)
        total = sum(status.values())
        rows.append(
            f"* Reading {label}: {diag['requests']} requests read, "
            f"{diag['requests_with_a_missing_round']} with a gap in the round numbering, "
            f"{diag['requests_whose_bank_shrank']} whose bank ever got smaller; "
            f"{100.0 * answered / total:.1f}% of the grown nuggets ended answered."
        )

    # The first round reads the request and nothing else, so it must be identical across the
    # three strategies. It is the anchor the whole comparison rests on.
    seeds = [diagnostics[label][3] for label, _, _, _ in STRATEGIES]
    common = sorted(set(seeds[0]) & set(seeds[1]) & set(seeds[2]))
    agree = sum(1 for t in common if seeds[0][t] == seeds[1][t] == seeds[2][t])
    rows.append(
        f"* The first round of nuggets, which reads the request and no passage, is identical "
        f"across all three strategies for {agree} of the {len(common)} requests."
    )

    for label, tree_name in SEARCH_SIDE:
        growth, (diag, _, _, _) = read_growth(args.trace / tree_name)
        final = [sizes[-1] for sizes in growth.values()]
        stops = [len(sizes) - 1 for sizes in growth.values()]
        rows.append(
            f"* {label} ({diag['requests']} requests, no nugget bank of its own in any "
            f"submitted file): final bank median {statistics.median(final):.0f}, "
            f"stopped on round {min(stops)}-{max(stops)}, median {statistics.median(stops):.0f}."
        )

    rows.append(
        "\n---\n\nThe same numbers are drawn in `docs/figures/nuggets_per_round.svg` and\n"
        "`docs/figures/nuggets_per_topic.svg`. All three files are produced by\n"
        "`analysis/nuggets_per_round_and_topic.py`, which reads the run trace and the submitted\n"
        "Task 3 files."
    )

    args.table.write_text(summary(rows), encoding="utf-8")
    print(summary(rows))


if __name__ == "__main__":
    main()
