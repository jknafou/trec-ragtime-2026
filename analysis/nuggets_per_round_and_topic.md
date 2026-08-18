# How the nugget bank grows, and how large it ends up

The pipeline writes a first set of nuggets from the request alone, answers what it
can, audits the result for gaps, writes more nuggets and repeats. Each pass leaves
the whole bank on disk, so bank size against round number is a direct read, and the
highest round a request wrote is the pass on which it stopped.

The three strategies differ only in the rendering of a retrieved passage the generator
was shown. Rows run native text, then the low-tier Opus-MT translation, then the
high-tier NLLB translation.

## Growth across rounds

| Reads | Requests | First bank | Final bank | Growth | Last round | Rounds (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 103 | 11 (mean 10.9) | 18 (mean 18.6, max 47) | 1.70x | 2-8 | 5 |
| Opus-MT (low tier) | 103 | 11 (mean 10.9) | 17 (mean 18.1, max 42) | 1.66x | 2-8 | 4 |
| NLLB (high tier) | 103 | 11 (mean 10.9) | 19 (mean 18.9, max 49) | 1.73x | 2-8 | 5 |

## Where each request stopped

A request stops when an audit round adds fewer than one new nugget twice running.
Round 8 is a configured ceiling, not a stopping condition, so the requests
in that last column ran out of budget rather than out of nuggets.

| Reads | round 0 | round 1 | round 2 | round 3 | round 4 | round 5 | round 6 | round 7 | round 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 0 | 0 | 20 | 13 | 14 | 15 | 7 | 8 | 26 |
| Opus-MT (low tier) | 0 | 0 | 29 | 15 | 12 | 8 | 8 | 4 | 27 |
| NLLB (high tier) | 0 | 0 | 16 | 14 | 17 | 13 | 8 | 7 | 28 |

* Reading native text: 26 of 103 requests reached the ceiling (25%), so they were still finding new nuggets when the loop was stopped.
* Reading Opus-MT (low tier): 27 of 103 requests reached the ceiling (26%), so they were still finding new nuggets when the loop was stopped.
* Reading NLLB (high tier): 28 of 103 requests reached the ceiling (27%), so they were still finding new nuggets when the loop was stopped.

## The submitted bank

| Reads | Nuggets | Per request | At the cap | Answers per nugget | With no answer | Aggregator |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| native text | 1728 | 17 (mean 16.8, 6-30) | 1 of 103 | 2.08 (median 2) | 11.4% | OR 100% |
| Opus-MT (low tier) | 1670 | 15 (mean 16.2, 8-27) | 0 of 103 | 2.11 (median 2) | 11.4% | OR 100% |
| NLLB (high tier) | 1725 | 16 (mean 16.7, 6-30) | 1 of 103 | 2.07 (median 2) | 13.6% | OR 100% |

## What the submission keeps of what was grown

Two things stand between the grown bank and the submitted one, and they are separable.
A near-duplicate merge runs first and folds together nuggets that ask the same thing in
different words; the 30-nugget ceiling applies afterwards. So for any request
whose submitted bank is short of 30, the whole difference is the merge, and the
ceiling removed nothing.

| Reads | Grown (median) | Submitted (median) | Requests at the 30-nugget ceiling | Merged away as near-duplicates |
| --- | ---: | ---: | ---: | ---: |
| native text | 18 | 17 | 1 of 103 | 167 of 1865 (9%) |
| Opus-MT (low tier) | 17 | 15 | 0 of 103 | 198 of 1868 (11%) |
| NLLB (high tier) | 19 | 16 | 1 of 103 | 205 of 1900 (11%) |

## Where the nuggets came from

| Reads | round 0 | round 1 | round 2 | round 3 | round 4 | round 5 | round 6 | round 7 | round 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 58.7% | 8.7% | 8.3% | 6.7% | 5.2% | 3.9% | 4.0% | 2.5% | 1.9% |
| Opus-MT (low tier) | 60.2% | 9.1% | 8.3% | 6.2% | 4.9% | 3.7% | 3.3% | 2.4% | 1.8% |
| NLLB (high tier) | 57.7% | 9.3% | 9.5% | 6.4% | 5.6% | 4.3% | 3.0% | 2.2% | 1.9% |

## Also from the trace

* Reading native text: 103 requests read, 0 with a gap in the round numbering, 0 whose bank ever got smaller; 79.7% of the grown nuggets ended answered.
* Reading Opus-MT (low tier): 103 requests read, 0 with a gap in the round numbering, 1 whose bank ever got smaller; 76.3% of the grown nuggets ended answered.
* Reading NLLB (high tier): 103 requests read, 0 with a gap in the round numbering, 0 whose bank ever got smaller; 76.7% of the grown nuggets ended answered.
* The first round of nuggets, which reads the request and no passage, is identical across all three strategies for 99 of the 103 requests.
* Searching the Opus-MT index (low tier) (103 requests, no nugget bank of its own in any submitted file): final bank median 17, stopped on round 2-8, median 4.
* Searching the NLLB index (high tier) (103 requests, no nugget bank of its own in any submitted file): final bank median 16, stopped on round 2-8, median 4.

---

The same numbers are drawn in `docs/figures/nuggets_per_round.svg` and
`docs/figures/nuggets_per_topic.svg`. All three files are produced by
`analysis/nuggets_per_round_and_topic.py`, which reads the run trace and the submitted
Task 3 files.
