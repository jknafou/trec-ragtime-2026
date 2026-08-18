# Agreement between the three strategies

Each task was answered three times. The three answers differ in one thing, and
which thing depends on the task: in Task 2 the three runs search a different
rendering of the collection (the native text, a low-tier Opus-MT translation, or
a high-tier NLLB translation); in Task 1 all three search the same native index and differ
only in which rendering of a retrieved passage the model reads. Both are measured
over the same 103 requests, and every document id is the original one whatever was
searched or read.

Throughout, *overlap* means the documents two runs share divided by the documents
in either of them, averaged over requests, with the interquartile range in
brackets. *Kendall's tau* is computed on the documents a pair has in common, from
their positions in each ranking: +1 is the same order, 0 is an unrelated order.

## Task 2 — does changing the index change what is found, or only the order?

| Comparison | Of the top 10, shared | Overlap, top 10 | Overlap, whole list | Order of shared documents | Same document ranked first |
| --- | ---: | ---: | ---: | ---: | ---: |
| native vs Opus-MT | 4.1 of 10 | 0.29 [0.18–0.43] | 0.33 [0.23–0.39] | +0.38 [+0.21 to +0.54] | 37 of 103 |
| native vs NLLB | 4.0 of 10 | 0.28 [0.18–0.43] | 0.33 [0.23–0.41] | +0.36 [+0.20 to +0.52] | 33 of 103 |
| Opus-MT vs NLLB | 4.5 of 10 | 0.33 [0.20–0.43] | 0.36 [0.25–0.45] | +0.41 [+0.24 to +0.57] | 33 of 103 |
| **all three at once** | | **0.17** | **0.21** | | |

Retrieval depth here is emergent — a run holds the documents its loops actually
surfaced, not a fixed thousand — and the deepest single ranked list in the whole
submission is 80 documents long. A cut at 100 therefore selects the entire
list every time, so that column is the whole list and not a truncation. The lists
also differ in length between runs, which by itself holds the overlap below 1.
Per-run depth:

| Index searched | Shortest | Median | Longest | Requests reaching rank 10 |
| --- | ---: | ---: | ---: | ---: |
| native text | 3 | 22 | 67 | 94 of 103 |
| Opus-MT (low tier) | 1 | 21 | 70 | 94 of 103 |
| NLLB (high tier) | 2 | 20 | 80 | 91 of 103 |

## Task 1 — do the three reports rest on the same documents?

All three of these runs searched the same native index. Any disagreement below is
therefore produced downstream of retrieval: by what the model read, by which
queries it chose to issue, and by its own run-to-run variation.

| Comparison | Documents cited by both | Overlap of cited documents |
| --- | ---: | ---: |
| native vs Opus-MT | 8.9 | 0.36 [0.24–0.44] |
| native vs NLLB | 8.5 | 0.35 [0.21–0.44] |
| Opus-MT vs NLLB | 8.9 | 0.37 [0.28–0.46] |
| **all three at once** | | **0.23** |

| Rendering read | Documents cited per report | Cited by no other strategy |
| --- | ---: | ---: |
| native text | 18.4 | 38% |
| Opus-MT (low tier) | 17.9 | 34% |
| NLLB (high tier) | 17.3 | 35% |

## What this shows

Changing the index changes what is found, and it changes the order as well.
Two runs share about 34% of their documents over the whole ranked list,
only 21% of the pooled documents are found by all
three, and a system that searched a single rendering would never see most of what
the other two surface.

Two details are worth more than the headline overlap. First, agreement is *worse*
at the top of the ranking than over the whole list (30% against
34%); the three runs share on average 4.2 of their
top ten documents and put the same document in first place in 34 of
103 requests. Second, the shared documents are not merely reshuffled slightly: the
rank correlation over them averages +0.38, clearly positive but a long
way from the +1 that a pure reordering of one common pool would give. The
disagreement is therefore both about which documents are found and about where they
are placed, and it is sharpest at the head, where a reader looks.

The same overlap measure on the Task 1 reports gives about 36% — the
same magnitude — from three runs that all searched the *same* index; about
36% of the documents a report cites are cited by neither of the other
two. It is a caution, not a result: the Task 2 disagreement cannot be credited to
the index alone, because holding the index fixed and varying only what the model
reads moves the answer about as much. The two are not measuring quite the same
object — Task 2 compares what was retrieved, Task 1 compares what survived into a
report, and that extra selection step carries variation of its own — so this bounds
the index effect loosely rather than isolating it.

## What this does not show

* Not quality. This track has no relevance judgements and none are coming, so
  nothing here says which of the three found the better documents, or whether any
  of them found good ones. Three strategies could disagree completely and all be
  right, or all be wrong. Agreement is not accuracy.
* Not an isolated index effect. Each strategy was run once, so there is no
  repeat at a fixed setting to measure run-to-run variation against. The pipeline
  is agentic: it issues its own queries over several rounds, and two runs of one
  setting would not be expected to agree perfectly either. The Task 1 figure above
  bounds how much of the Task 2 disagreement the index can be credited with, but it
  does not separate the two cleanly.
* Not independent runs. The native Task 2 run and the native Task 1 report come
  from a single execution — searching the native index while reading native
  passages is one system under two task descriptions — so those two files are not
  independent observations of one another.
* Not a ceiling. Overlap is measured over the documents each run actually
  returned. Depth is emergent and unequal between runs, which by itself puts an
  upper bound below 1 on any pair whose lists differ in length.

---

The same numbers are drawn in `docs/figures/strategy_agreement.svg`. Both are
produced by `analysis/strategy_agreement.py`, which reads only the submitted run
files.
