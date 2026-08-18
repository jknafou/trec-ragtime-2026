# Claims per round, and citations per report

Read from the run trace and from the submitted Task 1 files. Figures:
`docs/figures/claims_per_round.svg`, `docs/figures/report_citations.svg`.

All three runs searched the same native index and differ only in which rendering of a
retrieved passage the model was shown. Rows run native text, then the low-tier Opus-MT
translation, then the high-tier NLLB translation.

### Claims grounded per round

Unit: one round of one request.

| Rendering read | n | Median | IQR | Min | Max | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 557 | 4 | 2–9 | 0 | 42 | 7.66 |
| Opus-MT (low tier) | 527 | 5 | 2–11 | 0 | 42 | 8.02 |
| NLLB (high tier) | 557 | 5 | 2–9 | 0 | 38 | 7.54 |

### Cited sentences per report

Unit: one of the 103 requests.

| Rendering read | n | Median | IQR | Min | Max | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 103 | 28 | 22–32 | 0 | 40 | 26.25 |
| Opus-MT (low tier) | 103 | 29 | 20–33 | 0 | 42 | 26.24 |
| NLLB (high tier) | 103 | 29 | 20–32 | 0 | 43 | 25.93 |

Requests whose report carries no cited sentence at all: native text — 2023 (1 sentence), 2043 (1 sentence); Opus-MT (low tier) — 2043 (1 sentence); NLLB (high tier) — 2023 (1 sentence).

### Documents cited per sentence

| Rendering read | Sentences | 1 document | 2 documents | 3 documents | Uncited |
| --- | ---: | ---: | ---: | ---: | ---: |
| native text | 2706 | 2679 | 24 | 1 | 2 |
| Opus-MT (low tier) | 2704 | 2687 | 16 | 0 | 1 |
| NLLB (high tier) | 2672 | 2655 | 16 | 0 | 1 |

### Retrieval loops behind those rounds

| Rendering read | Loops | Answered | Unanswered | Errored | Grounding nothing |
| --- | ---: | ---: | ---: | ---: | ---: |
| native text | 2855 | 1950 | 870 | 35 | 904 |
| Opus-MT (low tier) | 3209 | 1943 | 1176 | 90 | 1265 |
| NLLB (high tier) | 3523 | 1933 | 1439 | 151 | 1589 |

### Claims grounded, by round index

Round 0 is the initial nugget bank and runs no retrieval, so the loops start at
round 1. A request stops when its coverage audit finds nothing left to ask, which is
why the later columns rest on fewer requests.

| Rendering read | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 2142 | 695 | 468 | 321 | 256 | 164 | 127 | 84 | 12 |
| Opus-MT (low tier) | 1894 | 765 | 534 | 374 | 284 | 147 | 122 | 88 | 21 |
| NLLB (high tier) | 1683 | 728 | 606 | 441 | 342 | 194 | 113 | 93 | 2 |
