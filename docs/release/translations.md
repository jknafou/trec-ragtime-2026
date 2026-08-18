# The sentence inventory and the two translation systems

The release carries two English renderings of every non-English sentence, one by NLLB-200-3.3B
and one by the small Helsinki OPUS-MT models. They are two translations of a single sentence
inventory, and the boundaries of that inventory were fixed in part by one of the two systems.

Segmentation ran on the native text before any translation existed, so to that point the
inventory owes nothing to either system. Short sentences translate badly on their own, so
consecutive short sentences were grouped into one unit, joined with a separator and handed
to NLLB as a single string, to be cut back apart on the separator afterwards. Where the
separator did not survive translation, the constituents could not be cut apart and were
fused into one sentence spanning them all. That fused sentence is what the `sentences`
config holds, and it is what the OPUS-MT arm was later asked to translate, one sentence
per call, with no grouping of its own.

Merge units, and the rate at which one failed to split back:

| Language | Merge units | Units that failed to split | Rate |
| --- | ---: | ---: | ---: |
| Spanish | 4,306,268 | 750,948 | 17.4 % |
| Russian | 4,812,333 | 1,601,810 | 33.3 % |
| Chinese | 3,530,376 | 2,083,915 | 59.0 % |

The effect on the inventory, at sentence grain:

| Language | From segmentation | Published | Absorbed into a fusion | Fused sentences |
| --- | ---: | ---: | ---: | ---: |
| Spanish | 22,611,885 | 21,435,940 | 1,926,893 (8.5 %) | 750,948 (3.5 %) |
| Russian | 21,331,755 | 18,614,576 | 4,318,989 (20.2 %) | 1,601,810 (8.6 %) |
| Chinese | 18,544,691 | 15,869,272 | 4,759,334 (25.7 %) | 2,083,915 (13.1 %) |
| English | 32,799,412 | 32,799,412 | 0 | 0 |

English is never merged and never translated, so it is untouched. Across the three translated
languages, 17.6 per cent of the sentences that came out of segmentation were absorbed into a
fused sentence, and 7.9 per cent of the published non-English sentences are fused ones. The
largest merge unit was 26 sentences in Russian, 21 in Spanish and 14 in Chinese.

The OPUS-MT arm did no grouping of its own, which its raw output shows directly: every one of
its 55,919,788 units has a constituent count of exactly one, none carries a merge-unit id,
and its per-language unit counts are the published sentence counts of 21,435,940 Spanish,
18,614,576 Russian and 15,869,272 Chinese. The NLLB output is 42,833,568 units covering
62,488,331 sentences.

So the asymmetry runs one way: NLLB influenced the sentence boundaries and OPUS-MT inherited
them, and the two tables are not two independent segmentations of the same text. The
influence is far from uniform across languages — a Chinese merge unit failed to split
three and a half times as often as a Spanish one — so a per-language comparison of the
two systems is also a comparison across inventories of different granularity.

One further difference between the two tables. A translation that came back with no
characters is kept as an empty string rather than dropped or nulled, so that the two tables
stay aligned row for row; a user who wants them gone can filter on `text != ""`.

| | `spa` | `rus` | `zho` | total |
| --- | ---: | ---: | ---: | ---: |
| `translations_nllb` | 2 | 0 | 0 | 2 |
| `translations_opus` | 871 | 8 | 73 | 952 |
