# Language mix of retrieved documents

Share of the documents each Task-2 run retrieves, by the language the document is
written in, pooled over all 103 requests. The three runs differ only in which
rendering of the collection is indexed and searched; the request is the same English
text in all three, and every ranked list is of original document ids whatever was
searched. Rows run native text, then the low-tier Opus-MT translation, then the
high-tier NLLB translation.

Retrieval depth is emergent rather than a fixed cut-off — a run holds the documents
its loops actually surfaced — so the deeper rows are computed over slightly fewer
requests. The last column records how many.

The collection itself is balanced across the four languages
(English 25.0%, Spanish 25.0%, Russian 25.0%, Chinese 25.0%), so any departure from an even quarter is
something retrieval did, not something the collection imposed.

## Down to rank 1

| Index searched | English | Spanish | Russian | Chinese | Documents | Requests reaching this rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 31.1% | 35.0% | 10.7% | 23.3% | 103 | 103 of 103 |
| Opus-MT (low tier) | 36.9% | 32.0% | 12.6% | 18.4% | 103 | 103 of 103 |
| NLLB (high tier) | 36.9% | 35.0% | 13.6% | 14.6% | 103 | 103 of 103 |

## Down to rank 3

| Index searched | English | Spanish | Russian | Chinese | Documents | Requests reaching this rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 38.2% | 29.1% | 9.7% | 23.0% | 309 | 103 of 103 |
| Opus-MT (low tier) | 40.7% | 29.6% | 11.4% | 18.2% | 307 | 102 of 103 |
| NLLB (high tier) | 39.9% | 28.6% | 14.3% | 17.2% | 308 | 102 of 103 |

## Down to rank 5

| Index searched | English | Spanish | Russian | Chinese | Documents | Requests reaching this rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 36.8% | 26.4% | 11.2% | 25.6% | 511 | 101 of 103 |
| Opus-MT (low tier) | 39.5% | 27.4% | 13.7% | 19.4% | 511 | 102 of 103 |
| NLLB (high tier) | 38.2% | 27.8% | 13.3% | 20.6% | 510 | 100 of 103 |

## Down to rank 10

| Index searched | English | Spanish | Russian | Chinese | Documents | Requests reaching this rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native text | 34.4% | 26.5% | 12.2% | 27.0% | 998 | 94 of 103 |
| Opus-MT (low tier) | 39.1% | 26.1% | 13.1% | 21.8% | 1001 | 94 of 103 |
| NLLB (high tier) | 39.6% | 25.3% | 13.3% | 21.7% | 984 | 91 of 103 |

---

The same numbers are drawn in `docs/figures/task2_language_mix.svg`. Both are
produced by `analysis/task2_language_mix.py`, which reads the submitted run files
and the two lookups cached beside it, `analysis/doc_languages.tsv` for the language
of each retrieved document and `analysis/collection_languages.tsv` for the totals of
the whole collection. Both are rewritten from the corpus document table whenever a
wanted document id is missing from the cache.
