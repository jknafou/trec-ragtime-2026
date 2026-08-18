# Analysis

Four measurements of the submitted runs, and the scripts that produce them. Each script writes one
markdown document beside itself and one or two figures under `docs/figures/`; the root `README.md`
shows the headline rows and links here for the rest.

| Script | Document | Figures |
| --- | --- | --- |
| `claims_per_round_and_topic.py` | `claims_per_round_and_topic.md` | `docs/figures/claims_per_round.svg`, `docs/figures/report_citations.svg` |
| `nuggets_per_round_and_topic.py` | `nuggets_per_round_and_topic.md` | `docs/figures/nuggets_per_round.svg`, `docs/figures/nuggets_per_topic.svg` |
| `strategy_agreement.py` | `strategy_agreement.md` | `docs/figures/strategy_agreement.svg` |
| `task2_language_mix.py` | `task2_language_mix.md` | `docs/figures/task2_language_mix.svg` |

Each takes the submission directory as its one positional argument, and each is a read: none
of them runs a model, opens an index or rewrites a submitted file.

```
uv run python analysis/<script>.py submission
```

`strategy_agreement.py` and `task2_language_mix.py` read nothing but `submission/`. The other two
also read the run trace, the artifact tree a run leaves on the cluster, which is not part of this
repository; `--trace` points them at it.

The language of a retrieved document is not in its id, so `task2_language_mix.py` resolves every
retrieved id against the corpus document table and caches the answer in `doc_languages.tsv`, with
the language totals of the whole collection in `collection_languages.tsv`. Both are committed,
which is what lets its tables and its figure rebuild from the submission alone, away from the
cluster.

Two files here serve the data release rather than these measurements. `passage_text.py` is the
reference implementation of the two rules that turn a published passage back into text, native and
translated. `upload_dataset.py` builds the four released configurations, writes the dataset card
from two documents under `docs/`, and pushes the result. Its two halves want opposite machines:
the build is CPU and disk with no network and runs from `slurm/upload_dataset.sbatch`, while the
transfer needs the internet and must run on the login node, because compute nodes here have no
route to it. The tool refuses to transfer from inside a SLURM allocation for that reason.
