# The dataset release

The pipeline's sentence-level view of the collection is published on the Hugging Face Hub as
[`jknafou/trec-ragtime-2026`](https://huggingface.co/datasets/jknafou/trec-ragtime-2026), a
derivative of [`trec-ragtime/ragtime2`](https://huggingface.co/datasets/trec-ragtime/ragtime2).
This page says what ships, where each part of it is specified, and under what licence. What built
each table is in [`pipeline.md`](pipeline.md).

## What ships

| Config | Splits | Rows | Approx. size | Contents |
| --- | --- | ---: | ---: | --- |
| `sentences` | eng, spa, rus, zho | 88,719,200 | 4.3 GB | Our sentence segmentation, with the text and the offsets of each sentence |
| `translations_nllb` | spa, rus, zho | 55,919,788 | 2.7 GB | Every non-English sentence in English, by NLLB-200-3.3B |
| `translations_opus` | spa, rus, zho | 55,919,788 | 2.7 GB | The same sentences by the three small Helsinki OPUS-MT models |
| `passages` | eng, spa, rus, zho | 9,941,840 | 0.3 GB | The retrieval units, each a contiguous run of sentences |

210,500,616 rows, roughly 10 GB. The `sentences` size is a measurement of the files on disk; the
other three are projections from measured compression, which is why the column is approximate.

The parent's 4,000,380 documents are not a config. The document text is the parent's, it joins
back on an unchanged `document_id`, and reshipping it would put four million rows of someone
else's prose in the middle of a release that is otherwise either ours or a pointer into theirs.

The two translation configs have no English split. An English sentence has no translation: its
English rendering is its own text, so a sentence with no translation row is English.

## Where the shape is specified

| File | What it settles |
| --- | --- |
| [`release/card-front-matter.md`](release/card-front-matter.md) | The YAML block at the head of the card: licence, language tags, `source_datasets`, and the `dataset_info` that declares every config's columns and per-split row counts |
| [`release/translations.md`](release/translations.md) | The two translation tables, how each was produced, and where they differ |

Where anything else disagrees with those two files, they are the current statement of the shape.

## Three things a consumer has to know

Offsets are Python character indices into our normalised document text, half-open as
`[char_start, char_end)`. They are not byte offsets, which several published collections use
instead, and a reader who assumes bytes gets a span that is wrong on every multi-byte character.
The normalised text is not published, so the offsets are not a way to cut text out; the sentence
text ships on the sentence row. What they carry is adjacency: two members of a passage abutted in
the source when the second's `char_start` equals the first's `char_end`.

Passages overlap. A passage begins with the last sentence or two of its predecessor, and because
sentence ids are minted at segmentation time and never re-minted, that is the same sentence under
the same id in both. 13.6 per cent of sentences belong to more than one passage, which is why the
membership is published as an ordered list on the passage row rather than as a `passage_id` column
on the sentence row.

The three renderings compose by two different rules, and using the wrong one produces text that is
nearly right. Native text is a concatenation of the members with a space only where the offsets
leave a gap: nothing separates sentences in Chinese source text, so a plain space join changes
the text of the large majority of multi-sentence Chinese passages. Either English rendering is an
unconditional space join of the member translations, whatever the source language, because the
strings being joined are English. An English-source passage has no translation rows and takes the
native rule. The card body below states the native rule as code, because it is the one a
consumer is most likely to get wrong.

## The sentence inventory is not neutral between the two translation systems

Sentences too short to translate well on their own were grouped and passed to NLLB-200-3.3B as one
string with a separator between them. Where that separator did not come back intact, the
constituents were fused into a single sentence, and that fused sentence is what `sentences` holds;
the OPUS-MT arm then translated the already-fused inventory. The two English tables are two
translations of one inventory that the NLLB run helped determine, rather than two independent
segmentations of the same text. A reader comparing the systems sentence by sentence should know
which of them the sentence boundaries owe something to.

## Licence

Every table here is Adapted Material of a CC-BY-SA-4.0 work, so share-alike fixes the licence. All
four configs ship under CC-BY-SA-4.0, with the full licence text in a `LICENSE` file and the
attribution and the statement of changes in the card body. The code in this repository is MIT; the
data is not.

NLLB-200-3.3B, which produced `translations_nllb`, is distributed under CC-BY-NC-4.0, and the two
terms cannot both bind one table. Creative Commons has approved exactly two BY-SA-compatible
licences for 4.0, the Free Art License 1.3 and GPLv3, and neither carries a NonCommercial element;
§3(b)(3) forbids attaching one by any other route. The position taken here is that CC-BY-NC-4.0
governs the weights, which are the material Meta applied it to, and not text produced by running
them: Adapted Material has to contain the licensed thing, and an English sentence contains no
weights. Our own translation run was non-commercial academic research, which the term permits on
any reading.

The card therefore names each translation model and its terms as attribution and imposes no
condition of its own. Stating a non-commercial condition on the data would attach an additional
term to material the source licence obliges us to release without one, so the card would breach
the licence it declares.

Meta has not stated a position on whether the term reaches model output. The question was put to
`facebookresearch/fairseq` as issue #5516 on 2 July 2024, and the repository was archived
read-only without an answer. This section is a reading of the licence texts, not legal advice.

## The dataset card

The card is assembled from two sources rather than written by hand:

```bash
DS_STAGE=card sbatch slurm/upload_dataset.sbatch
```

The front-matter comes from the block in
[`release/card-front-matter.md`](release/card-front-matter.md) and the body from the block below.
The step checks both before writing — four configs and no fifth, `default: true` on `sentences`,
and declared row counts that match the footers of what is staged — and writes nothing if a check
fails. Editing the card on the Hub instead puts the two out of step, and the field where that is
not survivable is `num_examples`: `datasets` verifies it on load, and a card wrong by one row
raises `NonMatchingSplitsSizesError` for every user until it is fixed.

The body names the repository as `jknafou/trec-ragtime-2026`, which `DS_CARD_ARGS='--repo-id
<owner>/<name>'` rewrites, and its quick start works only because the front-matter marks
`sentences` as the default config.

## The body, verbatim

````markdown
# TREC RAGTIME 2026 — sentence and passage renderings

A sentence-level view of the [TREC RAGTIME 2026](https://trec-ragtime.github.io/) news collection,
with two English machine translations of every non-English sentence and the passage boundaries used
for retrieval. Derived from [`trec-ragtime/ragtime2`](https://huggingface.co/datasets/trec-ragtime/ragtime2).

Pipeline, experiment design, run configurations and reproduction steps:
**[github.com/jknafou/trec-ragtime-2026](https://github.com/jknafou/trec-ragtime-2026)**

## What is in here

| Config | Splits | Rows | Contents |
| --- | --- | ---: | --- |
| `sentences` | eng, spa, rus, zho | 88,719,200 | The segmentation, with character offsets into the source document |
| `translations_nllb` | spa, rus, zho | 55,919,788 | Every non-English sentence in English, by NLLB-200-3.3B |
| `translations_opus` | spa, rus, zho | 55,919,788 | The same sentences by the small Helsinki OPUS-MT models |
| `passages` | eng, spa, rus, zho | 9,941,840 | Retrieval units, each a contiguous run of sentences |

Document and sentence ids are shared across every config, and unchanged from the parent, so a
passage can be read as native text or as either translation without any alignment step.

## Quick start

```py
from datasets import load_dataset

repo = "jknafou/trec-ragtime-2026"
sentences = load_dataset(repo, "sentences", split="zho")
print(sentences[0])
```

Rebuilding a document's passage split, and the text of each passage:

```py
from datasets import load_dataset

repo, split = "jknafou/trec-ragtime-2026", "zho"
passages = load_dataset(repo, "passages", split=split)
sentences = load_dataset(repo, "sentences", split=split)

doc = passages[0]["document_id"]
by_id = {r["sentence_id"]: r for r in sentences if r["document_id"] == doc}

for p in (r for r in passages if r["document_id"] == doc):
    members = [by_id[sid] for sid in p["sentence_ids"]]
    parts = [members[0]["text"]]
    for prev, cur in zip(members, members[1:]):
        if cur["char_start"] > prev["char_end"]:
            parts.append(" ")
        parts.append(cur["text"])
    print(p["passage_id"], len(members), "".join(parts))
```

The space goes in only where the offsets leave a gap: Chinese sentences often abut with no
separator, so a plain join would corrupt them. Passages overlap, so a sentence can belong to more
than one. For an English rendering, join the member translations with a single space.

## Licence and attribution

Released under **CC-BY-SA-4.0**, inherited from the parent collection, which requires attribution,
a statement of changes, and same-licence redistribution. The changes are sentence segmentation,
passage grouping, and machine translation of the non-English sentences.

The translation models carry their own terms, which apply to the models and not to this text:
NLLB-200-3.3B is CC-BY-NC-4.0, and the OPUS-MT checkpoints are Apache-2.0 and CC-BY-4.0.

## Citation

A paper describing these runs is in preparation. This section will carry the reference once it is
published.

```bibtex
% to be added
```
````
