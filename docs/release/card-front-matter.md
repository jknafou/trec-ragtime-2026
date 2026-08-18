# Dataset card front matter

The YAML block below is the front matter of the dataset card on the Hugging Face Hub.
`analysis/upload_dataset.py card` reads it from here, checks it against the Parquet files that are
staged for upload and writes the card, so the published card is reproducible from this repository.

## The block

```yaml
license: cc-by-sa-4.0
language:
  - en
  - es
  - ru
  - zh
multilinguality:
  - multilingual
  - translation
annotations_creators:
  - no-annotation
language_creators:
  - found
  - machine-generated
source_datasets:
  - trec-ragtime/ragtime2
  - extended|trec-ragtime/ragtime2
task_categories:
  - text-retrieval
  - translation
task_ids:
  - document-retrieval
size_categories:
  - 100M<n<1B
pretty_name: RAGTIME2 Sentence Renderings
tags:
  - multilingual
  - RAG
  - News
  - machine-translation
  - parallel-corpus
  - sentence-segmentation
  - trec
configs:
  - config_name: sentences
    default: true
    data_files:
      - split: eng
        path: sentences/eng-*.parquet
      - split: spa
        path: sentences/spa-*.parquet
      - split: rus
        path: sentences/rus-*.parquet
      - split: zho
        path: sentences/zho-*.parquet
  - config_name: translations_nllb
    data_files:
      - split: spa
        path: translations_nllb/spa-*.parquet
      - split: rus
        path: translations_nllb/rus-*.parquet
      - split: zho
        path: translations_nllb/zho-*.parquet
  - config_name: translations_opus
    data_files:
      - split: spa
        path: translations_opus/spa-*.parquet
      - split: rus
        path: translations_opus/rus-*.parquet
      - split: zho
        path: translations_opus/zho-*.parquet
  - config_name: passages
    data_files:
      - split: eng
        path: passages/eng-*.parquet
      - split: spa
        path: passages/spa-*.parquet
      - split: rus
        path: passages/rus-*.parquet
      - split: zho
        path: passages/zho-*.parquet
dataset_info:
  - config_name: sentences
    features:
      - name: document_id
        dtype: string
      - name: sentence_id
        dtype: string
      - name: sentence_index
        dtype: int32
      - name: lang
        dtype: string
      - name: char_start
        dtype: int32
      - name: char_end
        dtype: int32
      - name: text
        dtype: string
    splits:
      - name: eng
        num_examples: 32799412
      - name: spa
        num_examples: 21435940
      - name: rus
        num_examples: 18614576
      - name: zho
        num_examples: 15869272
  - config_name: translations_nllb
    features:
      - name: sentence_id
        dtype: string
      - name: document_id
        dtype: string
      - name: text
        dtype: string
    splits:
      - name: spa
        num_examples: 21435940
      - name: rus
        num_examples: 18614576
      - name: zho
        num_examples: 15869272
  - config_name: translations_opus
    features:
      - name: sentence_id
        dtype: string
      - name: document_id
        dtype: string
      - name: text
        dtype: string
    splits:
      - name: spa
        num_examples: 21435940
      - name: rus
        num_examples: 18614576
      - name: zho
        num_examples: 15869272
  - config_name: passages
    features:
      - name: passage_id
        dtype: string
      - name: document_id
        dtype: string
      - name: lang
        dtype: string
      - name: sentence_ids
        sequence: string
      - name: token_count
        dtype: int32
      - name: is_oversized
        dtype: bool
    splits:
      - name: eng
        num_examples: 2906906
      - name: spa
        num_examples: 2545034
      - name: rus
        num_examples: 1832768
      - name: zho
        num_examples: 2657132
```

## The values

`configs` is what makes the dataset viewer work. The four configs sit in top-level directories
whose names the Hub's automatic structure detection does not read as splits, so the mapping
is declared rather than inferred.

`sentences` is the default config. `load_dataset("<repo>")` with no config name resolves to
it, and the viewer opens on it rather than on `passages`, which is not usable on its own.

The language tags are two-letter and match the `lang` column of `sentences` and `passages`. The
split names are three-letter and match the parent collection: `eng`, `spa`, `rus` and `zho`
hold `en`, `es`, `ru` and `zh`. The correspondence is one to one.

`size_categories` is the sum over the four configs, 88,719,200 + 55,919,788 + 55,919,788 +
9,941,840 = 210,500,616 rows. The largest single config on its own would read `10M<n<100M`.

`license` is inherited from the parent collection, whose share-alike terms fix the value.

`dataset_info` is optional, since the viewer derives the schema and the row counts from the
Parquet itself, and it carries one hazard. `datasets` compares the declared `num_examples`
against what it reads at the default verification level, so a count wrong by one row raises
`NonMatchingSplitsSizesError` for every user until the card is fixed. `upload_dataset.py card`
re-derives the counts from the footers of the staged files before it writes the card.

`num_bytes`, `download_size` and `dataset_size` are absent. They describe the materialised
Arrow dataset, which is a different quantity from the compressed Parquet on disk, and the
Hub computes and displays the real file sizes itself.

The `data_files` patterns are `<config>/<split3>-*.parquet`. A change to the shard naming, or
a move of the four directories under a `data/` prefix, moves every `path` in the block with it.
