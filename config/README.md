# `config/` — the launch configs

Each `config/<run>.yml` is a self-contained, fully launchable run definition. A run is
started from one of these files and from nothing else:

```
run --config config/<run>.yml
```

Every choice a run makes is inlined in its own file: the shared infrastructure blocks
(`llm`, `claim_commit`, `decomposition`, `rag_loop`, `chunker`, `merge`,
`translation`, `reconcile`, `packing`, `index_build`, `serialize`, `topics`), the
run-specific translation axis, the execution widths, and the output routing. The shared
blocks are duplicated across files, so the file is the complete, reproducible record of that
run and nothing has to be resolved against an external registry to launch it or to read it
back later.

Before launching, set `execution.artifact_root` to a writable directory (the vector store
alone is about 1.9 TB), or leave the key unset and export `RAGTIME_ARTIFACT_ROOT`. All
six configs must name the same root, because the corpus is shared within a family.

## The two translation knobs

The experiment has one deliberate variable, the translation axis, and it has two
independent knobs.

* **Knob 1 — search space** (`retrieval.index` = `original` | `omt` | `omt_opus`): which of
  the three shared-method indexes retrieval queries, and so which passages come back.
  Whatever is searched, the run always emits the shared original document id, so this is
  the Task-2 axis.
* **Knob 2 — reading** (`passage_lang` = `original` | `omt` | `omt_opus`): once a passage
  has been retrieved by its shared id, which rendering's text is handed to the model. This
  is the Task-1 and Task-3 axis, and it never changes the id in a run file.

Each controlled run moves exactly one knob, which keeps the contrast clean. The `e2e-*` runs
move only Knob 2 and hold the index at `original`; the `mlir-*` runs move only Knob 1 and
let the model read the native text. A run moving both knobs at once would leave no
single-knob contrast, so `family_guard` admits one only when it is marked `status: optional`;
every run shipped here is controlled.

The three renderings are `original` (native text, no machine translation), `omt` (high tier,
NLLB-200-3.3B) and `omt_opus` (low tier, Opus-MT). All three share one sentence and passage
inventory, so `omt` against `omt_opus` isolates translation tier with every other knob,
including the passage ids, held fixed.

## The fairness invariant

Within each family the shared blocks must stay byte-identical, and only the family's one
allowed knob may differ. `ragtime.config.fairness.family_guard` checks this before any GPU is
touched. A change to a value in `llm`, `claim_commit`, `decomposition`, `rag_loop`,
`chunker`, `merge`, `translation`, `reconcile`, `packing`, `index_build`, `serialize` or
`topics` must be applied identically to every run in that family, or the deltas the
experiment reports measure infrastructure rather than translation.

The comparison is on raw text, so comments count. A comment edited inside a shared block of
one family member and not the others is a divergence, and the guard rejects it.

Two consequences follow.

* **One decomposition per run, shared by Tasks 1 and 3.** The seed round reads only the
  report request, so it is identical across runs; the coverage-audit rounds read retrieved
  passages, so they diverge per rendering. The same decomposition feeds both the report and
  the nugget bank.
* **`seeds: 1`.** The agentic loop samples at temperature 0.7, so one seed is one draw and
  per-topic variance is not measurable. Every number reported from these runs is a point
  estimate, and a translation delta is reported rather than established. The trade buys
  coverage of all 103 report requests.

## Runs, knobs and submissions

Each config's `outputs` block names its deliverable as `submissions/<track>/run_N.<ext>`, where
N is the run's per-track assessment priority. Tasks 1 and 3 come from the same `e2e-*`
execution, so one run produces two submissions. The six configs below define nine outputs, all
of which were filed, one per controlled run and track.

The filed tree uses different names. A path in the Output column below is published as
`submission/task<T>/T<T>_run_N.<ext>`, with `report_generation` as task 1, `retrieval` as task 2
and `nuggetization` as task 3: `submissions/retrieval/run_3.txt` is the file
`submission/task2/T2_run_3.txt`. The configs keep the names the run used, so they stay the
record of what ran; the filed names follow each track's own convention.

`run_N` is the assessment priority, and its order is not the same on every task. Tasks 1 and 3
rank the NLLB translation first, then Opus-MT, then native text. Task 2 ranks native text first,
then NLLB, then Opus-MT. So `run_1` is the NLLB arm on Tasks 1 and 3 and the native-text arm on
Task 2.

`T2_run_1.txt`, the `mlir-original` row below, was serialized from cells executed under
`e2e-original` rather than from an `mlir-original` execution. The two configs are identical on
both knobs (`retrieval.index: original`, `passage_lang: original`), so those cells hold exactly
the retrieval an `mlir-original` run would have produced. `slurm/serialize_all_arms.sbatch`
pairs the two, and `docs/REPRODUCE.md` records the substitution.

| Config | Kind | Knob moved | Status | Task | Track | Output |
|---|---|---|---|---|---|---|
| `e2e-omt.yml` | e2e agentic | Knob 2: `passage_lang=omt` | submitted | 1 | `report_generation/` | `run_1.jsonl` |
| `e2e-omt.yml` | e2e agentic | Knob 2: `passage_lang=omt` | submitted | 3 | `nuggetization/` | `run_1.jsonl` |
| `e2e-omt-weak.yml` | e2e agentic | Knob 2: `passage_lang=omt_opus` | submitted | 1 | `report_generation/` | `run_2.jsonl` |
| `e2e-omt-weak.yml` | e2e agentic | Knob 2: `passage_lang=omt_opus` | submitted | 3 | `nuggetization/` | `run_2.jsonl` |
| `e2e-original.yml` | e2e agentic | Knob 2: `passage_lang=original` | submitted | 1 | `report_generation/` | `run_3.jsonl` |
| `e2e-original.yml` | e2e agentic | Knob 2: `passage_lang=original` | submitted | 3 | `nuggetization/` | `run_3.jsonl` |
| `mlir-original.yml` | decomposition-driven retrieval | Knob 1: `index=original` | submitted | 2 | `retrieval/` | `run_1.txt` |
| `mlir-omt.yml` | decomposition-driven retrieval | Knob 1: `index=omt` | submitted | 2 | `retrieval/` | `run_2.txt` |
| `mlir-omt-weak.yml` | decomposition-driven retrieval | Knob 1: `index=omt_opus` | submitted | 2 | `retrieval/` | `run_3.txt` |

Querying the `original` index is cross-lingual: an English query over native multilingual
passages. Querying `omt` or `omt_opus` is monolingual, an English query over
English-translated passages, and the two differ only in translation tier.

## Output formats

* **Task 1, report generation** (`report_generation/`): JSONL, one report per line, with
  `metadata`, `responses` and `references`.
* **Task 2, multilingual information retrieval** (`retrieval/`): six-column TREC run,
  `request_id Q0 doc_id rank score run_id`.
* **Task 3, autonuggetization** (`nuggetization/`): JSONL, one nugget bank per line, with
  `metadata` and `nugget_bank`.

The in-file `run_id` carried in the RAGTIME `metadata` block is capped at 25 characters;
every run id here fits within it, so the run id, the filename and `run.id` stay legible.

## `config/serving/`

The files in `config/serving/` are hardware and serving facts, not run choices. They sit
outside the fairness glob, which reads `config/*.yml` non-recursively, so they can never
join a run family.

* `models.yml` — per-model dimensions the capacity calculator needs, and the acceptance set
  of replica shapes for each translation model.
* `bamboo.yml` — discovered node capacities for one cluster, plus the measured
  `shape_calibration` rows a translate replica takes its batch knobs from.
* `shapes.yml` — the launch-shape registry: the canonical retrieval-service allocation, how
  it is launched, the canonical vLLM shape, and the measurement status of each shape.
