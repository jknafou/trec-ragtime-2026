# Reproducing the runs

What to obtain, how to build the corpus and the index, how to stand the two services up, how to
launch and monitor a run, and how to produce the submission files. The design behind it is in
[`../config/README.md`](../config/README.md); what each stage does is in
[`pipeline.md`](pipeline.md).

## What to obtain first

Five things have to be obtained or built separately.

* **The collection.** The RAGTIME2 document collection is distributed by the track and is not
  redistributable, so it is not here in any form: no documents, no sentences, no passages, no
  vectors, no index. Obtain access from the track organisers; the pipeline then fetches it from the
  `trec-ragtime/ragtime2` dataset repository on the Hugging Face Hub and verifies every file against
  its recorded hash. The repository is access-controlled, so authenticate to the Hub first
  (`hf auth login`, or `HF_TOKEN` in the environment) and point `HF_HOME` at a cache the compute
  nodes can also read: they run with `HF_HUB_OFFLINE=1` and a cache written elsewhere reads as a
  missing model rather than as a missing credential.
* **The request set.** The topics file is published by the organisers. Put it at the path the
  configs name, `topics/topics.all.2026.v0625-fix.jsonl`, and do not edit it; the loader absorbs the
  file's formatting quirks rather than the file being normalised on disk.
* **Model weights.** All checkpoints come from the Hub, and compute nodes have no network, so fetch
  every one of them before the first job:

  | role | checkpoint |
  | --- | --- |
  | generation, and the claim-confirmation calls | `Qwen/Qwen3.5-122B-A10B-FP8` |
  | reranker | `Qwen/Qwen3-Reranker-4B` |
  | dense leg, and the paraphrase dedups | `BAAI/bge-m3` |
  | sparse-leg pivot tokenizer | `BAAI/bge-m3-unsupervised` |
  | learned-sparse leg | `omai-research/milco-650m` |
  | late-interaction leg | `hltcoe/plaidx-large-neuclir-mtd-mix-passages-mt5xxl-engeng` |
  | sentence segmenter | `sat-3l-sm` (through `wtpsplit`) |
  | machine translation, high tier | `facebook/nllb-200-3.3B` |
  | machine translation, low tier | `Helsinki-NLP/opus-mt-es-en`, `-ru-en`, `-zh-en` |

  The two index checkpoints that carry a `*_revision` leaf in the configs are pinned to that commit;
  fetch the pinned revision, not the branch tip.

  The high-tier translation model is served through CTranslate2 rather than transformers, so it has
  to be converted once, at `float16` to match `translation.config.compute_type`, and the converted
  directory named by `execution.ct2_model_dir`. That leaf is a machine-local path outside every
  hash; make it absolute unless every job starts from the repository root.
* **The run validator.** The organisers' `hltcoe/rag-run-validator`, at the commit pinned in
  `src/ragtime/pipeline/select_serialize/submission/validate.py`:

  ```bash
  git clone <the organisers' rag-run-validator repository> tools/rag-run-validator
  git -C tools/rag-run-validator checkout eb0811b229736746306d234081b348f77b2f646b
  ```

  Put it at `tools/rag-run-validator` or point `RAGTIME_RRV_HOME` at it. Serialization refuses to
  publish Task 1 or Task 3 without it.
* **The container image.** The retrieval service and the index-assembly stage run inside an
  Apptainer image built from this repository's `uv.lock` with the `heavy` and `index` extras,
  because the three index engines are installed there and not in the plain virtual environment. The
  image is named by `RAGTIME_SIF`; build one whose definition installs the locked environment and
  nothing else.

## Environment

The environment is uv-managed; `uv.lock` and `.python-version` are the source of truth, so the same
environment reinstalls on another machine and later in time.

```bash
uv sync --frozen --extra heavy --extra index   # the GPU serving stack and the index engines
UV_PROJECT_ENVIRONMENT=.venv-chunk uv sync --frozen --extra chunk    # segmentation, separately
```

`uv sync` is exact: it removes whatever the named extras do not ask for, so the two lines above are
two environments and not two steps into one. The `chunk` extra pulls the segmenter on a torch-free
ONNX backend and cannot share an environment with `heavy`; everything else in the offline chain runs
in the GPU environment. Run everything through `uv run`, or through `.venv/bin/run` as the SLURM
scripts do.

Compute nodes have no outbound network, so every download (the collection, model weights,
dependency resolution) happens where the Hub is reachable, and jobs then run with
`HF_HUB_OFFLINE=1`. The retrieval service and the index-assembly stage run inside the Apptainer
image named by `RAGTIME_SIF`, because the three index engines are installed there.

Set the artifact root once, and identically everywhere:

```bash
export RAGTIME_ARTIFACT_ROOT=/scratch/<user>/ragtime-runs
export RAGTIME_VLLM_REGISTRY=$RAGTIME_ARTIFACT_ROOT/serving/vllm_endpoints
export RAGTIME_RSVC_REGISTRY=$PWD/logs/rsvc/registry
export RAGTIME_SIF=$RAGTIME_ARTIFACT_ROOT/containers/ragtime-gpu.sif
mkdir -p logs/slurm
```

The six configs ship a placeholder root and refuse to launch until it is replaced. Each
one states `execution.artifact_root: /path/to/artifact-root`; a config value that disagrees with
`RAGTIME_ARTIFACT_ROOT` is a hard error, because two roots mean two half-built corpora, each
carrying its own success markers. Put your root in all six files, or delete the leaf from all
six and let the environment variable stand alone. They must agree with each other as well as with
the environment, since the corpus is shared within a family.

Beyond those four, the pipeline reads a handful of environment variables that have no config leaf,
each documented where it is used: `RAGTIME_REPO` (the repository root, for the SLURM scripts),
`RAGTIME_RRV_HOME` (the validator), `RAGTIME_RSVC_HTTP` (the retrieval bridge, when the worker and
the service share no filesystem), `RAGTIME_INVENTORY_DIR` (the low-tier translation arm),
`RAGTIME_SUBMISSION_ROOT` (where the serializer writes), `RAGTIME_VLLM_URL` (a generation endpoint
stated directly rather than resolved from the registry), `RAGTIME_DEV_ROOT` and `RAGTIME_PYPATH` (a
service pinned to a checkout, so that editing the tree cannot change what an already-loaded service
does), and the `SVC_*` family that parameterizes the retrieval launcher.

## Hardware

* **Generation.** One model instance per GPU pair. The served checkpoint has two key-value heads, so
  tensor parallelism is capped at two, and its weights do not fit one card of the sizes used here. A
  pair owns a request end to end; more throughput means more pairs, never one request split across
  pairs.
* **Retrieval.** One large-memory node per searched rendering. The runs behind this repository used
  six 80 GiB-class cards, 96 CPU cores and most of a terabyte of host memory, with the
  late-interaction leg sharded one language cell per card and a reranker instance on each of the
  six; two cards is the smallest shape the launcher derives on its own. What matters more than card
  memory is host memory, because the index working set has to stay in page cache, plus node-local
  scratch for the late-interaction blobs and shared memory for the passage store.
* **Corpus build.** Chunking, merging, reconciliation, the length sidecar and packing are CPU work
  fanned over many small nodes. Translation and vectorization are GPU work. Index assembly is CPU
  work inside the container.
* **Storage.** The artifact tree runs to a few terabytes: the vector blocks dominate, with the index
  itself and the passage store well behind. The collection is about four million documents, which
  become roughly ninety million sentences and ten million passages.

## Build the corpus and the index

Nine substages run in a fixed order, each a pull work queue with three roles: `seed` fills the queue
(CPU, once), `worker` drains it (an array of tasks, GPU or CPU depending on the substage), `drive`
merges and marks the substage done (CPU, once). The order is

```
chunk -> merge -> translate_omt_identity -> translate_omt -> reconcile
      -> len_max -> packing -> vectorize -> assemble
```

The whole chain is submitted for you, correctly wired with dependencies, by launching the run
(below). To drive one substage by hand:

```bash
PREPROCESS_SUBSTAGE=chunk PREPROCESS_ROLE=seed   run --config config/e2e-original.yml --stage preprocess
PREPROCESS_SUBSTAGE=chunk PREPROCESS_ROLE=worker run --config config/e2e-original.yml --stage preprocess
PREPROCESS_SUBSTAGE=chunk PREPROCESS_ROLE=drive  run --config config/e2e-original.yml --stage preprocess
```

The `chunk` seed role is what fetches the collection, so run that one where the Hub is reachable. It
is idempotent: a complete, marked download is a no-op, and a suspected partial download is re-fetched
by deleting the store's success marker.

Two steps sit outside the chain and are driven explicitly.

* **The low-tier translation arm.** `omt_opus` is produced by the same translate stage with three
  `translation.config` leaves changed: `translate_units: sentences` (one sentence in, one
  translation out, so no merge marker is written and none has to be split back), `model_family:
  marian`, and `omt_model` naming the OPUS-MT checkpoint per source language rather than one model
  for every direction. `model_family` is not cosmetic — it selects the source-sequence layout, and
  the NLLB layout prefixes a forced target-language token that Marian's vocabulary does not contain.
  The serving shape comes from `execution.translate_shape_key: "Helsinki-NLP/opus-mt@translate"`,
  declared in `config/serving/models.yml`. The stage is pointed at an already-reconciled inventory
  through `RAGTIME_INVENTORY_DIR`, because changing `translate_units` moves `translation_hash` and
  the arm can then no longer derive that path from its own configuration. On a from-scratch build it
  runs between `reconcile` and `len_max`; `len_max` refuses to start without all three renderings and
  says which one is missing.
* **The passage store.** The by-id LMDB store is built once from the final tables and one packing.
  It has a single writer, so it is a one-shard stage rather than a fan.

The corpus is family-shared and keyed by the semantic hash of the `chunker` block, not by the whole
run configuration, so all arms of a family read one build and a change to a scheduling knob never
invalidates it. Deeper levels carry their own keys: the reconciled inventory, the packing, and the
index recipe each hash separately, so a packing change re-derives only the grouping and a query-time
knob cannot orphan a built index. Assembly publishes the index manifest only after checking that
every planned cell is present, that the parts of a cell are disjoint, and that each rendering's
passage id set is exactly the packed table's.

## Stand up the two services

### Retrieval, one service per searched rendering

```bash
SVC_INDEX=original \
SVC_EXTRA="--rerank-devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5" \
SVC_REPLICAS=1 sbatch slurm/retrieval_service.sbatch
```

`SVC_INDEX` is the index that is searched, one of `original`, `omt`, `omt_opus`. The service opens
one whole rendering and publishes a descriptor with a heartbeat into `SVC_REGISTRY` when it is ready;
bring-up takes tens of minutes, dominated by loading the sparse and dense legs, so plan on one
service living for as long as that rendering is being searched. Switching the searched index costs a
full reload; switching the rendering the model *reads* costs nothing at all, because that is a
function argument on the client side. One service therefore covers the whole `e2e-*` family, and only
the `mlir-*` family needs a service per rendering.

The two overrides above are what reproduce the shape that ran, and neither is a default.
`--rerank-devices` is the only way to get more than one reranker instance: the launcher has no knob
of its own for it, and one instance makes the cross-encoder the dominant leg. Asking for fewer
replicas than there are cards is what makes the launcher shard the four language cells across cards
instead of standing up one whole stack per card, so `SVC_REPLICAS=1` on six cards gives the shape
that ran. `slurm/README.md` covers the rest of the `SVC_*` surface, including the step launcher that
swaps the rendering inside an allocation that is already held, and `slurm/verify_service.sh`, which
answers clause by clause what a running service is actually serving. Run that before trusting a
service: several of its properties fail silently, and one of them, whether the sparse fan is open,
cannot be told from a fan that is merely inefficient by any other reading.

The first query after boot pays lazy initialization and takes minutes; send one and discard it. If
the config's `retrieval.index` disagrees with `SVC_INDEX` the launcher refuses unless
`SVC_ALLOW_OVERRIDE=1`, in which case the substitution is recorded in the published descriptor
rather than hidden by editing a config, which would break the fairness gate. `omt_opus` is named by
no `e2e-*` config, so searching it from an `e2e-*` base config needs the override.

Quiesce the pipeline workers before restarting a retrieval service. A restart under a live fleet
fails every topic that is mid-search, and a failed topic consumes an attempt.

### Generation, one job per GPU pair

```bash
FLEET_WORK=1 FLEET_CONFIGS="config/e2e-original.yml config/e2e-omt.yml config/e2e-omt-weak.yml" \
  sbatch slurm/vllm_service.sbatch
```

The job brings up one instance, publishes it into `RAGTIME_VLLM_REGISTRY` under a lease, and with
`FLEET_WORK=1` also runs the pipeline workers that consume the topic queue, sized from the card it
landed on. `FLEET_CONFIGS` is a list because all six configs name the same checkpoint: moving from
one arm to the next costs a different queue, not a model reload. Workers reach retrieval through
`RAGTIME_RSVC_REGISTRY` when they can see its queue directory, or through `RAGTIME_RSVC_HTTP` when
they cannot. The two checks are not equally strong: on the HTTP route a worker refuses to start
unless the service answers a real health request, while on the registry route it only requires a
descriptor to be present, and a descriptor outlives the service that wrote it. Check a service with
`slurm/verify_service.sh` rather than inferring it from a worker having started.

Add workers to an allocation that is already up:

```bash
srun --jobid=<J> --overlap --mem=48G slurm/pipeline_workers.sh config/e2e-omt.yml 6 <port>
```

The `--mem` is not optional. An overlapping step draws on the job's own cgroup, so a step that takes
the default can starve the engine it is feeding. Allow about 8 GiB per worker.

Generation is not batch-invariant, so a co-tenant on the same instance changes sampled output for the
same seed. Keep the seed decomposition off an instance that is serving concurrent work if you need
round-0 parity to be exactly reproducible. For the same reason, do not mix GPU architectures within
one comparison: different cards select different quantized kernels, and the same seed then yields
different text.

## Launch a run

Everything a run does is in its config file, and the file is the only argument:

```bash
run --config config/e2e-omt.yml
```

The first thing that happens is the fairness gate. The config is loaded, its family siblings are
collected from the same directory, and `family_guard` refuses the launch if any shared block differs
across the family, if more than the family's one allowed knob moved, or if the seed counts disagree.
Nothing is submitted and no GPU is touched before that passes. The config is then expanded into a job
DAG and fanned onto SLURM.

The corpus node of that DAG is what built the corpus and the index behind the runs; the online half
was driven directly, stage by stage, as below.

Useful variants:

```bash
run --config config/e2e-omt.yml --dry-run     # print the DAG and stop
run --config config/e2e-omt.yml --local       # run the DAG in this process, no SLURM
```

Resuming needs no flag: every cell is skipped if its success marker is already there, so relaunching
the same command is the resume.

The online half was driven directly, because the fleet gains and loses GPU pairs over a long run.
Seed the topic queue once, then let each pair's workers drain it:

```bash
PIPELINE_ROLE=seed run --config config/e2e-omt.yml --stage pipeline
```

A shard is one `(topic, seed)` cell. Claiming is an atomic rename, so two workers can never take the
same topic and N workers on one node are exactly as safe as one. A worker brings its clients up once
and then drains, which is what gives a GPU pair topic affinity: it owns a topic from decomposition to
the last loop.

Only one arm of a family can drain at a time when their queue names collide. The pipeline queue
is named after `passage_lang`, which is `original` for all three `mlir-*` arms, so a second `mlir-*`
arm launched against a queue the first has drained reads 103 of 103 done and exits immediately
having produced nothing. Run them in sequence, and confirm the cell directory of the arm you meant
actually grew. The `e2e-*` arms have three distinct `passage_lang` values and do not collide.

The citation scorer runs inline at the end of each topic, inside a guard that logs
`citation_scoring_failed` and continues, so a topic can finish with its citations unranked (every
score 0.0) without failing. `run --config <cfg> --stage citation_scoring` is the standalone repair:
it recomputes the scores for finished cells, and Task 1 and Task 2 both read them.

## Monitor and resume

```bash
cp slurm/monitor.env.example slurm/monitor.env      # then edit the paths
sbatch --partition=shared-cpu --cpus-per-task=2 --mem=4G --time=00:15:00 \
       --output=logs/slurm/monitor-%j.out \
       --wrap="set -a && . slurm/monitor.env && set +a && bash slurm/monitor_run.sh"
```

One poll, then exit: queue counts per arm, one row per running topic (last completed round, bank
size, committed claims, heartbeat age) and the generation rate of each live instance. Judge liveness
by `done` and `running` moving, not by a process count, and not by a heartbeat alone: the heartbeat
is re-stamped by a forked child, so a fresh one proves that child is alive rather than that the topic
is progressing. Long quiet periods are normal in an agentic loop, which is why the monitor reports
both clocks and acts on neither.

This monitor is a shell poll over the queue and the artifact tree.

Resuming needs no special mode. Every writer is atomic and every unit carries a success marker, so a
relaunch recomputes only what is missing: a finished topic returns immediately, a finished corpus
substage is a no-op, and a worker that died mid-topic has its claim reclaimed by the stale-claim
reaper that every other worker runs before every claim. Inside a topic the granularity is the round,
so an interrupted cell loses only the round that was in flight.

One consequence of that design is easy to trip over: a cell path carries no configuration hash, so
genuinely re-running a topic means moving its cell directory aside, not just relaunching. Move it,
do not delete it; a finished cell is small and may be the only complete copy of that arm.

The `QUEUE` path in `monitor.env` contains the corpus's chunker hash, the semantic hash of the
`chunker` block, which is only knowable once a config exists. It is the directory the corpus build
creates under `$RAGTIME_ARTIFACT_ROOT/corpus/<family>/`, and `run --config <cfg> --dry-run` shows
its leading characters as the corpus node's key without needing anything built.

## Serialize the submissions

Select and serialize is a deterministic projection over one finished cell and needs no retrieval, but
it does need a generation endpoint for the answer-dedup confirmation.

```bash
run --config config/e2e-omt.yml --stage select_serialize
```

For the whole family in one pool, one shard per (arm, topic) and resumable:

```bash
SER_FAMILY=e2e  sbatch slurm/serialize_all_arms.sbatch     # Task 1 and Task 3
SER_FAMILY=mlir sbatch slurm/serialize_all_arms.sbatch     # Task 2
```

The reduce step concatenates the per-topic files in the same topic order the sequential path uses and
runs the official validator on the result. A validator failure, a validator crash and a validator
that could not run are all refusals: nothing is published. The Task 2 file is additionally re-read
from the bytes just written and checked column by column before it is accepted.

Each run writes a manifest fragment beside its deliverables. Turn the fragments into the family
manifest, which carries the collection id, the run mode and the assessment priority order, with:

```bash
python -m ragtime.pipeline.select_serialize.assemble_manifest --manifest-root <root>
```

Submission paths come verbatim from each config's `outputs` block, so the record and the emitted file
cannot disagree; the number in `run_N` is that run's per-track assessment priority. The six configs
declare nine deliverables: each `e2e-*` config declares two, because Tasks 1 and 3 are two
projections of one execution, and each `mlir-*` config declares the one Task 2 run. Nine files were
filed, and they are in [`../submission/`](../submission/).

One of those nine is a substitution, and `slurm/serialize_all_arms.sbatch` makes it explicitly:
the Task 2 low-baseline run is serialized from the `e2e-original` cells rather than from
`mlir-original`. The two configs are identical on both knobs (`retrieval.index: original` and
`passage_lang: original`), so those cells hold exactly the retrieval behaviour that arm describes and
the file is what an `mlir-original` execution would have produced. Anyone re-running the family from
scratch will produce that arm the ordinary way; anyone reproducing our filed files should keep the
substitution and declare it, as we did.

## Tests

```bash
uv run pytest -m small     # fast smoke tests on tiny fixtures
uv run pytest -m full      # the full-data tier
uv run ruff check src tests
```

The small tier runs anywhere. The full tier reads the collection and the index and belongs on a
compute node.

## Adapting this to another cluster

The scripts take every path, registry and account from the environment, but not the scheduling.
Partition names (`shared-gpu`, `shared-cpu,public-cpu`), the GPU constraint strings that name
specific card models, and the twelve-hour wall on both services are the ones from the cluster the
runs were made on, and they are in the `#SBATCH` headers of `slurm/*.sbatch` and in
`src/ragtime/orchestration/slurm/templates/`. Change those first; nothing else in the repository
encodes a site.

`config/serving/` holds two files that are also site-specific: a per-node hardware table used to
size a serving shape, and a table of shapes that were measured here. A cluster with different cards
will want its own.
