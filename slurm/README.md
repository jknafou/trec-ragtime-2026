# SLURM launchers

The scripts, in the order a run uses them. Everything is parameterized by environment variable;
no path, node name or account is baked in.

| script | what it does |
| --- | --- |
| `retrieval_service.sbatch` | One long-lived retrieval service for one corpus rendering. |
| `retrieval_service_step.sh` | Switches the rendering inside an allocation that is already held. |
| `verify_service.sh` | Checks a running retrieval service clause by clause, with the evidence. |
| `retrieval_probe.py` | One query against a live service, printing its own per-leg clocks. |
| `vllm_service.sbatch` | One vLLM instance on one GPU pair, plus the pipeline workers it feeds. |
| `pipeline_workers.sh` | Adds N pipeline workers to a vLLM allocation that is already running. |
| `monitor_run.sh` | One poll of the live run: queue state, per-topic progress, generation rate. |
| `serialize_all_arms.sbatch` | Runs `serialize_parallel.py`, the per-topic fan that builds the submission files. |
| `serialize_parallel.py` | The fan itself: one shard per (arm, topic) through the real `project()`. |
| `upload_dataset.sbatch` | Builds the released dataset shards and the card shipped with them. |

## Before the first submission

```bash
mkdir -p logs/slurm                 # SLURM will not create the --output directory
export RAGTIME_ARTIFACT_ROOT=/scratch/<user>/ragtime-runs
export RAGTIME_VLLM_REGISTRY=$RAGTIME_ARTIFACT_ROOT/serving/vllm_endpoints
export RAGTIME_RSVC_REGISTRY=$PWD/logs/rsvc/registry
```

Every script takes the repository root from `RAGTIME_REPO`, falling back to `SLURM_SUBMIT_DIR`,
so submit from the repository root or set the variable.

## Order to run them in

### 1. Retrieval, one service per rendering

```bash
SVC_INDEX=original sbatch slurm/retrieval_service.sbatch
```

`SVC_INDEX` is Knob 1, the index that is *searched* (`original`, `omt`, `omt_opus`). Switching it
costs a full reload of roughly half an hour, so one service serves one rendering for as long as
that rendering is being searched. Knob 2, `passage_lang`, is the rendering the model *reads*; it is
a function argument on the client side and needs no restart, so a single service covers every arm
of the `e2e-*` family.

The service publishes a descriptor into `SVC_REGISTRY` when it is ready. It needs 80 GiB-class
cards, about 490 GiB of host memory (the page cache holds the index working set) and roughly
250 GiB of node-local disk for the PLAID blob mirror. Useful overrides: `SVC_CONFIG`,
`SVC_REPLICAS`, `SVC_CELL_FAN`, `SVC_PART_WORKERS`, `SVC_CELL_PROCS`, `SVC_STORE_LOCAL`.

`SVC_ALLOW_OVERRIDE=1` permits `SVC_INDEX` to differ from the config's `retrieval.index`; the
substitution is recorded in the published descriptor rather than hidden by editing a config.

The service runs inside the project's Apptainer image (`RAGTIME_SIF`) and starts
`ragtime.devkit.rsvc`.

Reranking is one instance per card, which `retrieval_service.sbatch` has no knob for; it is passed
through `SVC_EXTRA="--rerank-devices cuda:0,...,cuda:5"`. Without it the service builds a single
instance and the reranker becomes the dominant leg. What the cross-encoder rescores is the fused
top `retrieval.reranker.depth`, in every submitted run.

### 1b. Switching the rendering without giving up the allocation

```bash
RSVC_JOBID=<allocation> SVC_INDEX=omt_opus \
  SVC_INDEX_ROOT=$RAGTIME_ARTIFACT_ROOT/corpus/e2e/<chunker>/corpus-preprocess/final/<final>/index/<build> \
  bash slurm/retrieval_service_step.sh
```

An `srun --overlap` step into an allocation that already holds the node, replacing the service and
keeping the allocation. It picks the config for the rendering, refuses to start unless the index
tree is complete (a partial tree ranks differently and reports nothing), and refuses to reuse a
PLAID mirror path, which would serve the previous rendering's blobs under the new label.

For `omt` and `omt_opus` the step keeps `config/e2e-original.yml` as the base and sets
`SVC_ALLOW_OVERRIDE=1`; the substitution is then published in the descriptor as `rendering_source`
rather than hidden by editing a config. Which base config is used reaches two published descriptor
fields and no ranking: the whole `retrieval` block is shared across a family, and the fields that do
differ — `passage_lang` and the output routing — are Knob 2 and downstream of retrieval, which
returns ids and scores and reads no text.

Quiesce the pipeline fleet first. A restart under live workers spends their retries on the reload,
and a topic whose attempts all fall inside it ends failed rather than being retried.

### 1c. Checking that it is serving what you think

```bash
srun --jobid=<allocation> --overlap -N1 -n1 --mem=8G --chdir=$PWD \
     env WANT_INDEX=omt_opus BOOT=logs/slurm/rsvc-<jobid>.out \
     bash slurm/verify_service.sh --live
```

Each clause prints PASS or FAIL and the evidence it read, and a clause that cannot be answered
fails rather than staying quiet. Statically: descriptor freshness first (`ready: true` outlives the
service that wrote it), the published rendering, late interaction on a GPU, `low_memory` false, one
reranker per card, the sparse fan gate, and the rerank depth. With `--live`, additionally: the
passage store in RAM, the CPU legs fanned, `sparse_cores` off 1.00, and the number of reranker
instances that actually scored the query.

The `sparse_cores` clause is the only one that separates a closed fan gate from a fan that is
merely inefficient: with the gate shut it reads exactly 1.00 in every language cell and nothing
else in the reply differs. It lives in `cpu_per_cell`, not in `timing`.

`WANT_INDEX` has no default, because a default is how a check passes against the one service that
happens to be right while the reader believes a different one was verified. The `--live` half needs
its own small `--mem`: an overlapping step draws on the service's cgroup, and one that took the
default has been measured holding 72 GiB and killing the service it was measuring.

`retrieval_probe.py` is the client it uses, and is worth running alone to read a single query's
per-leg cost. Discard the first query after a boot, which pays lazy initialization, and read the
`TIMING` legs rather than the client wall, which includes time queued behind other clients.

`rsvc_descriptor.example.json` is one such descriptor, and shows what a client resolves a service
through: the boot phases and their costs, the language cell part counts, the fan widths and the
host memory picture. Its measurements come from a real boot in the single-stack shape, on six
80 GiB cards searching the `original` rendering; the host name, job id, process id and paths in it
are placeholders. It predates three fields the current code also writes (`part_procs`,
`cell_workers`, `cell_plan`).

### 2. vLLM, one job per GPU pair

```bash
FLEET_WORK=1 FLEET_CONFIGS="config/e2e-original.yml config/e2e-omt.yml config/e2e-omt-weak.yml" \
  sbatch slurm/vllm_service.sbatch
```

One instance per pair: the model's two key-value heads cap tensor parallelism at 2, and a second
instance on the same cards starves both key-value caches. Scale out by submitting more jobs, never
by splitting one topic across pairs.

With `FLEET_WORK=1` the job also runs the pipeline workers that consume the topic queue, sized from
the card it landed on (16 concurrent workers on a 141 GiB card, 8 otherwise, overridable with
`SVC_TOTAL_WORKERS`). `FLEET_CONFIGS` is a list because every config names the same checkpoint:
draining one arm and starting the next costs no model reload.

Workers reach retrieval through `RAGTIME_RSVC_REGISTRY` when they can see the service's queue
directly, or through `RAGTIME_RSVC_HTTP` when they cannot. `RSVC_SSH_HOST` opens an SSH forward
first, for the case where the two services sit on clusters that share no filesystem.

To change a vLLM launch flag without giving up the allocation: `touch logs/vllm/RESTART_VLLM`.

### 3. More workers on an allocation that is already running

```bash
srun --jobid=<J> --overlap --mem=48G slurm/pipeline_workers.sh config/e2e-omt.yml 6 <port>
```

`vllm_service.sbatch` calls this itself, so it is only needed for a job that started before the
queue was populated. The `--mem` is not optional: an overlapping step draws on the job's own cgroup
and can starve the engine it is feeding. Allow about 8 GiB per worker.

Two workers cannot take the same topic: a claim is an atomic rename, and the first one wins.

### 4. Watching the run

```bash
cp slurm/monitor.env.example slurm/monitor.env    # then edit the paths
sbatch --partition=shared-cpu --cpus-per-task=2 --mem=4G --time=00:15:00 \
       --output=logs/slurm/monitor-%j.out \
       --wrap="set -a && . slurm/monitor.env && set +a && bash slurm/monitor_run.sh"
```

One poll, then exit. It reports queue counts per arm, one row per running topic (last completed
round, nugget bank size, committed claims, heartbeat age) and the generation rate of each live vLLM
instance. Judge liveness by `done` and `running` moving, not by a process count.

A heartbeat is re-stamped by a forked child, so a fresh heartbeat means that child is alive, not
that the worker is making progress. The per-topic `idle-Nm` flag is the second clock: the age of
the newest file under the topic's cell. Long gaps are normal in an agentic loop, which is why the
monitor reports both and acts on neither.

### 5. Building the submission

```bash
SER_FAMILY=e2e  sbatch slurm/serialize_all_arms.sbatch     # Task 1 and Task 3
SER_FAMILY=mlir sbatch slurm/serialize_all_arms.sbatch     # Task 2
```

The `e2e` family emits report generation (Task 1) and autonuggetization (Task 3) from the same
cells; the `mlir` family emits multilingual information retrieval (Task 2).

One shard per (arm, topic), all arms of a family in one pool, resumable: a shard whose declared
outputs already carry their `_SUCCESS` companion is skipped before a client is built, so a job that
hits its wall resumes exactly where it stopped. The reduce step concatenates the per-topic files in
the same lexicographic topic order the sequential path uses and runs the official validator on the
result.

`SER_WORKERS` (default 8) is the ceiling on concurrent requests to the fleet; the shards are spread
round-robin over the live endpoints rather than piled onto one. `SER_URLS` overrides registry
discovery with an explicit comma-separated list. `SER_OUT` overrides the output tree, which is
otherwise stable per family so that resume works across launches.

### 6. Building the released dataset

```bash
DS_STAGING=/scratch/<user>/staging sbatch slurm/upload_dataset.sbatch    # plan, once
DS_STAGING=... sbatch --array=0-<N-1>%8 slurm/upload_dataset.sbatch      # convert, one task/file
DS_STAGING=... DS_STAGE=card sbatch slurm/upload_dataset.sbatch          # the dataset card
```

The build half of the release: one parquet shard per output file, written out of the corpus spine
into a staging directory, then the card that ships with them. The array width comes from the
planning step and must not be guessed. Each task owns one output file and returns immediately if
that file is already built, so re-running the whole array after a partial failure is cheap.

The transfer half is not a batch job and cannot be. Compute nodes here have no route to the
internet, so `analysis/upload_dataset.py push` runs on the login node with the access token in the
environment, and it refuses to run inside an allocation. That script's header carries the full
sequence, including the manifest check and the dry run.
