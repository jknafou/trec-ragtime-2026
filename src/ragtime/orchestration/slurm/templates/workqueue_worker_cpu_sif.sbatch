#!/usr/bin/env bash
# Offline work-queue worker (CPU, inside the container). One array task self-claims
# shards from the shared queue and runs `run --stage preprocess`; the assemble half of
# the index build runs here. No logic beyond bring-up and `run --stage`.
#
# Why a third template. The plain CPU template runs on the host with no container, but
# faiss, pyseismic-lsr, pylate and fast-plaid live in the `index` extra and exist only
# inside the image, so a worker there dies on import before it claims anything. The GPU
# template has the image but hard-requests a card behind a five-way constraint, and
# assembly touches no GPU: its longest pole is the CPU-bound sparse index build, so
# queueing it behind scarce cards would idle one per part and put assembly on the encode
# critical path. Assembly is also configured to run on CPU for determinism, so
# scheduling and reproducibility point the same way.
#
#SBATCH --job-name=ragtime-wq-worker-cpu-sif
# Both CPU partitions: large nodes sit idle in each and a 12-hour request is valid in
# both, so SLURM takes whichever frees first.
#SBATCH --partition=shared-cpu,public-cpu
# 32 cores, the width the longest pole was measured at, and a quarter of a large node so
# four tasks can share one.
#SBATCH --cpus-per-task=32
# 80G, sized from the worker's peak rather than from the buffer handed to the index
# writer: fast_plaid.add_documents materializes an fp32 tensor whatever the input dtype,
# so it peaks at a multiple of that buffer. 80G covers the worst part at both the
# shipped float16 store dtype and float32. The margin is deliberate: the CPU chunk
# template's original 16G came from an English-only measurement that did not hold for
# the non-English tail and OOM-killed a whole fleet in a way that looked like a hang.
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail
# This trap does not requeue the in-flight shard, because bash cannot reach the Python
# worker's claimed shard. Recovery is via the heartbeat reaper: the killed worker stops
# refreshing, so the next launch's reap_stale returns the shard to pending/ after
# max_age. The trap only ends the job cleanly, with no spurious FAILED state.
trap 'echo "USR1: walltime reached; the reaper reclaims the shard after max_age"; exit 0' USR1

CONFIG="${CONFIG:?CONFIG must be set}"
WQ_DIR="${WQ_DIR:?WQ_DIR must be set (on the shared filesystem, never /tmp)}"
# Repo checkout this task runs from; override when the job is submitted from elsewhere.
REPO="${RAGTIME_REPO:-$PWD}"

# Stagger the array start, for the same reason as both sibling templates: an unstaggered
# wide array pulls the image and walks the whole site-packages graph off the one shared
# mount simultaneously, and the filesystem answers a metadata storm with spurious ENOENT
# for files that demonstrably exist. Deterministic per task, capped at about three
# minutes so a high task index does not idle away its own walltime.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  sleep $(( (SLURM_ARRAY_TASK_ID % 60) * 3 ))
fi

# The silicon this task landed on, recorded honestly as "none": there is no nvidia-smi
# on a CPU node, and `saturate.worker_provenance` drops empty values, so leaving it unset
# would make "assembled on a GPU-less node" indistinguishable from "we failed to record
# it". The other three provenance variables are real facts here and are forwarded below.
RAGTIME_GPU_MODEL="none"
echo "GPU=${RAGTIME_GPU_MODEL} node=${SLURMD_NODENAME:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"

# Make the bind-mounted source importable inside the container: the image is built with
# `uv sync --no-install-project`, so nothing installs the project inside it and `import
# ragtime` needs a PYTHONPATH entry pointing at the src directory.
RAGTIME_PYPATH="$REPO/src"

# The artifact root, when it is a site path rather than the repo-relative default. Both
# lines are needed, exactly as in the GPU template: --bind because a root outside
# $HOME/$PWD is invisible inside the container, and --env because --cleanenv drops the
# exported value, which would make the worker fall back to the repo-relative "runs".
ROOT_ARGS=()
if [ -n "${RAGTIME_ARTIFACT_ROOT:-}" ]; then
  ROOT_ARGS=(--env "RAGTIME_ARTIFACT_ROOT=${RAGTIME_ARTIFACT_ROOT}")
  case "${RAGTIME_ARTIFACT_ROOT}" in
    /*) ROOT_ARGS+=(--bind "${RAGTIME_ARTIFACT_ROOT}") ;;
  esac
fi

# --cleanenv keeps the host environment out of the container, so anything the worker
# needs is passed explicitly. No --nv: there is no card to expose and asking for one on
# a CPU node fails the exec. PREPROCESS_ROLE and PREPROCESS_SUBSTAGE are load-bearing,
# since `run --stage preprocess` reads both and without them every array task would
# seed, work and drive every substage. The three SLURM variables are read by
# `saturate.worker_provenance`, so without them a part cannot say where it was built.
# PYTHONUNBUFFERED keeps the last log block of a killed worker.
apptainer exec --cleanenv \
  "${ROOT_ARGS[@]}" \
  --env "PREPROCESS_ROLE=${PREPROCESS_ROLE:?PREPROCESS_ROLE must be set (seed|worker|drive)}" \
  --env "PREPROCESS_SUBSTAGE=${PREPROCESS_SUBSTAGE:?PREPROCESS_SUBSTAGE must be set}" \
  --env "WQ_DIR=${WQ_DIR}" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=${RAGTIME_PYPATH}" \
  --env "RAGTIME_GPU_MODEL=${RAGTIME_GPU_MODEL}" \
  --env "SLURMD_NODENAME=${SLURMD_NODENAME:-}" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}" \
  --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" \
  --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}" \
  "${RAGTIME_SIF:?RAGTIME_SIF must be set}" \
  uv run --frozen python -c 'from ragtime.orchestration.cli import main; raise SystemExit(main())' \
      --config "$CONFIG" --stage preprocess
# Not the `run` console script and not `python -m ragtime.orchestration.cli`, for the
# reasons the GPU template's closing note gives: the console script exists only once the
# project is installed, which this container never does, and runpy would execute the
# module a second time as `__main__`.
