#!/usr/bin/env bash
# Offline work-queue worker (GPU): one array task self-claims corpus shards from the
# shared queue and runs `run --stage preprocess`. No logic beyond bring-up and
# `run --stage`; the atomic claim, heartbeat, validate and poison semantics live in
# `ragtime.orchestration.slurm.workqueue`.
#
# The OR-set races every calibrated card whose measured max_batch_tokens clears the
# semantic bucket token budget of 16384. The RTX 3090 is excluded because its measured
# ceiling is below that budget, which would re-split every bucket and stop a shard
# composing identically across cards.
#SBATCH --job-name=ragtime-wq-worker
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="nvidia_a100_80gb_pcie|nvidia_h100_nvl|nvidia_h200_nvl|nvidia_rtx_pro_6000_blackwell|nvidia_geforce_rtx_5090"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

# Stagger the array start. Every task pulls a multi-GB translation checkpoint off the
# shared filesystem at bring-up, and an unstaggered wide array turns that into one
# synchronized read storm on the metadata server. Deterministic per task, so a resume
# reproduces the same schedule, and capped at about three minutes by the modulo so a
# high task index does not idle away its own walltime.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  sleep $(( (SLURM_ARRAY_TASK_ID % 60) * 3 ))
fi

# Which silicon this task landed on, recorded in the log and passed into the worker,
# which stamps it onto every shard's done/<shard>.result. The constraint above is
# heterogeneous and translation output is not bit-identical across architectures, so the
# artifact must be able to name its hardware.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
RAGTIME_GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
echo "GPU=${RAGTIME_GPU_MODEL:-unknown} node=${SLURMD_NODENAME:-?} task=${SLURM_ARRAY_TASK_ID:-none}"

# Make the bind-mounted source importable inside the container. The image is built with
# `uv sync --no-install-project` and sets UV_NO_SYNC=1, so it is keyed to uv.lock alone
# and a code edit needs no rebuild; nothing installs the project inside it, so `import
# ragtime` needs a PYTHONPATH entry pointing at the src directory.
RAGTIME_PYPATH="$REPO/src"

# The artifact root, when it is a site path rather than the repo-relative default. Both
# lines are needed: --bind because a root outside $HOME/$PWD is invisible inside the
# container under this cluster's bind configuration, and --env because --cleanenv drops
# the exported value, which would make the worker fall back to the repo-relative "runs".
ROOT_ARGS=()
if [ -n "${RAGTIME_ARTIFACT_ROOT:-}" ]; then
  ROOT_ARGS=(--env "RAGTIME_ARTIFACT_ROOT=${RAGTIME_ARTIFACT_ROOT}")
  case "${RAGTIME_ARTIFACT_ROOT}" in
    /*) ROOT_ARGS+=(--bind "${RAGTIME_ARTIFACT_ROOT}") ;;
  esac
fi

# --cleanenv keeps the host environment out of the container, so anything the worker
# needs is passed explicitly. PREPROCESS_ROLE and PREPROCESS_SUBSTAGE are load-bearing:
# `run --stage preprocess` reads both, and without them the role defaults to "all" and
# the substage filter to none, so every array task would seed, work and drive every
# substage over one queue. PYTHONUNBUFFERED keeps the last log block of a killed worker.
# WORKERS_PER_TASK is how many worker processes share this task's GPU. The default of 1
# leaves every existing arm untouched; a small MT model inverts the bottleneck, since a
# slow tokenizer on one core leaves the card partly idle, and N processes claiming from
# the same queue need no new coordination because the claim is an atomic rename. Only the
# `worker` role fans out: `seed` and `drive` are single-shot by construction.
WORKERS_PER_TASK="${WORKERS_PER_TASK:-1}"
[ "${PREPROCESS_ROLE}" = "worker" ] || WORKERS_PER_TASK=1

run_one() {
  apptainer exec --cleanenv --nv \
    "${ROOT_ARGS[@]}" \
    --env "PREPROCESS_ROLE=${PREPROCESS_ROLE:?PREPROCESS_ROLE must be set (seed|worker|drive)}" \
    --env "PREPROCESS_SUBSTAGE=${PREPROCESS_SUBSTAGE:?PREPROCESS_SUBSTAGE must be set}" \
    --env "WQ_DIR=${WQ_DIR}" \
    --env "RAGTIME_INVENTORY_DIR=${RAGTIME_INVENTORY_DIR:-}" \
    --env "PYTHONUNBUFFERED=1" \
    --env "PYTHONPATH=${RAGTIME_PYPATH}" \
    --env "RAGTIME_GPU_MODEL=${RAGTIME_GPU_MODEL:-}" \
    --env "SLURMD_NODENAME=${SLURMD_NODENAME:-}" \
    --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}" \
    --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" \
    --env "RAGTIME_WORKER_SLOT=${1}" \
    "${RAGTIME_SIF:?RAGTIME_SIF must be set}" \
    uv run --frozen python -c 'from ragtime.orchestration.cli import main; raise SystemExit(main())' \
        --config "$CONFIG" --stage preprocess
}

if [ "$WORKERS_PER_TASK" -le 1 ]; then
  run_one 0
else
  echo "launching $WORKERS_PER_TASK worker processes on one GPU"
  pids=""
  for slot in $(seq 0 $((WORKERS_PER_TASK - 1))); do
    # Stagger the bring-ups so N processes do not pull the checkpoints off the shared
    # filesystem in lockstep, for the same reason the array start is staggered above.
    sleep "$((slot * 5))"
    run_one "$slot" &
    pids="$pids $!"
  done
  # `wait <pid>` per child rather than a bare `wait`, which returns only the last child's
  # status and would report success while another worker died.
  rc=0
  for p in $pids; do
    wait "$p" || rc=$?
  done
  echo "all $WORKERS_PER_TASK workers exited; worst rc=$rc"
  exit "$rc"
fi
# Not the `run` console script: it is a [project.scripts] entrypoint that exists only
# once the project is installed, and this container never installs it. Not
# `python -m ragtime.orchestration.cli` either, because `ragtime.orchestration.__init__`
# already imports cli, so runpy would execute the module a second time as `__main__`.
