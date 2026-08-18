#!/usr/bin/env bash
# Offline work-queue worker (CPU). One array task self-claims shards from the shared
# queue and runs `run --stage preprocess`. The CPU substages (chunking, the unit map,
# the English identity translation arm, reconcile, len_max and packing) need no GPU and
# no container: they run the light `chunk` extra directly under uv, leaving the scarce
# cards for translation and the index build. No logic beyond bring-up and
# `run --stage`; the claim, heartbeat, validate and poison semantics live in
# `ragtime.orchestration.slurm.workqueue`.
#
#SBATCH --job-name=ragtime-wq-worker-cpu
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=8
# 32G, about 4 GB per pool child. An earlier 16G was derived from English shards alone
# and did not hold for the Chinese, Russian and Spanish tail, where every worker in a
# 120-task fleet was OOM-killed. That failure is invisible and expensive: an OOM-killed
# pool child makes `imap` block forever, the heartbeat keeps stamping, SLURM keeps
# reporting RUNNING, and the queue reads as neither failed nor progressing.
#SBATCH --mem=32G
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

# Compute-node environment, set explicitly because an sbatch environment is not a login
# shell and the interactive profile exports are not guaranteed even under --export=ALL:
#   UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR - a site-wide default may be unwritable here.
#   UV_NO_SYNC   - the shared venv is already synced; never attempt a resolve on a
#                  worker, which has no network.
#   HF_HUB_OFFLINE - compute nodes have no network, so models and the corpus snapshot
#                  resolve from the warm cache (populate it on the login node first).
# A `${VAR:-default}` would not be enough for the cache directory: the default only
# fires when the variable is unset, and an inherited but unwritable value makes uv abort
# in under a second. So probe writability and override a bad inherited value.
if [ -z "${UV_CACHE_DIR:-}" ] || ! mkdir -p "$UV_CACHE_DIR" 2>/dev/null || [ ! -w "$UV_CACHE_DIR" ]; then
  export UV_CACHE_DIR="$HOME/.cache/uv"
  mkdir -p "$UV_CACHE_DIR"
fi
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"
export UV_NO_SYNC=1
export HF_HUB_OFFLINE=1
# Line-buffer stdout. Without this Python block-buffers to a file, so small progress
# lines sit unflushed while multi-KB tracebacks flush the buffer immediately, and the
# log then shows only failures and reads like a dying fleet that is in fact healthy.
export PYTHONUNBUFFERED=1

# Stagger the array's import burst: every task's pool children import large libraries
# off the one shared venv, and a wide array turns that into a metadata storm under which
# the filesystem returns spurious ENOENT for files that demonstrably exist. Spreading
# the starts removes the burst and costs nothing, since a worker then runs for hours.
# Deterministic per task, so a resume reproduces the same schedule.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  sleep $(( SLURM_ARRAY_TASK_ID * 3 ))
fi

# No container and no GPU request: the light `chunk` extra runs straight under uv.
uv run --frozen --extra chunk run --config "$CONFIG" --stage preprocess
