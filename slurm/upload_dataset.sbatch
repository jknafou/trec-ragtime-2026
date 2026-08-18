#!/usr/bin/env bash
# Build the publishable copy of the four released configurations: one parquet shard per
# task, written out of the corpus spine into a staging directory.
#
# The transfer is not in this job, and cannot be. Compute nodes on this cluster have no
# route to the internet, so the upload runs on the login node. What runs here is the half
# that needs CPU and disk and no network: rewriting the published columns at the release's
# own compression and row-group size, and, for `sentences`, slicing each sentence's text
# out of the document it claims a span in.
#
#   1. plan, once                sbatch slurm/upload_dataset.sbatch
#      Reads the language and document-boundary columns of every source table once and
#      writes plan.json: one entry per output file, with the row range it covers and the
#      name it will carry. This is the only step that reads a whole column, which is why
#      it is separate: it is paid once instead of by every task in the array. It also
#      prints the array width for step 2.
#
#   2. convert, one task per file   sbatch --array=0-<N-1>%8 slurm/upload_dataset.sbatch
#      The width comes from step 1; do not guess it. Each task owns exactly one output
#      file and returns immediately if that file is already built, so re-running the whole
#      array after a partial failure costs one stat per finished shard. %8 throttles the
#      concurrent tasks: they all read the same few files off the shared filesystem, and
#      more readers past that point buy contention rather than throughput.
#
#      One shard alone, when a single task failed:
#        sbatch --array=<index> slurm/upload_dataset.sbatch
#
#   3. write the card            DS_STAGE=card sbatch slurm/upload_dataset.sbatch
#      Assembles README.md in the staging directory out of the two settled sources: the
#      YAML block in docs/release/card-front-matter.md and the body under "The body,
#      verbatim" in docs/HUGGINGFACE.md. It checks both before writing (four configs and
#      no fifth, `default: true` on `sentences`, and declared row counts that match the
#      footers of whatever is already staged) and writes nothing if a check fails. Add
#      DS_CARD_ARGS='--check-only' to run the checks alone, or '--repo-id <owner>/<name>'
#      to rewrite the repository id the examples use.
#
#   4. check what is staged      DS_STAGING=... .venv/bin/python \
#                                    analysis/upload_dataset.py manifest
#      Non-zero if anything is missing, and it prints the array line that would build it.
#
#   5. transfer, on the login node, with the token in the environment and nowhere else:
#        export HF_TOKEN=...
#        DS_STAGING=... .venv/bin/python analysis/upload_dataset.py push \
#            --repo-id <owner>/<name> --dry-run    # reports what would move, uploads nothing
#        DS_STAGING=... .venv/bin/python analysis/upload_dataset.py push \
#            --repo-id <owner>/<name>
#
# Required in the environment for steps 1 and 2:
#   RAGTIME_ARTIFACT_ROOT   the artifact root holding corpus/
#   DS_CHUNKER              the corpus build key under corpus/<family>/
#   DS_STAGING              where the publishable copy is assembled; needs room for the
#                           whole release, so put it on scratch
# Optional: DS_FAMILY (default e2e), DS_FINAL, DS_PACK, DS_PYTHON, DS_PLAN_ARGS,
#           DS_STAGE=card with DS_CARD_ARGS. The card stage needs only DS_STAGING.
# Run `mkdir -p logs/slurm` first: SLURM does not create the output directory.
#
# `push` refuses to upload unless the card from step 3 is staged.
#
#SBATCH --job-name=dataset-build
#SBATCH --partition=public-cpu,shared-cpu
#SBATCH --cpus-per-task=8
# The planning pass holds one boundary list, four million integers, plus one batch of a
# narrow column. A conversion task holds one row group of output plus, for `sentences`,
# the documents it is currently reading. Neither is large; 32G is that with room for
# Parquet's own read buffers, and the honest way to trim it is to read MaxRSS off the
# first array task rather than to reason about it here.
#SBATCH --mem=32G
# Measured on the same machine class: the `sentences` config, all sixteen shards, took
# 892 s single-threaded end to end, and it is the expensive one because it joins against
# the document table. A single shard is well inside an hour; the wall is loose because
# resume makes an over-long wall free and a truncated task is not.
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
set -uo pipefail

REPO="${RAGTIME_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1
PY="${DS_PYTHON:-.venv/bin/python}"
# HF_HUB_OFFLINE is belt and braces: nothing in the build stage opens a socket, and this
# makes a stray import that tries to fail loudly here rather than hang on a node with no
# route out.
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 no_proxy='*' NO_PROXY='*'
# Parquet decompression is threaded; leave it a few threads rather than the whole
# allocation, because several array tasks share the node.
export OMP_NUM_THREADS=4

: "${DS_STAGING:?set DS_STAGING to the staging directory}"
mkdir -p "$DS_STAGING"

echo "=== $(date) node=$(hostname -s) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK}"
echo "=== staging=$DS_STAGING"

if [ "${DS_STAGE:-}" = "card" ]; then
  echo "=== stage: card"
  # shellcheck disable=SC2086
  "$PY" analysis/upload_dataset.py card ${DS_CARD_ARGS:-}
  RC=$?
elif [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "=== stage: plan"
  "$PY" analysis/upload_dataset.py inspect || exit $?
  # shellcheck disable=SC2086
  "$PY" analysis/upload_dataset.py plan ${DS_PLAN_ARGS:-}
  RC=$?
else
  echo "=== stage: convert shard ${SLURM_ARRAY_TASK_ID}"
  "$PY" analysis/upload_dataset.py convert --shard "${SLURM_ARRAY_TASK_ID}"
  RC=$?
fi

echo "=== rc=$RC   $(date)"
exit $RC
