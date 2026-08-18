#!/usr/bin/env bash
# Select and serialize every arm of one run family, one shard per (arm, topic), resumable.
#   SER_FAMILY=e2e  sbatch slurm/serialize_all_arms.sbatch     # Task 1 + Task 3
#   SER_FAMILY=mlir sbatch slurm/serialize_all_arms.sbatch     # Task 2
#SBATCH --job-name=serialize
#SBATCH --partition=shared-cpu,public-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# The per-topic wall is dominated by a variable number of LLM dedup-confirm calls and the
# distribution has a long tail, so the wall is sized on the tail. Resume makes an over-long wall
# free and an over-short one merely wasteful.
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -uo pipefail

REPO="${RAGTIME_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 no_proxy='*' NO_PROXY='*'
# One CPU embedding model per worker process: cap its thread pool so the workers do not fight.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false

REAL="${RAGTIME_ARTIFACT_ROOT:?set the artifact root}"
FAMILY="${SER_FAMILY:-e2e}"
# The output root is stable across launches: resume only works if the second launch
# looks at the first launch's shards. The two families keep separate trees because e2e-original
# appears in both and emits different tracks in each.
OUT="${SER_OUT:-${RAGTIME_SUBMISSION_ROOT:-$REPO/logs/serialize}/$FAMILY}"
WORKERS="${SER_WORKERS:-8}"
mkdir -p "$OUT"
echo "=== $(date) node=$(hostname -s) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
echo "=== family=$FAMILY out=$OUT workers=$WORKERS"

# The three reading-knob arms: Task 1 and Task 3 come from the same cells.
E2E_SPECS=(
  --spec "e2e-original:config/e2e-original.yml:$REAL/e2e-original__original__seed0"
  --spec "e2e-omt:config/e2e-omt.yml:$REAL/e2e-omt__omt__seed0"
  --spec "e2e-omt-weak:config/e2e-omt-weak.yml:$REAL/e2e-omt-weak__omt_opus__seed0"
)
# The three search-knob arms. The third spec pairs mlir-original.yml, which declares the Task 2
# deliverable, with the e2e-original cells: the two configs are identical on both knobs
# (retrieval.index: original, passage_lang: original), so those cells hold exactly the retrieval
# behaviour that arm describes. Declare the substitution in the run description when submitting.
MLIR_SPECS=(
  --spec "mlir-omt:config/mlir-omt.yml:$REAL/mlir-omt__original__seed0"
  --spec "mlir-omt-weak:config/mlir-omt-weak.yml:$REAL/mlir-omt-weak__original__seed0"
  --spec "mlir-original:config/mlir-original.yml:$REAL/e2e-original__original__seed0"
)
case "$FAMILY" in
  e2e)  SPECS=("${E2E_SPECS[@]}") ;;
  mlir) SPECS=("${MLIR_SPECS[@]}") ;;
  *)    echo "SER_FAMILY must be e2e or mlir, got '$FAMILY'"; exit 2 ;;
esac

# The registry is the right default because it tracks a fleet that gains and loses pairs, but an
# explicit list is a sound override when the endpoints are known.
URLS_ARG=()
[ -n "${SER_URLS:-}" ] && URLS_ARG=(--urls "$SER_URLS")
echo "=== urls=${SER_URLS:-<from registry>}"

.venv/bin/python slurm/serialize_parallel.py \
  "${SPECS[@]}" "${URLS_ARG[@]}" --out "$OUT" --workers "$WORKERS"
RC=$?
echo "=== rc=$RC   $(date)"
echo "=== files:"
find "$OUT" -maxdepth 4 -name '*.jsonl' -not -path '*per_topic*' -exec ls -lh {} \;
exit $RC
