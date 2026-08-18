#!/usr/bin/env bash
# Corpus-scale acceptance pass (M05d) — one sbatch for all four phases.
#
# Staged 2026-08-05 while the assemble build was still draining, so that landing the build is a
# SUBMIT, not an authoring session. Modelled directly on workqueue_worker_cpu_sif.sbatch: same
# Apptainer bring-up, same PYTHONPATH shim, same artifact-root bind/env dance.
#
# WHY SIF-WRAP EVERY PHASE, including the two that don't need it:
#   The plan routes step0/A to the bare CPU template and B1/B2 to the SIF template. Wrapping all
#   four costs nothing (pyarrow is in the image; step0/A just never touch FAISS/Seismic/PyLate)
#   and removes an entire class of launch-time mistake at midnight — one script, one code path,
#   one set of env plumbing to get right. If step0/A ever become hot, split them then.
#
# USAGE (submit in order; step0 is a HARD gate — do not launch A/B1/B2 until it is green):
#   export CONFIG=config/e2e-original.yml
#   export RAGTIME_ARTIFACT_ROOT=/path/to/your/artifact-root
#   export RAGTIME_SIF="$RAGTIME_ARTIFACT_ROOT/containers/ragtime-gpu.sif"
#   export ACC_OUT="$RAGTIME_ARTIFACT_ROOT/acceptance/$(date +%Y%m%dT%H%M)"   # stamp OUTSIDE the job
#   mkdir -p "$ACC_OUT"
#   sbatch --export=ALL,PHASE=step0 acceptance.sbatch
#   # then, only if step0 passed:
#   sbatch --export=ALL,PHASE=A  acceptance.sbatch
#   sbatch --mem=220G --array=0-<NCELLS-1> --export=ALL,PHASE=B1 acceptance.sbatch  # --mem REQUIRED, see below
#   sbatch --array=0-2          --export=ALL,PHASE=B2 acceptance.sbatch   # es/zh/ru
#
# NCELLS comes from step0's own census output — read it, never guess it. Guessing the fan is how
# a pass silently validates a subset and reports green.
#
#SBATCH --job-name=ragtime-acceptance
#SBATCH --partition=shared-cpu,public-cpu
#SBATCH --cpus-per-task=8
# MEASURED, not estimated (2026-08-06): 64G is ENOUGH for step0/A/B2 and NOT ENOUGH for B1.
# B1 opens a whole cell's parts through open_lang and holds them all at once — the largest cell
# needed ~165 GB resident and was OOM-killed at 64G, then completed at 220G. The reasoning that
# produced "64G" below was wrong in one specific way worth naming: it compared against the
# assemble worker's 80G on the grounds that reading is cheaper than building, which is true PER
# PART and irrelevant HERE — assemble holds one part, B1 holds a cell (up to 23 of them).
#
#   step0 / A / B2 : the 64G default is fine (they never open a whole cell)
#   B1             : sbatch --mem=220G ...   <-- REQUIRED, override on the command line
#
# 220G and not 250G: the node ceiling is 245 GiB, and 250G is rejected at submit.
# (Pruning cut the SPARSE leg ~3x, so B1's peak should now be lower — but do not lower this
#  default on that expectation; re-measure B1's MaxRSS with sacct first, then edit this comment.)
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail
trap 'echo "USR1 — walltime; exiting cleanly"; exit 0' USR1

CONFIG="${CONFIG:?CONFIG must be set}"
PHASE="${PHASE:?PHASE must be set (step0|A|B1|B2)}"
ACC_OUT="${ACC_OUT:?ACC_OUT must be set (a directory on beegfs, NOT /tmp)}"

# Stagger, for the same proven reason the three worker templates do: an unstaggered array walks
# the SIF's site-packages graph off one shared beegfs mount simultaneously, and beegfs answers a
# metadata storm with spurious ENOENT for files that demonstrably exist (jobs 4163611 + 4165478).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  sleep $(( (SLURM_ARRAY_TASK_ID % 60) * 3 ))
fi

TASK="${SLURM_ARRAY_TASK_ID:-0}"
OUT="${ACC_OUT}/${PHASE}_${TASK}.json"
mkdir -p "$ACC_OUT"

# --index is required by B1 (cell) and B2 (non-English language); step0/A must NOT receive it.
IDX_ARGS=()
case "$PHASE" in
  B1|B2) IDX_ARGS=(--index "$TASK") ;;
esac

echo "phase=${PHASE} task=${TASK} node=${SLURMD_NODENAME:-?} out=${OUT}"

# Bind-mounted code importable inside the SIF: the image is built --no-install-project with
# UV_NO_SYNC=1, so `import ragtime` needs a PYTHONPATH dir holding ragtime -> code/.
RAGTIME_PYPATH="$HOME/.cache/ragtime-pypath/$(basename "$PWD")"
mkdir -p "$RAGTIME_PYPATH"
ln -sfn "$PWD/code" "$RAGTIME_PYPATH/ragtime"

# The artifact root must be BOTH bound (apptainer's default bind path is /etc/localtime+/etc/hosts,
# so an off-$HOME root is invisible inside) AND re-passed via --env (--cleanenv drops the export,
# and orchestration.cli.artifact_root would then silently fall back to a repo-relative "runs").
ROOT_ARGS=()
if [ -n "${RAGTIME_ARTIFACT_ROOT:-}" ]; then
  ROOT_ARGS=(--env "RAGTIME_ARTIFACT_ROOT=${RAGTIME_ARTIFACT_ROOT}")
  case "${RAGTIME_ARTIFACT_ROOT}" in
    /*) ROOT_ARGS+=(--bind "${RAGTIME_ARTIFACT_ROOT}") ;;
  esac
fi
# ACC_OUT may live outside the artifact root; bind it explicitly so the fragment write lands.
case "${ACC_OUT}" in
  /*) ROOT_ARGS+=(--bind "${ACC_OUT}") ;;
esac

# --nv EXPOSES THE CARD. Without it the container sees no driver and any encoder that
# defaults to device="cuda" dies with "Found no NVIDIA driver" even when SLURM allocated a
# GPU — which is exactly how B2 failed twice (jobs 4279048, 4279104): the second time WITH
# --gres=gpu:1, because this script was modelled on workqueue_worker_cpu_sif.sbatch, whose
# comment says "No --nv: there is no card to expose". True there, false here.
# Added conditionally so the CPU-only phases (step0, A, B1) are unaffected: passing --nv on
# a GPU-less node fails the exec outright.
NV_ARGS=()
if [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "${SLURM_GPUS_ON_NODE:-0}" != "0" ]; then
  NV_ARGS=(--nv)
  echo "GPU allocated (${SLURM_GPUS_ON_NODE}) -> apptainer --nv"
fi

apptainer exec --cleanenv "${NV_ARGS[@]}" \
  "${ROOT_ARGS[@]}" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=${RAGTIME_PYPATH}" \
  --env "SLURMD_NODENAME=${SLURMD_NODENAME:-}" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}" \
  --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" \
  --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}" \
  "${RAGTIME_SIF:?RAGTIME_SIF must be set}" \
  uv run --frozen python -c 'from ragtime.preprocess.acceptance import main; raise SystemExit(main())' \
      --config "$CONFIG" --phase "$PHASE" --out "$OUT" "${IDX_ARGS[@]}"
# NOT the `ragtime-acceptance` console script: [project.scripts] entrypoints are never installed
# inside the container (--no-install-project), so it would die with "Failed to spawn".
