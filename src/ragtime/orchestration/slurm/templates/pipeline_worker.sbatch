#!/usr/bin/env bash
# Online pipeline worker: one SLURM array task is one GPU node draining the topic queue
# for one seed. No logic beyond `run --stage`; the worker attaches to a vLLM instance
# published in the endpoint registry and runs the driver.
#
# NOT THE PATH THE FILED RUNS TOOK. The online half was driven directly by
# `slurm/pipeline_workers.sh` and `slurm/vllm_service.sbatch`, which run this same
# `run --stage pipeline` with `PIPELINE_ROLE=worker` inside an allocation that already
# holds the model. Two things keep this template from being that path, both recorded at
# `cli._TEMPLATE_OF`: it hardcodes `--stage pipeline` and never reads the `$STAGE` its
# submitter exports, so it cannot serve the citation-scoring and select-serialize nodes
# routed here as well; and the submitter exports no `PIPELINE_ROLE`, so a task takes the
# default "all" role and would seed, work and drive at once.
#
#SBATCH --job-name=ragtime-pipe-worker
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="nvidia_a100_80gb_pcie|nvidia_h100_nvl|nvidia_h200_nvl|nvidia_rtx_pro_6000_blackwell"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail
trap 'echo "USR1: walltime reached; the in-flight cell is reclaimed by the reaper"; exit 0' USR1

CONFIG="${CONFIG:?CONFIG must be set}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
# No `--topics`: the pipeline stage claims its shard off the work queue by atomic rename,
# so the caller cannot pick the topic and `run --stage pipeline` refuses the flag.
apptainer exec --cleanenv --nv "${RAGTIME_SIF:?RAGTIME_SIF must be set}" \
    uv run --frozen run --config "$CONFIG" --stage pipeline --seed "$SEED"
