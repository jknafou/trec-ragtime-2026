#!/usr/bin/env bash
# Monitoring rollup (CPU): the terminal afterok node that rolls each stage's counters
# into the reported metrics. No logic beyond bring-up and `run --stage`.
#
#SBATCH --job-name=ragtime-monitor
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG must be set}"
apptainer exec --cleanenv "${RAGTIME_SIF:?RAGTIME_SIF must be set}" \
    uv run --frozen run --config "$CONFIG" --stage monitor
