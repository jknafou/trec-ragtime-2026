#!/bin/bash
# Long-lived retrieval service for one corpus rendering (Knob 1).
#   SVC_INDEX=original|omt|omt_opus sbatch slurm/retrieval_service.sbatch
# Switching rendering costs a full reload; passage_lang (Knob 2) is free and needs no restart.
#SBATCH --job-name=rsvc
#SBATCH --partition=shared-gpu
# A whole rendering needs 80 GiB-class cards; 24 and 32 GiB cards run out of VRAM in PLAID.
#SBATCH --constraint="nvidia_a100_80gb_pcie|nvidia_rtx_pro_6000_blackwell|nvidia_h200_nvl"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=490G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -uo pipefail

REPO="${RAGTIME_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1

export RAGTIME_ARTIFACT_ROOT="${RAGTIME_ARTIFACT_ROOT:?set the artifact root}"
RAGTIME_SIF="${RAGTIME_SIF:-$RAGTIME_ARTIFACT_ROOT/containers/ragtime-gpu.sif}"
# The devkit writes to a sibling of the artifact root, so binding the root alone is not enough.
RAGTIME_DEV_ROOT="${RAGTIME_ARTIFACT_ROOT}-dev"
mkdir -p "$RAGTIME_DEV_ROOT"

# Which rendering this service searches. No default: a wrong index answers plausibly and silently.
SVC_INDEX="${SVC_INDEX:?SVC_INDEX must be one of original, omt, omt_opus}"
SVC_CONFIG="${SVC_CONFIG:-config/e2e-original.yml}"
SVC_QUEUE="${SVC_QUEUE:-$REPO/logs/rsvc/queue/$SVC_INDEX}"
SVC_REGISTRY="${SVC_REGISTRY:-$REPO/logs/rsvc/registry}"
SVC_NAME="${SVC_NAME:-rsvc-${SVC_INDEX}-${SLURM_JOB_ID:-manual}}"
SVC_SLOT="${SVC_SLOT:-$SVC_NAME}"
SVC_TIER="${SVC_TIER:-manual}"
SVC_GPU_MODEL="${SVC_GPU_MODEL:-unpinned}"
SVC_DEGRADED="${SVC_DEGRADED:-none}"
# The config's retrieval.index must match SVC_INDEX unless the override is explicit; the
# substitution is then recorded in the published descriptor.
SVC_ALLOW_OVERRIDE="${SVC_ALLOW_OVERRIDE:-0}"
SVC_EXTRA="${SVC_EXTRA:-}"

SVC_GPU_PLAID="${SVC_GPU_PLAID:-cuda:0}"
SVC_GPU_MTD="${SVC_GPU_MTD:-cuda:1}"
SVC_GPU_DENSE="${SVC_GPU_DENSE:-cuda:1}"
SVC_GPU_SPARSE="${SVC_GPU_SPARSE:-cuda:1}"
SVC_GPU_RERANK="${SVC_GPU_RERANK:-cuda:1}"
SVC_RERANK_BATCH="${SVC_RERANK_BATCH:-8}"
SVC_PLAID_LOW_MEMORY="${SVC_PLAID_LOW_MEMORY:-false}"
# Boolean gate on the sparse process fan. 0 disables it and leaves the sparse leg on one core;
# the per-part budget is a separate knob inside the service, not this one.
SVC_CELL_PROCS="${SVC_CELL_PROCS:-1}"

# ---------------------------------------------------------------------------
# Size the service from the allocation. Every knob below is a floor that an explicit SVC_* still
# overrides, so the tests use set-ness (${VAR+x}); an eager ${VAR:-N} default would make the whole
# derivation dead code.
# ---------------------------------------------------------------------------
_cpus="${SLURM_CPUS_PER_TASK:-0}"; [ "$_cpus" = "0" ] && _cpus="$(nproc)"
_gpus="${SLURM_GPUS_ON_NODE:-1}"; [ "$_gpus" = "0" ] && _gpus=1

# One replica is a whole stack (index legs, fusion and its own reranker) pinned to one card.
# One replica per card; a second replica on the same card costs latency and buys no throughput.
if [ -z "${SVC_REPLICAS+x}" ]; then
  _by_cpu=$(( _cpus / 12 ))
  _by_gpu=$_gpus
  SVC_REPLICAS=$(( _by_cpu < _by_gpu ? _by_cpu : _by_gpu ))
  [ "$SVC_REPLICAS" -lt 1 ] && SVC_REPLICAS=1
fi
_per_rep=$(( _cpus / SVC_REPLICAS )); [ "$_per_rep" -lt 1 ] && _per_rep=1

# Width of the language-cell fan inside a replica, capped at the four cells. 1 is the sequential
# path and is never derived: it has to be asked for.
if [ -z "${SVC_CELL_FAN+x}" ]; then
  SVC_CELL_FAN=$(( _per_rep / 3 ))
  [ "$SVC_CELL_FAN" -lt 2 ] && SVC_CELL_FAN=2
  [ "$SVC_CELL_FAN" -gt 4 ] && SVC_CELL_FAN=4
fi
if [ -z "${SVC_LI_WORKERS+x}" ]; then
  SVC_LI_WORKERS=$(( _per_rep / (SVC_CELL_FAN * 2) ))
  [ "$SVC_LI_WORKERS" -lt 1 ] && SVC_LI_WORKERS=1
  [ "$SVC_LI_WORKERS" -gt 4 ] && SVC_LI_WORKERS=4
fi
# A replica scatters to all four cells at once, so a cell's share is a quarter of the replica.
if [ -z "${SVC_PART_WORKERS+x}" ]; then
  SVC_PART_WORKERS=$(( _per_rep / 4 )); [ "$SVC_PART_WORKERS" -lt 1 ] && SVC_PART_WORKERS=1
  [ "$SVC_PART_WORKERS" -gt 8 ] && SVC_PART_WORKERS=8
fi
# The four widths multiply out (replicas x 4 cells x part workers x OpenMP), so the innermost one
# gets whatever the other three leave.
if [ -z "${SVC_CELL_OMP+x}" ]; then
  SVC_CELL_OMP=$(( _cpus / (SVC_REPLICAS * 4 * SVC_PART_WORKERS) ))
  [ "$SVC_CELL_OMP" -lt 1 ] && SVC_CELL_OMP=1
fi
if [ -z "${SVC_WARM_WORKERS+x}" ]; then
  SVC_WARM_WORKERS=$_cpus; [ "$SVC_WARM_WORKERS" -gt 12 ] && SVC_WARM_WORKERS=12
fi
if [ -z "${SVC_PREFAULT_THREADS+x}" ]; then
  SVC_PREFAULT_THREADS=$_cpus; [ "$SVC_PREFAULT_THREADS" -gt 12 ] && SVC_PREFAULT_THREADS=12
fi

# An inherited OMP_NUM_THREADS wins over faiss.omp_set_num_threads for every thread the fan
# spawns, so the ambient value is dropped and the innermost width is exported instead.
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
_omp="$SVC_CELL_OMP"; [ "$_omp" -lt 1 ] && _omp=1
export OMP_NUM_THREADS="$_omp"
export MKL_NUM_THREADS="$_omp"
export OPENBLAS_NUM_THREADS="$_omp"
export NUMEXPR_NUM_THREADS="$_omp"
# torch's intra-op pool lives in the replica parent, which drives PLAID and the reranker, and
# torch.set_num_threads is a real process global, so it takes the replica's whole share.
SVC_TORCH_THREADS="${SVC_TORCH_THREADS:-$_per_rep}"
# Seismic is Rust/Rayon; an unset pool size is one worker per node core inside every cell process.
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$SVC_PART_WORKERS}"

if [ -z "${SVC_REPLICA_DEVICES+x}" ]; then
  SVC_REPLICA_DEVICES=""
  for _i in $(seq 0 $(( SVC_REPLICAS - 1 )) ); do
    SVC_REPLICA_DEVICES="${SVC_REPLICA_DEVICES:+$SVC_REPLICA_DEVICES,}cuda:$(( _i % _gpus ))"
  done
fi

# More cards than replicas: shard one stack's four language cells over the spare cards instead of
# leaving them idle. Sharding and replicas are mutually exclusive, so this also selects the
# single-stack path (--replicas 0), which is the only mode where a shard is legal.
if [ -z "${SVC_PLAID_SHARD+x}" ] && [ "$SVC_REPLICAS" -lt "$_gpus" ]; then
  _i=0; SVC_PLAID_SHARD=""
  for _c in en es ru zh; do
    SVC_PLAID_SHARD="${SVC_PLAID_SHARD:+$SVC_PLAID_SHARD,}${_c}=cuda:$(( _i % _gpus ))"
    _i=$(( _i + 1 ))
  done
  SVC_REPLICAS=0
  SVC_REPLICA_DEVICES=""
  echo "plaid_shard=${SVC_PLAID_SHARD} (replicas -> 0, the single-stack path)"
fi

# Node-local mirror of the mmap'd PLAID blobs, on the host's local disk. Never tmpfs: those pages
# are unevictable and charged to this cgroup, which turns ~200 GiB of blobs into an OOM kill.
# The path carries the rendering: reusing one mirror across renderings serves the old blobs.
SVC_PLAID_LOCAL_HOST="${SVC_PLAID_LOCAL_HOST:-/tmp/rsvc-${SVC_INDEX}-${SLURM_JOB_ID:-manual}}"
SVC_PLAID_LOCAL="${SVC_PLAID_LOCAL:-/hosttmp/rsvc-${SVC_INDEX}-${SLURM_JOB_ID:-manual}}"
case "$SVC_PLAID_LOCAL" in
  /dev/shm*|/run/shm*)
    echo "ABORT: SVC_PLAID_LOCAL=$SVC_PLAID_LOCAL is tmpfs; use the /hosttmp bind."; exit 1 ;;
esac
# The passage store is read on every query, so it is mirrored into RAM; off a network filesystem
# the same fetch costs seconds rather than milliseconds.
SVC_STORE_LOCAL="${SVC_STORE_LOCAL:-/dev/shm/ragtime-store-$SVC_INDEX}"
SVC_MIRROR_OVERLAP="${SVC_MIRROR_OVERLAP:-on}"

mkdir -p "$REPO/logs" "$SVC_QUEUE/in" "$SVC_QUEUE/out" "$SVC_REGISTRY"
rm -f "$SVC_QUEUE/READY.json"

# A descriptor that outlives its job is a queue nobody answers, and a SIGKILL never reaches
# Python, so the descriptor is removed here too - but only if it is ours.
cleanup() {
  rm -f "$SVC_QUEUE/READY.json"
  D="$SVC_REGISTRY/$SVC_NAME.json"
  if [ -f "$D" ] && grep -q "\"job_id\": \"${SLURM_JOB_ID:-0}\"" "$D" 2>/dev/null; then rm -f "$D"; fi
  rm -rf "$SVC_PLAID_LOCAL_HOST"
}
trap cleanup EXIT

echo "node=$(hostname -f) job=${SLURM_JOB_ID:-?} cpus=${_cpus} gpus=${_gpus} mem=${SLURM_MEM_PER_NODE:-?}MB"
echo "index=$SVC_INDEX config=$SVC_CONFIG name=$SVC_NAME queue=$SVC_QUEUE"
echo "sizing: replicas=$SVC_REPLICAS (${_per_rep} c/replica) cell_fan=$SVC_CELL_FAN" \
     "li_workers=$SVC_LI_WORKERS part_workers=$SVC_PART_WORKERS cell_omp=$SVC_CELL_OMP" \
     "torch_threads=$SVC_TORCH_THREADS warm_workers=$SVC_WARM_WORKERS devices=$SVC_REPLICA_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || echo "no GPU visible"

# The blob mirror is ~200 GiB and a short local disk fails late, after the whole load.
df -BG "$(dirname "$SVC_PLAID_LOCAL_HOST")" | tail -1
FREE_G=$(df -BG --output=avail "$(dirname "$SVC_PLAID_LOCAL_HOST")" | tail -1 | tr -dc '0-9')
if [ "${FREE_G:-0}" -lt 250 ]; then
  echo "ABORT: $(dirname "$SVC_PLAID_LOCAL_HOST") has ${FREE_G}G free; the PLAID mirror needs ~200G."
  exit 1
fi

# Compute nodes may export an HTTP proxy; anything cluster-internal must bypass it.
export no_proxy="localhost,127.0.0.1,::1,$(hostname -f)"
export NO_PROXY="$no_proxy"

# A symlink rather than a copy, so the container imports the current tree. Point RAGTIME_PYPATH at
# a directory containing a `ragtime` entry to pin what a long-lived service imports.
if [ -z "${RAGTIME_PYPATH:-}" ]; then
  RAGTIME_PYPATH="$HOME/.cache/ragtime-pypath/$(basename "$REPO")"
  mkdir -p "$RAGTIME_PYPATH"
  ln -sfn "$REPO/src/ragtime" "$RAGTIME_PYPATH/ragtime"
fi
[ -e "$RAGTIME_PYPATH/ragtime" ] || { echo "ABORT: no 'ragtime' entry under $RAGTIME_PYPATH"; exit 1; }

NV=()
[ "${SLURM_GPUS_ON_NODE:-0}" != "0" ] && NV=(--nv)
OVERRIDE=()
[ "$SVC_ALLOW_OVERRIDE" = "1" ] && OVERRIDE=(--allow-index-override)

# --cleanenv drops every host variable not re-passed with --env, so each load-bearing one is
# listed below.
apptainer exec --cleanenv "${NV[@]}" \
  --bind "$RAGTIME_ARTIFACT_ROOT" \
  --bind "$RAGTIME_DEV_ROOT" \
  --bind "$REPO/logs" \
  --bind "$HOME/.cache/huggingface" \
  --bind /dev/shm \
  --bind /tmp:/hosttmp \
  --env "RAGTIME_ARTIFACT_ROOT=${RAGTIME_ARTIFACT_ROOT}" \
  --env "no_proxy=${no_proxy}" \
  --env "NO_PROXY=${no_proxy}" \
  --env "HF_HOME=${HOME}/.cache/huggingface" \
  --env "HF_HUB_OFFLINE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=${RAGTIME_PYPATH}" \
  --env "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  --env "MKL_NUM_THREADS=${MKL_NUM_THREADS}" \
  --env "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}" \
  --env "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}" \
  --env "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "PYTORCH_ALLOC_CONF=expandable_segments:True" \
  --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}" \
  --env "SLURMD_NODENAME=${SLURMD_NODENAME:-unknown}" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID:-0}" \
  --env "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-0}" \
  --env "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-0}" \
  --env "SVC_GPU_MODEL=${SVC_GPU_MODEL}" \
  --env "SVC_DEGRADED=${SVC_DEGRADED}" \
  "$RAGTIME_SIF" \
  uv run --frozen python -m ragtime.devkit.rsvc \
      --config "$SVC_CONFIG" \
      --index "$SVC_INDEX" \
      "${OVERRIDE[@]}" \
      --queue "$SVC_QUEUE" \
      --registry "$SVC_REGISTRY" \
      --name "$SVC_NAME" \
      --slot "$SVC_SLOT" \
      --tier "$SVC_TIER" \
      ${SVC_PLAID_SHARD:+--plaid-shard "$SVC_PLAID_SHARD"} \
      --replicas "$SVC_REPLICAS" \
      --replica-devices "$SVC_REPLICA_DEVICES" \
      --gpu-plaid "$SVC_GPU_PLAID" \
      --gpu-mtd "$SVC_GPU_MTD" \
      --gpu-dense "$SVC_GPU_DENSE" \
      --gpu-sparse "$SVC_GPU_SPARSE" \
      --gpu-rerank "$SVC_GPU_RERANK" \
      --rerank-batch "$SVC_RERANK_BATCH" \
      --plaid-low-memory "$SVC_PLAID_LOW_MEMORY" \
      --part-workers "$SVC_PART_WORKERS" \
      --cell-omp "$SVC_CELL_OMP" \
      --torch-threads "$SVC_TORCH_THREADS" \
      --li-workers "$SVC_LI_WORKERS" \
      --cell-procs "$SVC_CELL_PROCS" \
      --cell-fan "$SVC_CELL_FAN" \
      --prefault-threads "$SVC_PREFAULT_THREADS" \
      --warm-workers "$SVC_WARM_WORKERS" \
      --mirror-overlap "$SVC_MIRROR_OVERLAP" \
      --plaid-local "$SVC_PLAID_LOCAL" \
      --store-local "$SVC_STORE_LOCAL" \
      $SVC_EXTRA
RC=$?
# Nothing may run between the command and $?: sacct reports the batch step's status, so the real
# exit code has to be re-raised here.
echo "RSVC_RC=$RC"
exit "$RC"
