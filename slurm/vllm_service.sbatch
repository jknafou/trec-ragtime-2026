#!/usr/bin/env bash
# One vLLM instance on one GPU pair, plus the pipeline workers that feed it.
#   FLEET_WORK=1 FLEET_CONFIGS="config/e2e-original.yml config/e2e-omt.yml" \
#     sbatch slurm/vllm_service.sbatch      # submit one job per pair you want working
#SBATCH --job-name=vllm-fleet
#SBATCH --partition=shared-gpu
#SBATCH --constraint="nvidia_a100_80gb_pcie|nvidia_rtx_pro_6000_blackwell|nvidia_h200_nvl"
# One instance per GPU pair. num_key_value_heads=2 caps tensor parallelism at 2, and the FP8
# weights fit no single card below 141 GiB. Two instances on one pair starve each other's KV cache.
#SBATCH --gres=gpu:2
# 14 rather than 16: the A100 nodes reserve two cores for the system, so a 16-core ask can never
# land there whatever its GPU request.
#SBATCH --cpus-per-task=14
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -uo pipefail

REPO="${RAGTIME_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1
ENDPOINT="${RAGTIME_VLLM_ENDPOINT_FILE:-$REPO/logs/vllm/endpoint-${SLURM_JOB_ID}.json}"
mkdir -p "$(dirname "$ENDPOINT")"
export HF_HUB_OFFLINE=1 UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}" PYTHONUNBUFFERED=1

# SVC_TP wins over SLURM's count, which reports the whole job's cards when this runs as a step
# inside a larger allocation.
TP="${SVC_TP:-${SLURM_GPUS_ON_NODE:-2}}"

# max_model_len is bounded by the KV headroom left after the weights, not by raw VRAM.
# KV/token = kv_heads(2) * head_dim(256) * 2 (K and V) * layers(48) * 2 bytes = 96 KiB.
VRAM_ONE=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_TOT=$(( VRAM_ONE * TP ))
USABLE=$(( VRAM_TOT * 92 / 100 ))
WEIGHTS_MIB=121242
KV_MIB=$(( USABLE - WEIGHTS_MIB ))
if [ "$KV_MIB" -le 0 ]; then
  echo "FATAL: weights (${WEIGHTS_MIB} MiB) exceed usable VRAM (${USABLE} MiB at TP=$TP)"
  exit 1
fi
KV_TOKENS=$(( KV_MIB * 1024 / 96 ))
MML=$(( KV_TOKENS * 45 / 100 ))
[ "$MML" -gt 131072 ] && MML=131072
MML=$(( MML / 4096 * 4096 ))
if [ "$MML" -lt 8192 ]; then
  echo "FATAL: KV headroom ${KV_MIB} MiB admits only ${MML} tokens"
  exit 1
fi
echo "vram_total=${VRAM_TOT}MiB usable=${USABLE}MiB kv=${KV_MIB}MiB (~${KV_TOKENS} tok) MML=$MML"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in
  *A100*) export NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 ;;
  *)      export NCCL_NET_GDR_LEVEL=0 GLOO_SOCKET_IFNAME=lo ;;
esac
uv sync --frozen --extra heavy || { echo "UV_SYNC_FAILED"; exit 1; }
export PATH="$REPO/.venv/bin:$PATH"

PORT=$(( 8000 + (SLURM_JOB_ID % 1000) ))
HOST=$(hostname -f)
SRV="$REPO/logs/vllm/server-${SLURM_JOB_ID}.log"
echo "node=$HOST port=$PORT gpu=$GPU_NAME TP=$TP MML=$MML"

# The per-pair registry a worker leases an endpoint from. One shared endpoint file would send
# every worker to the same pair.
REG_BASE="${RAGTIME_VLLM_REGISTRY:-}"
REG_NAME="vllm-${SLURM_JOB_ID}"
reg() {
  [ -n "$REG_BASE" ] || return 0
  "$REPO/.venv/bin/python" -m ragtime.serving.vllm_registry "$@" --base "$REG_BASE" --name "$REG_NAME"
}

# The worker is stopped first and given a moment: it requeues its in-flight shard on the way out,
# which it cannot do while generating against a socket that has already gone.
cleanup() {
  if [ -n "${WORKER_PID:-}" ]; then kill "$WORKER_PID" 2>/dev/null; sleep 5; fi
  rm -f "$ENDPOINT"; reg unpublish; kill "${SRV_PID:-}" 2>/dev/null
}
trap cleanup EXIT

# 127.0.0.1 rather than localhost (vLLM binds IPv4 only), --noproxy, and an explicit timeout so a
# probe fails and retries instead of blocking forever. It also reports progress: a silent probe is
# indistinguishable from a hung engine.
wait_ready() {
  local i rc
  for i in $(seq 1 240); do
    curl -sf --noproxy '*' -m 10 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && {
      echo "[wait_ready] serving after ${i} probe(s) (~$((i * 5))s)"; return 0; }
    rc=$?
    if [ "$i" = "1" ] || [ $((i % 10)) = "0" ]; then
      echo "[wait_ready] probe ${i}/240 curl_rc=$rc port=${PORT} - not serving yet"
    fi
    kill -0 "$SRV_PID" 2>/dev/null || { echo "[wait_ready] server pid ${SRV_PID:-?} is gone"; return 1; }
    sleep 5
  done
  echo "[wait_ready] gave up after 240 probes with the server process still alive"
  return 1
}

# The launch flags have a single definition, in serving.node.vllm_server. --print-argv starts nothing;
# it prints the argv a scored run would use, so this service cannot drift from production.
MODEL="${RAGTIME_VLLM_MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
start_server() {
  local argv
  mapfile -t argv < <(uv run --frozen python -m ragtime.serving.node.vllm_server --print-argv \
      --model "$MODEL" --tensor-parallel-size "$TP" --max-model-len "$MML" \
      --gpu-memory-utilization 0.92 --port "$PORT" --python "$REPO/.venv/bin/python")
  if [ "${#argv[@]}" -lt 8 ]; then
    echo "FATAL: could not obtain the production vLLM argv (got ${#argv[@]} tokens)"
    exit 1
  fi
  echo "argv: ${argv[*]}"
  "${argv[@]}" >> "$SRV" 2>&1 &
  SRV_PID=$!
}

start_server
READY=0
if wait_ready; then READY=1; else echo "VLLM_NOT_READY on first start"; tail -30 "$SRV"; fi

publish_endpoint() {
  TMP="${ENDPOINT}.tmp-$$"
  cat > "$TMP" <<JSON
{"host": "$HOST", "port": $PORT, "url": "http://$HOST:$PORT/v1",
 "job_id": "$SLURM_JOB_ID", "gpu": "$GPU_NAME",
 "tp": $TP, "max_model_len": $MML, "started": "$(date -Is)"}
JSON
  mv -f "$TMP" "$ENDPOINT"
  reg publish --url "http://$HOST:$PORT/v1" --job-id "$SLURM_JOB_ID" \
      --pair "$HOST:${CUDA_VISIBLE_DEVICES:-unknown}" --model "$MODEL" --gpu-model "$GPU_NAME"
}

# A descriptor is a promise a worker will block on, so it is published only once the engine has
# answered. Advertising a crashed engine hangs every client that takes it.
if [ "$READY" = "1" ]; then
  publish_endpoint
  echo "READY - endpoint: $(cat "$ENDPOINT")"
else
  echo "NOT PUBLISHED - the engine never answered /v1/models; the supervise loop will retry."
fi

# ---------------------------------------------------------------------------
# The pair's own pipeline workers. A pair that finishes a topic and stops is wasted allocation, so
# the workers are born with the instance rather than coordinated from outside.
#
# FLEET_CONFIGS is a list, not one config: every config names the same checkpoint, so moving from
# one arm to the next costs no reload, only a different queue.
# ---------------------------------------------------------------------------
FLEET_WORK="${FLEET_WORK:-0}"
FLEET_CONFIGS="${FLEET_CONFIGS:-config/e2e-original.yml}"
WORKER_LOG="$REPO/logs/vllm/worker-${SLURM_JOB_ID}.log"
WORKER_PID=""
CUR_CONFIG=""

# Retrieval lives in its own service. Set RAGTIME_RSVC_HTTP when it answers over HTTP, or
# RAGTIME_RSVC_REGISTRY when this node can see its queue directly. RSVC_SSH_HOST opens a forward
# first, for the case where the two services sit on clusters that share no filesystem.
RSVC_HTTP="${RAGTIME_RSVC_HTTP:-}"
[ -n "$RSVC_HTTP" ] && export RAGTIME_RSVC_HTTP="$RSVC_HTTP"
RSVC_SSH_HOST="${RSVC_SSH_HOST:-}"
RSVC_LPORT="${RSVC_LPORT:-18799}"
RSVC_RPORT="${RSVC_RPORT:-8799}"

retrieval_up() {
  [ -n "$RSVC_HTTP" ] || return 0
  if [ -n "$RSVC_SSH_HOST" ]; then
    # ControlPath=none is not optional: a multiplexed session ignores a new -L and reports success
    # while the forward never binds.
    ss -ltn 2>/dev/null | grep -q ":${RSVC_LPORT}" || \
      ssh -f -N -L "${RSVC_LPORT}:127.0.0.1:${RSVC_RPORT}" -o ControlPath=none -o BatchMode=yes \
          -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes "$RSVC_SSH_HOST" 2>/dev/null
    sleep 3
  fi
  # A bound port with a dead service behind it accepts the connection and then hangs, so the check
  # is a real request rather than a listening socket.
  local code
  code=$(curl -s --noproxy '*' -m 20 -o /dev/null -w '%{http_code}' "$RSVC_HTTP/health" 2>/dev/null)
  [ "$code" = "200" ]
}

WORKER_ENV=(
  "RAGTIME_ARTIFACT_ROOT=${RAGTIME_ARTIFACT_ROOT:?set the artifact root}"
  "PIPELINE_ROLE=worker" "no_proxy=*" "NO_PROXY=*"
)
[ -n "$RSVC_HTTP" ] && WORKER_ENV+=("RAGTIME_RSVC_HTTP=$RSVC_HTTP")
[ -n "${RAGTIME_RSVC_REGISTRY:-}" ] && WORKER_ENV+=("RAGTIME_RSVC_REGISTRY=$RAGTIME_RSVC_REGISTRY")

start_worker() {
  retrieval_up || { echo "[worker] retrieval not healthy - not starting a worker this cycle"; return 1; }
  CUR_CONFIG="$1"
  echo "[worker] starting: config=$CUR_CONFIG"
  env "${WORKER_ENV[@]}" RAGTIME_VLLM_URL="http://localhost:${PORT}/v1" \
      "$REPO/.venv/bin/run" --config "$CUR_CONFIG" --stage pipeline \
      >> "$WORKER_LOG" 2>&1 &
  WORKER_PID=$!
  echo "[worker] pid=$WORKER_PID config=$CUR_CONFIG log=$WORKER_LOG"
}

# The first config this pair has not drained in this pass, wrapping round so a rescan re-offers
# the whole list.
DRAINED_SET=" "
next_config() {
  for c in $FLEET_CONFIGS; do
    case "$DRAINED_SET" in *" $c "*) continue ;; esac
    echo "$c"; return 0
  done
  return 1
}

if [ "$FLEET_WORK" = "1" ] && [ "$READY" = "1" ]; then
  c=$(next_config) && start_worker "$c" || true
elif [ "$FLEET_WORK" = "1" ]; then
  echo "[worker] engine not ready - the supervise loop starts the worker once it is."
fi

# Concurrent streams one instance carries, sized by KV pool. Past this the per-request rate falls
# far enough that a long generation approaches llm.call_timeout_s and legitimate turns get cut.
case "$GPU_NAME" in
  *H200*) TOTAL_WORKERS="${SVC_TOTAL_WORKERS:-16}" ;;
  *)      TOTAL_WORKERS="${SVC_TOTAL_WORKERS:-8}"  ;;
esac
N_ARMS=$(set -- $FLEET_CONFIGS; echo $#)
EXTRAS=$(( TOTAL_WORKERS - 1 )); [ "$EXTRAS" -lt 0 ] && EXTRAS=0
if [ "$N_ARMS" -gt 0 ] 2>/dev/null; then
  BASE=$(( EXTRAS / N_ARMS )); REM=$(( EXTRAS % N_ARMS ))
else
  BASE=0; REM=0
fi
echo "[extra] gpu=$GPU_NAME total_workers=$TOTAL_WORKERS (1 main + $EXTRAS extras over $N_ARMS arm(s))"
if [ "$FLEET_WORK" = "1" ] && [ "$READY" = "1" ] && [ "$EXTRAS" -gt 0 ] 2>/dev/null; then
  i=0
  for xc in $FLEET_CONFIGS; do
    n=$BASE; [ "$i" -lt "$REM" ] && n=$(( BASE + 1 ))
    i=$(( i + 1 ))
    [ "$n" -gt 0 ] || continue
    xtag=$(basename "$xc" .yml)
    xout="$REPO/logs/vllm/extra-${SLURM_JOB_ID}-${xtag}.log"
    # An `srun --overlap` step shares this job's cgroup memory, so it must carry its own --mem or
    # it can starve the engine it is feeding. A worker peaks at 1.5-3 GiB.
    xmem=$(( 8 * n ))
    echo "[extra] fanning $n worker(s) for $xc -> $xout (--mem=${xmem}G)"
    srun --jobid="$SLURM_JOB_ID" --overlap --mem="${xmem}G" \
         "$REPO/slurm/pipeline_workers.sh" "$xc" "$n" "$PORT" > "$xout" 2>&1 &
    sleep 20
  done
else
  echo "[extra] not fanning (FLEET_WORK=$FLEET_WORK READY=$READY EXTRAS=$EXTRAS)"
fi

# ---------------------------------------------------------------------------
# Supervise. The allocation is the scarce resource, not the process: a dead engine is restarted
# inside the same allocation rather than handing a queued node back.
# To cycle the server deliberately: touch logs/vllm/RESTART_VLLM
# ---------------------------------------------------------------------------
RESTART_FLAG="$REPO/logs/vllm/RESTART_VLLM"
restarts=0
while true; do
  sleep 60
  if [ -f "$RESTART_FLAG" ]; then
    echo "[$(date +%H:%M:%S)] restart requested - cycling vLLM, keeping the allocation"
    rm -f "$RESTART_FLAG" "$ENDPOINT"; reg unpublish
    kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null
    SRV_PID=""
  fi
  if [ -z "${SRV_PID:-}" ] || ! kill -0 "$SRV_PID" 2>/dev/null; then
    restarts=$((restarts+1))
    echo "[$(date +%H:%M:%S)] server is down - restart #$restarts (allocation retained)"
    # Withdraw before restarting: the window where the descriptor names a dead socket is the
    # window where a worker leases it and hangs.
    rm -f "$ENDPOINT"; reg unpublish
    start_server
    if wait_ready; then publish_endpoint; echo "[$(date +%H:%M:%S)] back up after restart #$restarts"
    else echo "[$(date +%H:%M:%S)] restart #$restarts did not become ready; retrying next cycle"; fi
    continue
  fi
  [ -f "$ENDPOINT" ] || publish_endpoint
  # A descriptor with no beat ages out and the pair leaves the fleet while the engine is healthy.
  reg beat >/dev/null

  if [ "$FLEET_WORK" = "1" ] && [ "$READY" = "1" ]; then
    if [ -n "${WORKER_PID:-}" ] && ! kill -0 "$WORKER_PID" 2>/dev/null; then
      wait "$WORKER_PID"; wrc=$?
      if [ "$wrc" = "0" ]; then
        # rc=0 means the queue drained, not that the worker failed: restarting it would spin
        # claiming nothing, so move to the next config in the list instead.
        DRAINED_SET="$DRAINED_SET$CUR_CONFIG "
        echo "[$(date +%H:%M:%S)] [worker] $CUR_CONFIG drained (rc=0)"
      else
        echo "[$(date +%H:%M:%S)] [worker] $CUR_CONFIG died rc=$wrc - restarting"
        tail -20 "$WORKER_LOG" 2>/dev/null
      fi
      WORKER_PID=""
    fi
    if [ -z "${WORKER_PID:-}" ]; then
      if c=$(next_config); then
        start_worker "$c" || true
      else
        # Every config is dry. Hold the pair and rescan: requeued shards do appear.
        DRAINED_SET=" "
        echo "[$(date +%H:%M:%S)] [worker] whole run list dry - pair held, rescanning next cycle"
      fi
    fi
  fi
  if [ $((SECONDS % 300)) -lt 60 ]; then
    echo "[$(date +%H:%M:%S)] alive - generations served:" \
         "$(grep -ac 'POST /v1/chat/completions' "$SRV" 2>/dev/null), restarts: $restarts"
  fi
done
