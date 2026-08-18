#!/usr/bin/env bash
# Add N pipeline workers to a live vLLM allocation, without touching the engine.
#   srun --jobid=<J> --overlap --mem=<8*N>G slurm/pipeline_workers.sh <config> <n> <vllm_port>
# The --mem is not optional: an overlapping step shares the job's cgroup.
set -uo pipefail

REPO="${RAGTIME_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO" || exit 1

CFG="${1:?config path}"
N="${2:-3}"
PORT="${3:?vllm port}"

export RAGTIME_ARTIFACT_ROOT="${RAGTIME_ARTIFACT_ROOT:?set the artifact root}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 no_proxy='*' NO_PROXY='*'

NODE=$(hostname -s)
LOGDIR="$REPO/logs/workers"
mkdir -p "$LOGDIR"

# Retrieval must already be reachable. A worker that cannot search would claim topics and fail
# them, which costs each one an attempt; not starting is the cheaper outcome.
RSVC_HTTP="${RAGTIME_RSVC_HTTP:-}"
if [ -n "$RSVC_HTTP" ]; then
  code=$(curl -s --noproxy '*' -m 20 -o /dev/null -w '%{http_code}' "$RSVC_HTTP/health" 2>/dev/null)
  if [ "$code" != "200" ]; then
    echo "[workers] retrieval not healthy (http=$code) - refusing to start"
    exit 4
  fi
elif [ -n "${RAGTIME_RSVC_REGISTRY:-}" ]; then
  if ! ls "$RAGTIME_RSVC_REGISTRY"/*.json >/dev/null 2>&1; then
    echo "[workers] no retrieval descriptor in $RAGTIME_RSVC_REGISTRY - refusing to start"
    exit 4
  fi
else
  echo "[workers] set RAGTIME_RSVC_HTTP or RAGTIME_RSVC_REGISTRY - refusing to start"
  exit 4
fi
echo "[workers] node=$NODE cfg=$CFG n=$N vllm=:$PORT retrieval=ok"

spawn() {   # spawn <log suffix>
  env RAGTIME_VLLM_URL="http://localhost:${PORT}/v1" \
      PIPELINE_ROLE=worker \
      "$REPO/.venv/bin/run" --config "$CFG" --stage pipeline \
      >> "$LOGDIR/${NODE}_${1}.log" 2>&1 &
}

# Two workers cannot take the same topic: a claim is an atomic rename and the first one wins, so N
# workers on one node are exactly as safe as one.
pids=()
for i in $(seq 1 "$N"); do
  spawn "$i"
  pids+=($!)
  echo "[workers] started worker $i pid=${pids[-1]}"
  sleep 20   # stagger: simultaneous client bring-ups hammer the same store and the same engine
done

# A worker exiting rc=0 has drained its queue and must not be restarted; anything else is a crash,
# and an idle slot on a held allocation is the expensive failure.
while true; do
  sleep 60
  alive=0
  for idx in "${!pids[@]}"; do
    p=${pids[$idx]}
    if kill -0 "$p" 2>/dev/null; then
      alive=$((alive+1)); continue
    fi
    wait "$p"; rc=$?
    if [ "$rc" = "0" ]; then
      echo "[workers] worker pid=$p drained (rc=0) - not restarting"
      continue
    fi
    echo "[workers] worker pid=$p died rc=$rc - restarting"
    spawn "r${idx}"
    pids[$idx]=$!
    alive=$((alive+1))
  done
  echo "[workers $(date +%H:%M:%S)] alive=$alive/$N"
done
