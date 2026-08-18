#!/usr/bin/env bash
# Check that the retrieval service is serving the index you think, in the shape you think.
#
#   WANT_INDEX=omt_opus BOOT=logs/slurm/rsvc-12345.out bash slurm/verify_service.sh
#   WANT_INDEX=omt_opus BOOT=logs/slurm/rsvc-12345.out bash slurm/verify_service.sh --live
#
# Every clause prints PASS or FAIL and the evidence it read. Exit status is 0 only if all of them
# pass. The script exists because a retrieval service can be serving the right index, answer every
# query correctly, and still run ten times slower than it should with nothing in its own output
# saying so.
#
# WANT_INDEX has no default. A default is how a check passes against the one service
# that happens to be right while the reader believes it verified a different one.
#
# --live starts a Python client, so it belongs in a job step rather than on a login node, and that
# step needs its own small --mem: an overlapping step draws on the service's cgroup, and one that
# took the default has been measured holding 72 GiB and OOM-killing the service it was measuring.
#
#   srun --jobid=<J> --overlap -N1 -n1 --mem=8G --chdir=<repo> \
#        env WANT_INDEX=omt_opus BOOT=<boot log> bash slurm/verify_service.sh --live
#
# Reference reading, from the service that answered the released runs while it searched omt_opus:
# every clause passes, sparse_cores 10.63 to 13.87 across the four language cells, cpu_legs_wall
# 0.9943 s, store_fetch 0.0024 s, and 2.523 s and 2.600 s on two queries it had never seen. Its
# cold load was 2039.3 s. A healthy service on comparable hardware should land near these.
set -uo pipefail

REPO="${RAGTIME_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1

WANT_INDEX="${WANT_INDEX:?WANT_INDEX must name the rendering you believe is being served}"
RSVC_REGISTRY="${RSVC_REGISTRY:-$REPO/logs/rsvc/registry}"
# The descriptor to read. Named explicitly when several lanes share a registry; otherwise the
# newest one, which is the only sane guess and is stated in the output so it can be checked.
REG="${REG:-$(ls -t "$RSVC_REGISTRY"/*.json 2>/dev/null | head -1)}"
# The service's own stdout. Several clauses can only be answered from what it logged at boot.
BOOT="${BOOT:-}"
SVC_CONFIG="${SVC_CONFIG:-config/e2e-original.yml}"
WANT_RERANK_INSTANCES="${WANT_RERANK_INSTANCES:-6}"
WANT_RERANK_DEPTH="${WANT_RERANK_DEPTH:-100}"
LIVE_QUERY="${LIVE_QUERY:-Which inspectors were dismissed after the ferry safety review}"
# The interpreter for the live half. The project environment, as the other launchers use it: a
# compute node has no route out, so this must not be a form that would resolve dependencies.
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

fail=0
pass_n=0
ok() { printf "  PASS  %-32s %s\n" "$1" "$2"; pass_n=$((pass_n + 1)); }
no() { printf "  FAIL  %-32s %s\n" "$1" "$2"; fail=1; }
# One "key": value pair out of the descriptor, last occurrence wins.
g() { grep -aoE "\"$1\": *[^,}]*" "$REG" 2>/dev/null | tail -1 | sed "s/.*: *//" | tr -d "\" "; }

echo "=== registry  ${REG:-none found under $RSVC_REGISTRY}"
echo "=== boot log  ${BOOT:-not given}"
echo

if [ -z "$REG" ] || [ ! -f "$REG" ]; then
  echo "  FAIL  no descriptor to read. Nothing below can be answered."
  exit 1
fi

# 1. Freshness first. `ready: true` outlives the service that wrote it; a stale descriptor has been
#    read as ready for 38 minutes after its service was cancelled, so every clause below this one
#    is meaningless if this one fails.
hb=$(g heartbeat); hb=${hb:-0}
age=$(( $(date +%s) - ${hb%%.*} ))
if [ "$age" -lt 120 ]; then
  ok "descriptor fresh" "heartbeat age ${age}s"
else
  no "descriptor stale" "heartbeat age ${age}s, so ready: means nothing and neither does the rest"
fi

# 2. The right rendering. This is the label the service published. Only ranking can prove which
#    index answered a given query, because the three renderings share passage ids and the build
#    hash is one build with the renderings as subdirectories.
r=$(g rendering)
[ -z "$r" ] && [ -n "$BOOT" ] && \
  r=$(grep -aoE "\"index\": \"[a-z_]+\"" "$BOOT" 2>/dev/null | tail -1 | sed 's/.*: *//' | tr -d '"')
if [ "$r" = "$WANT_INDEX" ]; then
  ok "index / rendering" "$r"
else
  no "index / rendering" "got ${r:-nothing}, want $WANT_INDEX"
fi

# 3. Late interaction on GPU, one card per language cell. This is the clause that exists because
#    the failure is silent: with the PLAID device absent the stack prints a warning, keeps going,
#    still reports three legs, and fuses over two pools. A measurement taken then measured a
#    different system.
if [ -z "$BOOT" ] || [ ! -f "$BOOT" ]; then
  # A clause that cannot be answered is a failure, not a silent absence. These three read only what
  # the service logged as it came up, and BOOT is the only place that is written.
  no "late-interaction on GPU" "no readable boot log, so this cannot be answered"
  no "low_memory false"        "no readable boot log, so this cannot be answered"
  no "reranker per GPU"        "no readable boot log, so this cannot be answered"
else
  shard=$(grep -aoE "plaid_shard[a-z_]*=[^ ]*" "$BOOT" 2>/dev/null | tail -1)
  [ -z "$shard" ] && shard=$(grep -aoE "\"plaid_(shard|device)\": *[^,}]*" "$BOOT" 2>/dev/null | tail -1)
  if echo "$shard" | grep -q "cuda:"; then
    ok "late-interaction on GPU" "$shard"
  else
    no "late-interaction on GPU" "no cuda device in the shard: ${shard:-nothing logged}"
  fi

  # 4. low_memory must be false. True does not converge: it has been measured with a spread of
  #    122x across queries against a converged median with it off.
  lm=$(grep -aoE "low_memory\"?: *(true|false|True|False)" "$BOOT" 2>/dev/null | tail -1)
  if echo "$lm" | grep -qi false; then
    ok "low_memory false" "$lm"
  else
    no "low_memory" "${lm:-not logged}"
  fi

  # 5. One reranker instance per card. Each logs itself as it comes up.
  n=$(grep -ac "rerank_instance_ready" "$BOOT" 2>/dev/null)
  if [ "${n:-0}" -ge "$WANT_RERANK_INSTANCES" ]; then
    ok "reranker per GPU" "$n instances"
  else
    no "reranker per GPU" "only ${n:-0}, want $WANT_RERANK_INSTANCES; pass --rerank-devices"
  fi
fi

# 6. Rerank depth. Three sources, strongest first: what the service reported for a live query, what
#    it logged, and failing both the config leaf it was told to load. The source is printed, because
#    the first is a measurement of the running service and the last is only a statement of intent.
depth=""; depth_src=""
if [ -n "$BOOT" ]; then
  depth=$(grep -aoE "\"rerank_depth\": *[0-9]+" "$BOOT" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  [ -n "$depth" ] && depth_src="boot log"
fi
if [ -z "$depth" ] && [ -f "$SVC_CONFIG" ]; then
  depth=$(grep -aoE "depth: *[0-9]+" "$SVC_CONFIG" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  [ -n "$depth" ] && depth_src="$SVC_CONFIG"
fi
# The live reading below is worth more than these two only because the probe sends no depth of its
# own. A request that carries one is answered at that depth, so a probe that always sent 100 would
# make this clause report its own argument back. See slurm/retrieval_probe.py.

# 7. The sparse process fan gate. cell_procs is a boolean gate, not a width: 0 skips the process
#    pool entirely and leaves the sparse leg on one core. The knob whose 0 means "one process per
#    part" is the separate part_procs.
cp=$(g cell_procs); pp=$(g part_procs); parts=""
[ -n "$BOOT" ] && parts=$(grep -aoE "\"total_parts\": *[0-9]+" "$BOOT" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
case "$cp" in
  1|true|True) ok "sparse fan gate open" "cell_procs=$cp part_procs=${pp:-unset} parts=${parts:-unknown}" ;;
  *)           no "sparse fan gate closed" "cell_procs=${cp:-unset}, relaunch with SVC_CELL_PROCS=1" ;;
esac

# The live half. Everything above reads what the service said about itself at boot; these read what
# it does now. A probe that fails leaves every number below empty, and each clause then fails on its
# own terms rather than being skipped.
if [ "${1:-}" = "--live" ]; then
  echo
  echo "--- live query. Run it when the fleet is quiet, or it measures the queue and not the service."
  out=$("$PYTHON" "$REPO/slurm/retrieval_probe.py" "$LIVE_QUERY" \
          --registry "$RSVC_REGISTRY" --rendering "$WANT_INDEX" 2>&1)
  probe_rc=$?
  [ "$probe_rc" -ne 0 ] && { echo "    probe exited $probe_rc:"; echo "$out" | sed "s/^/    /"; }
  echo "$out" | grep -aE "client_wall|^TIMING:|^CPUCELL:" | sed "s/^/    /"

  num() { echo "$out" | grep -oE "\"$1\": *[0-9.]+" | grep -oE "[0-9.]+$"; }

  # Passage text is read on every query. Off a network filesystem the same fetch costs seconds
  # rather than milliseconds, so this is the clause that catches a mirror that was never populated.
  sf=$(num store_fetch | tail -1)
  if awk -v v="${sf:-9}" "BEGIN{exit !(v < 0.05)}"; then
    ok "passages in RAM" "store_fetch=${sf}s"
  else
    no "passages not in RAM" "store_fetch=${sf:-unknown}s, off a shared filesystem this is ~25s"
  fi

  cl=$(num cpu_legs_wall | tail -1)
  if awk -v v="${cl:-99}" "BEGIN{exit !(v < 3)}"; then
    ok "cpu legs fanned" "cpu_legs_wall=${cl}s"
  else
    no "cpu legs slow" "cpu_legs_wall=${cl:-unknown}s, it is about 11s with the fan off"
  fi

  # The only clause that separates a closed gate from a fan that is merely inefficient. Clause 7
  # and cpu_legs_wall are both proxies; with the gate shut this reads exactly 1.00 in every cell
  # and nothing else in the reply differs. It lives in cpu_per_cell, not in timing.
  cores=$(echo "$out" | sed -n 's/^CPUCELL: //p' | grep -oE "\"sparse_cores\": *[0-9.]+" | grep -oE "[0-9.]+$")
  if [ -z "$cores" ]; then
    no "sparse_cores not reported" "absent from CPUCELL, so the fan is unverified"
  else
    hi=$(echo "$cores" | sort -g | tail -1)
    lo=$(echo "$cores" | sort -g | head -1)
    ncell=$(echo "$cores" | grep -c .)
    if awk -v v="$hi" "BEGIN{exit !(v > 1.05)}"; then
      ok "sparse fan engaged" "sparse_cores max=$hi min=$lo over $ncell cells"
    else
      no "sparse fan not engaged" "sparse_cores max=$hi, 1.00 is the closed-gate signature"
    fi
  fi

  live_depth=$(num rerank_depth | tail -1)
  live_inst=$(num rerank_instances | tail -1)
  [ -n "$live_depth" ] && { depth="$live_depth"; depth_src="live query"; }
  if [ -n "$live_inst" ]; then
    if [ "$live_inst" -ge "$WANT_RERANK_INSTANCES" ] 2>/dev/null; then
      ok "reranker instances (live)" "$live_inst scored this query"
    else
      no "reranker instances (live)" "$live_inst scored this query, want $WANT_RERANK_INSTANCES"
    fi
  fi

  echo "    note: client_wall includes queueing. 24 concurrent workers have turned a 2.5s service"
  echo "          into 21, 62 and 119s of client wall. Judge the service by the TIMING legs."
  echo "    note: the first query after a boot is lazy init, measured at 201 to 228s. Discard one."
fi

# Reported last, so the live query can supply the strongest source for it.
if [ "${depth:-0}" = "$WANT_RERANK_DEPTH" ]; then
  ok "rerank depth" "$depth (from $depth_src)"
else
  no "rerank depth" "got ${depth:-nothing}, want $WANT_RERANK_DEPTH (source ${depth_src:-none})"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "ALL $pass_n CLAUSES PASSED"
else
  echo "$pass_n passed, at least one failed. See the FAIL lines above."
fi
exit "$fail"
