#!/usr/bin/env bash
# One poll of a live run: queue state, per-topic progress, per-instance generation rate.
#   cp slurm/monitor.env.example slurm/monitor.env   # then edit the paths
#   set -a; . slurm/monitor.env; set +a; bash slurm/monitor_run.sh
# Prints a report and exits; it changes nothing and runs no project code.
set -uo pipefail

# Every targeting leaf is overridable. A monitor hardcoded to the wrong family cannot report that
# it is watching the wrong family: it goes quiet in the tone of "all is well".
QUEUE="${QUEUE:?queue root, e.g. <artifact root>/corpus/e2e/<hash>/corpus-preprocess/queue}"
ARTROOT="${ARTROOT:-${RAGTIME_ARTIFACT_ROOT:?set the artifact root}}"
ARMS="${ARMS:-original omt omt_opus}"
# Arm to cell directory. The mapping is not derivable: the omt_opus arm is produced by the
# e2e-omt-weak config, and every mlir arm reads `original` while its cells carry the searched
# index in the run id.
CELLDIR_original="${CELLDIR_original:-e2e-original__original__seed0}"
CELLDIR_omt="${CELLDIR_omt:-e2e-omt__omt__seed0}"
CELLDIR_omt_opus="${CELLDIR_omt_opus:-e2e-omt-weak__omt_opus__seed0}"
TOTAL_CELLS="${TOTAL_CELLS:-309}"
REGISTRY="${RAGTIME_VLLM_REGISTRY:-}"
STALE_S="${STALE_S:-900}"            # heartbeat age that means "investigate", not "slow"
PROGRESS_WARN_S="${PROGRESS_WARN_S:-5400}"   # no artifact written in this long: show the age
GAUGE_GAP="${GAUGE_GAP:-30}"         # seconds between the two vLLM counter readings

export no_proxy='*' NO_PROXY='*'
now=$(date +%s)
echo "=== $(date '+%Y-%m-%d %H:%M:%S')  queue=$QUEUE"

# --------------------------------------------------------------- queue state and topic detail
tot_d=0; tot_r=0; tot_p=0; tot_f=0
FLAGS=""
printf '\n%-10s %6s %8s %8s %8s\n' arm done running pending failed
for a in $ARMS; do
  q="$QUEUE/pipeline_$a"
  d=$(ls "$q/done"    2>/dev/null | grep -c '__seed0$')
  r=$(ls "$q/running" 2>/dev/null | grep -c '__seed0$')
  p=$(ls "$q/pending" 2>/dev/null | grep -c '__seed0$')
  f=$(ls "$q/failed"  2>/dev/null | grep -c '__seed0$')
  tot_d=$((tot_d+d)); tot_r=$((tot_r+r)); tot_p=$((tot_p+p)); tot_f=$((tot_f+f))
  printf '%-10s %6d %8d %8d %8d\n' "$a" "$d" "$r" "$p" "$f"
done
printf '%-10s %6d %8d %8d %8d   (%d cells expected)\n' TOTAL "$tot_d" "$tot_r" "$tot_p" "$tot_f" "$TOTAL_CELLS"

printf '\n%-6s %-10s %6s %8s %7s %8s %s\n' topic arm round nuggets claims hb flags
for a in $ARMS; do
  q="$QUEUE/pipeline_$a"
  eval "cell=\$CELLDIR_${a//-/_}"
  for s in $(ls "$q/running" 2>/dev/null | grep '__seed0$'); do
    hb="$q/meta/$s.hb"
    tid="${s%%__*}"
    # The heartbeat is a short epoch float rewritten in place every 30 s, so a reader can land
    # mid-rewrite and get nothing. Retry once, then believe it: an unreadable heartbeat must not
    # read as perfect health, and one unlucky read must not raise a false alarm.
    age=-1
    for _try in 1 2; do
      if [ -s "$hb" ]; then
        raw=$(cut -d. -f1 < "$hb" 2>/dev/null | tr -dc '0-9')
        [ -n "$raw" ] && { age=$(( $(date +%s) - raw )); break; }
      fi
      [ "$_try" = 1 ] && sleep 0.4
    done
    if   [ "$age" -lt 0 ];          then mark="no-heartbeat"; FLAGS="$FLAGS $a/$tid:no-heartbeat"
    elif [ "$age" -gt "$STALE_S" ]; then mark="stale-$((age/60))m"; FLAGS="$FLAGS $a/$tid:stale-$((age/60))m"
    else                                 mark=""
    fi

    tdir="$ARTROOT/$cell/topics/$tid"
    # A fresh heartbeat proves only that the forked heartbeat child is alive, so progress is read
    # from the cell as well: the age of the newest file written anywhere under it. Long gaps are
    # normal here, which is why this is reported and never acted on.
    if [ -d "$tdir" ]; then
      w=$(find "$tdir" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
      if [ -n "$w" ] && [ $(( now - w )) -gt "$PROGRESS_WARN_S" ]; then
        mark="$mark idle-$(( (now - w) / 60 ))m"
      fi
    fi

    # round: the highest decompose/round_N.jsonl on disk, written at the round boundary, so it is
    # the last completed round rather than the one in flight.
    # nuggets: lines in that round file; the bank grows across audit rounds.
    # claims: the loops' own claims_committed counter, summed.
    lastr="-"; nug="-"; cl=0
    if [ -d "$tdir/decompose" ]; then
      n=$(ls "$tdir"/decompose/round_*.jsonl 2>/dev/null | grep -v '_SUCCESS' \
          | sed 's/.*round_//;s/\.jsonl//' | sort -n | tail -1)
      if [ -n "$n" ]; then
        lastr="$n"
        nug=$(wc -l < "$tdir/decompose/round_${n}.jsonl" 2>/dev/null | tr -dc '0-9')
        cl=$(grep -ho '"claims_committed":[0-9]*' "$tdir"/rag_loop/*.jsonl 2>/dev/null \
             | cut -d: -f2 | awk '{s+=$1} END{print s+0}')
      fi
    fi
    printf '%-6s %-10s %6s %8s %7s %8s %s\n' \
      "$tid" "$a" "$lastr" "${nug:--}" "$cl" \
      "$( [ "$age" -lt 0 ] && echo none || echo "${age}s" )" "$mark"
  done
done

# --------------------------------------------------------------- per-instance vLLM gauges
# tok/s is the delta of the instance's own generation-token counter over real elapsed seconds.
# streams is num_requests_running: an instantaneous gauge, not a rate and not a topic count.
if [ -n "$REGISTRY" ] && ls "$REGISTRY"/*.json >/dev/null 2>&1; then
  printf '\n%-34s %10s %9s %9s\n' endpoint tok/s streams waiting
  for f in "$REGISTRY"/*.json; do
    url=$(grep -o '"url": *"[^"]*"' "$f" | head -1 | cut -d'"' -f4)
    [ -n "$url" ] || continue
    base="${url%/v1}"
    m1=$(curl -s --noproxy '*' -m 10 "$base/metrics" 2>/dev/null)
    [ -n "$m1" ] || { printf '%-34s %10s %9s %9s\n' "$base" unreachable - -; continue; }
    t1=$(date +%s)
    g1=$(echo "$m1" | awk '/^vllm:generation_tokens_total/{print $2; exit}')
    sleep "$GAUGE_GAP"
    m2=$(curl -s --noproxy '*' -m 10 "$base/metrics" 2>/dev/null)
    t2=$(date +%s)
    g2=$(echo "$m2" | awk '/^vllm:generation_tokens_total/{print $2; exit}')
    run=$(echo "$m2" | awk '/^vllm:num_requests_running/{print $2; exit}')
    wait=$(echo "$m2" | awk '/^vllm:num_requests_waiting/{print $2; exit}')
    rate=$(awk -v a="${g1:-0}" -v b="${g2:-0}" -v d="$(( t2 - t1 ))" \
           'BEGIN{ if (d > 0) printf "%.1f", (b-a)/d; else print "-" }')
    printf '%-34s %10s %9s %9s\n' "$base" "$rate" "${run:--}" "${wait:--}"
  done
else
  echo -e "\nno vLLM registry set (RAGTIME_VLLM_REGISTRY) - skipping instance gauges"
fi

if [ -n "$FLAGS" ]; then
  echo -e "\nflags:$FLAGS"
else
  echo -e "\nflags: none"
fi
