#!/usr/bin/env bash
# Replace the retrieval service inside an allocation that is already running.
#
#   RSVC_JOBID=<jobid> SVC_INDEX=omt_opus bash slurm/retrieval_service_step.sh
#
# This is not an sbatch submission. It is `srun --jobid=<J> --overlap`, attaching to an allocation
# that already holds the node, its GPUs and its memory, and it replaces only the step. That is why
# there is no --partition, --constraint or --gres here, and why --mem=0 is right: an overlap step
# inherits the allocation's resources, and the service is that allocation's payload rather than a
# guest inside it. A diagnostic step is the opposite case and must pass a small explicit --mem, or
# it draws on the same cgroup and can starve the service it is measuring; verify_service.sh does.
#
# Switching the searched rendering (Knob 1) costs a full reload, so the point of this script is to
# pay the reload without also paying the queue wait for a fresh allocation. Everything the service
# itself needs is in slurm/retrieval_service.sbatch, which this script runs with `bash`: the
# #SBATCH headers are inert under bash, and the resources come from the allocation instead.
#
# Quiesce the pipeline fleet before running this. A restart under live workers costs those workers
# their retries: a topic whose attempts are all spent during the reload ends failed and is not
# retried.
#
# ---------------------------------------------------------------------------------------------
# The values this was run with, for the runs behind this repository. Three near-identical scripts,
# one per rendering, merged here into one; these are the values each of them carried.
#
#   allocation      11517386, held for the whole campaign
#   node            gpu053, 6 x A100-80, 128 cores, mem=996G
#   step            srun --jobid=11517386 --overlap -N1 -n1 -c 96 --mem=0
#   repository      /home/users/k/knafouj/TREC_RAGTIME_2026
#   index root      <artifact-root>/corpus/e2e/8fbe879560a4/corpus-preprocess/final/f308301501d9/
#                     index/949acf17d993
#   name and slot   rsvc-gpu053-enc1, kept identical across all three renderings so that a switch
#                   replaces the descriptor rather than adding a second one
#   queue           <repo>/logs/dev/rsvc_queues/gpu053, one queue shared by every rendering
#   registry        <repo>/logs/dev/rsvc_registry
#   store mirror    /dev/shm/ragtime-mirror
#   PLAID mirror    /hosttmp/rsvc-keep, -keep-omt, -keep-omt_opus  (one per rendering)
#   per rendering   original  config/e2e-original.yml,  no index override
#                   omt       config/e2e-original.yml,  --allow-index-override
#                   omt_opus  config/e2e-original.yml,  --allow-index-override
#
# The base config named here is a substitution: the omt and omt_opus services were started from a
# config that moves both knobs at once, which this repository does not ship. The substitution is
# sound, and the reason is worth stating rather than glossing: the service reads exactly three
# things out of the config -- the corpus root, the retrieval block, and retrieval.index -- and the
# two files are identical on the first two. They differ only in run identity, passage_lang,
# retrieval.index and output routing.
# passage_lang is Knob 2, which retrieval never applies: the service returns ids and scores, reads
# no passage text, and only republishes the leaf in its descriptor. So both configs resolve to the
# same searched system, and the difference reaches exactly two published descriptor fields,
# rendering_source.from_config and passage_lang. It does not reach a ranking.
#
# Why the override is recorded rather than hidden. The service treats a disagreement between
# --index and the config's own retrieval.index as a hard error, because a silent Knob-1
# substitution is exactly what the fairness guard exists to prevent. No e2e config names omt_opus
# (e2e-omt-weak moves passage_lang, not the index), and the one config that does, mlir-omt-weak.yml,
# resolves its corpus root under the mlir family, which is a different tree. So the launcher keeps
# an e2e base config and makes the substitution explicit. The published descriptor then carries
# rendering_source = {"index": "omt_opus", "from_config": "original", "override": true}, which is
# what makes an artifact produced under it identifiable afterwards.
#
# Reranking: every run reranked the fused top retrieval.reranker.depth, sliced in the service.
# ---------------------------------------------------------------------------------------------
set -uo pipefail

REPO="${RAGTIME_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1

RSVC_JOBID="${RSVC_JOBID:?RSVC_JOBID must name the allocation to attach to}"
SVC_INDEX="${SVC_INDEX:?SVC_INDEX must be one of original, omt, omt_opus}"
STEP_CPUS="${STEP_CPUS:-96}"

# Which config carries the corpus root and the shared blocks for this rendering, and whether the
# rendering has to be substituted into it. Overridable, so a config named on the command line wins.
case "$SVC_INDEX" in
  original) SVC_CONFIG="${SVC_CONFIG:-config/e2e-original.yml}"
            SVC_ALLOW_OVERRIDE="${SVC_ALLOW_OVERRIDE:-0}" ;;
  omt)      SVC_CONFIG="${SVC_CONFIG:-config/e2e-original.yml}"
            SVC_ALLOW_OVERRIDE="${SVC_ALLOW_OVERRIDE:-1}" ;;
  omt_opus) SVC_CONFIG="${SVC_CONFIG:-config/e2e-original.yml}"
            SVC_ALLOW_OVERRIDE="${SVC_ALLOW_OVERRIDE:-1}" ;;
  *) echo "ABORT: SVC_INDEX=$SVC_INDEX is not one of original, omt, omt_opus"; exit 1 ;;
esac

# One lane for the whole allocation: the same name and slot across renderings, so a switch replaces
# the descriptor a client resolves through instead of leaving two of them in the registry.
SVC_NAME="${SVC_NAME:-rsvc-$RSVC_JOBID}"
SVC_SLOT="${SVC_SLOT:-$SVC_NAME}"
SVC_TIER="${SVC_TIER:-deploy}"
SVC_QUEUE="${SVC_QUEUE:-$REPO/logs/rsvc/queue/$RSVC_JOBID}"
SVC_REGISTRY="${SVC_REGISTRY:-$REPO/logs/rsvc/registry}"

# The mirror path carries the rendering. The mirror step builds <root>/<lang>/part-NNNNN, with no
# rendering in the path, and skips a part whose destination already exists, so reusing one root
# across renderings serves the previous rendering's blobs under the new label and reports nothing.
SVC_PLAID_LOCAL="${SVC_PLAID_LOCAL:-/hosttmp/rsvc-keep-$SVC_INDEX}"
SVC_STORE_LOCAL="${SVC_STORE_LOCAL:-/dev/shm/ragtime-mirror}"
SVC_PLAID_LOW_MEMORY="${SVC_PLAID_LOW_MEMORY:-false}"

# One replica, which is the input that makes retrieval_service.sbatch derive the four-card language
# shard and set replicas to 0. Do not read these back off a running process: the service's argv
# holds derived values, and feeding those back describes a different system.
SVC_REPLICAS="${SVC_REPLICAS:-1}"
SVC_GPU_PLAID="${SVC_GPU_PLAID:-cuda:0}"
SVC_GPU_MTD="${SVC_GPU_MTD:-cuda:1}"
SVC_GPU_DENSE="${SVC_GPU_DENSE:-cuda:1}"
SVC_GPU_SPARSE="${SVC_GPU_SPARSE:-cuda:1}"
SVC_GPU_RERANK="${SVC_GPU_RERANK:-cuda:5}"
SVC_RERANK_BATCH="${SVC_RERANK_BATCH:-8}"
SVC_PART_WORKERS="${SVC_PART_WORKERS:-6}"
SVC_LI_WORKERS="${SVC_LI_WORKERS:-3}"
SVC_TORCH_THREADS="${SVC_TORCH_THREADS:-24}"
SVC_CELL_FAN="${SVC_CELL_FAN:-4}"
SVC_CELL_OMP="${SVC_CELL_OMP:-1}"
# Boolean gate on the sparse process fan, not a width. 0 disables it and leaves the sparse leg on
# one core; the per-part process budget is a separate knob inside the service.
SVC_CELL_PROCS="${SVC_CELL_PROCS:-1}"
SVC_PREFAULT_THREADS="${SVC_PREFAULT_THREADS:-12}"
SVC_WARM_WORKERS="${SVC_WARM_WORKERS:-12}"
SVC_MIRROR_OVERLAP="${SVC_MIRROR_OVERLAP:-on}"
# One reranker instance per card. retrieval_service.sbatch has no knob for this, so it goes through
# SVC_EXTRA; without it the service builds a single instance.
SVC_RERANK_DEVICES="${SVC_RERANK_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5}"
SVC_EXTRA="${SVC_EXTRA:---rerank-devices $SVC_RERANK_DEVICES}"

# ---------------------------------------------------------------------------
# Preflight. A partial index tree does not fail loudly: the stack opens fewer parts, ranks
# differently and says nothing, which is the same failure class as a missing leg. So the part
# counts are checked before the reload rather than inferred from a query afterwards.
# ---------------------------------------------------------------------------
if [ "${RSVC_SKIP_PREFLIGHT:-0}" = "1" ]; then
  echo "preflight skipped by RSVC_SKIP_PREFLIGHT=1"
else
  SVC_INDEX_ROOT="${SVC_INDEX_ROOT:?SVC_INDEX_ROOT must point at the index build directory \
(<artifact-root>/corpus/<family>/<chunker>/corpus-preprocess/final/<final>/index/<build>), \
or set RSVC_SKIP_PREFLIGHT=1}"
  echo "=== preflight: is the $SVC_INDEX index complete on this cluster? $(date -Is) ==="
  ok=1
  # The three non-English cells live under the rendering; English is shared across renderings
  # because it is never translated, so it is reported below rather than gated on: it is the same
  # tree whichever rendering is being served.
  for spec in es:20 ru:14 zh:21; do
    lang=${spec%%:*}; want=${spec##*:}
    got=$(ls "$SVC_INDEX_ROOT/$SVC_INDEX/$lang" 2>/dev/null | wc -l)
    if [ "$got" = "$want" ]; then state=ok; else state=INCOMPLETE; ok=0; fi
    printf "  %s/%-3s %3s/%-3s %s\n" "$SVC_INDEX" "$lang" "$got" "$want" "$state"
  done
  printf "  _shared/en %3s/23\n" "$(ls "$SVC_INDEX_ROOT/_shared/en" 2>/dev/null | wc -l)"
  [ -f "$SVC_INDEX_ROOT/manifest.json" ] || { echo "  manifest.json missing"; ok=0; }

  # /hosttmp is the container's view of the host's /tmp, so the mirror is checked on the host path.
  host_mirror="${SVC_PLAID_LOCAL/#\/hosttmp/\/tmp}"
  if [ -e "$host_mirror" ]; then
    echo "  $SVC_PLAID_LOCAL already exists; the mirror step would skip the copy and serve whatever"
    echo "  is in it under a $SVC_INDEX label. Remove it or pick a fresh path."
    ok=0
  else
    echo "  mirror path $SVC_PLAID_LOCAL is fresh  ok"
  fi

  if [ "$ok" -ne 1 ]; then
    echo; echo "refusing to launch, see the failures above"
    exit 1
  fi
  echo "  preflight ok, launching"; echo
fi

# SVC_PLAID_LOCAL_HOST is left at its default. retrieval_service.sbatch removes it on
# exit, and the mirror above is meant to outlive the step so that a switch back is cheap.
exec srun --jobid="$RSVC_JOBID" --overlap -N1 -n1 -c "$STEP_CPUS" --mem=0 env \
  ${RAGTIME_PYPATH:+RAGTIME_PYPATH="$RAGTIME_PYPATH"} \
  RAGTIME_REPO="$REPO" \
  SVC_CONFIG="$SVC_CONFIG" \
  SVC_INDEX="$SVC_INDEX" \
  SVC_ALLOW_OVERRIDE="$SVC_ALLOW_OVERRIDE" \
  SVC_QUEUE="$SVC_QUEUE" \
  SVC_REGISTRY="$SVC_REGISTRY" \
  SVC_NAME="$SVC_NAME" \
  SVC_SLOT="$SVC_SLOT" \
  SVC_TIER="$SVC_TIER" \
  SVC_REPLICAS="$SVC_REPLICAS" \
  SVC_GPU_PLAID="$SVC_GPU_PLAID" \
  SVC_GPU_MTD="$SVC_GPU_MTD" \
  SVC_GPU_DENSE="$SVC_GPU_DENSE" \
  SVC_GPU_SPARSE="$SVC_GPU_SPARSE" \
  SVC_GPU_RERANK="$SVC_GPU_RERANK" \
  SVC_RERANK_BATCH="$SVC_RERANK_BATCH" \
  SVC_PART_WORKERS="$SVC_PART_WORKERS" \
  SVC_LI_WORKERS="$SVC_LI_WORKERS" \
  SVC_TORCH_THREADS="$SVC_TORCH_THREADS" \
  SVC_CELL_FAN="$SVC_CELL_FAN" \
  SVC_CELL_OMP="$SVC_CELL_OMP" \
  SVC_CELL_PROCS="$SVC_CELL_PROCS" \
  SVC_PREFAULT_THREADS="$SVC_PREFAULT_THREADS" \
  SVC_WARM_WORKERS="$SVC_WARM_WORKERS" \
  SVC_MIRROR_OVERLAP="$SVC_MIRROR_OVERLAP" \
  SVC_PLAID_LOCAL="$SVC_PLAID_LOCAL" \
  SVC_PLAID_LOW_MEMORY="$SVC_PLAID_LOW_MEMORY" \
  SVC_STORE_LOCAL="$SVC_STORE_LOCAL" \
  SVC_EXTRA="$SVC_EXTRA" \
  bash "$REPO/slurm/retrieval_service.sbatch"
