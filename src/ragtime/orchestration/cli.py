"""``run --config config/<run>.yml``, the single config-driven entrypoint.

The flow, in order:

1. ``config.check([path])`` runs ``load`` plus ``family_guard`` as one hard pre-launch
   gate. On a divergent family it raises and the process aborts non-zero having
   submitted nothing: no plan, no GPU, no ``sbatch``.
2. ``plan.build_plan(cfg)`` expands the ``afterok``-wired JobDAG.
3. ``--dry-run`` prints the DAG and stops, ``--local`` runs it in this process, and
   otherwise ``launcher.submit`` fans it onto SLURM.

``run --stage <x>`` is the worker-side dispatch a thin sbatch template invokes:
``preprocess``, ``pipeline``, ``citation_scoring`` and ``select_serialize`` each run their
stage, and ``monitor`` raises :class:`NotImplementedError` by name. The stages are reached two
ways: ``run --config <cfg>`` with no ``--stage``, whose CORPUS node builds the corpus and index
through :func:`_submit_corpus`, and a direct per-stage call -- ``--stage pipeline`` from
``slurm/``'s worker launchers, one fleet at a time, and ``--stage citation_scoring`` to score a
topic on its own, outside ``pipeline.driver``'s inline call.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime import config
from ragtime.common import Layout, get_logger

from . import plan
from .plan import CITATION_SCORING, CORPUS, MONITORING, PIPELINE, SELECT_SERIALIZE
from .run_identity import cell_key, run_family

if TYPE_CHECKING:
    from ragtime.config import RunConfig

    from .plan import JobDAG, JobNode

__all__ = ["ARTIFACT_ROOT_ENV", "artifact_root", "build_parser", "main"]

_log = get_logger("orchestration.cli")

# The artifact root when nothing says otherwise: repo-relative, and the value every dev
# run and test fixture uses. See :func:`artifact_root`, which decides the root.
_LOCAL_ROOT = "runs"

#: The ``execution`` leaf carrying the artifact root as part of the run record.
ARTIFACT_ROOT_KEY = "artifact_root"
#: The environment variable carrying the artifact root as a site fact, for a config
#: that states none. Exported by :func:`_submit_dag` so a worker resolves the same root
#: its submitter did.
ARTIFACT_ROOT_ENV = "RAGTIME_ARTIFACT_ROOT"


def artifact_root(cfg: RunConfig) -> str:
    """Return the artifact root every ``Layout``, adapter and queue path hangs off.

    Resolution order: ``execution.artifact_root`` (the record), then
    ``$RAGTIME_ARTIFACT_ROOT`` (the site fact), then ``"runs"`` (the repo-relative
    default). One resolver, because ``common.Layout`` is the only translator from a run
    to a path and this is the only thing that decides where it is rooted.

    The knob lives in ``execution`` rather than in a hashed block because a filesystem
    path is a machine fact, not a semantic identity: hashing it would re-key the whole
    corpus tree the day storage moves, for a change that alters no artifact byte. Since
    ``family_guard`` never compares ``execution``, the root is written equal across a
    family by hand rather than enforced equal by the gate.

    A config value that disagrees with the environment variable is a hard error rather
    than a silent precedence rule: two roots each look complete, each carrying its own
    ``_SUCCESS`` tree, while holding half a corpus.
    """
    stated = cfg.blocks.get("execution", {}).get(ARTIFACT_ROOT_KEY)
    stated = str(stated).strip() if stated is not None else ""
    from_env = os.environ.get(ARTIFACT_ROOT_ENV, "").strip()
    if stated and from_env and os.path.normpath(stated) != os.path.normpath(from_env):
        raise config.ConfigError(
            f"execution.{ARTIFACT_ROOT_KEY}={stated!r} disagrees with "
            f"${ARTIFACT_ROOT_ENV}={from_env!r}: two artifact roots means two half-built "
            "corpora, each carrying its own _SUCCESS tree, and no artifact can say which "
            "one produced it. Unset the environment variable or fix the config."
        )
    return stated or from_env or _LOCAL_ROOT


# The thin sbatch template each DAG node dispatches. The templates carry their own
# #SBATCH gres and constraint lines, so the launcher adds no GPU flags.
_TEMPLATES_DIR = Path(__file__).parent / "slurm" / "templates"
# CORPUS is absent: it is not one job with one template but an ordered chain
# of substages, each routing to its own adapter-declared template.
#
# Three of the four nodes share `pipeline_worker.sbatch`, which hardcodes `run --stage
# pipeline` and does not read the exported `$STAGE`, so under SLURM the citation-scoring and
# select-serialize nodes re-enter the pipeline stage rather than run their own; the submitter
# exports no `PIPELINE_ROLE` either, so a task takes the default "all" role and seeds, works
# and drives in every array task at once; and `monitor.sbatch` dispatches `run --stage
# monitor`, which raises by name. The `slurm/` launchers drive these stages one at a time
# (`plan`'s module docstring names one per stage).
_TEMPLATE_OF = {
    PIPELINE: "pipeline_worker.sbatch",
    CITATION_SCORING: "pipeline_worker.sbatch",
    SELECT_SERIALIZE: "pipeline_worker.sbatch",
    MONITORING: "monitor.sbatch",
}


# --------------------------------------------------------------------------- #
# The corpus substage chain: the CORPUS node is one plan node but several
# afterok-chained seed -> worker-array -> drive submissions, one per substage.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Substage:
    """One corpus substage: its adapter factory plus its ``execution`` knobs.

    ``name`` is the logical substage id, the ``PREPROCESS_SUBSTAGE`` a worker is
    dispatched with. It is not the queue name: every substage after
    ``chunk`` hash-suffixes its own stage name, so a semantic-block edit resolves to a
    fresh queue subtree. That is why every path and template here is read off
    ``adapter.stage`` and ``adapter.template`` rather than off a literal.

    ``adapter`` imports its stage module function-locally, so orchestration's module
    graph still depends on no stage code.
    """

    name: str
    adapter: Callable[[Any], Any]
    shards_key: str
    oversub_key: str
    #: The ``execution`` leaf carrying this substage's reaper window in seconds. Per
    #: substage rather than one module constant, because per-unit work spans three
    #: orders of magnitude across the chain.
    max_age_key: str
    #: chunk alone needs the raw corpus fetched before it can shard it.
    needs_raw_corpus: bool = False


def _chunk_adapter(cfg: RunConfig) -> Any:
    from ragtime.preprocess import ChunkAdapter

    return ChunkAdapter(base=artifact_root(cfg))


def _merge_adapter(cfg: RunConfig) -> Any:
    from ragtime.preprocess.merge import MergeAdapter

    return MergeAdapter.for_config(cfg, base=artifact_root(cfg))


def _translate_identity_adapter(cfg: RunConfig) -> Any:
    from ragtime.preprocess.translate_omt import OmtIdentityAdapter

    return OmtIdentityAdapter(cfg, base=artifact_root(cfg))


def _translate_omt_adapter(cfg: RunConfig) -> Any:
    from ragtime.preprocess.translate_omt import OmtTranslateAdapter

    return OmtTranslateAdapter(cfg, base=artifact_root(cfg))


def _reconcile_adapter(cfg: RunConfig) -> Any:
    from ragtime.preprocess.reconcile import ReconcileAdapter

    return ReconcileAdapter.for_config(cfg, base=artifact_root(cfg))


def _len_max_adapter(cfg: RunConfig) -> Any:
    from ragtime.preprocess.len_max import LenMaxAdapter

    return LenMaxAdapter.for_config(cfg, base=artifact_root(cfg))


def _packing_adapter(cfg: RunConfig) -> Any:
    from ragtime.preprocess.packing import PackAdapter

    return PackAdapter.for_config(cfg, base=artifact_root(cfg))


def _vectorize_adapter(cfg: RunConfig) -> Any:
    from ragtime.preprocess.vectorize import VectorizeAdapter

    return VectorizeAdapter.for_config(cfg, base=artifact_root(cfg))


def _assemble_adapter(cfg: RunConfig) -> Any:
    from ragtime.preprocess.assemble import AssembleAdapter

    return AssembleAdapter.for_config(cfg, base=artifact_root(cfg))


#: Every stage `_dispatch_stage` handles. One list, so the argparse choices and the
#: dispatcher cannot disagree.
_DISPATCHABLE_STAGES: tuple[str, ...] = (
    "preprocess",
    "pipeline",
    "citation_scoring",
    "select_serialize",
    "monitor",
)

#: The ordered corpus substage chain. Each entry gets its own queue, template and array
#: width, and starts only ``afterok`` the previous entry's drive job, because the
#: previous substage's artifact is the next one's input: merge reads the merged chunk
#: spine, translate reads the merged unit map, reconcile reads both raw translation
#: parts, len_max reads reconcile's inventory, packing reads len_max's sidecar,
#: vectorize reads packing's passages and assemble reads vectorize's blocks. A new
#: corpus step is exactly one more adapter appended here, never a bespoke driver.
_CORPUS_SUBSTAGES: tuple[_Substage, ...] = (
    _Substage(
        "chunk",
        _chunk_adapter,
        "corpus_shards",
        "oversubscription",
        max_age_key="chunk_max_age_s",
        needs_raw_corpus=True,
    ),
    # The translation-unit map. CPU only (a pinned tokenizer, no model), and it must be
    # merged before translate: unit boundaries are as much an input to the MT call as
    # the beam size is.
    _Substage(
        "merge",
        _merge_adapter,
        "merge_shards",
        "merge_oversubscription",
        max_age_key="merge_max_age_s",
    ),
    # English is CPU-only identity work and must never claim a GPU just to pass text
    # through.
    _Substage(
        "translate_omt_identity",
        _translate_identity_adapter,
        "translate_identity_shards",
        "translate_identity_oversubscription",
        max_age_key="translate_identity_max_age_s",
    ),
    _Substage(
        "translate_omt",
        _translate_omt_adapter,
        "translate_shards",
        "translate_oversubscription",
        max_age_key="translate_max_age_s",
    ),
    # The sole producer of the final sentence inventory. It needs both raw translation
    # parts merged, which the afterok chain guarantees, and is CPU only.
    _Substage(
        "reconcile",
        _reconcile_adapter,
        "reconcile_shards",
        "reconcile_oversubscription",
        max_age_key="reconcile_max_age_s",
    ),
    # The rendering-invariant length sidecar: one row per final sentence carrying its
    # length in every rendering plus their maximum. It runs after reconcile, which mints
    # the final ids, and before packing, which packs against it.
    #
    # It reads all three renderings, and the low-tier `omt_opus` table is produced by an
    # OPUS-MT arm launched outside this chain (it reuses the translate adapter pointed
    # at an existing final node via `RAGTIME_INVENTORY_DIR`, because its work table is
    # the reconciled inventory rather than the spine). On a from-scratch build that arm
    # must run between `reconcile` and here; `LenMaxAdapter.bringup` says so by name
    # rather than failing deep in a worker.
    _Substage(
        "len_max",
        _len_max_adapter,
        "len_max_shards",
        "len_max_oversubscription",
        max_age_key="len_max_max_age_s",
    ),
    # The sole producer of a passage, keyed by its own hash one level inside the
    # reconcile node. Split out of `reconcile` so a packing edit cannot re-key that node
    # and orphan the translations inside it. CPU, and it opens no text at all: packing
    # branches on counts and paragraph indices only.
    _Substage(
        "packing",
        _packing_adapter,
        "pack_shards",
        "pack_oversubscription",
        max_age_key="pack_max_age_s",
    ),
    # The index build, split in two by resource class. Vectorize (GPU) turns passages
    # into vector blocks; all three encoders run a forward pass, so its adapter declares
    # the GPU template. The unit is one encode block; how many blocks a worker claims is
    # the separate `vectorize_blocks_per_task`.
    _Substage(
        "vectorize",
        _vectorize_adapter,
        "vectorize_shards",
        "vectorize_oversubscription",
        max_age_key="vectorize_max_age_s",
    ),
    # Assemble (CPU, inside the container) turns vector blocks into the FAISS, Seismic
    # and PLAID artifacts plus the manifest. It must be last, since `_mark_corpus_done`
    # fires after the final substage's drive. Its adapter declares the Apptainer-wrapping
    # CPU template: the three engines exist only in the container, while the work itself
    # touches no card.
    _Substage(
        "assemble",
        _assemble_adapter,
        "assemble_shards",
        "assemble_oversubscription",
        max_age_key="assemble_max_age_s",
    ),
)


def _substage_array_size(cfg: RunConfig, sub: _Substage) -> int:
    """Return this substage's worker-array width: ``ceil(shards / oversubscription)``.

    The same oversubscription rule ``plan.corpus_workers`` applies to chunk,
    generalized to one knob pair per substage. Never hardcoded: tuning a width is a
    throughput decision in the non-shared ``execution`` block.
    """
    ex = cfg.blocks["execution"]
    if sub.shards_key not in ex:
        raise config.ConfigError(
            f"execution.{sub.shards_key} is required for corpus substage '{sub.name}' "
            f"(the config is the complete record; widths are never defaulted in code)"
        )
    shards = int(ex[sub.shards_key])
    oversub = int(ex.get(sub.oversub_key, 1) or 1)
    return max(1, math.ceil(shards / max(1, oversub)))


def _substage_max_age(cfg: RunConfig, sub: _Substage) -> float:
    """Return this substage's reaper window: ``execution.<sub>_max_age_s`` or the default.

    A throughput and liveness fact, so it lives in the non-shared ``execution`` block
    beside the array widths and is never hashed: how long a worker may be silent before
    its shard is reclaimed changes no artifact byte, and hashing it would re-key the
    whole corpus the day a stage is retuned.

    Optional per substage, but not free: a window at or below the heartbeat interval
    reaps every live shard on its first pass, so it is rejected at load rather than
    discovered under full concurrency.
    """
    from . import saturate

    stated = cfg.blocks.get("execution", {}).get(sub.max_age_key)
    if stated is None:
        return saturate.MAX_AGE_S
    try:
        value = float(stated)
    except (TypeError, ValueError) as exc:
        raise config.ConfigError(
            f"execution.{sub.max_age_key} must be a number of seconds; got {stated!r}"
        ) from exc
    if value <= saturate.HEARTBEAT_S:
        raise config.ConfigError(
            f"execution.{sub.max_age_key}={value} is <= the heartbeat interval "
            f"({saturate.HEARTBEAT_S}s): the reaper would reclaim every live shard on its "
            "first pass, which is a permanent claim/requeue churn (reap_stale requeues "
            "without incrementing attempts, so nothing ever poisons and it never stops)"
        )
    return value


def _family_roster(cfg_path: Path, cfg: RunConfig) -> list[Path]:
    """Return the same-family sibling configs in ``cfg_path``'s directory.

    ``config.fairness``'s cross-run shared-block and seed-bank checks only fire with at
    least two family members, so handing ``family_guard`` a lone ``--config`` would make
    them structurally unreachable. The gate therefore runs over the whole roster
    discovered on disk, with the family derived from the loaded run rather than from
    hardcoded names. The target run is always in its own roster, and a config with no
    siblings is a legitimate family of one.
    """
    fam = run_family(cfg)
    roster: list[Path] = []
    for p in sorted(cfg_path.parent.glob("*.yml")):
        if p.resolve() == cfg_path.resolve():
            roster.append(p)  # the target loaded fine already, always keep it
            continue
        try:
            sib = config.load(p)
        except Exception as exc:  # noqa: BLE001 - a bad sibling must not crash a valid launch
            # Not a loadable run config: a ConfigError, a YAML parse or duplicate-key
            # error, an OSError. Skip it, but warn so a malformed intended sibling is
            # visible rather than silently dropped.
            _log.warning("cli.roster.skip_unloadable_sibling", path=str(p), error=str(exc))
            continue
        if run_family(sib) == fam:
            roster.append(p)
    if not any(p.resolve() == cfg_path.resolve() for p in roster):
        roster.append(cfg_path)
    return roster


def build_parser() -> argparse.ArgumentParser:
    """Build the ``run`` argument parser."""
    p = argparse.ArgumentParser(
        prog="run",
        description="Launch one self-contained RAGTIME config as resumable SLURM work.",
    )
    p.add_argument("--config", required=True, help="path to config/<run>.yml (the run record)")
    # The choices come from the shared constant, so a stage cannot be implemented and
    # dispatched while argparse still rejects it. Stages that are not yet built are
    # listed too, because their handlers raise by name with a specific reason, which is
    # more useful than argparse's generic refusal.
    p.add_argument(
        "--stage",
        choices=_DISPATCHABLE_STAGES,
        help="worker-side stage dispatch (invoked by the sbatch templates)",
    )
    p.add_argument(
        "--recompute-t2",
        action="store_true",
        help=(
            "select_serialize: emit ONLY Task 2, synthesising a task-2 deliverable if the config "
            "declares none. Rebuilds the T2 run file from finished cells without touching T1/T3."
        ),
    )
    p.add_argument("--seed", type=int, help="restrict to a single seed cell")
    # citation_scoring only. The `pipeline` stage cannot honour it: its unit of work is a
    # queue shard a worker self-claims, so selection is the queue's, not the caller's, and
    # `_run_pipeline` refuses the flag rather than accepting and ignoring it.
    p.add_argument(
        "--topics",
        help="citation_scoring: restrict to a comma-separated list of topic ids",
    )
    # There is no `--resume`: resuming needs no flag. Every unit carries a `_SUCCESS`
    # marker and every writer is atomic, so relaunching the same command recomputes only
    # what is missing.
    p.add_argument("--dry-run", action="store_true", help="print the JobDAG and exit; no submit")
    p.add_argument(
        "--local",
        action="store_true",
        help=(
            "run the DAG in-process (no SLURM); ends at the unimplemented monitoring node, "
            "which raises by name"
        ),
    )
    return p


def _dispatch_stage(
    cfg: RunConfig,
    stage: str,
    seed: int | None,
    topics: str | None,
    *,
    recompute_t2: bool = False,
) -> int:
    """Worker-side ``run --stage`` dispatch.

    Orchestration imports no stage code at module level, so each stage handler is a
    function-local import: the module graph stays one-directional while the worker can
    still run the stage it was dispatched to.
    """
    if stage == "preprocess":
        return _run_preprocess(cfg)
    if stage == "pipeline":
        return _run_pipeline(cfg, seed, topics)
    if stage == "select_serialize":
        return _run_select_serialize(cfg, seed, recompute_t2=recompute_t2)
    if stage == "citation_scoring":
        return _run_citation_scoring(cfg, seed, topics)
    # A stage that is not yet built raises by name rather than through a generic
    # catch-all, so "not implemented" never reads as "unknown stage".
    if stage == "monitor":
        raise NotImplementedError(
            "run --stage monitor: the monitoring rollup is not implemented in this release. The "
            "counters it would aggregate are persisted per cell (see `Layout.metrics()`), so the "
            "rollup can be run later over a family that has already finished."
        )
    raise ValueError(
        f"run --stage {stage!r} is not a known stage; expected one of "
        f"preprocess/pipeline/citation_scoring/select_serialize/monitor. "
        f"run_id={cfg.run_id} seed={seed} topics={topics}"
    )


def _run_preprocess(cfg: RunConfig) -> int:
    """Run the corpus-build saturation role dispatched to this process.

    Two environment variables select the unit of work, both set by
    :func:`_submit_corpus`. ``PREPROCESS_SUBSTAGE`` picks the link of the chain, and
    ``PREPROCESS_ROLE`` picks one of three separate processes: ``seed`` (CPU, once,
    fetches the corpus for chunk then seeds the shards), ``worker`` (each array task,
    ``run_worker`` only) and ``drive`` (CPU, once after the array drains, merges).
    ``--local`` sets neither and runs every substage's full lifecycle in order in one
    process.

    ``_mark_corpus_done`` fires only after the last substage's drive, so
    ``already_done(CORPUS)`` keeps its meaning: the node is done when the whole chain
    is.
    """
    from . import saturate

    root = artifact_root(cfg)
    role = os.environ.get("PREPROCESS_ROLE", "all")
    only = os.environ.get("PREPROCESS_SUBSTAGE") or None
    substages = [s for s in _CORPUS_SUBSTAGES if only is None or s.name == only]
    if not substages:
        known = [s.name for s in _CORPUS_SUBSTAGES]
        raise ValueError(f"unknown PREPROCESS_SUBSTAGE={only!r} (known: {known})")

    for sub in substages:
        adapter = sub.adapter(cfg)
        wq = saturate.queue_for(cfg, adapter, base=root)
        max_age = _substage_max_age(cfg, sub)
        _log.info(
            "cli.preprocess",
            substage=sub.name,
            stage=adapter.stage,
            role=role,
            root=root,
            max_age_s=max_age,
        )
        if role in ("all", "seed"):
            if sub.needs_raw_corpus:
                from ragtime.preprocess import download

                download(cfg, base=root)  # the shards() reader needs the corpus first
            saturate.seed(cfg, adapter, wq)
        if role in ("all", "worker"):
            # The reaper window is the substage's own: per-unit work across the chain
            # ranges from seconds to tens of minutes, which no single constant fits.
            saturate.run_worker(cfg, adapter, wq, max_age=max_age)
        if role in ("all", "drive"):
            saturate.drive(cfg, adapter, wq)
    # Plan-level family _SUCCESS so already_done(CORPUS) is true, after the last link.
    if role in ("all", "drive") and substages[-1] is _CORPUS_SUBSTAGES[-1]:
        _mark_corpus_done(cfg)
    return 0


def _run_pipeline(cfg: RunConfig, seed: int | None, topics: str | None) -> int:
    """Run the online half, fanned exactly like the corpus build: seed, N workers, drive.

    This reuses ``saturate`` rather than adding a second fan. The exclusion the fleet
    needs, that two workers never take the same topic, is ``workqueue.claim``'s atomic
    rename, with the same forked heartbeat child and the same stale-claim reaper. What
    is specific to the online half is only what a shard is: one ``(topic, seed)`` cell,
    defined by ``pipeline.topic_fleet.TopicCellAdapter``.

    ``PIPELINE_ROLE`` selects the role, exactly as ``PREPROCESS_ROLE`` does for the
    corpus chain:

    * ``seed``: CPU, once. Fill the queue with one file per ``(topic, seed)``.
    * ``worker``: each array task. ``run_worker`` brings its clients up once and then
      drains: claim, drive one topic, validate, done, claim the next. A GPU pair
      therefore owns a topic end to end, which is the topic-affinity rule expressed as
      a work queue rather than as a static modulo shard.
    * ``drive``: CPU, once after the array drains. Hard-stops on ``failed/``, then
      merges. ``TopicCellAdapter.merge`` is a deliberate no-op, because
      ``select_serialize`` is its own DAG node over the finished cell and folding it in
      here would run the terminal dedup once per worker instead of once per cell.

    ``--local`` sets no role and runs all three in this process, which is what makes a
    one-topic dev run and a full production run the same code path.

    ``topics`` is refused rather than honoured. A shard here is claimed by whichever
    worker wins the atomic rename, so a caller cannot pick which topic this process gets;
    accepting the flag and quietly ignoring it would let an operator believe a run was
    narrowed when it was not. Narrow with ``--seed``, which does select a cell.
    """
    from . import saturate

    if topics is not None:
        raise ValueError(
            "run --stage pipeline does not support --topics: a shard is claimed off the "
            "work queue by atomic rename, so the queue decides which topic this worker "
            "gets, not the caller. Use --seed to restrict the cell, or seed a queue "
            "containing only the topics you want. (--topics is honoured by "
            "`run --stage citation_scoring`, which sweeps finished topic directories.)"
        )

    root = artifact_root(cfg)
    role = os.environ.get("PIPELINE_ROLE", "all")
    variant = str(getattr(cfg, "passage_lang", "original"))
    seeds = _pipeline_seeds(cfg, seed)

    from ragtime.pipeline.topic_fleet import TopicCellAdapter

    adapter = TopicCellAdapter(
        registry=os.environ.get("RAGTIME_RSVC_REGISTRY", ""),
        variant=variant,
        seeds=seeds,
        # One queue per run: `variant` alone collides across the whole mlir family,
        # whose three arms all read `original`, so only one arm could ever claim work.
        run_id=str(getattr(cfg, "run_id", "") or ""),
    )
    wq = saturate.queue_for(cfg, adapter, base=root)
    _log.info(
        "cli.pipeline", role=role, stage=adapter.stage, root=root,
        variant=variant, seeds=list(seeds),
    )
    if role in ("all", "seed"):
        saturate.seed(cfg, adapter, wq)
    if role in ("all", "worker"):
        # An unseeded queue is not a drained one. `run_worker` returns normally when
        # there is nothing to claim, which is what finishing looks like, but it returns
        # the same way when the queue was never seeded, and a supervisor reading rc=0 as
        # "this config is drained" would retire a whole arm having produced nothing.
        # The refusal is therefore on "no shards anywhere", not on "pending is empty",
        # which is what a genuinely finished run looks like.
        if not any(
            any(Path(d).glob("*"))
            for d in (wq.pending, wq.running, wq.done, wq.failed)
        ):
            raise RuntimeError(
                f"pipeline queue {wq.base} holds no shards in pending/running/done/failed: it "
                "was never seeded, or was moved or deleted underneath this worker. Refusing to "
                "exit 0, because a caller cannot distinguish that from a completed run. Run the "
                "`seed` role for this config first."
            )
        saturate.run_worker(cfg, adapter, wq)
    if role in ("all", "drive"):
        saturate.drive(cfg, adapter, wq)
    return 0


def _flush_merged(stats: Any, layout: Any, *, prefix: str) -> Any:
    """Flush a post-hoc stage's counters without destroying the cell's own.

    ``common.stats.flush`` rewrites ``layout.metrics()`` from the passed bus alone,
    which is correct for a cell re-run and wrong for a stage that runs afterwards over
    a finished cell, whose counters are already on disk and unknown to the post-hoc
    bus. So: re-emit what is on disk into the bus first, then flush once. Rows whose
    ``metric_id`` starts with ``prefix`` are dropped before re-emitting, so re-running
    the sweep replaces this stage's own counters rather than doubling them, since
    ``Statistics.emit`` accumulates on a repeated key.
    """
    from ragtime.common.stats import flush as _flush

    existing = layout.metrics()
    if existing.exists():
        for line in existing.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn line is not worth losing the rest of the file over
            mid = str(row.get("metric_id") or "")
            if not mid or mid.startswith(prefix):
                continue
            stats.emit(mid, float(row.get("value") or 0.0), **(row.get("slices") or {}))
    return _flush(stats, layout)


def _run_citation_scoring(cfg: RunConfig, seed: int | None, topics: str | None) -> int:
    """Score every finished topic's citations against the shared vLLM.

    The citation score is the assessment priority in the submission format, so a run
    serialized without this stage hands the assessors an all-zero column and no
    ordering at all. ``select_serialize`` distinguishes "the scorer never ran" from "it
    ran and found nothing", and this is the stage that makes the first case stop
    happening.

    The fan is per topic, not per cell, because ``Layout.citation_scores()`` is a
    per-topic directory: one call per cell could only ever write one topic's scores.
    The layout comes from ``driver.topic_layout`` rather than being string-built.

    Only finished topics are scored. A topic without ``_SUCCESS`` is mid-flight, its
    bank is not final and its loop records are still being written, so scoring it would
    produce a file that looks complete and is not. Skipped topics are counted and
    printed rather than silently passed over.

    Re-runnable by design: it recomputes and rewrites, so running it again over a
    growing family is safe and is how a fleet that finishes topics gradually gets
    scored without a barrier.
    """
    import asyncio

    from ragtime.common import Statistics
    from ragtime.pipeline.citation_scoring.scorer import score_citations
    from ragtime.pipeline.driver import topic_layout
    from ragtime.serving.registry import build_clients

    root = artifact_root(cfg)
    variant = str(getattr(cfg, "passage_lang", "original"))
    seeds = _pipeline_seeds(cfg, seed)
    only = {t.strip() for t in topics.split(",")} if topics else None
    clients = build_clients(cfg)

    scored = skipped = failed = 0
    for s in seeds:
        cell_dir = Path(root) / cell_key(cfg.run_id, variant, s)
        topics_dir = cell_dir / "topics"
        if not topics_dir.is_dir():
            _log.warning("cli.citation_scoring.no_cells", cell=str(cell_dir), seed=s)
            continue
        for topic_dir in sorted(p for p in topics_dir.iterdir() if p.is_dir()):
            topic_id = topic_dir.name
            if only is not None and topic_id not in only:
                continue
            if not (topic_dir / "_SUCCESS").exists():
                skipped += 1
                continue
            layout = topic_layout(cfg, variant, s, topic_id)
            stats = Statistics()
            try:
                path = asyncio.run(
                    score_citations(cfg, layout=layout, llm=clients.llm, seed=s, stats=stats)
                )
            except Exception as exc:  # noqa: BLE001 - one bad topic must not lose the rest
                failed += 1
                _log.warning(
                    "cli.citation_scoring.failed",
                    topic=topic_id, seed=s,
                    error_type=type(exc).__name__, error=repr(exc),
                )
                continue
            _flush_merged(stats, layout, prefix="citation_scoring.")
            scored += 1
            _log.info("cli.citation_scoring.wrote", topic=topic_id, seed=s, path=str(path))
    # The stage states its own coverage: "0 scored" and "0 topics existed" are different
    # outcomes and a bare rc=0 cannot tell them apart.
    _log.info("cli.citation_scoring.done", scored=scored, skipped_unfinished=skipped, failed=failed)
    print(f"citation_scoring: scored={scored} skipped_unfinished={skipped} failed={failed}")
    return 1 if failed and not scored else 0


def _run_select_serialize(cfg: RunConfig, seed: int | None, *, recompute_t2: bool = False) -> int:
    """Project this run's finished cells into the T1/T2/T3 files ``config.outputs`` declares.

    One job per cell, and no work queue: ``select_serialize`` reads
    directories that are already finished (``load.topic_dirs`` skips any topic without
    its ``_SUCCESS``), so there is no shard to claim, no heartbeat and nothing to reap.

    ``embed`` and ``confirm`` are sliced from the one ``ClientBundle`` and injected,
    because ``select_serialize`` builds no client of its own. ``confirm`` is the shared
    vLLM under greedy decode; ``embed`` is the same resident dense identity retrieval
    and decompose use, so the terminal dedup measures similarity in the space the rest
    of the run lives in.

    A validator rejection propagates: ``project`` raises ``SubmissionRejected`` with
    nothing written, because a file NIST rejects is worse than no file at all.
    """
    import asyncio

    from ragtime.common import Layout, Statistics
    from ragtime.common.stats import flush as flush_stats
    from ragtime.pipeline.select_serialize.project import project
    from ragtime.serving.registry import build_clients

    root = artifact_root(cfg)
    variant = str(getattr(cfg, "passage_lang", "original"))
    seeds = _pipeline_seeds(cfg, seed)
    topics_path = _topics_path(cfg)
    clients = build_clients(cfg)

    rc = 0
    for s in seeds:
        cell_dir = Path(root) / cell_key(cfg.run_id, variant, s)
        if not (cell_dir / "topics").is_dir():
            # Not an error: a family is serialized as its cells finish, so a seed whose
            # cells have not run yet is simply not ready.
            _log.warning("cli.select_serialize.no_cells", cell=str(cell_dir), seed=s)
            continue
        layout = Layout(run_dir=cell_dir, outputs=getattr(cfg, "outputs", None), base=root)
        stats = Statistics()
        _log.info("cli.select_serialize", cell=str(cell_dir), variant=variant, seed=s)
        result = asyncio.run(
            project(
                cfg,
                cell_dir=cell_dir,
                layout=layout,
                topics_path=topics_path,
                submission_root=root,
                embed=_serialize_embed(clients),
                confirm=_serialize_confirm(cfg, clients, s),
                stats=stats,
                variant=variant,
                seed=s,
                recompute_t2=recompute_t2,
            )
        )
        flush_stats(stats, layout)
        for p in result.paths:
            _log.info("cli.select_serialize.wrote", path=str(p))
        # The coverage dict states which tracks were emitted, which topics were skipped
        # for being incomplete, and whether the validator actually ran.
        _log.info("cli.select_serialize.done", seed=s, coverage=result.coverage)
        print(f"seed={s} wrote={[str(p) for p in result.paths]} coverage={result.coverage}")
    return rc


def _serialize_embed(clients: Any) -> Any:
    """Return ``clients.query_dense.embed`` with ``mode="dense"`` pinned at the slice point.

    ``query_dense``, not ``embedder``: the registry builds ``embedder`` from the
    query-time ``retrieval.dense`` leaf, which no shipped ``e2e-*`` config declares, so
    it is ``None`` there; and where it is declared it need not be the model the index
    was encoded with, which would put the terminal dedup in a different embedding space
    from the rest of the run.
    """
    from functools import partial

    return partial(clients.query_dense.embed, mode="dense")


def _serialize_confirm(cfg: RunConfig, clients: Any, seed: int) -> Any:
    """Return the answer-merge gate: an embedding candidate still needs the LLM to agree.

    Issued through the same ``clients.llm`` singleton and the same compiled ``dedup``
    schema decompose uses, so it is a different call site with the same vocabulary
    rather than a second gate. Greedy and seed-pinned, because the terminal dedup
    decides what reaches a submission and must not vary run to run.
    """
    from ragtime.pipeline.decompose.grow_nuggets import decoding_kwargs
    from ragtime.pipeline.decompose.prompts import dedup_confirm_prompt

    async def _confirm(kept: str, candidate: str) -> bool:
        obj = await clients.llm.generate(
            clients.schemas.dedup, dedup_confirm_prompt(kept, candidate), seed,
            **decoding_kwargs(cfg),
        )
        return bool(obj["duplicate"])

    return _confirm


def _topics_path(cfg: RunConfig) -> Path:
    """Return the run's own topics file, from the config record."""
    blocks = getattr(cfg, "blocks", None) or {}
    block = blocks.get("topics")
    rel = block if isinstance(block, str) else (block or {}).get("path")
    if not rel:
        raise ValueError("config has no `topics:`; select_serialize will not guess a topics file")
    source = getattr(cfg, "source_path", None)
    base = Path(source).resolve().parent.parent if source else Path.cwd()
    return base / str(rel)


def _pipeline_seeds(cfg: RunConfig, seed: int | None) -> tuple[int, ...]:
    """Return the seeds this run fans over, read from the config.

    An explicit ``--seed`` narrows to one cell, which is the dev path and the
    production-scale unit; otherwise every seed the run record declares is fanned. An
    integer ``cfg.seeds`` means "seeds 0..n-1". It is read here rather than in the
    adapter, so the adapter takes a plain sequence and stays testable without a config.
    """
    if seed is not None:
        return (int(seed),)
    declared = getattr(cfg, "seeds", 1)
    if isinstance(declared, int):
        return tuple(range(max(1, declared)))
    return tuple(int(s) for s in declared)


def _mark_corpus_done(cfg: RunConfig) -> None:
    """Write the CORPUS node's family ``_SUCCESS`` where ``plan.cell_artifact`` checks."""
    from ragtime.common.io import success_marker
    from ragtime.config import all_hashes

    root = artifact_root(cfg)
    layout = Layout(run_dir=root, base=root)
    corpus = layout.corpus_dir(run_family(cfg), all_hashes(cfg)["chunker"])
    marker = success_marker(corpus)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(b"")


def _local_runner(cfg: RunConfig) -> Callable[[JobNode], None]:
    """A per-node runner for ``--local`` that dispatches each node's stage in-process.

    Unlike the SLURM path, this one really does dispatch ``node.stage``, so ``--local``
    is the only way to drive the whole declared DAG. It writes no node-level ``_SUCCESS``
    except the corpus node's (:func:`_mark_corpus_done`), so a second ``--local`` run
    re-enters the online nodes and leans on their own inner idempotence; and it ends on
    the terminal monitoring node, whose handler raises ``NotImplementedError`` by name.
    A ``--local`` run therefore always finishes with that refusal, after every implemented
    stage has run.
    """

    def _run(node: JobNode) -> None:
        _dispatch_stage(cfg, node.stage, None, None)

    return _run


def _print_dag(dag: JobDAG) -> None:
    print(dag.render())
    for parent, child in dag.edges:
        print(f"edge: {parent} -> {child} (afterok)")


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for the ``run`` console script. Returns the process exit code.

    The fairness gate runs first and is not caught here: a ``FairnessError`` or
    ``ConfigError`` propagates, so ``run`` aborts non-zero with nothing submitted.
    """
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)

    # 1. Hard fairness gate: load the target, then family_guard the whole family roster
    #    (two or more members make the cross-run checks reachable), before any plan,
    #    GPU or sbatch.
    cfg = config.load(cfg_path)
    config.check(_family_roster(cfg_path, cfg))

    # Worker-side stage dispatch bypasses planning: a template already picked the cell.
    if args.stage is not None:
        return _dispatch_stage(
            cfg,
            args.stage,
            args.seed,
            args.topics,
            recompute_t2=bool(getattr(args, "recompute_t2", False)),
        )

    # 2. Expand the plan.
    dag = plan.build_plan(cfg)
    _log.info("cli.plan", run_id=dag.run_id, family=dag.family, nodes=len(dag.nodes))

    # 3. Route.
    if args.dry_run:
        _print_dag(dag)
        return 0
    if args.local:
        executed = plan.run_local(dag, _local_runner(cfg), root=artifact_root(cfg))
        _log.info("cli.local.done", run_id=dag.run_id, executed=executed)
        return 0

    # SLURM submission path, reachable only where `sbatch` exists.
    _submit_dag(dag, cfg_path, cfg)
    return 0


def _submit_dag(dag: JobDAG, cfg_path: Path, cfg: RunConfig) -> dict[str, int]:
    """Submit each DAG node to SLURM via its thin template, wired with ``afterok``.

    Each node dispatches its own ``slurm/templates/*.sbatch`` with ``--export``
    carrying ``CONFIG`` and ``STAGE`` (plus ``ALL``, so the login environment
    propagates); the array-shaped ``PIPELINE`` node fans as a ``0..array_size-1``
    array. The templates own their own gres and constraint lines, so the launcher adds
    none. ``WQ_DIR``, ``RAGTIME_SIF`` and ``RAGTIME_ARTIFACT_ROOT`` are derived from
    ``Layout`` and :func:`artifact_root` and exported here, so a worker's Layout is
    rooted where the submitter's was even when the root came from the environment.

    ``STAGE`` is a record, not a dispatch: every template hardcodes its own
    ``run --stage`` and none reads ``$STAGE``, so the exported value only tells a reader
    of the job environment which node a job belongs to.

    The CORPUS node un-fuses into an ordered chain of substages, each of them three
    ``afterok``-wired submissions: ``seed`` (CPU, once), a ``worker`` array, and
    ``drive`` (CPU, once). Downstream ``afterok`` on CORPUS binds to the last
    substage's drive job, so the pipeline starts only once the whole chain is merged.
    """
    from .slurm import launcher

    # Layout is the one path translator: WQ_DIR and RAGTIME_SIF are read from it rather
    # than duplicated as strings. The corpus queue is keyed off the semantic chunker
    # hash carried in the CORPUS node key.
    root = artifact_root(cfg)
    layout = Layout(run_dir=root, base=root)
    chunker_hash = dag.node(CORPUS).key.split(":", 1)[1]
    sif = layout.sif_path()

    jobids: dict[str, int] = {}
    for node in dag.nodes:
        after = [jobids[parent] for parent in node.after]
        base_export = (
            f"ALL,CONFIG={cfg_path},STAGE={node.stage},RAGTIME_SIF={sif},"
            f"{ARTIFACT_ROOT_ENV}={root}"
        )
        if node.name == CORPUS:
            jobids[node.name] = _submit_corpus(
                launcher, cfg, after, base_export, layout, dag.family, chunker_hash
            )
            continue
        template = _TEMPLATES_DIR / _TEMPLATE_OF[node.name]
        if node.array_size > 1:
            jobid = launcher.submit_array(
                str(template), node.array_size, after=after, export=base_export, gpu=False
            )
        else:
            jobid = launcher.submit(str(template), after=after, export=base_export, gpu=False)
        jobids[node.name] = jobid
        _log.info(
            "cli.submit",
            node=node.name,
            jobid=jobid,
            template=template.name,
            array=node.array_size,
        )
    return jobids


def _submit_corpus(
    launcher: object,
    cfg: RunConfig,
    after: list[int],
    base_export: str,
    layout: Layout,
    family: str,
    chunker_hash: str,
) -> int:
    """Submit the corpus substage chain; return the last substage's drive job id.

    Per substage, three role-separated SLURM jobs (``PREPROCESS_ROLE`` plus the
    substage's own ``WQ_DIR`` and ``PREPROCESS_SUBSTAGE``) with an ``sbatch --array``
    in the middle. Each substage starts ``afterok`` the previous substage's drive, and
    the first keeps the CORPUS node's own dependencies, so the corpus is done only when
    the last link has merged.

    Every per-substage value is read off the adapter (``adapter.stage`` for the queue
    path, ``adapter.template`` for the sbatch script), never off a literal: every
    post-chunk adapter hash-suffixes its stage name, so a literal here would point at a
    queue nobody writes to. Templates own their gres and constraint lines, so
    ``gpu=False`` keeps the launcher from adding a second, conflicting constraint.
    """
    prev: list[int] = list(after)
    drive_id = 0
    for sub in _CORPUS_SUBSTAGES:
        adapter = sub.adapter(cfg)
        template = _TEMPLATES_DIR / adapter.template
        wq_dir = layout.wq_dir(family, chunker_hash, adapter.stage)
        width = _substage_array_size(cfg, sub)
        e = f"{base_export},WQ_DIR={wq_dir},PREPROCESS_SUBSTAGE={sub.name}"
        seed_id = launcher.submit(  # type: ignore[attr-defined]
            str(template), after=prev, export=f"{e},PREPROCESS_ROLE=seed", gpu=False
        )
        worker_id = launcher.submit_array(  # type: ignore[attr-defined]
            str(template),
            width,
            after=[seed_id],
            export=f"{e},PREPROCESS_ROLE=worker",
            gpu=False,
        )
        # afterany, not afterok: `afterok` on an N-task worker array is satisfied only
        # if every task exits 0, so a single preempted task would strand the merge as
        # DependencyNeverSatisfied with the queue fully drained. `drive` carries the
        # stronger, artifact-level guard anyway.
        drive_id = launcher.submit(  # type: ignore[attr-defined]
            str(template), after_any=[worker_id], export=f"{e},PREPROCESS_ROLE=drive", gpu=False
        )
        _log.info(
            "cli.submit.corpus",
            substage=sub.name,
            stage=adapter.stage,
            template=template.name,
            seed=seed_id,
            worker_array=worker_id,
            drive=drive_id,
            width=width,
        )
        prev = [drive_id]
    return drive_id


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
