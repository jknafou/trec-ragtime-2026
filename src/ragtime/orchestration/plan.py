"""The config-to-JobDAG planner and the ``--local`` execution engine.

``build_plan(cfg)`` expands one validated config into the ``afterok`` chain
``corpus-preprocess`` (planned once per family, since the corpus is byte-identical
across its renderings) -> a per-run ``pipeline`` array (seeds x topic shards) ->
post-hoc ``citation_scoring`` -> ``select_serialize`` -> ``monitoring``. Nodes carry a
``config_hash``-derived skip key, so ``already_done`` (and therefore resume) is a
no-op over completed work. ``run_local`` runs the DAG in dependency order in this
process, skipping any node already done.

Each node is also reachable on its own, which is how the stages are driven in practice.
``cli._submit_corpus`` fans ``corpus-preprocess``'s nine substages onto SLURM;
``slurm/pipeline_workers.sh`` and ``slurm/vllm_service.sbatch`` invoke ``run --config <cfg>
--stage pipeline`` directly, so a fleet can gain and lose GPU pairs while the run continues; the
citation scorer runs inline at the end of each topic (``pipeline.driver``), with ``run --stage
citation_scoring`` as the standalone entrypoint; ``slurm/serialize_all_arms.sbatch`` serialises the
arms; and ``slurm/monitor_run.sh`` is the live monitor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ragtime.common import Layout, get_logger
from ragtime.common.io import is_done
from ragtime.config import all_hashes

from .determinism import expand_seeds
from .run_identity import cell_key, run_family, run_id, variant

if TYPE_CHECKING:
    from ragtime.config import RunConfig

__all__ = [
    "JobDAG",
    "JobNode",
    "already_done",
    "build_plan",
    "cell_artifact",
    "corpus_workers",
    "node_artifact",
    "node_cells",
    "run_local",
]

_log = get_logger("orchestration.plan")

# The single shared corpus stage, planned once per family.
CORPUS = "corpus-preprocess"
# The per-run linear tail after the corpus is built.
PIPELINE = "pipeline"
# The post-hoc scorer node, named for the `citation_scoring` stage it dispatches.
CITATION_SCORING = "citation_scoring"
SELECT_SERIALIZE = "select_serialize"
MONITORING = "monitoring"

# node.name -> the `run --stage` value the node dispatches. This is what `--local` runs
# in-process and what `cli._submit_dag` exports as `STAGE=`; note that the sbatch
# templates hardcode their own `run --stage` rather than reading `$STAGE` (see
# `cli._TEMPLATE_OF`).
_STAGE_OF = {
    CORPUS: "preprocess",
    PIPELINE: "pipeline",
    CITATION_SCORING: "citation_scoring",
    SELECT_SERIALIZE: "select_serialize",
    MONITORING: "monitor",
}


@dataclass(frozen=True, slots=True)
class JobNode:
    """One node of a run's JobDAG.

    ``name`` is unique within the DAG; ``stage`` is the ``run --stage`` value a thin
    sbatch template dispatches; ``key`` is the ``config_hash``-derived skip key;
    ``after`` names the ``afterok`` parents; ``array_size`` is the static-array width
    (seeds x topic shards for the pipeline node, otherwise 1); ``family_shared`` marks
    the corpus node whose identity is shared across a family.
    """

    name: str
    stage: str
    key: str
    after: tuple[str, ...]
    array_size: int = 1
    family_shared: bool = False


@dataclass(frozen=True, slots=True)
class JobDAG:
    """A run's job graph: an ordered tuple of nodes wired by ``afterok`` edges.

    Frozen and value-equal, so ``build_plan`` called twice on the same config yields
    an equal DAG with stable node and edge ordering.
    """

    run_id: str
    family: str
    variant: str
    nodes: tuple[JobNode, ...]

    def node(self, name: str) -> JobNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"no node {name!r} in DAG for run {self.run_id!r}")

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        """The ``afterok`` edges as ordered ``(parent, child)`` pairs."""
        return tuple(
            (parent, n.name) for n in self.nodes for parent in n.after
        )

    def render(self) -> str:
        """Return a deterministic multi-line rendering of the DAG, for ``--dry-run``."""
        lines = [
            (
                f"run_id={self.run_id} (<=25: {len(self.run_id)} chars)  "
                f"family={self.family}  variant={self.variant}"
            ),
            "nodes:",
        ]
        for n in self.nodes:
            after = f" afterok:{','.join(n.after)}" if n.after else ""
            arr = f" array=0-{n.array_size - 1}" if n.array_size > 1 else ""
            shared = " [family-shared]" if n.family_shared else ""
            lines.append(f"  - {n.name} (stage={n.stage} key={n.key[:12]}){arr}{after}{shared}")
        return "\n".join(lines)


def corpus_workers(cfg: RunConfig) -> int:
    """Return the corpus work-queue's worker-array width, from config.

    The corpus build seeds roughly ``corpus_shards`` doc-balanced shards and saturates
    ``ceil(corpus_shards / oversubscription)`` concurrent workers. Both knobs live in
    the non-shared, non-hashed ``execution`` block (execution parallelism, not chunking
    semantics), so a per-config tweak neither trips ``family_guard`` nor changes the
    corpus artifact path.
    """
    c = cfg.blocks["execution"]
    shards = int(c["corpus_shards"])
    oversub = int(c["oversubscription"])
    return max(1, math.ceil(shards / max(1, oversub)))


def _topic_shards(cfg: RunConfig) -> int:
    """Return the per-topic fan width of the online ``PIPELINE`` array.

    The online topic fan needs the resolved topic count, which does not exist until the
    topics file is read at pipeline time, so this is currently 1 and widening it is a
    single-function change. Distinct from the corpus work queue's document sharding,
    which ``saturate`` owns through the stage adapter.
    """
    return 1


def build_plan(cfg: RunConfig) -> JobDAG:
    """Expand ``cfg`` into its ``afterok``-wired JobDAG. Pure and deterministic.

    The corpus node is keyed on the family-invariant ``chunker`` block hash, so every
    family member plans the same corpus node; the per-run tail is keyed on the run's
    own block hashes.
    """
    rid = run_id(cfg)
    fam = run_family(cfg)
    var = variant(cfg)
    hashes = all_hashes(cfg)

    corpus_key = f"{fam}:{hashes['chunker']}"
    n_seeds = len(expand_seeds(cfg))
    topic_shards = _topic_shards(cfg)
    run_key = f"{rid}:{hashes['llm']}"

    nodes = (
        JobNode(
            CORPUS,
            _STAGE_OF[CORPUS],
            corpus_key,
            after=(),
            array_size=corpus_workers(cfg),
            family_shared=True,
        ),
        JobNode(
            PIPELINE,
            _STAGE_OF[PIPELINE],
            f"{run_key}:pipeline",
            after=(CORPUS,),
            array_size=n_seeds * topic_shards,
        ),
        JobNode(
            CITATION_SCORING,
            _STAGE_OF[CITATION_SCORING],
            f"{run_key}:citation_scoring",
            after=(PIPELINE,),
        ),
        JobNode(
            SELECT_SERIALIZE,
            _STAGE_OF[SELECT_SERIALIZE],
            f"{run_key}:select_serialize",
            after=(CITATION_SCORING,),
        ),
        JobNode(
            MONITORING,
            _STAGE_OF[MONITORING],
            f"{run_key}:monitoring",
            after=(SELECT_SERIALIZE,),
        ),
    )
    return JobDAG(run_id=rid, family=fam, variant=var, nodes=nodes)


def node_cells(node: JobNode) -> range:
    """Return the cell indices of ``node`` (``0..array_size-1``).

    A single-cell node has exactly cell 0; the ``PIPELINE`` array node has one cell per
    ``(seed, topic-shard)``. Since ``build_plan`` sets ``array_size = n_seeds *
    topic_shards`` and ``expand_seeds`` yields ``0..n_seeds-1``, with ``topic_shards``
    currently 1, cell index ``i`` maps to seed ``i``.
    """
    return range(max(1, node.array_size))


def cell_artifact(root: str | Path, dag: JobDAG, node: JobNode, cell: int) -> Path:
    """Return the artifact path of one ``(variant, seed/cell)`` cell of ``node``.

    Family-shared corpus nodes live under a family root and are cell-invariant;
    per-run nodes live under the ``(run_id, variant, seed)`` cell directory, so each
    seed's cell carries its own ``_SUCCESS`` marker.
    """
    base = Path(root)
    if node.family_shared:
        # The corpus artifact is keyed off the semantic chunker hash carried in the
        # node key, and resolved through Layout so the planner and the corpus-build
        # driver cannot diverge on the path.
        chunker_hash = node.key.split(":", 1)[1]
        return Layout(run_dir=base, base=base).corpus_dir(dag.family, chunker_hash)
    return base / cell_key(dag.run_id, dag.variant, cell) / node.name


def node_artifact(root: str | Path, dag: JobDAG, node: JobNode) -> Path:
    """Return the cell-0 artifact path of ``node``, the single-cell convenience form."""
    return cell_artifact(root, dag, node, 0)


def already_done(node: JobNode, root: str | Path, dag: JobDAG) -> bool:
    """Return True only if every cell of ``node`` has its ``_SUCCESS`` marker.

    A multi-seed ``PIPELINE`` node is done only when all of its cells are complete, so
    a resume with seed 0 done and later seeds missing does not skip the node.
    """
    return all(is_done(cell_artifact(root, dag, node, c)) for c in node_cells(node))


def run_local(
    dag: JobDAG,
    runner: Callable[[JobNode], None],
    *,
    root: str | Path,
) -> list[str]:
    """Run the DAG in dependency order in this process, skipping done nodes.

    Nodes are visited in ``afterok`` order (the tuple is already topologically
    ordered) and a node whose ``_SUCCESS`` marker exists is skipped, so a second
    ``run_local`` over a completed run is a no-op. ``runner`` is responsible for
    producing each node's artifact and marker. Returns the names of the nodes actually
    executed, empty on a full-resume no-op.

    Node-level skipping is only as good as the markers the runner writes, and the shipped
    runner (``cli._local_runner``) writes one only for the corpus node
    (``cli._mark_corpus_done``). The online nodes are re-entered on every ``--local``
    invocation and rely on their own inner idempotence instead: the corpus and pipeline
    queues are resumable shard by shard, and a finished ``(topic, seed)`` cell returns
    immediately. So a second ``--local`` run recomputes nothing of substance, but it does
    not short-circuit at the node level the way this function's contract allows.
    """
    executed: list[str] = []
    for node in dag.nodes:
        if already_done(node, root, dag):
            _log.info("run_local.skip", run_id=dag.run_id, node=node.name, reason="already_done")
            continue
        _log.info("run_local.exec", run_id=dag.run_id, node=node.name)
        runner(node)
        executed.append(node.name)
    return executed
