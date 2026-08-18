"""The online fleet as a ``saturate.StageAdapter``: one claimable shard per ``(topic, seed)``.

The exclusion the fleet needs, "two workers must never work the same topic", is
``orchestration.slurm.workqueue.claim``'s atomic ``mv`` (first mv wins), the same protocol the
translation fan runs on. This module only says what a shard is for the online half;
``preprocess/translate_omt.py`` is the same contract with a different unit of work.

The shard is ``(topic, seed)`` rather than a topic because ``pipeline.driver.drive`` takes
``(cfg, topic, variant, seed)`` and ``Layout`` keys a cell by all of them. A topic-only shard would
either serialize the seeds inside one worker or need a second claim level inside ``work``; one flat
unit keeps ``claim`` the only exclusion. A pair owns its topic end to end: the k RAG loops inside a
topic are asyncio coroutines against that pair's own instance, so a topic is indivisible here by
construction and this adapter never subdivides one.

Running the adapter at width greater than one requires the per-pair vLLM registry. A single shared
endpoint file would put every worker on one pair, which is not a fan, and the co-tenancy would also
break seeded reproducibility. ``bringup`` therefore refuses rather than silently sharing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ragtime.common.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

__all__ = ["TopicCellAdapter", "TopicShard"]

_log = get_logger("pipeline.topic_fleet")

#: The env var a launcher sets to give one worker its own vLLM endpoint. Injected per worker
#: rather than read from a shared endpoint file; its absence is what tells us the fan is unsafe.
ENDPOINT_ENV = "RAGTIME_VLLM_URL"

#: How often a worker re-stamps the lease it holds. Well under the registry's staleness bound,
#: because a topic runs for tens of minutes and without a beat the reaper would reclaim every
#: healthy lease and hand the pair to a second worker.
LEASE_HEARTBEAT_S = 30.0


@dataclass(frozen=True, slots=True)
class TopicShard:
    """One claimable ``(topic, seed)`` cell. ``name`` is the queue filename."""

    name: str
    payload: dict


class TopicCellAdapter:
    """``saturate.StageAdapter`` for the online half: claim a ``(topic, seed)``, then drive it.

    ``bringup`` builds the per-worker clients once. Note the asymmetry with the preprocess
    adapters: those load a resident model into the worker, whereas here the LLM is a remote vLLM
    reached over HTTP and retrieval is a filesystem-queue client. A worker's expensive resource is
    not in this process at all; it is the vLLM pair the worker was pointed at, and the point of
    the fan is that each worker gets a different one.
    """

    template = "pipeline_worker"

    def __init__(
        self, *, registry: str | Path, variant: str, seeds: Sequence[int], run_id: str = ""
    ) -> None:
        """Build the adapter for one run.

        The queue is keyed on ``run_id``, because the runs are the experiment and ``run_id`` is
        already what separates the cells (``cell_key`` yields ``<run_id>__<variant>__seed<N>``),
        so queue and artifact tree agree on what a run is. Keying on ``variant`` would be wrong
        for the mlir family, whose three arms all read ``original`` and move only
        ``retrieval.index``: they would share one queue, and the second arm launched would find it
        already drained and exit cleanly having done nothing. Falls back to ``variant`` when no
        ``run_id`` is given, so a synthesized adapter keeps working.

        Re-enumeration is safe: changing the key means a completed run's next launch walks a fresh
        queue, but ``driver.drive`` resume-noops on the existing ``_SUCCESS`` cells, because the
        artifact tree rather than the queue is the checkpoint.
        """
        self.registry = str(registry)
        self.variant = str(variant)
        self.run_id = str(run_id or "")
        self.seeds = tuple(int(s) for s in seeds)
        self.stage = f"pipeline_{self.run_id or self.variant}"

    # -- seed ------------------------------------------------------------ #
    def shards(self, cfg: Any) -> Iterator[TopicShard]:
        """Yield one shard per ``(topic, seed)``, in a deterministic order.

        Deterministic because the queue is seeded once and then claimed concurrently: the claim
        order is nondeterministic by design, but which units exist must not be, or a re-seed after
        a partial run would mint different work.

        The enumeration is seed-major rather than topic-major, and ``claim()`` takes shards in
        sorted order, so this is the order a fleet works them. Topic-major would give one worker
        all draws of topic 2000 before it ever saw topic 2001, and a submission needs coverage of
        every topic rather than depth on two. Seed-major makes every intermediate state a valid
        submission: all topics at seed 0, then all at seed 1, so stopping early costs precision
        rather than coverage. Shard names and payloads are unaffected, so a re-seed over an
        existing queue is a no-op for work already done.
        """
        from ragtime.common.topics import load_topics

        topics_path = self._topics_path(cfg)
        topics = list(load_topics(topics_path))
        for seed in self.seeds:
            for topic in topics:
                topic_id = str(getattr(topic, "topic_id", ""))
                yield TopicShard(
                    name=f"{topic_id}__seed{seed}",
                    payload={"topic_id": topic_id, "seed": seed, "variant": self.variant},
                )
        _log.info(
            "pipeline.topic_fleet.shards",
            topics=len(topics), seeds=len(self.seeds),
            shards=len(topics) * len(self.seeds), topics_path=str(topics_path),
        )

    @staticmethod
    def _topics_path(cfg: Any) -> Path:
        """Return the run's own topics file from the config, never a module constant.

        ``topics`` is a shared block, so two family members reading different request sets would
        make every delta an artifact of the input set rather than of translation.
        """
        from pathlib import Path as _P

        blocks = getattr(cfg, "blocks", None) or {}
        block = blocks.get("topics")
        rel = block if isinstance(block, str) else (block or {}).get("path")
        if not rel:
            raise ValueError(
                "config has no `topics:` and the fleet will not guess a topics file. The config "
                "is the run record, and a hardcoded default would let two family members read "
                "different requests while claiming to be comparable."
            )
        source = getattr(cfg, "source_path", None)
        base = _P(source).resolve().parent.parent if source else _P.cwd()
        return base / str(rel)

    # -- worker ---------------------------------------------------------- #
    def bringup(self, cfg: Any) -> Any:
        """Build the per-worker clients and retrieval context, on a pair of this worker's own.

        The endpoint comes from one of two places, in order:

        1. ``RAGTIME_VLLM_URL`` in the environment, an explicit per-worker assignment. A launcher
           that has already decided which pair this worker gets is not second-guessed.
        2. Otherwise an exclusive lease from ``serving.vllm_registry``, held for this worker's
           whole life and released when it exits.

        There is no third branch and no fallback to a shared endpoint file, which would put every
        worker on one pair and break seeded reproducibility through co-tenancy. If nothing is
        free this raises: a worker with no pair of its own must not start.
        """
        import os

        endpoint = os.environ.get(ENDPOINT_ENV, "").strip()
        lease = None
        if not endpoint:
            lease = self._lease_a_pair(cfg)
            endpoint = lease.env[ENDPOINT_ENV]
        from ragtime.preprocess.index import index_hash
        from ragtime.preprocess.packing import packing_hash
        from ragtime.preprocess.reconcile import reconcile_hash
        from ragtime.retrieval.context import service_context
        from ragtime.serving.registry import build_clients

        clients = build_clients(cfg)
        ctx = service_context(
            cfg, self._cell_layout(cfg, topic_id="_bringup", seed=self.seeds[0]),
            registry=self.registry,
            recon_hash=reconcile_hash(cfg), pack_hash=packing_hash(cfg), idx_hash=index_hash(cfg),
        )
        # No transport is named here or logged here: how retrieval is reached is `retrieval`'s
        # decision, and a stage that can name it becomes a second place that has to stay in
        # agreement. `retrieval.context.up` records it from the folder that owns it.
        _log.info(
            "pipeline.topic_fleet.bringup", endpoint=endpoint,
            leased=lease is not None, pair=(lease.pair if lease else None),
        )
        return {"cfg": cfg, "clients": clients, "ctx": ctx, "endpoint": endpoint, "lease": lease}

    def _lease_a_pair(self, cfg: Any) -> Any:
        """Take exclusive ownership of one vLLM pair for this worker's lifetime.

        The lease is per lifetime, not per shard. A worker drains many ``(topic, seed)`` cells and
        each one is indivisible on its pair, so churning the lease between cells would buy nothing
        and open a window where two workers alternate on one instance.

        Three things happen together because they are one contract: the atomic claim, which is the
        exclusion; a heartbeat on the leased descriptor, because a topic runs for tens of minutes
        against a much shorter staleness bound; and an ``atexit`` release, so a pair is not lost
        to the fleet when the worker finishes. A worker killed outright releases nothing, which is
        what the reaper is for.
        """
        import atexit

        from ragtime.common.layout import Layout
        from ragtime.orchestration.cli import artifact_root
        from ragtime.orchestration.slurm.workqueue import start_heartbeat
        from ragtime.serving.vllm_registry import claim_endpoint

        # The registry hangs off the artifact root, not off any one cell's run_dir: the pool of
        # pairs is shared by every cell of every family on this filesystem.
        base = Layout(artifact_root(cfg)).vllm_registry_dir()
        model = self._llm_model(cfg)
        gpu_model = self._lane(base)
        lease = claim_endpoint(base, model=model, gpu_model=gpu_model)
        if lease is None:
            raise RuntimeError(
                f"no free vLLM pair in {base} for model={model!r} gpu_model={gpu_model!r}. "
                f"A fleet worker must own its own pair: {ENDPOINT_ENV} is unset and the registry "
                "has nothing claimable in this lane. Either fewer workers than pairs are live, or "
                "the vLLM jobs have not published yet (they publish only once the engine answers)."
            )
        hb = start_heartbeat(lease.path, LEASE_HEARTBEAT_S)
        atexit.register(self._release, base, lease, hb)
        _log.info(
            "pipeline.topic_fleet.leased", name=lease.name, pair=lease.pair,
            job_id=lease.job_id, gpu_model=lease.gpu_model,
        )
        return lease

    @staticmethod
    def _llm_model(cfg: Any) -> str | None:
        """Return the checkpoint this run's config declares. A pair serving anything else is refused."""
        blocks = getattr(cfg, "blocks", None) or {}
        llm = blocks.get("llm") or {}
        model = llm.get("model") if isinstance(llm, dict) else None
        return str(model) if model else None

    @staticmethod
    def _lane(base: Any) -> str | None:
        """Return which GPU architecture this worker will accept.

        A100 runs the Marlin FP8 kernel and Blackwell runs Triton, so seeded generation does not
        carry across them and a family whose cells were generated on a mix would have a kernel
        term sitting inside the translation delta.

        ``RAGTIME_VLLM_GPU_MODEL`` declares the lane explicitly. With it unset the lane is read
        off the registry: if every live pair is the same architecture that is the lane, and if
        they differ this raises, because picking one silently is how half a family ends up on the
        wrong kernel.
        """
        import os

        from ragtime.serving.vllm_registry import live_endpoints

        declared = os.environ.get("RAGTIME_VLLM_GPU_MODEL", "").strip()
        if declared:
            return declared
        seen = {str(d.get("gpu_model") or "") for d in live_endpoints(base)}
        seen.discard("")
        if len(seen) > 1:
            raise RuntimeError(
                f"the vLLM registry at {base} offers more than one GPU architecture "
                f"({sorted(seen)}) and no lane was declared. Set RAGTIME_VLLM_GPU_MODEL to the "
                "one this run family uses: A100 runs Marlin FP8 and Blackwell runs Triton, so "
                "seeded generation is not reproducible across them and cells generated on a mix "
                "are not comparable."
            )
        return next(iter(seen), None)

    @staticmethod
    def _release(base: Any, lease: Any, hb: Any) -> None:
        """Stop beating, then give the pair back. Never raises: this runs at interpreter exit."""
        from ragtime.serving.vllm_registry import release_endpoint

        try:
            if hb is not None:
                hb.stop()
        except Exception as exc:  # noqa: BLE001 - exit path: log, never propagate
            _log.warning("pipeline.topic_fleet.heartbeat_stop_failed", error=str(exc))
        try:
            release_endpoint(base, lease)
        except Exception as exc:  # noqa: BLE001 - the reaper reclaims what we fail to release
            _log.warning("pipeline.topic_fleet.release_failed", name=lease.name, error=str(exc))

    @staticmethod
    def _cell_layout(cfg: Any, *, topic_id: str, seed: int) -> Any:
        from ragtime.pipeline.driver import topic_layout

        return topic_layout(cfg, str(getattr(cfg, "passage_lang", "original")), seed, topic_id)

    def work(self, ctx: Any, shard: Path) -> Path:
        """Drive one claimed cell and return its run dir. The pair owns it end to end."""
        import json

        from ragtime.common.topics import load_topics
        from ragtime.pipeline.driver import drive

        payload = json.loads(shard.read_text(encoding="utf-8"))
        topic_id, seed = str(payload["topic_id"]), int(payload["seed"])
        cfg = ctx["cfg"]
        topics = {str(getattr(t, "topic_id", "")): t for t in load_topics(self._topics_path(cfg))}
        topic = topics.get(topic_id)
        if topic is None:
            raise KeyError(f"claimed shard names topic {topic_id!r}, absent from the run's topics file")
        # A worker never adopts a shard's arm. Per-run queues already make a cross-arm claim
        # unreachable; this is the second lock, because driving another arm would put a
        # translation delta inside what is supposed to be measuring it, with nothing downstream
        # that would notice and no repair after the fact.
        shard_variant = str(payload.get("variant") or self.variant)
        if shard_variant != self.variant:
            raise ValueError(
                f"claimed shard {shard.name!r} is stamped variant {shard_variant!r} but this "
                f"worker runs {self.variant!r}. Check that the queue was seeded by the same "
                "config this worker was launched with."
            )
        # One counter bus per cell, handed to both consumers. `RetrievalContext` is built once per
        # worker, so its own bus outlives every cell the worker drains and is never flushed;
        # flushing it after each cell would instead re-write earlier cells' counters, since
        # `Statistics` has no reset. Minting the bus at the cell grain gives both sets of counters
        # one grain and one flush, which `drive` performs before the resume witness.
        from ragtime.common import Statistics

        bus = Statistics()
        ctx["ctx"].stats = bus  # slots=True but not frozen: assignment is the intended seam
        return drive(
            cfg, topic, self.variant, seed,
            ctx=ctx["ctx"], clients=ctx["clients"], stats=bus,
        )

    # -- validate / merge ------------------------------------------------ #
    def validate(self, path: Path) -> bool:
        """Return whether the cell carries the ``_SUCCESS`` witness ``drive`` writes last.

        Not "the directory is non-empty": a crashed cell leaves partial artifacts,
        and treating those as done would retire a topic that was never finished.
        """
        return (path / "_SUCCESS").exists()

    def merge(self, cfg: Any, shard_paths: Sequence[Path]) -> None:
        """No-op: select-and-serialize is its own DAG stage over the finished cell.

        Folding it in here would run the terminal dedup once per worker instead of once per cell,
        which is both wasteful and wrong, since the dedup is defined over the whole bank.
        """
        _log.info("pipeline.topic_fleet.merge_noop", cells=len(shard_paths))
