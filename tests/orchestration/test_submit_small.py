"""Real, non-``--local`` SLURM submission wiring.

The submit path must dispatch each DAG node through its thin ``slurm/templates/*.sbatch``
rather than the raw YAML, carry ``--export CONFIG=<path>``, fan the pipeline node as an
``sbatch --array`` rather than a single job, and wire ``afterok`` dependencies. It is
cluster-independent: ``sbatch`` is spied, never executed.

The corpus node is not one seed, worker and drive chain but an ordered chain of substages
(chunk, merge, the identity pass, translation, reconcile and the rest), each with its own
queue, template and config-derived array width, chained ``afterok`` on the previous
substage's drive. Every expectation below is derived from the config and the adapters rather
than from a hardcoded width or stage string: a width literal rots when a throughput knob is
tuned, and the post-chunk stage names carry hash suffixes, so a literal would point at a
queue nobody writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragtime.config import load
from ragtime.orchestration import cli
from ragtime.orchestration.plan import PIPELINE, build_plan

pytestmark = pytest.mark.small

_E2E = Path(__file__).resolve().parents[2] / "config" / "e2e-omt.yml"
_N_SUBSTAGES = len(cli._CORPUS_SUBSTAGES)
#: 3 role-separated jobs (seed / worker array / drive) per corpus substage.
_N_CORPUS_JOBS = 3 * _N_SUBSTAGES


@pytest.fixture
def cfg():
    return load(_E2E)


@pytest.fixture
def submitted_calls(sbatch_spy) -> list[list[str]]:
    """Drive a full non-local submit of a real e2e config; return the spied sbatch argvs."""
    assert cli.main(["--config", str(_E2E)]) == 0  # no --dry-run / --local -> submit path
    assert sbatch_spy.calls, "expected sbatch submissions"
    return sbatch_spy.calls


# --------------------------------------------------------------------------- #
# Helpers: every one reads the submission back the way SLURM would.
# --------------------------------------------------------------------------- #
def _export(argv: list[str]) -> str:
    return next(a for a in argv if a.startswith("--export="))


def _var(argv: list[str], name: str) -> str | None:
    export = _export(argv)
    if f"{name}=" not in export:
        return None
    return export.split(f"{name}=", 1)[1].split(",")[0]


def _dep_ids(argv: list[str], kind: str = "afterok") -> set[int]:
    for a in argv:
        if a.startswith(f"--dependency={kind}:"):
            return {int(x) for x in a.split(":", 1)[1].split(":")}
    return set()


def _corpus_calls(calls: list[list[str]]) -> list[list[str]]:
    """The corpus submissions, in submission order (STAGE=preprocess is the marker)."""
    return [argv for argv in calls if _var(argv, "STAGE") == "preprocess"]


# --------------------------------------------------------------------------- #
# The submission invariants, expressed over the widened corpus node.
# --------------------------------------------------------------------------- #
def test_every_node_dispatches_a_template_not_the_raw_yaml(submitted_calls) -> None:
    # 3 jobs per corpus substage, plus the 4 online tail nodes
    # (PIPELINE/CITATION_SCORING/SELECT_SERIALIZE/MONITORING).
    assert len(submitted_calls) == _N_CORPUS_JOBS + 4
    for argv in submitted_calls:
        script = argv[-1]
        assert script.endswith(".sbatch"), script
        assert not script.endswith(".yml")  # never submit the config file as the script


def test_export_carries_the_config_path(submitted_calls) -> None:
    for argv in submitted_calls:
        exports = [a for a in argv if a.startswith("--export=")]
        assert exports, argv
        assert any(f"CONFIG={_E2E}" in a for a in exports)


def test_pipeline_node_array_width_follows_the_config(submitted_calls) -> None:
    """The pipeline fan is `seeds x topic_shards` wide, and at width 1 there is no `--array`.

    `plan` emits `--array=0-{n-1}` only when `array_size > 1`, so a one-wide fan is an
    ordinary job. That is why this cannot simply count arrays: a literal width and a fixed
    array count would encode the seed count and report a missing array where the right answer
    is a plain job.
    """
    cfg = load(_E2E)
    width = build_plan(cfg).node(PIPELINE).array_size
    # The template does not identify the pipeline node: `pipeline`, `citation_scoring` and
    # `select_serialize` all submit `pipeline_worker.sbatch`. So the test asserts the property
    # over that group rather than naming one member: exactly one fan is array-shaped and it is
    # `width` wide, or none is when the fan is 1.
    pipe = [a for a in submitted_calls if str(a[-1]).endswith("pipeline_worker.sbatch")]
    assert pipe, "no pipeline_worker submission at all"
    flags = [f for argv in pipe for f in argv if f.startswith("--array=")]
    if width > 1:
        assert flags == [f"--array=0-{width - 1}"], (
            f"expected exactly one {width}-wide fan among the pipeline nodes, got {flags}"
        )
    else:
        assert flags == [], "a one-wide fan is a plain job, not a 1-element array"
    # Every corpus substage still fans as an array, whatever the pipeline width is.
    corpus_arrays = [
        argv for argv in submitted_calls
        if any(a.startswith("--array=") for a in argv) and _var(argv, "STAGE") == "preprocess"
    ]
    assert len(corpus_arrays) == _N_SUBSTAGES


def test_downstream_nodes_wire_afterok(submitted_calls) -> None:
    deps = [
        argv for argv in submitted_calls if any(a.startswith("--dependency=afterok:") for a in argv)
    ]
    # Per substage the worker waits on seed, while drive waits on `afterany` rather than
    # `afterok`, which its own test explains. Add the seed of every substage after the first,
    # chained on the previous drive, then the four chained online tail nodes
    # (pipeline/citation_scoring/select_serialize/monitoring).
    assert len(deps) == _N_SUBSTAGES + (_N_SUBSTAGES - 1) + 4


def test_pipeline_afterok_targets_the_last_substages_drive_job(sbatch_spy) -> None:
    """PIPELINE must depend on the last substage's DRIVE (the corpus is done only then)."""
    assert cli.main(["--config", str(_E2E)]) == 0
    calls, ids = sbatch_spy.calls, sbatch_spy.ids

    corpus = _corpus_calls(calls)
    last_drive_idx = max(
        i
        for i, argv in enumerate(calls)
        if _var(argv, "STAGE") == "preprocess"
        and _var(argv, "PREPROCESS_ROLE") == "drive"
        and _var(argv, "PREPROCESS_SUBSTAGE") == cli._CORPUS_SUBSTAGES[-1].name
    )
    assert corpus[-1] is calls[last_drive_idx]  # the chain's last submission
    pipe_idx = next(i for i, argv in enumerate(calls) if _var(argv, "STAGE") == "pipeline")
    assert _dep_ids(calls[pipe_idx]) == {ids[last_drive_idx]}  # not seed/worker, not chunk


# --------------------------------------------------------------------------- #
# The corpus substage chain, asserted structurally.
# --------------------------------------------------------------------------- #
def test_corpus_submits_the_substages_in_declared_order(submitted_calls) -> None:
    """chunk -> merge -> identity -> translate -> reconcile -> len_max -> packing ->
    vectorize -> assemble."""
    corpus = _corpus_calls(submitted_calls)
    assert len(corpus) == _N_CORPUS_JOBS
    got = [(_var(a, "PREPROCESS_SUBSTAGE"), _var(a, "PREPROCESS_ROLE")) for a in corpus]
    expected = [
        (sub.name, role) for sub in cli._CORPUS_SUBSTAGES for role in ("seed", "worker", "drive")
    ]
    assert got == expected
    # The order is the data-dependency order, pinned here because it is not recoverable from
    # the submission: merge reads chunk's spine, translate reads the map merge wrote,
    # reconcile reads both raw translation parts and so must follow the later of them,
    # len_max reads the final inventory reconcile published, packing reads that sidecar,
    # vectorize reads the passages packing published, and assemble reads the vector blocks
    # vectorize wrote, so it is necessarily last.
    #
    # len_max and packing sit between reconcile and vectorize because reconcile emits no
    # passages -- the `packing` block owns that -- so a chain ending at reconcile would build
    # an inventory and produce no passages at all. The vectorize and assemble halves are the
    # index build, and there is no fused `index` link beside them: a declared but unreachable
    # substage is a false pin, since its width keys would look load-bearing while nothing read
    # them. `IndexAdapter` itself is still used, since vectorize composes it and assemble
    # imports its three engine writers; it is simply not a link in the chain.
    assert [s.name for s in cli._CORPUS_SUBSTAGES] == [
        "chunk",
        "merge",
        "translate_omt_identity",
        "translate_omt",
        "reconcile",
        "len_max",
        "packing",
        "vectorize",
        "assemble",
    ]
    assert "index" not in {s.name for s in cli._CORPUS_SUBSTAGES}


def test_each_substage_routes_to_its_adapters_template_and_queue(submitted_calls, cfg) -> None:
    """Template and WQ_DIR come from the adapter, with its hash-suffixed stage."""
    corpus = _corpus_calls(submitted_calls)
    by_substage: dict[str, list[list[str]]] = {}
    for argv in corpus:
        by_substage.setdefault(_var(argv, "PREPROCESS_SUBSTAGE"), []).append(argv)

    for sub in cli._CORPUS_SUBSTAGES:
        adapter = sub.adapter(cfg)
        jobs = by_substage[sub.name]
        assert len(jobs) == 3
        for argv in jobs:
            assert argv[-1].endswith(adapter.template), (sub.name, argv[-1])
            wq = _var(argv, "WQ_DIR")
            # ...one queue per substage, ending in the adapter's own stage name.
            assert wq is not None and Path(wq).name == adapter.stage
    # the queues are genuinely distinct (a shared WQ_DIR would cross-claim shards)
    queues = {_var(a, "WQ_DIR") for a in corpus}
    assert len(queues) == _N_SUBSTAGES


def test_only_the_gpu_arms_take_the_gpu_template(cfg) -> None:
    """Only the model-loading arms take a card; the seven others need none."""
    got = {sub.name: sub.adapter(cfg).template for sub in cli._CORPUS_SUBSTAGES}
    assert got == {
        "chunk": "workqueue_worker_cpu.sbatch",
        "merge": "workqueue_worker_cpu.sbatch",
        "translate_omt_identity": "workqueue_worker_cpu.sbatch",
        "translate_omt": "workqueue_worker.sbatch",
        # reconciliation loads no model, not even a tokenizer.
        "reconcile": "workqueue_worker_cpu.sbatch",
        # the length sidecar is a tokenizer, and packing branches on counts alone, so
        # neither takes a card.
        "len_max": "workqueue_worker_cpu.sbatch",
        "packing": "workqueue_worker_cpu.sbatch",
        # The GPU half of the index build: the dense/sparse/late-interaction encoders all
        # run a forward pass.
        "vectorize": "workqueue_worker.sbatch",
        # The CPU half, and the reason the third template exists: it needs the SIF (faiss /
        # pyseismic / pylate live only in the container) and not a card (Seismic 268.1 s per
        # part on 32 cores vs PLAID 76.8 s and FAISS ~2 s, all CPU).
        "assemble": "workqueue_worker_cpu_sif.sbatch",
    }


def test_post_chunk_stage_names_carry_their_semantic_hashes(cfg) -> None:
    """A merge/translation-config edit must resolve to an ENTIRELY fresh queue subtree.

    The translate arms carry both hashes: unit boundaries are as much an input to the MT
    call as the beam size is, so a re-run under a different merge map must not find a
    stale ``done/`` from the previous grouping.
    """
    from ragtime.config import all_hashes

    h = all_hashes(cfg)
    t12, m12 = h["translation"][:12], h["merge"][:12]
    stages = {sub.name: sub.adapter(cfg).stage for sub in cli._CORPUS_SUBSTAGES}
    assert stages["chunk"] == "chunk"  # the chunk queue path carries no hash
    assert stages["merge"] == f"merge_{m12}"
    assert stages["translate_omt_identity"] == f"translate_omt_identity_{t12}_{m12}"
    assert stages["translate_omt"] == f"translate_omt_{t12}_{m12}"
    # Reconciliation's stage carries the composite hash: an edit to any of its four inputs
    # must resolve to an entirely fresh queue.
    from ragtime.preprocess.reconcile import reconcile_hash

    assert stages["reconcile"] == f"reconcile_{reconcile_hash(cfg)[:12]}"


# --------------------------------------------------------------------------- #
# The index build is launchable in its two halves. Its widths were declared in every config's
# `execution.*_shards` while the chain still ended at `packing`, so `run --config` could not
# start an index build at all.
# --------------------------------------------------------------------------- #
def test_the_vectorize_substage_is_second_to_last_and_lands_on_the_gpu_template(
    submitted_calls, cfg
) -> None:
    """The GPU half: submitted, just before assemble, GPU template, own width knobs."""
    from ragtime.config import config_hash
    from ragtime.preprocess.index import LEGS, index_build_options, index_hash, leg_encode_hash
    from ragtime.preprocess.reconcile import reconcile_hash
    from ragtime.preprocess.vectorize import VectorizeAdapter

    sub = next(s for s in cli._CORPUS_SUBSTAGES if s.name == "vectorize")
    assert cli._CORPUS_SUBSTAGES[-2] is sub  # it reads what packing published
    adapter = sub.adapter(cfg)
    assert isinstance(adapter, VectorizeAdapter)
    # Three encoders and three forward passes, read off the adapter, not a cli literal.
    assert adapter.template == "workqueue_worker.sbatch"
    # The queue is keyed by the corpus and the per-leg encode hashes, not by
    # `index_hash`: an assemble-only recipe change must re-encode and re-fan nothing.
    opts = index_build_options(cfg)
    encode12 = config_hash({leg: leg_encode_hash(opts, leg) for leg in LEGS})[:12]
    assert adapter.stage == f"vectorize_{encode12}_{reconcile_hash(cfg)[:12]}"
    assert index_hash(cfg)[:12] not in adapter.stage
    # widths come from the knobs the configs already declare, never a code-side literal
    assert (sub.shards_key, sub.oversub_key) == (
        "vectorize_shards",
        "vectorize_oversubscription",
    )
    assert cli._substage_array_size(cfg, sub) == -(
        -int(cfg.blocks["execution"]["vectorize_shards"])
        // int(cfg.blocks["execution"]["vectorize_oversubscription"])
    )

    jobs = [
        a for a in _corpus_calls(submitted_calls) if _var(a, "PREPROCESS_SUBSTAGE") == "vectorize"
    ]
    assert [_var(a, "PREPROCESS_ROLE") for a in jobs] == ["seed", "worker", "drive"]
    for argv in jobs:
        assert argv[-1].endswith("workqueue_worker.sbatch"), argv[-1]
        assert Path(_var(argv, "WQ_DIR")).name == adapter.stage


def test_the_assemble_substage_is_last_and_lands_on_the_cpu_sif_template(
    submitted_calls, cfg
) -> None:
    """The CPU half: submitted, last in the chain (so `_mark_corpus_done` waits for the
    engines, not just the vectors), on the Apptainer-wrapping CPU template."""
    from ragtime.preprocess.assemble import AssembleAdapter
    from ragtime.preprocess.index import index_hash
    from ragtime.preprocess.reconcile import reconcile_hash

    sub = next(s for s in cli._CORPUS_SUBSTAGES if s.name == "assemble")
    assert cli._CORPUS_SUBSTAGES[-1] is sub  # it reads what vectorize published
    adapter = sub.adapter(cfg)
    assert isinstance(adapter, AssembleAdapter)
    # It needs the SIF (faiss/pyseismic/pylate are container-only) and not a card.
    assert adapter.template == "workqueue_worker_cpu_sif.sbatch"
    # both hashes in the queue name: a corpus edit OR a recipe edit gets a fresh subtree.
    assert adapter.stage == f"assemble_{index_hash(cfg)[:12]}_{reconcile_hash(cfg)[:12]}"
    assert (sub.shards_key, sub.oversub_key) == ("assemble_shards", "assemble_oversubscription")
    assert cli._substage_array_size(cfg, sub) == -(
        -int(cfg.blocks["execution"]["assemble_shards"])
        // int(cfg.blocks["execution"]["assemble_oversubscription"])
    )

    jobs = [
        a for a in _corpus_calls(submitted_calls) if _var(a, "PREPROCESS_SUBSTAGE") == "assemble"
    ]
    assert [_var(a, "PREPROCESS_ROLE") for a in jobs] == ["seed", "worker", "drive"]
    for argv in jobs:
        assert argv[-1].endswith("workqueue_worker_cpu_sif.sbatch"), argv[-1]
        assert Path(_var(argv, "WQ_DIR")).name == adapter.stage


def test_each_index_half_dispatches_its_own_adapter_worker_side(
    monkeypatch, tmp_path, cfg
) -> None:
    """``PREPROCESS_SUBSTAGE=vectorize|assemble`` must select that half's own adapter.

    And `index` is not a substage: a launcher (or a hand-run worker) asking for it gets the
    named error rather than quietly running the whole chain.
    """
    from ragtime.orchestration import saturate
    from ragtime.preprocess.assemble import AssembleAdapter
    from ragtime.preprocess.vectorize import VectorizeAdapter

    seen: list[object] = []
    monkeypatch.setattr(cli, "_LOCAL_ROOT", str(tmp_path))
    monkeypatch.setattr(saturate, "seed", lambda c, a, q: seen.append(a) or 0)
    monkeypatch.setattr(saturate, "run_worker", lambda c, a, q: 0)
    monkeypatch.setattr(saturate, "drive", lambda c, a, q: None)
    monkeypatch.setattr(cli, "_mark_corpus_done", lambda c: None)
    monkeypatch.setenv("PREPROCESS_ROLE", "seed")

    for substage, cls in (("vectorize", VectorizeAdapter), ("assemble", AssembleAdapter)):
        seen.clear()
        monkeypatch.setenv("PREPROCESS_SUBSTAGE", substage)
        assert cli._run_preprocess(cfg) == 0
        assert len(seen) == 1 and isinstance(seen[0], cls)

    monkeypatch.setenv("PREPROCESS_SUBSTAGE", "index")
    with pytest.raises(ValueError, match="index"):
        cli._run_preprocess(cfg)


def test_substage_array_widths_are_derived_from_config(submitted_calls, cfg) -> None:
    """Widths come from ``execution.<substage>_shards`` and ``_oversubscription``, never from
    a literal, which rots the moment a throughput knob is tuned. Chunk's width must stay
    exactly ``plan.corpus_workers``."""
    from ragtime.orchestration import plan

    arrays = {
        _var(argv, "PREPROCESS_SUBSTAGE"): next(a for a in argv if a.startswith("--array="))
        for argv in _corpus_calls(submitted_calls)
        if any(a.startswith("--array=") for a in argv)
    }
    assert set(arrays) == {sub.name for sub in cli._CORPUS_SUBSTAGES}
    for sub in cli._CORPUS_SUBSTAGES:
        width = cli._substage_array_size(cfg, sub)
        ex = cfg.blocks["execution"]
        assert width == -(-int(ex[sub.shards_key]) // int(ex[sub.oversub_key]))  # ceil()
        assert arrays[sub.name] == f"--array=0-{width - 1}"
    assert cli._substage_array_size(cfg, cli._CORPUS_SUBSTAGES[0]) == plan.corpus_workers(cfg)


def test_a_missing_shards_key_fails_loudly_rather_than_defaulting(cfg) -> None:
    """The config is the complete record: a width is never invented in code."""
    from ragtime import config as _config

    ghost = cli._Substage(
        "ghost",
        lambda c: None,
        "ghost_shards",
        "ghost_oversubscription",
        max_age_key="ghost_max_age_s",
    )
    with pytest.raises(_config.ConfigError, match="ghost_shards"):
        cli._substage_array_size(cfg, ghost)


def test_only_the_worker_role_fans_as_an_array(submitted_calls) -> None:
    for argv in _corpus_calls(submitted_calls):
        is_array = any(a.startswith("--array=") for a in argv)
        assert is_array == (_var(argv, "PREPROCESS_ROLE") == "worker"), argv


def test_each_substage_chains_afterok_on_the_previous_substages_drive(sbatch_spy) -> None:
    """seed(afterok prev drive) -> worker(afterok seed) -> drive(afterANY worker)."""
    assert cli.main(["--config", str(_E2E)]) == 0
    calls, ids = sbatch_spy.calls, sbatch_spy.ids
    idx = [i for i, argv in enumerate(calls) if _var(argv, "STAGE") == "preprocess"]

    prev_drive: int | None = None
    for n, sub in enumerate(cli._CORPUS_SUBSTAGES):
        seed_i, worker_i, drive_i = idx[3 * n : 3 * n + 3]
        assert _var(calls[seed_i], "PREPROCESS_SUBSTAGE") == sub.name
        if prev_drive is None:
            assert _dep_ids(calls[seed_i]) == set()  # the CORPUS node has no parent
        else:
            assert _dep_ids(calls[seed_i]) == {prev_drive}  # chained on the previous DRIVE
        assert _dep_ids(calls[worker_i]) == {ids[seed_i]}
        assert _dep_ids(calls[drive_i], "afterany") == {ids[worker_i]}
        prev_drive = ids[drive_i]


def test_drive_waits_on_afterany_so_one_bad_array_task_cannot_strand_the_merge(
    sbatch_spy,
) -> None:
    """``afterok`` on an N-task worker array is satisfied only if every task exits 0, so one
    preempted or OOM-killed task out of many leaves the merge pending forever with
    ``DependencyNeverSatisfied``, the queue drained and every output already on disk.
    ``drive``'s real guard is at the artifact level, since it stops on a non-empty
    ``failed/``, so it waits on ``afterany`` and checks the queue itself."""
    assert cli.main(["--config", str(_E2E)]) == 0
    drives = [
        argv
        for argv in sbatch_spy.calls
        if _var(argv, "STAGE") == "preprocess" and _var(argv, "PREPROCESS_ROLE") == "drive"
    ]
    assert len(drives) == _N_SUBSTAGES
    for argv in drives:
        assert _dep_ids(argv, "afterany"), argv
        assert not _dep_ids(argv), f"drive must NOT be afterok-gated on the array: {argv}"
    # the SEED of the next substage still is afterok: it genuinely needs the artifact.
    seeds = [
        argv
        for argv in sbatch_spy.calls
        if _var(argv, "STAGE") == "preprocess" and _var(argv, "PREPROCESS_ROLE") == "seed"
    ]
    assert all(not _dep_ids(a, "afterany") for a in seeds)


def test_no_gpu_flags_are_added_by_the_launcher(submitted_calls) -> None:
    """Templates own their own ``#SBATCH --gres/--constraint`` (the GPU one included)."""
    for argv in submitted_calls:
        assert not any(a.startswith(("--gres", "--constraint")) for a in argv)


# --------------------------------------------------------------------------- #
# The worker-side half of the contract: PREPROCESS_SUBSTAGE selects the link.
# --------------------------------------------------------------------------- #
def test_corpus_success_is_marked_only_after_the_last_substages_drive(
    monkeypatch, tmp_path, cfg
) -> None:
    """``_mark_corpus_done`` (what ``already_done(CORPUS)`` reads) fires once, at the end."""
    from ragtime.orchestration import saturate

    marked: list[str] = []
    monkeypatch.setattr(cli, "_mark_corpus_done", lambda c: marked.append("done"))
    monkeypatch.setattr(cli, "_LOCAL_ROOT", str(tmp_path))
    monkeypatch.setattr(saturate, "seed", lambda *a, **k: 0)
    monkeypatch.setattr(saturate, "run_worker", lambda *a, **k: 0)
    monkeypatch.setattr(saturate, "drive", lambda *a, **k: None)

    for sub in cli._CORPUS_SUBSTAGES[:-1]:
        monkeypatch.setenv("PREPROCESS_SUBSTAGE", sub.name)
        monkeypatch.setenv("PREPROCESS_ROLE", "drive")
        assert cli._run_preprocess(cfg) == 0
        assert marked == []  # an intermediate drive must not mark the corpus done

    monkeypatch.setenv("PREPROCESS_SUBSTAGE", cli._CORPUS_SUBSTAGES[-1].name)
    assert cli._run_preprocess(cfg) == 0
    assert marked == ["done"]


def test_unknown_substage_env_fails_loudly(monkeypatch, cfg) -> None:
    monkeypatch.setenv("PREPROCESS_SUBSTAGE", "translate_omt_typo")
    with pytest.raises(ValueError, match="translate_omt_typo"):
        cli._run_preprocess(cfg)
