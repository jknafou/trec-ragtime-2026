"""Lint of the Apptainer recipe ``container/ragtime-gpu.def``, plus the sbatch templates.

No image is built here. The recipe must be well-formed and must encode the documented
contract: the devel CUDA base, because flashinfer compiles kernels at runtime, the exact
``uv sync`` flag set, and the ``uv run --frozen run`` entrypoint. These skip when the recipe
is absent, since building the image is a site decision.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.small

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEF = _REPO_ROOT / "container" / "ragtime-gpu.def"
_NO_DEF = "the Apptainer recipe is not distributed with this repo"
_TEMPLATES = _REPO_ROOT / "src" / "ragtime" / "orchestration" / "slurm" / "templates"


@pytest.fixture
def def_text() -> str:
    if not _DEF.is_file():
        pytest.skip(_NO_DEF)
    return _DEF.read_text(encoding="utf-8")


def _environment_block(def_text: str) -> str:
    m = re.search(r"^%environment\b(.*?)(?=^%\w+\b|\Z)", def_text, re.MULTILINE | re.DOTALL)
    assert m is not None
    return m.group(1)


@pytest.mark.skipif(not _DEF.is_file(), reason=_NO_DEF)
def test_def_file_exists() -> None:
    assert _DEF.is_file()


def test_has_bootstrap_and_devel_base(def_text: str) -> None:
    assert re.search(r"^Bootstrap:\s*docker\s*$", def_text, re.MULTILINE)
    from_line = re.search(r"^From:\s*(\S+)\s*$", def_text, re.MULTILINE)
    assert from_line is not None
    image = from_line.group(1)
    assert image == "nvidia/cuda:12.9.2-devel-ubuntu22.04"
    assert "-devel-" in image  # not the runtime image: flashinfer's JIT needs the toolchain


def test_required_sections_present(def_text: str) -> None:
    for section in ("%post", "%environment", "%runscript"):
        assert re.search(rf"^{section}\b", def_text, re.MULTILINE), section


def test_post_runs_the_exact_uv_sync_flag_set(def_text: str) -> None:
    # `--extra index` (faiss-cpu, pyseismic-lsr, pylate and fast-plaid) is additive to
    # `heavy`, with the same torch and sentence-transformers pins, so the image holds
    # one environment covering both serving and index building.
    assert "uv sync --frozen --no-dev --extra heavy --extra index --no-install-project" in def_text


def test_runscript_invokes_uv_run_frozen_run(def_text: str) -> None:
    assert "uv run --frozen run" in def_text


# --------------------------------------------------------------------------- #
# The recipe fixes, and the sbatch templates that close over WQ_DIR and RAGTIME_SIF.
# --------------------------------------------------------------------------- #
def test_environment_pins_uv_cache_dir_and_keeps_uv_no_sync(def_text: str) -> None:
    """UV_CACHE_DIR must resolve to a writable node-local path, through RAGTIME_SCRATCH.

    Writing ``${SCRATCH:-/tmp}/uv-cache`` is not enough: where ``$SCRATCH`` is unset, every
    such expansion collapses to ``/tmp`` and the intent is never actually expressed. The
    recipe resolves node-local scratch once into ``RAGTIME_SCRATCH``
    (``${SCRATCH:-${TMPDIR:-/tmp}}``) and roots the write paths in it, so this pins
    RAGTIME_SCRATCH, and separately pins that RAGTIME_SCRATCH is itself derived from SCRATCH
    or TMPDIR rather than hardcoded.
    """
    env = _environment_block(def_text)
    assert re.search(r'export RAGTIME_SCRATCH="?\$\{SCRATCH:-\$\{TMPDIR:-/tmp\}\}', env), (
        "RAGTIME_SCRATCH must be derived from $SCRATCH, falling back to $TMPDIR then /tmp"
    )
    assert re.search(r'export UV_CACHE_DIR="?\$\{RAGTIME_SCRATCH\}', env), (
        "UV_CACHE_DIR must point at a RAGTIME_SCRATCH-rooted (writable, non-read-only) path"
    )
    assert "UV_NO_SYNC=1" in env  # load-bearing, so guard against its removal


def test_environment_keeps_hf_cache_warm_and_offline(def_text: str) -> None:
    """Two further runtime settings, pinned so a later edit cannot drop them.

    Compute nodes have no network, so the first hub call inside the image must never fire,
    which is what ``HF_HUB_OFFLINE=1`` guarantees. That is only safe if HF_HOME points at the
    warm ``/home`` cache, which apptainer binds, since ``${SCRATCH:-/tmp}/hf`` resolves to an
    empty node-local directory and turns offline mode into a hard failure with the pre-cached
    weights invisible. The JIT and compile caches stay node-local, because those are write
    paths in a read-only image.
    """
    env = _environment_block(def_text)
    assert "export HF_HUB_OFFLINE=1" in env
    assert re.search(r'export HF_HOME="\$\{HF_HOME:-\$HOME/\.cache/huggingface\}"', env), (
        "HF_HOME must default to the warm $HOME cache, never node-local scratch"
    )
    for jit in ("FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR"):
        assert re.search(rf'export {jit}="?\$\{{RAGTIME_SCRATCH\}}', env), jit


def test_apptainer_exec_lines_use_cleanenv() -> None:
    for tmpl in _TEMPLATES.glob("*.sbatch"):
        for line in tmpl.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("apptainer exec"):  # command line, not a comment
                assert "--cleanenv" in line, f"{tmpl.name}: apptainer exec missing --cleanenv"


def test_cpu_workqueue_template_has_no_gres_and_no_apptainer() -> None:
    cpu = _TEMPLATES / "workqueue_worker_cpu.sbatch"
    assert cpu.is_file()
    text = cpu.read_text(encoding="utf-8")
    directives = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
    assert not any("--gres" in ln for ln in directives)  # CPU stage requests no GPU
    commands = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("apptainer exec" in ln for ln in commands)  # runs uv directly, no SIF
    assert "uv run --frozen --extra chunk run" in text


def test_corpus_substages_route_to_their_adapters_own_template() -> None:
    """Each corpus substage's template is read off its adapter, never a cli-side literal.

    Three templates, three resource classes. Chunk, merge, the English identity pass,
    reconcile, len_max and packing are host CPU, with no card and no container. The NLLB arm
    and the index build's vectorize half take the GPU template. The assemble half takes the
    container-wrapping CPU one, because its engines exist only in the image and none of them
    needs a card. A literal table here would rot the moment an adapter changed template, and
    every one of these names must also exist on disk, which the loop below checks.
    """
    from ragtime.config import load
    from ragtime.orchestration import cli
    from ragtime.orchestration.plan import CORPUS

    assert CORPUS not in cli._TEMPLATE_OF  # the corpus node is a chain, not one template
    cfg = load(_REPO_ROOT / "config" / "e2e-omt.yml")
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
        # The GPU half of the index build: dense/sparse/late-interaction each run a forward
        # pass.
        "vectorize": "workqueue_worker.sbatch",
        # The CPU half needs the container image, not a card.
        "assemble": "workqueue_worker_cpu_sif.sbatch",
    }
    for name in set(got.values()):
        assert (_TEMPLATES / name).is_file(), name


def test_cpu_sif_workqueue_template_wraps_apptainer_without_asking_for_a_card() -> None:
    """The third template: the container image, no ``--gres``, many cores, a peak ``--mem``.

    Every clause is load-bearing. The image, because faiss, pyseismic and pylate with
    fast-plaid exist only inside it and the host CPU template cannot import them. No card,
    because assembly's two poles are the Seismic and PLAID document adds, both of which are
    CPU work, so a `--gres` here would idle a scarce card per part and put assembly back on
    the encode critical path. `--mem` is sized from the worker's peak
    (`PLAID_ADD_PEAK_MULTIPLIER` x the buffer, so ~40 GiB at fp16 and ~53 GiB at fp32), never
    from the buffer alone, which would under-provision by 2-3x.
    """
    tmpl = _TEMPLATES / "workqueue_worker_cpu_sif.sbatch"
    assert tmpl.is_file()
    text = tmpl.read_text(encoding="utf-8")
    directives = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
    commands = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]

    assert not any("--gres" in ln for ln in directives)  # CPU stage requests no GPU
    partition = next(ln for ln in directives if "--partition" in ln)
    assert "shared-cpu" in partition and "public-cpu" in partition
    cpus = int(next(ln for ln in directives if "--cpus-per-task" in ln).split("=")[-1])
    assert cpus >= 32  # Seismic sustained 19.3x parallelism; the pole wants cores
    mem = next(ln for ln in directives if "--mem" in ln).split("=")[-1]
    assert mem.endswith("G") and int(mem[:-1]) >= 64  # above the fp32 worst-case peak

    exec_line = next(ln for ln in commands if "apptainer exec" in ln)
    assert "--cleanenv" in exec_line
    assert "--nv" not in exec_line  # there is no card on this node to expose
    assert "${RAGTIME_SIF:?" in text  # the image path is required, never defaulted
    assert "uv run --frozen python -c" in text


def test_cpu_sif_template_forwards_the_worker_provenance_envs_through_cleanenv() -> None:
    """``--cleanenv`` drops the ``--export``ed environment, so everything the worker reads off
    the environment must be forwarded explicitly.

    Two groups. The two that select the unit of work, role and substage: without them every
    array task would seed, work and drive every substage. And the three
    ``saturate.worker_provenance`` reads: without them the manifest's per-shard ``worker``
    field is empty and an assembled part cannot say where it was built. ``RAGTIME_GPU_MODEL``
    is set to the literal ``none`` rather than left unset, because ``worker_provenance`` drops
    empty values, so an unset variable would delete the ``gpu`` key and make "ran on a
    GPU-less node" indistinguishable from "the model was not recorded".
    """
    text = (_TEMPLATES / "workqueue_worker_cpu_sif.sbatch").read_text(encoding="utf-8")
    from ragtime.orchestration.saturate import _PROVENANCE_ENV

    for env in ("PREPROCESS_ROLE=", "PREPROCESS_SUBSTAGE=", "PYTHONUNBUFFERED=1"):
        assert f'--env "{env}' in text, env
    for name in _PROVENANCE_ENV:  # the exact set the worker records, read off the source
        assert f'--env "{name}=' in text, name
    commands = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("nvidia-smi" in ln for ln in commands)  # it does not exist on a CPU node
    assert 'RAGTIME_GPU_MODEL="none"' in text
    # deterministic per-task start stagger, derived from the task id rather than $RANDOM,
    # so a resume reproduces the same schedule.
    assert "SLURM_ARRAY_TASK_ID % 60" in text
    assert not any("$RANDOM" in ln for ln in commands)


def test_gpu_workqueue_template_races_the_calibrated_cards_and_excludes_the_3090() -> None:
    """The or-set is the calibrated set, and the 3090 is excluded by measurement.

    Its ceiling of 9 216 max_batch_tokens is below the fairness-hashed 16 384 semantic bucket
    budget, so CT2 would re-split every bucket and the same shard would stop composing
    identically across cards. That is a silent fairness break, not a slow job.
    """
    text = (_TEMPLATES / "workqueue_worker.sbatch").read_text(encoding="utf-8")
    constraint = next(ln for ln in text.splitlines() if ln.startswith("#SBATCH --constraint"))
    for card in (
        "nvidia_rtx_pro_6000_blackwell",
        "nvidia_h200_nvl",
        "nvidia_h100_nvl",
        "nvidia_a100_80gb_pcie",
        "nvidia_geforce_rtx_5090",
    ):
        assert card in constraint, card
    assert "3090" not in constraint
    assert "#SBATCH --gres=gpu:1" in text


def test_gpu_template_forwards_the_role_and_substage_through_cleanenv() -> None:
    """``--cleanenv`` drops the ``--export``ed environment, so the two variables that select
    the unit of work must be forwarded explicitly. Without them the worker defaults to role
    ``all`` and substage ``None``, and every array task would seed, work and drive every
    substage. ``PYTHONUNBUFFERED`` keeps a preempted worker's last log block."""
    text = (_TEMPLATES / "workqueue_worker.sbatch").read_text(encoding="utf-8")
    for env in ("PREPROCESS_ROLE=", "PREPROCESS_SUBSTAGE=", "PYTHONUNBUFFERED=1"):
        assert f'--env "{env}' in text, env
    # deterministic per-task start stagger, derived from the task id rather than $RANDOM,
    # so a resume reproduces the same schedule.
    assert "SLURM_ARRAY_TASK_ID % 60" in text
    assert "$RANDOM" not in text


def test_submit_exports_layout_derived_wq_dir_and_sif(sbatch_spy, monkeypatch) -> None:
    from ragtime.orchestration import cli

    monkeypatch.setattr(cli.Layout, "wq_dir", lambda self, fam, ch, stage: Path("/sentinel/wq"))
    monkeypatch.setattr(cli.Layout, "sif_path", lambda self: Path("/sentinel/sif"))
    cfg = _REPO_ROOT / "config" / "e2e-omt.yml"
    assert cli.main(["--config", str(cfg)]) == 0

    corpus_exports = [
        a
        for argv in sbatch_spy.calls
        for a in argv
        if a.startswith("--export=") and "STAGE=preprocess" in a
    ]
    # the CORPUS node un-fuses into 3 role-separated jobs PER SUBSTAGE; every one carries
    # the Layout-derived WQ_DIR + RAGTIME_SIF (read from Layout, not a hardcoded 2nd path).
    from ragtime.orchestration import cli as _cli

    assert len(corpus_exports) == 3 * len(_cli._CORPUS_SUBSTAGES)
    for export in corpus_exports:
        assert "WQ_DIR=/sentinel/wq" in export
        assert "RAGTIME_SIF=/sentinel/sif" in export
