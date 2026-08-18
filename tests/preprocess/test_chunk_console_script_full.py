"""The chunk work-queue worker, driven through the real ``run`` console script.

A real ``run --config <fixture> --stage preprocess`` subprocess, so ``__main__`` is the
actual uv-generated, extension-less console script. That is the condition a test running
the pool as an imported module cannot reproduce, because under pytest ``__main__`` is a
real module and a start method that could not survive a console script still passed.
Invoking the installed script directly, rather than through ``uv run``, keeps this off the
network and out of uv's cache and lock path.

Everything else is real too: ``ChunkAdapter`` bring-up, a two-process ``_ChunkPool``, the
SaT segmenter and the pinned ``bge-m3`` tokenizer, and the ``saturate.run_worker`` claim
loop over a seeded queue on disk. It therefore needs the console script installed, the
``chunk`` extra, and a warm model cache. The same start-method property is covered without
models in ``test_chunk_pool_entrypoint_small.py``.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import signal
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from ragtime.common import Layout
from ragtime.config import all_hashes, load
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family

pytestmark = pytest.mark.full

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "config" / "e2e-original.yml"
_TIMEOUT_S = 600
_N_DOCS = 8

_TEXT = (
    "First sentence here. Second sentence follows.\n\n"
    "A new paragraph appears now. It has two sentences."
)


def _console_script() -> Path:
    """The installed ``run`` entrypoint sitting next to the running interpreter."""
    script = Path(sys.executable).with_name("run")
    if not script.exists():
        pytest.skip(f"console script not installed at {script} (run `uv sync`)")
    return script


# A model or hub failure inside the subprocess means a cold model cache or no network, not
# a defect, so it skips. The markers are matched against the worker's `saturate.work.failed`
# record. The models are deliberately not probed by loading them in the test process: that
# would leave an onnxruntime session and its native thread pool in an interpreter that later
# tests fork from, which is the fork-after-threads hazard this file exists to catch.
_ENV_GAP_MARKERS = (
    "OfflineModeIsEnabled",
    "LocalEntryNotFound",
    "EntryNotFound",
    "RepositoryNotFound",
    "HFValidationError",
    "ConnectionError",
    "Couldn't reach",
    "Max retries exceeded",
)


def _skip_if_env_gap(log: str) -> None:
    for marker in _ENV_GAP_MARKERS:
        if marker in log:
            pytest.skip(f"SaT/bge-m3 unavailable to this box ({marker}); full gate needs a "
                        f"warm HF cache")


def _stage_and_seed(tmp_path: Path, cfg) -> tuple[Path, int]:
    """A fixture raw corpus and a seeded chunk queue under the subprocess's working root."""
    from ragtime.preprocess.chunk import ChunkAdapter

    base = str(tmp_path / "runs")  # the local root the CLI resolves against its cwd
    fam, ch = run_family(cfg), all_hashes(cfg)["chunker"]
    raw = Layout(run_dir=base, base=base).corpus_raw_dir(fam, ch)
    raw.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw / "eng-docs.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(_N_DOCS):
            f.write(
                json.dumps(
                    {"id": f"eng-docs/{i:07d}", "text": _TEXT, "url": "u", "date": "2026-01-01"}
                )
                + "\n"
            )
    adapter = ChunkAdapter(base=base)
    wq = saturate.queue_for(cfg, adapter, base=base)
    n = saturate.seed(cfg, adapter, wq)
    assert n == _N_DOCS  # far more shards than documents, so one document per shard
    return wq.base, n


def _run_worker_subprocess(script: Path, cwd: Path, cfg_path: Path) -> subprocess.CompletedProcess:
    """``run --config … --stage preprocess`` in the worker role, in its own session."""
    env = {
        **os.environ,
        # State the artefact root rather than inherit it. `cli.artifact_root` falls back to
        # `$RAGTIME_ARTIFACT_ROOT`, which `**os.environ` would otherwise pass through and
        # point the worker at whatever tree the caller was using.
        "RAGTIME_ARTIFACT_ROOT": str(cwd / "runs"),
        "PREPROCESS_ROLE": "worker",  # what the job template exports for an array task
        # The template exports the substage as well, and leaving it out is not harmless:
        # with an empty filter the worker runs every corpus substage, so a chunk test would
        # drag in translate bring-up and load a translation tokenizer it does not need.
        "PREPROCESS_SUBSTAGE": "chunk",
        "SLURM_CPUS_PER_TASK": "2",  # two chunk workers, so the real pool path is taken
        "HF_HUB_OFFLINE": "1",  # a compute node has a warm cache and no network
        "TOKENIZERS_PARALLELISM": "false",
    }
    argv = [str(script), "--config", str(cfg_path), "--stage", "preprocess"]
    proc = subprocess.Popen(  # fixed argv, no shell
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, err = proc.communicate()
        pytest.fail(f"`run --stage preprocess` hung > {_TIMEOUT_S}s\nstderr:\n{err}")
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def test_chunk_worker_through_real_console_script(tmp_path: Path) -> None:
    pytest.importorskip("wtpsplit")  # an import only, so no session and no threads
    pytest.importorskip("onnxruntime")
    script = _console_script()
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    # Work from a copy, because a tracked config is never edited, and strip
    # `execution.artifact_root` from it. The shipped configs name an absolute scratch root,
    # so a verbatim copy would send the worker there while this test watched an empty
    # temporary queue: the shards would sit in `pending/` for ever and the failure would
    # read as a work-queue bug. Stripping the line closes the config half of that trap and
    # `_run_worker_subprocess` closes the environment half. Both are needed, because
    # `cli.artifact_root` treats a stated config root that disagrees with the environment
    # variable as a hard error.
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / _CONFIG.name).write_text(
        "".join(
            line
            for line in _CONFIG.read_text(encoding="utf-8").splitlines(keepends=True)
            if not re.match(r"^\s+artifact_root:", line)
        ),
        encoding="utf-8",
    )
    cfg = load(cfg_dir / _CONFIG.name)
    queue, n_shards = _stage_and_seed(tmp_path, cfg)

    res = _run_worker_subprocess(script, tmp_path, cfg_dir / _CONFIG.name)
    log = res.stdout + res.stderr
    _skip_if_env_gap(log)

    # The start-method failure itself, and any shard failure the worker swallowed.
    assert "bootstrapping phase" not in log, log
    assert "saturate.work.failed" not in log, log  # the record carries the traceback
    assert res.returncode == 0, f"exit={res.returncode}\n{log}"

    # The queue drained through the pool: every shard done, none poisoned.
    assert not list((queue / "failed").glob("*")), "shards were poisoned to failed/"
    assert not list((queue / "pending").glob("*")) and not list((queue / "running").glob("*"))
    outs = sorted(p for p in (queue / "out").glob("*") if not p.name.endswith("._SUCCESS"))
    assert len(outs) == n_shards
    for p in outs:
        assert p.with_name(f"{p.name}._SUCCESS").exists()  # validated before being marked done

    # The shard outputs are nested {document, sentences} records, and every sentence is a
    # verbatim NFC span of its own document text.
    recs = [json.loads(line) for p in outs for line in p.read_text(encoding="utf-8").splitlines()]
    assert len(recs) >= n_shards
    assert {r["document_id"] for r in recs} == {f"eng-docs/{i:07d}" for i in range(_N_DOCS)}
    for r in recs:
        assert set(r) == {"document_id", "lang", "text", "sentences"}
        assert r["sentences"]
        text = r["text"]
        for j, sent in enumerate(r["sentences"]):
            assert sent["sentence_index"] == j  # dense, zero-based, in document order
            assert sent["sentence_id"] == f"{r['document_id']}#s{j}"
            span = text[sent["start"] : sent["end"]]
            assert span and span.strip() == span
            assert unicodedata.is_normalized("NFC", span)
