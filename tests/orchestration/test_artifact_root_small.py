"""The artifact-root knob: one resolver, no hash movement, one rooted tree.

Three properties, and the third is the reason the knob sits where it does:

1. Resolution: ``execution.artifact_root``, which is the run record, then
   ``$RAGTIME_ARTIFACT_ROOT``, which is the site fact, then ``"runs"``. A config and a
   disagreeing environment are a hard error rather than a silent precedence rule, because two
   roots mean two half-built corpora, each with its own ``_SUCCESS`` tree.
2. Rooting: every corpus, queue and image path a submission carries hangs off that one value,
   and the submitted ``--export`` carries it, so the worker roots its Layout where its
   submitter did.
3. Hash invariance: a filesystem path is a machine fact, so stating it must move no semantic
   block hash. ``chunker`` keys ``corpus/<family>/<chunker12>/``, and the reconcile, packing,
   index and three per-leg encode hashes are pure functions of ``chunker``, ``merge``,
   ``translation``, ``reconcile``, ``packing`` and ``index_build``. If any of those moved,
   re-homing the store would orphan the translated corpus and the whole vector store by path,
   which is why the knob lives in the non-shared, never-compared ``execution`` block.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ragtime import config
from ragtime.config import all_hashes
from ragtime.orchestration import cli

pytestmark = pytest.mark.small

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG_DIR = _REPO_ROOT / "config"
_E2E = _CFG_DIR / "e2e-omt.yml"

#: A root that is nothing like the default, so a stale "runs" cannot pass by accident.
_SCRATCH = "/srv/beegfs/scratch/users/x/xyz/ragtime-runs"


def _without_root(tmp_path: Path, src: Path) -> Path:
    """A temporary copy of a real config with ``execution.artifact_root`` removed.

    The shipped configs carry the leaf, so a silent config is a state a test must construct
    rather than assume. Asserting the default against a config that sets the value would test
    nothing.
    """
    lines = [
        line
        for line in src.read_text(encoding="utf-8").splitlines(keepends=True)
        if not re.match(r"^\s+artifact_root:", line)
    ]
    dst = tmp_path / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(lines), encoding="utf-8")
    return dst


def _with_root(tmp_path: Path, src: Path, root: str) -> Path:
    """A temporary copy of a real config carrying ``execution.artifact_root``.

    The tracked ``config/*.yml`` are read-only test data, and landing that leaf in them is a
    fairness-family decision made by a human, not by a test.
    """
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    # Replace the leaf if it is already there, otherwise insert it. The shipped configs carry
    # `execution.artifact_root`, so a blind insert would produce a second key and make ruamel
    # raise DuplicateKeyError, which reads as a resolver bug rather than a fixture one.
    for i, line in enumerate(lines):
        if re.match(r"^\s+artifact_root:", line):
            lines[i] = f"  artifact_root: {root}\n"
            break
    else:
        for i, line in enumerate(lines):
            if line.startswith("execution:"):
                lines.insert(i + 1, f"  artifact_root: {root}\n")
                break
        else:  # pragma: no cover - every shipped config has an execution block
            raise AssertionError(f"{src} has no execution block")
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / src.name
    out.write_text("".join(lines), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# 1. Resolution.
# --------------------------------------------------------------------------- #
def test_default_root_is_the_repo_relative_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cli.ARTIFACT_ROOT_ENV, raising=False)
    assert cli.artifact_root(config.load(_without_root(tmp_path, _E2E))) == "runs"


def test_environment_supplies_the_root_when_the_config_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cli.ARTIFACT_ROOT_ENV, _SCRATCH)
    assert cli.artifact_root(config.load(_without_root(tmp_path, _E2E))) == _SCRATCH


def test_config_leaf_is_the_record_and_resolves_without_any_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cli.ARTIFACT_ROOT_ENV, raising=False)
    cfg = config.load(_with_root(tmp_path, _E2E, _SCRATCH))
    assert cli.artifact_root(cfg) == _SCRATCH


def test_config_and_environment_disagreement_is_a_hard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cli.ARTIFACT_ROOT_ENV, "/some/other/root")
    cfg = config.load(_with_root(tmp_path, _E2E, _SCRATCH))
    with pytest.raises(config.ConfigError, match="artifact_root"):
        cli.artifact_root(cfg)


def test_config_and_environment_agreement_is_accepted_modulo_a_trailing_slash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same path spelled two ways is agreement, not a launch failure."""
    monkeypatch.setenv(cli.ARTIFACT_ROOT_ENV, _SCRATCH + "/")
    cfg = config.load(_with_root(tmp_path, _E2E, _SCRATCH))
    assert cli.artifact_root(cfg) == _SCRATCH


def test_the_new_execution_leaf_loads_on_every_real_config(tmp_path: Path) -> None:
    """``execution`` is a CLOSED schema: without its entry every config fails to load."""
    for src in sorted(_CFG_DIR.glob("*.yml")):
        cfg = config.load(_with_root(tmp_path / src.stem, src, _SCRATCH))
        assert cfg.blocks["execution"]["artifact_root"] == _SCRATCH


# --------------------------------------------------------------------------- #
# 2. Rooting: the submitted tree hangs off the one value.
# --------------------------------------------------------------------------- #
def test_every_submission_exports_the_root_and_roots_its_queue_under_it(
    sbatch_spy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cli.ARTIFACT_ROOT_ENV, _SCRATCH)
    # The config must be silent for the environment to supply the root: the shipped configs
    # carry the leaf, and a config value beside a disagreeing environment is a hard
    # ConfigError by design rather than a precedence rule.
    assert cli.main(["--config", str(_without_root(tmp_path, _E2E))]) == 0
    assert sbatch_spy.calls

    for argv in sbatch_spy.calls:
        export = next(a for a in argv if a.startswith("--export="))
        assert f"{cli.ARTIFACT_ROOT_ENV}={_SCRATCH}" in export
        assert f"RAGTIME_SIF={_SCRATCH}/containers/" in export
        if "WQ_DIR=" in export:
            wq = export.split("WQ_DIR=", 1)[1].split(",")[0]
            assert wq.startswith(f"{_SCRATCH}/corpus/"), wq


def test_the_default_root_submission_is_unchanged_by_the_knob(
    sbatch_spy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: with nothing set, every path is the repo-relative one it was."""
    monkeypatch.delenv(cli.ARTIFACT_ROOT_ENV, raising=False)
    # Neither knob set: the resolver must fall back to the repo-relative "runs".
    assert cli.main(["--config", str(_without_root(tmp_path, _E2E))]) == 0
    corpus = [a for a in sbatch_spy.calls if any("WQ_DIR=" in x for x in a)]
    assert corpus
    for argv in corpus:
        export = next(a for a in argv if a.startswith("--export="))
        wq = export.split("WQ_DIR=", 1)[1].split(",")[0]
        assert wq.startswith("runs/corpus/"), wq


# --------------------------------------------------------------------------- #
# 3. Hash invariance: the reason for the knob's placement.
# --------------------------------------------------------------------------- #
def test_stating_the_root_moves_no_semantic_block_hash(tmp_path: Path) -> None:
    """Only ``execution``'s own hash moves; every other block is byte-for-byte identical.

    That ``execution`` itself moves is asserted too, since otherwise the test would pass
    just as happily against a knob that was never read.
    """
    for src in sorted(_CFG_DIR.glob("*.yml")):
        before = all_hashes(config.load(src))
        after = all_hashes(config.load(_with_root(tmp_path / src.stem, src, _SCRATCH)))
        assert set(before) == set(after), src.name
        assert after["execution"] != before["execution"], src.name
        for block, digest in before.items():
            if block == "execution":
                continue
            assert after[block] == digest, f"{src.name}: {block} hash moved"


def test_the_corpus_path_hash_level_is_unmoved(tmp_path: Path) -> None:
    """``chunker12``, the level ``Layout.corpus_dir`` keys the whole build on, is fixed.

    Stated separately from the loop above because this is the one whose movement would
    orphan the shipped corpus by path rather than merely re-key a derived artifact.
    """
    for src in sorted(_CFG_DIR.glob("*.yml")):
        before = all_hashes(config.load(src))["chunker"][:12]
        after = all_hashes(config.load(_with_root(tmp_path / src.stem, src, _SCRATCH)))
        assert after["chunker"][:12] == before, src.name
