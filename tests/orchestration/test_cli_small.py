"""The CLI hard gate, and the rule that it never inlines its own fairness recipe.

The fairness gate is hard and first: a divergent config aborts with a ``FairnessError``
before ``plan.build_plan`` runs and before any ``sbatch`` ever fires. ``cli.main`` accepts
a single ``--config`` (its shipped signature), so the fairness trip here is a genuine
single-config knob violation (an mlir run that moved the reading knob), the exact shape
``config.check`` catches before a plan exists.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ragtime import config
from ragtime.orchestration import cli, plan

pytestmark = pytest.mark.small

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _mutated_mlir(tmp_path: Path) -> Path:
    """A real mlir config that also moves the reading knob: a single-config fairness trip."""
    text = (_REPO_ROOT / "config" / "mlir-original.yml").read_text(encoding="utf-8")
    p = tmp_path / "cfg.yml"
    p.write_text(text + "\npassage_lang: omt\n", encoding="utf-8")
    return p


def test_fairness_aborts_before_build_plan_and_sbatch(
    tmp_path: Path,
    sbatch_spy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A divergent config raises FairnessError before build_plan/sbatch (invariant 1)."""
    calls: list[object] = []
    monkeypatch.setattr(plan, "build_plan", lambda cfg: calls.append(cfg))

    with pytest.raises(config.FairnessError):
        cli.main(["--config", str(_mutated_mlir(tmp_path))])

    assert calls == []  # build_plan never reached
    assert sbatch_spy.calls == []  # no job submitted


def test_fairness_error_is_propagated_not_swallowed(tmp_path: Path, sbatch_spy) -> None:
    """The raised type is exactly config.FairnessError (not re-wrapped) and never returns 0."""
    with pytest.raises(config.FairnessError):
        cli.main(["--config", str(_mutated_mlir(tmp_path))])
    assert sbatch_spy.calls == []


def test_positive_control_unmutated_config_passes_gate(sbatch_spy) -> None:
    """The same config, unmutated, clears the gate, so the test is not vacuously green."""
    rc = cli.main(["--config", str(_REPO_ROOT / "config" / "mlir-original.yml"), "--dry-run"])
    assert rc == 0
    assert sbatch_spy.calls == []  # --dry-run never submits


def test_cli_uses_shipped_config_check_not_an_inlined_copy() -> None:
    """cli routes the gate through ragtime.config.check rather than its own recipe."""
    assert cli.config.check is config.check
    # No copy of the raw-text fairness algorithm lives in orchestration.cli.
    src = inspect.getsource(cli)
    assert "top_level_blocks" not in src
    assert "shared_block_hash" not in src


# --------------------------------------------------------------------------- #
# The cross-run family gate: a divergent SIBLING (not a self-violation) must abort.
# The gate discovers the same-family roster on disk and hands the whole family to
# family_guard, so a >=2-member shared-block divergence is actually reachable.
# --------------------------------------------------------------------------- #
def _copy_cfg(src_name: str, dst_dir: Path, transform=None) -> Path:
    text = (_REPO_ROOT / "config" / src_name).read_text(encoding="utf-8")
    if transform is not None:
        text = transform(text)
    out = dst_dir / src_name
    out.write_text(text, encoding="utf-8")
    return out


def test_clean_two_member_family_passes_gate(tmp_path: Path, sbatch_spy) -> None:
    """Two fairness-clean e2e siblings in one dir clear the gate (positive control)."""
    d = tmp_path / "config"
    d.mkdir()
    _copy_cfg("e2e-omt.yml", d)
    target = _copy_cfg("e2e-omt-weak.yml", d)
    assert cli.main(["--config", str(target), "--dry-run"]) == 0
    assert sbatch_spy.calls == []


def test_unloadable_sibling_yml_does_not_crash_a_valid_launch(tmp_path: Path, sbatch_spy) -> None:
    """A malformed / duplicate-key stray ``*.yml`` in the config dir must be skipped
    during roster discovery, never crash an otherwise-valid launch (regression: the
    roster's except clause only caught ConfigError, letting ruamel ParserError /
    DuplicateKeyError from a bad sibling propagate and abort a good target)."""
    d = tmp_path / "config"
    d.mkdir()
    _copy_cfg("e2e-omt.yml", d)  # clean family member
    target = _copy_cfg("e2e-omt-weak.yml", d)  # valid target
    (d / "zz-broken.yml").write_text("run:\n  id: x\n    bad-indent: [\n", encoding="utf-8")  # ParserError
    (d / "zz-dup.yml").write_text("run:\n  id: a\nrun:\n  id: b\n", encoding="utf-8")  # DuplicateKeyError
    assert cli.main(["--config", str(target), "--dry-run"]) == 0  # bad siblings skipped, not fatal
    assert sbatch_spy.calls == []


def test_divergent_sibling_shared_block_aborts_through_cli(
    tmp_path: Path, sbatch_spy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared-block divergence between TWO siblings aborts before build_plan/sbatch.

    This is the case a single-config gate could never catch: the family_guard shared-block
    check only fires with >=2 members, so the cli must discover + gate the whole roster.
    """
    d = tmp_path / "config"
    d.mkdir()
    _copy_cfg("e2e-omt.yml", d)  # clean sibling
    target = _copy_cfg(
        "e2e-omt-weak.yml",
        d,
        # Replace every occurrence, not just the first. `llm.model` and `decomposition.model`
        # must agree, and `schema.validate` enforces it, since there is no separate
        # decomposer model, only one shared vLLM, so a count=1 replace diverges them and raises
        # ConfigError during *validation*, before `family_guard` ever runs. This test is about
        # the FAIRNESS abort, so the config must stay internally valid and differ from its
        # sibling only in the shared `llm` block.
        transform=lambda t: t.replace("Qwen/Qwen3.5-122B-A10B-FP8", "Qwen/DIVERGED"),
    )
    calls: list[object] = []
    monkeypatch.setattr(plan, "build_plan", lambda cfg: calls.append(cfg))

    with pytest.raises(config.FairnessError):
        cli.main(["--config", str(target), "--dry-run"])

    assert calls == []  # aborted before the plan was built
    assert sbatch_spy.calls == []  # and before any job was submitted
