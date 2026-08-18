"""The translate ShapeSet and its measured calibration rows, cross-checked three ways.

Three files have to agree before a GPU is ever allocated, and nothing else checks them
against each other:

* ``config/<run>.yml``: ``execution.translate_shape_key`` names the acceptance set.
* ``config/serving/models.yml``: authors that set, meaning which cards and what allocation
  shape.
* the cluster capacity profile: the measured ``shape_calibration`` rows, what each card
  actually withstood, and the only sanctioned source of batch knobs.

The binding fact is ``semantic_composition_ok``. ``translation.config``'s
``bucket_token_budget`` x ``max_sentences_per_bucket`` is fairness-hashed and therefore
identical on every card, so a card whose measured ceiling falls below it would make CT2
re-split every bucket: the same shard would stop composing identically across cards, and the
translation delta would become a hardware artefact. That is why the 3090, measured at 9 216,
is absent both here and from the launcher's OR-set, and why this compares the YAML against
the launcher constraint rather than trusting either alone.

These are pure data reads: nothing imports torch or ctranslate2, and nothing touches a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML  # the loader `serving.modelspec` uses (never PyYAML)

from ragtime.config import load
from ragtime.orchestration.slurm import launcher

pytestmark = pytest.mark.small

_YAML = YAML(typ="safe")

_REPO = Path(__file__).resolve().parents[2]
_MODELS = _REPO / "config" / "serving" / "models.yml"
_BAMBOO = _REPO / "config" / "serving" / "bamboo.yml"
_E2E_OMT = _REPO / "config" / "e2e-omt.yml"

#: Authored but never calibrated, per arm. Carried as an explicit gap rather than an invented
#: row, because an unmeasured ceiling one config read away from a production allocation is
#: exactly the failure this file exists to prevent.
#:
#: The OPUS-MT arm lists every shape: its three Marian checkpoints differ from NLLB-3.3B by a
#: factor of 45 in weights, so not one of the NLLB rows carries over. Until they are measured
#: the honest state is a hole, not a copied number.
_UNCALIBRATED: dict[str, set[str]] = {
    "facebook/nllb-200-3.3B@translate": {"h200x1"},
    "Helsinki-NLP/opus-mt@translate": {"bwx1", "h200x1", "h100x1", "a100x1", "5090x1"},
}
#: Post-calibration packaging option, not raced ahead of the gpu:1 shapes.
_CANDIDATE_ONLY = {"a100x4"}


def _yaml(path: Path) -> dict:
    return _YAML.load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shape_key() -> str:
    return str(load(_E2E_OMT).blocks["execution"]["translate_shape_key"])


@pytest.fixture(scope="module")
def authored(shape_key: str) -> list[dict]:
    model, role = shape_key.split("@", 1)
    return list(_yaml(_MODELS)["mt"][model]["shapes"][role])


@pytest.fixture(scope="module")
def calibrated(shape_key: str) -> dict[str, dict]:
    """The measured rows for the configured arm: ``{}`` when none exist yet."""
    return dict(_yaml(_BAMBOO).get("shape_calibration", {}).get(shape_key, {}))


@pytest.fixture(scope="module")
def uncalibrated(shape_key: str) -> set[str]:
    assert shape_key in _UNCALIBRATED, (
        f"{shape_key!r} has no declared calibration status: a new arm must state which "
        "of its shapes are measured before it can be raced"
    )
    return _UNCALIBRATED[shape_key]


def test_the_configured_shape_key_actually_resolves(shape_key: str, authored) -> None:
    """``execution.translate_shape_key`` must name a set that EXISTS in models.yml."""
    assert shape_key in _UNCALIBRATED, f"unknown arm {shape_key!r}"
    assert authored, "the authored ShapeSet is empty"


def test_every_calibrated_shape_is_an_authored_one(authored, calibrated, uncalibrated) -> None:
    names = {s["name"] for s in authored}
    assert set(calibrated) <= names
    assert set(calibrated) == names - uncalibrated - _CANDIDATE_ONLY


def test_the_uncalibrated_shape_is_carried_as_a_gap_not_as_a_proven_row(
    authored, calibrated, uncalibrated
) -> None:
    """A shape is authored because the fleet has the card, not because a number exists.

    Inventing a row would put an unmeasured batch ceiling one config read away from a
    production allocation. The correct state is a visible hole.
    """
    assert uncalibrated <= {s["name"] for s in authored}
    for name in uncalibrated:
        assert name not in calibrated


def test_every_calibrated_shape_hosts_the_fairness_hashed_composition(calibrated) -> None:
    tcfg = load(_E2E_OMT).blocks["translation"]["config"]
    budget = int(tcfg["bucket_token_budget"])
    sentences = int(tcfg["max_sentences_per_bucket"])
    for name, row in calibrated.items():
        assert row["status"] == "proven", name
        assert row["semantic_composition_ok"] is True, name
        assert row["semantic_max_batch_tokens"] == budget, name
        assert row["semantic_max_sentences_per_batch"] == sentences, name
        # ...and the card's own measured ceiling clears the shared budget, which is what
        # keeps CT2 from re-splitting a conforming bucket on that card.
        assert int(row["max_batch_tokens"]) >= budget, name


def test_the_gpu_or_set_is_exactly_the_usable_shapes_hardware(authored, calibrated) -> None:
    """A card may be raced only if it is authored and calibrated (or authored-but-not-yet-
    calibrated, which the OR-set still allows so a fresh card can be tried); a card that
    failed the semantic composition must never appear."""
    constraint = next(
        a for a in launcher.gpu_constraint_args() if a.startswith("--constraint=")
    ).split("=", 1)[1]
    cards = set(constraint.split("|"))
    gpu1 = {s["gpu_model"] for s in authored if s.get("status", "proven") != "candidate"}
    assert cards == gpu1
    for row in calibrated.values():
        assert row["gpu_model"] in cards
    assert "nvidia_geforce_rtx_3090" not in cards  # measured 9 216 < the 16 384 budget


def test_gpu1_shapes_are_parallelism_none(authored) -> None:
    """NLLB-3.3B fits on one card, so there is nothing to tensor-shard: 1 GPU, 1 CT2
    translator, 1 array task. Anything multi-GPU is `data` packaging and stays candidate."""
    for s in authored:
        if s["gpu_count"] == 1:
            assert s["parallelism"] == "none", s["name"]
        else:
            assert s["parallelism"] == "data" and s.get("status") == "candidate", s["name"]
