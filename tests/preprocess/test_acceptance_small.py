"""Corpus acceptance: the step-0 precondition, on every refusal path.

``step0_precondition`` decides whether the corpus-scale acceptance pass may run at all. Its
value is that it refuses a bounded-shard fixture, a half-built tree or a mis-resolved path,
and a precondition nobody has ever seen fail is not a precondition. This repository has
produced three green results that had quietly skipped themselves, and one assemble build
was stopped at 3 of 188 parts; an unproven step 0 pointed at that tree is the failure this
file forecloses.

So every ``raise`` in ``step0_precondition`` has a test here, and one test covers the
accepting side on a synthetic tree that really is at corpus scale. A check that only ever
says no is as useless as one that only ever says yes.

The tests build a synthetic index tree of censuses and a manifest, with no engines and no
vectors, and drive the real function, so they need neither faiss, seismic nor pylate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout
from ragtime.common.io import write_jsonl
from ragtime.preprocess.acceptance import (
    MIN_CORPUS_PASSAGES,
    AcceptanceError,
    _Resolved,
    cells,
    step0_precondition,
)
from ragtime.preprocess.index import LEGS, index_part_name

pytestmark = pytest.mark.small

_RECON = "a" * 64
_PACK = "b" * 64
_IDX = "c" * 64
_PART = 131_072
#: Passages per language in the synthetic tree. Four languages at 3,000,000 gives 12 M,
#: above the real corpus at about 9.9 M and well over MIN_CORPUS_PASSAGES.
_PER_LANG = 3_000_000
_LANGS = ("es", "ru", "zh")


class _Cfg:
    """The two attributes ``step0_precondition`` reads off a RunConfig."""

    languages = ("en", *_LANGS)
    source_path = Path("config/synthetic.yml")


def _resolved(root: Path) -> _Resolved:
    layout = Layout(run_dir=str(root), base=str(root), family="e2e", chunker_hash="d" * 64)
    return _Resolved(
        cfg=_Cfg(),
        adapter=None,  # unused by step0
        layout=layout,
        opts=None,  # unused by step0
        recon_hash=_RECON,
        pack_hash=_PACK,
        idx_hash=_IDX,
        part_passages=_PART,
        langs=_LANGS,
        encode_hashes={leg: f"{leg}-hash" for leg in LEGS},
    )


def _write_cell(res: _Resolved, variant: str | None, lang: str, passages: int) -> None:
    """One cell's part census: what ``read_shard_parts`` cross-checks against the disk."""
    parts = max(1, -(-passages // _PART))
    for part in range(parts):
        d = res.shard_dir(variant, lang, part)
        d.mkdir(parents=True, exist_ok=True)
        n = min(_PART, passages - part * _PART)
        write_jsonl(
            d / "part.json",
            [
                {
                    "part": part,
                    "parts": parts,
                    "part_passages": _PART,
                    "passages": n,
                    "source_lang": lang,
                    "variant": variant,
                }
            ],
            skip_if_done=False,
        )
        # The census sits in the directory its own index names. read_shard_parts rejects a
        # part in the wrong directory, so the synthetic tree has to get this right too.
        assert d.name == index_part_name(part)


def _write_manifest(res: _Resolved, **overrides: Any) -> None:
    payload: dict[str, Any] = {
        "index_hash": _IDX,
        "reconcile_hash": _RECON,
        "packing_hash": _PACK,
        "legs": list(LEGS),
        "variants": {v: {"passages": _PER_LANG * 3 + _PER_LANG} for v in ("original",)},
        "leg_encode_hash": dict(res.encode_hashes),
        "created_at": "2026-08-05T00:00:00+00:00",
    }
    payload.update(overrides)
    path = res.layout.index_manifest_path(_RECON, _IDX)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, [payload], skip_if_done=False)


def _corpus_scale_tree(tmp_path: Path, *, per_lang: int = _PER_LANG) -> _Resolved:
    res = _resolved(tmp_path)
    for variant, lang in cells(res):
        _write_cell(res, variant, lang, per_lang)
    _write_manifest(res)
    return res


# --------------------------------------------------------------------------- #
# The accepting side: a tree that really is at corpus scale.
# --------------------------------------------------------------------------- #
def test_step0_accepts_a_corpus_scale_tree(tmp_path: Path) -> None:
    """SA01: four languages at 3 M passages each is accepted, and reported."""
    res = _corpus_scale_tree(tmp_path)
    got = step0_precondition(res)
    assert got["passed"] is True
    assert got["witnessed_passages_total"] == 4 * _PER_LANG
    assert got["witnessed_passages_by_lang"] == {lang: _PER_LANG for lang in ("en", *_LANGS)}
    # Ten cells, each reporting the parts its own census witnesses.
    assert len(got["cells"]) == 10
    assert got["cells"]["_shared/en"]["parts"] == -(-_PER_LANG // _PART)


# --------------------------------------------------------------------------- #
# Every refusal path, one test each.
# --------------------------------------------------------------------------- #
def test_step0_refuses_when_the_manifest_is_absent(tmp_path: Path) -> None:
    """SA02: a build that never reached ``merge()`` publishes no manifest, so refuse.

    That is the state one real assemble run was left in, 3 of 188 parts and no manifest.
    The pass stops here rather than validating a partial index.
    """
    res = _resolved(tmp_path)
    for variant, lang in cells(res):
        _write_cell(res, variant, lang, _PER_LANG)
    with pytest.raises(AcceptanceError, match="no published index manifest"):
        step0_precondition(res)


def test_step0_refuses_a_bounded_shard_fixture(tmp_path: Path) -> None:
    """SA03: a complete but small index is refused.

    Everything here is internally consistent: the manifest is present, every cell's census
    is whole and the ids are fine. It is simply not the corpus. A pass that ran green
    against this would be green for having validated the wrong artefact.
    """
    res = _corpus_scale_tree(tmp_path, per_lang=600)
    with pytest.raises(AcceptanceError, match="step 0 failed") as exc:
        step0_precondition(res)
    assert "not the corpus-scale index" in str(exc.value).lower() or "NOT the" in str(exc.value)
    assert f"{4 * 600:,}" in str(exc.value)


def test_the_floor_is_the_stated_one() -> None:
    """SA04a: the floor is pinned to a literal, so moving it is a deliberate act.

    It is pinned separately from SA04b so that the boundary test below can use plain
    literals. A boundary test that derives its inputs from the constant it is testing cannot
    notice that constant being weakened: with the floor mutated to 0, ``(0 - 4) // 4`` goes
    negative and such a test passes for the wrong reason. With this pin and those literals, a
    floor of 0 fails SA03 and SA04b.
    """
    assert MIN_CORPUS_PASSAGES == 1_000_000


@pytest.mark.parametrize(
    ("per_lang", "refused"),
    [(249_999, True), (250_000, False)],  # 4 x these straddle the 1,000,000 floor exactly
)
def test_step0_boundary_at_the_floor(tmp_path: Path, per_lang: int, refused: bool) -> None:
    """SA04b: the boundary bites. 999,996 is refused and 1,000,000 is accepted."""
    res = _corpus_scale_tree(tmp_path, per_lang=per_lang)
    if refused:
        with pytest.raises(AcceptanceError, match="step 0 failed"):
            step0_precondition(res)
    else:
        assert step0_precondition(res)["witnessed_passages_total"] == 4 * per_lang


def test_step0_refuses_a_missing_cell(tmp_path: Path) -> None:
    """SA05: a rendering that was never built is refused, not averaged away."""
    res = _corpus_scale_tree(tmp_path)
    import shutil

    shutil.rmtree(res.lang_dir("omt_opus", "zh"))
    with pytest.raises(AcceptanceError, match="cell directory is missing"):
        step0_precondition(res)


def test_step0_refuses_renderings_that_disagree_on_a_count(tmp_path: Path) -> None:
    """SA06: the three renderings of one language witness the same passage count.

    One build cannot produce anything else, because parts are cut on the single text-free
    table, which has no ``variant`` column. A disagreement therefore means the tree mixes
    two builds, either two packing or reconcile hashes or a hand-copied directory. Refuse
    before any check reads it.
    """
    res = _corpus_scale_tree(tmp_path)
    import shutil

    shutil.rmtree(res.lang_dir("omt", "es"))
    _write_cell(res, "omt", "es", _PER_LANG - _PART)
    with pytest.raises(AcceptanceError, match="witness different passage counts"):
        step0_precondition(res)


def test_step0_refuses_a_manifest_describing_another_build(tmp_path: Path) -> None:
    """SA07: the manifest identities equal the ones the config resolves.

    A stale manifest at the right path is the quietest failure of all: every downstream
    check would run happily against an artefact the config does not name.
    """
    res = _corpus_scale_tree(tmp_path)
    _write_manifest(res, packing_hash="deadbeef" * 8)
    with pytest.raises(AcceptanceError, match="disagree with the ones"):
        step0_precondition(res)


def test_step0_refuses_a_cell_whose_census_is_short(tmp_path: Path) -> None:
    """SA08: a part directory deleted after the census was written is refused.

    ``read_shard_parts`` requires the parts on disk to cover ``0..parts-1`` exactly. Step 0
    inherits that rather than re-implementing it, so a cell that would be searched short
    cannot reach a check. The error is an ``IndexIntegrityError`` rather than an
    ``AcceptanceError`` because the shared reader raises first, which is the point of
    reusing it.
    """
    from ragtime.preprocess.index import IndexIntegrityError

    res = _corpus_scale_tree(tmp_path)
    import shutil

    census = res.shard_dir(None, "en", 1)
    shutil.rmtree(census)
    with pytest.raises((AcceptanceError, IndexIntegrityError)):
        step0_precondition(res)


def test_step0_runs_in_every_phase_and_is_not_skippable() -> None:
    """SA09: ``main`` calls step 0 before dispatching any phase, including ``--phase A``.

    This reads the source rather than the behaviour, because observing the call in each
    phase would need the real artefact tree. What has to hold is that no phase branch is
    reachable without step 0 having run first.
    """
    import inspect

    from ragtime.preprocess import acceptance

    src = inspect.getsource(acceptance.main)
    step0_at = src.index("step0_precondition(res)")
    for branch in ('args.phase == "A"', 'args.phase == "B1"', "phase_b2("):
        assert src.index(branch) > step0_at, f"{branch} is reachable before Step 0"


# --------------------------------------------------------------------------- #
# FA01-dense: a rank-0 miss explained by a duplicate tie is not a build defect.
#
# Three of ten real cells failed on ties while an independent census of all 24,011,708
# dense vectors found no NaN and no zero-norm vector. The duplicate exception was already
# named in the check's own docstring and the competitors' scores were already recorded;
# only the comparison was missing.
#
# The rows below are the recorded evidence from those runs rather than invented shapes, so
# a regression here means the real cells fail again.
# --------------------------------------------------------------------------- #
from ragtime.preprocess.acceptance import duplicate_tie_explains_miss

_REAL_TIE_ROWS = [
    # _shared/en: probe at rank 1, both window entries at the maximum
    {"rank": 1, "top_score": 1.0, "competitors": [{"score": 1.0}, {"score": 1.0}]},
    # original/ru: the probe is pushed out of top-k by five maximum-scoring duplicates
    {"rank": -1, "top_score": 1.0000001192092896,
     "competitors": [{"score": 1.0000001192092896}] * 5},
    # omt_opus/es: same shape as _shared/en, at the float32-ulp score
    {"rank": 1, "top_score": 1.0000001192092896,
     "competitors": [{"score": 1.0000001192092896}] * 2},
]


@pytest.mark.small
@pytest.mark.parametrize("row", _REAL_TIE_ROWS, ids=["shared_en", "original_ru", "omt_opus_es"])
def test_fa01_duplicate_tie_is_explained(row: dict[str, Any]) -> None:
    """The three cells that failed are recognised as ties."""
    assert duplicate_tie_explains_miss(row) is True


@pytest.mark.small
@pytest.mark.parametrize(
    "row",
    [
        # The defect this check exists for: a mis-keyed id map. The probe's vector is not
        # where the idmap says, so the window holds varied scores below the maximum, which
        # a duplicate tie cannot produce because <v,v> = 1 cannot be beaten.
        {"rank": -1, "top_score": 0.83, "competitors": [{"score": 0.83}, {"score": 0.79}]},
        # Only part of the window is at the maximum, so it is not saturated.
        {"rank": -1, "top_score": 1.0, "competitors": [{"score": 1.0}, {"score": 0.4}]},
        # No evidence recorded at all: fail closed rather than explain away a blind miss.
        {"rank": -1, "top_score": 0.9, "competitors": []},
        {"rank": -1, "top_score": None, "competitors": [{"score": 1.0}]},
        # Malformed evidence is not agreement.
        {"rank": -1, "top_score": float("nan"), "competitors": [{"score": 1.0}]},
        {"rank": -1, "top_score": 1.0, "competitors": [{"score": None}]},
        {"rank": -1, "top_score": 1.0, "competitors": [{"score": float("nan")}]},
    ],
    ids=["miskeyed_idmap", "partially_saturated", "no_competitors", "no_top_score",
         "nan_top_score", "none_score", "nan_score"],
)
def test_fa01_a_real_defect_is_never_explained_away(row: dict[str, Any]) -> None:
    """Anything that is not a saturated maximum-score window stays a failure."""
    assert duplicate_tie_explains_miss(row) is False


@pytest.mark.small
def test_fa01_tie_tolerance_is_one_float32_ulp_scale_not_a_loose_window() -> None:
    """The epsilon admits float32 self-match noise and nothing wider.

    A self-match returns 1.0 give or take one float32 ulp, observed at 1.0000001192092896,
    so the tolerance has to cover that. A score genuinely below the maximum, which can only
    mean a mis-keyed idmap, still has to fail. A looser window would disable the check
    without looking like it.
    """
    ulp = {"rank": -1, "top_score": 1.0, "competitors": [{"score": 1.0000001192092896}]}
    assert duplicate_tie_explains_miss(ulp) is True
    just_below = {"rank": -1, "top_score": 1.0, "competitors": [{"score": 1.0 - 1e-4}]}
    assert duplicate_tie_explains_miss(just_below) is False
