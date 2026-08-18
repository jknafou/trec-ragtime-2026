"""Decompose units: k-band, exemplars, bank ops, weighting, saturation, on_topic, fingerprint.

FT-B13/B14, FT-B17/B18, FT-B21/B22, FT-B24/B25, FT-B26, FT-B36..B42.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from ragtime.common import Answer, Nugget, Statistics
from ragtime.pipeline.decompose import bank as bank_ops
from ragtime.pipeline.decompose import (
    limit_to_k,
    on_topic,
    saturated,
    select_exemplars,
    weight_and_dedup,
)
from ragtime.pipeline.decompose.exemplars import (
    EXEMPLAR_SETS,
    MAX_EXEMPLARS,
    MIN_EXEMPLARS,
    exemplar_set_fingerprint,
)
from ragtime.pipeline.decompose.kband import KBAND_OUT_OF_RANGE
from ragtime.pipeline.decompose.weighting import WEIGHT_OUT_OF_RANGE, clamp_weight

from .conftest import make_bundle

pytestmark = pytest.mark.small

REAL_K_BAND = [[2000, 4, 8], [5000, 8, 14], [10000, 14, 24]]


def _n(nid: str, question: str, **kw) -> Nugget:
    return Nugget(nugget_id=nid, question=question, **kw)


# --------------------------------------------------------------------------- #
# FT-B17 / FT-B18: the limit -> k_band lookup
# --------------------------------------------------------------------------- #
def test_ft_b17_band_boundaries_are_exact_and_inclusive():
    assert limit_to_k(2000, REAL_K_BAND) == (4, 8)
    assert limit_to_k(5000, REAL_K_BAND) == (8, 14)
    assert limit_to_k(10000, REAL_K_BAND) == (14, 24)
    # one below / one above each boundary, to pin off-by-one
    assert limit_to_k(1999, REAL_K_BAND) == (4, 8)
    assert limit_to_k(2001, REAL_K_BAND) == (8, 14)
    assert limit_to_k(5001, REAL_K_BAND) == (14, 24)


def test_ft_b18_out_of_range_limit_clamps_to_widest_band_and_flags():
    stats = Statistics()
    assert limit_to_k(15000, REAL_K_BAND, stats=stats) == (14, 24)
    assert stats.value(KBAND_OUT_OF_RANGE, round=0) == 1.0


def test_ft_b18b_in_range_limit_does_not_flag():
    stats = Statistics()
    limit_to_k(5000, REAL_K_BAND, stats=stats)
    assert stats.total(KBAND_OUT_OF_RANGE) == 0.0


def test_k_band_table_is_validated_not_trusted():
    with pytest.raises(ValueError, match="ascend"):
        limit_to_k(100, [[5000, 8, 14], [2000, 4, 8]])
    with pytest.raises(ValueError, match="empty"):
        limit_to_k(100, [])
    with pytest.raises(ValueError, match="triple"):
        limit_to_k(100, [[5000, 8]])
    with pytest.raises(ValueError, match="k_min > k_max"):
        limit_to_k(100, [[5000, 14, 8]])


# --------------------------------------------------------------------------- #
# FT-B13 / FT-B14: exemplars
# --------------------------------------------------------------------------- #
def test_ft_b13_exemplar_sets_versioned_and_n_slices_correctly():
    v1 = EXEMPLAR_SETS["autonuggetizer_v1"]
    assert MIN_EXEMPLARS <= len(v1) <= MAX_EXEMPLARS
    for n in (2, 3, 4):
        picked = select_exemplars("autonuggetizer_v1", n)
        assert len(picked) == n
        assert picked == v1[:n], "the slice is a stable PREFIX, not a sample"
    for bad in (0, 1, 5, 99):
        with pytest.raises(ValueError):
            select_exemplars("autonuggetizer_v1", bad)
    with pytest.raises(KeyError):
        select_exemplars("no_such_set", 3)


def test_ft_b13b_exemplars_are_house_style_single_sentence_questions():
    for ex in EXEMPLAR_SETS["autonuggetizer_v1"]:
        for q in ex.nuggets:
            assert q.endswith("?"), q
            assert q.count("?") == 1, q
            # A nugget carries one fact, so a coordinated question is two nuggets, not one.
            assert " and " not in q.lower(), f"compound question: {q}"
            assert " or " not in q.lower(), f"compound question: {q}"


def test_ft_b14_exemplar_set_v1_is_byte_frozen():
    """FT-B14: a v2 is a new constant; editing v1 in place re-keys prompt_hash silently."""
    assert (
        exemplar_set_fingerprint("autonuggetizer_v1")
        == "f3e603ab38f895b1147d99d64643fbed6e6262e8fc3c9c0164a20c77ae95b83c"
    )


# --------------------------------------------------------------------------- #
# FT-B21 / FT-B22: bank ops
# --------------------------------------------------------------------------- #
def test_ft_b21_bank_add_is_replace_based_not_mutating():
    bank0 = (_n("t#n0", "Q0"), _n("t#n1", "Q1"))
    snapshot = tuple(dataclasses.asdict(n) for n in bank0)
    grown = bank_ops.add(bank0, (_n("t#n2", "Q2"),))
    assert len(grown) == 3
    assert len(bank0) == 2
    assert tuple(dataclasses.asdict(n) for n in bank0) == snapshot
    assert grown[0] is bank0[0], "existing entries are carried by identity, not rebuilt"


def test_ft_b21b_add_skips_an_id_already_present():
    bank0 = (_n("t#n0", "Q0"),)
    assert bank_ops.add(bank0, (_n("t#n0", "DIFFERENT"),)) == bank0


def test_ft_b22_open_close_prune_semantics():
    bank0 = (_n("t#n0", "Q0"), _n("t#n1", "Q1"), _n("t#n2", "Q2"))
    assert bank_ops.open_nuggets(bank0) == bank0

    closed = bank_ops.close(bank0, "t#n1")
    assert closed[1].status == "answered"
    assert {n.nugget_id for n in bank_ops.open_nuggets(closed)} == {"t#n0", "t#n2"}

    pruned = bank_ops.prune(closed, ["t#n2"])
    assert pruned[2].status == "pruned"
    assert len(pruned) == 3, "prune flags, it never deletes (the id must stay retired)"
    assert {n.nugget_id for n in bank_ops.open_nuggets(pruned)} == {"t#n0"}

    # a pruned nugget is not resurrected by close()
    assert bank_ops.close(pruned, "t#n2")[2].status == "pruned"


def test_next_index_is_monotone_never_reused():
    bank0 = (_n("t#n0", "Q0"), _n("t#n7", "Q7"))
    assert bank_ops.next_index(bank0) == 8
    assert bank_ops.next_index(()) == 0


def test_mint_sets_the_seed_defaults():
    minted = bank_ops.mint("2000", [("Q?", 0.6), "R?"], start_index=3)
    assert [n.nugget_id for n in minted] == ["2000#n3", "2000#n4"]
    assert minted[0].weight == 0.6
    assert minted[1].weight == 0.0
    assert all(n.aggregator_type == "OR" and n.status == "unanswered" for n in minted)
    assert all(n.retrieved == () and n.answers == () for n in minted)


# --------------------------------------------------------------------------- #
# FT-B24 / FT-B25: weighting
# --------------------------------------------------------------------------- #
def test_ft_b24_out_of_range_weights_are_clamped_not_propagated():
    for raw, expected in ((-0.1, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.1, 1.0)):
        value, bad = clamp_weight(raw)
        assert value == expected
        assert bad is (raw < 0.0 or raw > 1.0)
    assert clamp_weight(float("nan")) == (0.0, True)
    assert clamp_weight(float("inf")) == (0.0, True)


def test_ft_b24b_weight_and_dedup_clamps_and_counts(decompose_cfg):
    stats = Statistics()
    bank = tuple(
        _n(f"t#n{i}", f"Q{i}", weight=w) for i, w in enumerate([-0.1, 0.0, 0.5, 1.0, 1.1])
    )

    async def _identity(b):
        return b

    out = asyncio.run(
        weight_and_dedup(bank, cfg=decompose_cfg, dedup_fn=_identity, stats=stats)
    )
    assert [n.weight for n in out] == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert all(0.0 <= n.weight <= 1.0 for n in out)
    assert stats.value(WEIGHT_OUT_OF_RANGE) == 2.0


def test_ft_b25_vital_is_a_pure_derivation_with_no_extra_llm_call(decompose_cfg):
    """FT-B25: ``vital`` costs zero ``guided_json`` calls beyond the ones already made."""
    clients = make_bundle(nuggets=[])
    decompose_cfg.blocks["decomposition"]["vital_cutoff"] = 0.5
    bank = (_n("t#n0", "Q0", weight=0.6), _n("t#n1", "Q1", weight=0.4))

    async def _identity(b):
        return b

    out = asyncio.run(weight_and_dedup(bank, cfg=decompose_cfg, dedup_fn=_identity))
    assert [n.vital for n in out] == [True, False]
    assert clients.llm.calls == []


def test_vital_cutoff_is_read_from_config_not_hardcoded(decompose_cfg):
    bank = (_n("t#n0", "Q0", weight=0.6),)

    async def _identity(b):
        return b

    decompose_cfg.blocks["decomposition"]["vital_cutoff"] = 0.9
    assert not asyncio.run(
        weight_and_dedup(bank, cfg=decompose_cfg, dedup_fn=_identity)
    )[0].vital
    decompose_cfg.blocks["decomposition"]["vital_cutoff"] = 0.1
    assert asyncio.run(weight_and_dedup(bank, cfg=decompose_cfg, dedup_fn=_identity))[0].vital


# --------------------------------------------------------------------------- #
# FT-B26: on_topic as a standalone predicate
# --------------------------------------------------------------------------- #
def test_ft_b26_on_topic_admits_and_rejects(decompose_cfg):
    ps = "I need a report on flood defences in the Rhine delta."

    rejecting = make_bundle(on_topic={"rationale": "adjacent subject", "on_topic": False})
    assert (
        asyncio.run(
            on_topic(
                "What is the population of Buenos Aires?",
                ps,
                None,
                cfg=decompose_cfg,
                clients=rejecting,
                seed=0,
            )
        )
        is False
    )

    admitting = make_bundle(on_topic={"rationale": "a genuine facet", "on_topic": True})
    assert (
        asyncio.run(
            on_topic(
                "Which agency maintains the Rhine delta dykes?",
                ps,
                None,
                cfg=decompose_cfg,
                clients=admitting,
                seed=0,
            )
        )
        is True
    )


def test_ft_b26b_on_topic_uses_its_own_minimal_schema_and_counts(decompose_cfg):
    from ragtime.pipeline.decompose.on_topic import ON_TOPIC_REJECTED
    from ragtime.serving import compile_schemas

    stats = Statistics()
    clients = make_bundle(on_topic={"rationale": "off topic", "on_topic": False})
    asyncio.run(
        on_topic("Q?", "PS", None, cfg=decompose_cfg, clients=clients, seed=0, stats=stats)
    )
    call = clients.llm.calls_for("on_topic_gate")[0]
    assert call.schema is compile_schemas().on_topic
    assert set(dict(call.schema.schema)["required"]) == {"rationale", "on_topic"}
    assert stats.value(ON_TOPIC_REJECTED) == 1.0


def test_on_topic_passages_are_context_the_nugget_need_not_be_answered_by_them(decompose_cfg):
    """The retrieved text is shown for topicality only.

    The gate stays a topic judgement; it never becomes a check that the passages already
    answer the nugget.
    """
    clients = make_bundle(on_topic={"rationale": "ok", "on_topic": True})
    asyncio.run(
        on_topic("Q?", "PS", "some retrieved text", cfg=decompose_cfg, clients=clients, seed=0)
    )
    prompt = clients.llm.calls_for("on_topic_gate")[0].prompt
    assert "some retrieved text" in prompt
    assert "does NOT have" in prompt


# --------------------------------------------------------------------------- #
# FT-B36..B39: saturation
# --------------------------------------------------------------------------- #
def test_ft_b36_fires_on_low_streak():
    assert saturated((5, 1, 0), min_new=1, low_streak=2) is True


def test_ft_b37_does_not_fire_above_min_new():
    assert saturated((5, 3, 4), min_new=1, low_streak=2) is False


def test_ft_b38_boundary_exact_min_new_and_short_history():
    assert saturated((1, 1), min_new=1, low_streak=2) is True, "<= is the boundary, not <"
    assert saturated((0,), min_new=1, low_streak=2) is False, "history shorter than the window"
    assert saturated((), min_new=1, low_streak=2) is False
    assert saturated((0, 5), min_new=1, low_streak=2) is False, "the streak must be the LAST k"


def test_ft_b39_reads_its_kwargs_not_module_constants():
    history = (2, 2, 2)
    assert saturated(history, min_new=1, low_streak=2) is False
    assert saturated(history, min_new=2, low_streak=2) is True
    assert saturated(history, min_new=2, low_streak=4) is False
    with pytest.raises(ValueError):
        saturated(history, min_new=1, low_streak=0)


# --------------------------------------------------------------------------- #
# FT-B40..B42: bank_fingerprint
# --------------------------------------------------------------------------- #
def test_ft_b40_fingerprint_excludes_weight_and_vital():
    a = (_n("t#n0", "Q0", weight=0.1, vital=False), _n("t#n1", "Q1", weight=0.2))
    b = (_n("t#n0", "Q0", weight=0.9, vital=True), _n("t#n1", "Q1", weight=0.8, vital=True))
    assert bank_ops.bank_fingerprint(a) == bank_ops.bank_fingerprint(b)


def test_ft_b40b_fingerprint_excludes_round_ge1_state():
    base = _n("t#n0", "Q0")
    grown = dataclasses.replace(
        base, status="answered", answers=(Answer(answer="x", sentence="X."),)
    )
    assert bank_ops.bank_fingerprint((base,)) == bank_ops.bank_fingerprint((grown,))


def test_ft_b41_fingerprint_is_order_independent():
    a = (_n("t#n0", "Q0"), _n("t#n1", "Q1"), _n("t#n2", "Q2"))
    assert bank_ops.bank_fingerprint(a) == bank_ops.bank_fingerprint(tuple(reversed(a)))


def test_ft_b42_fingerprint_changes_on_question_or_aggregator_diff():
    a = (_n("t#n0", "Q0"), _n("t#n1", "Q1"))
    assert bank_ops.bank_fingerprint(a) != bank_ops.bank_fingerprint(
        (a[0], dataclasses.replace(a[1], question="Q1 but different"))
    )
    assert bank_ops.bank_fingerprint(a) != bank_ops.bank_fingerprint(
        (a[0], dataclasses.replace(a[1], aggregator_type="and"))
    )
    assert bank_ops.bank_fingerprint(a) != bank_ops.bank_fingerprint(a[:1])


def test_fingerprint_reuses_the_one_canonical_hasher():
    """The hasher is ``config.hashing.config_hash``, not a second sha256 recipe."""
    from ragtime.config.hashing import config_hash

    a = (_n("t#n1", "Q1"), _n("t#n0", "Q0"))
    expected = config_hash(
        {
            "nuggets": [
                {"question": "Q0", "aggregator_type": "OR"},
                {"question": "Q1", "aggregator_type": "OR"},
            ]
        }
    )
    assert bank_ops.bank_fingerprint(a) == expected
    assert len(expected) == 64
