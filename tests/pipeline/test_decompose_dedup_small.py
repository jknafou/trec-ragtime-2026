"""``dedup_nuggets``: the paraphrase and embedding merge (FT-B28..B35).

The cascade under test: a generous cosine pre-filter proposes pairs, and the LLM and-gate
decides. Every knob is injected, so each test moves exactly one of them.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from ragtime.common import Answer, Nugget, Statistics
from ragtime.pipeline.decompose import dedup_nuggets
from ragtime.pipeline.decompose.dedup import DEDUP_MERGED, cosine

pytestmark = pytest.mark.small


def _n(nid: str, question: str, **kw) -> Nugget:
    return Nugget(nugget_id=nid, question=question, **kw)


def _unit(*xs: float) -> list[float]:
    norm = math.sqrt(sum(x * x for x in xs)) or 1.0
    return [x / norm for x in xs]


class _Embed:
    """An injected embedder returning a canned vector per question text."""

    def __init__(self, by_text: dict[str, list[float]]) -> None:
        self.by_text = by_text
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.by_text[t] for t in texts]


class _Confirm:
    def __init__(self, verdict: bool = True) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, kept: str, candidate: str) -> bool:
        self.calls.append((kept, candidate))
        return self.verdict


# Two hand-crafted paraphrases (near-identical vectors) plus one unrelated nugget.
Q_A = "Which agency issued the recall notice?"
Q_B = "Which regulator put out the recall notice?"
Q_C = "How many people were hospitalised?"
_VECTORS = {
    Q_A: _unit(1.0, 0.0, 0.0),
    Q_B: _unit(0.99, 0.14, 0.0),  # cosine(Q_A, Q_B) ~= 0.990
    Q_C: _unit(0.0, 0.0, 1.0),  # orthogonal to both
}
PAIR_BANK = (_n("t#n0", Q_A), _n("t#n1", Q_B), _n("t#n2", Q_C))


def _run(bank, *, cutoff=0.80, llm=True, confirm=None, embed=None, stats=None):
    return asyncio.run(
        dedup_nuggets(
            bank,
            cosine_cutoff=cutoff,
            llm_paraphrase_merge=llm,
            embed=embed or _Embed(_VECTORS),
            llm_confirm=confirm if confirm is not None else _Confirm(True),
            stats=stats,
        )
    )


def test_cosine_helper_is_a_plain_dot_product():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine(_VECTORS[Q_A], _VECTORS[Q_B]) == pytest.approx(0.990, abs=1e-3)


def test_ft_b28_merges_a_handcrafted_paraphrase_pair():
    stats = Statistics()
    out = _run(PAIR_BANK, stats=stats)
    assert [n.nugget_id for n in out] == ["t#n0", "t#n2"]
    assert stats.value(DEDUP_MERGED) == 1.0


def test_ft_b28b_llm_veto_keeps_both():
    """The cosine pre-filter only proposes; a `duplicate: false` verdict is decisive."""
    confirm = _Confirm(False)
    out = _run(PAIR_BANK, confirm=confirm)
    assert [n.nugget_id for n in out] == ["t#n0", "t#n1", "t#n2"]
    assert confirm.calls == [(Q_A, Q_B)]


def test_ft_b29_never_unmerge_append_only_growth():
    """FT-B29, identity monotonicity: a surviving id survives the next pass too."""
    first = _run(PAIR_BANK)
    survivors = {n.nugget_id for n in first}
    assert survivors == {"t#n0", "t#n2"}

    q_d = "What penalty was imposed?"
    q_e = "Which body put out the recall notice?"  # another paraphrase of Q_A
    vectors = dict(_VECTORS) | {q_d: _unit(0.0, 1.0, 0.0), q_e: _unit(0.98, 0.2, 0.0)}
    extended = (*first, _n("t#n3", q_d), _n("t#n4", q_e))

    second = _run(extended, embed=_Embed(vectors))
    ids = {n.nugget_id for n in second}
    assert survivors <= ids, "an id that survived round one must still exist"
    assert "t#n1" not in ids, "a merged-away id never resurfaces"
    assert "t#n4" not in ids, "the new paraphrase merges INTO the older nugget"


def test_ft_b30_cosine_cutoff_is_read_from_config_not_hardcoded():
    assert len(_run(PAIR_BANK, cutoff=0.99999)) == 3, "an unreachable cutoff merges nothing"
    assert len(_run(PAIR_BANK, cutoff=0.80)) == 2, "the shipped cutoff merges the paraphrase"
    # Q_C is orthogonal to both (cosine 0.0), so only a cutoff at/below 0 reaches it.
    assert len(_run(PAIR_BANK, cutoff=0.10)) == 2
    assert len(_run(PAIR_BANK, cutoff=-1.0)) == 1, "a floor cutoff merges everything"


def test_ft_b31_llm_paraphrase_merge_false_skips_llm_confirm():
    confirm = _Confirm(True)
    out = _run(PAIR_BANK, llm=False, confirm=confirm)
    assert confirm.calls == [], "cosine-only mode must never call the gate"
    assert [n.nugget_id for n in out] == ["t#n0", "t#n2"]


def test_llm_paraphrase_merge_true_without_a_gate_is_a_hard_error():
    with pytest.raises(ValueError, match="llm_confirm"):
        asyncio.run(
            dedup_nuggets(
                PAIR_BANK,
                cosine_cutoff=0.8,
                llm_paraphrase_merge=True,
                embed=_Embed(_VECTORS),
                llm_confirm=None,
            )
        )


def test_ft_b32_embed_is_injected_and_called_once_for_the_whole_bank():
    embed = _Embed(_VECTORS)
    _run(PAIR_BANK, embed=embed)
    assert len(embed.calls) == 1, "one batched embed call, never one per pair"
    assert embed.calls[0] == [Q_A, Q_B, Q_C]


def test_ft_b32b_embed_is_the_exact_callable_the_caller_sliced_from_the_bundle():
    """The identity chain: what ``grow_nuggets`` binds is what ``dedup`` invokes."""
    from functools import partial

    seen: list[str] = []

    class _Bundle:
        def embed(self, texts, mode="dense"):
            seen.append(mode)
            return [_VECTORS[t] for t in texts]

    bound = partial(_Bundle().embed, mode="dense")
    asyncio.run(
        dedup_nuggets(
            PAIR_BANK,
            cosine_cutoff=0.8,
            llm_paraphrase_merge=False,
            embed=bound,
            llm_confirm=None,
        )
    )
    assert seen == ["dense"], "the ONE resident dense embedder, mode pinned at the call site"


def test_merge_preserves_the_kept_nugget_and_inherits_its_evidence():
    answer = Answer(answer="the FDA", sentence="The FDA issued it.")
    bank = (
        _n("t#n0", Q_A, weight=0.9, vital=True),
        _n("t#n1", Q_B, weight=0.2, answers=(answer,)),
        _n("t#n2", Q_C),
    )
    out = _run(bank)
    kept = out[0]
    assert kept.nugget_id == "t#n0"
    assert kept.weight == 0.9 and kept.vital is True, "the survivor's own scores are kept"
    assert kept.answers == (answer,), "the merged-away nugget's evidence is not lost"


def test_short_bank_short_circuits_without_embedding():
    embed = _Embed(_VECTORS)
    assert _run((), embed=embed) == ()
    assert _run((_n("t#n0", Q_A),), embed=embed) == (_n("t#n0", Q_A),)
    assert embed.calls == []


def test_a_partial_embedding_batch_is_a_hard_error():
    def short(texts):
        return [_VECTORS[texts[0]]]

    with pytest.raises(ValueError, match="partial embedding batch"):
        _run(PAIR_BANK, embed=short)


def test_ragged_vectors_are_a_hard_error():
    def ragged(texts):
        return [[1.0, 0.0], [1.0], [0.0, 1.0]]

    with pytest.raises(ValueError, match="ragged"):
        _run(PAIR_BANK, embed=ragged)
