"""Full gate for the reranker: the real ``Qwen/Qwen3-Reranker-4B`` on a GPU.

This is the tier the defect needed. The shipped code scored with
``sentence_transformers.CrossEncoder``, which loads the checkpoint as
``AutoModelForSequenceClassification``: the LM head is discarded and ``score.weight`` is
freshly random, so the reranker returned a random permutation of the pool and raised nothing
while doing it. No hermetic test can see that, because the numbers have the right type, the
right shape and the right range. Only the real weights say whether the ranking means
anything.

Asserted here (each one fails against the random head):

- FL-RR1 two independent loads in one process return byte-identical scores. A random head
  re-initialises per load, so its two score vectors disagree.
- FL-RR2 a relevant passage outranks irrelevant ones (the model is actually reading).
- FL-RR3 every score is finite, ``<= 0`` (a log-probability), and no two are equal: the
  regression guard for returning ``exp`` instead of ``log``: under ``exp`` the irrelevant tail
  all scored exactly ``0.0``, which ties it and hands its ordering to ``retrieval._rerank``'s
  ``(-score, passage_id)`` tie-break instead of to the model.
- FL-RR4 batching does not change the ranking: the proof that left padding is in force
  (with right padding, a batched short row is scored at a pad position).
- FL-RR5 the scores are unchanged in a process whose transformers weight tying has been
  globally neutered. A leaked ``no_tie_weights()`` patch left this checkpoint's tied
  ``lm_head.weight`` stranded on the ``meta`` device, and loading with ``device_map`` alone
  would pass FL-RR1 and FL-RR4 while scoring every passage at ``log 0.5``, so this case gets
  its own test.

These skip outside a GPU allocation and must run inside one. A skip is not a pass.
"""

from __future__ import annotations

import math
import os

import pytest

from ragtime.serving.reranker import Reranker

pytestmark = pytest.mark.full

_RERANKER_ID = "Qwen/Qwen3-Reranker-4B"

_QUERY = "What health effects does air pollution have on city residents?"
#: index 0 is the only passage that answers the query; 1 is same-topic-but-not-an-answer;
#: 2 and 3 are unrelated. Not in a ranking-friendly order.
_PASSAGES = [
    (
        "Long-term exposure to fine particulate matter in cities raises the risk of asthma, "
        "chronic bronchitis, heart disease and stroke among residents, and is estimated to "
        "cause millions of premature deaths worldwide each year."
    ),
    (
        "The city council voted on Tuesday to extend the tram network by two stops, a project "
        "financed jointly by the canton and the federal transport office."
    ),
    (
        "Sourdough fermentation relies on a stable culture of wild yeasts and lactobacilli; "
        "bakers refresh the starter with flour and water at room temperature."
    ),
    (
        "The 1994 world championship final went to a penalty shoot-out after 120 goalless "
        "minutes in front of a crowd of more than ninety thousand."
    ),
]


def _rank(scores: list[float]) -> list[int]:
    """Candidate indices best-first, under ``retrieval._rerank``'s own ``(-score, id)`` order."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def _require_gpu() -> None:
    """Skip unless this is an allocation with a visible CUDA device and the heavy extras."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.skip("not in a SLURM allocation (no $SLURM_JOB_ID)")
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device")


def _score_once(query: str, passages: list[str], *, batch_size: int = 8) -> tuple[list[float], str]:
    """One independent load + score; returns the scores and the loaded model's class name.

    The model is dropped and the CUDA cache emptied before returning, so two loads cost one
    resident copy (~7.5 GiB of bf16 weights) rather than two.
    """
    import torch

    rr = Reranker(model=_RERANKER_ID, device="cuda", batch_size=batch_size)
    scores = rr.score(query, passages)
    klass = type(rr._model).__name__
    # `lm_head` is the head this whole fix is about: a sequence-classification load has `score`
    # (randomly initialised) instead, and would fail here rather than silently rank at random.
    assert hasattr(rr._model, "lm_head"), f"{klass} has no lm_head: wrong head loaded"
    rr._model = rr._tok = None
    torch.cuda.empty_cache()
    return scores, klass


@pytest.fixture(scope="module")
def two_loads() -> tuple[list[float], list[float], str]:
    _require_gpu()
    first, klass = _score_once(_QUERY, _PASSAGES)
    second, _ = _score_once(_QUERY, _PASSAGES)
    return first, second, klass


# --------------------------------------------------------------------------- #
# FL-RR1: determinism across two independent loads
# --------------------------------------------------------------------------- #
def test_flrr1_two_independent_loads_score_identically(
    two_loads: tuple[list[float], list[float], str],
) -> None:
    first, second, klass = two_loads
    assert klass.endswith("ForCausalLM"), f"loaded {klass}, not the checkpoint's causal LM"
    assert first == second, (
        "two loads of the same checkpoint disagree: the scoring head is not the checkpoint's "
        f"own weights: {first} vs {second}"
    )


# --------------------------------------------------------------------------- #
# FL-RR2: the model is actually reading the passages
# --------------------------------------------------------------------------- #
def test_flrr2_the_relevant_passage_ranks_first(
    two_loads: tuple[list[float], list[float], str],
) -> None:
    scores = two_loads[0]
    assert _rank(scores)[0] == 0, f"the answering passage did not rank first: {scores}"
    assert scores[0] > max(scores[1:]), scores


# --------------------------------------------------------------------------- #
# FL-RR3: log-probabilities, and a tail that is ORDERED rather than tied at zero
# --------------------------------------------------------------------------- #
def test_flrr3_scores_are_finite_log_probs_with_no_ties(
    two_loads: tuple[list[float], list[float], str],
) -> None:
    scores = two_loads[0]
    assert all(math.isfinite(s) for s in scores), scores
    assert all(s <= 0.0 for s in scores), f"not log-probabilities: {scores}"
    assert len(set(scores)) == len(scores), (
        f"tied scores {scores}: under `exp` the irrelevant tail collapses to 0.0 and the "
        "passage-id tie-break, not the model, decides its order"
    )


# --------------------------------------------------------------------------- #
# FL-RR4: batching (hence left padding) does not move the ranking
# --------------------------------------------------------------------------- #
def test_flrr4_batch_size_does_not_change_the_ranking(
    two_loads: tuple[list[float], list[float], str],
) -> None:
    batched = two_loads[0]  # batch_size 8 -> one batch of 4 (padded rows)
    unbatched, _ = _score_once(_QUERY, _PASSAGES, batch_size=1)  # every row alone, no padding
    assert _rank(batched) == _rank(unbatched), (
        f"batching changed the ranking ({batched} vs {unbatched}): the last position of a "
        "padded row is not the passage's last token, i.e. padding is not on the left"
    )


# --------------------------------------------------------------------------- #
# FL-RR5: the checkpoint's own head, even in a process with neutered weight tying
# --------------------------------------------------------------------------- #
def test_flrr5_scores_survive_neutered_weight_tying(
    two_loads: tuple[list[float], list[float], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-create that process state and demand the same scores.

    ``transformers.initialization.no_tie_weights()`` disables tying by monkeypatching the class
    attribute ``PreTrainedModel.tie_weights`` for the duration of every ``from_pretrained`` and
    restoring it in a ``finally``: a process-global, non-re-entrant patch that two overlapping
    loads (e.g. from a ``ThreadPoolExecutor``) can leave permanently neutered. This checkpoint
    ships ``tie_word_embeddings: true`` and no ``lm_head.weight``, so in such a process the
    whole scoring head lands on the ``meta`` device. Load-then-``.to()`` raised
    ``NotImplementedError: Cannot copy out of meta tensor``, and a naive ``device_map`` fix
    instead returned ``log 0.5`` for every passage: silent, and invisible to FL-RR1 and
    FL-RR4, which is why this asserts against the clean baseline rather than against merely
    not crashing.

    The patch is applied inside the test body, so the ``two_loads`` baseline (fixture setup) is
    taken in a clean process, and ``monkeypatch`` restores tying for every later test.
    """
    _require_gpu()
    from transformers.modeling_utils import PreTrainedModel

    monkeypatch.setattr(PreTrainedModel, "tie_weights", lambda *a, **k: None, raising=True)
    neutered, klass = _score_once(_QUERY, _PASSAGES)
    assert klass.endswith("ForCausalLM"), f"loaded {klass}, not the checkpoint's causal LM"
    assert neutered == two_loads[0], (
        "a process with neutered weight tying scored differently from a clean one "
        f"({neutered} vs {two_loads[0]}): the head that scored is not the checkpoint's"
    )
    assert _rank(neutered)[0] == 0, f"the answering passage did not rank first: {neutered}"
    assert len(set(neutered)) == len(neutered), (
        f"tied scores {neutered}: a meta/uninitialised head collapses every passage to the same "
        "log 0.5 and hands the pool's order to the passage-id tie-break"
    )
