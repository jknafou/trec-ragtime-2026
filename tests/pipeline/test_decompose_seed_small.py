"""The round-0 seed path of ``grow_nuggets`` (FT-A1/A3/A4, FT-B1..B23, FT-C1/C3).

Everything here runs against the stub ``ClientBundle``: round 0 is retrieval-free by
construction, so no index, no rag loop and no GPU is involved. The real-model
counterparts live in ``test_decompose_full.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest

from ragtime.common import Nugget, Statistics
from ragtime.config import load
from ragtime.orchestration.determinism import expand_seeds
from ragtime.pipeline.decompose import bank as bank_ops
from ragtime.pipeline.decompose import fairness_anchor, grow_nuggets
from ragtime.pipeline.decompose.grow_nuggets import BANK_SIZE, THIN_SEED
from ragtime.serving import compile_schemas

from .conftest import SEED_QUESTIONS, make_bundle, nuggets_response

pytestmark = pytest.mark.small

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_A = _REPO_ROOT / "config" / "e2e-original.yml"
CONFIG_B = _REPO_ROOT / "config" / "e2e-omt.yml"


def _seed(clients, cfg, topic, *, stats=None, bank=(), passages=None, round=0):
    return asyncio.run(
        grow_nuggets(
            topic.problem_statement,
            topic.background,
            bank,
            passages,
            round,
            cfg=cfg,
            clients=clients,
            topic_id=topic.topic_id,
            limit=topic.limit,
            seed=7,
            stats=stats,
        )
    )


# --------------------------------------------------------------------------- #
# A. The blocking acceptance criteria
# --------------------------------------------------------------------------- #
def test_ft_a1_round0_bank_hash_equal_across_two_variants_same_seed(decompose_cfg, seed_topic):
    """FT-A1: the runtime fairness anchor, on the stub.

    Round 0 takes no variant-dependent input (no passages, no index), so two calls that
    differ only in the variant label must produce the same bank fingerprint.
    """
    banks = {}
    for variant in ("original", "omt"):
        decompose_cfg.passage_lang = variant
        clients = make_bundle(
            nuggets=[
                nuggets_response(SEED_QUESTIONS),
                nuggets_response(SEED_QUESTIONS, weights=[0.9, 0.7, 0.4, 0.2]),
            ]
        )
        banks[variant] = _seed(clients, decompose_cfg, seed_topic)

    assert bank_ops.bank_fingerprint(banks["original"]) == bank_ops.bank_fingerprint(banks["omt"])
    # ... and the anchor itself accepts them
    assert fairness_anchor.assert_seed_parity(banks, seed=7)


def test_ft_a3_assert_seed_parity_raises_on_genuine_divergence(decompose_cfg, seed_topic):
    """FT-A3: a fairness anchor that can never fail is decoration, not a test."""
    clients = make_bundle(
        nuggets=[
            nuggets_response(SEED_QUESTIONS),
            nuggets_response(SEED_QUESTIONS, weights=[0.9, 0.7, 0.4, 0.2]),
        ]
    )
    bank_a = _seed(clients, decompose_cfg, seed_topic)
    leaked = dataclasses.replace(bank_a[1], question="What did the TRANSLATED passage say?")
    bank_b = (bank_a[0], leaked, *bank_a[2:])

    with pytest.raises(fairness_anchor.SeedParityError) as exc:
        fairness_anchor.assert_seed_parity({"original": bank_a, "omt": bank_b})
    assert leaked.nugget_id in str(exc.value)
    assert "TRANSLATED" in str(exc.value)


def test_ft_a4_seed_decompose_and_stubbed_loop_share_one_client_bundle(
    real_run_config, seed_topic, monkeypatch
):
    """FT-A4: one ``ClientBundle``, obtained once and shared by every stage.

    ``build_clients`` is called exactly once here; the seed-decompose path and a stand-in for
    the RAG loop's ``run_loop`` both receive that object, and the stand-in returns the same
    object, compared by identity rather than equality.
    """
    import ragtime.serving.llm as serving_llm
    from ragtime.serving import build_clients

    calls: list = []

    async def fake_guided_json(schema, ctx):
        calls.append((schema, ctx))
        # one nugget => dedup short-circuits and the real (lazy) Encoder is never loaded
        return nuggets_response(["What legislation applies?"], weights=[0.9])

    monkeypatch.setattr(serving_llm, "guided_json", fake_guided_json)

    clients = build_clients(real_run_config)  # <- the one call

    bank = asyncio.run(
        grow_nuggets(
            seed_topic.problem_statement,
            seed_topic.background,
            (),
            None,
            0,
            cfg=real_run_config,
            clients=clients,
            topic_id=seed_topic.topic_id,
            limit=seed_topic.limit,
            seed=0,
        )
    )
    assert len(bank) == 1

    def fake_run_loop(nugget, cfg, bundle):
        """Stands in for the RAG loop's ``run_loop``: it never builds a client of its own."""
        assert nugget.nugget_id
        return bundle

    returned = fake_run_loop(bank[0], real_run_config, clients)
    assert returned is clients
    assert returned.llm is clients.llm
    assert returned.query_dense is clients.query_dense  # the field dedup actually uses
    assert returned.schemas is clients.schemas
    assert calls, "the seed path must have gone through serving.llm.guided_json"


# --------------------------------------------------------------------------- #
# B. grow_nuggets shape and round dispatch
# --------------------------------------------------------------------------- #
def test_ft_b1_round0_returns_nonempty_bank(stub_clients, decompose_cfg, seed_topic):
    bank = _seed(stub_clients, decompose_cfg, seed_topic)
    assert isinstance(bank, tuple)
    assert bank and all(isinstance(n, Nugget) for n in bank)


def test_ft_b2_round_ge1_refuses_evidence_from_nowhere(stub_clients, decompose_cfg, seed_topic):
    """FT-B2: from round 1 on, `grow_nuggets` takes the loops' own evidence and nothing else.

    The branch must decide it cannot proceed before spending a generation on the shared vLLM,
    which is what the second assertion pins. What it refuses is a passages argument that is not
    a `coverage_audit.Evidence`: under `audit_evidence: reuse` the audit reads only what the
    loops retrieved and answered, so a raw blob would be evidence from nowhere.
    """
    with pytest.raises(TypeError) as exc:
        _seed(stub_clients, decompose_cfg, seed_topic, round=1, passages=["p"])
    assert "Evidence" in str(exc.value)
    assert stub_clients.llm.calls == [], "it must refuse before issuing any LLM call"


def test_ft_b3_round0_rejects_non_none_passages(stub_clients, decompose_cfg, seed_topic):
    with pytest.raises(ValueError) as exc:
        _seed(stub_clients, decompose_cfg, seed_topic, passages=["a passage"])
    assert "retrieval-free" in str(exc.value)
    assert stub_clients.llm.calls == []


# --------------------------------------------------------------------------- #
# B. Seed prompt shape and self-critique
# --------------------------------------------------------------------------- #
def test_ft_b5_seed_prompt_orders_problem_statement_first_and_labels_it_primary():
    from ragtime.pipeline.decompose import seed_prompt

    text = seed_prompt("MARKER_PROBLEM", "MARKER_BACKGROUND", (4, 8))
    assert text.index("MARKER_PROBLEM") < text.index("MARKER_BACKGROUND")
    head = text[: text.index("MARKER_PROBLEM")]
    assert "primary" in head.lower()
    assert "spine" in head.lower()


def test_ft_b6_seed_prompt_requests_facet_enumeration_from_an_empty_bank():
    from ragtime.pipeline.decompose import seed_prompt

    text = seed_prompt("ps", "bg", (4, 8)).lower()
    assert "enumerate the distinct aspects" in text
    assert "bank is currently empty" in text
    # ... and not the later rounds' update-and-audit framing, which is a different prompt
    for audit_framing in (
        "already answered",
        "mark answered",
        "current bank",
        "carry the bank forward",
        "retrieved passages",
    ):
        assert audit_framing not in text, audit_framing


def test_ft_b7_self_critique_is_exactly_one_scoped_call(stub_clients, decompose_cfg, seed_topic):
    """FT-B7: one draft + one self-critique on the `nuggets` schema, scoped to 3 things."""
    _seed(stub_clients, decompose_cfg, seed_topic)
    nugget_calls = stub_clients.llm.calls_for("nuggets")
    assert len(nugget_calls) == 2, "exactly one draft + one self-critique"

    draft, critique = nugget_calls
    assert "enumerate the distinct aspects" in draft.prompt.lower()
    body = critique.prompt.lower()
    assert "merge near-duplicates" in body
    assert "coverage" in body
    assert "weight" in body
    assert "do exactly three things and nothing else" in body
    for open_ended in ("reconsider your answer", "rethink", "re-argue your reasoning"):
        assert open_ended not in body


# --------------------------------------------------------------------------- #
# B. Emission reuse
# --------------------------------------------------------------------------- #
def test_ft_b9_never_passes_max_tokens(stub_clients, decompose_cfg, seed_topic):
    """FT-B9, behaviourally and structurally: ``GenCtx`` has no ``max_tokens`` field."""
    from ragtime.serving import GenCtx

    _seed(stub_clients, decompose_cfg, seed_topic)
    assert stub_clients.llm.calls
    for call in stub_clients.llm.calls:
        assert "max_tokens" not in call.kwargs
    assert "max_tokens" not in {f.name for f in dataclasses.fields(GenCtx)}


def test_ft_b10_draft_call_uses_the_compiled_nuggets_schema(
    stub_clients, decompose_cfg, seed_topic
):
    _seed(stub_clients, decompose_cfg, seed_topic)
    draft = stub_clients.llm.calls_for("nuggets")[0]
    assert draft.schema is compile_schemas().nuggets


def test_ft_b11_nuggets_schema_requires_a_leading_rationale_field():
    schema = dict(compile_schemas().nuggets.schema)
    props = list(schema["properties"])
    assert props[0] == "rationale"
    assert props.index("rationale") < props.index("nuggets")
    assert set(schema["required"]) >= {"rationale", "nuggets"}


def test_ft_b12_decomposition_output_value_is_advisory_only(
    stub_clients, decompose_cfg, seed_topic
):
    """FT-B12: ``decomposition.output`` never branches behaviour."""
    baseline = _seed(stub_clients, decompose_cfg, seed_topic)
    baseline_calls = [(c.name, c.prompt) for c in stub_clients.llm.calls]

    other = make_bundle(
        nuggets=[
            nuggets_response(SEED_QUESTIONS),
            nuggets_response(SEED_QUESTIONS, weights=[0.9, 0.7, 0.4, 0.2]),
        ]
    )
    decompose_cfg.blocks["decomposition"]["output"] = "not_a_real_output_mode"
    after = _seed(other, decompose_cfg, seed_topic)

    assert bank_ops.bank_fingerprint(after) == bank_ops.bank_fingerprint(baseline)
    assert [(c.name, c.prompt) for c in other.llm.calls] == baseline_calls


# --------------------------------------------------------------------------- #
# B. Decoding passthrough
# --------------------------------------------------------------------------- #
def test_ft_b15_decoding_kwargs_come_from_config(stub_clients, decompose_cfg, seed_topic):
    decompose_cfg.blocks["decomposition"]["decoding"] = {
        "temperature": 0.42,
        "top_p": 0.55,
        "top_k": 7,
    }
    _seed(stub_clients, decompose_cfg, seed_topic)
    assert stub_clients.llm.calls
    for call in stub_clients.llm.calls:
        assert call.kwargs["temperature"] == 0.42
        assert call.kwargs["top_p"] == 0.55
        assert call.kwargs["top_k"] == 7


def test_ft_b15b_real_config_decoding_reaches_the_gen_ctx(
    real_run_config, seed_topic, monkeypatch
):
    """The same passthrough end-to-end through the real ``GenCtx`` (no hardcoded knobs)."""
    import ragtime.serving.llm as serving_llm
    from ragtime.serving import build_clients

    seen: list = []

    async def fake_guided_json(schema, ctx):
        seen.append(ctx)
        return nuggets_response(["What legislation applies?"], weights=[0.5])

    monkeypatch.setattr(serving_llm, "guided_json", fake_guided_json)
    clients = build_clients(real_run_config)
    asyncio.run(
        grow_nuggets(
            seed_topic.problem_statement,
            seed_topic.background,
            (),
            None,
            0,
            cfg=real_run_config,
            clients=clients,
            topic_id=seed_topic.topic_id,
            limit=seed_topic.limit,
            seed=3,
        )
    )
    decoding = real_run_config.blocks["decomposition"]["decoding"]
    assert seen
    for ctx in seen:
        assert ctx.temperature == decoding["temperature"]
        assert ctx.top_p == decoding["top_p"]
        assert ctx.top_k == decoding["top_k"]
        assert ctx.seed == 3


def test_ft_b16_seed_is_pinned_into_every_call(stub_clients, decompose_cfg, seed_topic):
    _seed(stub_clients, decompose_cfg, seed_topic)
    assert stub_clients.llm.calls
    assert {c.seed for c in stub_clients.llm.calls} == {7}


# --------------------------------------------------------------------------- #
# B19: the k band is a hint, never a clamp
# --------------------------------------------------------------------------- #
def test_ft_b19_seed_bank_is_never_clamped_to_the_k_band_over_or_under(
    decompose_cfg, seed_topic
):
    """FT-B19: 20 proposed nuggets are not sliced to 8; 2 are not padded up to 4."""
    decompose_cfg.blocks["decomposition"]["k_band"] = [[10000, 4, 8]]

    over = [f"What is fact number {i}?" for i in range(20)]
    clients_over = make_bundle(
        nuggets=[nuggets_response(over), nuggets_response(over, weights=[0.5] * 20)]
    )
    bank_over = _seed(clients_over, decompose_cfg, seed_topic)
    assert len(bank_over) == 20, "the LLM's 20 nuggets must survive an 8-wide band"

    under = ["What is the single fact?", "Who reported it?"]
    clients_under = make_bundle(
        nuggets=[nuggets_response(under), nuggets_response(under, weights=[0.9, 0.8])]
    )
    bank_under = _seed(clients_under, decompose_cfg, seed_topic)
    assert len(bank_under) == 2, "a 2-nugget bank must not be padded up to k_min"


def test_ft_b19b_band_is_rendered_into_the_prompt_as_a_hint(
    stub_clients, decompose_cfg, seed_topic
):
    _seed(stub_clients, decompose_cfg, seed_topic)
    draft = stub_clients.llm.calls_for("nuggets")[0].prompt
    # every released 2026 topic has limit 5000 -> the middle band (8, 14)
    assert "at least 8" in draft
    assert "14" in draft
    assert "not a quota" in draft


# --------------------------------------------------------------------------- #
# B20: thin and degenerate seeds
# --------------------------------------------------------------------------- #
def test_ft_b20_thin_problem_statement_falls_back_to_kmin_and_flags(
    stub_clients, decompose_cfg, seed_topic
):
    stats = Statistics()
    thin = "Report on floods."
    asyncio.run(
        grow_nuggets(
            thin,
            seed_topic.background,
            (),
            None,
            0,
            cfg=decompose_cfg,
            clients=stub_clients,
            topic_id=seed_topic.topic_id,
            limit=seed_topic.limit,
            seed=7,
            stats=stats,
        )
    )
    assert stats.value(THIN_SEED, round=0) == 1.0
    draft = stub_clients.llm.calls_for("nuggets")[0].prompt
    assert "at least 8" in draft
    assert "around 8" in draft, "a thin seed collapses the band onto k_min"


def test_ft_b20b_full_problem_statement_does_not_flag_thin(
    stub_clients, decompose_cfg, seed_topic
):
    stats = Statistics()
    _seed(stub_clients, decompose_cfg, seed_topic, stats=stats)
    assert stats.value(THIN_SEED, round=0) == 0.0


# --------------------------------------------------------------------------- #
# B23 / B43 / B44 / B45: the seed nugget's own record shape
# --------------------------------------------------------------------------- #
def test_ft_b23_round0_nuggets_default_to_or_aggregator(stub_clients, decompose_cfg, seed_topic):
    bank = _seed(stub_clients, decompose_cfg, seed_topic)
    assert {n.aggregator_type for n in bank} == {"OR"}


def test_ft_b43_origin_round_zero_and_trigger_passage_none(
    stub_clients, decompose_cfg, seed_topic
):
    bank = _seed(stub_clients, decompose_cfg, seed_topic)
    assert all(n.origin_round == 0 for n in bank)
    assert all(n.trigger_passage_id is None for n in bank)


def test_ft_b44_bank_size_metric_emitted_once(stub_clients, decompose_cfg, seed_topic):
    stats = Statistics()
    bank = _seed(stub_clients, decompose_cfg, seed_topic, stats=stats)
    assert stats.value(BANK_SIZE, round=0, variant="original", seed=7) == float(len(bank))
    assert stats.total(BANK_SIZE) == float(len(bank)), "emitted exactly once"


def test_ft_b45_round0_nugget_shape_matches_the_io_schema(
    stub_clients, decompose_cfg, seed_topic
):
    bank = _seed(stub_clients, decompose_cfg, seed_topic)
    for n in bank:
        assert isinstance(n, Nugget)
        assert n.status == "unanswered"
        assert n.retrieved == ()
        assert n.answers == ()
        assert n.nugget_id.startswith(f"{seed_topic.topic_id}#n")
        assert 0.0 <= n.weight <= 1.0


# --------------------------------------------------------------------------- #
# B47 / B48: config guardrails
# --------------------------------------------------------------------------- #
def test_ft_b47_retrieval_access_seed_true_raises_before_any_llm_call(
    stub_clients, decompose_cfg, seed_topic
):
    decompose_cfg.blocks["decomposition"]["retrieval_access"]["seed"] = True
    with pytest.raises(ValueError) as exc:
        _seed(stub_clients, decompose_cfg, seed_topic)
    assert "retrieval_access.seed" in str(exc.value)
    assert stub_clients.llm.calls == []


def test_ft_b48_unknown_grounding_gate_raises(stub_clients, decompose_cfg, seed_topic):
    decompose_cfg.blocks["decomposition"]["grounding_gate"] = "something_else"
    with pytest.raises(ValueError) as exc:
        _seed(stub_clients, decompose_cfg, seed_topic)
    assert "grounding_gate" in str(exc.value)
    assert stub_clients.llm.calls == []


# --------------------------------------------------------------------------- #
# B49: the `background` on/off ablation
# --------------------------------------------------------------------------- #
def test_ft_b49_seed_prompt_background_false_omits_section_without_crash():
    from ragtime.pipeline.decompose import seed_prompt

    for background in ("", "   ", None):
        text = seed_prompt("MARKER_PROBLEM", background, (4, 8))
        assert "MARKER_PROBLEM" in text
        assert "BACKGROUND (context/persona" not in text


def test_ft_b49b_background_false_config_threads_through_to_the_prompt(
    stub_clients, decompose_cfg, seed_topic
):
    decompose_cfg.blocks["decomposition"]["background"] = False
    _seed(stub_clients, decompose_cfg, seed_topic)
    draft = stub_clients.llm.calls_for("nuggets")[0].prompt
    assert seed_topic.background not in draft
    assert seed_topic.problem_statement in draft


# --------------------------------------------------------------------------- #
# C. Cross-stage I/O: producers into decompose, consumers of decompose
# --------------------------------------------------------------------------- #
def test_ft_c1_real_topic_and_real_config_feed_grow_nuggets_without_reshaping(
    small_topics, real_run_config
):
    """FT-C1: a real ``Topic`` and a real ``RunConfig`` wire straight in, with no renaming."""
    topic = small_topics[1]
    clients = make_bundle(
        nuggets=[
            nuggets_response(SEED_QUESTIONS),
            nuggets_response(SEED_QUESTIONS, weights=[0.9, 0.7, 0.4, 0.2]),
        ]
    )
    bank = asyncio.run(
        grow_nuggets(
            topic.problem_statement,
            topic.background,
            (),
            None,
            0,
            cfg=real_run_config,
            clients=clients,
            topic_id=topic.topic_id,
            limit=topic.limit,
            seed=0,
        )
    )
    assert bank
    assert all(n.nugget_id.startswith(f"{topic.topic_id}#n") for n in bank)


def test_ft_c3_round0_bank_is_a_valid_round_ge1_bank_argument(
    stub_clients, decompose_cfg, seed_topic
):
    """FT-C3: the exact ``bank`` shape the round loop's later rounds receive."""
    bank = _seed(stub_clients, decompose_cfg, seed_topic)
    assert type(bank) is tuple
    for n in bank:
        assert isinstance(n, Nugget)
        # Frozen and slotted, so `dataclasses.replace` is the accumulation mechanism.
        grown = dataclasses.replace(n, status="answered")
        assert grown.status == "answered" and n.status == "unanswered"
        with pytest.raises(dataclasses.FrozenInstanceError):
            n.status = "answered"


# --------------------------------------------------------------------------- #
# The equal-budget precondition for a run family
# --------------------------------------------------------------------------- #
def test_a_family_runs_the_same_seeds_equal_budget():
    """Both siblings of a run family expand to the same seed list, and it is ``cfg.seeds`` long.

    This is a property of the two config files alone: no model, no endpoint, no card, no client
    bundle, so it belongs in this tier rather than inside a full test that makes real
    decompositions.

    It is a fairness precondition. If the two arms ran different seeds they would not have equal
    budget, and every cross-variant comparison downstream would be confounded. FT-A2 and FT-D2
    assert that the arms agree; this asserts they were asked the same thing in the first place.
    """
    cfg_a, cfg_b = load(CONFIG_A), load(CONFIG_B)
    seeds = expand_seeds(cfg_a)
    assert seeds == expand_seeds(cfg_b), "a family runs the same seeds (equal budget)"
    assert len(seeds) == cfg_a.seeds
