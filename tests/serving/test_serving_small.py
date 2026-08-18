"""CPU-only tests of the serving package: registry singletons, the ``guided_json`` emission
discipline, ``compile_schemas`` and the pinned null-versus-absent shapes, and the public API
name contract. Everything runs against stubs, and no test imports torch or vLLM."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap

import jsonschema
import pytest

from ragtime.serving import (
    GenCtx,
    GuidedJsonError,
    build_stub_clients,
    compile_schemas,
    guided_json,
)
from ragtime.serving.schemas import (
    MODE_RESPONSE_FORMAT,
    MODE_STRUCTURED,
    CompiledSchema,
)

pytestmark = pytest.mark.small


def _run(coro):
    return asyncio.run(coro)


def _ctx(client, **kw):
    kw.setdefault("model", "stub-model")
    kw.setdefault("prompt", "hello")
    kw.setdefault("seed", 1234)
    return GenCtx(client=client, **kw)


# --------------------------------------------------------------------------- #
# A. Registry: per-node singletons.
# --------------------------------------------------------------------------- #
def test_build_clients_llm_singleton_identity():
    bundle = build_stub_clients(llm_model="m")
    decompose_site = bundle.llm
    loop_site = bundle.llm
    assert decompose_site is loop_site  # one shared vLLM client (invariant 3)


def test_build_clients_embedder_singleton_identity():
    bundle = build_stub_clients()
    assert bundle.embedder is bundle.embedder  # retrieval + dedup share one encoder


def test_build_clients_two_independent_builds_are_not_forced_equal():
    a = build_stub_clients()
    b = build_stub_clients()
    assert a.llm is not b.llm  # per-build singleton, not a global process singleton


def test_build_clients_model_identity_from_config_no_hardcode():
    b1 = build_stub_clients(llm_model="model-A")
    b2 = build_stub_clients(llm_model="model-B")
    assert b1.llm.model == "model-A"
    assert b2.llm.model == "model-B"


# --------------------------------------------------------------------------- #
# B. guided_json wrapper: the whole emission discipline.
# --------------------------------------------------------------------------- #
_ACTION = compile_schemas().action
_GOOD_ACTION = json.dumps({"rationale": "r", "action": "abstain"})
_BAD = "{ this is not valid json"


def test_guided_json_valid_response_validates_and_returns_dict(make_stub_client):
    client = make_stub_client([_GOOD_ACTION])
    obj = _run(guided_json(_ACTION, _ctx(client)))
    assert isinstance(obj, dict)
    jsonschema.validate(obj, dict(_ACTION.schema))


def test_guided_json_invalid_then_valid_retries_and_succeeds(make_stub_client):
    client = make_stub_client([_BAD, _GOOD_ACTION])
    obj = _run(guided_json(_ACTION, _ctx(client, max_attempts=3)))
    assert obj["action"] == "abstain"
    assert len(client.calls) == 2  # one retry


def test_guided_json_exhausts_retries_then_raises(make_stub_client):
    client = make_stub_client([_BAD, _BAD, _BAD, _BAD])
    with pytest.raises(GuidedJsonError):
        _run(guided_json(_ACTION, _ctx(client, max_attempts=3)))
    assert len(client.calls) == 3  # exactly the bound, never forever


def test_guided_json_never_passes_max_tokens(make_stub_client):
    client = make_stub_client([_BAD, _GOOD_ACTION])
    _run(guided_json(_ACTION, _ctx(client, max_attempts=3)))
    assert all("max_tokens" not in kw for kw in client.calls)


def test_guided_json_seed_threaded_into_request(make_stub_client):
    client = make_stub_client([_BAD, _GOOD_ACTION])
    _run(guided_json(_ACTION, _ctx(client, seed=4242, max_attempts=3)))
    assert client.calls and all(kw["seed"] == 4242 for kw in client.calls)


def test_guided_json_enable_thinking_false_in_extra_body(make_stub_client):
    client = make_stub_client([_GOOD_ACTION])
    _run(guided_json(_ACTION, _ctx(client)))
    kw = client.calls[0]
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_guided_json_uses_response_format_or_structured_outputs_not_dead_field(make_stub_client):
    client = make_stub_client([_GOOD_ACTION])
    _run(guided_json(_ACTION, _ctx(client)))  # action -> structured_outputs shape
    kw = client.calls[0]
    has_struct = "structured_outputs" in kw["extra_body"]
    has_rf = "response_format" in kw
    assert has_struct or has_rf
    assert "guided_json" not in kw and "guided_decoding_backend" not in kw

    # A response_format (strict) schema takes the other branch.
    flat = compile_schemas().dedup
    good_dedup = json.dumps(
        {"duplicate": False, "paraphrase_match": False, "entity_match": False, "reason": None}
    )
    client2 = make_stub_client([good_dedup])
    _run(guided_json(flat, _ctx(client2)))
    kw2 = client2.calls[0]
    assert kw2["response_format"]["type"] == "json_schema"
    assert kw2["response_format"]["json_schema"]["strict"] is True
    assert "guided_json" not in kw2


def _run_isolated(body: str) -> None:
    """Run ``body`` in a CLEAN interpreter (torch may be installed for the full
    gate, so an in-process ``sys.modules`` check is order-fragile: isolate it)."""
    script = "import sys\n" + textwrap.dedent(body)
    r = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr


def test_guided_json_never_instantiates_a_model():
    # In a clean interpreter, importing serving + running guided_json against a fake
    # client must not pull torch/vllm (guided_json is a pure client call, not a loader).
    _run_isolated(
        """
        import asyncio, json
        from ragtime.serving import GenCtx, compile_schemas, guided_json
        class _C:
            def __init__(self):
                self.chat = type("x", (), {"completions": self})()
            async def create(self, **k):
                content = json.dumps({"rationale": "r", "action": "abstain"})
                msg = type("m", (), {"content": content})()
                return type("r", (), {"choices": [type("c", (), {"message": msg})()]})()
        obj = asyncio.run(guided_json(
            compile_schemas().action,
            GenCtx(client=_C(), model="m", prompt="p", seed=1),
        ))
        assert obj["action"] == "abstain"
        assert "torch" not in sys.modules, "guided_json pulled torch"
        assert "vllm" not in sys.modules, "guided_json pulled vllm"
        """
    )


# --------------------------------------------------------------------------- #
# C. compile_schemas: compiled once, fixed set, null-vs-absent semantics.
# --------------------------------------------------------------------------- #
def test_compile_schemas_called_twice_returns_same_object():
    assert compile_schemas() is compile_schemas()


def test_each_schema_validates_known_good_payload(good_payloads):
    schemas = compile_schemas().by_name()
    for name, compiled in schemas.items():
        jsonschema.validate(good_payloads[name], dict(compiled.schema))


def test_each_schema_rejects_known_bad_payload(bad_payloads):
    schemas = compile_schemas().by_name()
    for name, compiled in schemas.items():
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad_payloads[name], dict(compiled.schema))


def test_action_union_uses_structured_outputs_shape_with_true_optional_fields():
    action = compile_schemas().action
    assert action.mode == MODE_STRUCTURED
    # a `search` payload that OMITS submit_claim's `passage_id` entirely validates:
    # absent key means inapplicable (no None-check needed downstream).
    payload = {"rationale": "r", "action": "search", "query": "q"}
    jsonschema.validate(payload, dict(action.schema))
    assert "passage_id" not in payload


def test_flat_schema_uses_response_format_strict_shape_with_nullable_not_absent():
    dedup = compile_schemas().dedup
    assert dedup.mode == MODE_RESPONSE_FORMAT
    base = {"duplicate": True, "paraphrase_match": True, "entity_match": True}
    # OMITTING the inapplicable `reason` is INVALID (strict: every prop required)...
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(base), dict(dedup.schema))
    # ...the same payload with `reason: null` validates (nullable, not absent).
    jsonschema.validate({**base, "reason": None}, dict(dedup.schema))


# --------------------------------------------------------------------------- #
# J. Cross-stage I/O: public names.
# --------------------------------------------------------------------------- #
def test_public_api_names_stable():
    import ragtime.serving as s

    for name in (
        "build_clients",
        "ClientBundle",
        "guided_json",
        "compile_schemas",
        "discover_nodes",
        "capacity_for",
    ):
        assert hasattr(s, name), name


def test_import_serving_is_heavy_free():
    # In a fresh interpreter, importing the package + the light entrypoints must not
    # pull any heavy lib (A7: lazy backends).
    _run_isolated(
        """
        import ragtime.serving
        from ragtime.serving import compile_schemas, build_stub_clients, capacity_for
        compile_schemas()
        build_stub_clients()
        for heavy in ("torch", "vllm", "transformers", "sentence_transformers", "FlagEmbedding"):
            assert heavy not in sys.modules, f"import ragtime.serving pulled {heavy}"
        """
    )


def test_ad_hoc_mapping_schema_defaults_to_response_format(make_stub_client):
    # a bare dict schema (not a CompiledSchema) is treated as strict response_format.
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    assert not isinstance(schema, CompiledSchema)
    client = make_stub_client([json.dumps({"x": 1})])
    obj = _run(guided_json(schema, _ctx(client)))
    assert obj == {"x": 1}
    assert "response_format" in client.calls[0]


# --------------------------------------------------------------------------- #
# Fast twins for the harness in tests/serving/test_serving_full.py.
#
# The two helpers below decide whether that module runs at all, and one of them encodes the
# rule that a job-referenced file on node-local /tmp is a silent no-op. Neither depends on
# scale, so both are answered here rather than inside a GPU allocation.
# --------------------------------------------------------------------------- #
def test_the_full_gates_home_not_tmp_guard_is_a_real_predicate() -> None:
    """``_assert_on_home`` must reject node-local ``/tmp`` and accept ``/home``, both directions.

    A guard that accepted everything would restore the failure it exists to stop: a job that
    names a file under the node's own ``/tmp`` writes it where nothing else can read it, and
    the job still looks like it worked.
    """
    from tests.serving import test_serving_full as full

    with pytest.raises(AssertionError, match="node-local"):
        full._assert_on_home("/tmp/vllm.log")
    full._assert_on_home(full._home_scratch() / "vllm.log")
    assert not str(full._home_scratch()).startswith("/tmp")


def test_the_full_gates_gpu_precondition_skips_only_where_the_tier_has_no_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_require_gpu_allocation`` skips with no allocation: it must not fail, nor pass.

    That module's contract is asymmetric. No allocation is a tier limit and skips, while a
    visible CUDA device that cannot import vLLM is a broken environment and fails. Only the
    first branch is reachable without a GPU, and it is the branch that decides whether the
    whole module is inert, so it is the one pinned here.
    """
    from tests.serving import test_serving_full as full

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(pytest.skip.Exception, match="SLURM_JOB_ID"):
        full._require_gpu_allocation()
