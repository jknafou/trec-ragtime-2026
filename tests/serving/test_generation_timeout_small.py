"""The wall-clock bound on one generation call: a harness test, no GPU.

The defect this guards is a degenerate generation that ran well past ``REQUEST_TIMEOUT_S``
and raised nothing, because that timeout is an httpx read timeout measuring the gap between
reads, which a trickling response resets forever. The server was still reporting
``Running: 1 reqs`` throughout.

It matters beyond one slow call. The RAG loop's termination guarantee is that the monotonic
counters alone bound the loop, and that holds only if each turn returns. A turn that never
comes back cannot reach the next backstop check, so an unbounded call makes the guarantee
vacuous in exactly the case it exists for.

These are harness tests: a stub client that sleeps proves the mechanism in 0.1 s rather than
needing a real degenerate generation. The full tier asserts the complementary fact a stub
cannot show, that the shipped client carries a bound
(``test_ragloop_full.py::test_flm07_05``).
"""

from __future__ import annotations

import asyncio
import types
from typing import Any

import pytest

from ragtime.serving.llm import GenCtx, GenerationTimeout, LlmClient, guided_json

pytestmark = pytest.mark.small

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


class SleepyClient:
    """A client whose ``create`` takes ``delay`` seconds. Stands in for a degenerate generation.

    Sleeps rather than trickling bytes: the bound under test is total elapsed, so a
    stub that merely delayed between reads would be testing httpx, not us.
    """

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **_: Any) -> Any:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"ok": true}'))]
        )


def _ctx(client: Any, **kw: Any) -> GenCtx:
    return GenCtx(client=client, model="m", prompt="p", seed=1, **kw)


def test_a_call_that_overruns_its_bound_raises_instead_of_hanging():
    client = SleepyClient(30.0)
    with pytest.raises(GenerationTimeout) as exc:
        asyncio.run(guided_json(SCHEMA, _ctx(client, call_timeout_s=0.2)))
    assert "wall-clock" in str(exc.value)
    assert "not the httpx" in str(exc.value), (
        "the message must say which timeout fired: the httpx read timeout is reset forever by a "
        "trickling degenerate generation, so only the wall-clock bound can end one"
    )


def test_a_call_inside_its_bound_is_untouched():
    """The bound must not change the normal path. A guard that alters healthy behaviour is a
    regression dressed as a fix."""
    out = asyncio.run(guided_json(SCHEMA, _ctx(SleepyClient(0.01), call_timeout_s=5.0)))
    assert out == {"ok": True}


def test_the_timeout_does_not_burn_the_retry_budget():
    """A timeout must not be retried. ``max_attempts`` exists for an invalid or truncated
    object; a generation that overran is still occupying the shared instance, and re-issuing
    it would multiply load on the one vLLM every other loop is waiting on."""
    client = SleepyClient(30.0)
    with pytest.raises(GenerationTimeout):
        asyncio.run(guided_json(SCHEMA, _ctx(client, call_timeout_s=0.2, max_attempts=3)))
    assert client.calls == 1, f"timed-out call was retried {client.calls} times"


def test_the_bound_can_be_disabled_explicitly_but_never_by_accident():
    """``None`` opts out, but the default is a real number, so forgetting the knob leaves the
    bound on. The default was once effectively no bound, which is how this shipped."""
    assert GenCtx(client=None, model="m", prompt="p", seed=0).call_timeout_s == 1800.0
    out = asyncio.run(guided_json(SCHEMA, _ctx(SleepyClient(0.01), call_timeout_s=None)))
    assert out == {"ok": True}


def test_the_shared_client_singleton_carries_the_bound_and_passes_it_down():
    """The guarantee must be a property of the shared client, not of whoever remembered to
    pass it: decompose, the k loops, the coverage audit and the citation scorer's
    claim-importance judge all go through this one object."""
    assert LlmClient(model="m").call_timeout_s == 1800.0
    assert LlmClient(model="m", call_timeout_s=42.0).call_timeout_s == 42.0


def test_the_config_declares_the_bound_so_it_is_in_the_run_record():
    """``llm.call_timeout_s`` is a fairness-shared leaf: every run in a family bounds a
    generation identically, so a variant cannot quietly get more thinking time than its
    siblings. A knob that changes results and lives only in code is not in the run record."""
    from ragtime.config.loader import load

    values = {}
    for name in ("e2e-original", "e2e-omt", "e2e-omt-weak", "mlir-original"):
        cfg = load(f"config/{name}.yml")
        values[name] = cfg.blocks["llm"].get("call_timeout_s")
    assert all(v is not None for v in values.values()), f"undeclared in some config: {values}"
    assert len(set(values.values())) == 1, f"the bound diverges across the family: {values}"
