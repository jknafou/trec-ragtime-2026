"""``guided_json``: the LLM emission discipline in one wrapper.

vLLM 0.17.1 silently ignores the legacy ``guided_json`` request field, so this wrapper
keeps the internal name but issues either the enforced ``response_format`` json_schema
with ``strict: true`` or the native
``extra_body={"structured_outputs": {"json": S}}`` shape, according to the schema's
pinned ``mode``. Because a silent bypass is possible, every response is checked with
``jsonschema.validate``, with bounded retry on invalid or truncated output before
raising.

Operational rules baked in here:

* ``max_tokens`` is never sent: a tight budget on a thinking model returns
  ``content=None`` and JSON truncated mid-span.
* ``extra_body["chat_template_kwargs"]["enable_thinking"]`` is set to ``False``.
* the per-run ``seed`` is threaded into every call, including retries.
* the prompt is NFC-normalized via ``common.nfc``, the verbatim-commit normalizer.

This module imports no heavy library. The OpenAI client is injected via
``GenCtx.client``, so importing it and running ``guided_json`` against a stub works on
a machine with none of the model stack installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import jsonschema
from ragtime.common import get_logger, nfc

from .schemas import MODE_STRUCTURED, CompiledSchema

__all__ = [
    "GenCtx",
    "GenerationTimeout",
    "GuidedJsonError",
    "LlmClient",
    "decoding_kwargs",
    "guided_json",
]

_log = get_logger("serving.llm")


class GuidedJsonError(RuntimeError):
    """Raised when guided generation never returns a schema-valid object in bound."""


class GenerationTimeout(TimeoutError):
    """One generation call exceeded its wall-clock bound, ``llm.call_timeout_s``.

    It subclasses :class:`TimeoutError` so a caller that already handles timeouts keeps
    working, but it is its own type because it means something specific: the call did not
    fail, it never came back. It is distinct from ``openai.APITimeoutError``, the httpx read
    timeout, which means the socket went quiet; this one fires while the server is still
    streaming.
    """


@dataclass(frozen=True)
class GenCtx:
    """Per-call generation context threaded into one ``guided_json`` request.

    ``client`` is an ``openai.AsyncOpenAI``-shaped object, of which only
    ``client.chat.completions.create(**kwargs)`` is used; it is injected rather than
    instantiated here. ``seed`` is the per-run seed, pinned into every call, and
    ``max_attempts`` is the total number of ``create()`` calls allowed.
    """

    client: Any
    model: str
    prompt: str
    seed: int
    system: str | None = None
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    max_attempts: int = 3
    logger: Any = field(default=None)
    #: Wall-clock bound for one ``create()`` call, in seconds, from ``llm.call_timeout_s``.
    #: ``None`` leaves the call unbounded, so the default is a real number rather than
    #: ``None``: an unbounded default would silently reappear for anyone who forgets the knob.
    call_timeout_s: float | None = 1800.0


def decoding_kwargs(block: Any) -> dict[str, Any]:
    """Extract ``temperature``, ``top_p`` and ``top_k`` from one config block for ``GenCtx``.

    It takes the block rather than the whole config, because two stages legitimately decode
    from different blocks: the RAG loop from ``llm.decoding`` and decompose from
    ``decomposition.decoding``. What is shared is the extraction itself: read from config,
    never default, and omit a missing knob rather than substituting a value.

    A missing knob is not passed, so :class:`GenCtx`'s own default applies and the omission
    stays visible. Defaulting here would let the config and the generation that actually ran
    state two different facts.

    ``max_tokens`` is absent by construction, since ``GenCtx`` has no such field: a mid-object
    cut corrupts the JSON a schema-constrained generation is producing.
    """
    d = dict(block or {})
    out: dict[str, Any] = {}
    if "temperature" in d:
        out["temperature"] = float(d["temperature"])
    if "top_p" in d:
        out["top_p"] = float(d["top_p"])
    if "top_k" in d:
        out["top_k"] = int(d["top_k"])
    return out


def _as_compiled(schema: CompiledSchema | Any) -> CompiledSchema:
    if isinstance(schema, CompiledSchema):
        return schema
    # A bare mapping defaults to the strict response_format shape.
    from .schemas import MODE_RESPONSE_FORMAT

    return CompiledSchema("ad_hoc", schema, MODE_RESPONSE_FORMAT)


def _build_request(compiled: CompiledSchema, ctx: GenCtx) -> dict[str, Any]:
    """Build the ``create()`` kwargs: NFC prompt, seed, ``enable_thinking=False``, the
    enforced structured-output shape, and no ``max_tokens`` or legacy fields."""
    messages: list[dict[str, str]] = []
    if ctx.system:
        messages.append({"role": "system", "content": nfc(ctx.system)})
    messages.append({"role": "user", "content": nfc(ctx.prompt)})

    extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
    if ctx.top_k is not None:
        extra_body["top_k"] = ctx.top_k

    kwargs: dict[str, Any] = {
        "model": ctx.model,
        "messages": messages,
        "seed": ctx.seed,
        "temperature": ctx.temperature,
        "extra_body": extra_body,
    }
    if ctx.top_p is not None:
        kwargs["top_p"] = ctx.top_p

    if compiled.mode == MODE_STRUCTURED:
        # Native vLLM/xgrammar field, which allows genuinely optional keys.
        extra_body["structured_outputs"] = {
            "json": dict(compiled.schema),
            # Structural whitespace is the only unbounded production in this grammar: every
            # content field carries `maxLength`, every array `maxItems`, and
            # `additionalProperties: false` forbids inventing more, but JSON's padding between
            # tokens was unbounded, and `max_tokens` is never sent. A generation
            # that wanders into it never comes back, since the probability of emitting more
            # whitespace after whitespace approaches 1. In-string whitespace is untouched: past
            # the opening quote a space is string content bounded by that string's own
            # `maxLength`, so the verbatim-span contract is unaffected.
            "disable_any_whitespace": True,
        }
    else:
        # OpenAI-native json_schema in strict mode.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": compiled.name,
                "schema": dict(compiled.schema),
                "strict": True,
            },
        }
    # `max_tokens` is intentionally never set, and the legacy `guided_json` and
    # `guided_decoding_backend` fields are never built.
    return kwargs


def _extract_content(resp: Any) -> str:
    """Pull ``choices[0].message.content``, or ``""`` if the model returned None."""
    try:
        content = resp.choices[0].message.content
    except (AttributeError, IndexError):
        return ""
    return content if isinstance(content, str) else ""


async def guided_json(schema: CompiledSchema | Any, ctx: GenCtx) -> dict[str, Any]:
    """Issue one enforced structured-output generation and return the validated object.

    Builds the request (NFC prompt, seed pinned, ``enable_thinking=False``, the schema's
    pinned enforced shape, no ``max_tokens``), calls the injected client, then validates the
    parsed object with ``jsonschema`` every time, which guards against a silent grammar
    bypass. Retries invalid or truncated output up to ``ctx.max_attempts`` total calls, then
    raises :class:`GuidedJsonError`.
    """
    compiled = _as_compiled(schema)
    kwargs = _build_request(compiled, ctx)
    last_err: Exception | None = None

    for attempt in range(max(1, ctx.max_attempts)):
        # A wall-clock bound on the call itself. `REQUEST_TIMEOUT_S` below is an httpx read
        # timeout, which measures the gap between reads, so a degenerate generation that
        # trickles one token at a time resets it forever and the call never returns. That
        # would break the RAG loop's budget backstop, since `max_iters`, `max_searches` and
        # `token_budget` can only force a terminal between turns. `asyncio.timeout` bounds
        # total elapsed time, and cancelling the coroutine closes the socket, so vLLM aborts
        # the request and the shared instance gets its KV cache back.
        t0 = time.perf_counter()
        try:
            if ctx.call_timeout_s is None:
                resp = await ctx.client.chat.completions.create(**kwargs)
            else:
                async with asyncio.timeout(ctx.call_timeout_s):
                    resp = await ctx.client.chat.completions.create(**kwargs)
        except TimeoutError as exc:
            elapsed = time.perf_counter() - t0
            _log.error(
                "guided_json.timeout",
                schema=compiled.name,
                attempt=attempt + 1,
                elapsed_s=round(elapsed, 1),
                call_timeout_s=ctx.call_timeout_s,
                seed=ctx.seed,
            )
            raise GenerationTimeout(
                f"guided_json({compiled.name}) exceeded its wall-clock bound of "
                f"{ctx.call_timeout_s:g}s (elapsed {elapsed:.1f}s, seed={ctx.seed}). The call "
                "was cancelled, so the shared vLLM instance is freed. This is not the httpx "
                "read timeout, which a trickling degenerate generation resets forever. Raise "
                "llm.call_timeout_s only if a legitimate generation needs longer."
            ) from exc
        content = _extract_content(resp)
        try:
            obj = json.loads(content or "")
            jsonschema.validate(obj, dict(compiled.schema))
            return obj
        except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
            last_err = exc
            _log.warning(
                "guided_json.retry",
                schema=compiled.name,
                attempt=attempt + 1,
                max_attempts=ctx.max_attempts,
                error=type(exc).__name__,
            )

    raise GuidedJsonError(
        f"guided_json({compiled.name}) failed to return a schema-valid object "
        f"after {ctx.max_attempts} attempt(s)"
    ) from last_err


#: HTTP read timeout for one generation call, in seconds. It is generous and is
#: not a latency budget: it bounds how long one request may sit in vLLM's internal queue
#: before the client gives up.
#:
#: Single-stream generation on this model runs at roughly ten tokens per second, so a long
#: structured generation legitimately takes minutes, and throughput does not degrade at high
#: concurrency. Client-side throttling therefore buys nothing, while a short timeout
#: manufactures failures.
#:
#: This must stay strictly greater than ``llm.call_timeout_s``, or it silently becomes the real
#: timeout. There are two bounds on one generation and they are not interchangeable:
#:
#: * ``llm.call_timeout_s`` is the semantic bound. ``guided_json`` wraps the call in
#:   ``asyncio.timeout``, so it raises :class:`GenerationTimeout`, which the RAG loop turns
#:   into a clean ``closed_by="timeout"`` record. It is configurable and fairness-shared.
#: * ``REQUEST_TIMEOUT_S`` is the transport bound, handed to the OpenAI SDK.
#:
#: Whichever is smaller is the one that fires, and the transport bound is the worse of the two
#: to hit: it raises ``APITimeoutError``, which the SDK retries, so one slow generation could
#: cost several times the bound in wall clock and GPU. Keep this above ``call_timeout_s`` with
#: margin and treat it as a backstop for a wedged socket. Nothing asserts the ordering; it
#: holds because this constant sits far above every ``llm.call_timeout_s`` the configs carry.
REQUEST_TIMEOUT_S = 99999.0

#: Transport-level retries. Retrying is right for a connection error and wrong for a timeout,
#: and the two must not be conflated. A read error against a server that logged every request
#: as 200 OK means the socket died rather than that the model is still working, so a retry
#: duplicates nothing; the likely mechanism is an idle pooled connection reused past the
#: server's keep-alive window, which is exactly what decompose's long gaps between calls
#: produce. The SDK distinguishes the two for us, raising ``APIConnectionError`` for a dead
#: socket and ``APITimeoutError`` when :data:`REQUEST_TIMEOUT_S` elapses. Two retries cover a
#: stale-socket blip without turning a genuinely slow generation into a load multiplier.
#: ``guided_json`` still owns the only semantic retry, for invalid or truncated schema output.
REQUEST_MAX_RETRIES = 2


class LlmClient:
    """The shared vLLM client used by decompose, the k RAG loops, the aggregator and dedup.

    Holds the config-resolved ``model`` id and a lazily constructed ``openai.AsyncOpenAI``,
    with ``openai`` imported only on first use so the stub path needs no dependencies.
    ``generate`` is a thin ``guided_json`` call.
    """

    __slots__ = ("_client", "_client_loop", "api_key", "base_url", "call_timeout_s", "model")

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str = "EMPTY",
        call_timeout_s: float | None = 1800.0,
    ) -> None:
        self.model = model
        # The config-declared wall-clock bound, applied to every call this client makes, so
        # the guarantee is a property of the shared client rather than of each call site.
        self.call_timeout_s = call_timeout_s
        self.base_url = base_url or os.environ.get("RAGTIME_VLLM_URL", "http://localhost:8000/v1")
        self.api_key = api_key
        self._client: Any = None
        self._client_loop: Any = None

    def _ensure_client(self) -> Any:
        """Return the cached ``AsyncOpenAI``, rebuilding it if its event loop changed.

        There is still one client per loop, which is what sharing a single vLLM connection
        means here; a second client on the same loop would be the violation, and that cannot
        happen.

        ``AsyncOpenAI`` binds an ``httpx`` connection pool to the loop running when it is
        constructed, and ``asyncio.run()`` creates a fresh loop and closes it on exit. A caller
        that invokes ``asyncio.run`` more than once would otherwise get a cached client whose
        pool belongs to a dead loop and fail with ``RuntimeError: Event loop is closed``
        wrapped in ``openai.APIConnectionError``. Production runs one loop per cell, but a
        client whose correctness depends on how the caller drove asyncio is a trap, and
        rebinding is cheap: an idle pool holds no sockets.
        """
        try:
            loop: Any = asyncio.get_running_loop()
        except RuntimeError:  # sync context: keep whatever client we have
            loop = self._client_loop
        if self._client is None or loop is not self._client_loop:
            from openai import AsyncOpenAI

            self._client_loop = loop
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=REQUEST_TIMEOUT_S,
                max_retries=REQUEST_MAX_RETRIES,
            )
        return self._client

    async def generate(
        self, schema: CompiledSchema | Any, prompt: str, seed: int, **kw: Any
    ) -> dict[str, Any]:
        kw.setdefault("call_timeout_s", self.call_timeout_s)
        ctx = GenCtx(client=self._ensure_client(), model=self.model, prompt=prompt, seed=seed, **kw)
        return await guided_json(schema, ctx)
