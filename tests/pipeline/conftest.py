"""Fixtures for the decompose tests.

Two rules this file exists to enforce:

* Real topics, never hand-written ones. ``small_topics`` (root ``tests/conftest.py``)
  is a 3-topic slice of the canonical ``topics/topics.all.2026.v0625-fix.jsonl``, loaded through
  ``common.topics.load_topics``: the loader that absorbs TREC's concatenated
  single-line-JSONL quirk. Nothing here parses that format itself.
* Real config values, never invented ones. ``decompose_cfg`` carries the actual
  ``decomposition`` block of ``config/e2e-original.yml`` (loaded via ``config.load``),
  deep-copied to plain dicts so a test can move one knob to prove it is read from config
  rather than baked into the code. The shipped file itself is never edited.

The stub ``ClientBundle`` is a fake in the ``serving.registry.build_stub_clients``
spirit: no ``torch``/``vllm``/``sentence_transformers``/``numpy`` anywhere in this tier.
It records every call, so a test can assert on prompt text, schema identity, decoding
kwargs and the number of calls.
"""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ragtime.serving import compile_schemas

# ---------------------------------------------------------------------------------------------
# The RAG loop's half of this conftest: the retrieval fixtures, re-exported so `tests/pipeline` can
# see them. pytest resolves fixtures by conftest directory, not by import, so a fixture defined in
# `tests/retrieval/conftest.py` is invisible here and a test requesting it errors at setup with
# "fixture not found" -- which reads as a missing dependency rather than a scoping mistake.
#
# The two halves are disjoint: this block serves `test_ragloop_full.py` (real tri-leg index, real
# encoders, the real retrieve -> display path); everything below serves the decompose tests with
# stubs.
from tests.retrieval.conftest import (  # noqa: F401  -- re-exported for pytest, not for import
    Built,
    build_index,
    built,
    context_for,
    retrieval_cfg,
    retrieval_docs,
    scaled_legs,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = _REPO_ROOT / "config" / "e2e-original.yml"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class StubCfg:
    """A ``RunConfig``-shaped holder with mutable blocks (the real one is frozen).

    Only the two attributes decompose reads, ``blocks`` and ``passage_lang``, plus
    ``seeds`` for shape parity. Tests mutate ``blocks["decomposition"][...]`` to prove a
    knob is read from config; the real ``RunConfig`` is ``MappingProxyType`` all the way
    down and cannot be tweaked in place.
    """

    blocks: dict[str, Any]
    passage_lang: str = "original"
    seeds: int = 5


@pytest.fixture(scope="session")
def real_run_config():
    """The real, validated ``config/e2e-original.yml`` (a genuine ``RunConfig``)."""
    from ragtime.config import load

    return load(REAL_CONFIG)


@pytest.fixture
def decompose_cfg(real_run_config) -> StubCfg:
    """A mutable copy of the real config's blocks (decomposition values are the real ones)."""
    return StubCfg(blocks=_plain(dict(real_run_config.blocks)))


def _plain(obj: Any) -> Any:
    """Deep-copy a frozen ``RunConfig`` block tree into plain dicts/lists."""
    if hasattr(obj, "items"):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return copy.deepcopy(obj)


# --------------------------------------------------------------------------- #
# Stub ClientBundle
# --------------------------------------------------------------------------- #
@dataclass
class Recorded:
    """One captured ``llm.generate`` call."""

    schema: Any
    prompt: str
    seed: int
    kwargs: dict[str, Any]

    @property
    def name(self) -> str:
        return getattr(self.schema, "name", str(self.schema))


class StubLlm:
    """A canned-JSON LLM singleton keyed by compiled-schema name, then call order.

    Keying by schema name rather than by a global counter lets a test assert that the
    draft call and the self-critique call both used the ``nuggets`` schema, in that
    order, without becoming brittle to an unrelated dedup-confirm call landing between
    them. The last response of a sequence repeats, so a test never has to predict the
    exact number of dedup confirms.
    """

    def __init__(self, by_schema: dict[str, list[Any]]) -> None:
        self._by = {k: list(v) for k, v in by_schema.items()}
        self._i: dict[str, int] = defaultdict(int)
        self.calls: list[Recorded] = []
        self.model = "stub-llm"

    async def generate(self, schema: Any, prompt: str, seed: int, **kwargs: Any) -> dict:
        name = getattr(schema, "name", str(schema))
        self.calls.append(Recorded(schema=schema, prompt=prompt, seed=seed, kwargs=kwargs))
        seq = self._by.get(name)
        if not seq:
            raise KeyError(f"stub LLM has no canned response for schema {name!r}")
        obj = seq[min(self._i[name], len(seq) - 1)]
        self._i[name] += 1
        return obj(prompt) if callable(obj) else copy.deepcopy(obj)

    def calls_for(self, name: str) -> list[Recorded]:
        return [c for c in self.calls if c.name == name]


class StubEmbedder:
    """Deterministic one-hot embeddings: identical text -> cosine 1.0, distinct -> 0.0.

    Chosen over pseudo-random vectors so that a test about something *else* (seed bank
    shape, stats, prompts) can never fail on an accidental dedup merge, while the dedup
    tests that DO care inject their own vectors.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.model = "stub-embedder"

    def embed(self, texts: list[str], mode: str = "dense") -> list[list[float]]:
        self.calls.append((list(texts), mode))
        uniq = sorted({t for t in texts}, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
        index = {t: i for i, t in enumerate(uniq)}
        width = max(1, len(uniq))
        out = []
        for t in texts:
            row = [0.0] * width
            row[index[t]] = 1.0
            out.append(row)
        return out


class DeadEmbedder:
    """``ClientBundle.embedder`` as decompose must treat it: off limits, and loud about it.

    ``serving.registry`` builds ``embedder`` from the query-time leaf ``retrieval.dense``.
    On every shipped config that leaf names ``BAAI/bge-m3``, so this is a constructible
    encoder rather than a broken one -- but it carries **no revision**, while
    ``index_build.config.dense_revision`` pins the checkout that actually encoded the corpus.
    The two therefore do not collapse onto one object in ``_index_dense_client``, and
    embedding a query with ``.embedder`` would compare it against vectors a possibly
    different checkout of the same model id produced. No stage reads the field: the search
    path uses ``.index_dense`` / ``.query_dense`` and dedup uses ``.query_dense``.

    ``.embedder`` is not dead inside ``serving.registry`` itself -- it is the object
    ``_index_dense_client`` collapses onto when the retrieval leaf happens to name the same
    model, revision and device, which is what stops a node holding two copies of one model.
    What is dead is every use of it as an encoder by a caller, which is the only thing this
    stub is about.

    An earlier stub bundle carried a working ``embedder`` and no ``index_dense`` at all,
    mirroring the implementation instead of the contract, so the small tier stayed green
    while real data failed every time. Raising here keeps that defect catchable without a
    GPU: anything reaching for ``.embedder`` fails immediately, with the reason attached.
    """

    model = ""

    def embed(self, texts: list[str], mode: str = "dense") -> Any:
        raise AssertionError(
            "ClientBundle.embedder is not the encoder to use: it is built from the "
            "query-time `retrieval.dense` leaf with no revision, so it is not pinned to the "
            "checkout that built the index. Use clients.index_dense (or clients.query_dense "
            "for the query-side device): the encoder keyed by the hashed "
            "`index_build.config.dense_model` and `dense_revision`. See grow_nuggets.py."
        )


@dataclass
class StubBundle:
    """A ``ClientBundle``-shaped fake carrying only what decompose touches.

    ``index_dense`` is the working encoder; ``embedder`` is a ``DeadEmbedder``
    so the stub reproduces production's shape rather than a convenient fiction.

    ``query_dense`` mirrors production's collapse rather than being a second stub: with no
    ``index_build.config.query_encode_device`` leaf, which is the case in every shipped
    config, ``serving.registry`` returns the same object for both fields, so the stub does
    too. A fixture handing out two distinct recorders here would let dedup read the wrong
    one while every assertion about ``index_dense.calls`` still passed.
    """

    llm: StubLlm
    index_dense: StubEmbedder
    query_dense: Any = None
    embedder: Any = field(default_factory=DeadEmbedder)
    schemas: Any = field(default_factory=compile_schemas)

    def __post_init__(self) -> None:
        if self.query_dense is None:
            self.query_dense = self.index_dense


def nuggets_response(questions, *, weights=None, rationale="because") -> dict:
    """Build a ``{rationale, nuggets}`` payload (the widened decompose shape)."""
    weights = weights if weights is not None else [None] * len(questions)
    return {
        "rationale": rationale,
        "nuggets": [
            {"nugget_id": None, "question": q, "weight": w}
            for q, w in zip(questions, weights, strict=True)
        ],
    }


def audit_response(*, coverage=(), add=(), prune=(), rationale="audited") -> dict:
    """Build a ``{rationale, coverage, add, prune}`` audit-delta payload (the round loop).

    ``coverage`` is ``[(nugget_id, label)]`` and ``add`` is ``[(question, trigger, weight)]``:
    positional tuples rather than dicts, so a test reads as the delta it asserts about and not
    as a JSON blob whose keys have to be re-checked against the schema at every call site.
    """
    return {
        "rationale": rationale,
        "coverage": [{"nugget_id": nid, "coverage": label} for nid, label in coverage],
        "add": [
            {"question": q, "trigger_passage_id": trigger, "weight": weight}
            for q, trigger, weight in add
        ],
        "prune": list(prune),
    }


SEED_QUESTIONS = (
    "What legislation regulates the sale of the product?",
    "Which agency enforces that legislation?",
    "What penalties apply to a violation?",
    "When did the legislation take effect?",
)


@pytest.fixture
def stub_clients() -> StubBundle:
    """The default bundle: a 4-nugget draft, a 4-nugget weighted self-critique.

    The self-critique response differs from the draft in ``weight`` only,
    so a test that cares about bank CONTENT gets a stable, predictable question set.
    """
    draft = nuggets_response(SEED_QUESTIONS)
    critique = nuggets_response(SEED_QUESTIONS, weights=[0.9, 0.7, 0.4, 0.2])
    return StubBundle(
        llm=StubLlm(
            {
                "nuggets": [draft, critique],
                "dedup_and_gate": [
                    {
                        "duplicate": True,
                        "paraphrase_match": True,
                        "entity_match": True,
                        "reason": "same fact",
                    }
                ],
                "on_topic_gate": [{"rationale": "a genuine facet", "on_topic": True}],
            }
        ),
        index_dense=StubEmbedder(),
    )


def make_bundle(*, nuggets=(), dedup=None, on_topic=None, audit=()) -> StubBundle:
    """Build a stub bundle with explicit canned responses per schema.

    ``audit`` is the ``coverage_audit`` sequence, one entry per audit ROUND (the last repeats,
    like every other schema here), so a two-round test hands in two deltas and does not have
    to predict how many dedup confirms land between them.
    """
    return StubBundle(
        llm=StubLlm(
            {
                "nuggets": list(nuggets),
                "coverage_audit": list(audit) or [audit_response()],
                "dedup_and_gate": [
                    dedup
                    or {
                        "duplicate": False,
                        "paraphrase_match": False,
                        "entity_match": False,
                        "reason": None,
                    }
                ],
                "on_topic_gate": [
                    on_topic or {"rationale": "a genuine facet", "on_topic": True}
                ],
            }
        ),
        index_dense=StubEmbedder(),
    )


@pytest.fixture
def seed_topic(small_topics):
    """The first real topic of the small slice, as a plain namespace of its fields."""
    t = small_topics[0]
    return SimpleNamespace(
        topic_id=t.topic_id,
        problem_statement=t.problem_statement,
        background=t.background,
        limit=t.limit,
    )
