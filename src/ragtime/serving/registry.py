"""The single ``config -> client`` factory: per-node singletons, no stage loads weights.

``build_clients(config) -> ClientBundle`` is the only factory. The returned bundle holds
per-node singletons. Decompose, the k RAG loops, the aggregator and the dedup-confirm gate
all share ``.llm``, one vLLM client. Retrieval and decompose's dedup share the index's dense
identity: retrieval through ``.index_dense``, dedup through ``.query_dense``, which is the
same object unless ``index_build.config.query_encode_device`` names a different device.
``.embedder``, built from the query-time ``retrieval`` block, is unused on every ``e2e-*``
config. The offline index build has its own three clients, ``.index_dense``, ``.milco`` and
``.mtd_colbert``, because their identities live in the hashed ``index_build.config`` block
that the index manifest records; ``.index_dense`` collapses onto ``.embedder`` when both name
the same model, revision and device, so an identity is never resident twice. Every model
identity and backend is read from config rather than hardcoded.

``build_stub_clients(...)`` returns a CPU fake bundle with a canned-JSON LLM and no-op
clients for the small tests, so no test imports ``torch`` or ``vllm``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .encoders import DEFAULT_INDEX_DENSE_MODEL, Encoder
from .late_interaction import (
    DEFAULT_MTD_CHECKPOINT,
    DEFAULT_MTD_REVISION,
    MtdEncoder,
)
from .llm import LlmClient
from .mt import DEFAULT_COMPUTE_TYPE, FAMILY_NLLB, MtClient
from .reranker import Reranker
from .schemas import SchemaSet, compile_schemas
from .sparse_milco import (
    DEFAULT_MAX_LENGTH as DEFAULT_MILCO_MAX_LENGTH,
)
from .sparse_milco import (
    DEFAULT_MILCO_MODEL,
    DEFAULT_MILCO_REVISION,
    MilcoEncoder,
)

__all__ = ["ClientBundle", "build_clients", "build_stub_clients"]


@dataclass(frozen=True, slots=True)
class ClientBundle:
    """Frozen holder of the per-node serving singletons.

    Two reads of any field return the same object, which a test can check by identity.
    """

    llm: Any
    embedder: Any
    #: The index build's dense encoder, deliberately its own field rather than ``embedder``.
    #: Its identity lives in the hashed ``index_build.config.dense_model``, which is what the
    #: index manifest and the per-leg config hash record, while ``embedder``'s lives in the
    #: query-time ``retrieval`` block that no shipped ``e2e-*`` config carries. Pointing the
    #: index build at ``embedder`` made the manifest a false record of what encoded the corpus.
    #: The one exception: when both resolve to the same model, revision and device, this is
    #: ``embedder``, so an identity is never resident twice.
    index_dense: Any
    #: The same dense identity as ``index_dense``, on the query side's device.
    #:
    #: Same model, revision and weights, so a query embedding lands in the index's embedding
    #: space. The only thing that may differ is the device, read from
    #: ``index_build.config.query_encode_device``, which is excluded from ``index_hash``
    #: because embedding a query writes no stored byte. When that leaf is absent, as in every
    #: shipped config, this field is ``index_dense`` by object identity.
    #:
    #: Its consumer is decompose's paraphrase-dedup pre-filter. ``retrieval.context``
    #: still encodes queries with ``index_dense``, because the retrieval service
    #: places encoders per process and routing it here would silently override that placement.
    query_dense: Any
    reranker: Any
    mt: Any
    #: The index's learned-sparse leg (MILCO) and late-interaction leg (PyLate). Both are
    #: constructed like every other client, from the hashed config block and lazily loaded, so
    #: ``preprocess.index`` instantiates no model of its own. They are index-build clients but
    #: live on this registry because a query-time consumer needs the identical encoders to
    #: embed a query into the same spaces.
    milco: Any
    mtd_colbert: Any
    schemas: SchemaSet


def _blk(config: Any, name: str) -> dict:
    return dict(config.blocks.get(name, {}) if config is not None else {})


def _mt_client(config: Any) -> MtClient:
    """Build the MT client from the ``translation`` block and ``execution.ct2_model_dir``.

    Two things must not be confused here:

    - Shape. ``translation`` is ``{config: {...}, hash: "..."}``, so every knob sits one level
      down under ``config``. A direct ``translation.get("engine")`` always misses and silently
      falls through to its default, which hands CTranslate2 a model identity string
      where it expects a checkpoint path. The reads below go through ``translation["config"]``.
    - Identity against path. ``translation.config.omt_model`` is the semantic identity, which
      is fairness-shared, hashed and stamped on every ``Translation`` row's ``model_id``;
      ``execution.ct2_model_dir`` is the machine-local directory of the converted checkpoint,
      kept in the unshared ``execution`` block so a filesystem path never enters the fairness
      hash. They stay separate all the way into :class:`~ragtime.serving.mt.MtClient`.

    ``compute_type`` also comes from the hashed block and is passed to CTranslate2 explicitly,
    so the declared quantization and the one that runs are a single fact.

    The directory is validated at first use rather than here. Validating at construction would
    couple every caller of :func:`build_clients` to an offline MT artifact it may never touch,
    so a chunk worker or an online vLLM node would fail to start over an absent checkpoint it
    does not use. ``MtClient`` re-checks the path when it loads, so a genuinely missing
    checkpoint still fails with a message naming the config key.
    """
    tcfg = dict(_blk(config, "translation").get("config") or {})
    execution = _blk(config, "execution")
    # Both knobs are either a scalar, for one multilingual checkpoint such as the NLLB arm, or
    # a per-source-language mapping, for N bilingual checkpoints such as the Opus-MT arm where
    # the direction is the model. The identity side stays in the hashed block, so a mapping
    # puts all N identities into the translation hash and changing any one of them re-homes the
    # whole arm. The path side stays in the unhashed `execution` block, since a filesystem
    # layout is not a semantic fact.
    model = _scalar_or_map(tcfg.get("omt_model"))
    if not model:
        raise ValueError(
            "translation.config.omt_model is missing; it is the semantic MT model identity "
            "stamped on every Translation row's model_id, while the on-disk checkpoint is "
            "the separate, unhashed execution.ct2_model_dir"
        )
    model_dir = _scalar_or_map(execution.get("ct2_model_dir"))
    if not model_dir:
        raise ValueError(
            "execution.ct2_model_dir is missing; serving needs the machine-local "
            f"CTranslate2 checkpoint directory for {model!r}, produced by "
            "ct2-transformers-converter. It is not translation.config."
        )
    if isinstance(model, Mapping) != isinstance(model_dir, Mapping):
        raise ValueError(
            "translation.config.omt_model and execution.ct2_model_dir must have the same "
            f"shape; got {type(model).__name__} and {type(model_dir).__name__}. A "
            "per-direction arm needs an identity and a checkpoint for every direction, and "
            "mixing the two shapes would stamp one identity on text several models wrote."
        )
    if isinstance(model, Mapping) and set(model) != set(model_dir):
        raise ValueError(
            "translation.config.omt_model and execution.ct2_model_dir cover different "
            f"source languages ({sorted(model)} vs {sorted(model_dir)}); every direction "
            "must have both, or a row would claim a model that never ran."
        )
    # The directory is checked by MtClient on load; see the docstring.
    return MtClient(
        model=model,
        engine=str(tcfg.get("engine", "ctranslate2") or "ctranslate2"),
        model_path=model_dir,
        compute_type=str(tcfg.get("compute_type", DEFAULT_COMPUTE_TYPE) or DEFAULT_COMPUTE_TYPE),
        family=str(tcfg.get("model_family", FAMILY_NLLB) or FAMILY_NLLB),
    )


def _index_build(config: Any) -> dict:
    """Return the ``index_build`` block's own ``config`` sub-mapping.

    Same shape as ``translation``, ``{config: {...}, hash: "..."}``, so a direct
    ``index_build.get("sparse_model")`` always misses and silently falls through to a default.
    Every read goes through ``["config"]``.

    A missing block resolves to ``{}`` and the clients below fall back to their documented
    default identities, so an unrelated ``build_clients`` caller such as a chunk worker or an
    online vLLM node is not broken by an offline index knob it never touches.
    """
    return dict(_blk(config, "index_build").get("config") or {})


def _milco_client(config: Any) -> MilcoEncoder:
    """Build the learned-sparse client from ``index_build.config``.

    ``sparse_model`` and ``sparse_model_revision`` are the semantic identity, hashed and
    recorded in the index manifest. The revision is a git SHA because the repository
    publishes no tag or release, so a branch pin would let the weights move under a built
    index.

    ``sparse_max_length`` is read here for the same reason ``document_length`` is read for the
    late-interaction client: it decides which tokens of a passage the leg sees, so leaving it
    to a module constant would hide the encode window behind a hashed artifact. The fallback
    is the client's own default, imported rather than re-typed, so the recipe and the engine
    cannot state two different windows.
    """
    cfg = _index_build(config)
    return MilcoEncoder(
        model=str(cfg.get("sparse_model", DEFAULT_MILCO_MODEL) or DEFAULT_MILCO_MODEL),
        revision=str(cfg.get("sparse_model_revision", DEFAULT_MILCO_REVISION) or ""),
        device=str(cfg.get("encode_device", "auto") or "auto"),
        max_length=int(
            cfg.get("sparse_max_length", DEFAULT_MILCO_MAX_LENGTH) or DEFAULT_MILCO_MAX_LENGTH
        ),
    )


def _index_dense_client(config: Any, embedder: Encoder) -> Encoder:
    """Build the dense encoder the index build uses, from ``index_build.config``.

    ``dense_model`` and ``dense_revision`` are the semantic identity: they are hashed into the
    ``index_build`` block, folded into the index path, and written into the manifest's per-leg
    config hash and provenance. Reading them here is what makes that record true. Taking the
    encoder from the query-time ``retrieval.dense`` key instead would make the manifest claim
    one model while another encoded the corpus.

    ``device`` comes from the same block's ``encode_device``, because device and dtype change
    the forward-pass numerics and the hashed record must match the engine.

    When the retrieval block happens to name the same model, revision and device, the same
    object is returned rather than a second one, so no node holds two copies of one model.
    """
    cfg = _index_build(config)
    model = str(cfg.get("dense_model", DEFAULT_INDEX_DENSE_MODEL) or DEFAULT_INDEX_DENSE_MODEL)
    revision = str(cfg.get("dense_revision", "") or "")
    device = str(cfg.get("encode_device", "") or "") or embedder.device
    if (embedder.model, embedder.revision, embedder.device) == (model, revision, device):
        return embedder
    return Encoder(model=model, device=device, revision=revision)


def _query_dense_client(config: Any, index_dense: Encoder) -> Encoder:
    """Build the query-side dense encoder: ``index_dense`` on the query device.

    There is one implementation with a device parameter, never a second encoder and never a
    CPU-versus-GPU branch in a caller. The model, the revision and therefore the embedding
    space come from ``index_dense`` itself, so this cannot drift from the encoder that built
    the index. Only ``index_build.config.query_encode_device`` moves, and when it is absent or
    names the device ``index_dense`` already holds, the same object is returned.

    It is a separate field rather than an override of ``index_dense``'s own device because
    ``index_dense`` is what ``preprocess.index`` hands to the vectorize step, and its device is
    the hashed, manifest-recorded ``encode_device``. The build's device and the query's device
    are two facts, so they are two leaves and two fields.

    This device is not a free machine knob. A forward pass on CPU and one on CUDA are not the
    same vector, decompose's dedup compares them against a fixed ``cosine_cutoff``, and a pair
    either side of that cutoff merges on one device and not on the other. That is why the leaf
    lives in the fairness-shared ``index_build`` block rather than in ``execution``, and why a
    CPU arm is a different arm rather than a cheaper way to run the same one.
    """
    device = str(_index_build(config).get("query_encode_device", "") or "")
    if not device or device == index_dense.device:
        return index_dense
    return Encoder(model=index_dense.model, device=device, revision=index_dense.revision)


def _mtd_client(config: Any) -> MtdEncoder:
    """Build the late-interaction client from ``index_build.config``.

    ``document_length`` and ``query_length`` default to 0, meaning "use the checkpoint's own
    metadata", which PyLate reads natively, so an unset knob never silently becomes a
    different truncation window from the one the checkpoint was trained for. The per-language
    truncation that encode-time windowing once addressed is now removed upstream by
    ``preprocess.packing``, which packs every passage to fit the window in every rendering.
    """
    cfg = _index_build(config)
    return MtdEncoder(
        checkpoint=str(cfg.get("spine_model", DEFAULT_MTD_CHECKPOINT) or DEFAULT_MTD_CHECKPOINT),
        revision=str(cfg.get("spine_model_revision", DEFAULT_MTD_REVISION) or ""),
        device=str(cfg.get("encode_device", "auto") or "auto"),
        document_length=int(cfg.get("document_length", 0) or 0),
        query_length=int(cfg.get("query_length", 0) or 0),
    )


def _scalar_or_map(value: Any) -> str | Mapping[str, str]:
    """A config knob that may be one string or a ``{source_lang: string}`` mapping."""
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    return str(value or "")


def build_clients(config: Any) -> ClientBundle:
    """Build the per-node singleton bundle from ``config``.

    Model identities and backends come from the validated config blocks: the LLM from
    ``llm.model``, the encoder and reranker from the ``retrieval`` block, and MT from
    ``translation.config`` together with ``execution.ct2_model_dir`` (see
    :func:`_mt_client`). Constructing a client opens no port and loads no weights, since heavy
    backends are lazy on first call; the vLLM server is stood up separately by ``node``.
    """
    llm = _blk(config, "llm")
    retrieval = _blk(config, "retrieval")
    reranker_cfg = dict(retrieval.get("reranker", {})) if retrieval else {}

    embedder = Encoder(model=str(retrieval.get("dense", "")))
    reranker = Reranker(model=str(reranker_cfg.get("model", "")))
    index_dense = _index_dense_client(config, embedder)

    return ClientBundle(
        llm=LlmClient(
            model=str(llm.get("model", "")),
            # `llm.call_timeout_s` is a fairness-shared config leaf: every run in a family
            # bounds a generation identically, so no variant gets more time than its siblings.
            call_timeout_s=llm.get("call_timeout_s", 1800.0),
        ),
        embedder=embedder,
        index_dense=index_dense,
        query_dense=_query_dense_client(config, index_dense),
        reranker=reranker,
        mt=_mt_client(config),
        milco=_milco_client(config),
        mtd_colbert=_mtd_client(config),
        schemas=compile_schemas(),
    )


# CPU stub bundle for the small tests; torch and vllm are never imported.
class _StubResp:
    def __init__(self, content: str) -> None:
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _StubCompletions:
    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self._i = 0
        self.calls: list[dict] = []

    async def create(self, **kwargs: Any) -> _StubResp:
        self.calls.append(kwargs)
        if not self._contents:
            content = "{}"
        else:
            content = self._contents[min(self._i, len(self._contents) - 1)]
        self._i += 1
        return _StubResp(content)


class _StubOpenAI:
    """Minimal ``AsyncOpenAI``-shaped fake exposing ``chat.completions.create``."""

    def __init__(self, contents: list[str]) -> None:
        self.chat = type("Chat", (), {"completions": _StubCompletions(contents)})()


class _StubLlm:
    """Stub LLM singleton carrying ``.model`` + a canned-JSON client."""

    __slots__ = ("client", "model")

    def __init__(self, model: str, contents: list[str]) -> None:
        self.model = model
        self.client = _StubOpenAI(contents)


class _StubClient:
    """A no-op fake for the small co-located clients; carries a ``.model`` tag."""

    __slots__ = ("model",)

    def __init__(self, model: str) -> None:
        self.model = model


def build_stub_clients(
    config: Any = None,
    *,
    responses: list[str] | None = None,
    llm_model: str | None = None,
) -> ClientBundle:
    """Return a CPU fake ``ClientBundle`` for the small tests, with no heavy imports.

    The LLM id is read from ``config`` when given, which lets a test prove the identity comes
    from config, otherwise from the explicit ``llm_model`` override, and otherwise from a stub
    default. ``responses`` seeds the canned-JSON LLM client.
    """
    if config is not None:
        llm_model = llm_model or str(_blk(config, "llm").get("model", ""))
    index_dense = _index_dense_client(config, Encoder(model="", device="cuda"))
    return ClientBundle(
        llm=_StubLlm(model=llm_model or "stub-llm", contents=responses or []),
        embedder=_StubClient("stub-embedder"),
        # Real but lazy, like the two index-leg clients below: it loads nothing at
        # construction, and keeping it real lets a small test prove the index's dense identity
        # comes from the hashed block rather than from a stub label.
        index_dense=index_dense,
        # Real for the same reason, and derived from `index_dense` rather than stubbed, so a
        # small test can prove the collapse to one object without a GPU.
        query_dense=_query_dense_client(config, index_dense),
        reranker=_StubClient("stub-reranker"),
        mt=_StubClient("stub-mt"),
        # The two index-leg encoders are real client objects even in the stub bundle. Both are
        # lazy, importing nothing heavy until an encode call, so constructing them is free and
        # a small test can still prove the identity comes from config.
        milco=_milco_client(config),
        mtd_colbert=_mtd_client(config),
        schemas=compile_schemas(),
    )
