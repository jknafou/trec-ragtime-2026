"""MILCO, the multilingual learned-sparse encoder, behind one client.

This is the index's sparse leg. ``omai-research/milco-650m`` (Apache-2.0 weights) is loaded
through ``transformers`` with ``trust_remote_code``, since the vendor ships modelling code
in the repository rather than a PyPI package, and it is pinned by git revision SHA because
the repository carries no release or immutable tag.

MILCO projects every language into a single English SPLADE lexical space, which is why the
sparse leg can be built identically on the ``original``, ``omt`` and ``omt_opus``
renderings and still fire cross-lingually. That property is what makes a learned sparse
leg preferable to plain BM25 here.

Four operational facts this module keeps in one place:

1. Device placement is owned here, once. The vendor's ``encode_text`` calls ``.to(device)``
   on every invocation, so the model's device is a side effect of calling it. No stage may
   move a model, so the single ``.to()`` call site is this client, constructed by
   ``serving.registry``, and never ``preprocess.index``.
2. Batching is not owned here. ``encode_text(texts, batch_size=...)`` runs its own internal
   fixed-size loop, so passing a whole shard would make it a second batcher disagreeing
   with ``serving.batching.Batcher``. :meth:`MilcoEncoder.encode_text` therefore passes
   ``batch_size=len(texts)``: one already-sized bucket per call.
3. MILCO resolves a second repository at load time by repo id,
   ``AutoTokenizer.from_pretrained("BAAI/bge-m3-unsupervised")``, out of the Hugging Face
   cache rather than a local snapshot. It must be pre-cached on a machine with network
   access, which is what :func:`prefetch_pivot_tokenizer` does; that call omits
   ``revision=``, because a revision-pinned ``snapshot_download`` writes no ``refs/main`` and
   offline repo-id resolution then fails with ``LocalEntryNotFoundError``.
4. ``naver/splade-v3`` is a gated repository and is not needed. It only names terms via
   ``get_vocab()``, while the retrieval path keys components by integer dimension id, so
   ``encode_text`` is always called with ``return_dict=False``.

Heavy imports (``torch`` and ``transformers``) happen inside the backend on first use, so
importing this module and constructing the client stays cheap on a CPU-only machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragtime.common import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

__all__ = [
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MILCO_MODEL",
    "DEFAULT_MILCO_REVISION",
    "MODEL_MAX_LENGTH",
    "NO_TOP_K",
    "PIVOT_TOKENIZER_REPO",
    "SEISMIC_COMPONENT_MAXLEN",
    "MilcoEncoder",
    "SeismicComponentError",
    "prefetch_pivot_tokenizer",
    "seismic_components",
]

_log = get_logger("serving.sparse_milco")

#: The adopted checkpoint and its git-SHA pin. These are fallbacks; a run takes its value
#: from the hashed ``index_build.config`` block. They exist so that a caller without that
#: block does not silently encode with an unpinned model.
#:
#: The smaller ``milco-300m`` was evaluated as a way to decorrelate this leg from the dense
#: leg, since it uses a different pretraining lineage. It matched on rank-1 retrieval but
#: gave equal or tighter margins in every language, about 40 percent more non-zeros, and
#: needed three loading workarounds, one of them bypassing ``from_pretrained`` because the
#: meta-device init leaves its non-persistent rope buffers uninitialised. The shared
#: backbone between the dense and sparse legs is therefore recorded as a residual risk
#: rather than traded for a decorrelation benefit that cannot be measured without qrels.
DEFAULT_MILCO_MODEL = "omai-research/milco-650m"
DEFAULT_MILCO_REVISION = "567aeb03a5756c1e42793b9113925b4945ed5767"

#: The second repository MILCO resolves by repo id at load time.
PIVOT_TOKENIZER_REPO = "BAAI/bge-m3-unsupervised"

#: ``pyseismic-lsr`` stores a query or document component as a string with numpy dtype U30.
#: A longer string is truncated silently, which would merge two vocabulary dimensions into
#: one posting list, so the cap is asserted at the boundary. Stringified integer dimension
#: ids are at most six characters in every vocabulary used here, so the assertion never
#: fires on real input; it fires on a change that would otherwise break the leg silently.
SEISMIC_COMPONENT_MAXLEN = 30

#: Keep every component: the ``top_k`` of an unpruned vector, and the default, so a caller
#: that says nothing gets the encoder's raw output. It is named rather than spelled ``0``
#: because it is a meaningful value rather than an absent one, and it must never be folded
#: into an ``x or default`` idiom, which would turn "no cut" into "the default cut".
NO_TOP_K = 0

#: MILCO's own ``model_max_length``.
MODEL_MAX_LENGTH = 8192

#: The encode window. It decides which tail is dropped, so a caller reads it from the hashed
#: ``index_build.config.sparse_max_length`` leaf and this is only the fallback.
#:
#: It is the model's ceiling rather than the corpus's observed maximum, since sizing to
#: today's longest passage would reintroduce truncation the first time a longer document
#: arrives. An earlier value of 512 had no architectural justification here (unlike the
#: late-interaction leg, where 512 is XLM-R's real ceiling) and cost a per-language content
#: loss: measured over the whole corpus, Chinese passages exceeded 512 tokens roughly two
#: orders of magnitude more often than Spanish ones, in the one leg whose purpose is
#: unbiased cross-lingual firing. Truncation is never silent either way, since
#: :meth:`MilcoEncoder.truncated_tokens` measures the loss on the path that runs.
DEFAULT_MAX_LENGTH = MODEL_MAX_LENGTH


class SeismicComponentError(ValueError):
    """A sparse component id would not survive Seismic's ``U30`` string cap."""


def prefetch_pivot_tokenizer(repo_id: str = PIVOT_TOKENIZER_REPO) -> str:
    """Pre-cache MILCO's second repository and return the snapshot path.

    Called without ``revision=``: a revision-pinned ``snapshot_download`` writes
    the blobs but no ``refs/main``, and MILCO's own
    ``from_pretrained("BAAI/bge-m3-unsupervised")`` uses a repo id rather than a path, so it
    then fails to resolve offline with ``LocalEntryNotFoundError``. The pin that matters for
    reproducibility is on the MILCO checkpoint itself; this repository contributes only a
    tokenizer vocabulary.

    This needs network access, so it runs where network is available rather than inside a
    build job.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id)
    _log.info("serving.sparse_milco.prefetch", repo_id=repo_id, path=str(path))
    return str(path)


def seismic_components(
    weights: Mapping[int, float] | Mapping[str, float],
    *,
    top_k: int = NO_TOP_K,
) -> tuple[list[str], list[float]]:
    """Convert one sparse vector into ``(components, values)`` in Seismic's string form.

    Components are the stringified vocabulary dimension ids, ordered by that string, which
    is the form Seismic keys on and a total order, so two builds of the same weights produce
    byte-identical arrays. Every component is checked against
    :data:`SEISMIC_COMPONENT_MAXLEN`; one that would be truncated raises
    :class:`SeismicComponentError` rather than silently colliding with another dimension's
    posting list.

    ``top_k`` keeps only the heaviest components, with :data:`NO_TOP_K` keeping them all.
    MILCO's raw output averages several hundred non-zeros per passage, and the model's own
    ablation reports near-identical nDCG@10 down to a few hundred kept dimensions, so this
    is where the sparse leg's size is bounded. Two properties make the cut safe:

    * It is a total order, so the build stays reproducible. Ties on the weight break on the
      component id ascending, which is the same string order the returned pair is already
      sorted by, so the rule reads identically whether the caller's mapping is keyed by
      integer dimension ids or by the stored component strings.
    * The length cap is checked on every component, pruned or not, because it is a property
      of the vocabulary rather than of which dimensions survived. Checking only the
      survivors would let an unrepresentable vocabulary pass at one ``k`` and raise at
      another.

    The returned pair stays in component-string order whatever ``top_k`` is, so a pruned
    vector is exactly the surviving sub-sequence of the unpruned one.
    """
    components: list[str] = []
    values: list[float] = []
    for term, weight in sorted(weights.items(), key=lambda kv: str(kv[0])):
        component = str(term)
        if len(component) > SEISMIC_COMPONENT_MAXLEN:
            raise SeismicComponentError(
                f"sparse component {component!r} is {len(component)} chars, over Seismic's "
                f"U30 cap ({SEISMIC_COMPONENT_MAXLEN}); it would be truncated silently and "
                "merge two vocabulary dimensions into one posting list"
            )
        components.append(component)
        values.append(float(weight))
    if top_k > NO_TOP_K and len(components) > top_k:
        # Heaviest first, ties on the component string ascending, then back to positional
        # order, which is component-string order because the lists were built in it.
        keep = sorted(range(len(components)), key=lambda i: (-values[i], components[i]))
        kept = sorted(keep[:top_k])
        components = [components[i] for i in kept]
        values = [values[i] for i in kept]
    return components, values


class MilcoEncoder:
    """One resident MILCO replica, with identity, revision and device from config.

    ``model`` and ``revision`` are the semantic identity, hashed in ``index_build`` and
    recorded in the index manifest, while ``device`` is a machine fact. Nothing is loaded
    until the first encode call, so constructing the client imports neither ``torch`` nor
    ``transformers``.
    """

    __slots__ = (
        "_backend",
        "_tokenizer_obj",
        "batch_hint",
        "device",
        "max_length",
        "model",
        "revision",
    )

    def __init__(
        self,
        model: str,
        revision: str = "",
        device: str = "auto",
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_hint: int = 32,
    ) -> None:
        self.model = model
        self.revision = revision
        self.device = device
        self.max_length = int(max_length)
        # Only used on the convenience path where a caller hands over more than one bucket's
        # worth of text; the index build always passes exactly one Batcher bucket.
        self.batch_hint = int(batch_hint)
        self._backend: Any = None
        self._tokenizer_obj: Any = None

    # Backend, and the single device-placement call site.
    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def backend(self) -> Any:
        """The loaded vendor model, loaded once on first use.

        ``trust_remote_code=True`` is required, since the vendor's ``milco.py`` is the model
        implementation. Loading also resolves :data:`PIVOT_TOKENIZER_REPO` by repo id from
        the offline cache, so :func:`prefetch_pivot_tokenizer` must have run first.
        """
        if self._backend is None:
            from transformers import AutoModel

            kwargs: dict[str, Any] = {"trust_remote_code": True}
            if self.revision:
                kwargs["revision"] = self.revision
            model = AutoModel.from_pretrained(self.model, **kwargs)
            model.eval()
            device = self._resolve_device()
            # The only .to() call site for this model: the vendor's encode_text re-applies it
            # per call, but no stage ever does.
            model.to(device)
            _log.info(
                "serving.sparse_milco.loaded",
                model=self.model,
                revision=self.revision or "<unpinned>",
                device=device,
            )
            self._backend = model
        return self._backend

    # Encoding.
    def encode_text(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Encode one already-sized bucket into one ``{dim_id: weight}`` map per text.

        ``batch_size=len(texts)`` makes the caller's ``serving.batching.Batcher`` bucket the
        batch, so the vendor's internal loop runs exactly once and never becomes a second
        batching policy. ``return_dict=False`` keeps the gated ``naver/splade-v3`` vocabulary
        out of the path; the returned ids are integer vocabulary dimensions, which is all
        Seismic keys on.
        """
        items = list(texts)
        if not items:
            return []
        reps = self.backend().encode_text(
            items,
            batch_size=max(1, len(items)),
            max_length=self.max_length,
            return_dict=False,
        )
        return _sparse_rows(reps, len(items))

    def encode_query(self, query: str) -> dict[int, float]:
        """Encode one query into the same English SPLADE pivot space as the documents."""
        rows = self.encode_text([query])
        return rows[0] if rows else {}

    # Truncation diagnostic.
    def _tokenizer(self) -> Any:
        """The tokenizer this encoder truncates with: the model's own, never a proxy.

        Prefers whatever the loaded vendor model exposes, and otherwise resolves
        :data:`PIVOT_TOKENIZER_REPO` by repo id, which is the repository MILCO itself loads
        at bring-up, so the vocabulary is the same one rather than a look-alike. It is loaded
        once and cached, so the diagnostic never becomes a second model load per call.
        """
        if self._tokenizer_obj is None:
            tokenizer = getattr(self.backend(), "tokenizer", None)
            if tokenizer is None:
                from transformers import AutoTokenizer

                # No revision= here, for the reason given in prefetch_pivot_tokenizer.
                tokenizer = AutoTokenizer.from_pretrained(PIVOT_TOKENIZER_REPO)
            self._tokenizer_obj = tokenizer
        return self._tokenizer_obj

    def truncated_tokens(self, texts: Sequence[str]) -> list[int]:
        """Per text, how many tokens :meth:`encode_text` drops on the path that runs.

        This is the sparse leg's counterpart to
        ``serving.late_interaction.MtdEncoder.truncated_tokens`` and exists for the same
        reason: without it, a per-language truncation asymmetry is invisible. It is measured
        with the model's own tokenizer and ``truncation=False``, so the number is the real
        loss rather than a character-length proxy. At :data:`MODEL_MAX_LENGTH` it should read
        0, and any non-zero value means a passage is being indexed as a prefix of itself.
        """
        items = list(texts)
        if not items:
            return []
        encoded = self._tokenizer()(items, add_special_tokens=True, truncation=False)
        return [max(0, len(ids) - self.max_length) for ids in encoded["input_ids"]]


def _sparse_rows(reps: Any, n: int) -> list[dict[int, float]]:
    """Turn a torch sparse ``[N, V]`` tensor into N dicts, dropping non-positive weights.

    ``coalesce()`` runs first, because the vendor returns an uncoalesced sparse tensor and
    reading ``indices()`` off one raises. A dense or array-shaped return is tolerated so that
    a future vendor change surfaces as wrong numbers in a test rather than an AttributeError
    inside a build job.

    ``coalesce`` is the only discriminator, and ``indices`` and ``values`` are probed on the
    coalesced object rather than on ``reps`` itself, since on an uncoalesced tensor those are
    exactly the attributes that must not be touched.
    """
    sparse = reps.coalesce() if hasattr(reps, "coalesce") else reps
    if hasattr(sparse, "indices") and hasattr(sparse, "values"):
        rows: list[dict[int, float]] = [{} for _ in range(n)]
        idx = sparse.indices().tolist()
        vals = sparse.values().tolist()
        for row, dim, weight in zip(idx[0], idx[1], vals, strict=True):
            if weight > 0.0:
                rows[int(row)][int(dim)] = float(weight)
        return rows
    return [
        {int(dim): float(w) for dim, w in enumerate(row) if float(w) > 0.0}
        for row in list(reps)[:n]
    ]
