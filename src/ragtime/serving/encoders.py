"""The dense encoder of the index build and of decompose's dedup pre-filter.

The three retrieval legs are three separate checkpoints, not three outputs of one
model: this class serves the dense leg (``BAAI/bge-m3``, through
``sentence-transformers``), ``serving.sparse_milco`` serves the learned-sparse leg and
``serving.late_interaction`` the late-interaction leg. Retrieval and dedup share one
resident dense encoder, because the registry hands the same object to both call sites.

``embed`` takes a ``mode`` and ``"dense"`` is the only one it serves. It is kept explicit at
the call sites, rather than dropped, because this module names one of three legs and a bare
``embed`` would read as "the encoder" and invite the tri-output reading the three separate
checkpoints exist to rule out.

Heavy libraries are imported inside the backend on first use, so importing this
module and constructing the client pulls in nothing large.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DEFAULT_INDEX_DENSE_MODEL", "Encoder"]

#: Fallback identity of the index build's dense leg, used only when
#: ``index_build.config.dense_model`` is absent; a run's value normally comes from that
#: hashed block. It lives beside its client for the same reason as
#: ``sparse_milco.DEFAULT_MILCO_MODEL`` and ``late_interaction.DEFAULT_MTD_CHECKPOINT``:
#: one literal per model identity, imported by ``preprocess.index`` rather than copied.
DEFAULT_INDEX_DENSE_MODEL = "BAAI/bge-m3"


class Encoder:
    """One resident dense encoder, with the model id taken from config.

    ``revision`` pins the checkpoint the way ``serving.late_interaction`` and
    ``serving.sparse_milco`` do: the index manifest records it, so an unpinned load
    would let the weights move under an already-built index. It is forwarded to
    ``sentence-transformers`` only when non-empty, so leaving it unset keeps the
    library's own resolution.
    """

    __slots__ = ("_backend", "device", "model", "revision")

    def __init__(self, model: str, device: str = "cuda", *, revision: str = "") -> None:
        self.model = model
        self.device = device
        self.revision = revision
        self._backend: Any = None

    def _resolve_device(self) -> str:
        """Resolve ``"auto"`` to cuda when available and cpu otherwise.

        Only ``"auto"`` imports torch; any other value is returned untouched, so no
        import is added to the default path.
        """
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _dense_backend(self) -> Any:
        if self._backend is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"device": self._resolve_device()}
            if self.revision:
                kwargs["revision"] = self.revision
            self._backend = SentenceTransformer(self.model, **kwargs)
        return self._backend

    def embed(self, texts: list[str], mode: str = "dense") -> Any:
        """Encode ``texts``. ``mode`` must be ``"dense"``, the only leg this class serves.

        Passing ``batch_size=len(texts)`` on the dense leg makes the caller's
        ``serving.batching.Batcher`` bucket the batch, as in
        :meth:`serving.sparse_milco.MilcoEncoder.encode_text`. Left to itself
        ``sentence-transformers`` defaults to ``batch_size=32`` and length-sorts
        internally, which re-partitions the bucket by text length; since length differs
        across renderings of the same passage, that would make batch composition
        rendering-dependent. Holding composition fixed is what keeps repeated encodes
        bitwise identical, so there is one batching policy in the system and it lives in
        the ``Batcher``.
        """
        if mode != "dense":
            raise ValueError(
                f"unknown embed mode {mode!r}; this client is the dense leg only. The "
                "learned-sparse leg is serving.sparse_milco and the late-interaction leg is "
                "serving.late_interaction, three separate checkpoints rather than three "
                "outputs of one model."
            )
        return self._dense_backend().encode(
            texts, batch_size=max(1, len(texts)), normalize_embeddings=True
        )
