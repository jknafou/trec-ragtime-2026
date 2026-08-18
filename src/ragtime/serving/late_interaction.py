"""Late-interaction (multi-vector) encoder behind one client.

This is the index's third leg: HLTCOE's Multilingual Translate-Distill ColBERT checkpoint
served through PyLate, which loads the checkpoint natively (it detects
``architectures[0] == "HF_ColBERT"``, calls ``Dense.from_stanford_weights`` and reads the
prefixes and window lengths out of ``artifact.metadata``), so no hand-rolled safetensors
wrapper is needed.

Of the two Multilingual Translate-Distill checkpoints published, coverage is disjoint. The
one adopted here covers two of the three non-English corpus languages and scored slightly
better on the third, Spanish, which it was never trained on, than its sibling did with
Spanish in training. The cross-lingual alignment comes from the XLM-R backbone rather than
from the cross-language fine-tuning.

Document truncation is a real concern for this leg: Chinese passages tokenize longer than
their English or Russian counterparts, so at the checkpoint's own ``document_length`` they
are the ones that get cut, which would introduce a per-language bias from the retrieval
method rather than from the corpus. The remedy lives upstream in ``preprocess.packing``,
which packs against each sentence's length maximised over the three renderings so every
passage fits the retrieval window in every rendering by construction, and
``preprocess.index`` asserts at bring-up that the window is at least the pack budget.
:meth:`MtdEncoder.truncated_tokens` remains as the alarm that fires if that upstream
guarantee ever stops holding.

Heavy imports (``pylate`` and ``torch``) happen inside the backend on first use, so
importing this module and constructing the client stays cheap on a CPU-only machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragtime.common import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_DOCUMENT_LENGTH",
    "DEFAULT_MTD_CHECKPOINT",
    "DEFAULT_MTD_REVISION",
    "DEFAULT_QUERY_LENGTH",
    "MtdEncoder",
]

_log = get_logger("serving.late_interaction")

#: The adopted checkpoint and its git-SHA pin. These are fallbacks; a run takes its values
#: from the hashed ``index_build.config`` block. The sibling checkpoint
#: ``plaidx-large-clef-mtd-mix-passages-mt5xxl-engeng`` covers en->de/es/fr rather than the
#: en->zh/fa/ru this one covers.
DEFAULT_MTD_CHECKPOINT = "hltcoe/plaidx-large-neuclir-mtd-mix-passages-mt5xxl-engeng"
DEFAULT_MTD_REVISION = "1fb92d4d51a3e60d9942c0654e71428f29c7b5d8"

#: The adopted checkpoint's own window lengths, read off its ``artifact.metadata``. These are
#: fallbacks: a caller passes the values from the hashed ``index_build`` block, and passing 0
#: lets the checkpoint's own metadata decide, which PyLate reads natively.
DEFAULT_DOCUMENT_LENGTH = 220
DEFAULT_QUERY_LENGTH = 32

#: Positions the ColBERT document path costs on top of the two sequence specials a pack
#: budget already reserves: PyLate prepends a document marker token, so a text of *n* content
#: tokens occupies ``n + 3`` positions. A pack budget of ``B`` therefore needs a document
#: window of at least ``B + 1``, and at ``B == document_length`` a budget-filling passage
#: loses exactly one token.
DOCUMENT_MARKER_TOKENS = 1


class MtdEncoder:
    """One resident PyLate ColBERT replica, configured by identity, revision, device and
    window lengths.

    ``encode_docs`` and ``encode_query`` are the only calls the index build makes.
    ``encode_docs`` takes one already-sized ``serving.batching.Batcher`` bucket, so this
    client owns no batching policy of its own, matching ``serving.sparse_milco``.
    """

    __slots__ = (
        "_backend",
        "checkpoint",
        "device",
        "document_length",
        "query_length",
        "revision",
    )

    def __init__(
        self,
        checkpoint: str,
        revision: str = "",
        device: str = "auto",
        *,
        document_length: int = 0,
        query_length: int = 0,
    ) -> None:
        self.checkpoint = checkpoint
        self.revision = revision
        self.device = device
        # 0 means do not override; PyLate then reads the checkpoint's own metadata.
        self.document_length = int(document_length)
        self.query_length = int(query_length)
        self._backend: Any = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def backend(self) -> Any:
        """The loaded ``pylate.models.ColBERT`` (loaded once, on first use)."""
        if self._backend is None:
            from pylate import models

            kwargs: dict[str, Any] = {
                "model_name_or_path": self.checkpoint,
                "device": self._resolve_device(),
            }
            if self.revision:
                kwargs["revision"] = self.revision
            # Only override what config actually declared: the checkpoint carries its own
            # lengths and PyLate honours them, so an unset knob must not silently become a
            # different truncation window than the one the checkpoint was fit on.
            if self.document_length:
                kwargs["document_length"] = self.document_length
            if self.query_length:
                kwargs["query_length"] = self.query_length
            self._backend = models.ColBERT(**kwargs)
            _log.info(
                "serving.late_interaction.loaded",
                checkpoint=self.checkpoint,
                revision=self.revision or "<unpinned>",
                device=kwargs["device"],
                document_length=self.document_length or "<checkpoint>",
                query_length=self.query_length or "<checkpoint>",
            )
        return self._backend

    # Tokenizer used by the truncation counter.
    def _tokenizer(self) -> Any:
        tokenizer = getattr(self.backend(), "tokenizer", None)
        if tokenizer is None:  # pragma: no cover - vendor surface change
            raise AttributeError(
                "the loaded PyLate model exposes no .tokenizer, so the truncation counter "
                "cannot be computed; reporting 0 would hide the asymmetry"
            )
        return tokenizer

    # Encoding.
    def encode_docs(self, texts: Sequence[str]) -> list[Any]:
        """Encode one bucket of passages into one multi-vector matrix per passage."""
        items = list(texts)
        if not items:
            return []
        out = self.backend().encode(
            items,
            is_query=False,
            batch_size=max(1, len(items)),
            show_progress_bar=False,
        )
        return list(out)

    def encode_query(self, query: str) -> Any:
        """Encode one query with the query prefix and length regime."""
        out = self.backend().encode(
            [query],
            is_query=True,
            batch_size=1,
            show_progress_bar=False,
        )
        return next(iter(out))

    # Truncation diagnostic.
    def effective_document_length(self) -> int:
        """The window documents are actually truncated to (config override or checkpoint)."""
        if self.document_length:
            return self.document_length
        backend = self.backend()
        for attr in ("document_length", "max_seq_length"):
            value = getattr(backend, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return DEFAULT_DOCUMENT_LENGTH

    def document_prefix(self) -> str:
        """The document marker PyLate prepends before encoding, one token wide.

        Read off the loaded model rather than hardcoded, since it is a property of the
        checkpoint's ColBERT head and a wrong literal would make the truncation counter
        under-report.
        """
        return str(getattr(self.backend(), "document_prefix", "") or "")

    def truncated_tokens(self, texts: Sequence[str]) -> list[int]:
        """Per text, how many tokens the document window drops on the path that runs.

        Measured with the checkpoint's own tokenizer, without truncation, over the sequence
        PyLate actually builds for a document: ``document_prefix + text`` plus the two
        sequence specials. The prefix occupies a position, so omitting it would under-report
        by one token and report zero for a passage sitting exactly on the budget while the
        encoder really dropped a token.

        This is an alarm rather than a remedy. Packing upstream guarantees every passage
        fits the window in every rendering, so this should read 0; a non-zero value means a
        pack budget larger than the window, or an inventory packed under a different recipe.
        """
        items = list(texts)
        if not items:
            return []
        limit = self.effective_document_length()
        prefix = self.document_prefix()
        encoded = self._tokenizer()(
            [prefix + text for text in items], add_special_tokens=True, truncation=False
        )
        return [max(0, len(ids) - limit) for ids in encoded["input_ids"]]
