"""Pack tokenizer for the 512-token budget: XLM-R-large via ``BAAI/bge-m3``, pinned by revision.

The chunker's token budget is bound to the tightest-window consumer, the retriever
(PLAID-X and BGE-M3 are both XLM-R-large), rather than to NLLB's 512 cap. Counting
uses the standalone ``tokenizers`` package loading ``BAAI/bge-m3``'s ``tokenizer.json``
at a pinned revision sha, so the budget is reproducible. Special tokens are excluded
from the count (``count("") == 0``); the retriever's special tokens are reserved at
pack time via :meth:`PackTokenizer.num_special`.

``tokenizers`` is imported lazily inside :func:`_load` so importing this module never
pulls the Rust extension at collection time.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ragtime.config import RunConfig

__all__ = [
    "PackTokenizer",
    "load_pack_tokenizer",
    "load_pack_tokenizer_by_id",
    "parse_tokenizer_id",
]


def parse_tokenizer_id(tokenizer_id: str) -> tuple[str, str | None]:
    """Split ``"<repo>@<revision>"`` into ``(repo, revision)``; revision is ``None`` if absent.

    The pinned form is ``BAAI/bge-m3@<sha>``; the revision is what makes the budget
    reproducible.
    """
    if "@" in tokenizer_id:
        repo, rev = tokenizer_id.split("@", 1)
        return repo, (rev or None)
    return tokenizer_id, None


@cache
def _load(repo: str, revision: str | None) -> Any:
    """Load (and module-cache) the ``tokenizers.Tokenizer`` for a pinned repo@revision."""
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(repo, revision=revision)


class PackTokenizer:
    """A thin, deterministic length counter over one pinned ``tokenizers.Tokenizer``."""

    __slots__ = ("_tok",)

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    def count(self, text: str) -> int:
        """Number of content tokens in ``text`` (special tokens excluded; ``count("")==0``)."""
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_batch(self, texts: Sequence[str]) -> list[int]:
        """Apply :meth:`count` to many texts in one call, using the Rust batch path.

        Semantics are identical to calling :meth:`count` per text. ``encode_batch_fast``
        skips building the offset and word-id side tables, which are not read here; older
        ``tokenizers`` builds without it fall back to ``encode_batch``.
        """
        items = list(texts)
        if not items:
            return []
        fast = getattr(self._tok, "encode_batch_fast", None)
        encode = fast if fast is not None else self._tok.encode_batch
        return [len(enc.ids) for enc in encode(items, add_special_tokens=False)]

    def num_special(self) -> int:
        """Special tokens the model adds to one encoded sequence (2 for bge-m3).

        XLM-R and bge-m3 wrap a single sequence as ``<s> ... </s>``. The chunker packs to
        ``token_budget - num_special()`` content tokens so a full passage encoded with
        specials stays within the model's 512-token window.
        """
        return self._tok.num_special_tokens_to_add(is_pair=False)


def load_pack_tokenizer_by_id(tokenizer_id: str) -> PackTokenizer:
    """Build the pack tokenizer from a ``"<repo>@<revision>"`` id string.

    The multiprocessing chunk workers use this because a plain string crosses the
    ``spawn`` boundary while a ``RunConfig`` need not.
    """
    repo, revision = parse_tokenizer_id(tokenizer_id)
    return PackTokenizer(_load(repo, revision))


def load_pack_tokenizer(cfg: RunConfig) -> PackTokenizer:
    """Build the pack tokenizer from ``cfg.chunker.config.tokenizer_id``."""
    tid = cfg.blocks["chunker"]["config"]["tokenizer_id"]
    repo, revision = parse_tokenizer_id(tid)
    return PackTokenizer(_load(repo, revision))
