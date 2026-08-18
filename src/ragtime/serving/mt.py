"""Machine translation behind one call, with the engine isolated from callers.

``translate_batch(items, ...) -> {sentence_id: english}``. It serves both translation-tier
arms on CTranslate2: the high tier (``facebook/nllb-200-3.3B``, one multilingual checkpoint)
and the low tier (``Helsinki-NLP/opus-mt-{es,ru,zh}-en``, three bilingual Marian checkpoints
selected per direction). ``translation.config.model_family`` picks the source-sequence
layout; see :data:`FAMILY_NLLB` and :data:`FAMILY_MARIAN`. The organisers' shipped
translations are not used, so there is no alignment arm and no Sockeye engine here.

Engine, semantic model id, compute type and the on-disk CTranslate2 checkpoint directory all
come from config; see :class:`MtClient`. The client is built only by
``serving.registry.build_clients``, and heavy imports are lazy, so importing this module
never touches CUDA.

Items arrive pre-tokenized and mixed-language, one length bucket per call. Bucket
composition belongs to the caller, since putting it here would leave the index leg without a
reusable composer. This module owns the three things only the MT engine can own: the source
token layout, the per-bucket decode cap, and the CTranslate2 call itself, including its
out-of-memory retry ladder.

Note that the source sequence must not be built via ``tokenizer.src_lang``; see
:meth:`MtClient.tokenize`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping  # runtime import: `_pick` isinstance-checks it
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from .batching import Tier

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_COMPUTE_TYPE",
    "DEFAULT_MAX_INPUT_LENGTH",
    "EOS_TOKEN",
    "FAMILY_MARIAN",
    "FAMILY_NLLB",
    "MtClient",
    "MtSentence",
    "decode_cap",
    "truncate_tokens",
]

#: The two source-sequence layouts this client knows, selected by the hashed
#: ``translation.config.model_family``. They differ in the bytes sent to the model, which is
#: why the knob is semantic rather than an execution detail:
#:
#: - ``nllb``: one multilingual checkpoint. The source is ``[src_lang, *content, "</s>"]``
#:   and the decoder is forced with ``target_prefix=[[target_lang]]``.
#: - ``marian``: N bilingual checkpoints, one per direction. The direction is the checkpoint,
#:   so the source carries no language token, just ``[*content, "</s>"]``, and there is no
#:   target prefix; forcing one would emit a literal language token that Marian's vocabulary
#:   does not contain.
FAMILY_NLLB = "nllb"
FAMILY_MARIAN = "marian"
_FAMILIES = frozenset({FAMILY_NLLB, FAMILY_MARIAN})

#: The end-of-sequence token, always the last token of a source sequence.
EOS_TOKEN = "</s>"
#: The source-length cap. CTranslate2's own default is 1024, so this is always passed
#: explicitly.
DEFAULT_MAX_INPUT_LENGTH = 512
#: The converted checkpoint's saved quantization. It is passed explicitly to CTranslate2 and
#: read from ``translation.config.compute_type`` by the registry, so the hashed record and the
#: engine cannot disagree.
DEFAULT_COMPUTE_TYPE = "float16"
#: SentencePiece's word-start prefix (U+2581); a piece may be just the marker with or without
#: it. Used only by :meth:`MtClient.assert_marker_atomic`.
_SP_WORD_PREFIX = "▁"

# CT2 may surface an allocator failure as a RuntimeError carrying the CUDA message
# rather than as MemoryError; match on the message (lowercased) as well.
_OOM_MARKERS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "bad_alloc",
)


class MtSentence(NamedTuple):
    """One sentence or merge unit to translate, pre-tokenized at gather time.

    ``src_lang`` is a FLORES-200 code (``spa_Latn``, ``rus_Cyrl``, ``zho_Hans``,
    ``zho_Hant``); English never reaches this client, as the CPU-only identity substage
    handles it. ``tokens`` is the full source sequence produced by
    :meth:`MtClient.tokenize`.

    The corpus-side caller passes its own structurally identical item,
    ``preprocess.translate_omt.MtInput``. Every read below is by attribute, so the two never
    have to be the same class. That caller also resolves the FLORES code, since it holds the
    text that decides simplified from traditional Chinese, so there is no second language
    table in this module.
    """

    sentence_id: str
    src_lang: str
    text: str
    tokens: tuple[str, ...] = ()


def truncate_tokens(
    tokens: Sequence[str],
    max_input_length: int = DEFAULT_MAX_INPUT_LENGTH,
    *,
    leading_specials: int = 1,
) -> tuple[str, ...]:
    """Truncate a source sequence to ``max_input_length``, keeping its special tokens.

    A naive ``tokens[:512]`` drops the trailing ``</s>`` the model needs in order to
    terminate, so the content is cut and the specials are re-attached.

    ``leading_specials`` is how many tokens at the front are special rather than content: 1 for
    the NLLB layout ``[src_lang, *content, "</s>"]`` and 0 for the Marian layout
    ``[*content, "</s>"]``, where the first token is real content and keeping it while dropping
    its successors would splice the sentence.
    """
    t = tuple(tokens)
    if len(t) <= max_input_length:
        return t
    keep = leading_specials + 1  # the leading specials plus the trailing EOS
    if max_input_length < keep:
        raise ValueError(f"max_input_length must be >= {keep}, got {max_input_length}")
    head = t[:leading_specials]
    body = t[leading_specials : max_input_length - 1]
    return (*head, *body, t[-1])


def decode_cap(
    items: Sequence[Any],
    *,
    len_ratio_a: Mapping[str, float],
    len_ratio_b: float = 10.0,
    max_decoding_cap: int = DEFAULT_MAX_INPUT_LENGTH,
    require_ratio: bool = False,
) -> int:
    """Return the direction-aware decode cap for one bucket.

    The cap is a scalar, since CTranslate2 takes no per-item cap. It is
    ``ceil(max_ratio * bucket_max_src_tokens + len_ratio_b)`` clamped to ``max_decoding_cap``,
    where ``max_ratio`` is the largest configured expansion ratio among the languages present
    in the bucket. Buckets are formed by length alone and may be mixed-language, so taking the
    maximum keeps the longest-expanding direction from being clipped.
    """
    if not items:
        raise ValueError("decode_cap: empty bucket")
    if require_ratio:
        # A miss here would be silent and destructive: the ratio falls back to 1.0, the cap
        # tightens to roughly the source length, and a direction that genuinely expands is
        # truncated mid-sentence with no exception and no counter. The keys must therefore
        # match whatever the arm sets as `src_lang` on its items, which is why both arms
        # label directions with the same FLORES codes even though only the multilingual one
        # sends that code to the model.
        missing = sorted({it.src_lang for it in items} - set(len_ratio_a))
        if missing:
            raise KeyError(
                f"translation.config.len_ratio_a has no entry for {missing}; the decode "
                f"cap would silently fall back to ratio 1.0 and truncate the output "
                f"(configured: {sorted(len_ratio_a)})"
            )
    max_src = max(len(it.tokens) for it in items)
    ratio = max(float(len_ratio_a.get(it.src_lang, 1.0)) for it in items)
    return max(1, min(int(max_decoding_cap), math.ceil(ratio * max_src + len_ratio_b)))


def _check_family(family: str) -> str:
    if family not in _FAMILIES:
        raise ValueError(
            f"unsupported translation.config.model_family {family!r}: the source-sequence "
            f"layout must be one of {sorted(_FAMILIES)}, and guessing it from the checkpoint "
            "would let a model swap silently change the bytes sent to MT"
        )
    return family


def _pick(value: Any, src_lang: str, key: str) -> str:
    """Resolve a knob that is either one value or a per-source-language mapping.

    A missing direction is a hard error rather than a silent fallback to another language's
    checkpoint, which would translate Chinese through the Spanish model and show up only as
    bad English that nobody could attribute.
    """
    if isinstance(value, Mapping):
        try:
            return str(value[src_lang])
        except KeyError:
            raise KeyError(
                f"{key} has no entry for source language {src_lang!r}; a per-direction "
                f"arm must map every language in translation.config.source_langs "
                f"(configured: {sorted(value)})"
            ) from None
    return str(value)


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _OOM_MARKERS)


class MtClient:
    """One MT engine, with identity, compute type and checkpoint path taken from config.

    ``model`` and ``model_path`` are two different facts and are not collapsed:

    - ``model`` is the semantic model identity, ``translation.config.omt_model``. It lives in
      the fairness-shared, hashed block and lands on every ``Translation`` row's ``model_id``,
      so it must never be a filesystem path.
    - ``model_path`` is the machine-local directory holding the converted checkpoint,
      ``execution.ct2_model_dir``. It is unshared and unhashed, so it may differ per machine
      without moving the fairness hash.

    ``compute_type`` is a third semantic knob, ``translation.config.compute_type``, passed
    explicitly to CTranslate2 rather than left to the checkpoint's saved quantization. Leaving
    it implicit lets the hashed block declare one precision while another actually runs.

    Only ``serving.registry.build_clients`` constructs this, and it is what validates
    ``model_path`` before a GPU is allocated.
    """

    __slots__ = (
        "_lang_checked",
        "_toks",
        "_translators",
        "compute_type",
        "device",
        "engine",
        "family",
        "model",
        "model_path",
    )

    def __init__(
        self,
        model: str | Mapping[str, str],
        engine: str = "ctranslate2",
        device: str = "cuda",
        *,
        model_path: str | Mapping[str, str] | None = None,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        family: str = FAMILY_NLLB,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.engine = engine
        self.device = device
        self.compute_type = compute_type
        self.family = _check_family(family)
        # Keyed by the resolved checkpoint directory rather than by source language: the
        # Opus-MT arm maps both Chinese scripts onto one checkpoint, and keying by path makes
        # that a single resident replica instead of two identical copies.
        self._translators: dict[str, Any] = {}
        self._toks: dict[str, Any] = {}
        self._lang_checked: set[str] = set()

    # Per-direction resolution.
    def per_language(self) -> bool:
        """True when this arm is N bilingual checkpoints rather than one multilingual one."""
        return isinstance(self.model_path, Mapping)

    def model_id_for(self, src_lang: str) -> str:
        """The semantic identity stamped on this direction's rows, never a path.

        It is a scalar for a multilingual arm and per-direction for a bilingual arm, where
        ``model_id`` genuinely differs row to row because a different checkpoint produced the
        text.
        """
        return _pick(self.model, src_lang, "translation.config.omt_model")

    def _resolve_path(self, src_lang: str) -> str:
        """This direction's configured checkpoint directory, which is the replica cache key.

        It does not touch the filesystem: it is called on every batch to pick a
        replica, and it must return the same key for an already-resident model whether or not
        the directory is still reachable.
        """
        # There is no fallback to `self.model`, which is a semantic identity rather than a
        # path; handing it to CTranslate2 is the mis-wiring this split exists to prevent.
        if not self.model_path:
            raise ValueError(
                f"MtClient({self.model!r}) has no on-disk model_path; the "
                "CTranslate2 checkpoint directory comes from "
                "execution.ct2_model_dir via serving.registry.build_clients"
            )
        return _pick(self.model_path, src_lang, "execution.ct2_model_dir")

    def _require_path(self, src_lang: str) -> str:
        """:meth:`_resolve_path` plus the existence check, called only before a load."""
        path = self._resolve_path(src_lang)
        # Checked here, at first use, rather than in build_clients: validating at construction
        # would couple every consumer of the bundle to an artifact it may never touch, such as
        # a chunk worker or an online node standing up vLLM. Note that ct2_model_dir is
        # typically relative, so this resolves against the process working directory.
        if not Path(path).is_dir():
            raise ValueError(
                f"execution.ct2_model_dir {path!r} (source language {src_lang!r}) is not a "
                f"directory; the CTranslate2 checkpoint for "
                f"{self.model_id_for(src_lang)!r} must exist on this machine before a "
                f"translate replica is allocated (resolved from cwd {Path.cwd()!r})"
            )
        return path

    def _load_tok(self, src_lang: str) -> Any:
        """Load only this direction's tokenizer, never the translation weights.

        Split out from :meth:`_load` so the bring-up assertions, the source-token layout and
        the merge join marker, can be checked against the real tokenizer without a GPU and
        without reading the checkpoint off shared storage. The translate substage's bring-up
        runs before it has claimed a shard and must not pay a weight load to discover that a
        tokenizer swap broke the pipeline.
        """
        path = self._resolve_path(src_lang)
        tok = self._toks.get(path)
        if tok is None:
            self._require_path(src_lang)  # existence is checked at first use
            from transformers import AutoTokenizer

            # The converted checkpoint directory carries the copied tokenizer files, so this
            # resolves offline with no repository lookup on a compute node.
            tok = AutoTokenizer.from_pretrained(path)
            self._toks[path] = tok
        return tok

    def _load(self, src_lang: str) -> tuple[Any, Any]:
        """This direction's (translator, tokenizer), loading each at most once."""
        if self.engine != "ctranslate2":
            raise ValueError(f"unsupported MT engine {self.engine!r} (only ctranslate2)")
        path = self._resolve_path(src_lang)
        tr = self._translators.get(path)
        if tr is None:
            self._require_path(src_lang)  # validated before any heavy import
            import ctranslate2

            # `compute_type` is passed explicitly rather than left to CTranslate2's "use
            # whatever the checkpoint saved" default. That default is not wrong, but it is
            # unverifiable, and it would let `translation.config` declare one precision while
            # another ran with nothing in the run record able to contradict it. The value
            # comes from the hashed block, so config and engine state the same fact by
            # construction, and declaring a type the checkpoint cannot serve fails here at
            # bring-up.
            tr = ctranslate2.Translator(path, device=self.device, compute_type=self.compute_type)
            self._translators[path] = tr
        return tr, self._load_tok(src_lang)

    # Tokenization: the source-token layout is owned here and asserted at bring-up.
    def _assert_lang_layout(self, src_lang: str) -> None:
        """Assert once per language that the special tokens resolve.

        On the pinned ``transformers`` release the ``tokenizer.src_lang`` path is broken:
        setting it and tokenizing emits no leading language token and appends ``<unk>`` for
        the code. The model then gets no source-language signal and never terminates,
        returning long runs of repeated punctuation with no exception and no counter, which
        nothing else in the pipeline would notice. This assertion is therefore mandatory
        rather than defensive: a future tokenizer upgrade must fail here.

        For the ``marian`` family the direction is the checkpoint, so there is no language
        token to resolve and only the ``</s>`` half applies, but it applies just as hard.
        Marian's ``eos_token_id`` is 0, a falsy id that an ``if not eos_id`` check would treat
        as missing, and without a real end-of-sequence token the model never terminates.
        """
        if src_lang in self._lang_checked:
            return
        tok = self._load_tok(src_lang)
        unk = getattr(tok, "unk_token_id", None)
        wanted = [EOS_TOKEN] if self.family == FAMILY_MARIAN else [src_lang, EOS_TOKEN]
        ids = tok.convert_tokens_to_ids(wanted)
        eos_id = ids[-1]
        if self.family != FAMILY_MARIAN:
            lang_id = ids[0]
            if lang_id is None or lang_id == unk:
                raise RuntimeError(
                    f"NLLB source-token layout broken: language token {src_lang!r} does not "
                    f"resolve in this tokenizer's vocabulary (got id {lang_id!r}, "
                    f"unk={unk!r}), so the sequence would carry no source-language signal"
                )
        if eos_id is None or eos_id == unk:
            raise RuntimeError(
                f"{self.family} source-token layout broken: {EOS_TOKEN!r} does not resolve "
                f"(got id {eos_id!r}, unk={unk!r}), so the model would never terminate"
            )
        self._lang_checked.add(src_lang)

    def assert_marker_atomic(self, marker: str, src_lang: str = "spa_Latn") -> None:
        """Assert that the merged-unit join marker is one atomic piece in this tokenizer.

        ``preprocess.translate_omt`` joins a merged unit's constituents with ``marker`` and
        splits the English back on it. That works only because the SentencePiece vocabulary
        encodes the marker as a single piece and decodes it losslessly, which is a property of
        the pinned tokenizer rather than a guarantee of the format. A tokenizer or model swap
        that fused the marker into its neighbours, or mapped it to ``<unk>``, would raise
        nothing by itself: every merged unit would come back unsplittable and the corpus would
        quietly lose most of the granularity the marker exists to preserve.

        The probe checks, on the real tokenize path, that the marker encodes as exactly one
        piece, that the piece carries nothing but the marker apart from SentencePiece's
        word-start prefix, that it is not ``<unk>``, and that it survives the round trip back
        to text. It uses the tokenizer only, so bring-up pays no weight load.
        """
        tok = self._load_tok(src_lang)
        self._assert_lang_layout(src_lang)
        probe = f"A {marker} B"
        ids = tok(probe, add_special_tokens=False).input_ids
        pieces = tok.convert_ids_to_tokens(ids)
        carrying = [(i, p) for i, p in zip(ids, pieces, strict=True) if marker in p]
        unk = getattr(tok, "unk_token_id", None)
        if len(carrying) != 1:
            raise RuntimeError(
                f"merge join marker {marker!r} is not atomic in this tokenizer: it "
                f"encodes to {len(carrying)} piece(s) of {pieces!r}, so the split-back "
                "would silently stop recovering per-sentence translations"
            )
        piece_id, piece = carrying[0]
        if piece.removeprefix(_SP_WORD_PREFIX) != marker or piece_id == unk:
            raise RuntimeError(
                f"merge join marker {marker!r} does not survive this tokenizer: it came "
                f"back as piece {piece!r} (id {piece_id!r}, unk={unk!r}); it must be its "
                "own piece, fused to no neighbouring character"
            )
        if marker not in tok.decode(ids, skip_special_tokens=True):
            raise RuntimeError(
                f"merge join marker {marker!r} does not round-trip through this "
                "tokenizer's decode, so the split-back would never find it in the output"
            )

    def tokenize(
        self,
        text: str,
        src_lang: str,
        *,
        max_input_length: int = DEFAULT_MAX_INPUT_LENGTH,
        stats: Any = None,
    ) -> tuple[str, ...]:
        """Build one source sequence explicitly, as ``[src_lang, *content, "</s>"]``.

        It is never built via ``tokenizer.src_lang``; see :meth:`_assert_lang_layout`, since
        that attribute is broken on the pinned ``transformers`` and its failure is silent.
        Building the sequence here also keeps tokenization pure, with no stateful mutation of
        the whole tokenizer, so it is parallel-safe and can happen once at gather time rather
        than repeatedly per sentence.

        Over-long sequences are truncated by :func:`truncate_tokens`, keeping both specials,
        and counted as ``translate.truncated_at_max_input``. Token counts are not persisted:
        the sentence spine is keyed by the ``chunker`` hash while a token length is a
        ``translation``-block fact, so storing it there would cross two hash boundaries.
        """
        _, tok = self._load(src_lang)
        self._assert_lang_layout(src_lang)
        ids = tok(text, add_special_tokens=False).input_ids
        core = tok.convert_ids_to_tokens(ids)
        # The direction is carried by the language token on a multilingual checkpoint and by
        # the checkpoint itself on a bilingual one, so prefixing a Marian sequence with a
        # language code would feed it an out-of-vocabulary string as if it were content.
        seq: tuple[str, ...] = (
            (*core, EOS_TOKEN) if self.family == FAMILY_MARIAN else (src_lang, *core, EOS_TOKEN)
        )
        if len(seq) > max_input_length:
            if stats is not None:
                stats.emit("translate.truncated_at_max_input", lang=src_lang)
            seq = truncate_tokens(
                seq,
                max_input_length,
                leading_specials=0 if self.family == FAMILY_MARIAN else 1,
            )
        return seq

    # Translation: one bucket per call.
    def translate_batch(
        self,
        items: Sequence[Any],
        *,
        target_lang: str = "eng_Latn",
        beam_size: int = 4,
        len_ratio_a: Mapping[str, float],
        len_ratio_b: float = 10.0,
        max_decoding_cap: int = DEFAULT_MAX_INPUT_LENGTH,
        max_input_length: int = DEFAULT_MAX_INPUT_LENGTH,
        tier: Tier | None = None,
        no_repeat_ngram_size: int,
        stats: Any = None,
    ) -> dict[str, str]:
        """Translate one pre-tokenized, length-bucketed, possibly mixed-language bucket.

        Returns ``{sentence_id: english_text}``. Four properties are contract rather than
        preference:

        - Items arrive pre-tokenized in ``MtSentence.tokens``; this call never tokenizes, so
          there is no per-item tokenizer state to race.
        - Exactly one bucket per CTranslate2 call. Bucket composition belongs to the caller,
          and CTranslate2 re-sorts and re-splits internally once ``max_batch_size`` binds,
          which would take composition away from the caller.
        - Deterministic beam search with no sampling, so there is no random state here.
        - Every memory knob is passed explicitly: ``max_input_length``, which otherwise
          defaults to 1024; ``max_batch_size`` with ``batch_type="tokens"``, whose default of
          0 makes token batching inert; ``max_decoding_length``; and ``beam_size``.

        ``no_repeat_ngram_size`` is required and has no default. It materially
        changes the output text, since a repetition loop otherwise drags the whole batch to
        the decode cap while blocking too aggressively corrupts legitimately repeating text
        such as dates, so it is a hashed semantic knob in ``translation.config``. A default
        here would let a config omit it and still change bytes.

        Only ``tier.token_budget`` is read, as CTranslate2's token-denominated
        ``max_batch_size`` safety clamp; ``batch_type="tokens"`` makes that argument a token
        count rather than a sentence count. The sentence cap belongs upstream, in bucket
        composition, where it is a hashed semantic knob.
        """
        bucket = list(items)
        if not bucket:
            return {}
        missing = [it.sentence_id for it in bucket if not it.tokens]
        if missing:
            raise ValueError(
                f"translate_batch: {len(missing)} item(s) arrived un-tokenized "
                f"(first: {missing[0]!r}); tokenize once, at gather time"
            )
        t = tier or Tier()
        out: dict[str, str] = {}
        # Buckets are composed by length alone, never by language, so a bucket may be
        # mixed-language. A multilingual arm translates it in one call, which is the point of
        # the language token. A per-direction arm cannot: two languages mean two checkpoints,
        # so the bucket is split by resolved checkpoint and each group gets its own call. The
        # split is a deterministic function of the bucket's own contents rather than of the
        # GPU, so the same shard composes the same batches on every card. For a multilingual
        # arm there is exactly one group.
        for group in self._by_checkpoint(bucket):
            cap = decode_cap(
                group,
                len_ratio_a=len_ratio_a,
                len_ratio_b=len_ratio_b,
                max_decoding_cap=max_decoding_cap,
                require_ratio=True,  # a missing direction is a bug, not a 1.0 fallback
            )
            results = self._call(
                [list(it.tokens) for it in group],
                src_lang=group[0].src_lang,
                target_lang=target_lang,
                beam_size=beam_size,
                decode_length=cap,
                max_input_length=max_input_length,
                max_batch_size=t.token_budget,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stats=stats,
            )
            tok = self._load_tok(group[0].src_lang)
            for it, res in zip(group, results, strict=True):
                out[it.sentence_id] = self._detokenize(res, target_lang, tok)
        return out

    def _by_checkpoint(self, bucket: list[Any]) -> list[list[Any]]:
        """Split the bucket into groups one replica can serve, preserving order."""
        if not self.per_language():
            return [bucket]
        groups: dict[str, list[Any]] = {}
        for it in bucket:
            groups.setdefault(self._resolve_path(it.src_lang), []).append(it)
        return list(groups.values())

    def _call(
        self,
        tokenized: list[list[str]],
        *,
        src_lang: str,
        target_lang: str,
        beam_size: int,
        decode_length: int,
        max_input_length: int,
        max_batch_size: int,
        no_repeat_ngram_size: int,
        stats: Any = None,
    ) -> list[Any]:
        """The CTranslate2 call, with an in-process halve-and-retry ladder around an OOM.

        Without it an OOM is terminal for the whole build: the worker fails the shard, every
        retry re-runs the same shard under the same shape and fails identically, and the shard
        ends up in ``failed/`` with nothing produced. Halving the bucket is the only place
        that can shrink the offending batch, so it happens here; only a single-sentence OOM
        propagates.

        This is defence in depth rather than the primary control, which is conservative
        sizing: CTranslate2 may abort the process on some allocator failures, in which case no
        exception reaches this handler. A halved retry also changes batch composition, so its
        numerics may differ from the un-split call, which is acceptable on a recovery path and
        another reason sizing must not rely on it.
        """
        translator, _ = self._load(src_lang)
        # A bilingual checkpoint has no target-language token: its decoder already speaks only
        # English, and forcing a language tag would prepend an out-of-vocabulary string to
        # every hypothesis.
        prefix = None if self.family == FAMILY_MARIAN else [[target_lang]] * len(tokenized)
        try:
            return translator.translate_batch(
                tokenized,
                target_prefix=prefix,
                beam_size=beam_size,
                max_decoding_length=decode_length,
                max_input_length=max_input_length,
                max_batch_size=max_batch_size,
                batch_type="tokens",
                no_repeat_ngram_size=no_repeat_ngram_size,
                return_scores=False,
            )
        except Exception as exc:  # re-raised unless it is a survivable, splittable OOM
            if len(tokenized) <= 1 or not _is_oom(exc):
                raise
            if stats is not None:
                stats.emit("translate.oom_halve_retry")
            mid = len(tokenized) // 2
            kwargs = {
                "src_lang": src_lang,
                "target_lang": target_lang,
                "beam_size": beam_size,
                "decode_length": decode_length,
                "max_input_length": max_input_length,
                "max_batch_size": max_batch_size,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "stats": stats,
            }
            return self._call(tokenized[:mid], **kwargs) + self._call(tokenized[mid:], **kwargs)

    def _detokenize(self, result: Any, target_lang: str, tok: Any) -> str:
        """Turn one result into its best hypothesis as text, stripping the language tag.

        ``tok`` is passed in rather than read off the client, because with per-direction
        checkpoints there is no single tokenizer, and decoding a hypothesis against the wrong
        vocabulary would silently produce plausible-looking nonsense.
        """
        hyp = [t for t in result.hypotheses[0] if t != target_lang]
        return tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True)
