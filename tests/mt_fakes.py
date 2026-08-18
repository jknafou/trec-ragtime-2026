"""Dependency-free stand-ins for the NLLB tokenizer + the CTranslate2 translator.

Shared by the serving and preprocess small suites so that no small test imports
``ctranslate2``, ``torch`` or ``transformers``. Both fakes are deterministic and
record their call kwargs, which is what lets the tests assert the CT2 contract
(exactly one bucket per call, every memory knob passed explicitly, no sampling
parameter anywhere) without a GPU.
"""

from __future__ import annotations

import types
from typing import Any

# Real NLLB ids for the tokens the layout assertion cares about, taken from the
# pinned transformers stack.
SPECIAL_IDS: dict[str, int] = {
    "</s>": 2,
    "<unk>": 3,
    "eng_Latn": 256047,
    "spa_Latn": 256161,
    "rus_Cyrl": 256147,
    "zho_Hans": 256200,
    "zho_Hant": 256201,
}
WORD_PREFIX = "▁"  # the SentencePiece word-start marker


class FakeNllbTokenizer:
    """Whitespace tokenizer with an NLLB-shaped id space and vocabulary lookups.

    Implements exactly the surface ``serving.mt.MtClient`` uses: ``__call__(text,
    add_special_tokens=False).input_ids``, ``convert_ids_to_tokens``,
    ``convert_tokens_to_ids``, ``unk_token_id`` and ``decode``. ``unknown_specials``
    makes a token resolve to ``<unk>``, which is the silent breakage seen when
    ``tokenizer.src_lang`` is unknown, so the layout assertion can be shown to fire
    instead of merely being present.
    """

    def __init__(self, unknown_specials: tuple[str, ...] = ()) -> None:
        self.unk_token_id = SPECIAL_IDS["<unk>"]
        self._unknown = set(unknown_specials)
        self._to_id: dict[str, int] = dict(SPECIAL_IDS)
        self._to_tok: dict[int, str] = {v: k for k, v in SPECIAL_IDS.items()}
        self._next = 1000

    def _id(self, token: str) -> int:
        if token in self._unknown:
            return self.unk_token_id
        if token not in self._to_id:
            self._to_id[token] = self._next
            self._to_tok[self._next] = token
            self._next += 1
        return self._to_id[token]

    def __call__(self, text: str, add_special_tokens: bool = True) -> Any:
        ids = [self._id(WORD_PREFIX + w) for w in text.split()]
        if add_special_tokens:  # the path the translation stage must never take
            ids = [*ids, SPECIAL_IDS["</s>"], self.unk_token_id]
        return types.SimpleNamespace(input_ids=ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._to_tok.get(i, "<unk>") for i in ids]

    def convert_tokens_to_ids(self, tokens: list[str] | str) -> Any:
        if isinstance(tokens, str):
            return self._id(tokens)
        return [self._id(t) for t in tokens]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        toks = [self._to_tok.get(i, "<unk>") for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in SPECIAL_IDS]
        return " ".join(t.removeprefix(WORD_PREFIX) for t in toks)


class FakeCt2Translator:
    """Records every call and echoes each source as ``EN <content...>``.

    ``oom_above`` makes a call raise a CUDA-OOM-shaped ``RuntimeError`` when the
    batch holds more than that many sequences, so the halve-and-retry ladder can be
    exercised with no GPU.
    """

    def __init__(self, oom_above: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.oom_above = oom_above

    def translate_batch(self, source: list[list[str]], **kwargs: Any) -> list[Any]:
        self.calls.append({"source": [list(s) for s in source], **kwargs})
        if self.oom_above is not None and len(source) > self.oom_above:
            raise RuntimeError("CUDA failed with error out of memory")
        prefix = kwargs.get("target_prefix")
        if prefix is None:
            # The `marian` layout: a bilingual checkpoint takes no target prefix and its
            # source carries no leading language token, so everything but the trailing
            # EOS is content.
            return [
                types.SimpleNamespace(hypotheses=[[WORD_PREFIX + "EN", *seq[:-1]]])
                for seq in source
            ]
        target = prefix[0][0]
        return [
            types.SimpleNamespace(hypotheses=[[target, WORD_PREFIX + "EN", *seq[1:-1]]])
            for seq in source
        ]


#: The stand-in checkpoint dir the fakes are cached under. It never exists on disk: the
#: point of pre-filling the replica caches is that no load, and hence no existence check,
#: is ever reached.
FAKE_DIR = "/fake/nllb"


def fake_mt_client(
    *, oom_above: int | None = None, unknown_specials: tuple[str, ...] = ()
) -> Any:
    """A real :class:`ragtime.serving.mt.MtClient` wired to the two fakes.

    The client's own ``_load`` is bypassed, since both replica caches are pre-filled under
    :data:`FAKE_DIR`, so the real tokenize, decode-cap, CT2-kwargs and OOM-ladder code
    runs and only the engine is fake.
    """
    from ragtime.serving.mt import MtClient

    client = MtClient(
        model="fake-nllb", engine="ctranslate2", device="cpu", model_path=FAKE_DIR
    )
    client._translators[FAKE_DIR] = FakeCt2Translator(oom_above=oom_above)
    client._toks[FAKE_DIR] = FakeNllbTokenizer(unknown_specials=unknown_specials)
    return client


def fake_marian_client(dirs: dict[str, str] | None = None) -> Any:
    """A per-direction, OPUS-MT-shaped client: one bilingual checkpoint per direction.

    Two directions share one dir (``zho_Hans`` and ``zho_Hant`` both map to
    the zh checkpoint), because that is the real config and it is what shows the client
    caches replicas by resolved path rather than by language.
    """
    from ragtime.serving.mt import FAMILY_MARIAN, MtClient

    dirs = dirs or {
        "spa_Latn": "/fake/opus-es",
        "rus_Cyrl": "/fake/opus-ru",
        "zho_Hans": "/fake/opus-zh",
        "zho_Hant": "/fake/opus-zh",
    }
    ids = {lg: f"Helsinki-NLP/opus-mt-{p.rsplit('-', 1)[-1]}-en" for lg, p in dirs.items()}
    client = MtClient(
        model=ids, engine="ctranslate2", device="cpu", model_path=dirs, family=FAMILY_MARIAN
    )
    for path in set(dirs.values()):
        client._translators[path] = FakeCt2Translator()
        client._toks[path] = FakeNllbTokenizer()
    return client
