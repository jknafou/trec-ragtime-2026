"""The reranker's yes/no head, hermetically: no torch, no transformers, no checkpoint.

What is pinned here is the shape of the correctness fix, meaning the things that would let
the original defect come back:

- SM-RR1 the module must load the checkpoint's own LM head (``AutoModelForCausalLM``) and
  must never go through ``sentence_transformers.CrossEncoder`` or
  ``AutoModelForSequenceClassification``, which discard that head and randomly initialise a
  ``score`` layer, turning a good fused ranking into a random permutation right before the
  LLM reads it. This is a source-level guard, because the failure is a silent wrong number
  rather than an error.
- SM-RR2 padding is on the left and the pad token is fixed up, or ``logits[:, -1, :]`` reads
  a pad position for every short row of a batch.
- SM-RR3 ``score`` batches by ``batch_size`` and returns one score per passage in input
  order, since the retrieval layer zips the scores back onto its candidate ids with
  ``strict=True``.
- SM-RR4 the constructor is the one ``registry.build_clients`` and the retrieval service both
  call, and it loads nothing.
- SM-RR5 the loaded model is audited by ``bind_checkpoint_head`` before it can score. A tied
  head left stranded on the ``meta`` device is re-tied to the input embedding, which is the
  failure that was observed, while a head genuinely absent from the checkpoint, or any other
  tensor still on ``meta``, raises. Both halves matter: loading onto the device without
  re-tying does not crash, it scores every passage at ``log 0.5``, which is the same silent
  wrong-number class as the CrossEncoder defect.

The arithmetic itself, the log-probability of "yes" and the absence of a tie collapsing the
tail, needs the real 4B checkpoint and lives in ``test_reranker_full.py``.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ragtime.serving.reranker import RERANK_TASK, Reranker, rerank_prompt

pytestmark = pytest.mark.small

_SOURCE = (Path(__file__).resolve().parents[2] / "src" / "ragtime" / "serving" / "reranker.py").read_text()


def _imported_names(source: str) -> set[str]:
    """Every module and symbol the module IMPORTS (prose in a docstring is not an import)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


# --------------------------------------------------------------------------- #
# Fake transformers / torch (the only heavy things ``load()`` touches).
# --------------------------------------------------------------------------- #
class _FakeTokenizer:
    """Emulates the one behaviour the fixup depends on: setting ``pad_token`` to a token that
    is in the vocabulary (EOS) also resolves ``pad_token_id``."""

    def __init__(self, *, eos: str | None = "<|endoftext|>", ids: dict[str, int] | None = None):
        self.eos_token = eos
        self.eos_token_id = 7 if eos is not None else None
        self.pad_token_id: int | None = None
        self._pad_token: str | None = None
        self._ids = ids if ids is not None else {"yes": 9693, "no": 2152}
        self.kwargs: dict[str, Any] = {}

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str | None) -> None:
        self._pad_token = value
        self.pad_token_id = self.eos_token_id if value == self.eos_token else None

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._ids.get(token)


class _FakeConfig:
    def __init__(self, *, tie_word_embeddings: bool = True) -> None:
        self.pad_token_id: int | None = None
        self.tie_word_embeddings = tie_word_embeddings


#: Storage identity of the input embedding. A head sharing this "pointer" is the tied head.
_EMBED_PTR = 4242


class _FakeTensor:
    """The two things ``bind_checkpoint_head`` asks a tensor: its device type and its storage."""

    def __init__(self, ptr: int, device: str = "cuda") -> None:
        self._ptr = ptr
        self.device = types.SimpleNamespace(type=device)

    def data_ptr(self) -> int:
        return self._ptr


class _FakeEmbedding:
    def __init__(self, weight: _FakeTensor) -> None:
        self.weight = weight


class _FakeModel:
    """A model whose *checkpoint health* is configurable, because that is what is under test.

    ``head_device="meta"`` + ``missing_keys=("lm_head.weight",)`` reproduces a load in a process
    whose weight tying was globally disabled, which is the state that strands Qwen3-Reranker's
    tied ``lm_head.weight`` on the meta device and makes ``model.to(device)`` raise.
    """

    def __init__(
        self,
        name: str,
        *,
        tie_word_embeddings: bool = True,
        head_device: str = "cuda",
        head_ptr: int = _EMBED_PTR,
        stranded: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.kwargs = kwargs
        self.config = _FakeConfig(tie_word_embeddings=tie_word_embeddings)
        self.moved_to: str | None = None
        self.eval_called = False
        self._embed = _FakeEmbedding(_FakeTensor(_EMBED_PTR))
        self.lm_head = _FakeEmbedding(_FakeTensor(head_ptr, head_device))
        self._stranded = stranded
        self.all_tied_weights_keys: dict[str, str] = {}

    # -- the surface `bind_checkpoint_head` reads ---------------------------- #
    def get_input_embeddings(self) -> _FakeEmbedding:
        return self._embed

    def get_output_embeddings(self) -> _FakeEmbedding:
        return self.lm_head

    def named_parameters(self, remove_duplicate: bool = True) -> list[tuple[str, _FakeTensor]]:
        # Read LIVE, so a re-tie performed by the code under test is visible to the meta scan.
        return [
            ("model.embed_tokens.weight", self._embed.weight),
            ("lm_head.weight", self.lm_head.weight),
            *((n, _FakeTensor(0, "meta")) for n in self._stranded),
        ]

    def named_buffers(self) -> list[tuple[str, _FakeTensor]]:
        return []

    def to(self, device: str) -> _FakeModel:
        self.moved_to = device
        return self

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self


def _fake_transformers(
    tok: _FakeTokenizer,
    *,
    missing_keys: tuple[str, ...] = (),
    tied_map: dict[str, str] | None = None,
    **model_kwargs: Any,
) -> types.ModuleType:
    """A ``transformers`` exposing only the two Auto classes the fixed code may use.

    Missing ``AutoModelForSequenceClassification``: a regression to the
    CrossEncoder path would fail here with ``ImportError``, not with a wrong number.

    ``from_pretrained`` honours the real contract: with ``output_loading_info=True`` it returns
    ``(model, loading_info)``, not a bare model: the production guard needs that key diff to tell
    a legitimately tied parameter from a freshly-invented one.
    """
    mod = types.ModuleType("transformers")
    seen: dict[str, Any] = {}

    class _AutoTok:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeTokenizer:
            tok.kwargs = {"name": name, **kwargs}
            return tok

    class _AutoCausal:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> Any:
            model = _FakeModel(name, **model_kwargs, **kwargs)
            model.all_tied_weights_keys = dict(tied_map or {})
            seen["model"] = model
            if kwargs.get("output_loading_info"):
                return model, {"missing_keys": set(missing_keys), "unexpected_keys": set()}
            return model

    mod.AutoTokenizer = _AutoTok  # type: ignore[attr-defined]
    mod.AutoModelForCausalLM = _AutoCausal  # type: ignore[attr-defined]
    mod.seen = seen  # type: ignore[attr-defined]
    return mod


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, tok: _FakeTokenizer, **scenario: Any
) -> types.ModuleType:
    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bfloat16-sentinel"  # type: ignore[attr-defined]
    transformers = _fake_transformers(tok, **scenario)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return transformers


# --------------------------------------------------------------------------- #
# SM-RR1: the head that scores is the checkpoint's own
# --------------------------------------------------------------------------- #
def test_smrr1_module_never_reaches_for_the_random_sequence_classification_head() -> None:
    imported = _imported_names(_SOURCE)
    assert "AutoModelForCausalLM" in imported
    for banned in ("CrossEncoder", "sentence_transformers", "AutoModelForSequenceClassification"):
        assert banned not in imported, (
            f"{banned} is back: it discards Qwen3-Reranker's LM head and randomly initialises "
            "`score.weight`, which shuffles the candidate pool"
        )


def test_smrr1_load_goes_through_the_causal_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = _FakeTokenizer()
    transformers = _install_fakes(monkeypatch, tok)
    rr = Reranker(model="Qwen/Qwen3-Reranker-4B", device="cuda")
    rr.load()

    model = transformers.seen["model"]  # type: ignore[attr-defined]
    assert model.name == "Qwen/Qwen3-Reranker-4B"  # the same cached checkpoint, no download
    # The weights land ON the device as they are read, and the key diff comes back with them.
    assert model.kwargs == {
        "dtype": "bfloat16-sentinel",
        "device_map": {"": "cuda"},
        "output_loading_info": True,
    }
    # never load-then-`.to(device)`: a tied head stranded on `meta` makes that raise
    # `NotImplementedError: Cannot copy out of meta tensor`.
    assert model.moved_to is None
    assert model.eval_called
    assert (rr._yes, rr._no) == (9693, 2152)


def test_smrr1_load_refuses_a_tokenizer_without_single_yes_no_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, _FakeTokenizer(ids={"yes": 9693}))
    with pytest.raises(RuntimeError, match="yes/no"):
        Reranker(model="m").load()


# --------------------------------------------------------------------------- #
# SM-RR2: left padding + the pad-token fixup
# --------------------------------------------------------------------------- #
def test_smrr2_tokenizer_pads_on_the_left_and_borrows_eos_as_pad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok = _FakeTokenizer()
    _install_fakes(monkeypatch, tok)
    rr = Reranker(model="m")
    rr.load()

    # Left, or `logits[:, -1, :]` reads a pad position for every short row of a batch.
    assert tok.kwargs == {"name": "m", "padding_side": "left"}
    assert tok.pad_token == tok.eos_token  # the family convention for Qwen checkpoints
    assert rr._model.config.pad_token_id == tok.eos_token_id  # tokenizer and config agree


def test_smrr2_load_refuses_when_there_is_nothing_safe_to_pad_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, _FakeTokenizer(eos=None))
    with pytest.raises(RuntimeError, match="pad_token"):
        Reranker(model="m").load()


# --------------------------------------------------------------------------- #
# SM-RR3: the prompt, the batching, and the input order
# --------------------------------------------------------------------------- #
def test_smrr3_prompt_is_qwens_own_instruct_template() -> None:
    prompt = rerank_prompt("who won?", "the passage text")
    assert prompt.startswith("<|im_start|>system\nJudge whether the Document meets")
    assert f"<Instruct>: {RERANK_TASK}\n<Query>: who won?\n<Document>: the passage text" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_smrr3_score_batches_by_batch_size_and_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def fake_batch(self: Reranker, prompts: list[str]) -> list[float]:
        seen.append(list(prompts))
        return [-float(len(p)) for p in prompts]

    monkeypatch.setattr(Reranker, "_batch_logprobs", fake_batch)
    rr = Reranker(model="m", batch_size=2)
    rr._model = object()  # already "loaded": no checkpoint is touched
    passages = ["a", "bb", "ccc", "dddd", "eeeee"]
    scores = rr.score("q", passages)

    assert [len(b) for b in seen] == [2, 2, 1]
    flat = [p for batch in seen for p in batch]
    assert flat == [rerank_prompt("q", p) for p in passages]  # order never permuted
    assert scores == [-float(len(p)) for p in flat]
    assert len(scores) == len(passages)  # retrieval zips these back with strict=True


def test_smrr3_empty_pool_is_no_forward_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(self: Reranker, prompts: list[str]) -> list[float]:
        raise AssertionError("an empty pool must not reach the model")

    monkeypatch.setattr(Reranker, "_batch_logprobs", explode)
    assert Reranker(model="m").score("q", []) == []


# --------------------------------------------------------------------------- #
# SM-RR4: the registry's construction interface
# --------------------------------------------------------------------------- #
def test_smrr4_registry_interface_is_unchanged() -> None:
    """``build_clients`` constructs with a model id only, and construction loads nothing."""
    rr = Reranker(model="Qwen/Qwen3-Reranker-4B")
    assert rr.device == "cuda"
    assert rr._model is None and rr._tok is None  # constructing loads nothing


# --------------------------------------------------------------------------- #
# SM-RR5: the loaded checkpoint is audited before it is allowed to score
# --------------------------------------------------------------------------- #
def test_smrr5_a_stranded_tied_head_is_retied_to_the_input_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure state: tying was globally neutered, so the tied head never materialised.

    ``lm_head.weight`` is absent from this checkpoint (``tie_word_embeddings: true``) and, being a
    declared tied key, transformers leaves it on ``meta`` for the tying step that no
    longer happens. The fix binds it to the input embedding: the checkpoint's own weights.
    ``to_empty()`` would also clear the ``meta`` flag and is not acceptable: it allocates
    uninitialised storage, i.e. the random head this module exists to prevent.
    """
    transformers = _install_fakes(
        monkeypatch,
        _FakeTokenizer(),
        missing_keys=("lm_head.weight",),
        tied_map={"lm_head.weight": "model.embed_tokens.weight"},
        head_device="meta",
        head_ptr=0,  # meta storage: no data
    )
    rr = Reranker(model="Qwen/Qwen3-Reranker-4B")
    rr.load()

    model = transformers.seen["model"]  # type: ignore[attr-defined]
    head = model.get_output_embeddings().weight
    assert head is model.get_input_embeddings().weight, "the head was not re-tied"
    assert head.device.type != "meta" and head.data_ptr() == _EMBED_PTR
    assert not [n for n, t in model.named_parameters() if t.device.type == "meta"]


def test_smrr5_an_unexplained_missing_parameter_refuses_to_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parameter absent from the checkpoint and not tied was invented at random: refuse.

    This is the general form of the CrossEncoder defect: a ``score``/``lm_head`` tensor that no
    training ever saw produces well-typed, well-shaped, meaningless numbers.
    """
    _install_fakes(
        monkeypatch,
        _FakeTokenizer(),
        missing_keys=("score.weight",),
        tied_map={},
    )
    with pytest.raises(RuntimeError, match="freshly initialised"):
        Reranker(model="Qwen/Qwen3-Reranker-4B").load()


def test_smrr5_a_tensor_left_on_the_meta_device_refuses_to_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anything still on ``meta`` after the repair has no data at all: fail, never materialise."""
    _install_fakes(
        monkeypatch,
        _FakeTokenizer(),
        stranded=("model.layers.0.mlp.down_proj.weight",),
    )
    with pytest.raises(RuntimeError, match="meta device"):
        Reranker(model="Qwen/Qwen3-Reranker-4B").load()
