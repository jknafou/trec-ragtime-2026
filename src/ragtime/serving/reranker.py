"""Cross-encoder reranker behind one interface.

``score(query, passages) -> list[float]``, where higher means more relevant. The backend is
Qwen3-Reranker-4B scored the way it was trained: its own instruct template through
``AutoModelForCausalLM``, read at the ``yes`` and ``no`` logits of its LM head. The model id
comes from config, and heavy imports are lazy.

The score is ``log P(yes)`` and is therefore never positive. It is the retriever's own
relevance number, returned to the RAG loop with the passage ids; it is unrelated to the
Task 2 document score, which is a much later sum over the claims citing a document and uses
this number only as a tie-break.
"""

from __future__ import annotations

from itertools import chain
from typing import Any

from ragtime.common import get_logger

_log = get_logger("serving.reranker")

__all__ = ["RERANK_TASK", "Reranker", "bind_checkpoint_head", "rerank_prompt"]

#: Qwen3-Reranker's own instruct template, taken verbatim from the checkpoint's model card.
#: The trained head only produces a relevance signal for the prompt it was trained on.
_PROMPT_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

#: The instruct line. It is a prompt constant rather than a run choice: it stays
#: byte-identical across every variant of a run family, and ``retrieval.reranker`` carries
#: only ``{model, depth}``.
RERANK_TASK = "Given a web search query, retrieve relevant passages that answer the query"

#: Rows per forward pass and the token window each row is truncated to. These are throughput
#: facts rather than semantics: a larger batch measured no faster while costing several GiB
#: of activations, and 1024 tokens is roughly four times the packed passage length, so it
#: bounds a pathological input without truncating a production one.
_BATCH_SIZE = 8
_MAX_LENGTH = 1024


def rerank_prompt(query: str, passage: str, task: str = RERANK_TASK) -> str:
    """Build the prompt string the reranker scores. Module-level so a test can pin it."""
    return (
        _PROMPT_PREFIX + f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {passage}"
        + _PROMPT_SUFFIX
    )


def bind_checkpoint_head(model: Any, loading_info: dict[str, Any], model_id: str) -> None:
    """Verify, and where necessary restore, that the scoring head holds checkpoint weights.

    ``Qwen/Qwen3-Reranker-4B`` ships ``tie_word_embeddings: true`` and so carries no
    ``lm_head.weight`` of its own; transformers ties it to ``model.embed_tokens.weight`` at
    the end of ``from_pretrained``. Everything this module computes hangs off that tensor, so
    the model does not leave :meth:`Reranker.load` unless the tensor is real and came from the
    checkpoint.

    The tying can fail. ``transformers.initialization.no_tie_weights()``, entered by every
    ``from_pretrained``, disables tying by monkeypatching the class attribute
    ``PreTrainedModel.tie_weights`` to a no-op and restoring it in a ``finally``. That patch
    is process-global and neither re-entrant nor thread-safe, so two overlapping loads can
    interleave and leave the class attribute neutered for the rest of the interpreter. A later
    load then reports ``lm_head.weight`` as missing and, because it is a declared tied key,
    skips moving it off the ``meta`` device, so it ends up with no storage. Loading onto the
    device with ``device_map`` hides that rather than fixing it: a meta head scores every
    passage identically, which hands the whole ordering to the caller's tie-break.

    ``to_empty()`` is not a fix either, since it allocates uninitialised storage. The correct
    repair is to bind the output embedding to the input embedding exactly as the checkpoint's
    config declares, which restores identical scores in a clean process.

    Raises ``RuntimeError`` if any tensor is still on ``meta`` after the repair, or if any
    checkpoint-missing parameter remains unexplained, since a freshly initialised parameter
    anywhere in this model makes the ranking noise.
    """
    cfg = getattr(model, "config", None)
    missing = set(loading_info.get("missing_keys") or ())
    # `all_tied_weights_keys` is computed in `post_init` from the config and is unaffected by
    # a neutered `tie_weights`, so it still names the ties the checkpoint intends.
    resolved = set(getattr(model, "all_tied_weights_keys", None) or {})
    inp = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if getattr(cfg, "tie_word_embeddings", False) and inp is not None and out is not None:
        out_w = out.weight
        untied = out_w.device.type == "meta" or out_w.data_ptr() != inp.weight.data_ptr()
        if untied:
            name = next(
                (n for n, p in model.named_parameters(remove_duplicate=False) if p is out_w),
                None,
            )
            out.weight = inp.weight  # the tie the config declares, done by hand
            if name is not None:
                resolved.add(name)
            _log.warning(
                "reranker.retied_output_head",
                model=model_id,
                param=name,
                was_meta=out_w.device.type == "meta",
                detail=(
                    "transformers' weight tying was inactive for this load (a leaked global "
                    "no_tie_weights() patch); the output head was re-tied to the input "
                    "embedding so the checkpoint's own weights score, not a fresh tensor"
                ),
            )
    stranded = [
        n
        for n, t in chain(model.named_parameters(), model.named_buffers())
        if t.device.type == "meta"
    ]
    if stranded:
        raise RuntimeError(
            f"reranker {model_id!r}: {len(stranded)} tensor(s) are still on the meta device "
            f"after loading (e.g. {stranded[:5]}). They hold no data, so every score would be "
            "meaningless; refusing rather than materialising uninitialised storage."
        )
    unresolved = sorted(missing - resolved)
    if unresolved:
        raise RuntimeError(
            f"reranker {model_id!r}: parameters {unresolved} are absent from the checkpoint and "
            "were freshly initialised, so the scoring head is not the trained one."
        )


class Reranker:
    """Query-passage reranker using Qwen3-Reranker-4B's own yes/no head.

    This is not a ``sentence_transformers.CrossEncoder``.
    ``Qwen/Qwen3-Reranker-4B`` is a ``Qwen3ForCausalLM`` whose relevance signal is the yes/no
    token logit of its LM head, while ``CrossEncoder`` dispatches to
    ``AutoModelForSequenceClassification``, which discards that head and randomly initialises
    a fresh ``score.weight``. The result is not a degraded ranking but a random permutation of
    the pool, which is worse than no reranker at all, since reranking is the last thing to
    touch passage order before the LLM reads it. Instead this class loads
    ``AutoModelForCausalLM`` from the cached checkpoint and reads
    ``log_softmax(logits[-1])`` at the ``yes`` and ``no`` ids.

    There is one backend. ``retrieval.reranker`` carries only ``{model, depth}``, so no config
    leaf ever selected a second one.
    """

    __slots__ = (
        "_model",
        "_no",
        "_tok",
        "_yes",
        "batch_size",
        "device",
        "max_length",
        "model",
    )

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        *,
        batch_size: int = _BATCH_SIZE,
        max_length: int = _MAX_LENGTH,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._tok: Any = None
        self._model: Any = None
        self._yes = self._no = -1

    def load(self) -> None:
        """Load the tokenizer and causal LM, and resolve the ``yes`` and ``no`` token ids.

        The weights land on ``device`` directly through ``device_map``, and the loaded model is
        audited by :func:`bind_checkpoint_head` before it is stored; see that function for why
        placing the weights on the device is only half of the fix.

        Padding is left-side so that ``logits[:, -1, :]`` is the real last token of every row in
        a batch rather than a pad position. With right padding, every short row would be scored
        at a pad token and the reranking would be noise.

        Qwen checkpoints ship with no ``pad_token`` and the family convention is to reuse EOS.
        Scoring a pool of ``retrieval.reranker.depth`` candidates always tokenizes a batch, so
        without a pad token transformers raises ``ValueError: Cannot handle batch sizes > 1 if
        no padding token is defined``. Setting it on the tokenizer alone is not enough, because
        the guard also reads the model config, so both are set here.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model, padding_side="left")
        if getattr(tok, "pad_token", None) is None:
            eos = getattr(tok, "eos_token", None)
            if eos is None:  # nothing safe to pad with
                raise RuntimeError(
                    f"reranker {self.model!r} defines neither pad_token nor eos_token, "
                    "so it cannot tokenize a batch; scoring a pool is impossible"
                )
            tok.pad_token = eos
        # The checkpoint's own weights are bf16, so this is a load rather than a cast.
        # `device_map` puts every weight on the target device as it is read, which avoids the
        # load-to-CPU-then-`model.to(device)` two-step and its meta-tensor failure, and
        # `output_loading_info` returns the checkpoint-vs-model key diff that
        # `bind_checkpoint_head` audits.
        mdl, info = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=torch.bfloat16,
            device_map={"": self.device},
            output_loading_info=True,
        )
        bind_checkpoint_head(mdl, info, self.model)
        mdl = mdl.eval()
        cfg = getattr(mdl, "config", None)
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
        if cfg is not None and getattr(cfg, "pad_token_id", None) is None and pad_id is not None:
            cfg.pad_token_id = pad_id
        yes = tok.convert_tokens_to_ids("yes")
        no = tok.convert_tokens_to_ids("no")
        if yes is None or no is None or yes < 0 or no < 0:
            raise RuntimeError(
                f"reranker {self.model!r}: tokenizer has no single 'yes'/'no' token, so the "
                "yes/no logit head is unaddressable and scoring would be meaningless"
            )
        self._tok, self._model, self._yes, self._no = tok, mdl, int(yes), int(no)
        _log.info(
            "reranker.loaded",
            model=self.model,
            device=self.device,
            pad_token=str(getattr(tok, "pad_token", None)),
            yes_id=self._yes,
            no_id=self._no,
        )

    def _batch_logprobs(self, prompts: list[str]) -> list[float]:
        """Score one already-sized batch of prompts: ``log P(yes)`` per row, in input order."""
        import torch

        enc = self._tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self._model(**enc).logits[:, -1, :]
        pair = torch.stack([logits[:, self._no], logits[:, self._yes]], dim=1).float()
        # The log probability of "yes" rather than its exponential: monotone-identical for
        # ranking, and it does not underflow. Under `.exp()` clearly irrelevant passages all
        # score exactly 0.0, which ties the whole tail and hands its ordering to the caller's
        # tie-break instead of the model.
        return [float(x) for x in torch.log_softmax(pair, dim=1)[:, 1].tolist()]

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage for ``query``, higher being more relevant.

        The scoring is deterministic, with no sampling and no freshly initialised parameters,
        so two loads of the same checkpoint score a pool identically. Scores are
        log-probabilities and therefore at most 0; callers rank by them and must not assume a
        ``[0, 1]`` scale.
        """
        if not passages:
            return []
        if self._model is None:
            self.load()
        prompts = [rerank_prompt(query, p) for p in passages]
        out: list[float] = []
        for i in range(0, len(prompts), self.batch_size):
            out.extend(self._batch_logprobs(prompts[i : i + self.batch_size]))
        return out
