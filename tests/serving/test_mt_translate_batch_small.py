"""``MtClient.translate_batch``: the CT2 contract, the token layout, the OOM ladder.

Every test here is GPU-free: the client is the real
:class:`ragtime.serving.mt.MtClient` with a fake tokenizer + fake CT2 engine
(``tests.mt_fakes``), so the code under test is the shipped tokenize / decode-cap /
call-kwargs / halve-and-retry logic and only the engine is stubbed.
"""

from __future__ import annotations

import pytest

from ragtime.common import Statistics
from ragtime.serving.batching import Tier
from ragtime.serving.mt import (
    DEFAULT_MAX_INPUT_LENGTH,
    EOS_TOKEN,
    MtSentence,
    decode_cap,
    truncate_tokens,
)
from tests.mt_fakes import FAKE_DIR, fake_marian_client, fake_mt_client

pytestmark = pytest.mark.small

_RATIOS = {"zho_Hans": 2.0, "zho_Hant": 2.0, "rus_Cyrl": 1.4, "spa_Latn": 1.3}


def _item(sid: str, lang: str, n_tokens: int) -> MtSentence:
    body = tuple(f"t{i}" for i in range(n_tokens - 2))
    return MtSentence(sentence_id=sid, src_lang=lang, text=" ".join(body),
                      tokens=(lang, *body, EOS_TOKEN))


# --------------------------------------------------------------------------- #
# The source-token layout, and the guard on its silent failure.
# --------------------------------------------------------------------------- #
def test_tokenize_builds_lang_token_first_and_eos_last_never_via_src_lang() -> None:
    client = fake_mt_client()
    tokens = client.tokenize("北京 今天 天气", "zho_Hans")
    assert tokens[0] == "zho_Hans"  # the language signal the model needs
    assert tokens[-1] == EOS_TOKEN  # ... and the terminator it needs
    ids = client._toks[FAKE_DIR].convert_tokens_to_ids(list(tokens))
    assert client._toks[FAKE_DIR].unk_token_id not in ids  # never the `<unk>`-tailed src_lang path
    assert ids[0] == 256200 and ids[-1] == 2  # the pinned zho_Hans / </s> ids


def test_tokenize_raises_loudly_when_the_language_token_resolves_to_unk() -> None:
    """The tokenizer failure this guards against was silent.

    A tokenizer that cannot resolve the language code must fail the shard rather than emit
    54 M looping translations with no exception and no counter.
    """
    client = fake_mt_client(unknown_specials=("zho_Hans",))
    with pytest.raises(RuntimeError, match="source-token layout broken"):
        client.tokenize("北京", "zho_Hans")


# --------------------------------------------------------------------------- #
# Truncation.
# --------------------------------------------------------------------------- #
def test_truncate_keeps_both_special_tokens_at_512() -> None:
    tokens = ["spa_Latn", *[f"w{i}" for i in range(598)], EOS_TOKEN]
    out = truncate_tokens(tokens, DEFAULT_MAX_INPUT_LENGTH)
    assert len(out) == DEFAULT_MAX_INPUT_LENGTH
    assert out[0] == "spa_Latn" and out[-1] == EOS_TOKEN
    # ... which the naive slice this replaces does not do (the bug being fixed).
    assert tokens[:DEFAULT_MAX_INPUT_LENGTH][-1] != EOS_TOKEN


def test_tokenize_truncates_and_counts_truncated_at_max_input() -> None:
    client = fake_mt_client()
    stats = Statistics()
    long_text = " ".join(f"w{i}" for i in range(40))
    tokens = client.tokenize(long_text, "spa_Latn", max_input_length=12, stats=stats)
    assert len(tokens) == 12
    assert tokens[0] == "spa_Latn" and tokens[-1] == EOS_TOKEN
    assert stats.value("translate.truncated_at_max_input", lang="spa_Latn") == 1.0
    # a short sentence does not fire the counter
    client.tokenize("w0 w1", "spa_Latn", max_input_length=12, stats=stats)
    assert stats.total("translate.truncated_at_max_input") == 1.0


# --------------------------------------------------------------------------- #
# Decode cap (scalar per call, direction-aware over the languages PRESENT).
# --------------------------------------------------------------------------- #
def test_decode_cap_uses_max_ratio_across_bucket_languages() -> None:
    bucket = [_item("a#p0#s0", "spa_Latn", 10), _item("b#p0#s0", "zho_Hans", 20)]
    # max ratio (zh 2.0) x max src tokens (20) + 10
    assert decode_cap(bucket, len_ratio_a=_RATIOS, len_ratio_b=10, max_decoding_cap=512) == 50
    # ... and the configured hard cap always wins
    assert decode_cap(bucket, len_ratio_a=_RATIOS, len_ratio_b=10, max_decoding_cap=32) == 32


# --------------------------------------------------------------------------- #
# The CT2 call itself.
# --------------------------------------------------------------------------- #
def test_translate_batch_passes_every_memory_knob_and_no_sampling_params() -> None:
    client = fake_mt_client()
    bucket = [_item("d#p0#s0", "rus_Cyrl", 8), _item("d#p0#s1", "spa_Latn", 12)]
    out = client.translate_batch(
        bucket,
        len_ratio_a=_RATIOS,
        beam_size=4,
        no_repeat_ngram_size=4,
        tier=Tier(token_budget=16384, max_items=512),
    )
    assert set(out) == {"d#p0#s0", "d#p0#s1"}
    (call,) = client._translators[FAKE_DIR].calls  # exactly one bucket per CT2 call
    assert call["beam_size"] == 4
    assert call["batch_type"] == "tokens"
    assert call["max_batch_size"] == 16384  # default 0 makes token batching inert
    assert call["max_input_length"] == DEFAULT_MAX_INPUT_LENGTH  # CT2 defaults to 1024
    assert call["max_decoding_length"] == decode_cap(bucket, len_ratio_a=_RATIOS)
    assert call["no_repeat_ngram_size"] == 4  # ...exactly what the CALLER passed
    assert call["return_scores"] is False
    # deterministic beam search: No sampling knob is passed, at all.
    assert not {"sampling_topk", "sampling_topp", "sampling_temperature"} & set(call)
    assert call["target_prefix"] == [["eng_Latn"], ["eng_Latn"]]


def test_no_repeat_ngram_size_has_no_client_side_default() -> None:
    """It materially changes the output text: a repetition loop drags the whole batch to the
    decode cap, and blocking too hard corrupts legitimately repeating text such as dates. So
    it is a hashed knob in ``translation.config``. A Python default here would let a config
    omit it and still move the bytes, and the config would stop being the record."""
    client = fake_mt_client()
    with pytest.raises(TypeError, match="no_repeat_ngram_size"):
        client.translate_batch(  # type: ignore[call-arg]
            [_item("d#p0#s0", "spa_Latn", 6)], len_ratio_a=_RATIOS
        )


def test_the_ngram_value_reaches_ct2_unchanged_from_the_caller() -> None:
    client = fake_mt_client()
    client.translate_batch(
        [_item("d#p0#s0", "spa_Latn", 6)], len_ratio_a=_RATIOS, no_repeat_ngram_size=7
    )
    assert client._translators[FAKE_DIR].calls[0]["no_repeat_ngram_size"] == 7


def test_translate_batch_refuses_untokenized_items() -> None:
    client = fake_mt_client()
    raw = MtSentence(sentence_id="d#p0#s0", src_lang="spa_Latn", text="hola", tokens=())
    with pytest.raises(ValueError, match="un-tokenized"):
        client.translate_batch([raw], len_ratio_a=_RATIOS, no_repeat_ngram_size=4)


def test_translate_batch_maps_results_back_by_sentence_id_in_order() -> None:
    client = fake_mt_client()
    bucket = [_item("d#p0#s2", "spa_Latn", 5), _item("d#p0#s0", "rus_Cyrl", 9)]
    out = client.translate_batch(bucket, len_ratio_a=_RATIOS, no_repeat_ngram_size=4)
    assert out["d#p0#s2"].startswith("EN ")
    assert out["d#p0#s0"].startswith("EN ")
    assert out["d#p0#s2"] != out["d#p0#s0"]


# --------------------------------------------------------------------------- #
# OOM: the build must survive it.
# --------------------------------------------------------------------------- #
def test_oom_halve_and_retry_ladder_shrinks_bucket_until_it_fits() -> None:
    """Without the ladder one OOM costs 90-360 GPU-h: run_worker fails the shard, all
    k_max retries repeat under the same shape, ``drive`` then refuses to merge."""
    client = fake_mt_client(oom_above=2)
    stats = Statistics()
    bucket = [_item(f"d#p0#s{i}", "spa_Latn", 6) for i in range(8)]
    out = client.translate_batch(
        bucket, len_ratio_a=_RATIOS, no_repeat_ngram_size=4, stats=stats
    )
    assert set(out) == {it.sentence_id for it in bucket}  # nothing lost
    sizes = [len(c["source"]) for c in client._translators[FAKE_DIR].calls]
    assert sizes == [8, 4, 2, 2, 4, 2, 2]  # whole bucket, then halve until it fits
    assert stats.total("translate.oom_halve_retry") == 3


def test_single_sentence_oom_propagates_to_the_poison_path() -> None:
    client = fake_mt_client(oom_above=0)
    with pytest.raises(RuntimeError, match="out of memory"):
        client.translate_batch(
            [_item("d#p0#s0", "spa_Latn", 6)], len_ratio_a=_RATIOS, no_repeat_ngram_size=4
        )


def test_non_oom_failure_is_never_swallowed_by_the_ladder() -> None:
    client = fake_mt_client()

    def _boom(source, **kwargs):
        raise RuntimeError("invalid model file")

    client._translators[FAKE_DIR].translate_batch = _boom
    with pytest.raises(RuntimeError, match="invalid model file"):
        client.translate_batch(
            [_item(f"d#p0#s{i}", "spa_Latn", 6) for i in range(4)],
            len_ratio_a=_RATIOS,
            no_repeat_ngram_size=4,
        )


# --------------------------------------------------------------------------- #
# The per-direction arm (OPUS-MT): N bilingual checkpoints, and no language token.
# --------------------------------------------------------------------------- #
def _marian_item(sid: str, lang: str, n_tokens: int) -> MtSentence:
    """Marian's layout: ``[*content, </s>]``, no leading language token."""
    body = tuple(f"t{i}" for i in range(n_tokens - 1))
    return MtSentence(sentence_id=sid, src_lang=lang, text=" ".join(body),
                      tokens=(*body, EOS_TOKEN))


def test_marian_tokenize_omits_the_language_token_and_keeps_the_eos() -> None:
    """The direction is the checkpoint, so a language token would be stray content."""
    client = fake_marian_client()
    tokens = client.tokenize("hola mundo", "spa_Latn")
    assert tokens[0] != "spa_Latn"
    assert tokens[-1] == EOS_TOKEN


def test_marian_sends_no_target_prefix() -> None:
    """Forcing `eng_Latn` would prepend a string Marian's vocabulary does not contain."""
    client = fake_marian_client()
    client.translate_batch(
        [_marian_item("d#p0#s0", "spa_Latn", 6)], len_ratio_a=_RATIOS, no_repeat_ngram_size=4
    )
    (call,) = client._translators["/fake/opus-es"].calls
    assert call["target_prefix"] is None


def test_a_mixed_language_bucket_is_split_across_the_right_checkpoints() -> None:
    """Buckets are composed by length alone, so a boundary shard mixes languages. Each group
    must reach its own replica: never one model translating another's text."""
    client = fake_marian_client()
    items = [
        _marian_item("d#p0#s0", "spa_Latn", 6),
        _marian_item("d#p0#s1", "rus_Cyrl", 6),
        _marian_item("d#p0#s2", "zho_Hans", 6),
        _marian_item("d#p0#s3", "zho_Hant", 6),
    ]
    out = client.translate_batch(items, len_ratio_a=_RATIOS, no_repeat_ngram_size=4)
    assert set(out) == {i.sentence_id for i in items}  # every item came back
    per_dir = {p: len(t.calls) for p, t in client._translators.items()}
    assert per_dir == {"/fake/opus-es": 1, "/fake/opus-ru": 1, "/fake/opus-zh": 1}
    # zho_Hans + zho_Hant share one checkpoint, so they share one call of two sequences.
    assert len(client._translators["/fake/opus-zh"].calls[0]["source"]) == 2


def test_the_zh_directions_are_one_resident_replica_not_two() -> None:
    """Keying the replica cache by resolved PATH is what makes that true."""
    client = fake_marian_client()
    assert client.per_language()
    assert client._resolve_path("zho_Hans") == client._resolve_path("zho_Hant")
    assert len(client._translators) == 3  # es, ru, zh: not four


def test_per_direction_model_id_names_the_checkpoint_that_produced_the_row() -> None:
    client = fake_marian_client()
    assert client.model_id_for("spa_Latn") == "Helsinki-NLP/opus-mt-es-en"
    assert client.model_id_for("zho_Hant") == "Helsinki-NLP/opus-mt-zh-en"


def test_a_direction_with_no_configured_checkpoint_fails_loudly() -> None:
    """Never a silent fallback: translating zh through the Spanish model would only ever
    show up as bad English nobody could attribute."""
    client = fake_marian_client()
    with pytest.raises(KeyError, match="no entry for source language"):
        client._resolve_path("deu_Latn")


def test_decode_cap_refuses_a_len_ratio_miss_instead_of_defaulting_to_1() -> None:
    """A miss is silent and destructive: the cap tightens to ~the source length and
    zh->en (which genuinely expands ~2x) is truncated mid-sentence."""
    from ragtime.serving.mt import decode_cap

    items = [_marian_item("d#p0#s0", "deu_Latn", 6)]
    assert decode_cap(items, len_ratio_a=_RATIOS) > 0  # lenient by default
    with pytest.raises(KeyError, match="len_ratio_a has no entry"):
        decode_cap(items, len_ratio_a=_RATIOS, require_ratio=True)


def test_marian_truncation_keeps_content_from_the_front_not_a_stray_first_token() -> None:
    """With no leading special, keeping ``t[0]`` and dropping its successors would splice
    the sentence: the NLLB layout's rule is wrong here."""
    from ragtime.serving.mt import truncate_tokens

    seq = ("a", "b", "c", "d", EOS_TOKEN)
    assert truncate_tokens(seq, 3, leading_specials=0) == ("a", "b", EOS_TOKEN)
    assert truncate_tokens(seq, 3, leading_specials=1) == ("a", "b", EOS_TOKEN)
    seq2 = ("spa_Latn", "b", "c", "d", EOS_TOKEN)
    assert truncate_tokens(seq2, 3, leading_specials=1) == ("spa_Latn", "b", EOS_TOKEN)
