"""Pack tokenizer determinism and the revision pin.

Determinism, ``count("") == 0`` and monotonicity are checked against a synthetic local
tokenizer, so they need no network. The cross-check that the loader reads the pinned
``BAAI/bge-m3@<sha>`` revision rather than the latest main has to load bge-m3 itself, and is
skipped when it cannot be fetched.
"""

from __future__ import annotations

import pytest

from ragtime.preprocess import tokenizer as tk
from ragtime.preprocess.tokenizer import PackTokenizer, load_pack_tokenizer, parse_tokenizer_id

pytestmark = pytest.mark.small


def _synthetic_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return tok


def test_parse_tokenizer_id() -> None:
    assert parse_tokenizer_id("BAAI/bge-m3@abc123") == ("BAAI/bge-m3", "abc123")
    assert parse_tokenizer_id("BAAI/bge-m3") == ("BAAI/bge-m3", None)


def test_count_is_deterministic_and_empty_is_zero() -> None:
    pt = PackTokenizer(_synthetic_tokenizer())
    for s in ("uno dos tres", "привет мир", "你好 世界", "hello"):
        assert pt.count(s) == pt.count(s)  # identical across calls (no mutable state)
    assert pt.count("") == 0
    assert PackTokenizer(_synthetic_tokenizer()).count("a b c") == pt.count("a b c")


def test_count_monotone_non_decreasing_under_concat() -> None:
    pt = PackTokenizer(_synthetic_tokenizer())
    a, b = "uno dos", "tres cuatro cinco"
    assert pt.count(a + " " + b) >= pt.count(a)
    assert pt.count(a + " " + b) >= pt.count(b)
    assert pt.count(a + " " + b) == pt.count(a) + pt.count(b)  # whitespace-separated


def test_count_batch_is_exactly_count_per_text() -> None:
    """``len_max`` tokenises about 177 M strings through ``count_batch``, so a batched path
    that drifted from :meth:`count` would silently change the pack budget for the whole
    corpus. The empty string and the empty batch are included."""
    pt = PackTokenizer(_synthetic_tokenizer())
    texts = ["uno dos tres", "", "привет мир", "你好 世界", "hello"]
    assert pt.count_batch(texts) == [pt.count(t) for t in texts]
    assert pt.count_batch([]) == []


def test_num_special_is_zero_for_a_plain_tokenizer() -> None:
    # a WordLevel tokenizer with no post-processor adds no <s>/</s>: 0 reserved tokens.
    assert PackTokenizer(_synthetic_tokenizer()).num_special() == 0


def test_load_pack_tokenizer_matches_pinned_revision(make_cfg) -> None:
    """The loader reads the same pinned revision the config records.

    Skipped where bge-m3 cannot be fetched, so the offline suite stays green.
    """
    from tokenizers import Tokenizer

    cfg = make_cfg()
    repo, rev = parse_tokenizer_id(cfg.blocks["chunker"]["config"]["tokenizer_id"])
    try:
        pinned = Tokenizer.from_pretrained(repo, revision=rev)
        pt = load_pack_tokenizer(cfg)
    except Exception as exc:  # noqa: BLE001, unavailable offline: skip rather than fail
        pytest.skip(f"bge-m3 tokenizer unavailable offline: {exc}")
    ref = PackTokenizer(pinned)
    for s in ("el café tiene dos gramos", "报告 关于 电子烟", "hello world"):
        assert pt.count(s) == ref.count(s)  # same pinned revision, not latest main
    # bge-m3 (XLM-R) wraps a single sequence as <s> … </s> -> 2 reserved tokens; the pack
    # budget reserves exactly these (512 - 2 = 510 content), so a full passage fits the window.
    assert pt.num_special() == 2
    tk._load.cache_clear()
