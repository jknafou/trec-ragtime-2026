"""Smoke tests of the real clients on CPU, skipped when the library or the model is absent.

These exercise the SaT segmenter and the batcher. They skip cleanly on a base environment
without the heavy extras, so they never break the small tier, and on a machine with the
extras installed and the model cached they prove the interface end to end."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.small


def test_sat_segmenter_split_tiny_text():
    pytest.importorskip("wtpsplit")
    from ragtime.serving.segmenter import Segmenter

    seg = Segmenter()
    try:
        out = seg.split("Nordic walking uses poles. It is popular in Finland.")
    except (OSError, RuntimeError, ImportError, ValueError) as exc:  # model not cached / offline
        pytest.skip(f"SaT model unavailable: {exc}")
    assert isinstance(out, list) and len(out) >= 1
    assert all(s.strip() for s in out)


def test_batcher_respects_token_budget_and_never_drops():
    from ragtime.serving.batching import Batcher, Tier

    items = ["a" * 10, "b" * 4000, "c" * 5000, "d" * 20]
    batches = Batcher().batch(items, Tier(token_budget=6000, max_items=8))
    assert sum(len(b) for b in batches) == len(items)  # nothing dropped
    for b in batches:
        assert sum(len(x) for x in b) <= 6000 or len(b) == 1  # over-budget item stands alone
