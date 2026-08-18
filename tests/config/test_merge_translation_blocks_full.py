"""The corpus-build blocks over the six real configs.

Covers `merge`, `translation` and `reconcile`. All three are semantically
fairness-shared: the family builds one merge map, one translated corpus and one final
inventory, and each block's hash keys the artifact path (``reconcile``'s as one quarter
of the composite ``final/<recon12>/`` key), so a divergent member would silently build a
different corpus rather than a different rendering of the same one.

The check here is byte-identical block text and identical ``all_hashes`` entries across
the whole six-config roster.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ragtime.config import all_hashes, load
from ragtime.config.schema import _ALLOWED, SHARED_BLOCKS

pytestmark = pytest.mark.full

_BLOCKS = ("merge", "translation", "reconcile")


def _block_text(text: str, name: str) -> str:
    """The block's raw body, column-0 key through the line before the next column-0 key."""
    lines = text.splitlines(keepends=True)
    start = next(
        (i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(name)}:", ln)), None
    )
    assert start is not None, f"top-level block {name!r} not found"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^[A-Za-z_][\w-]*:", lines[j]):
            end = j
            break
    return "".join(lines[start:end])


@pytest.mark.parametrize("name", _BLOCKS)
def test_block_is_present_in_every_real_config(name, real_e2e_paths, real_mlir_paths):
    for p in real_e2e_paths + real_mlir_paths:
        assert f"\n{name}:" in p.read_text(encoding="utf-8"), p.name


@pytest.mark.parametrize("name", _BLOCKS)
def test_block_text_is_byte_identical_across_the_whole_roster(
    name, real_e2e_paths, real_mlir_paths
):
    bodies = {
        p.name: _block_text(p.read_text(encoding="utf-8"), name)
        for p in real_e2e_paths + real_mlir_paths
    }
    distinct = set(bodies.values())
    assert len(distinct) == 1, (
        f"{name}: {len(distinct)} distinct block bodies across the roster; "
        f"{sorted(bodies)} must be byte-identical"
    )


@pytest.mark.parametrize("name", _BLOCKS)
def test_block_hash_is_identical_across_the_whole_roster(
    name, real_e2e_paths, real_mlir_paths
):
    """The semantic hash keys the artifact, so it must agree, not just the text."""
    hashes = {p.name: all_hashes(load(p))[name] for p in real_e2e_paths + real_mlir_paths}
    assert len(set(hashes.values())) == 1, hashes


@pytest.mark.parametrize("name", _BLOCKS)
def test_block_is_declared_in_allowed_so_the_configs_load(name):
    """A block absent from ``_ALLOWED`` is rejected as an unknown key by every config."""
    assert name in _ALLOWED


def test_shared_blocks_records_the_completed_promotion():
    """The fairness-shared set is pinned here, name for name.

    Adding a name without updating the standalone copies is an immediate launch failure,
    so the set is pinned explicitly and a change to it is a deliberate, cross-file edit:
    ``schema.validate`` raises ``missing required block`` for any name in this tuple that
    a config lacks.

    Each entry keys either a corpus artifact the whole family shares or a policy that
    would otherwise let two members of one family differ in something other than the
    translation knob. ``merge``, ``translation``, ``reconcile``, ``packing`` and
    ``index_build`` key the corpus and the index; ``rag_loop`` fixes how much evidence a
    loop gathers before it answers, so differing budgets would make a translation delta
    an artifact of effort; ``serialize`` fixes the terminal selection policy; ``topics``
    names the request set, and two members reading different requests are not comparable
    at all.
    """
    assert SHARED_BLOCKS == (
        "llm",
        "claim_commit",
        "decomposition",
        "rag_loop",
        "chunker",
        "merge",
        "translation",
        "reconcile",
        "packing",
        "index_build",
        # The terminal selection policy (k_t3, answers_cap, dedup cutoffs, rrf_k,
        # top_docs). Two members of a family selecting differently would make a
        # translation delta indistinguishable from a selection artifact. It keys
        # nothing on disk, so no corpus artifact is re-keyed by its presence here.
        "serialize",
        # The request set the run reads. The T1 and T3 contrast is Original against OMT
        # over the same requests, so two members reading different topics files are not
        # comparable and every delta would be an artifact of the input set. It keys
        # nothing on disk: the corpus and index are keyed by chunker, merge,
        # translation, reconcile, packing and index_build. One honest limit is that
        # `family_guard` groups by family, so this is enforced within `e2e*` and within
        # `mlir*` but never across the two; that all runs share one request set is
        # asserted separately by
        # `test_all_6_real_configs_name_the_same_existing_topics_file`.
        "topics",
    )
    # The three corpus-build blocks are part of that set.
    assert set(_BLOCKS) <= set(SHARED_BLOCKS)


def test_hardware_facts_stay_out_of_the_hashed_translation_block(
    real_e2e_paths, real_mlir_paths
):
    """A filesystem path, a GPU shape key or a shard count must never enter a fairness hash.

    Checked on the parsed block, which is what ``config_hash`` hashes, rather than on the
    raw text: the block's prose names ``execution.ct2_model_dir`` to say where the path
    lives instead, and a raw-text grep would flag its own signpost.
    """
    for p in real_e2e_paths + real_mlir_paths:
        keys = set(load(p).blocks["translation"]["config"])
        assert "ct2_model_dir" not in keys, p.name
        assert "translate_shape_key" not in keys, p.name
        assert not any(k.endswith(("_shards", "_oversubscription")) for k in keys), p.name


def test_execution_substage_widths_are_declared_but_unhashed(real_e2e_paths):
    """Throughput knobs live in ``execution``, where a per-config tweak changes no artifact."""
    for key in (
        "merge_shards",
        "merge_oversubscription",
        "translate_shards",
        "translate_oversubscription",
        "translate_identity_shards",
        "translate_identity_oversubscription",
        "reconcile_shards",
        "reconcile_oversubscription",
        "translate_shape_key",
        "ct2_model_dir",
    ):
        assert key in _ALLOWED["execution"], key
    cfg = load(Path(real_e2e_paths[0]))
    for key in (
        "merge_shards",
        "translate_shards",
        "reconcile_shards",
        "translate_shape_key",
        "ct2_model_dir",
    ):
        assert key in cfg.blocks["execution"], key


def test_reconcile_composite_key_moves_with_every_one_of_its_four_inputs(real_e2e_paths):
    """``recon12`` must change when any of chunker, merge, translation or reconcile changes.

    This is the load-bearing property of the composite key: the final corpus is a join of
    independently versioned inputs, so a key naming only one would leave the others
    silently substitutable at the same path, and a v1 inventory beside a v2 translation
    would find the previous run's ``_SUCCESS``. Perturbing each input in turn shows the
    stale pairing is unaddressable rather than merely asserted against.
    """
    import types

    from ragtime.preprocess.reconcile import reconcile_hash

    cfg = load(Path(real_e2e_paths[0]))
    base = reconcile_hash(cfg)
    for block, edit in (
        ("chunker", {"config": {"token_budget": 384}, "hash": "x"}),
        ("merge", {"min_sentence_tokens": 99}),
        ("translation", {"config": {"beam_size": 1}, "hash": "x"}),
        ("reconcile", {"split_back": False, "min_segment_chars": 1}),
    ):
        blocks = dict(cfg.blocks)
        blocks[block] = edit
        moved = reconcile_hash(types.SimpleNamespace(blocks=blocks))
        assert moved != base, f"{block}: recon12 did not move"
