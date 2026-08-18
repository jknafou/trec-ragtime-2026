"""Small-fixture smoke tests for config fairness.

Everything here is synthetic; no tracked ``config/*.yml`` is opened for writing. Covers
unknown-key rejection, ``seeds`` enforcement, ``config_hash`` and ``all_hashes``
stability, sensitivity and recursive NFC, the seed-bank precondition in isolation,
``DuplicateKeyError`` propagation, and the import-purity checks.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
import sys
import unicodedata

import pytest

from ragtime.config import (
    ConfigError,
    FairnessError,
    RunConfig,
    all_hashes,
    config_hash,
    family_guard,
    load,
    shared_block_hash,
)

pytestmark = pytest.mark.small

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- #
# Unknown-key rejection (the "file stays the complete record" invariant)
# --------------------------------------------------------------------------- #
def test_valid_synthetic_configs_pass(tmp_path, write_config, synthetic_e2e_text, synthetic_mlir_text):
    e2e = load(write_config(tmp_path / "a", synthetic_e2e_text))
    mlir = load(write_config(tmp_path / "b", synthetic_mlir_text))
    assert isinstance(e2e, RunConfig) and isinstance(mlir, RunConfig)
    # Both read `e2e_agentic`, and that is the contract, not a copy-paste slip: `run.kind` is a
    # one-value enum. The two fixtures differ by FAMILY (`run.id` prefix) and by which knob they
    # move, never by kind -- exactly as the six shipped configs do.
    assert e2e.kind == "e2e_agentic" and mlir.kind == "e2e_agentic"


def test_unknown_run_kind_raises(tmp_path, write_config, synthetic_e2e_text):
    """`run.kind` is a one-value enum, and a one-value enum must still be able to FAIL.

    Worth its own test precisely because there is only one legal value: a refactor that
    dropped the membership check, or widened it to "any non-empty string", would leave every
    shipped config passing and nothing would notice. `pipeline.driver` carries the twin of
    this test on the dispatch side (`test_drive_refuses_an_unknown_run_kind`).
    """
    for bad in ("mlir_decomposition_driven", "something_new"):
        text = synthetic_e2e_text.replace("kind: e2e_agentic", f"kind: {bad}")
        with pytest.raises(ConfigError, match="run.kind"):
            load(write_config(tmp_path / bad, text))


def test_reject_unknown_top_level_key(tmp_path, write_config, synthetic_e2e_text):
    text = synthetic_e2e_text + "bogus_block: {}\n"
    with pytest.raises(ConfigError, match="bogus_block"):
        load(write_config(tmp_path, text))


def test_reject_unknown_key_in_nested_block(tmp_path, write_config, mutate_block, synthetic_e2e_text):
    text = mutate_block(synthetic_e2e_text, "llm", lambda b: b + "  rogue_field: x\n")
    with pytest.raises(ConfigError, match="rogue_field"):
        load(write_config(tmp_path, text))


def test_reject_unknown_key_in_outputs_item(tmp_path, write_config, mutate_block, synthetic_e2e_text):
    text = mutate_block(synthetic_e2e_text, "outputs", lambda b: b + "    rogue: x\n")
    with pytest.raises(ConfigError, match="rogue"):
        load(write_config(tmp_path, text))


@pytest.mark.parametrize(
    "block", ["llm", "claim_commit", "rag_loop", "decomposition", "chunker", "topics"]
)
def test_missing_shared_block_raises(tmp_path, write_config, mutate_block, synthetic_e2e_text, block):
    # A config missing a shared block must not launch (else it hashes to "" and
    # passes family_guard vacuously).
    text = mutate_block(synthetic_e2e_text, block, lambda _b: "")
    with pytest.raises(ConfigError, match=block):
        load(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("line", "field"),
    [
        ("    track: report_generation\n", "track"),
        ('    path: "submissions/report_generation/run_synth.jsonl"\n', "path"),
        ('    run_id: "e2e-synth"\n', "run_id"),
    ],
)
def test_outputs_item_missing_required_field_raises(
    tmp_path, write_config, synthetic_e2e_text, line, field
):
    text = synthetic_e2e_text.replace(line, "")
    with pytest.raises(ConfigError, match=field):
        load(write_config(tmp_path, text))


# --------------------------------------------------------------------------- #
# seeds enforcement: never silently defaulted
# --------------------------------------------------------------------------- #
def test_seeds_missing_raises(tmp_path, write_config, synthetic_e2e_text):
    text = synthetic_e2e_text.replace("seeds: 1\n", "")
    with pytest.raises(ConfigError, match="seeds"):
        load(write_config(tmp_path, text))


def test_seeds_wrong_value_for_e2e_raises(tmp_path, write_config, synthetic_e2e_text):
    text = synthetic_e2e_text.replace("seeds: 1", "seeds: 3")
    with pytest.raises(ConfigError, match="seeds must be 1"):
        load(write_config(tmp_path, text))


def test_seeds_wrong_value_for_the_mlir_family_raises(tmp_path, write_config, synthetic_mlir_text):
    text = synthetic_mlir_text.replace("seeds: 1", "seeds: 5")
    with pytest.raises(ConfigError, match="seeds must be 1"):
        load(write_config(tmp_path, text))


# --------------------------------------------------------------------------- #
# Seed-bank precondition in isolation
# --------------------------------------------------------------------------- #
def test_family_guard_seed_bank_precondition_unequal_seeds_raises(
    tmp_path, write_config, synthetic_e2e_text
):
    cfg1 = load(write_config(tmp_path, synthetic_e2e_text))
    # Otherwise fairness-compliant (identical shared blocks + one knob moved), but
    # a differing seed count. The kind-derived check in validate is bypassed with
    # replace so that the seed-bank precondition is isolated.
    cfg2 = dataclasses.replace(cfg1, run_id="e2e-synth2", seeds=3)
    with pytest.raises(FairnessError, match="seed"):
        family_guard([cfg1, cfg2])


# --------------------------------------------------------------------------- #
# config_hash and all_hashes
# --------------------------------------------------------------------------- #
_REFLOWED_LLM = (
    "llm:\n"
    "  # a reordered, re-quoted, commented copy with the same semantic content\n"
    "  pin: [model_id, prompt_version_hash, seed]\n"
    "  decoding: { top_k: 20, top_p: 0.8, temperature: 0.7 }\n"
    "  role: 'decomposer + k loops'\n"
    "  model: synth-model\n"
)


def test_config_hash_stable_under_cosmetic_reflow(
    tmp_path, write_config, mutate_block, synthetic_e2e_text
):
    reflowed = mutate_block(synthetic_e2e_text, "llm", lambda _b: _REFLOWED_LLM)
    cfg1 = load(write_config(tmp_path / "a", synthetic_e2e_text))
    cfg2 = load(write_config(tmp_path / "b", reflowed))
    assert config_hash(cfg1.blocks["llm"]) == config_hash(cfg2.blocks["llm"])


def test_config_hash_sensitive_to_value_change(tmp_path, write_config, synthetic_e2e_text):
    changed = synthetic_e2e_text.replace("temperature: 0.7", "temperature: 0.5")
    cfg1 = load(write_config(tmp_path / "a", synthetic_e2e_text))
    cfg2 = load(write_config(tmp_path / "b", changed))
    assert config_hash(cfg1.blocks["llm"]) != config_hash(cfg2.blocks["llm"])


def test_config_hash_recursive_nfc(tmp_path, write_config, synthetic_e2e_text):
    precomposed = "café"  # 'é' = U+00E9, at a nested path (llm.role)
    decomposed = unicodedata.normalize("NFD", precomposed)  # 'e' + U+0301
    assert precomposed != decomposed
    role = 'role: "decomposer + k loops"'
    text_a = synthetic_e2e_text.replace(role, f'role: "decomposer {precomposed}"')
    text_b = synthetic_e2e_text.replace(role, f'role: "decomposer {decomposed}"')
    cfg_a = load(write_config(tmp_path / "a", text_a))
    cfg_b = load(write_config(tmp_path / "b", text_b))
    assert config_hash(cfg_a.blocks["llm"]) == config_hash(cfg_b.blocks["llm"])


def test_all_hashes_shape(tmp_path, write_config, synthetic_e2e_text):
    cfg = load(write_config(tmp_path, synthetic_e2e_text))
    hashes = all_hashes(cfg)
    assert isinstance(hashes, dict)
    for name, value in hashes.items():
        assert isinstance(name, str)
        assert isinstance(value, str) and _HEX64.match(value)
    assert {"llm", "claim_commit", "rag_loop", "decomposition", "chunker"} <= set(hashes)


def test_retrieval_default_materialized_into_blocks(tmp_path, write_config, synthetic_e2e_text):
    # synthetic_e2e_text has no retrieval block; the resolved index default must be
    # materialised so all_hashes has a uniform key set across a family (Knob 1
    # symmetric with Knob 2's passage_lang write-back).
    cfg = load(write_config(tmp_path, synthetic_e2e_text))
    assert cfg.retrieval_index == "original"
    assert "retrieval" in cfg.blocks
    assert cfg.blocks["retrieval"]["index"] == "original"
    assert "retrieval" in all_hashes(cfg)


# --------------------------------------------------------------------------- #
# DuplicateKeyError propagation (not swallowed, not re-wrapped)
# --------------------------------------------------------------------------- #
def test_duplicate_top_level_block_raises(tmp_path, write_config, synthetic_e2e_text):
    from ruamel.yaml.constructor import DuplicateKeyError

    dup = synthetic_e2e_text + (
        "llm:\n"
        '  model: "dup"\n'
        '  role: "dup"\n'
        "  decoding: { temperature: 0.7, top_p: 0.8, top_k: 20 }\n"
        "  pin: [x]\n"
    )
    with pytest.raises(DuplicateKeyError):
        load(write_config(tmp_path, dup))


# --------------------------------------------------------------------------- #
# The shape the downstream stages consume
# --------------------------------------------------------------------------- #
def test_all_hashes_shape_is_layout_consumable(tmp_path, write_config, synthetic_e2e_text):
    cfg = load(write_config(tmp_path, synthetic_e2e_text))
    hashes = all_hashes(cfg)
    # A flat dict[str, str] mapping block name to 64-character lowercase hex, with no
    # nested structures and no non-string values: the shape layout and provenance consume.
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in hashes.items())
    assert all(_HEX64.match(v) for v in hashes.values())


def test_gate_sequence_is_pure_cpu_no_heavy_import(tmp_path, write_config, synthetic_e2e_text):
    # Run in a fresh interpreter (subprocess): an in-process `sys.modules` check is a
    # false-positive magnet once another module has polluted the session.
    cfg_path = write_config(tmp_path, synthetic_e2e_text)
    code = (
        "import sys\n"
        "from ragtime.config import load, family_guard\n"
        f"family_guard([load(r'{cfg_path}')])\n"
        "bad = [m for m in ('torch', 'vllm', 'transformers') if m in sys.modules]\n"
        "assert not bad, f'config gate imported heavy libs: {bad}'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_config_does_not_import_orchestration():
    # Fresh interpreter: importing `ragtime.config` must not drag in
    # `ragtime.orchestration`. Subprocess-isolated so a polluted session cannot pass.
    code = "import sys, ragtime.config; assert 'ragtime.orchestration' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_renderings_and_knob_values_agree() -> None:
    """``passage_store.RENDERINGS`` and ``schema.KNOB_VALUES`` are one vocabulary.

    They are two hand-maintained lists of the same thing, and they cannot be unified in
    code: ``common`` sits below ``config`` on the dependency spine, so importing
    ``KNOB_VALUES`` into ``passage_store`` would invert the spine. The cross-check has to
    live in a test.

    They have silently disagreed before: `omt_opus` was added to `KNOB_VALUES` so a config
    could select the low-tier rendering while `RENDERINGS` still listed the older set.
    Config validation passed, `family_guard` passed and every artifact check passed, and
    `render(pid, "omt_opus")` would have raised `unknown rendering` the moment retrieval
    first called it.
    """
    from ragtime.common.passage_store import RENDERINGS
    from ragtime.config.schema import KNOB_VALUES

    assert set(RENDERINGS) == set(KNOB_VALUES), (
        f"rendering vocabulary diverged: passage_store.RENDERINGS={sorted(RENDERINGS)} "
        f"vs config.schema.KNOB_VALUES={sorted(KNOB_VALUES)}; a config could name a "
        "rendering the passage store cannot serve (or vice versa)"
    )
    assert len(RENDERINGS) == len(set(RENDERINGS)), "RENDERINGS has duplicates"
    assert "original" in RENDERINGS, "the untranslated id spine must always be a rendering"


# --------------------------------------------------------------------------- #
# `topics`: the request set is part of the run record and is fairness-shared
# --------------------------------------------------------------------------- #
def test_topics_path_is_a_first_class_resolved_field(tmp_path, write_config, synthetic_e2e_text):
    cfg = load(write_config(tmp_path, synthetic_e2e_text))
    # Verbatim as written, never resolved against a root here (that is the caller's
    # decision, as with Layout.submission), so the value stays greppable and portable.
    assert cfg.topics_path == "topics/topics.all.2026.v0625-fix.jsonl"
    assert cfg.blocks["topics"]["path"] == cfg.topics_path


def test_topics_is_a_shared_block(tmp_path, write_config, synthetic_e2e_text):
    """Two members of a family reading different requests are not comparable at all."""
    from ragtime.config.schema import SHARED_BLOCKS

    assert "topics" in SHARED_BLOCKS
    # And it really participates in the byte-identity hash, not just the name list.
    a = shared_block_hash(synthetic_e2e_text)
    b = shared_block_hash(synthetic_e2e_text.replace("v0625-fix.jsonl", "v0625.jsonl"))
    assert a != b


def test_topics_must_be_a_mapping_not_a_bare_scalar(tmp_path, write_config, synthetic_e2e_text):
    text = synthetic_e2e_text.replace(
        "topics:\n  path: topics/topics.all.2026.v0625-fix.jsonl\n",
        "topics: topics/topics.all.2026.v0625-fix.jsonl\n",
    )
    with pytest.raises(ConfigError, match="topics"):
        load(write_config(tmp_path, text))


def test_topics_missing_path_raises(tmp_path, write_config, synthetic_e2e_text):
    text = synthetic_e2e_text.replace("  path: topics/topics.all.2026.v0625-fix.jsonl\n", "")
    with pytest.raises(ConfigError, match="topics.path"):
        load(write_config(tmp_path, text))


def test_topics_typoed_subkey_is_rejected_at_load(tmp_path, write_config, synthetic_e2e_text):
    """The reason `topics` is a mapping: `pathh:` dies here, not on a GPU node."""
    text = synthetic_e2e_text.replace(
        "  path: topics/topics.all.2026.v0625-fix.jsonl\n",
        "  pathh: topics/topics.all.2026.v0625-fix.jsonl\n",
    )
    with pytest.raises(ConfigError, match="pathh"):
        load(write_config(tmp_path, text))


def test_topics_empty_path_raises(tmp_path, write_config, synthetic_e2e_text):
    text = synthetic_e2e_text.replace(
        "  path: topics/topics.all.2026.v0625-fix.jsonl\n", '  path: ""\n'
    )
    with pytest.raises(ConfigError, match="topics.path"):
        load(write_config(tmp_path, text))
