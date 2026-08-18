"""The MT client's config wiring: the exact trap this file exists to keep shut.

``translation``'s real shape is ``{config: {...}, hash: "..."}``, so a read of
``translation.get("model")`` / ``translation.get("engine")`` misses and falls through to its
default, handing CTranslate2 the literal string ``facebook/nllb-200-3.3B`` as a checkpoint
path on an offline compute node. The client is built from ``translation``'s ``config``
sub-mapping instead (``registry._mt_client``).

Two facts are asserted apart, never together. ``translation.config.omt_model`` is the
semantic identity: fairness-hashed, and stamped on every row's ``model_id``.
``execution.ct2_model_dir`` is the machine-local checkpoint directory: non-shared and
unhashed. Nothing here loads weights, and constructing a client is inert.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from ragtime.serving.mt import MtClient
from ragtime.serving.registry import build_clients

pytestmark = pytest.mark.small

_OMT_MODEL = "nllb-200-3.3B"
#: The pre-fix default. A config whose omt_model is not this value is what makes the
#: regression detectable at all.
_OLD_HARDCODED_DEFAULT = "facebook/nllb-200-3.3B"


def _cfg(
    *,
    ct2_model_dir: str | Path | None,
    omt_model: str | None = _OMT_MODEL,
    engine: str = "ctranslate2",
    compute_type: str | None = None,
) -> types.SimpleNamespace:
    """A config with the real block shape: knobs one level down, under ``config``."""
    translation_config: dict = {"engine": engine, "target_lang": "eng_Latn", "beam_size": 4}
    if compute_type is not None:
        translation_config["compute_type"] = compute_type
    if omt_model is not None:
        translation_config["omt_model"] = omt_model
    execution: dict = {"translate_shards": 1000}
    if ct2_model_dir is not None:
        execution["ct2_model_dir"] = str(ct2_model_dir)
    return types.SimpleNamespace(
        run_id="e2e-omt",
        blocks={
            "llm": {"model": "Qwen/Qwen3.5-122B-A10B-FP8"},
            "translation": {"config": translation_config, "hash": "c" * 64},
            "execution": execution,
        },
    )


@pytest.fixture
def ct2_dir(tmp_path: Path) -> Path:
    """A stand-in for the converted checkpoint dir (only its existence is checked)."""
    d = tmp_path / "nllb-200-3.3B-ct2-fp16"
    d.mkdir()
    (d / "model.bin").write_bytes(b"")
    return d


# --------------------------------------------------------------------------- #
# The wiring itself.
# --------------------------------------------------------------------------- #
def test_mt_client_reads_the_nested_translation_config_block(ct2_dir: Path) -> None:
    """The regression guard: a ``translation.get("model")``-style read would miss."""
    mt = build_clients(_cfg(ct2_model_dir=ct2_dir)).mt
    assert isinstance(mt, MtClient)
    assert mt.model == _OMT_MODEL  # from translation["config"]["omt_model"]
    assert mt.model != _OLD_HARDCODED_DEFAULT  # ... not the pre-fix fall-through default


def test_mt_engine_comes_from_the_nested_block_not_a_default(ct2_dir: Path) -> None:
    """A non-default engine value must survive the read; the old one always saw the
    ``ctranslate2`` default, whatever the block said. Only ctranslate2 is supported at load
    time, so this asserts the read, using a sentinel that could only come from config."""
    mt = build_clients(_cfg(ct2_model_dir=ct2_dir, engine="engine-from-config")).mt
    assert mt.engine == "engine-from-config"


def test_ct2_checkpoint_dir_comes_from_execution_not_from_the_hashed_block(
    ct2_dir: Path,
) -> None:
    """Semantic identity and filesystem path stay separate, both ways round."""
    mt = build_clients(_cfg(ct2_model_dir=ct2_dir)).mt
    assert mt.model_path == str(ct2_dir)  # the on-disk dir CT2 will open
    assert mt.model == _OMT_MODEL  # ... and the identity is untouched by it
    assert str(ct2_dir) not in mt.model  # a path must never become an identity


def test_compute_type_comes_from_the_hashed_block_and_is_not_left_implicit(
    ct2_dir: Path,
) -> None:
    """The declared quantization must be the one CT2 is ASKED for.

    Leaving ``compute_type`` unset let ``translation.config`` declare ``bfloat16`` while CT2
    silently used the checkpoint's own fp16: a false statement inside a fairness-hashed block,
    with nothing able to contradict it. It is now carried on the client and passed to the
    engine, so the two cannot disagree.
    """
    mt = build_clients(_cfg(ct2_model_dir=ct2_dir, compute_type="int8_float16")).mt
    assert mt.compute_type == "int8_float16"


def test_compute_type_defaults_to_the_checkpoints_own_float16(ct2_dir: Path) -> None:
    """A config that omits the key gets fp16: what the converted checkpoint actually is,
    and what every measurement and translated shard so far ran in."""
    mt = build_clients(_cfg(ct2_model_dir=ct2_dir)).mt
    assert mt.compute_type == "float16"


def test_the_shipped_configs_declare_the_compute_type_that_will_be_passed() -> None:
    """End-to-end on the real run record: the hashed value is the engine's value."""
    from ragtime.config import load

    repo = Path(__file__).resolve().parents[2]
    for name in ("e2e-omt.yml", "mlir-omt.yml"):
        cfg = load(repo / "config" / name)
        assert cfg.blocks["translation"]["config"]["compute_type"] == "float16"


def test_mt_client_is_the_same_singleton_within_one_bundle(ct2_dir: Path) -> None:
    bundle = build_clients(_cfg(ct2_model_dir=ct2_dir))
    assert bundle.mt is bundle.mt


def test_build_clients_does_not_load_any_mt_weights(ct2_dir: Path) -> None:
    """Constructing the bundle is inert: the CT2 engine is lazy on first call."""
    mt = build_clients(_cfg(ct2_model_dir=ct2_dir)).mt
    assert mt._translators == {} and mt._toks == {}


# --------------------------------------------------------------------------- #
# Loud, early failure: never an obscure CT2 error inside a GPU allocation.
# --------------------------------------------------------------------------- #
def test_missing_ct2_model_dir_raises_naming_the_config_key() -> None:
    with pytest.raises(ValueError, match="execution.ct2_model_dir is missing"):
        build_clients(_cfg(ct2_model_dir=None))


def test_a_bad_ct2_model_dir_fails_at_FIRST_USE_not_at_bring_up(tmp_path: Path) -> None:
    """Building the bundle must not touch the checkpoint; using the MT client must.

    Validating at construction coupled every consumer of ``build_clients`` to an offline MT
    artifact it may never touch. It broke ``test_chunk_worker_through_real_console_script``,
    a chunk test with no MT involvement, because ``ct2_model_dir`` is relative and that test
    runs from a temporary working directory. An online node standing up vLLM for the RAG loop
    would have failed the same way.
    """
    for bad in (tmp_path / "model.bin", tmp_path / "never-converted"):
        if bad.name.endswith(".bin"):
            bad.write_bytes(b"")
        # Bring-up succeeds: nothing has asked for a translation yet.
        bundle = build_clients(_cfg(ct2_model_dir=bad))
        # First use fails, still naming the config key, and now also the cwd it
        # resolved against: the detail that made the original failure opaque.
        with pytest.raises(ValueError, match="is not a directory"):
            bundle.mt._require_path("spa_Latn")


def test_missing_omt_model_raises_rather_than_stamping_an_empty_model_id(
    ct2_dir: Path,
) -> None:
    """An empty semantic identity would land silently on ~54 M rows' ``model_id``."""
    with pytest.raises(ValueError, match="translation.config.omt_model is missing"):
        build_clients(_cfg(ct2_model_dir=ct2_dir, omt_model=None))


def test_load_passes_the_compute_type_and_the_path_to_ctranslate2(
    ct2_dir: Path, monkeypatch
) -> None:
    """The engine call itself: the half a config assertion cannot reach.

    A fake ``ctranslate2`` module records the ``Translator(...)`` kwargs, so neither CT2 nor a
    GPU is involved, and the tokenizer slot is pre-filled so ``_load`` does not import
    transformers either.
    """
    import sys
    import types as _types

    seen: dict = {}

    class _FakeTranslator:
        def __init__(self, path, **kwargs):
            seen.update(path=path, **kwargs)

    monkeypatch.setitem(
        sys.modules, "ctranslate2", _types.SimpleNamespace(Translator=_FakeTranslator)
    )
    client = build_clients(_cfg(ct2_model_dir=ct2_dir, compute_type="float16")).mt
    client._toks[str(ct2_dir)] = object()  # skip the tokenizer load; this is about the engine
    client._load("spa_Latn")
    assert seen["path"] == str(ct2_dir)  # the checkpoint DIR, never the semantic identity
    assert seen["compute_type"] == "float16"  # declared == what CT2 is asked for


def test_mt_client_load_never_falls_back_to_the_semantic_identity_as_a_path() -> None:
    """Without an explicit ``model_path``, ``_load`` refuses: it does not try to open
    ``nllb-200-3.3B`` as a directory or resolve it as an HF repo id."""
    client = MtClient(model=_OMT_MODEL, engine="ctranslate2")
    assert client.model_path is None
    with pytest.raises(ValueError, match="no on-disk model_path"):
        client._load("spa_Latn")
