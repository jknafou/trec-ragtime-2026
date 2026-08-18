"""The two index-leg encoder clients and their registry wiring.

Hermetic: nothing here loads ``torch``, ``transformers``, ``pylate`` or a checkpoint. What is
asserted is the three failures an earlier bring-up experiment hit (or that would break a
build), and that config rather than code owns the encoder identities:

- SM20 ``pyseismic-lsr`` components are ``U30`` strings and truncate silently past 30
  characters, which would merge two vocabulary dimensions into one posting list. The cap is
  asserted at the boundary, never relied upon.
- SM21 MILCO fetches a second repo (``BAAI/bge-m3-unsupervised``) by repo id at load time,
  so the pre-cache call must be made without ``revision=``, or no ``refs/main`` is written
  and offline resolution dies with ``LocalEntryNotFoundError``.
- Both clients come from ``serving.registry.build_clients``, read their identity from the
  hashed ``index_build`` block, and load nothing at construction time, so no stage ever
  instantiates a model and no unrelated ``build_clients`` caller pays for an index knob.
- The dense leg is a third such client (``index_dense``). It once read the query-time
  ``retrieval.dense`` key while the manifest and ``leg_config_hash`` claimed
  ``index_build.config.dense_model``. All six shipped configs now name ``BAAI/bge-m3`` under
  both, so the two disagree only in that ``index_build`` also pins ``dense_revision`` -- which
  is exactly why the tests below use synthetic configs where the two ids differ: on a real
  config the defect produces the right model at an unpinned revision and is invisible.
- The truncation counter on the late-interaction client is a real tokenizer operation and it
  reads only a token count, so it is the same path for en, es, ru and zh; a
  language-conditional measurement would break that fairness property. It outlived the MaxP
  windowing it once measured, since the truncation is now removed upstream by
  ``preprocess.packing``: the remedy moved, the measurement did not.
"""

from __future__ import annotations

import re
import types

import pytest

from ragtime.serving import late_interaction, sparse_milco
from ragtime.serving.encoders import DEFAULT_INDEX_DENSE_MODEL, Encoder
from ragtime.serving.late_interaction import MtdEncoder
from ragtime.serving.registry import build_clients, build_stub_clients
from ragtime.serving.sparse_milco import (
    SEISMIC_COMPONENT_MAXLEN,
    MilcoEncoder,
    SeismicComponentError,
    seismic_components,
)

pytestmark = pytest.mark.small


def _cfg(retrieval_dense: str = "BAAI/bge-m3", **index_build: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        blocks={
            "llm": {"model": "Qwen/Qwen3.5-122B-A10B-FP8"},
            "retrieval": {"dense": retrieval_dense, "sparse": "BAAI/bge-m3"},
            "translation": {"config": {"omt_model": "nllb-200-3.3B"}},
            "execution": {"ct2_model_dir": "runs/models/nllb-ct2"},
            "index_build": {"config": dict(index_build)},
        }
    )


# --------------------------------------------------------------------------- #
# SM20: the U30 component cap
# --------------------------------------------------------------------------- #
def test_sparse_components_are_stringified_dim_ids_in_a_total_order() -> None:
    components, values = seismic_components({305: 1.5, 12: 0.25})
    assert components == ["12", "305"]  # ordered by the string Seismic keys on
    assert values == [0.25, 1.5]
    assert all(len(c) <= SEISMIC_COMPONENT_MAXLEN for c in components)


def test_an_over_long_component_raises_instead_of_truncating_silently() -> None:
    """SM20: assert the cap rather than rely on it. Truncation would merge two dimensions."""
    too_long = "9" * (SEISMIC_COMPONENT_MAXLEN + 1)
    with pytest.raises(SeismicComponentError, match="U30 cap"):
        seismic_components({too_long: 1.0})
    # the boundary itself is legal
    assert seismic_components({"9" * SEISMIC_COMPONENT_MAXLEN: 1.0})[0] == [
        "9" * SEISMIC_COMPONENT_MAXLEN
    ]


def test_empty_weights_are_an_empty_vector_not_an_error() -> None:
    """A degenerate encode must stay countable: dropping the passage would break id equality."""
    assert seismic_components({}) == ([], [])


# --------------------------------------------------------------------------- #
# SM33: the top-k cut
#
# MILCO's raw output is a measured 819.4 nnz/passage, and the sparse leg costs 7.35 GiB per
# 131,072-passage part unpruned against 2.44 GiB at k=300 (~505 -> ~167 GiB per rendering).
# The cut is applied at ASSEMBLE time (``preprocess.index._SeismicWriter``) over vectors that
# were STORED unpruned, so what is asserted here is the function's contract: exactly k kept,
# a total order (hence a byte-reproducible build), and no behaviour change at all when the
# caller asks for none: which is what keeps ``preprocess.vectors``' stored form and the
# query path exactly as they were.
# --------------------------------------------------------------------------- #
def test_the_top_k_cut_keeps_exactly_the_k_heaviest_components() -> None:
    """SM33: k of n survive, and they are the heaviest, nothing else about the pair moves."""
    weights = {1: 0.1, 2: 0.9, 3: 0.5, 4: 0.7, 5: 0.3}
    components, values = seismic_components(weights, top_k=3)
    # component-STRING order is preserved (the pruned vector is a SUB-SEQUENCE of the full
    # one), and the survivors are the three largest weights
    assert components == ["2", "3", "4"]
    assert values == [0.9, 0.5, 0.7]
    assert sorted(values, reverse=True) == [0.9, 0.7, 0.5]
    # and the survivors are exactly a prefix of the full vector's weight ordering
    full_c, full_v = seismic_components(weights)
    assert len(full_c) == 5
    assert set(zip(components, values, strict=True)) <= set(
        zip(full_c, full_v, strict=True)
    )


def test_a_vector_at_or_below_k_is_untouched_by_the_cut() -> None:
    """SM33: no cut is not "a cut of zero", the pair must be byte-identical to the unpruned.

    Includes the boundary ``nnz == k``, where an off-by-one would silently drop the lightest
    component of every short passage in the corpus.
    """
    weights = {7: 0.4, 3: 0.9, 11: 0.2}
    unpruned = seismic_components(weights)
    assert seismic_components(weights, top_k=3) == unpruned  # nnz == k
    assert seismic_components(weights, top_k=4) == unpruned  # nnz < k
    assert seismic_components(weights, top_k=sparse_milco.NO_TOP_K) == unpruned
    assert unpruned == (["11", "3", "7"], [0.2, 0.9, 0.4])


def test_ties_are_broken_on_the_component_id_ascending() -> None:
    """SM33: the cut is a total order, which is what makes the build byte-reproducible.

    Equal weights are the case a "take the k largest" implementation leaves to the sort's
    accidental input order, and the input here is a dict whose iteration order is the
    encoder's, not ours. The rule is the component id ascending, and it is asserted against a
    mapping whose insertion order DISAGREES with it, so a fall-through to input order fails.
    """
    weights = {9: 1.0, 4: 1.0, 40: 1.0, 2: 1.0}  # all tied, inserted out of order
    assert seismic_components(weights, top_k=2)[0] == ["2", "4"]
    # ... and the rule reads identically whether the caller keys by int dim id (the encoder's
    # own output) or by the stored component STRINGS (what assemble reconstructs from
    # vectors.parquet): str(4) == str("4"), so the two spellings prune to the same set.
    as_strings = {"9": 1.0, "4": 1.0, "40": 1.0, "2": 1.0}
    assert seismic_components(as_strings, top_k=2) == seismic_components(weights, top_k=2)
    # a partial tie: the heavier component wins outright, the tie decides the second slot
    mixed = {9: 2.0, 4: 1.0, 40: 1.0, 2: 0.5}
    assert seismic_components(mixed, top_k=2)[0] == ["4", "9"]


def test_the_cut_is_byte_reproducible_across_calls_and_across_dict_orders() -> None:
    """SM33: two builds of the same weights produce the same postings, including at a tie."""
    weights = {i: (i % 7) / 7.0 for i in range(50)}
    first = seismic_components(weights, top_k=10)
    assert seismic_components(weights, top_k=10) == first
    # the same weights in a different dict order (an encoder is free to yield either)
    shuffled = dict(reversed(list(weights.items())))
    assert shuffled != weights or list(shuffled) != list(weights)
    assert seismic_components(shuffled, top_k=10) == first
    assert len(first[0]) == 10


def test_the_u30_cap_is_checked_on_every_component_pruned_or_not() -> None:
    """SM33 with SM20: the cap is a property of the vocabulary, not of what survived.

    An over-long component with a tiny weight would be cut away at k=1, and a corpus whose
    ids cannot be represented at all would then pass at one k and raise at another, which is
    exactly the silent-merge failure SM20 exists to prevent.
    """
    too_long = "9" * (SEISMIC_COMPONENT_MAXLEN + 1)
    with pytest.raises(SeismicComponentError, match="U30 cap"):
        seismic_components({too_long: 0.0001, "5": 1.0}, top_k=1)


# --------------------------------------------------------------------------- #
# SM21: the second-repo pre-cache is called without revision=
# --------------------------------------------------------------------------- #
def test_pivot_tokenizer_prefetch_is_called_without_a_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SM21: a revision-pinned snapshot writes no ``refs/main`` -> offline load fails."""
    seen: list[tuple[tuple, dict]] = []

    def fake_snapshot_download(*args, **kwargs):
        seen.append((args, kwargs))
        return "/cache/models--BAAI--bge-m3-unsupervised/snapshots/deadbeef"

    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    path = sparse_milco.prefetch_pivot_tokenizer()
    assert path.endswith("deadbeef")
    (args, kwargs) = seen[0]
    assert args == (sparse_milco.PIVOT_TOKENIZER_REPO,)
    assert "revision" not in kwargs, "a pinned revision writes no refs/main"
    assert sparse_milco.PIVOT_TOKENIZER_REPO == "BAAI/bge-m3-unsupervised"


# --------------------------------------------------------------------------- #
# The clients: lazy, identity-from-config, device owned here
# --------------------------------------------------------------------------- #
def test_both_clients_are_built_from_the_hashed_index_build_block() -> None:
    cfg = _cfg(
        sparse_model="omai-research/milco-650m",
        sparse_model_revision="567aeb03a5756c1e42793b9113925b4945ed5767",
        spine_model="hltcoe/plaidx-large-neuclir-mtd",
        spine_model_revision="1fb92d4d51a3e60d9942c0654e71428f29c7b5d8",
        document_length=220,
        query_length=32,
    )
    bundle = build_clients(cfg)
    assert bundle.milco.model == "omai-research/milco-650m"
    assert bundle.milco.revision == "567aeb03a5756c1e42793b9113925b4945ed5767"
    assert bundle.mtd_colbert.checkpoint == "hltcoe/plaidx-large-neuclir-mtd"
    assert bundle.mtd_colbert.revision == "1fb92d4d51a3e60d9942c0654e71428f29c7b5d8"
    assert bundle.mtd_colbert.document_length == 220
    assert bundle.mtd_colbert.query_length == 32
    # nothing is loaded at construction (no torch/pylate import happened)
    assert bundle.milco._backend is None
    assert bundle.mtd_colbert._backend is None


def test_a_missing_index_build_block_falls_back_to_the_pinned_defaults() -> None:
    """A chunk worker or an online vLLM node must not fail over an offline index knob."""
    cfg = types.SimpleNamespace(
        blocks={
            "llm": {"model": "m"},
            "retrieval": {"dense": "d", "sparse": "s"},
            "translation": {"config": {"omt_model": "nllb"}},
            "execution": {"ct2_model_dir": "dir"},
        }
    )
    bundle = build_clients(cfg)
    assert bundle.milco.model == sparse_milco.DEFAULT_MILCO_MODEL
    assert bundle.milco.revision == sparse_milco.DEFAULT_MILCO_REVISION
    assert bundle.mtd_colbert.checkpoint == late_interaction.DEFAULT_MTD_CHECKPOINT
    assert bundle.mtd_colbert.revision == late_interaction.DEFAULT_MTD_REVISION
    # 0 => "use the checkpoint's own metadata", never a silently different window
    assert bundle.mtd_colbert.document_length == 0
    assert bundle.mtd_colbert.query_length == 0


def test_unset_lengths_mean_the_checkpoint_decides_and_are_pinned_by_it() -> None:
    encoder = late_interaction.MtdEncoder("ckpt", document_length=0)
    assert encoder.document_length == 0
    assert encoder.effective_document_length.__doc__  # documented, not magic
    assert late_interaction.DEFAULT_DOCUMENT_LENGTH == 220
    assert late_interaction.DEFAULT_QUERY_LENGTH == 32


def test_the_bundle_exposes_both_index_legs_as_singletons() -> None:
    bundle = build_stub_clients(_cfg())
    assert isinstance(bundle.milco, MilcoEncoder)
    assert isinstance(bundle.mtd_colbert, late_interaction.MtdEncoder)
    assert bundle.milco is bundle.milco  # one field, one object


def test_milco_feeds_one_bucket_per_call_never_a_second_batching_policy() -> None:
    """The vendor's ``encode_text`` runs its own loop, so it must get exactly one bucket."""
    encoder = MilcoEncoder("m", "rev")
    seen: list[dict] = []

    class _Backend:
        def encode_text(self, texts, **kwargs):
            seen.append({"n": len(texts), **kwargs})
            import types as _t

            return _t.SimpleNamespace(
                coalesce=lambda: _t.SimpleNamespace(
                    indices=lambda: _FakeTensor([[0, 1], [7, 9]]),
                    values=lambda: _FakeTensor([1.5, 2.5]),
                )
            )

    encoder._backend = _Backend()
    rows = encoder.encode_text(["a", "b"])
    assert seen[0]["n"] == 2 and seen[0]["batch_size"] == 2
    assert seen[0]["return_dict"] is False  # keeps the GATED splade-v3 vocab out of the path
    assert rows == [{7: 1.5}, {9: 2.5}]
    assert encoder.encode_text([]) == []  # an empty bucket never reaches the vendor
    assert len(seen) == 1


class _FakeTensor(list):
    """Minimal stand-in for the torch sparse tensor's ``.tolist()`` surface."""

    def tolist(self):
        return list(self)


# --------------------------------------------------------------------------- #
# The DENSE index client: identity from index_build.config, never retrieval.dense
# --------------------------------------------------------------------------- #
def test_the_index_dense_client_is_built_from_the_hashed_index_build_block() -> None:
    """The regression: the manifest's ``dense_model`` must be the encoder that runs.

    Synthetic on purpose. The two leaves disagree here so the wiring is visible; in every shipped
    config they name the same model id and differ only in that ``index_build`` also pins
    ``dense_revision``, which is why the same defect would be invisible on a real config.
    """
    cfg = _cfg(
        retrieval_dense="Qwen/Qwen3-Embedding-4B",  # synthetic: no shipped config names this
        dense_model="BAAI/bge-m3",
        dense_revision="5617a9f61b028005a4858fdac845db406aefb181",
        encode_device="cuda",
    )
    bundle = build_clients(cfg)
    assert bundle.index_dense.model == "BAAI/bge-m3"
    assert bundle.index_dense.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert bundle.index_dense.device == "cuda"
    assert bundle.index_dense._backend is None  # lazy: nothing loaded at construction
    # and the query-time embedder is left exactly where it was (decompose dedup's open decision)
    assert bundle.embedder.model == "Qwen/Qwen3-Embedding-4B"
    assert bundle.index_dense is not bundle.embedder


def test_a_config_without_a_retrieval_dense_key_still_gets_the_index_model() -> None:
    """With the query-time leaf absent the old wiring built ``Encoder("")`` and encoded with it.

    Every shipped config does set ``retrieval.dense``, so this is the fallback branch rather than
    a picture of production; what it pins is that ``index_dense`` never depends on that leaf.
    """
    cfg = types.SimpleNamespace(
        blocks={
            "llm": {"model": "m"},
            "translation": {"config": {"omt_model": "nllb"}},
            "execution": {"ct2_model_dir": "dir"},
            "index_build": {"config": {"dense_model": "BAAI/bge-m3"}},
        }
    )
    bundle = build_clients(cfg)
    assert bundle.embedder.model == ""  # the retrieval knob is absent in this synthetic config
    assert bundle.index_dense.model == "BAAI/bge-m3"


def test_the_index_dense_client_is_the_embedder_when_the_identity_is_the_same() -> None:
    """One resident model per identity: never two copies of one checkpoint."""
    bundle = build_clients(_cfg(retrieval_dense="BAAI/bge-m3", dense_model="BAAI/bge-m3"))
    assert bundle.index_dense is bundle.embedder


def test_a_missing_index_build_block_falls_back_to_the_pinned_dense_default() -> None:
    cfg = types.SimpleNamespace(
        blocks={
            "llm": {"model": "m"},
            "retrieval": {"dense": "d"},
            "translation": {"config": {"omt_model": "nllb"}},
            "execution": {"ct2_model_dir": "dir"},
        }
    )
    bundle = build_clients(cfg)
    assert bundle.index_dense.model == DEFAULT_INDEX_DENSE_MODEL == "BAAI/bge-m3"
    assert bundle.index_dense is not bundle.embedder


# --------------------------------------------------------------------------- #
# The QUERY-side dense client: the same identity, on its own device
# --------------------------------------------------------------------------- #
def test_query_dense_is_the_index_encoder_itself_when_the_query_device_leaf_is_absent() -> None:
    """No shipped config carries ``query_encode_device``, so nothing may change.

    Object identity, not equality: the collapse is what keeps one resident dense encoder per
    node. If this ever became a second ``Encoder``, a decompose node would load the checkpoint
    twice for no reason and the small tier would not notice.
    """
    bundle = build_clients(_cfg(dense_model="BAAI/bge-m3", encode_device="cuda"))
    assert bundle.query_dense is bundle.index_dense
    assert bundle.query_dense.device == "cuda"


def test_query_dense_is_the_index_encoder_when_the_leaf_merely_repeats_the_build_device() -> None:
    """Stating the device you already had is not a reason to hold two copies of a model."""
    bundle = build_clients(
        _cfg(dense_model="BAAI/bge-m3", encode_device="cuda", query_encode_device="cuda")
    )
    assert bundle.query_dense is bundle.index_dense


def test_query_encode_device_moves_the_query_encoder_and_leaves_the_build_encoder_alone() -> None:
    """The point: a CPU decompose client, with the index's own encoder identity.

    ``index_dense`` is what ``preprocess.index`` hands the vectorize step, and its device is
    the hashed, manifest-recorded ``encode_device``, so it must not move here, or the manifest
    becomes a false record of what encoded the corpus.
    """
    bundle = build_clients(
        _cfg(
            dense_model="BAAI/bge-m3",
            dense_revision="5617a9f61b028005a4858fdac845db406aefb181",
            encode_device="cuda",
            query_encode_device="cpu",
        )
    )
    assert bundle.query_dense is not bundle.index_dense
    assert bundle.query_dense.device == "cpu"
    assert bundle.index_dense.device == "cuda"  # the BUILD encoder is untouched
    # same identity => same embedding space as the index; only the hardware moved
    assert bundle.query_dense.model == bundle.index_dense.model == "BAAI/bge-m3"
    assert (
        bundle.query_dense.revision
        == bundle.index_dense.revision
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert bundle.query_dense._backend is None  # still lazy: nothing loaded at construction


def test_the_query_device_leaf_is_excluded_from_index_hash_so_it_cannot_orphan_the_index() -> None:
    """A query embedding stores no byte, so stating it must re-key nothing.

    Asserted against the live ``QUERY_TIME_KEYS`` membership and against ``index_hash``
    itself, because a key can be declared excluded and still be missing from the exclusion
    list, which is what happened to ``query_plaid_device``.
    """
    from ragtime.preprocess.index import QUERY_TIME_KEYS, index_hash

    assert "query_encode_device" in QUERY_TIME_KEYS

    def _cfg_with(**index_build):
        return types.SimpleNamespace(
            blocks={
                "index_build": {"config": {"dense_model": "BAAI/bge-m3", **index_build}},
                "packing": {"config": {"target": 1}},
            }
        )

    assert index_hash(_cfg_with(encode_device="cuda")) == index_hash(
        _cfg_with(encode_device="cuda", query_encode_device="cpu")
    )
    # and the BUILD device is still hashed: the exclusion must not have leaked onto it
    assert index_hash(_cfg_with(encode_device="cuda")) != index_hash(
        _cfg_with(encode_device="cpu")
    )


def test_an_absent_query_device_inherits_the_build_device_rather_than_a_constant() -> None:
    """A config that moved the BUILD to CPU must not silently embed queries on a card."""
    from ragtime.preprocess.index import index_build_options

    cfg = types.SimpleNamespace(
        blocks={
            "index_build": {"config": {"dense_model": "BAAI/bge-m3", "encode_device": "cpu"}},
            "packing": {"config": {"target": 1}},
        }
    )
    opts = index_build_options(cfg)
    assert opts.encode_device == "cpu"
    assert opts.query_encode_device == "cpu"


def test_a_revision_is_passed_to_sentence_transformers_only_when_declared() -> None:
    """An unset pin must keep the previous call shape, never a silently different default."""
    seen: list[dict] = []

    class _ST:
        def __init__(self, model, **kwargs):
            seen.append({"model": model, **kwargs})

    sentence_transformers = pytest.importorskip("sentence_transformers")
    orig = sentence_transformers.SentenceTransformer
    try:
        sentence_transformers.SentenceTransformer = _ST
        Encoder(model="m", device="cpu")._dense_backend()
        Encoder(model="m", device="cpu", revision="abc123")._dense_backend()
    finally:
        sentence_transformers.SentenceTransformer = orig
    assert seen[0] == {"model": "m", "device": "cpu"}
    assert seen[1] == {"model": "m", "device": "cpu", "revision": "abc123"}


def test_the_dense_leg_feeds_one_bucket_per_call_never_a_second_batching_policy() -> None:
    """The pinned ``Batcher`` bucket is the batch: sentence-transformers must not re-split.

    Its default ``batch_size=32`` and internal length sort would re-partition the bucket by
    text length, which differs across renderings for the same passage: a rendering-correlated
    perturbation of exactly the batch composition this project pins for bitwise
    reproducibility (100 % identical with composition held, against 60.97 % when it moves).
    The sparse leg already gets this right in ``MilcoEncoder.encode_text``; the dense one
    shipped without it and without a test, which is why this one exists.
    """
    seen: list[dict] = []

    class _Backend:
        def encode(self, texts, **kwargs):
            seen.append({"n": len(texts), **kwargs})
            return [[0.0]] * len(texts)

    encoder = Encoder(model="m", device="cpu")
    encoder._backend = _Backend()
    bucket = ["a", "bb", "ccc", "d" * 4000]  # length-heterogeneous, like a real bucket
    encoder.embed(bucket, mode="dense")
    assert seen[0]["n"] == len(bucket)
    assert seen[0]["batch_size"] == len(bucket), "the whole bucket must be ONE batch"
    assert seen[0]["normalize_embeddings"] is True  # unchanged
    # a single-query encode is still one batch, and an empty bucket never asks for 0
    encoder.embed(["q"], mode="dense")
    assert seen[1]["batch_size"] == 1
    encoder.embed([], mode="dense")
    assert seen[2] == {"n": 0, "batch_size": 1, "normalize_embeddings": True}


# --------------------------------------------------------------------------- #
# MaxP windowing: the remedy for the measured per-language truncation asymmetry
# --------------------------------------------------------------------------- #
class _WordTokenizer:
    """A hermetic stand-in for the checkpoint's fast tokenizer.

    One token per whitespace-delimited run OR per CJK codepoint: the property that matters
    for these tests is that Chinese text tokenizes to many tokens with no spaces, exactly the
    corpus condition that made zh truncate. Returns ``offset_mapping`` like a fast tokenizer.
    """

    _TOKEN = re.compile(r"[一-鿿]|[^\s一-鿿]+")
    SPECIALS = 2

    def __call__(
        self,
        texts,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict:
        items = [texts] if isinstance(texts, str) else list(texts)
        offsets = [[(m.start(), m.end()) for m in self._TOKEN.finditer(t)] for t in items]
        extra = self.SPECIALS if add_special_tokens else 0
        out: dict = {"input_ids": [[0] * (len(o) + extra) for o in offsets]}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


class _FakeColbert:
    """Just enough PyLate surface for the tokenizer-side tests (no torch, no weights)."""

    def __init__(self) -> None:
        self.tokenizer = _WordTokenizer()
        self.document_length = 12


def _windowed(**knobs) -> MtdEncoder:
    encoder = MtdEncoder("ckpt", document_length=12, **knobs)
    encoder._backend = _FakeColbert()
    return encoder


def test_the_truncation_counter_measures_the_path_that_actually_runs() -> None:
    """The counter measures the path that actually runs, not a windowed one.

    There is no encode-time windowing - the truncation is removed upstream by
    ``preprocess.packing`` - and this counter is the alarm that fires if that upstream guarantee
    ever stops holding, so it must report the real loss of the path that runs: the whole passage
    against ``document_length``, specials included, no truncation during measurement.
    """
    encoder = _windowed()
    long_text = " ".join(f"w{i}" for i in range(30))  # 30 tokens against a 12-token window
    assert encoder.truncated_tokens([long_text]) == [30 + 2 - 12]
    # a passage that fits reports zero: which is what the packed corpus must produce
    assert encoder.truncated_tokens(["a b c"]) == [0]
    assert encoder.truncated_tokens([]) == []


def test_the_encoder_has_no_windowing_surface_left() -> None:
    """The encoder exposes no windowing surface at all: a dormant knob is the phantom-knob defect.

    A ``window_texts``/``maxp_*`` attribute would invite a caller to enable half a remedy with no
    config behind it to read, while the real guarantee lives upstream in ``preprocess.packing``.
    """
    encoder = _windowed()
    for gone in ("window_texts", "window_bounds", "_token_offsets",
                 "maxp_enabled", "maxp_window_tokens", "maxp_stride_tokens"):
        assert not hasattr(encoder, gone), gone
    with pytest.raises(TypeError):
        MtdEncoder("ckpt", maxp_enabled=True)
