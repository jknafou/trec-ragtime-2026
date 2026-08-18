"""The three-leg index over the three renderings: one adapter, nine leg-variant builds.

The last corpus substage (chunk, merge, translate, reconcile, index) applies one build method to
three text columns (``original``, ``omt``, ``omt_opus`` -- ``common.passage_store.RENDERINGS``)
with three legs each:

- dense: ``BAAI/bge-m3`` (1024-dimensional, no instruction prefix) into FAISS, exact
  ``IndexFlatIP`` by default;
- sparse: MILCO (``milco-650m``) learned-sparse into a Seismic impact index;
- late_interaction: the MTD ColBERT checkpoint via PyLate and PLAID, one PLAID document per
  passage.

There is no encode-time MaxP windowing. The per-language truncation it would remedy is removed
where it is created, in ``preprocess.packing``: every passage is packed to fit the retrieval
window in every rendering, so the requirement is ``document_length >= packing.pack_budget``,
checked at bring-up by :func:`_assert_window_covers_packing`.

There is no BM25 leg and no fourth leg of any kind. A leg buildable only on the two English
renderings would make ``original`` a three-leg index and the other two four-leg ones, and "only
the rendering varies" would stop being true; MILCO already carries cross-lingual lexical signal on
all three. :class:`IndexAdapter` rejects any leg set that is not exactly :data:`LEGS`.

Four properties are structural rather than conventional:

1. Search returns ``(passage_id, score)`` and nothing else, by record schema. The only
   per-passage table any leg writes is :func:`idmap_arrow_schema` (``passage_id``,
   ``document_id``, ``source_lang``, ``ordinal``), which has no text column, and every engine is
   handed integer ordinals as document keys. ``document_id`` is a column rather than something
   re-derived at query time, so the always-original citation is carried.
2. The method is identical across renderings: one code path, one hashed ``index_build`` block,
   and a per-leg ``config_hash`` (embedder, tokenizer, ANN parameters) equal for all three
   because the rendering is not one of its inputs.
3. Batch composition is pinned to a rendering-invariant key. BGE-M3 vectors are bitwise-identical
   across the GPU architectures in use, but the same shard batched differently drops to about 61%
   bitwise-identical vectors while agreeing on 99.95% of the top ten, and
   ``serving.batching.Batcher`` buckets by length. Ordering and bucketing by
   ``(token_count, passage_id)`` -- from the shared, text-free passage table -- makes bucket
   membership identical across renderings, so a rank delta is attributable to translation content.
4. English is built once. :meth:`Layout.index_shard_dir` collapses ``source_lang == "en"`` to one
   ``_shared/en`` path whatever the variant, so reuse needs no copy or symlink step. Each
   variant's manifest names that path and ``merge`` still asserts the variant's id set covers
   every English ``passage_id``. The shared build encodes the native composition and counts
   (``index.shared_en_composition_divergence``) any English passage whose translated composition
   differs.

Each leg's output directory is published independently -- built into ``<leg>.tmp-<pid>``,
renamed, then given ``common.io``'s ``_SUCCESS`` marker -- and ``work`` skips any leg whose marker
exists, so an OOM in one leg of a three-encoder worker does not discard the other two. The
shard-level claim, heartbeat and poison machinery stays in
``orchestration.slurm.workqueue`` via ``orchestration.saturate``.

Passage text is composed, never stored: this stage streams
``common.passage_store.iter_final_passages`` and shares one ``compose_passage_text`` with the
passage store, so the native-slice-versus-translated-join rule is not re-derived here. Only the
claimed shard's passages are materialised, once, and reused by all three legs.

Two library surfaces are isolated in the three ``_*Writer``, ``open`` and ``search`` methods and
nowhere else: Seismic's ``save``/``load`` round-trip and PyLate's PLAID ``add_documents`` and
query surface. ``add_documents`` is called exactly once per part -- its incremental ``update``
path re-clusters and aborts the worker on real embeddings (see :class:`_PlaidWriter`).

The shard grain is ``(variant, source_lang, part)``. A ``(variant, source_lang)`` cell is cut into
parts of at most ``index_build.config.plaid_part_passages`` passages, each part a claimable work
unit with its own three legs, ordinal space and ``idmap.parquet``. The English cell alone is
nearly three million passages, roughly eleven hours of encode against a twelve-hour partition cap,
and an eleven-hour shard that dies at hour ten loses everything.

The cut is ordinal arithmetic: part ``p`` holds ordinals ``[p*N, (p+1)*N)`` of one language's rows
in the shared, text-free passage table. Two orders exist and must not be conflated. The shipped
``vectorize``/``assemble`` split cuts on the table's row order (``_row_order_lang_ids``), which
makes a part document-contiguous and the read proportional to the slice; the fused
:class:`IndexAdapter` cuts on the pinned ``(token_count, passage_id)`` order
(``_ordered_lang_keys``), which the split keeps as the within-block order that pins batch
composition. Either way the cut is a function of that one table, so part membership is identical
across renderings, and :meth:`IndexAdapter.merge` re-proves it per part by digest comparison.

A reader may never glob for parts. Every part writes ``part.json`` (its own index, the cell total,
the part size, its passage count) as the last act of ``work``, so a part directory without one is
an incomplete build. :func:`read_shard_parts` requires those witnesses to agree unanimously, to be
contiguous ``0..P-1``, and to number exactly ``P`` on disk, raising
:class:`IndexIntegrityError` otherwise. A fan over nine of ten parts would otherwise return a
full-looking, silently truncated result list.

Fan-in across the parts of one cell merges on score rather than RRF
(:func:`query_lang_leg`): parts of one cell share one score space, and RRF would rank a weak
part-2 hit equal to a strong part-1 hit. RRF stays in ``common.fusion``, where the score spaces
genuinely differ -- across legs and across languages.

The late-interaction leg is the one place a leg is not a single index: its single
``add_documents`` call would have to hold a whole unit's token embeddings in host RAM, hundreds of
gigabytes in fp32 for the English shard. So it cuts to ``(variant, source_lang, part)``, each part
an independent PLAID index, and :meth:`PlaidLateInteractionLeg.search` fans over every part and
merges by score. The part count is written in the leg (``parts.json``) and cross-checked against
the directories on disk; the merge sorts on ``(-score, ordinal)``, a total order; and
``idmap.parquet`` is untouched, so ordinals stay shard-global and a part is invisible to every
id-integrity check and to every caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping  # Runtime: isinstance in _without_query_time_keys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ragtime.common import Layout, Statistics, get_logger
from ragtime.common.io import (
    is_done,
    iter_parquet,
    iter_parquet_batches,
    read_jsonl,
    success_marker,
    write_jsonl,
)
from ragtime.common.io import write_parquet_stream as _write_parquet_stream
from ragtime.common.layout import INDEX_PART_GLOB, index_part_name
from ragtime.common.passage_store import RENDERINGS, iter_final_passages
from ragtime.config import ConfigError, all_hashes, config_hash
from ragtime.config.schema import INDEX_BUILD_CONFIG
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family
from ragtime.serving.batching import Batcher, Tier
from ragtime.serving.encoders import DEFAULT_INDEX_DENSE_MODEL
from ragtime.serving.late_interaction import DOCUMENT_MARKER_TOKENS
from ragtime.serving.sparse_milco import DEFAULT_MAX_LENGTH as DEFAULT_SPARSE_MAX_LENGTH
from ragtime.serving.sparse_milco import seismic_components

from .packing import packing_hash
from .reconcile import reconcile_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from ragtime.config import RunConfig

__all__ = [
    "ALL_LEG_KEYS",
    "ASSEMBLE_KEYS",
    "DENSE_LEG",
    "ENCODE_KEYS",
    "LATE_INTERACTION_LEG",
    "LEGS",
    "PLAID_ADD_PEAK_MULTIPLIER",
    "QUERY_TIME_KEYS",
    "QUERY_WORKERS_CAP",
    "QUERY_WORKERS_KEY",
    "SHARED_LANG",
    "SPARSE_LEG",
    "STRUCTURAL_KEYS",
    "FaissDenseLeg",
    "IndexAdapter",
    "IndexBuildOptions",
    "IndexCtx",
    "IndexIntegrityError",
    "IndexShardSpec",
    "LangHandle",
    "Leg",
    "LegHandle",
    "PlaidBufferTooLargeError",
    "PlaidLateInteractionLeg",
    "SeismicSparseLeg",
    "default_legs",
    "default_query_workers",
    "idmap_arrow_schema",
    "index_build_options",
    "index_hash",
    "kendall_tau_b",
    "leg_config_hash",
    "leg_encode_hash",
    "leg_of_config_key",
    "leg_recipe_keys",
    "open_lang",
    "open_shard",
    "plaid_part_count",
    "query_lang_leg",
    "query_leg",
    "query_part_workers",
    "rbo",
    "read_shard_parts",
    "resolve_query_plaid_device",
    "resolve_query_workers",
    "search_with_rep",
]

_log = get_logger("preprocess.index")

_DEFAULT_BASE = "runs"  # mirrors cli._LOCAL_ROOT; real runs pass an explicit base
_HASH_DIRLEN = 12  # same truncation as Layout's hash path levels

#: The three legs. Their names are contractual: they are the ``leg`` slice value on every
#: counter, the sub-directory of a shard, and the ``leg`` argument of :func:`query_leg`.
DENSE_LEG = "dense"
SPARSE_LEG = "sparse"
LATE_INTERACTION_LEG = "late_interaction"
LEGS: tuple[str, ...] = (DENSE_LEG, SPARSE_LEG, LATE_INTERACTION_LEG)

#: The one language whose shard is built once for all three variants. Structural, not a run
#: choice: English is the target of both MT arms, so its rendering is identity pass-through.
#: Kept equal to ``Layout``'s own private constant by :func:`Layout.index_shard_dir`'s
#: behaviour (asserted by a test), never by two hand-maintained literals agreeing by luck.
SHARED_LANG = "en"

#: The per-leg id map: the only per-passage table this stage writes, and the structural
#: reason a search cannot return text (see property 1 in the module docstring).
IDMAP_FILENAME = "idmap.parquet"

# Per-engine artifact names inside a leg directory.
_DENSE_FILENAME = "dense.faiss"

#: The stem every late-interaction part directory is named from: a leg holds
#: ``plaid-00000``, ``plaid-00001``, ...: never a bare ``plaid``, not even when there is
#: exactly one part. Uniformity is the point: a reader that had to handle "one part is named
#: differently" would have a code path in which a missing part looks like the normal case.
_PLAID_INDEX_NAME = "plaid"

#: The late-interaction leg's part census, written inside the leg's temp dir (so it is
#: published by the same atomic rename as the engine files) and read by every opener. A
#: reader never infers the part count: inferring it is exactly how a shard would be
#: searched short: a fan over 9 of 10 parts returns a full-looking, silently truncated result
#: list, and nothing downstream could tell. See :func:`_read_parts`.
_PARTS_FILENAME = "parts.json"

#: The shard-level census, one level up from :data:`_PARTS_FILENAME` and written by each
#: part into its own directory as the last act of :meth:`IndexAdapter.work`. Every part is
#: an independent witness to the same plan (``part``, ``parts``, ``part_passages``), so
#: :func:`read_shard_parts` needs no single writer that knows the whole cell: and unanimity
#: among P witnesses is strictly stronger evidence than one writer's claim. Its presence is
#: also what distinguishes a complete part from a crashed one: a ``part-*`` directory
#: without it raises rather than being fanned over.
_PART_FILENAME = "part.json"

# Seismic's save/load pair is asymmetric, so these two names are not interchangeable:
#   SeismicIndex.save(p)  -> path without extension, writes p + ".index.seismic"
#   SeismicIndex.load(q)  -> requires the full ".index.seismic" filename
# Passing one string to both publishes a leg that cannot be reopened, and the failure
# surfaces only at query time.
_SPARSE_SAVE_STEM = "sparse.seismic"
_SPARSE_FILENAME = _SPARSE_SAVE_STEM + ".index.seismic"

#: The engine each leg is actually built with. Derived from the artifact names above, so an
#: engine name exists exactly once in this module. These are structural, not run knobs: the
#: writers (``FaissDenseLeg`` / ``_SeismicWriter`` / ``_PlaidWriter``) are hard-wired, and
#: swapping an engine is a code change. They are named here so the corresponding
#: ``index_build.config`` keys can be validated against reality rather than silently accepted
#: (see :func:`_assert_structural_keys`).
_SPINE_ENGINE = "pylate+fast_plaid"  # pylate's PLAID index over the fast-plaid backend
ENGINE_OF: dict[str, str] = {
    DENSE_LEG: _DENSE_FILENAME.rsplit(".", 1)[-1],
    SPARSE_LEG: _SPARSE_SAVE_STEM.rsplit(".", 1)[-1],
    LATE_INTERACTION_LEG: _SPINE_ENGINE,
}

#: pyseismic-lsr 0.5.1's build call, verbatim from the extension module's own signature
#: string. It is recorded here because of what it does not contain: a seed. The clustering it
#: runs is ``RandomKmeans*``, and the Rust binary draws its random state from the OS
#: (``getrandom``), with no seed argument and no env knob to pin it.
SEISMIC_BUILD_SIGNATURE = (
    "SeismicIndex.build_from_dataset(dataset, n_postings=3500, centroid_fraction=0.1, "
    "min_cluster_size=2, summary_energy=0.4, max_fraction=1.5, doc_cut=15, nknn=0, "
    "knn_path=None, batched_indexing=None, num_threads=0)"
)

#: What ``PLAID.add_documents`` costs in host RAM, as a multiple of the buffer handed to it.
#: fast-plaid materializes an fp32 tensor whatever the input dtype, so halving the buffer
#: does not halve the peak and the fp16 multiplier is the larger of the two; this takes the
#: larger, since a run may store either. A measured property of the vendor call rather than
#: a run choice, so it is a code constant and not an ``index_build.config`` key. Re-measure
#: if fast-plaid is bumped.
PLAID_ADD_PEAK_MULTIPLIER = 3.3

#: fast-plaid's k-means assignment step has two implementations, and this stage pins the
#: reproducible one. ``fastkmeans`` resolves ``use_triton=None`` to True whenever Triton
#: imports, which it does in this container, so the shipped default runs a fused Triton kernel
#: that writes ``best_ids`` outside torch's determinism contract entirely. Repeating a build
#: under ``use_triton=True`` gives four distinct centroid sets; under ``use_triton=False`` it
#: gives one, with the determinism flag on or off.
#:
#: One layer up, ``compute_kmeans`` looks device-dependent, but device and kernel are not
#: independent there: the vendor resolves ``use_triton`` to True on cuda and False on cpu.
#: With the two variables split, cuda plus ``use_triton=False`` is byte-identical to the CPU
#: build, and ``fast_plaid_rust.create`` given fixed centroids is byte-identical either way.
#: The Triton kernel was therefore the whole of the late-interaction leg's irreproducibility;
#: every differing file is downstream of those centroids.
#:
#: Hard-wired rather than exposed as an ``index_build.config`` key, for the same reason
#: ``spine_engine`` is: an unpinned clustering would be a per-rendering difference with no
#: cause in the text, which is the confound the batch-composition pin exists to remove.
_PLAID_USE_TRITON_KMEANS = False

#: ``leg -> why its clustering cannot be seeded``. A documented deviation from the
#: deterministic offline build the index stage otherwise promises: two of the three legs are
#: seeded from the hashed block
#: (:attr:`IndexBuildOptions.train_seed` reaches FAISS's clustering parameters and PLAID's
#: k-means), and this one is not, because the vendor exposes nothing to seed. Stated in the
#: manifest and logged at build time rather than left as an unbacked spec claim, since a
#: per-rendering clustering starting from different random state is the kind of hidden,
#: rendering-correlated confound the batch-composition pin exists to remove. What survives
#: the deviation: the leg's input (vectors and their order) is still byte-identical to a
#: re-run's, and only Seismic's internal cluster assignment can differ.
UNSEEDABLE_LEGS: dict[str, str] = {SPARSE_LEG: SEISMIC_BUILD_SIGNATURE}

# --- counters (counter-only invariant: emit here, monitoring rolls up) ------- #
STAT_PASSAGES = "index.passages"
STAT_ENCODED = "index.encoded_passages"
STAT_OVERSIZED = "index.oversized_passages"
STAT_TRUNCATED = "index.late_interaction_truncated_tokens"
#: The sparse leg's own truncation counter. A separate metric id rather than a ``leg=`` slice
#: of the one above, because the name of a counter is part of its meaning (see
#: :data:`_TRUNCATION_SOURCE`).
STAT_SPARSE_TRUNCATED = "index.sparse_truncated_tokens"
STAT_LEG_BYTES = "index.leg_bytes"
STAT_LEG_RESUMED = "index.leg_skipped_resume"
STAT_SPARSE_EMPTY = "index.sparse_empty_vectors"
#: Sparse components dropped by the assemble-time top-k cut (``seismic_top_k``). Its own
#: metric id rather than a slice of the truncation counter, because it is a loss at a
#: different boundary: which dimensions of a vector are indexed, not which tokens were read.
#: Structurally zero when ``seismic_top_k`` is 0.
STAT_SPARSE_PRUNED = "index.sparse_pruned_components"
#: The late-interaction leg's part census, as counters: how many parts a shard was cut into
#: and the peak buffer any one of them needed (fp32 on the host). The second decides whether
#: a corpus-scale build fits the node at all, so it is measured on every build rather than
#: extrapolated from a bounded shard.
#:
#: ``STAT_PLAID_PART_BYTES`` is the buffer, not the worker's requirement: ``add_documents``'
#: own peak RSS is :data:`PLAID_ADD_PEAK_MULTIPLIER` times this, so sizing ``--mem`` from it
#: directly under-provisions by 2-3x. The same caveat applies to ``parts.json``'s
#: ``peak_part_bytes``.
STAT_PLAID_PARTS = "index.plaid_parts"
STAT_PLAID_PART_BYTES = "index.plaid_part_peak_bytes"
#: English passages whose ``omt``/``omt_opus`` composition differs from their ``original``.
#: Under ``reconcile.store_identity_translations: false`` no English translation row exists
#: and English resolves to the native slice in all three renderings, so this is structurally
#: zero. It is kept as a canary: a non-zero value means an identity row has reappeared in
#: some translations table, which is the one way the "English is one text" property breaks.
STAT_EN_DIVERGENCE = "index.shared_en_composition_divergence"
STAT_ID_INTEGRITY = "index.id_integrity_passages"

#: ``leg -> (IndexCtx client attribute, counter)`` for the legs whose encode imposes a window
#: of its own. The dense leg is absent by verification rather than oversight: ``Encoder.embed``
#: passes no ``max_length``, so BGE-M3 runs at its own 8192 ``max_seq_length`` against a
#: corpus maximum of 5,277 tokens. Adding a leg here is the whole cost of making its
#: truncation visible.
_TRUNCATION_SOURCE: dict[str, tuple[str, str]] = {
    LATE_INTERACTION_LEG: ("mtd", STAT_TRUNCATED),
    SPARSE_LEG: ("milco", STAT_SPARSE_TRUNCATED),
}

# Semantic defaults for the hashed ``index_build`` block. Used only when the block omits a
# key: every one of them changes the artifact, so a real run states them in config (which
# folds them into ``index_hash``). Same discipline as ``translate_omt``'s ``_DEFAULT_*``.
#: Imported, never re-typed: ``serving.encoders`` owns the one literal for this identity, and
#: the client the build gets is constructed from the same constant, so a second copy here
#: could drift from the encoder the registry hands out.
_DEFAULT_DENSE_MODEL = DEFAULT_INDEX_DENSE_MODEL
_DEFAULT_ANN_FACTORY = "Flat"  # exact IndexFlatIP, the default to beat
_DEFAULT_NPROBE = 0
_DEFAULT_TRAIN_SEED = 0
_DEFAULT_TRAIN_SAMPLE = 0  # 0 => train on every buffered vector (non-default path only)
_DEFAULT_OMP_THREADS = 0  # 0 => leave FAISS's own default alone
_DEFAULT_PLAID_NBITS = 2  # measured 0.46 TB corpus-wide at 12/12 rank-1
#: Passages per late-interaction part: the knob that makes a corpus-scale build possible.
#:
#: Derived from a 32 GiB per-part host-RAM budget rather than picked. fast-plaid's ``create``
#: must receive a part's whole token-embedding buffer in one ``add_documents`` call (see
#: :class:`_PlaidWriter`), so the part size is the peak that buffer reaches. ``packing``
#: guarantees every passage is at most ``pack_budget`` (511) tokens in every rendering, i.e.
#: 512 document positions, and the MTD encoder returns float32 on the host at 128 dims, so
#: 512 bytes per vector. The worst-case part is ``N * 512 * 128 * 4`` bytes, and ``2**17 =
#: 131,072`` puts that at exactly 32 GiB.
#:
#: The real corpus sits well below the ceiling, at 277.8 vectors per English passage, because
#: ColBERT drops punctuation embeddings; per-part buffers run 5.5-24.5 GiB across the four
#: languages. Doubling to ``2**18`` would put the worst English part at 48.9 GiB, over both
#: the budget and the worker allocation once ``add_documents``' copy is counted, which is why
#: the bound rather than the mean is what this is sized against.
#:
#: A run states it in config like every other recipe leaf: it changes the stored bytes, since
#: each part is clustered independently, so it is hashed into the leg's ``config_hash``.
_DEFAULT_PLAID_PART_PASSAGES = 131072
#: Imported, never re-typed (same discipline as ``_DEFAULT_DENSE_MODEL``): ``serving`` owns
#: the one literal for the sparse encode window, and the client the build gets is constructed
#: from that same constant, so the recipe this stage records and the window the encoder runs
#: cannot drift apart.
_DEFAULT_SPARSE_MAX_LENGTH = DEFAULT_SPARSE_MAX_LENGTH
_DEFAULT_SEISMIC_MIN_CLUSTER = 2
_DEFAULT_SEISMIC_CENTROID_FRACTION = 0.1
_DEFAULT_SEISMIC_QUERY_CUT = 200
_DEFAULT_SEISMIC_HEAP_FACTOR = 1.0
#: How many of a passage's sparse components reach Seismic: the assemble-time cut that
#: bounds this leg's size, applied by :class:`_SeismicWriter` and by nothing else.
#:
#: MILCO's raw density on a real part is 819.4 nnz per passage, and the leg costs 7.35 GiB
#: unpruned against 2.44 GiB at ``k=300``. The value is the model's published operating
#: point: arXiv:2510.00671 Table 11 puts MIRACL nDCG@10 at 72.7 for k=1000, 72.1 for k=300
#: and 68.4 for k=100, with the authors reporting only marginal gains beyond 300 tokens.
#:
#: It is an assemble key (:data:`ASSEMBLE_KEYS`, never :data:`ENCODE_KEYS`): the cut is
#: applied when the stored vectors are fed to the engine, so the stored sparse vectors stay
#: valid and a change of ``k`` costs a CPU re-assemble rather than a GPU re-encode.
#:
#: Rank churn against an unpruned twin is substantial (top-10 overlap 0.615, RBO 0.606 at
#: k=300). With no qrels the churn is all that can be observed directly, while the literature
#: predicts a quality cost under a point of nDCG@10; self-retrieval did not degrade at any k
#: down to 100.
_DEFAULT_SEISMIC_TOP_K = 300

#: Four Seismic build parameters that are run choices, so they are declared, hashed and
#: passed explicitly rather than left to ``pyseismic-lsr``'s own defaults. The values are the
#: vendor's current defaults: declaring them changes no byte, and changing them is
#: a separate decision with its own evidence. In particular ``max_fraction`` is deliberately
#: not aligned to the vendor's recommended 6, which would be an unevidenced behaviour change
#: riding along on an evidenced one.
#:
#: Each literal is cross-checked against :data:`SEISMIC_BUILD_SIGNATURE` (the vendor's own
#: signature string, recorded verbatim) by a test, so a vendor bump that moves a default
#: fails loudly instead of silently re-defining what "unchanged" meant.
_DEFAULT_SEISMIC_N_POSTINGS = 3500
_DEFAULT_SEISMIC_SUMMARY_ENERGY = 0.4
_DEFAULT_SEISMIC_MAX_FRACTION = 1.5
_DEFAULT_SEISMIC_DOC_CUT = 15

# Encode-composition defaults. These are semantic, and they live in the hashed
# ``index_build.config`` block (``encode_batch_token_budget``/``encode_max_items``), not in
# ``execution``: for MT, bucket composition is a reproducibility concern only, but here the
# same passage is encoded three times, once per rendering, and the comparison between those
# three is the experiment. The composition must therefore match across the per-rendering
# builds, which only a fairness-shared, hashed key guarantees: the same shard batched
# differently is only about 61 % bitwise-identical.
_DEFAULT_BATCH_TOKENS = 8192  # == serving.batching.Tier's own default
_DEFAULT_BATCH_ITEMS = 64
#: Passages per VECTORIZE block: the unit the encode half of the split produces, and the
#: unit an assemble reads. Derived, not picked: it is set equal to
#: :data:`_DEFAULT_PLAID_PART_PASSAGES` so one encode block is exactly one assemble part,
#: which is what makes an assemble read whole blocks and keeps every part boundary landing
#: where it would have landed in the un-split build.
#:
#: It is a hashed recipe key rather than an ``execution`` width, for the same reason
#: ``encode_batch_token_budget`` is: the block is the total order ``Batcher`` buckets within,
#: so its size decides batch membership and therefore the vectors bitwise. The throughput
#: knob is ``execution.vectorize_blocks_per_task``, which moves no boundary.
_DEFAULT_ENCODE_BLOCK_PASSAGES = _DEFAULT_PLAID_PART_PASSAGES

#: The two devices a build step may declare. Never ``"auto"``, which would hash identically
#: while resolving to different numerics, so one hash would cover two different artifacts.
_DEVICES: frozenset[str] = frozenset({"cuda", "cpu"})
#: The query device additionally admits an ordinal (``cuda:0``, ``cuda:1``), because a service
#: pins late-interaction to a specific card so the four language cells do not contend for one.
#: ``encode_device`` and ``assemble_device`` do not admit ordinals: they decide stored bytes,
#: and which card wrote an artifact may not vary under one hash. Reading is different, since
#: the ordinal selects hardware and hardware is not part of the recipe.
_QUERY_DEVICES: frozenset[str] = frozenset(
    {"cpu", "cuda", *(f"cuda:{i}" for i in range(8))}
)
_DEFAULT_ENCODE_DEVICE = "cuda"
#: The assemble step's device, declared, hashed and closed-set validated rather than left for
#: the vendor to pick off the worker: an ``original`` rendering assembled on CPU and an
#: ``omt`` one on GPU would be a per-rendering difference with no cause in the text.
#:
#: The default is CPU because assembly's expensive steps are CPU-bound: Seismic's
#: ``build_from_dataset`` touches no GPU, PLAID's ``add_documents`` is host-RAM-bound rather
#: than VRAM-bound, and FAISS takes seconds. Assembly wants cores and RAM rather than a card,
#: which is what makes the vectorize/assemble split a separation by resource class.
_DEFAULT_ASSEMBLE_DEVICE = "cpu"

#: The device PLAID searches on. Separate from :data:`_DEFAULT_ASSEMBLE_DEVICE` because building
#: and searching are different acts on different sides of the artifact: assembly writes bytes (so
#: its device is hashed into ``index_hash``), search only reads them (so its device is a
#: ``QUERY_TIME_KEYS`` member and re-keys nothing).
#:
#: Late-interaction dominates query cost when it runs on the CPU: 10.103 s against 0.390 s
#: sparse and 0.085 s dense per (leg, cell), i.e. 88-96 % of the total, and a whole rendering
#: is 78 parts.
#:
#: CUDA is not a free speed-up but a re-keyed change of the query stack. On identical parts,
#: CPU against CUDA gives 8/20 byte-identical results, 14/20 the same rank order and 19/20 the
#: same id set, with deltas quantized to one fp16 ULP in the MaxSim reduction and the earliest
#: disagreement at rank 42. The loop consumes top-10, which the score margin protects, but a
#: tail measured on the CPU path is not comparable to a CUDA one. CUDA is byte-identical to
#: itself run to run.
#:
#: ``low_memory=True`` is the only viable CUDA form: ``False`` needs 100-105 GiB of VRAM per
#: rendering, while ``True`` is around 12 GiB, byte-identical to the full-VRAM path, and still
#: roughly 10x CPU. That fits alongside a served model on the same node.
#:
#: Defaults to CPU so adopting CUDA is a deliberate, recorded config change rather than
#: something a scheduler decides by handing a worker a GPU.
_DEFAULT_QUERY_PLAID_DEVICE = "cpu"

#: How a leg's vectors are stored between vectorize and assemble. Closed sets, because the
#: stored bytes are the artifact the split publishes: an fp16 store is a genuinely different
#: vector set, and an assemble reading it is reading different input. The defaults here are
#: the lossless forms the encoders return today: float32 on the host for the MTD
#: encoder (512 B/vector), float32 from ``Encoder.embed``, and Seismic's own
#: ``(U30 components, f32 values)`` pair from :func:`serving.sparse_milco.seismic_components`
#: so a run states what it stores, as with every other recipe leaf.
#:
#: fp16 for the late-interaction leg is viable and the shipped configs take it: the assembled
#: index is byte-identical to the fp32-stored one apart from vendor ``mtime`` fields, for
#: 1.51 TB of intermediate storage instead of 3.03 TB. It does not halve the worker's peak,
#: because fast-plaid materializes an fp32 tensor whatever the input dtype, so the worst
#: English part falls by around a quarter (see :data:`PLAID_ADD_PEAK_MULTIPLIER`). Sparse
#: storage is the larger surprise at roughly 1.30 TB corpus-wide, comparable to
#: late-interaction's 1.51 TB.
_STORE_DTYPES: frozenset[str] = frozenset({"float32", "float16"})
_DEFAULT_STORE_DTYPE = "float32"
#: Seismic's stored sparse form. Named as a format rather than a dtype because a sparse
#: vector is a pair (components + values), not an array: the component side is Seismic's
#: ``U30`` string form and only the value side has a width to choose.
_SPARSE_STORE_FORMATS: frozenset[str] = frozenset({"seismic_u30_f32", "seismic_u30_f16"})
_DEFAULT_SPARSE_STORE_FORMAT = "seismic_u30_f32"

#: Renderings whose text is a join over per-sentence translations (vs the native slice).
_TRANSLATED = tuple(name for name in RENDERINGS if name != "original")


class IndexIntegrityError(RuntimeError):
    """An index would be published with an id set that is not the corpus's.

    Never recovered from: a passage silently missing from one leg (an OOM'd batch, a
    truncation edge case) makes that rendering's index a different index from its siblings,
    and every cross-rendering comparison downstream would then be measuring the build bug
    instead of translation quality. Raising aborts publication, nothing partial is written.
    """


def idmap_arrow_schema() -> Any:
    """Pinned Arrow schema of a leg's ``idmap.parquet``: the column order is the contract.

    Four columns and no fifth: ``ordinal`` (the engine's own document key, the
    row's position in this shard's pinned batch order), ``passage_id`` (what a search
    returns), ``document_id`` (the always-original citation key, carried not re-derived) and
    ``source_lang`` (so a shard is self-describing). A text column here would be the one
    way the search/display decoupling could break; its absence is the guarantee.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("ordinal", pa.int64()),
            pa.field("passage_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("source_lang", pa.string()),
        ]
    )


@dataclass(frozen=True, slots=True)
class IndexBuildOptions:
    """The semantic build recipe: everything that changes an index byte.

    Read from the hashed, fairness-shared ``index_build.config`` block (never from
    ``execution``, whose knobs are throughput-only). The rendering is not a
    field: it is what varies, and it must not be able to reach a leg's ``config_hash``.
    """

    # Every field here reaches a call site. A knob in a recipe hash advertises a pin; one that
    # reaches no call is a false pin, and a false pin is worse than an absent one, so a new
    # field lands together with the code that reads it.
    dense_model: str = _DEFAULT_DENSE_MODEL
    dense_revision: str = ""
    #: Recorded provenance for the two legs whose engine had no field at all, added so that
    #: Every key filed in :data:`ENCODE_KEYS`/:data:`ASSEMBLE_KEYS` can be read off this
    #: object by name (which is how the two hashes build their bodies). Validated against
    #: :data:`ENGINE_OF` exactly as ``spine_engine`` is: the engine is hard-wired in code, so
    #: a config naming another one fails at load rather than being ignored.
    dense_engine: str = ""
    sparse_engine: str = ""
    ann_factory: str = _DEFAULT_ANN_FACTORY
    nprobe: int = _DEFAULT_NPROBE
    quantizer_train_seed: int = _DEFAULT_TRAIN_SEED
    quantizer_train_sample: int = _DEFAULT_TRAIN_SAMPLE
    omp_threads: int = _DEFAULT_OMP_THREADS
    #: Only warranted for the sparse + late-interaction legs: the dense leg was measured
    #: bitwise-identical across three architectures, so it needs no constraint at all.
    gpu_arch_pin: str = ""
    sparse_model: str = ""
    sparse_model_revision: str = ""
    #: The sparse leg's encode window (``index_build.config.sparse_max_length``). It decides
    #: which tokens of a passage MILCO ever sees, so it belongs in the leg's recipe hash for
    #: the same reason ``document_length`` belongs in the late-interaction leg's. A 512-token
    #: window against MILCO's own 8192 ceiling truncated 2.36 % of Chinese passage-renderings
    #: and 0.024 % of Spanish ones, a per-language asymmetry produced by nothing but a
    #: default. Never 0-means-the-client-decides: an encode window this stage records must be
    #: a number, and the fallback is the client's own imported constant.
    sparse_max_length: int = _DEFAULT_SPARSE_MAX_LENGTH
    spine_model: str = ""
    spine_model_revision: str = ""
    #: Recorded provenance, and validated against :data:`ENGINE_OF`: a config naming another
    #: engine fails at load instead of being ignored (the engine is hard-wired in code).
    spine_engine: str = _SPINE_ENGINE
    plaid_nbits: int = _DEFAULT_PLAID_NBITS
    #: How many passages one late-interaction part holds
    #: (``index_build.config.plaid_part_passages``). Not a throughput knob and therefore not
    #: in ``execution``: each part is clustered and residual-coded independently, so moving it
    #: changes every stored byte of this leg, which is why it is hashed
    #: (:func:`leg_config_hash`) exactly as ``plaid_nbits`` and ``document_length`` are.
    plaid_part_passages: int = _DEFAULT_PLAID_PART_PASSAGES
    #: The late-interaction encode window (``index_build.config.document_length``). 0 => the
    #: checkpoint's own metadata decides, which is what ``serving.late_interaction`` does with
    #: it; it is carried here because it belongs in the leg's recipe hash and provenance.
    document_length: int = 0
    #: Encode composition: semantic and fairness-shared (see the constants above), so it is
    #: read from ``index_build.config``, never from the non-shared ``execution`` block.
    encode_batch_token_budget: int = _DEFAULT_BATCH_TOKENS
    encode_max_items: int = _DEFAULT_BATCH_ITEMS
    #: The VECTORIZE unit (see :data:`_DEFAULT_ENCODE_BLOCK_PASSAGES`): how many passages of
    #: the pinned order one encode block covers. In :data:`ENCODE_KEYS` for every leg: it
    #: decides batch membership, hence the vectors themselves.
    encode_block_passages: int = _DEFAULT_ENCODE_BLOCK_PASSAGES
    #: The device the encode runs on: a first-class recipe leaf over the closed set
    #: :data:`_DEVICES`, never ``"auto"``, so the recipe records the numerics that produced
    #: the vectors.
    encode_device: str = _DEFAULT_ENCODE_DEVICE
    #: The device a query-side dense embedding runs on: a :data:`QUERY_TIME_KEYS` member,
    #: so it is excluded from ``index_hash`` and flipping it never orphans a built index.
    #: Resolved by :func:`index_build_options` to ``encode_device`` when the leaf is absent,
    #: so an unset knob is byte-identical to the behaviour that shipped before it existed,
    #: and never ``"auto"``, for :data:`_DEVICES`' reason (one recorded fact, two numerics).
    #: Read by ``serving.registry`` (the ``query_dense`` client), never by this module: no
    #: step here embeds a query.
    query_encode_device: str = _DEFAULT_ENCODE_DEVICE
    #: The device the assemble runs on (:data:`_DEFAULT_ASSEMBLE_DEVICE`): in
    #: :data:`ASSEMBLE_KEYS` for every leg, never in the encode set: it cannot move a vector,
    #: only the index built over one.
    assemble_device: str = _DEFAULT_ASSEMBLE_DEVICE
    #: The device PLAID searches on (:data:`_DEFAULT_QUERY_PLAID_DEVICE`): a
    #: :data:`QUERY_TIME_KEYS` member, so it is excluded from ``index_hash`` and flipping it
    #: never orphans a built index. Reads bytes, writes none.
    query_plaid_device: str = _DEFAULT_QUERY_PLAID_DEVICE
    #: Per-leg store forms: the bytes vectorize publishes and assemble consumes.
    dense_store_dtype: str = _DEFAULT_STORE_DTYPE
    late_interaction_store_dtype: str = _DEFAULT_STORE_DTYPE
    sparse_store_format: str = _DEFAULT_SPARSE_STORE_FORMAT
    #: ``pack12``: Not an ``index_build.config`` key, and carried here anyway:
    #: the passages are as much an input to a vector as the encoder is (a different grouping
    #: of the same sentences is a different text, a different id and a different ordinal), so
    #: :func:`leg_encode_hash` must fold it in. Resolved by :func:`index_build_options` from
    #: the separate ``packing`` block, exactly as :func:`index_hash` folds that block in one
    #: level up. It is excluded from the key partition for the same reason: the partition is
    #: over ``index_build.config``, and this is not one of its keys.
    packing_hash: str = ""
    seismic_min_cluster_size: int = _DEFAULT_SEISMIC_MIN_CLUSTER
    seismic_centroid_fraction: float = _DEFAULT_SEISMIC_CENTROID_FRACTION
    seismic_query_cut: int = _DEFAULT_SEISMIC_QUERY_CUT
    seismic_heap_factor: float = _DEFAULT_SEISMIC_HEAP_FACTOR
    #: The assemble-time top-k cut on a passage's sparse components (see
    #: :data:`_DEFAULT_SEISMIC_TOP_K`). ``0`` keeps every component
    #: (:data:`~ragtime.serving.sparse_milco.NO_TOP_K`) and is a legal, meaningful value,
    #: it is what the vectors on disk hold and what the query side uses.
    seismic_top_k: int = _DEFAULT_SEISMIC_TOP_K
    #: The four vendor build parameters, declared so the config is the complete record
    #: (see :data:`_DEFAULT_SEISMIC_N_POSTINGS`). Applied: every one of them is passed to
    #: ``SeismicIndex.build_from_dataset`` by :meth:`_SeismicWriter.finish`, so none is a
    #: false pin.
    seismic_n_postings: int = _DEFAULT_SEISMIC_N_POSTINGS
    seismic_summary_energy: float = _DEFAULT_SEISMIC_SUMMARY_ENERGY
    seismic_max_fraction: float = _DEFAULT_SEISMIC_MAX_FRACTION
    seismic_doc_cut: int = _DEFAULT_SEISMIC_DOC_CUT
    #: Tokenizer identity is recorded per leg: the three legs use three different
    #: vocabularies (XLM-R 250002 for dense, the English SPLADE pivot 30522 for sparse,
    #: XLM-R again but a separate model instance for late interaction). One shared
    #: ``tokenizer_id`` leaf would misstate that.
    dense_tokenizer_id: str = ""
    sparse_pivot_tokenizer_id: str = ""
    late_interaction_tokenizer_id: str = ""

    @property
    def exact(self) -> bool:
        """True for the shipped default: exact flat search, so nothing is trained."""
        return self.ann_factory.strip().lower() in {"flat", "indexflatip"}

    @property
    def train_seed(self) -> int:
        """The one declared training seed, for every leg that trains anything.

        The config key is ``index_build.config.quantizer_train_seed`` (declared in all six
        configs, folded into ``index_hash``); this property is the name for what it means
        now that more than the dense quantizer reads it, FAISS's clustering parameters
        (:func:`_apply_faiss_train_seed`) and PLAID's k-means (:func:`_open_plaid`). One
        declared seed, not three, because a second seed knob is a second thing that can be
        set differently for one rendering than for another. The sparse leg cannot take it at
        all (:data:`UNSEEDABLE_LEGS`).
        """
        return self.quantizer_train_seed


# --------------------------------------------------------------------------- #
# The encode / assemble partition of the recipe.
#
# The build is two halves, vectorize (encode passages into stored vectors) and assemble
# (turn stored vectors into a searchable index), and the point of the split is that an
# assemble-only recipe change reuses the vectors. That reuse is only safe if the two key
# sets are a true partition of the recipe: were "encode-relevant" an allowlist, a newly
# added hashed key would default to "not encode-relevant" and the build would reuse stale
# vectors under a moved recipe hash.
#
# So the sets below are declared and the universe they must cover is derived from
# ``config.schema.INDEX_BUILD_CONFIG`` (:data:`ALL_LEG_KEYS`); a key added to the schema and
# filed in neither set fails the partition test. Concretely, raising ``sparse_max_length``
# then re-encodes only the sparse leg and changing ``plaid_nbits`` re-encodes nothing.
# --------------------------------------------------------------------------- #
#: ``index_build.config`` keys that name a structural fact rather than a recipe input: they
#: reach no leg hash at all, because the thing they describe is not a run choice. Each is
#: value-checked by :func:`_assert_structural_keys` instead (asserted by a test that reads
#: that function's source), which is what keeps this from being an escape hatch.
STRUCTURAL_KEYS: frozenset[str] = frozenset({"variants", "shard_by", "english_once"})

#: ``index_build.config`` keys that are query-time only: they change no stored byte in any
#: leg, so they belong to neither half of the build. ``query_length`` is the late-interaction
#: query encode window (``serving.registry._mtd_client``), never the document one; the
#: document window is ``document_length``, which is an encode key.
#:
#: ``query_plaid_device`` decides the device PLAID searches on, and searching moves no stored
#: byte, so it is excluded from ``index_hash`` and flipping it does not orphan a built index.
#: It is nonetheless not an ``execution`` key, since that block's rule is that a key changes
#: no returned hit and this one moves result lists (see
#: :data:`_DEFAULT_QUERY_PLAID_DEVICE`).
#:
#: ``query_encode_device`` follows the identical argument one level over: it decides the
#: device the dense encoder embeds a query-side string on, and embedding a query stores
#: nothing. It is excluded from ``index_hash`` and still outside ``execution``, because a
#: device moves a forward pass and a cosine near the dedup cutoff can merge on one device and
#: not the other.
QUERY_TIME_KEYS: frozenset[str] = frozenset(
    {"query_length", "query_plaid_device", "query_encode_device"}
)

#: ``leg -> the name markers that make a config key that leg's``. The one place key
#: ownership is expressed; :data:`ALL_LEG_KEYS` is derived from it and from the schema, so
#: the partition test's expectation is never a restatement of the declared sets it checks.
_LEG_KEY_MARKERS: dict[str, tuple[str, ...]] = {
    DENSE_LEG: ("dense_", "ann_", "nprobe", "quantizer_"),
    SPARSE_LEG: ("sparse_", "seismic_"),
    LATE_INTERACTION_LEG: ("spine_", "plaid_", "late_interaction_", "document_length"),
}


def leg_of_config_key(key: str) -> str | None:
    """Which leg owns ``key``: or ``None`` for a key every leg shares.

    Ownership is by name (:data:`_LEG_KEY_MARKERS`), which is the convention the block has
    always followed, so a new ``sparse_*`` key is the sparse leg's without anyone saying so.
    A key claimed by two legs raises rather than being assigned to whichever matched first:
    an ambiguous name would make the partition below depend on dict order.
    """
    owners = [leg for leg, markers in _LEG_KEY_MARKERS.items() if key.startswith(markers)]
    if len(owners) > 1:
        raise ValueError(
            f"index_build.config key {key!r} matches the markers of {owners}, key ownership "
            "must be unambiguous, or the encode/assemble partition depends on dict order"
        )
    return owners[0] if owners else None


#: The recipe's key universe, read off the live schema (``config.schema``) rather than
#: restated here: so a key added there and filed in neither half below cannot hide.
_INDEX_BUILD_CONFIG_KEYS: frozenset[str] = frozenset(INDEX_BUILD_CONFIG)

def leg_recipe_keys(universe: Iterable[str] | None = None) -> dict[str, frozenset[str]]:
    """``leg -> every recipe key that is part of that leg``, derived from the schema.

    ``universe`` defaults to the live ``INDEX_BUILD_CONFIG`` and exists so a test can ask what
    would happen if a new key were declared, instead of waiting for someone to declare one.
    A key is a leg's if its name says so (:func:`leg_of_config_key`) and every leg's if it
    does not; the structural and query-time keys are removed because they are part of no
    leg's recipe at all.
    """
    keys = _INDEX_BUILD_CONFIG_KEYS if universe is None else frozenset(universe)
    return {
        leg: frozenset(
            key
            for key in keys
            if key not in STRUCTURAL_KEYS
            and key not in QUERY_TIME_KEYS
            and leg_of_config_key(key) in (None, leg)
        )
        for leg in LEGS
    }


#: ``leg -> every ``index_build.config`` key that is part of that leg's recipe``. Derived
#: from the live schema, never restated: a key added to ``INDEX_BUILD_CONFIG`` lands here
#: automatically (in its own leg's set if its name says so, in all three if it does not),
#: and then fails the partition test until it is filed as encode or assemble.
ALL_LEG_KEYS: dict[str, frozenset[str]] = leg_recipe_keys()

#: Keys that decide the vectors, shared by all three legs. ``encode_block_passages`` and the
#: two ``Batcher`` bounds decide batch membership (measured: the same shard batched
#: differently is 60.97 % bitwise-identical), ``encode_device`` decides the forward-pass
#: numerics. None of them can be an ``execution`` knob for that reason.
_SHARED_ENCODE_KEYS: frozenset[str] = frozenset(
    {
        "encode_block_passages",
        "encode_batch_token_budget",
        "encode_max_items",
        "encode_device",
    }
)
#: Keys that decide only the assembled index, shared by all three legs. ``assemble_device``
#: is here because it changes the assembled bytes (see :data:`_DEFAULT_ASSEMBLE_DEVICE`) and
#: cannot change a vector.
_SHARED_ASSEMBLE_KEYS: frozenset[str] = frozenset({"assemble_device"})

#: ``leg -> the keys whose value decides that leg's vectors``. Read by
#: :func:`leg_encode_hash`, so this is the hash input, not a parallel description of it.
ENCODE_KEYS: dict[str, frozenset[str]] = {
    DENSE_LEG: _SHARED_ENCODE_KEYS
    | {"dense_model", "dense_revision", "dense_tokenizer_id", "dense_store_dtype"},
    SPARSE_LEG: _SHARED_ENCODE_KEYS
    | {
        "sparse_model",
        "sparse_model_revision",
        "sparse_pivot_tokenizer_id",
        # The encode window: which of a passage's tokens the model ever sees. The one key
        # whose re-encode this split is most meant to bound: raising it costs the sparse
        # leg's 13 GPU-h, not the whole build's 82.
        "sparse_max_length",
        "sparse_store_format",
    },
    LATE_INTERACTION_LEG: _SHARED_ENCODE_KEYS
    | {
        "spine_model",
        "spine_model_revision",
        "late_interaction_tokenizer_id",
        "document_length",
        "late_interaction_store_dtype",
    },
}

#: ``leg -> the keys that decide only how those vectors are assembled``. The complement, and
#: declared rather than computed as one: a key must be filed explicitly on one side, so
#: that "everything I did not think about is assemble-only": the stale-vector failure, is
#: not expressible.
ASSEMBLE_KEYS: dict[str, frozenset[str]] = {
    DENSE_LEG: _SHARED_ASSEMBLE_KEYS
    | {
        "dense_engine",
        "ann_factory",
        "nprobe",
        "quantizer_train_seed",
        "quantizer_train_sample",
    },
    SPARSE_LEG: _SHARED_ASSEMBLE_KEYS
    | {
        "sparse_engine",
        "seismic_min_cluster_size",
        "seismic_centroid_fraction",
        # The top-k cut belongs here and not in the encode set, which is the reason it
        # is affordable: it is applied by ``_SeismicWriter`` to vectors that were already
        # stored unpruned, so changing it re-assembles on CPU and re-encodes nothing. Filing
        # it as an encode key would move ``leg_encode_hash`` and invalidate 87.7 GB of stored
        # sparse vectors: 13 GPU-h, for a change that touches no forward pass.
        "seismic_top_k",
        # The four vendor build parameters, declared and passed (see IndexBuildOptions).
        # Assemble by the same argument as ``min_cluster_size``: they shape the built index,
        # never a vector.
        "seismic_n_postings",
        "seismic_summary_energy",
        "seismic_max_fraction",
        "seismic_doc_cut",
        # The two search parameters are filed as assemble, not as query-time:
        # over-including here costs a re-assemble (cheap, no GPU encode), while
        # under-including would be the failure this partition exists to prevent.
        "seismic_query_cut",
        "seismic_heap_factor",
    },
    LATE_INTERACTION_LEG: _SHARED_ASSEMBLE_KEYS
    | {"spine_engine", "plaid_nbits", "plaid_part_passages"},
}


def _icfg(cfg: RunConfig) -> Mapping[str, Any]:
    """The ``index_build`` block's ``config`` sub-mapping (one level down, like
    ``translation``)."""
    return dict(cfg.blocks.get("index_build", {}).get("config", {}) or {})


def _shard_by_column() -> str:
    """The column shards are keyed on, read off :class:`IndexShardSpec`, never re-typed.

    A shard is one ``(variant, source_lang, part)`` unit, so the spec's language field is
    still the sharding column; taking it from the dataclass means renaming that field can
    never leave this validation pointing at a column nobody writes.

    ``part`` does not appear here, and ``shard_by: source_lang`` stays the true
    record: the part axis is a sub-division of the language cell on the pinned ordinal (the
    same relationship ``plaid_part_passages`` has to the leg it cuts), not a second sharding
    column. Its size is declared by its own key, ``plaid_part_passages``, and there is
    exactly one part size.
    """
    return next(f for f in IndexShardSpec.__dataclass_fields__ if f.endswith("lang"))


def _assert_structural_keys(c: Mapping[str, Any]) -> None:
    """Reject an ``index_build.config`` that disagrees with a structural fact of the build.

    ``index_build.config`` is a free-form schema leaf, so nothing upstream rejects a key,
    which makes "declared but not applied" the failure mode to design against. The rule for
    this block: every key is either applied or validated; none is merely accepted. The
    structural keys here name real, correct behaviour that is not a run knob (per-language
    shards, English built once, the three engines), so deleting them would drop true
    information from the record while leaving them unchecked would let ``english_once: false``
    or ``dense_engine: qdrant`` be silently ignored by a file that claims to be the complete
    reproducible record.

    Three families of check live here, and they fail for different reasons:
    Structural facts (validated against :data:`RENDERINGS`/:data:`ENGINE_OF`/
    :class:`IndexShardSpec`), sizes that must be at least one passage (the two part axes and
    the vectorize block), and closed sets (:data:`_DEVICES`, :data:`_STORE_DTYPES`,
    :data:`_SPARSE_STORE_FORMATS`) for the leaves whose value decides numerics or stored
    bytes.

    Every expectation is read from the structural constant it describes (:data:`RENDERINGS`,
    :data:`SHARED_LANG`, :data:`ENGINE_OF`, :class:`IndexShardSpec`), so this function can
    never drift from the code it is asserting about. An absent key is fine: the structure is
    what it is whether or not the file restates it.
    """
    problems: list[str] = []
    if "variants" in c:
        declared = [str(v) for v in c["variants"] or ()]
        # Membership, not order: the build iterates RENDERINGS, so a reordered list describes
        # the same build: a missing or extra rendering does not.
        if set(declared) != set(RENDERINGS):
            problems.append(
                f"variants={declared}: the build always covers {list(RENDERINGS)} "
                "(common.passage_store.RENDERINGS), the set of renderings is not a run knob"
            )
    if "shard_by" in c and str(c["shard_by"]) != _shard_by_column():
        problems.append(
            f"shard_by={c['shard_by']!r}: shards are always "
            f"variants x {_shard_by_column()} (IndexShardSpec), never another column"
        )
    if "english_once" in c and not bool(c["english_once"]):
        problems.append(
            f"english_once={c['english_once']!r}: {SHARED_LANG!r} is identity pass-through in "
            "both MT arms, so its shard is built once for all renderings (shard_specs), "
            "structural, not a run knob"
        )
    if "plaid_part_passages" in c and int(c["plaid_part_passages"] or 0) < 1:
        problems.append(
            f"plaid_part_passages={c['plaid_part_passages']!r}: a late-interaction part holds "
            "at least one passage, the key bounds the single add_documents buffer "
            "(_PlaidWriter), it does not switch part-sharding off"
        )
    if "seismic_top_k" in c and int(c["seismic_top_k"] or 0) < 0:
        problems.append(
            f"seismic_top_k={c['seismic_top_k']!r}: the sparse cut keeps the k heaviest "
            "components of a passage, so k is a count, 0 is the legal, meaningful 'keep "
            "them all' (serving.sparse_milco.NO_TOP_K), and a negative k describes no build"
        )
    if "encode_block_passages" in c and int(c["encode_block_passages"] or 0) < 1:
        problems.append(
            f"encode_block_passages={c['encode_block_passages']!r}: a vectorize block holds at "
            "least one passage, the key is the encode unit (and therefore the batch "
            "composition), not a switch that turns block-vectorizing off"
        )
    # The two device leaves and the three store forms: closed sets, and closed for the same
    # reason. A device decides numerics (measured: PLAID's k-means is identical 3/3 on CPU
    # and 3-distinct on CUDA), a store form decides the stored bytes: so a
    # value outside the set describes an artifact this code cannot produce, and "auto"
    # describes two artifacts under one hash.
    _CLOSED_LEAVES: tuple[tuple[str, frozenset[str], str], ...] = (
        (
            "encode_device",
            _DEVICES,
            (
                "device decides the forward-pass numerics, so an 'auto' that resolves to CPU "
                "on one worker and CUDA on another would hash identically while producing "
                "different vectors"
            ),
        ),
        (
            "assemble_device",
            _DEVICES,
            (
                "the assembled bytes depend on it (fast-plaid's compute_kmeans is identical "
                "3/3 on cpu and 3-distinct on cuda), so CPU assembly is a "
                "deliberate, re-keyed alternative, never an opportunistic fallback"
            ),
        ),
        (
            "query_plaid_device",
            _QUERY_DEVICES,
            (
                "CPU and CUDA MaxSim disagree from rank 42 (8/20 byte-identical, 19/20 same id "
                "set), so the query device changes what is returned and must be a "
                "written-down choice, an 'auto' resolving per worker would make one run's tail "
                "depend on which node the scheduler happened to give it"
            ),
        ),
        (
            "query_encode_device",
            _QUERY_DEVICES,
            (
                "a device moves the forward pass, so a query embedding taken on CPU and one "
                "taken on CUDA are not the same vector, decompose's dedup pre-filter compares "
                "them against a fixed cosine cutoff, and a pair either side of it merges on one "
                "device and not the other. An 'auto' resolving per worker would make a nugget "
                "bank depend on which node the scheduler happened to give it"
            ),
        ),
        ("dense_store_dtype", _STORE_DTYPES, "an fp16 store is a different vector set"),
        (
            "late_interaction_store_dtype",
            _STORE_DTYPES,
            (
                "an fp16 store is a different vector set (the MTD encoder returns float32 on "
                "the host, 512 B/vector measured)"
            ),
        ),
        (
            "sparse_store_format",
            _SPARSE_STORE_FORMATS,
            (
                "Seismic keys on U30 component strings, so only the value side has a width "
                "to choose (serving.sparse_milco.seismic_components)"
            ),
        ),
    )
    for key, allowed, why in _CLOSED_LEAVES:
        if key in c and str(c[key]) not in allowed:
            problems.append(f"{key}={c[key]!r}: expected one of {sorted(allowed)}, {why}")
    for key, leg in (
        ("dense_engine", DENSE_LEG),
        ("sparse_engine", SPARSE_LEG),
        ("spine_engine", LATE_INTERACTION_LEG),
    ):
        if key in c and str(c[key]) != ENGINE_OF[leg]:
            problems.append(
                f"{key}={c[key]!r}: the {leg} leg is built with {ENGINE_OF[leg]!r} "
                "(ENGINE_OF), a different engine is a code change, not a config edit"
            )
    if problems:
        raise ConfigError(
            "index_build.config disagrees with the build's structure: "
            + "; ".join(problems)
            + ". These keys are validated rather than applied, the config is the complete "
            "record, so a value the code cannot honour must fail at load, not be ignored."
        )


def index_build_options(cfg: RunConfig) -> IndexBuildOptions:
    """Resolve :class:`IndexBuildOptions` from the hashed ``index_build`` block.

    The rule for this block, enforced here: every key is either applied (it reaches a call
    that changes an index byte, or it is the provenance this stage records) or validated
    against the structural constant it names (:func:`_assert_structural_keys`); none is
    merely accepted. A key that reaches nothing is a false pin, it moves ``index_hash``
    and ``leg_config_hash`` while changing no byte, which is a record that lies in the one
    direction that matters.
    """
    c = _icfg(cfg)
    _assert_structural_keys(c)
    return IndexBuildOptions(
        dense_model=str(c.get("dense_model", _DEFAULT_DENSE_MODEL) or _DEFAULT_DENSE_MODEL),
        dense_revision=str(c.get("dense_revision", "") or ""),
        dense_engine=str(c.get("dense_engine", ENGINE_OF[DENSE_LEG]) or ENGINE_OF[DENSE_LEG]),
        sparse_engine=str(
            c.get("sparse_engine", ENGINE_OF[SPARSE_LEG]) or ENGINE_OF[SPARSE_LEG]
        ),
        ann_factory=str(c.get("ann_factory", _DEFAULT_ANN_FACTORY) or _DEFAULT_ANN_FACTORY),
        nprobe=int(c.get("nprobe", _DEFAULT_NPROBE) or 0),
        quantizer_train_seed=int(c.get("quantizer_train_seed", _DEFAULT_TRAIN_SEED) or 0),
        quantizer_train_sample=int(c.get("quantizer_train_sample", _DEFAULT_TRAIN_SAMPLE) or 0),
        omp_threads=int(c.get("omp_threads", _DEFAULT_OMP_THREADS) or 0),
        gpu_arch_pin=str(c.get("gpu_arch_pin", "") or ""),
        sparse_model=str(c.get("sparse_model", "") or ""),
        sparse_model_revision=str(c.get("sparse_model_revision", "") or ""),
        sparse_max_length=int(
            c.get("sparse_max_length", _DEFAULT_SPARSE_MAX_LENGTH) or _DEFAULT_SPARSE_MAX_LENGTH
        ),
        spine_model=str(c.get("spine_model", "") or ""),
        spine_model_revision=str(c.get("spine_model_revision", "") or ""),
        spine_engine=str(c.get("spine_engine", _SPINE_ENGINE) or _SPINE_ENGINE),
        plaid_nbits=int(c.get("plaid_nbits", _DEFAULT_PLAID_NBITS) or _DEFAULT_PLAID_NBITS),
        plaid_part_passages=int(
            c.get("plaid_part_passages", _DEFAULT_PLAID_PART_PASSAGES)
            or _DEFAULT_PLAID_PART_PASSAGES
        ),
        document_length=int(c.get("document_length", 0) or 0),
        encode_batch_token_budget=int(
            c.get("encode_batch_token_budget", _DEFAULT_BATCH_TOKENS) or _DEFAULT_BATCH_TOKENS
        ),
        encode_max_items=int(c.get("encode_max_items", _DEFAULT_BATCH_ITEMS) or _DEFAULT_BATCH_ITEMS),
        encode_block_passages=int(
            c.get("encode_block_passages", _DEFAULT_ENCODE_BLOCK_PASSAGES)
            or _DEFAULT_ENCODE_BLOCK_PASSAGES
        ),
        encode_device=str(c.get("encode_device", _DEFAULT_ENCODE_DEVICE) or _DEFAULT_ENCODE_DEVICE),
        # Inherits `encode_device` when absent, rather than defaulting to a constant: the two
        # are the same encoder, and a config that moved the build to CPU must not
        # silently keep embedding queries on a card the job may not even hold. An unset leaf is
        # therefore exactly the pre-behaviour, on every config.
        query_encode_device=str(
            c.get("query_encode_device", "")
            or c.get("encode_device", _DEFAULT_ENCODE_DEVICE)
            or _DEFAULT_ENCODE_DEVICE
        ),
        assemble_device=str(
            c.get("assemble_device", _DEFAULT_ASSEMBLE_DEVICE) or _DEFAULT_ASSEMBLE_DEVICE
        ),
        query_plaid_device=str(
            c.get("query_plaid_device", _DEFAULT_QUERY_PLAID_DEVICE)
            or _DEFAULT_QUERY_PLAID_DEVICE
        ),
        dense_store_dtype=str(
            c.get("dense_store_dtype", _DEFAULT_STORE_DTYPE) or _DEFAULT_STORE_DTYPE
        ),
        late_interaction_store_dtype=str(
            c.get("late_interaction_store_dtype", _DEFAULT_STORE_DTYPE) or _DEFAULT_STORE_DTYPE
        ),
        sparse_store_format=str(
            c.get("sparse_store_format", _DEFAULT_SPARSE_STORE_FORMAT)
            or _DEFAULT_SPARSE_STORE_FORMAT
        ),
        # Not an ``index_build.config`` key: ``pack12``, resolved from the separate
        # ``packing`` block so :func:`leg_encode_hash` can fold in the passages themselves.
        packing_hash=packing_hash(cfg),
        seismic_min_cluster_size=int(
            c.get("seismic_min_cluster_size", _DEFAULT_SEISMIC_MIN_CLUSTER)
            or _DEFAULT_SEISMIC_MIN_CLUSTER
        ),
        seismic_centroid_fraction=float(
            c.get("seismic_centroid_fraction", _DEFAULT_SEISMIC_CENTROID_FRACTION)
            or _DEFAULT_SEISMIC_CENTROID_FRACTION
        ),
        seismic_query_cut=int(
            c.get("seismic_query_cut", _DEFAULT_SEISMIC_QUERY_CUT) or _DEFAULT_SEISMIC_QUERY_CUT
        ),
        seismic_heap_factor=float(
            c.get("seismic_heap_factor", _DEFAULT_SEISMIC_HEAP_FACTOR)
            or _DEFAULT_SEISMIC_HEAP_FACTOR
        ),
        # Not the ``x or default`` idiom every leaf above uses, and the exception is
        # load-bearing: ``0`` is a meaningful value here (keep every component), so ``or``
        # would silently turn a declared "no cut" into the default 300-component cut: a
        # config that says one thing while the build does another, which is the whole class
        # of defect this block's rule exists to prevent.
        seismic_top_k=int(
            _DEFAULT_SEISMIC_TOP_K
            if c.get("seismic_top_k") is None
            else c["seismic_top_k"]
        ),
        seismic_n_postings=int(
            c.get("seismic_n_postings", _DEFAULT_SEISMIC_N_POSTINGS)
            or _DEFAULT_SEISMIC_N_POSTINGS
        ),
        seismic_summary_energy=float(
            c.get("seismic_summary_energy", _DEFAULT_SEISMIC_SUMMARY_ENERGY)
            or _DEFAULT_SEISMIC_SUMMARY_ENERGY
        ),
        seismic_max_fraction=float(
            c.get("seismic_max_fraction", _DEFAULT_SEISMIC_MAX_FRACTION)
            or _DEFAULT_SEISMIC_MAX_FRACTION
        ),
        seismic_doc_cut=int(
            c.get("seismic_doc_cut", _DEFAULT_SEISMIC_DOC_CUT) or _DEFAULT_SEISMIC_DOC_CUT
        ),
        dense_tokenizer_id=str(c.get("dense_tokenizer_id", "") or ""),
        sparse_pivot_tokenizer_id=str(c.get("sparse_pivot_tokenizer_id", "") or ""),
        late_interaction_tokenizer_id=str(c.get("late_interaction_tokenizer_id", "") or ""),
    )


def index_hash(cfg: RunConfig) -> str:
    """The build key: ``config_hash`` of the ``index_build`` block and the ``packing`` one.

    The one canonical hasher (``config.config_hash``), never a second sha256 call site.

    ``packing`` is in here because the passages are as much an input to this index as the
    encoder is. ``recon12``, the level this node lives under, pins which sentences exist,
    but since packing was split out two different groupings of those sentences into passages legitimately
    coexist beneath it (``passages/<pack12>/``), and an index built from one of them has
    different vectors, a different id set and different ordinals than an index built from the
    other. Keying this level by the recipe alone would let the second silently reuse the
    first's ``_SUCCESS``, the same stale-artifact hazard ``recon12`` exists to make
    unaddressable, one level down. So it is folded in rather than merely recorded.
    """
    return config_hash(
        {
            "index_build": _without_query_time_keys(cfg.blocks.get("index_build", {})),
            "packing": cfg.blocks.get("packing", {}),
        }
    )


def _without_query_time_keys(index_build: Mapping[str, Any]) -> dict[str, Any]:
    """``index_build`` with every :data:`QUERY_TIME_KEYS` leaf removed, for hashing only.

    ``index_hash`` keys the build: the tree at ``Layout.index_dir`` and the manifest field
    ``assemble`` writes. A key deciding how a built index is read has no business in it,
    because reading writes nothing.

    This changes ``index_hash`` once, because ``query_length`` is a query-time key that was
    being hashed. The built tree is migrated by moving it to the new key, the vectors are not
    touched at all (they live under a disjoint ``leg_encode_hash`` subtree), and ``_SUCCESS``
    markers are siblings inside the moved tree. The same re-key was already executed on this
    corpus once (``seismic_top_k``: ``a6ca11038f77`` -> ``2072b711bfa2``, byte-identical
    ``encode_hashes``), so the shape of the migration is precedented rather than theoretical.
    """
    out = dict(index_build)
    leaf = out.get("config")
    if isinstance(leaf, Mapping):
        out["config"] = {k: v for k, v in leaf.items() if k not in QUERY_TIME_KEYS}
    return out


def leg_encode_hash(opts: IndexBuildOptions, leg: str) -> str:
    """The vector key: a hash over only the inputs that decide this leg's stored vectors.

    This is the point of the vectorize/assemble split, expressed as an artifact key. An
    assemble-only recipe change (``plaid_nbits``, ``ann_factory``, the Seismic parameters,
    ``plaid_part_passages``) leaves this hash unmoved, so the stored vectors are reused and
    only the index is rebuilt; a per-leg encode change moves only that leg's hash. Measured
    against the 82 GPU-h whole-build cost: raising ``sparse_max_length`` now re-encodes the
    sparse leg alone (13 GPU-h) and changing ``plaid_nbits`` re-encodes nothing, where today
    either costs the full 82.

    The body is built from :data:`ENCODE_KEYS`, not typed out beside it. That is what
    makes the declared partition load-bearing rather than decorative: a key filed on the wrong
    side does not merely mis-document the split, it changes what this hash covers, and the
    partition test (``ENCODE_KEYS[leg] | ASSEMBLE_KEYS[leg] == ALL_LEG_KEYS[leg]``, derived
    from the live schema) is what stops a new key from silently defaulting to "not
    encode-relevant" and letting a moved recipe hash reuse stale vectors.

    ``packing_hash`` is folded in for the reason :func:`index_hash` folds the ``packing``
    block in: the passages are an input to a vector. A re-grouping of the same sentences is a
    different text under a different id at a different ordinal, so vectors keyed by the
    encoder alone would be reused across two genuinely different passage sets. ``leg`` is in
    the body so the three legs cannot collide even if their key sets coincided.
    """
    if leg not in ENCODE_KEYS:
        raise ValueError(f"unknown leg {leg!r}; expected one of {LEGS}")
    body = {key: getattr(opts, key) for key in sorted(ENCODE_KEYS[leg])}
    return config_hash({"leg": leg, "packing_hash": opts.packing_hash, **body})


def leg_config_hash(opts: IndexBuildOptions, leg: str) -> str:
    """The per-leg ``config_hash`` recorded in the manifest: embedder + tokenizer + ANN.

    Its inputs are the leg's own recipe only, so it is equal across ``original``/``omt``/
    ``omt_opus`` by construction rather than by convention: there is no rendering-shaped
    input it could pick up. That equality is the machine-checkable form of "same methodology,
    only the text differs".
    """
    if leg == DENSE_LEG:
        body: dict[str, Any] = {
            "model": opts.dense_model,
            "revision": opts.dense_revision,
            "tokenizer_id": opts.dense_tokenizer_id,
            "ann_factory": opts.ann_factory,
            "nprobe": opts.nprobe,
            # Every key here reaches a call: the seed is applied by _apply_faiss_train_seed
            # before train(). Every key in this body reaches a call; none is decoration.
            "quantizer_train_seed": opts.quantizer_train_seed,
        }
    elif leg == SPARSE_LEG:
        body = {
            "model": opts.sparse_model,
            "revision": opts.sparse_model_revision,
            "tokenizer_id": opts.sparse_pivot_tokenizer_id,
            # The encode window is part of this leg's recipe, exactly as document_length is
            # for the late-interaction one: it decides which of a passage's tokens reach the
            # SPLADE projection at all. Two indexes built at 512 and 8192 are different
            # indexes, and a per-leg hash that omitted this would advertise them as the same.
            "max_length": opts.sparse_max_length,
            "min_cluster_size": opts.seismic_min_cluster_size,
            "centroid_fraction": opts.seismic_centroid_fraction,
            # The assemble-time cut: two indexes built from the same stored vectors at
            # k=300 and unpruned hold different postings (measured top-10 overlap 0.615),
            # so a leg identity that omitted it would advertise them as one index. It is
            # absent from ``leg_encode_hash`` for the opposite reason: it
            # moves no vector (see ASSEMBLE_KEYS).
            "top_k": opts.seismic_top_k,
            # Four Seismic build parameters declared here rather than left to the vendor.
            # Every one reaches ``build_from_dataset`` (_SeismicWriter.finish).
            "n_postings": opts.seismic_n_postings,
            "summary_energy": opts.seismic_summary_energy,
            "max_fraction": opts.seismic_max_fraction,
            "doc_cut": opts.seismic_doc_cut,
        }
    elif leg == LATE_INTERACTION_LEG:
        body = {
            "model": opts.spine_model,
            "revision": opts.spine_model_revision,
            "tokenizer_id": opts.late_interaction_tokenizer_id,
            "engine": opts.spine_engine,
            "nbits": opts.plaid_nbits,
            # The part size is a stored-byte input, not a scheduling one: a shard cut into
            # 10 parts is 10 independently k-means'd, independently residual-coded indexes,
            # so the same passages at 262,144/part and at 131,072/part are different stored
            # vectors. Omitting it would advertise two different artifacts as one leg
            # identity, which is the false-pin failure this block is designed against.
            "part_passages": opts.plaid_part_passages,
            # The encode window is part of this leg's recipe: it decides how many token
            # vectors a passage contributes and, at 220, which of its tokens are dropped
            # (measured: 66-79 % of passages exceed it in every language). A
            # per-leg hash that omitted it would understate the recipe: two indexes built at
            # 220 and 512 would advertise the same leg identity.
            "document_length": opts.document_length,
            # This leg trains (k-means over the token vectors), so its RNG seed is part of
            # its recipe exactly as nbits is: two builds under different seeds have
            # different centroids and must not advertise the same leg identity. The sparse
            # leg carries no such key: it has no seed to record
            # (UNSEEDABLE_LEGS), and a hash entry would claim a pin that does not exist.
            "train_seed": opts.train_seed,
        }
    else:  # pragma: no cover - guarded by IndexAdapter.__post_init__
        raise ValueError(f"unknown leg {leg!r}; expected one of {LEGS}")
    return config_hash({"leg": leg, **body})


# --------------------------------------------------------------------------- #
# Rank-agreement diagnostics (the validation-without-qrels battery's metrics).
#
# No qrels, no dev set and no training data exist for this system, so nothing here
# measures retrieval quality. These two functions measure agreement between two ranked
# lists, which is what the liveness/consistency checks need: cross-rendering overlap
# (`omt` vs `omt_opus`, the causally clean comparison) and a degeneracy smoke test that
# no leg returns a near-constant ranking. Thresholds are not defined here: they are read
# off the measured noise floor by the test that applies them.
# --------------------------------------------------------------------------- #
def kendall_tau_b(left: Sequence[str], right: Sequence[str]) -> float:
    """Kendall's tau-b over the items present in both ranked lists (ties handled).

    Returns 1.0 for identical orders, -1.0 for exact reversal, 0.0 when fewer than two
    items are shared (no agreement is measurable, and inventing one would let a degenerate
    leg look fine).
    """
    shared = [item for item in left if item in set(right)]
    if len(shared) < 2:
        return 0.0
    rank_r = {item: i for i, item in enumerate(right)}
    concordant = discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            a, b = rank_r[shared[i]], rank_r[shared[j]]
            if a == b:
                continue
            if a < b:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return 0.0 if total == 0 else (concordant - discordant) / total


def rbo(left: Sequence[str], right: Sequence[str], p: float = 0.9) -> float:
    """Rank-biased overlap (Webber et al.), top-weighted, truncated at the shorter list.

    ``p`` is the persistence: 0.9 weights roughly the first ten ranks. Truncated RBO (no
    extrapolation), because an index shard's returned list is genuinely finite and
    extrapolating would report agreement over ranks nobody looked at.
    """
    depth = min(len(left), len(right))
    if depth == 0:
        return 0.0
    seen_l: set[str] = set()
    seen_r: set[str] = set()
    weighted = 0.0
    norm = 0.0
    for d in range(1, depth + 1):
        seen_l.add(left[d - 1])
        seen_r.add(right[d - 1])
        weight = p ** (d - 1)
        weighted += weight * (len(seen_l & seen_r) / d)
        norm += weight
    return weighted / norm if norm else 0.0


# --------------------------------------------------------------------------- #
# The leg contract. Three implementations, one shared driver.
# --------------------------------------------------------------------------- #
@runtime_checkable
class Leg(Protocol):
    """One index leg: encode -> incrementally write -> reopen -> search.

    Narrow, and id-free on the way out: ``search`` returns
    ``(ordinal, score)`` pairs and the shared code maps ordinals to ``passage_id``s through
    ``idmap.parquet``. A leg therefore never sees, stores or returns a passage's text.
    """

    name: str

    def encode_docs(self, ctx: IndexCtx, texts: Sequence[str]) -> Sequence[Any]:
        """Encode one already-sized bucket of passage texts."""
        ...

    def encode_query(self, ctx: IndexCtx, query: str) -> Any:
        """Encode one query into this leg's own space."""
        ...

    def writer(self, out_dir: Path, ctx: IndexCtx) -> LegWriter:
        """Open an incremental writer over ``out_dir`` (a private temp dir)."""
        ...

    def open(self, leg_dir: Path, ctx: IndexCtx) -> Any:
        """Reopen a published leg directory for search (read-only)."""
        ...

    def search(
        self, reader: Any, ctx: IndexCtx, query_rep: Any, top_k: int
    ) -> list[tuple[int, float]]:
        """``[(ordinal, score)]``, best first: never a passage_id, never text."""
        ...


class LegWriter(Protocol):
    """The incremental half of a leg: add bucket by bucket, then finish once."""

    def add(self, ordinals: Sequence[int], reps: Sequence[Any]) -> None: ...

    def finish(self) -> None: ...


# --- dense: FAISS ----------------------------------------------------------- #
class _FaissWriter:
    """Accumulate dense vectors into a FAISS index, then serialize it.

    Exact ``Flat`` (the default) adds straight through, no training call is ever made, which
    is what makes "the shipped default has no quantizer step" structural. The retained
    compressed factory string buffers its vectors so it can train before adding: that is one
    of the reasons it is not the default and needs a measured justification before use.
    """

    __slots__ = ("_buffer", "_index", "dim", "opts", "out_dir")

    def __init__(self, out_dir: Path, opts: IndexBuildOptions) -> None:
        self.out_dir = out_dir
        self.opts = opts
        self._index: Any = None
        self._buffer: list[Any] = []
        self.dim = 0

    def _vectors(self, reps: Sequence[Any]) -> Any:
        import numpy as np

        arr = np.asarray(reps, dtype="float32")
        if arr.ndim != 2:
            raise ValueError(f"dense encode returned shape {arr.shape}, expected (n, dim)")
        if self.dim == 0:
            self.dim = int(arr.shape[1])
        elif int(arr.shape[1]) != self.dim:
            raise ValueError(f"dense dim changed mid-shard: {arr.shape[1]} != {self.dim}")
        # The encoder normalizes (``normalize_embeddings=True``), and IndexFlatIP's inner
        # product is cosine only on unit vectors: so this asserts rather than re-normalizes:
        # silently fixing it up would hide a mis-wired encoder behind plausible scores.
        norm = float((arr[0] * arr[0]).sum() ** 0.5)
        if abs(norm - 1.0) > 1e-2:
            raise ValueError(
                f"dense vectors are not unit-norm (|v| = {norm:.4f}); IndexFlatIP's inner "
                "product is only cosine on normalized vectors"
            )
        return arr

    def add(self, ordinals: Sequence[int], reps: Sequence[Any]) -> None:
        arr = self._vectors(reps)
        if not self.opts.exact:
            self._buffer.append(arr)
            return
        if self._index is None:
            self._index = _new_faiss_index(self.dim, self.opts)
        if self._index.ntotal != int(ordinals[0]):
            raise IndexIntegrityError(
                f"FAISS internal ids must equal the shard's ordinals: index holds "
                f"{self._index.ntotal} vectors but this bucket starts at ordinal "
                f"{ordinals[0]}, the idmap would address the wrong vectors"
            )
        self._index.add(arr)

    def finish(self) -> None:
        import faiss
        import numpy as np

        if not self.opts.exact:
            if self.opts.omp_threads:
                faiss.omp_set_num_threads(self.opts.omp_threads)
            arr = np.vstack(self._buffer) if self._buffer else np.zeros((0, self.dim), "float32")
            self._index = _new_faiss_index(self.dim or 1, self.opts)
            if arr.shape[0]:
                sample = self.opts.quantizer_train_sample
                _apply_faiss_train_seed(self._index, self.opts)
                self._index.train(arr[:sample] if sample else arr)
                self._index.add(arr)
        if self._index is None:  # an empty shard is a legal artifact, not a failure
            self._index = _new_faiss_index(self.dim or 1, self.opts)
        faiss.write_index(self._index, str(self.out_dir / _DENSE_FILENAME))


def _apply_faiss_train_seed(index: Any, opts: IndexBuildOptions) -> list[str]:
    """Pin every clustering RNG inside ``index`` to the config's seed, before ``train``.

    FAISS seeds its k-means through ``ClusteringParameters.seed``, which lives on a different
    object per index shape: the coarse quantizer's ``cp`` on an ``IndexIVF`` (itself reachable
    only through ``extract_index_ivf`` when the factory string wrapped it in a pre-transform,
    e.g. ``OPQ...,IVF...,PQ...``) and a second ``cp`` on each ``ProductQuantizer``. Setting
    ``quantizer_train_seed`` in config and then never handing it to FAISS, which is what
    shipped, leaves each rendering's quantizer trained from a different random state while
    the manifest records a seed, i.e. a hidden per-rendering confound with a paper trail that
    denies it.

    Returns the knobs it actually set (for the log). A trainable index with no seedable knob
    raises instead of training unseeded: an unpinnable clustering must be a visible failure,
    never a silent one. Nothing is called on the shipped exact path, ``IndexFlatIP`` trains
    nothing, which is why this is reached only from the non-``exact`` branch.
    """
    applied: list[str] = []

    def _seed(owner: Any, path: str) -> None:
        cp = getattr(owner, "cp", None)
        if cp is not None and hasattr(cp, "seed"):
            cp.seed = int(opts.train_seed)
            applied.append(path)

    targets: list[tuple[Any, str]] = [(index, "index")]
    try:
        import faiss

        ivf = faiss.extract_index_ivf(index)
    except Exception:  # noqa: BLE001, not an IVF index; the other knobs still apply
        ivf = None
    if ivf is not None and ivf is not index:
        targets.append((ivf, "ivf"))
    for owner, name in targets:
        _seed(owner, f"{name}.cp")
        pq = getattr(owner, "pq", None)
        if pq is not None:
            _seed(pq, f"{name}.pq.cp")

    if not applied and not bool(getattr(index, "is_trained", True)):
        raise IndexIntegrityError(
            f"the dense leg's ann_factory {opts.ann_factory!r} builds an index that must be "
            "trained but exposes no ClusteringParameters.seed to pin, so each rendering "
            "would cluster from a different random state. Refusing to train unseeded, "
            "either use the exact 'Flat' recipe or name the seedable knob here."
        )
    _log.info(
        "preprocess.index.dense.train_seed",
        seed=int(opts.train_seed),
        knobs=applied,
        ann_factory=opts.ann_factory,
    )
    return applied


def _new_faiss_index(dim: int, opts: IndexBuildOptions) -> Any:
    """The dense index for this recipe: exact ``IndexFlatIP``, or the factory string.

    ``IndexFlatIP`` is the shipped default because the system is unsupervised: with no qrels,
    an approximate index's recall loss cannot be measured, and it could vary by rendering,
    a hidden confound inside the very comparison this corpus exists to make. Exact search
    removes the approximation rather than bounding it, and at 1024 dims it is affordable
    (~115.5 GB fp32 for the whole corpus against 375-500 GB node RAM, mmap'd).
    """
    import faiss

    if opts.exact:
        return faiss.IndexFlatIP(dim)
    return faiss.index_factory(dim, opts.ann_factory, faiss.METRIC_INNER_PRODUCT)


class FaissDenseLeg:
    """``BAAI/bge-m3`` dense vectors in FAISS. No instruction prefix, ever.

    The prefix matters: the swapped-out alternative (Qwen3-Embedding) requires an
    English-authored instruction template, which is an unvalidatable per-language bias
    channel in a study about English translation. BGE-M3 needs none, so this leg passes the
    composed passage text through unmodified, the encode call site below is the whole
    proof, there is no prompt/prefix/template anywhere in it.
    """

    name = DENSE_LEG

    def encode_docs(self, ctx: IndexCtx, texts: Sequence[str]) -> Sequence[Any]:
        return ctx.dense.embed(list(texts), mode="dense")

    def encode_query(self, ctx: IndexCtx, query: str) -> Any:
        return ctx.dense.embed([query], mode="dense")[0]

    def writer(self, out_dir: Path, ctx: IndexCtx) -> LegWriter:
        return _FaissWriter(out_dir, ctx.opts)

    def open(self, leg_dir: Path, ctx: IndexCtx) -> Any:
        import faiss

        index = faiss.read_index(str(leg_dir / _DENSE_FILENAME))
        if ctx.opts.nprobe and hasattr(index, "nprobe"):
            index.nprobe = ctx.opts.nprobe
        return index

    def search(
        self, reader: Any, ctx: IndexCtx, query_rep: Any, top_k: int
    ) -> list[tuple[int, float]]:
        import numpy as np

        q = np.asarray(query_rep, dtype="float32").reshape(1, -1)
        scores, ordinals = reader.search(q, int(top_k))
        return [
            (int(o), float(s)) for o, s in zip(ordinals[0], scores[0], strict=True) if int(o) >= 0
        ]


# --- sparse: MILCO -> Seismic ----------------------------------------------- #
class _SeismicWriter:
    """Accumulate MILCO sparse vectors into a ``SeismicDataset``, then build + save.

    Documents are keyed by the stringified ordinal, not the ``passage_id``: Seismic's own
    string type is ``U30``-capped and truncates silently, so keying on an id whose length is
    a property of the corpus would be a latent correctness bug. Ordinals are <= 8 chars
    forever, and the ordinal -> ``passage_id`` mapping lives in ``idmap.parquet`` exactly as
    it does for the other two legs, one mechanism, not three.

    This is the one place the sparse top-k cut is applied (``seismic_top_k``), and it is
    here rather than in ``preprocess.vectors``' writer: the vectors on disk stay
    the encoder's raw output, so re-tuning k is a CPU re-assemble over 87.7 GB of already-
    encoded vectors instead of a 13 GPU-h re-encode. Both this and ``vectors.py`` reach the
    cut through the same :func:`~ragtime.serving.sparse_milco.seismic_components`, one
    implementation of "a sparse vector in Seismic's form", with the cut as an argument, never
    a second pruning path that could order or tie-break differently.
    """

    __slots__ = ("_dataset", "_dtype", "_empty", "_pruned", "opts", "out_dir")

    def __init__(self, out_dir: Path, opts: IndexBuildOptions) -> None:
        import seismic

        self.out_dir = out_dir
        self.opts = opts
        self._dataset = seismic.SeismicDataset()
        self._dtype = seismic.get_seismic_string()
        self._empty = 0
        self._pruned = 0

    def add(self, ordinals: Sequence[int], reps: Sequence[Any]) -> None:
        import numpy as np

        for ordinal, weights in zip(ordinals, reps, strict=True):
            components, values = seismic_components(weights, top_k=self.opts.seismic_top_k)
            # Counted, never assumed: the cut is a real content loss, and a content loss
            # with no counter is invisible.
            self._pruned += max(0, len(weights) - len(components))
            if not components:
                # Kept, not skipped: dropping it would break the id-set equality every
                # rendering's index is asserted on. Counted so a degenerate encode is visible.
                self._empty += 1
            self._dataset.add_document(
                str(ordinal),
                np.array(components, dtype=self._dtype),
                np.array(values, dtype=np.float32),
            )

    def finish(self) -> None:
        from seismic import SeismicIndex

        # Documented deviation, logged at every build rather than assumed away: this is the
        # one leg whose clustering takes no seed (:data:`UNSEEDABLE_LEGS` carries the vendor
        # signature that proves it). The two seedable legs read opts.train_seed; here the
        # honest record is that the RNG is the OS's. Do not "fix" this with a global
        # random/torch seed: the randomness is inside the Rust extension and unreachable.
        _log.info(
            "preprocess.index.sparse.unseeded_clustering",
            path=str(self.out_dir),
            vendor_call=UNSEEDABLE_LEGS[SPARSE_LEG],
        )
        # Every parameter of the vendor call is passed explicitly. The shipped values are the
        # vendor.s current defaults, so declaring them changes no byte today and makes a future
        # vendor change visible as a diff rather than as a silently different index.
        index = SeismicIndex.build_from_dataset(
            self._dataset,
            n_postings=self.opts.seismic_n_postings,
            centroid_fraction=self.opts.seismic_centroid_fraction,
            min_cluster_size=self.opts.seismic_min_cluster_size,
            summary_energy=self.opts.seismic_summary_energy,
            max_fraction=self.opts.seismic_max_fraction,
            doc_cut=self.opts.seismic_doc_cut,
        )
        # Stem, not the full filename: save() appends ".index.seismic" itself.
        _seismic_save(index, self.out_dir / _SPARSE_SAVE_STEM)

    @property
    def empty(self) -> int:
        return self._empty

    @property
    def pruned(self) -> int:
        """Components dropped by the top-k cut across every passage this writer took."""
        return self._pruned


def _seismic_save(index: Any, path: Path) -> None:
    """Serialize a Seismic index, naming the vendor call that has to exist.

    ``build_from_dataset``/``search`` are spike-verified; the save/load pair is
    the one call in this module exercised for the first time by the full gate. It is wrapped
    so a vendor-surface change fails with a message that says what to look for, instead of an
    ``AttributeError`` 90 minutes into a build.
    """
    save = getattr(index, "save", None)
    if save is None:  # pragma: no cover - vendor surface change
        raise AttributeError(
            "seismic index exposes no .save(); the sparse leg cannot be serialized. Check "
            "pyseismic-lsr 0.5.1's persistence API (save/load) before changing the pin."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    save(str(path))


def _seismic_load(path: Path) -> Any:
    """Reopen a saved Seismic index (the read half of :func:`_seismic_save`)."""
    from seismic import SeismicIndex

    load = getattr(SeismicIndex, "load", None)
    if load is None:  # pragma: no cover - vendor surface change
        raise AttributeError(
            "seismic.SeismicIndex exposes no .load(); a published sparse leg cannot be "
            "reopened. Check pyseismic-lsr 0.5.1's persistence API before changing the pin."
        )
    return load(str(path))


class SeismicSparseLeg:
    """MILCO learned-sparse weights in a Seismic impact index.

    This is the leg that makes the sparse role work on all three renderings: MILCO projects
    every source language into one English SPLADE lexical space, so it fires cross-lingually
    on ``original`` too (9/9 rank-1) where a raw lexical leg shares near-zero
    surface tokens with Cyrillic or CJK text.
    """

    name = SPARSE_LEG

    def encode_docs(self, ctx: IndexCtx, texts: Sequence[str]) -> Sequence[Any]:
        return ctx.milco.encode_text(list(texts))

    def encode_query(self, ctx: IndexCtx, query: str) -> Any:
        return ctx.milco.encode_query(query)

    def writer(self, out_dir: Path, ctx: IndexCtx) -> LegWriter:
        return _SeismicWriter(out_dir, ctx.opts)

    def open(self, leg_dir: Path, ctx: IndexCtx) -> Any:
        return _seismic_load(leg_dir / _SPARSE_FILENAME)

    def search(
        self, reader: Any, ctx: IndexCtx, query_rep: Any, top_k: int
    ) -> list[tuple[int, float]]:
        import numpy as np
        import seismic

        components, values = seismic_components(query_rep)
        dtype = seismic.get_seismic_string()
        hits = reader.search(
            query_id="q",
            query_components=np.array(components, dtype=dtype),
            query_values=np.array(values, dtype=np.float32),
            k=int(top_k),
            query_cut=ctx.opts.seismic_query_cut,
            heap_factor=ctx.opts.seismic_heap_factor,
        )
        # Verified return shape: (query_id, score, doc_id): doc_id is our ordinal string.
        return [(int(doc_id), float(score)) for _, score, doc_id in hits]


# --- late interaction: PyLate / PLAID --------------------------------------- #
class PlaidBufferTooLargeError(RuntimeError):
    """One PLAID part's units do not fit in one ``add_documents`` call on this node.

    Raised, never worked around: the alternatives are all worse. Dropping units would publish
    a short index under a full ``idmap.parquet`` (the id-set-integrity invariant
    :meth:`IndexAdapter.validate` enforces would then be violated by construction), and
    splitting a part's add into several calls is what this class exists to prevent, see
    :class:`_PlaidWriter` for the measured crash that costs.

    Part-sharding (``index_build.config.plaid_part_passages``) made this guard satisfiable,
    it did not make it unnecessary. The part size bounds the buffer in passages; whether that
    many passages fit is still a property of the node, so a part configured (or a corpus
    packed) beyond this worker's RAM must fail here, as a Python exception with the
    arithmetic, rather than as a SIGABRT or a silently short index.
    """


def _rep_nbytes(rep: Any) -> int:
    """One window embedding's size in bytes: the buffer guard's only measurement.

    Not a knob and not a dtype assumption: it reads the object's own ``nbytes`` where the
    encoder returned an array (always, on the real path, numpy and torch both expose it) and
    falls back to 8 bytes per component only for a Python-list stand-in.
    """
    nbytes = getattr(rep, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    try:
        return sum(len(row) for row in rep) * 8
    except TypeError:
        pass
    try:
        return len(rep) * 8
    except TypeError:  # pragma: no cover - a rep with no length is not an embedding
        return 0


def _host_memory_headroom_bytes() -> int:
    """Bytes this process could still allocate: an environment fact, never a config knob.

    Read fresh at each check, cgroup first: SLURM confines a job to a memory cgroup, so
    ``/proc/meminfo`` on a shared node overstates by the whole rest of the machine. Returns
    ``0`` for "unknowable", and the caller then does not invent a limit: a guard that guessed
    a ceiling would be a run choice hardcoded in code.
    """
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
        if limit != "max":
            current = Path("/sys/fs/cgroup/memory.current").read_text(encoding="utf-8").strip()
            return max(0, int(limit) - int(current))
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


class _PlaidWriter:
    """Buffer one part of the shard, issue exactly one ``add_documents``, repeat.

    The unit PLAID stores is one document per passage, keyed by the passage ordinal, the
    same unit every other leg stores. (It stored MaxP *windows* until with a
    ``window_ordinal -> passage ordinal`` sidecar to attribute them; both are gone now that
    packing guarantees a passage fits the window, which is why this writer needs no map at
    all and an ordinal means the same thing in all three legs.) ``idmap.parquet``
    is still exactly the corpus's, which is what the id-set integrity assertion checks.

    One ``add_documents`` per part is a correctness requirement rather than a tuning choice.
    The first call on an index takes fast-plaid's ``create`` path; every later one takes
    ``update`` -> ``process_update`` -> ``update_centroids`` -> ``compute_kmeans`` ->
    ``_kmeans_torch_double_chunked``, whose ``cluster_sums.index_add_(0, best_ids, ...)``
    trips a device-side ``dstIndex < dstAddDimSize`` assert and aborts the worker once enough
    units have accumulated. The threshold is where the failure was observed rather than a
    proven safe bound, which is why this writer never issues a second call against the same
    index: every part is a fresh index taking its own ``create``.

    Parts exist because buffering a whole per-language shard is 376 GiB of token embeddings
    for English alone (2.9 M passages x 295 content tokens x 0.92 vectors/token x 512 bytes
    fp32 on the host) against 375-500 GB nodes already holding three encoders, with
    ``add_documents`` needing roughly a second copy. Streaming the buffer to disk does not
    help, since ``create`` materializes it regardless, and re-packing does not help, since
    total vectors track total tokens however passages are cut. So the index is cut instead:
    ``(variant, source_lang)`` becomes ``(variant, source_lang, part)``, each part an
    independent PLAID index of at most ``index_build.config.plaid_part_passages`` passages,
    fan-merged at query time by :meth:`PlaidLateInteractionLeg.search`.

    Part boundaries are ordinal arithmetic, not bucket arithmetic: part ``p`` holds
    ordinals ``[p*N, (p+1)*N)``, so a part's membership is a function of the shared, text-free
    table's order and ``N`` alone, identical across the three renderings (which share that
    table, which has no ``variant`` column) and reproducible on a re-run whatever the
    ``Batcher`` produced. Under the shipped ``vectorize``/``assemble`` split that order is the
    table's row order (:func:`_row_order_lang_ids`); under the legacy fused
    :class:`IndexAdapter` it is the pinned ``(token_count, passage_id)`` one
    (:func:`_ordered_lang_keys`). The final part is a remainder and may be small; at corpus
    grain even a 1-passage remainder carries hundreds of token vectors, well above
    fast-plaid's ~10-vector codec-training floor. Under the split, a part is not sorted ascending
    by token count (see ``preprocess.vectorize``'s module docstring): it holds a row-order slice
    with a normal length distribution, so every part clears that floor by millions of vectors.

    ``Batcher`` still drives the encode side (GPU batching is wanted, and bucket membership is
    the pinned, rendering-invariant composition of property 3); only the index insertion is
    part-grained. Peak residency is therefore one part, measured against the node's own
    headroom, with :class:`PlaidBufferTooLargeError` raised rather than a unit dropped or a
    part's add split.
    """

    __slots__ = ("_buffer", "_bytes", "_ids", "_parts", "_peak", "_units", "opts", "out_dir")

    def __init__(self, out_dir: Path, opts: IndexBuildOptions) -> None:
        self.out_dir = out_dir
        self.opts = opts
        # The current part's single ``add_documents`` call's two arguments, accumulated
        # across buckets and dropped the moment the part is flushed.
        self._ids: list[str] = []
        self._buffer: list[Any] = []
        self._bytes = 0
        self._parts = 0
        self._units = 0
        self._peak = 0

    def add(self, ordinals: Sequence[int], reps: Sequence[Any]) -> None:
        """Buffer one bucket, flushing a part whenever it fills. ``reps[i]`` is ``ordinals[i]``.

        A bucket may straddle a part boundary and that is deliberate: cutting on the ordinal
        makes the part plan independent of the encode batching, which is what keeps it equal
        across renderings. See the class docstring for why a second ``add_documents`` against
        an already-created part index is a crash, not a slowdown.
        """
        limit = max(1, int(self.opts.plaid_part_passages))
        for ordinal, rep in zip(ordinals, reps, strict=True):
            self._ids.append(str(int(ordinal)))
            self._buffer.append(rep)
            self._bytes += _rep_nbytes(rep)
            if len(self._ids) >= limit:
                self._flush_part()
        # Once per bucket, not once per passage: the probe reads the cgroup off /sys and the
        # buffer cannot grow past ``limit`` between two checks anyway.
        self._assert_buffer_fits("buffering this part's passages")

    def _flush_part(self) -> None:
        """Create part ``self._parts`` and hand it its whole buffer in one call."""
        if not self._ids:
            return
        self._assert_buffer_fits("adding this part's passages to PLAID")
        # The index is created here, so the one add below is unambiguously its `create` call.
        index = _open_plaid(self.out_dir, self.opts, override=True, part=self._parts)
        pinned = _pin_torch_rng(int(self.opts.train_seed))
        _log.info(
            "preprocess.index.plaid.part_add",
            part=self._parts,
            units=len(self._ids),
            buffered_bytes=self._bytes,
            rng_pinned=pinned,
            use_triton_kmeans=_PLAID_USE_TRITON_KMEANS,
            path=str(self.out_dir),
        )
        # The RNG pin, the Triton pin (:data:`_PLAID_USE_TRITON_KMEANS`) and this one are the
        # Three things that make this leg reproducible; all three are needed and none of them
        # alone is sufficient.
        with _deterministic_torch_ops():
            index.add_documents(documents_ids=self._ids, documents_embeddings=self._buffer)
        self._peak = max(self._peak, self._bytes)
        self._parts += 1
        self._units += len(self._ids)
        # The engine owns these vectors now; dropping the buffer is what bounds residency to
        # One part rather than one shard.
        self._ids = []
        self._buffer = []
        self._bytes = 0

    def finish(self) -> None:
        self._flush_part()
        if self._parts == 0:
            # An empty shard is a legal artifact: create one empty part so the leg is
            # readable rather than an absent directory (the same rule the dense leg follows).
            _open_plaid(self.out_dir, self.opts, override=True, part=0)
            self._parts = 1
        # The part census, published inside the leg by the same atomic rename as the engine
        # files. A reader fans over exactly these parts and no others; nothing infers the
        # count. The PLAID document id is the passage ordinal, so ``idmap.parquet`` still
        # resolves a hit on its own: parts add no second id space.
        write_jsonl(
            self.out_dir / _PARTS_FILENAME,
            [
                {
                    "parts": self._parts,
                    "part_passages": int(self.opts.plaid_part_passages),
                    "units": self._units,
                    "peak_part_bytes": self._peak,
                    "dirs": [_plaid_part_name(i) for i in range(self._parts)],
                }
            ],
            skip_if_done=False,
        )

    def _required_bytes(self) -> int:
        """What the ``add_documents`` call needs, not what our buffer holds.

        ``self._bytes`` x :data:`PLAID_ADD_PEAK_MULTIPLIER`, see that constant for the
        measurement. Kept as its own method so the guard, the error message and the tests all
        read one number.
        """
        return int(self._bytes * PLAID_ADD_PEAK_MULTIPLIER)

    def _assert_buffer_fits(self, phase: str) -> None:
        """Abort while the failure is still a Python exception, never a truncated index.

        The condition is the measured requirement of the call rather than the size of the
        buffer: ``add_documents``' peak RSS is 2.17x the buffer under fp32 and 3.23x under
        fp16. A guard comparing against 1.0x would pass at 1.5x headroom and then OOM inside
        the vendor call, losing the whole leg for that part, and it would grow less protective
        the more fp16 shrank the buffer.

        On failure the leg's temp directory is discarded unpublished, leaving the dense/sparse
        legs' ``_SUCCESS`` markers (and this leg's absence) exactly as they were.

        Still live after part-sharding: ``plaid_part_passages`` bounds the buffer in passages,
        and whether that many passages fit is a property of this node.
        """
        headroom = _host_memory_headroom_bytes()
        required = self._required_bytes()
        if not headroom or headroom >= required:
            return
        raise PlaidBufferTooLargeError(
            f"{self.out_dir}: {phase} needs {required / 2**30:.1f} GiB, "
            f"{self._bytes / 2**30:.1f} GiB of buffered vectors x "
            f"{PLAID_ADD_PEAK_MULTIPLIER} (add_documents' measured peak RSS), for "
            f"{len(self._ids)} passages (part {self._parts}, "
            f"index_build.config.plaid_part_passages={self.opts.plaid_part_passages}), and "
            f"only {headroom / 2**30:.1f} GiB of RAM is left. The late-interaction leg adds "
            "every part in one add_documents call because fast-plaid's incremental update "
            "path re-clusters and aborts the worker with a device-side assert, so this "
            "cannot be split into further calls "
            "and units are never dropped. Give the worker more memory, or lower "
            "plaid_part_passages (a hashed index_build key, i.e. a fairness-family decision "
            "that re-derives this leg)."
        )

    @property
    def parts(self) -> int:
        """How many PLAID parts this leg holds: derived from the units seen, never assumed."""
        return self._parts

    @property
    def buffered_bytes(self) -> int:
        """Peak bytes any single part held: measured, not estimated.

        This is our buffer, not the worker's requirement. Sizing ``--mem`` from it
        under-provisions by 2-3x: multiply by :data:`PLAID_ADD_PEAK_MULTIPLIER` first. Same
        caveat applies to the counter it feeds (:data:`STAT_PLAID_PART_BYTES`) and to the
        ``peak_part_bytes`` field of the leg's ``parts.json``.
        """
        return max(self._peak, self._bytes)


def _pin_torch_rng(seed: int) -> bool:
    """Seed the ambient torch RNG before an ``add_documents``. Measured, not precautionary.

    fast-plaid's ``create`` takes a ``seed`` and passes it to ``FastKMeans`` and to the Rust
    codec, but ``compute_kmeans`` picks which documents train the codec with a bare
    ``torch.randperm(num_documents)`` (pylate 1.6.0 / fast-plaid), i.e. from the process-global
    RNG that nothing seeded. So two builds of the same passages under the same declared seed
    cluster on different samples and produce different centroids, codes, residuals and ivf,
    measured here, on CPU, as six differing files out of 21 between two runs of the identical
    single-call path (SM29's control assertion).

    That is the same defect ``_apply_faiss_train_seed`` exists for on the dense leg: a seed
    declared in the recipe, recorded in the manifest, and never reaching the RNG that decides
    the artifact. Pinning it here makes the leg reproducible, which the artifact-tree-is-the-
    checkpoint invariant requires, and which a cross-rendering comparison requires even more
    (an unpinned clustering is a per-rendering difference with no cause in the text).

    Returns whether the pin was applied, so the log records the truth rather than the
    intention; torch is imported lazily because this module must stay importable without it.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - the real path always has torch (pylate needs it)
        return False
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - GPU-only branch
        torch.cuda.manual_seed_all(seed)
    return True


@contextmanager
def _deterministic_torch_ops() -> Iterator[bool]:
    """Run the enclosed block under torch's deterministic-algorithm kernels, then restore.

    Scoped to the ``add_documents`` call rather than set process-wide: this stage shares a
    worker with three encoders whose forward passes have no reproducibility requirement, and a
    global flag would impose (and could break) determinism on them too.

    Measured at op level on the k-means loop's own shapes,
    ``cluster_sums.index_add_(0, best_ids, data_chunk.float())`` uses float32 CUDA atomics and
    returns three distinct results with the flag off and one with it
    on; the fp16 cuBLAS GEMM beside it is already identical either way. In the torch k-means
    path the fp16 downcast of the new centroids happens to mask that atomic-order difference at
    the k this gate measures, which is precisely why the flag is set rather than relied on to
    be unnecessary: masking is a property of one (k, chunk size, dtype) triple, not a
    guarantee, and the corpus build runs at a different one.

    ``warn_only=True`` deliberately: a strict flag turns any op with no deterministic kernel
    into a hard failure deep inside a vendor call (and, on CUDA >= 10.2, demands a
    ``CUBLAS_WORKSPACE_CONFIG`` env var), which would trade a reproducibility improvement for a
    new class of build abort. The deterministic kernels are selected either way, ``warn_only``
    changes only what happens when one does not exist.

    Yields whether the flag was actually applied, so a caller can log the truth.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - the real path always has torch
        yield False
        return
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield True
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


def _plaid_part_name(part: int) -> str:
    """``plaid-00000``: the engine directory of one part inside a leg directory.

    Zero-padded so the on-disk order is the part order, and applied to every part including a
    lone one (see :data:`_PLAID_INDEX_NAME`).
    """
    return f"{_PLAID_INDEX_NAME}-{part:05d}"


def _read_parts(leg_dir: Path) -> dict[str, Any]:
    """The late-interaction leg's part census, cross-checked against what is on disk.

    Two independent facts have to agree: the count the writer recorded, and the part
    directories that exist. Trusting either alone is the failure this guards, a reader that
    globbed would silently fan over 9 of 10 parts if one were lost, and a reader that only
    read the number would open a directory that is not there. Disagreement raises; there is
    no "search what we found" fallback, because a partial index that looks complete is worse
    than a failed read.
    """
    path = leg_dir / _PARTS_FILENAME
    if not path.exists():
        raise IndexIntegrityError(
            f"{leg_dir}: no {_PARTS_FILENAME}, a published late-interaction leg always "
            "records how many PLAID parts it holds, and a reader must never infer that "
            "count (an under-count returns a full-looking but silently short result list)"
        )
    record = read_jsonl(path)[0]
    parts = int(record.get("parts") or 0)
    named = [str(name) for name in record.get("dirs") or ()]
    if parts < 1 or len(named) != parts:
        raise IndexIntegrityError(
            f"{leg_dir}: {_PARTS_FILENAME} claims {parts} part(s) but names {named}"
        )
    on_disk = sorted(p.name for p in leg_dir.glob(f"{_PLAID_INDEX_NAME}-*") if p.is_dir())
    if on_disk != sorted(named):
        raise IndexIntegrityError(
            f"{leg_dir}: the recorded parts {sorted(named)} are not the parts on disk "
            f"{on_disk}, this leg would be searched short (or over a stale part)"
        )
    return record


def plaid_part_count(leg_dir: Path) -> int | None:
    """Parts in a published leg, or ``None`` for a leg that has none (dense/sparse).

    The manifest's source for the count, so the number a reader fans across and the number an
    auditor reads come from one place.
    """
    if not (leg_dir / _PARTS_FILENAME).exists():
        return None
    return int(_read_parts(leg_dir)["parts"])


#: Set once ``enable_plaid_idmap_cache`` has run, so the vendor patch is applied exactly once per
#: process no matter how many of the 78 parts are opened.
_IDMAP_CACHE_PATCHED: list[int] = []


def enable_plaid_idmap_cache() -> int:
    """Memoize PyLate's two per-call id-map unpickles, byte-identical results.

    ``FastPlaid._load_documents_ids_to_plaid_ids`` and its inverse unpickle the part's id maps on
    every search call. Memoizing them on the instance changes no result, the maps are immutable
    for the life of an opened part, and it is the largest query-time win available on this stack.

    Returns the number of methods patched. A 0 is a signal rather than a no-op: the vendor has
    renamed them and the speed-up is not in effect, so callers record it instead of assuming
    success.
    """
    if _IDMAP_CACHE_PATCHED:
        return _IDMAP_CACHE_PATCHED[0]
    try:
        from pylate.indexes import fast_plaid as fp
    except Exception:  # noqa: BLE001 - pylate absent (small tier) must never break an import
        return 0

    patched = 0
    for name in ("_load_documents_ids_to_plaid_ids", "_load_plaid_ids_to_documents_ids"):
        orig = getattr(fp.FastPlaid, name, None)
        if orig is None:
            continue
        attr = f"__ragtime_idmap_{name}"

        def make(orig=orig, attr=attr):  # closure binds the vendor method at patch time
            def cached(self):
                got = getattr(self, attr, None)
                if got is None:
                    got = orig(self)
                    object.__setattr__(self, attr, got)
                return got

            return cached

        setattr(fp.FastPlaid, name, make())
        patched += 1
    _IDMAP_CACHE_PATCHED.append(patched)
    _log.info("preprocess.index.plaid.idmap_cache", patched=patched, speedup="52.93x measured")
    return patched


def resolve_query_plaid_device(device: str, env_device: str | None) -> str:
    """The query-side PLAID device: the config decides the class, the replica decides the ordinal.

    Kept out of :func:`_open_plaid` so the rule is testable without a GPU or an index.

    * A differing ordinal is adopted (``cuda:0`` configured, ``cuda:1`` in the environment gives
      ``cuda:1``). A replica is a whole stack pinned to one card, and the service sets this
      variable to that replica's own device.
    * A differing class raises. ``cuda`` against ``cpu`` is a fairness-shared variable that
      reorders results, and nothing on disk would record which device produced them.
    """
    if not env_device:
        return device
    if env_device.split(":", 1)[0] != str(device).split(":", 1)[0]:
        raise IndexIntegrityError(
            f"RAGTIME_QUERY_PLAID_DEVICE={env_device!r} selects a different device class than "
            f"the configured query_plaid_device={device!r}. A differing ordinal (cuda:0 vs "
            "cuda:1) is fine, a replica pins its own card. Crossing cuda<->cpu is not: the "
            "config is the run record, the device reorders results, and nothing on disk would "
            "say which device produced them. Change the config, or unset the variable."
        )
    return env_device


def _open_plaid(index_dir: Path, opts: IndexBuildOptions, *, override: bool, part: int) -> Any:
    """Construct one part's PyLate PLAID index under ``index_dir`` (nbits + seed from config).

    ``part`` selects the engine sub-directory (:func:`_plaid_part_name`) and is deliberately
    required: a defaulted part number would let a caller that forgot about parts silently
    create, and then overwrite, part 0.

    ``nbits=2`` is the recommended default: measured 0.46 TB corpus-wide at 12/12 rank-1
    self-retrieval, the less compressed of the two settings that both held
    rank-1 perfectly, this project defaults to the least-compressed affordable option
    because it has no qrels with which to measure what tighter compression would cost.

    ``seed`` is the leg's k-means RNG and it is not optional here: PyLate 1.6.0 defaults it to
    a vendor literal (42) and hands it to fast-plaid's ``FastKMeans`` and Rust codec. Leaving
    the vendor default in place would mean the centroids of the three renderings' indexes come
    from a seed the run record does not state. It reads the same declared knob the dense leg
    does.

    The kwarg is necessary and not sufficient: fast-plaid does not seed the global generators
    itself, and ``compute_kmeans`` selects its training sample with an unseeded
    ``torch.randperm``. :func:`_pin_torch_rng` covers that gap at the call site.

    ``use_triton`` is the third pin, and the one that actually decided this leg's
    reproducibility on GPU: pylate passes it straight through to fast-plaid's
    ``compute_kmeans``, where ``None`` means "Triton if it imports" and the Triton assignment
    kernel is not reproducible run to run. See :data:`_PLAID_USE_TRITON_KMEANS` for the
    measurement. Passing it explicitly also means the default flipping in a future pylate
    cannot silently change what this stage builds.

    ``device`` is the fourth. Left unset, fast-plaid picks one off the worker, and a build
    whose bytes depend on whether the scheduler handed it a GPU is a per-rendering confound
    with no cause in the text. It comes from ``index_build.config.assemble_device``, hashed
    and closed-set validated (never ``"auto"``); see :data:`_DEFAULT_ASSEMBLE_DEVICE`.
    """
    from pylate import indexes

    # Build and query take different devices, and `override` is the discriminator: the two
    # build call sites pass override=True, the one query call site passes override=False.
    # Assembly's device is hashed into `index_hash` because it decides stored bytes; the query
    # device is a QUERY_TIME_KEYS member because searching writes none.
    build = bool(override)
    if not build:
        # Query path only: memoize the vendor's per-call id-map unpickles. Idempotent and
        # process-wide, so opening a whole rendering's parts patches once.
        enable_plaid_idmap_cache()
    device = opts.assemble_device if build else opts.query_plaid_device
    if not build:
        # A replica may select its own card through the environment; it may not cross the
        # device class the config declares.
        device = resolve_query_plaid_device(
            device, os.environ.get("RAGTIME_QUERY_PLAID_DEVICE")
        )
    kwargs: dict[str, Any] = {
        "index_folder": str(index_dir),
        "index_name": _plaid_part_name(part),
        "nbits": opts.plaid_nbits,
        "seed": int(opts.train_seed),
        "use_triton": _PLAID_USE_TRITON_KMEANS,
        "device": device,
        "override": override,
    }
    # `low_memory=True` on the CUDA query path only: it is meaningless on CPU and the build
    # path's bytes must not change. It bounds the load, not the search allocation, which is
    # decided by three vendor defaults (`n_full_scores`, `n_ivf_probe`, `batch_size`) that
    # trade recall for memory and are therefore left at their documented values.
    if not build and device != "cpu":
        kwargs["low_memory"] = True
    return indexes.PLAID(**kwargs)


class PlaidLateInteractionLeg:
    """The MTD/PyLate late-interaction leg (MaxSim over token vectors).

    Two things this leg must not hide. First, the checkpoint's fixed ``document_length``
    truncates non-English passages more aggressively than English: a Chinese passage runs
    312-343 tokens against a 220-token window. ``preprocess.packing`` removes that upstream,
    so this leg encodes whole passages and :func:`_assert_window_covers_packing` refuses to
    build if ``document_length`` is smaller than the budget those passages were packed to;
    whatever the encode still loses is counted as
    ``index.late_interaction_truncated_tokens``, which is the alarm if the upstream guarantee
    stops holding. Second, MaxSim scores are not magnitude-comparable to the sparse leg's dot
    products, only their rank-1 rates are, which is why fusion downstream is rank-based RRF
    rather than a score merge.
    """

    name = LATE_INTERACTION_LEG

    def encode_docs(self, ctx: IndexCtx, texts: Sequence[str]) -> Sequence[Any]:
        return ctx.mtd.encode_docs(list(texts))

    def encode_query(self, ctx: IndexCtx, query: str) -> Any:
        return ctx.mtd.encode_query(query)

    def writer(self, out_dir: Path, ctx: IndexCtx) -> LegWriter:
        return _PlaidWriter(out_dir, ctx.opts)

    def open(self, leg_dir: Path, ctx: IndexCtx) -> Any:
        """Every part of this leg, in part order: the fan :meth:`search` merges over.

        The count comes from the leg's own census (:func:`_read_parts`), which is
        cross-checked against the directories on disk. Nothing here infers it.

        ``leg_dir`` is resolved through :func:`resolve_leg_dir` first, so a caller that has
        mirrored this leg onto faster local storage is served by a supported indirection rather
        than by reassigning this method (see :func:`set_leg_dir_mirror`).
        """
        leg_dir = resolve_leg_dir(leg_dir)
        record = _read_parts(leg_dir)
        return tuple(
            _open_plaid(leg_dir, ctx.opts, override=False, part=i)
            for i in range(int(record["parts"]))
        )

    def search(
        self, reader: Any, ctx: IndexCtx, query_rep: Any, top_k: int
    ) -> list[tuple[int, float]]:
        """Top-``top_k`` passages by MaxSim, fan-merged over the shard's parts.

        One PLAID document is one passage, so there is no collapse step and no over-fetch
        beyond the fan: each part returns its own top-``k`` and the global top-``k`` is a
        subset of their union, so ``k`` per part is exactly sufficient.

        Why the merge is a score sort and not RRF. A ColBERT/PLAID score is
        ``sum over query tokens of max over document tokens of <q_i, d_j>``, a function of
        the (query, document) pair alone, with no index-global normalization anywhere in
        fast-plaid's return path (`search_on_device` hands back the Rust `pysearch` scores
        verbatim; the token-level scores are documented as plain dot products). So two parts
        of one shard report scores in the same space and comparing them is meaningful, unlike
        comparing this leg to the sparse one, whose dot products are not magnitude-comparable
        with it, which is why fusion across legs is rank-based RRF in ``common.fusion`` and is
        not reused here. What the split does perturb is the residual codec: each part quantizes
        against its own centroids, so a score carries part-local compression noise exactly as
        it already carried index-local compression noise before parts existed. That is a
        precision effect on a shared scale, not two different scales, and it is measured
        (the gate compares a document's score under a 1-part and a multi-part build of the
        same passages) rather than assumed.

        The sort is total and deterministic, ``(-score, ordinal)``, and an ordinal is unique
        within a shard, so two runs of one query over one shard return the identical list.
        Anything less would make the cross-rendering agreement diagnostics meaningless: they
        would measure tie-breaking order, not translation.

        This loop stays sequential while the cell-level fan (:func:`query_lang_leg`) is
        concurrent, and the asymmetry is geometric. There is one part size: an index part holds at
        most ``plaid_part_passages`` passages and :class:`_PlaidWriter` flushes a PLAID part
        every ``plaid_part_passages`` passages, so ``reader`` here is a one-tuple in every
        shipped cell (asserted by the acceptance census). Threading a one-iteration loop buys
        nothing and costs something real: this loop already runs inside a worker of the
        cell-level fan, so a pool here would nest, and 23 outer threads x N inner threads x
        torch's own intra-op pool is precisely the oversubscription :data:`QUERY_WORKERS_CAP`
        exists to bound. If ``plaid_part_passages`` is ever cut below the index part size,
        revisit this, and note that ``(-score, ordinal)`` is not a total order across parts
        (ordinals are part-local), so any concurrency added here must preserve part order,
        exactly as :func:`_fan_parts` does.
        """
        out: list[tuple[int, float]] = []
        for part in reader:
            hits = part(queries_embeddings=[query_rep], k=int(top_k))
            out.extend(
                # PyLate returns {"id": <our passage ordinal string>, "score": float}.
                (int(hit["id"]), float(hit["score"]))
                for hit in (next(iter(hits), []) if hits else [])
            )
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out[: int(top_k)]


def default_legs() -> tuple[Leg, ...]:
    """The three legs, in :data:`LEGS` order: the only leg set a build may have."""
    return (FaissDenseLeg(), SeismicSparseLeg(), PlaidLateInteractionLeg())


# --------------------------------------------------------------------------- #
# Query: the per-part fan's thread pool. Throughput only: the result is identical.
# --------------------------------------------------------------------------- #
#: The ``execution`` key naming the width of the per-part query fan.
#:
#: In ``execution`` and not in ``index_build.config``. It changes no stored
#: byte and no returned hit: it decides only how many of a cell's parts are searched at
#: once: so it is the same kind of knob as ``execution.vectorize_blocks_per_task`` (a
#: Width), not the same kind as ``encode_block_passages`` (a recipe). Putting it in the
#: hashed, fairness-shared ``index_build`` block would re-key the index for a latency retune
#: and force the whole family to agree on a machine fact; putting it in ``retrieval`` would
#: fold a machine fact into a block whose leaves are search semantics (``rrf``, ``reranker``,
#: and the ``index`` knob ``family_guard`` polices).
QUERY_WORKERS_KEY = "query_part_workers"

#: Ceiling on the auto-resolved fan width (an explicit config value is honoured above it).
#:
#: Every part search under this fan is a native call that may also thread internally: FAISS
#: through OpenMP, Seismic through Rayon, PyLate/PLAID through torch's intra-op pool. The
#: machine-level parallelism is therefore the product of this width and theirs, and a fan as
#: wide as the part count (en 23 / es 20 / zh 21 / ru 14) on a node that also hosts the
#: shared vLLM would multiply out to hundreds of runnable threads.
#:
#: The fan helps the dense leg and not the sparse one, because FAISS releases the GIL inside
#: ``search`` and ``pyseismic-lsr`` does not; the sparse leg pays a small thread overhead at
#: every width and needs Seismic's own multi-query API or separate processes instead. On the
#: dense leg the gain flattens by about width 6, where the remaining cost is memory
#: bandwidth rather than threading, so 8 is a bound rather than a target.
QUERY_WORKERS_CAP = 8

#: Named so a stack dump or a ``py-spy`` sample says which pool a thread belongs to.
_QUERY_THREAD_PREFIX = "ragtime-query-fan"


def default_query_workers() -> int:
    """Fan width when the config names none: half the visible CPUs, capped, at least 1.

    ``SLURM_CPUS_PER_TASK`` first (the cores this task was actually allocated, ``os.cpu_count``
    reports the whole node, which on a 128-core shared node would over-fan by 16x), then the
    machine. Halved because of the internal-threading argument in :data:`QUERY_WORKERS_CAP`:
    the engines below this fan take the other half. On the measured numbers there is nothing
    to lose by that half, both legs' wall time had flattened by width 6 (0.318 s dense,
    0.331 s sparse), and something real to lose without it.
    """
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        cpus = int(allocated) if allocated else (os.cpu_count() or 1)
    except ValueError:  # pragma: no cover - a non-numeric SLURM env is not worth a raise
        cpus = os.cpu_count() or 1
    return max(1, min(QUERY_WORKERS_CAP, cpus // 2))


def resolve_query_workers(value: Any = None) -> int:
    """The raw ``execution.query_part_workers`` value -> a usable width. One owner.

    ``None`` (absent from the config) means :func:`default_query_workers`; anything else is
    the operator's explicit choice and is honoured, floored at 1 so a ``0`` degrades to the
    sequential path rather than to a pool that can never run. Interpretation lives here, in
    the module that owns the fan, so a caller (``retrieval.context._index_ctx``) forwards the
    config value untouched instead of re-implementing the policy.
    """
    if value is None:
        return default_query_workers()
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        raise ConfigError(
            f"execution.{QUERY_WORKERS_KEY}={value!r}: the per-part query fan's width must be "
            "an integer (absent = derive it from the task's CPU allocation)"
        ) from None


def query_part_workers(cfg: RunConfig) -> int:
    """The resolved fan width for one run config (``execution.query_part_workers``)."""
    return resolve_query_workers(cfg.blocks.get("execution", {}).get(QUERY_WORKERS_KEY))


def _fan_parts(
    parts: Sequence[LegHandle],
    search_part: Callable[[LegHandle], list[tuple[str, float]]],
    workers: int,
    *,
    omp: int = 0,
) -> list[tuple[str, float]]:
    """Run ``search_part`` over every part, concatenating results in part order.

    The whole contract of this function is that it is invisible in the output. Concurrency
    here may not reorder anything, for a reason that is specific and not merely stylistic: the
    merge one level up sorts on ``(-score, passage_id)``, which is a total order, but the
    PLAID fan one level down sorts on ``(-score, ordinal)``, and an ordinal is unique only
    within a part, so two parts can present an identical sort key, and Python's stable sort
    would then resolve them by arrival order. Ordering by completion would make that tie
    depend on which part's engine happened to finish first. ``Executor.map`` yields in
    submission order (documented, and the reason ``as_completed`` is not used here), so the
    concatenation is byte-identical to the sequential loop this replaced.

    ``workers <= 1`` and single-part cells take the plain loop, no pool is created, so the
    degenerate case is the original code path rather than a one-thread emulation of it.
    """
    if workers <= 1 or len(parts) <= 1:
        return [hit for part in parts for hit in search_part(part)]

    def one(part: LegHandle) -> list[tuple[str, float]]:
        # The FAISS cap must be applied here, on the pool worker, not by the caller.
        # ``OMP_NUM_THREADS``/``omp_set_num_threads`` bind ``nthreads-var``, a per-thread ICV:
        # a call on the submitting thread binds only that thread, and every thread this pool
        # creates starts from the environment value instead. Measured:
        #     fan 1: omp 1/2/4/8/16 -> 0.84/2.01/3.89/7.67/15.75 cores   (obeyed exactly)
        #     fan 2: omp 1/2/4/8/16 -> 22.8/21.9/24.2/22.5/21.5 cores    (ignored entirely)
        # So a fan that believes it capped FAISS at one thread runs it at ~11 per worker, and
        # the product oversubscribes the node. Calling it per task is what makes the cap real.
        # This is a thread-count change only: DEDUP-0 asserts no `omp` value moves a score.
        if omp:
            _faiss_omp(omp)
        return search_part(part)

    with ThreadPoolExecutor(
        max_workers=min(int(workers), len(parts)), thread_name_prefix=_QUERY_THREAD_PREFIX
    ) as pool:
        per_part = list(pool.map(one, parts))
    return [hit for hits in per_part for hit in hits]


#: Leg-directory mirrors: ``{canonical leg dir -> local mirror}``. Empty by default, so the
#: shipped path is unchanged and this costs one dict lookup per leg open.
_LEG_DIR_MIRROR: dict[str, str] = {}


def set_leg_dir_mirror(mapping: dict[str, str] | None) -> None:
    """Serve these leg directories from a mirror (same bytes, cheaper page faults).

    Staging a rendering's blobs onto node-local disk before serving it is a real optimization,
    it cut cold load ~27%, and it needs the engine to open the mirror instead of the canonical
    path. Before this existed the only way to get that was to reassign
    ``PlaidLateInteractionLeg.open`` from outside (``devkit.rsvc.patch_plaid_open``), which is a
    monkeypatch of production internals: it silently breaks the moment ``open``'s signature
    moves, it is invisible to anyone reading this class, and two of them installed at once
    compose in an order nobody controls.

    The mapping is bytes-identical by contract, a mirror is a copy, never a different index,
    so this cannot change a result, only where the bytes are read from. Pass ``None`` or ``{}``
    to clear it.
    """
    _LEG_DIR_MIRROR.clear()
    if mapping:
        _LEG_DIR_MIRROR.update({str(k): str(v) for k, v in mapping.items()})


def resolve_leg_dir(leg_dir: Path) -> Path:
    """The path a leg should actually be opened from, honouring any registered mirror."""
    mirrored = _LEG_DIR_MIRROR.get(str(leg_dir))
    return Path(mirrored) if mirrored else leg_dir


def _faiss_omp(threads: int) -> None:
    """Set FAISS's own OpenMP width in this thread, at query time.

    ``OMP_NUM_THREADS`` is read once when libgomp initialises, so it cannot be changed after
    start-up by the environment; ``faiss`` exposes the setter, which is the only way to bind a
    pool worker. A no-op for the sparse leg, Seismic is Rust/Rayon and never consults OpenMP,
    and for late-interaction, which is torch.

    The single implementation. ``devkit.rsvc`` had its own copy; two spellings of a thread cap
    is how one path ends up oversubscribed while the other looks fine.
    """
    if threads <= 0:
        return
    try:
        import faiss

        faiss.omp_set_num_threads(int(threads))
    except Exception:  # noqa: BLE001 - a missing/failed setter must never fail a query
        return


def search_cell_with_rep(
    handle: LangHandle,
    leg: str,
    rep: Any,
    top_k: int,
    *,
    workers: int,
    omp: int = 0,
) -> list[tuple[str, float]]:
    """Search one leg of a whole cell from an already-computed query representation.

    The one implementation of the per-cell merge, extracted so that the two callers cannot
    drift: :func:`query_lang_leg` (which encodes the query itself) and the resident service in
    ``devkit.rsvc`` (which encodes in the parent, on the GPU, and ships the ``rep`` to a
    forked CPU worker that must not re-encode). Before this existed the service carried its own
    copy whose equivalence to production was asserted only in a docstring, so every latency
    number measured through the service described a merge nothing else executed.

    Semantics are exactly what ``query_lang_leg`` documents: per-part top-k, a raw-score merge
    across parts (they share one score space), and a total ``(-score, passage_id)`` order. The
    global top-k is a subset of the union of the per-part top-ks, so k per part is sufficient.
    """
    parts = handle.parts
    if not parts:
        return []
    out = _fan_parts(
        parts, lambda part: search_with_rep(part, leg, rep, top_k), workers, omp=omp
    )
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out[: int(top_k)]


# --------------------------------------------------------------------------- #
# Query: the minimal per-leg primitive the retrieval layer reuses rather than re-implements.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LegHandle:
    """An opened index shard: the three legs' readers plus the shard's id map.

    Holds no passage text and cannot acquire any, ``idmap.parquet`` has no text column, so
    ``passage_ids``/``document_ids`` are all a search can resolve to. Readers and the id map
    are loaded lazily, per leg, so opening a handle to query one leg never pays for three.
    """

    shard_dir: Path
    ctx: IndexCtx
    legs: dict[str, Leg]
    _readers: dict[str, Any] = field(default_factory=dict)
    _idmaps: dict[str, list[str]] = field(default_factory=dict)

    def reader(self, leg: str) -> Any:
        if leg not in self._readers:
            self._readers[leg] = self.leg_impl(leg).open(self.shard_dir / leg, self.ctx)
        return self._readers[leg]

    def passage_ids(self, leg: str) -> list[str]:
        """Ordinal-indexed ``passage_id``s for one leg, from its own ``idmap.parquet``."""
        if leg not in self._idmaps:
            self._idmaps[leg] = [
                row["passage_id"]
                for row in iter_parquet(
                    self.shard_dir / leg / IDMAP_FILENAME, columns=["ordinal", "passage_id"]
                )
            ]
        return self._idmaps[leg]

    def leg_impl(self, leg: str) -> Leg:
        """The leg implementation for ``leg`` (``KeyError`` if this index has no such leg)."""
        try:
            return self.legs[leg]
        except KeyError:
            raise KeyError(f"unknown leg {leg!r}; this index carries {sorted(self.legs)}") from None


def open_shard(shard_dir: str | os.PathLike[str], ctx: IndexCtx) -> LegHandle:
    """Open one published part (``.../<lang>/part-000NN``) for querying.

    A part is a shard, not a language: searching one part of a multi-part cell returns a
    full-looking list over a fraction of the passages. Callers that mean "this language"
    call :func:`open_lang` on the parent directory, which reads the census and fans.
    """
    path = Path(shard_dir)
    legs = {leg.name: leg for leg in ctx.legs}
    for name in legs:
        if not is_done(path / name):
            raise IndexIntegrityError(
                f"{path}: leg {name!r} is not published (no _SUCCESS marker), querying a "
                "partially built shard would silently search fewer passages"
            )
    return LegHandle(shard_dir=path, ctx=ctx, legs=legs)


def read_shard_parts(lang_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """One ``(variant, source_lang)`` cell's part census, cross-checked against the disk.

    The shard-level mirror of :func:`_read_parts`, with the same failure semantics and for
    the same reason: a reader must never infer a part count. A fan over 9 of 10 parts
    returns a full-looking, silently truncated result set, and nothing downstream, not the
    id maps (each is correct for its own part), not the manifest, not a score, could tell.

    The census is not one writer's claim but P independent witnesses: each part writes its
    own :data:`_PART_FILENAME` recording the whole plan it was built under, and this
    function requires them to
    (a) exist for every ``part-*`` directory on disk (one without it is a crashed build, not
    a part), (b) agree unanimously on ``parts`` and ``part_passages``, (c) sit in the
    directory their own index names, and (d) cover ``0..parts-1`` with exactly ``parts``
    directories. Any disagreement raises :class:`IndexIntegrityError`; there is no
    "search what we found" fallback.
    """
    path = Path(lang_dir)
    dirs = sorted(p for p in path.glob(INDEX_PART_GLOB) if p.is_dir())
    if not dirs:
        raise IndexIntegrityError(
            f"{path}: no {INDEX_PART_GLOB} directories, a published index cell holds at "
            "least one part, and a reader never falls back to treating the cell directory "
            "itself as a shard"
        )
    by_part: dict[int, tuple[Path, Mapping[str, Any]]] = {}
    for part_dir in dirs:
        census = part_dir / _PART_FILENAME
        if not census.exists():
            raise IndexIntegrityError(
                f"{part_dir}: no {_PART_FILENAME}, every complete part records the plan it "
                "was built under (as the last act of IndexAdapter.work), so a part "
                "directory without one is an abandoned attempt and this cell must not be "
                "searched until it is rebuilt"
            )
        record = read_jsonl(census)[0]
        index = int(record["part"])
        if part_dir.name != index_part_name(index):
            raise IndexIntegrityError(
                f"{part_dir}: {_PART_FILENAME} claims part {index}, i.e. "
                f"{index_part_name(index)!r}, a part in the wrong directory would be "
                "fanned over twice or not at all"
            )
        if index in by_part:  # pragma: no cover - unreachable while names are unique
            raise IndexIntegrityError(f"{path}: two directories claim part {index}")
        by_part[index] = (part_dir, record)
    claimed = {int(r["parts"]) for _, r in by_part.values()}
    sizes = {int(r["part_passages"]) for _, r in by_part.values()}
    if len(claimed) != 1 or len(sizes) != 1:
        raise IndexIntegrityError(
            f"{path}: the parts disagree about the cell's plan (parts={sorted(claimed)}, "
            f"part_passages={sorted(sizes)}), they were not built from one shard plan, so "
            "which of them together form the language is unknowable"
        )
    parts = claimed.pop()
    if sorted(by_part) != list(range(parts)):
        raise IndexIntegrityError(
            f"{path}: parts on disk are {sorted(by_part)} but the census claims {parts} "
            f"part(s) 0..{parts - 1}, this cell would be searched short (a fan over a "
            "subset returns a full-looking but silently truncated result list)"
        )
    return {
        "parts": parts,
        "part_passages": sizes.pop(),
        "dirs": [by_part[i][0].name for i in range(parts)],
        "passages": [int(by_part[i][1].get("passages") or 0) for i in range(parts)],
        "source_lang": str(by_part[0][1].get("source_lang") or ""),
        "variant": by_part[0][1].get("variant"),
    }


@dataclass(slots=True)
class LangHandle:
    """Every part of one ``(variant, source_lang)`` cell: the unit a query really means.

    Built from the census (:func:`read_shard_parts`), never from a glob, so "this handle
    covers the whole language" is checked rather than assumed. Holds no text, for the same
    structural reason a :class:`LegHandle` does not.
    """

    lang_dir: Path
    ctx: IndexCtx
    parts: tuple[LegHandle, ...]
    census: dict[str, Any]


def open_lang(lang_dir: str | os.PathLike[str], ctx: IndexCtx) -> LangHandle:
    """Open all parts of one ``(variant, source_lang)`` cell, in part order."""
    path = Path(lang_dir)
    census = read_shard_parts(path)
    return LangHandle(
        lang_dir=path,
        ctx=ctx,
        parts=tuple(open_shard(path / name, ctx) for name in census["dirs"]),
        census=census,
    )


def _resolve_ordinals(
    handle: LegHandle, leg: str, hits: Sequence[tuple[int, float]]
) -> list[tuple[str, float]]:
    """``(ordinal, score)`` -> ``(passage_id, score)`` through this part's own id map.

    Ordinals are part-local (each part is an independent index over its own ordinal space),
    so they are only meaningful against the handle they came from, which is exactly why
    this takes the handle and why the cross-part merge below keys on ``passage_id``.
    """
    ids = handle.passage_ids(leg)
    return [(ids[o], float(s)) for o, s in hits if 0 <= o < len(ids)]


def query_leg(
    handle: LegHandle, leg: str, query: str, top_k: int
) -> list[tuple[str, float]]:
    """Search one leg of one part: ``[(passage_id, score)]``, best first. No text.

    The per-part primitive. :func:`query_lang_leg` is the one a caller that means "search
    this language" wants; this one exists because the fan has to be built out of something,
    and because a single-part cell's diagnostics read better without it.

    The frozen contract, and the primitive the query side's fused ``retrieve``/``service.search``
    composes (RRF over the legs + rerank) rather than re-implementing: a second raw-leg query
    path is exactly how the search/display decoupling would drift. ``rank`` and
    ``search_index`` are the caller's to add, this returns the two fields the engine
    actually produced.
    """
    impl = handle.leg_impl(leg)
    return search_with_rep(handle, leg, impl.encode_query(handle.ctx, query), top_k)


def search_with_rep(
    handle: LegHandle, leg: str, rep: Any, top_k: int
) -> list[tuple[str, float]]:
    """Search one leg of one part with an already-computed representation. No encode.

    :func:`query_leg` is ``encode_query(ctx, text)`` then this, and is literally written that
    way one function up, so this is the second half of the one query path, factored out,
    never a second disagreeing one. ``query_leg``'s signature and behaviour are unchanged
    (it remains the frozen contract); this is a sibling, not an edit.

    It exists because a caller may already hold the representation it wants to search with:
    the corpus-acceptance pass (:mod:`ragtime.preprocess.acceptance`) queries a leg with a
    passage's own stored vector, read back through
    :func:`~ragtime.preprocess.vectors.stream_part`. Re-encoding that passage's text to get
    the same probe would test the encoder (and, on a CPU node, a different device than the
    one that wrote the vectors) instead of the index, the artifact under test.

    Returns ``[(passage_id, score)]``, best first, resolved through this part's own id map.
    No text, for the same structural reason :class:`LegHandle` holds none.
    """
    impl = handle.leg_impl(leg)
    return _resolve_ordinals(
        handle, leg, impl.search(handle.reader(leg), handle.ctx, rep, top_k)
    )


def query_lang_leg(
    handle: LangHandle, leg: str, query: str, top_k: int
) -> list[tuple[str, float]]:
    """Search one leg of a whole language cell, fanning over its parts. No text.

    The merge is a score sort, not RRF, and the two levels of this build differ here.
    Parts of one cell are the same encoder over the same leg, differing only in which
    passages they hold: a dense inner product, a Seismic dot product and a PLAID MaxSim are
    each a function of the ``(query, passage)`` pair alone, with no index-global
    normalization anywhere in their return paths, so two parts of one cell report scores in
    the same space. RRF over parts would discard exactly the magnitude that makes them
    comparable, a weak part-2 hit would tie with a strong part-1 hit purely because each
    topped its own list, and would additionally make the result depend on how the corpus
    happened to be cut, which is a scheduling choice. Rank fusion stays where the score
    spaces genuinely differ: Across legs and across languages (``common.fusion``'s ``rrf``),
    never here. This is the same argument :meth:`PlaidLateInteractionLeg.search` makes for
    its own PLAID-part fan, one level down, and the same mechanism.

    The sort is total and deterministic, ``(-score, passage_id)``. The tie-break is the
    passage ID and not the ordinal (which is what the PLAID fan uses) because an ordinal is
    unique only within a part, so ordinals would not order the fan at all; a ``passage_id``
    is globally unique, so two runs return the identical list.

    Each part is asked for its own top-``k``: the global top-``k`` is a subset of the union
    of the per-part top-``k``s, so ``k`` per part is exactly sufficient and never
    approximate. The query is encoded once and the representation reused across parts,
    faster, and it removes any possibility of a per-part query representation drifting.

    The fan is concurrent, and that changes nothing about what it returns. A cell is 14-23
    parts on the shipped geometry, so the sequential loop this replaced paid 14-23 engine
    round-trips end to end for every (leg, cell, query). The gain is leg-dependent, and
    :data:`QUERY_WORKERS_CAP` explains why: it is worth having on dense and worth nothing on
    sparse. Width comes from ``execution.query_part_workers``
    (:func:`resolve_query_workers`), never from a literal, and the results are collected in
    submission order by :func:`_fan_parts`, whose docstring explains why completion order
    would be a tie-break hazard one level down.

    Readers and id maps are warmed sequentially first, before the pool exists. Each part owns
    its own lazily-populated caches so a fan could safely fill them, but doing it here keeps
    the concurrent region to native search alone: no two threads open an engine (or a Parquet
    reader) at once, and the one-off open cost is paid exactly where it was paid before.
    """
    if not handle.parts:  # pragma: no cover - read_shard_parts rejects an empty census
        return []
    impl = handle.parts[0].leg_impl(leg)
    rep = impl.encode_query(handle.ctx, query)
    for part in handle.parts:
        part.reader(leg)
        part.passage_ids(leg)
    workers = resolve_query_workers(handle.ctx.query_workers)
    started = time.perf_counter()
    # omp=1 when fanning, and only then. `concurrency.py`'s sweep measured `OMP_NUM_THREADS=1`
    # + a fan as the best configuration at the part level (it beat OMP=6), and `_fan_parts`
    # documents why the cap has to be applied on the worker rather than here. At `workers <= 1`
    # the cap is left alone: a sequential cell should use FAISS's own intra-op width, and
    # pinning it to 1 there would be a silent slowdown, not a fix.
    out = _fan_parts(
        handle.parts,
        lambda part: search_with_rep(part, leg, rep, top_k),
        workers,
        omp=1 if workers > 1 else 0,
    )
    # The fan's own wall time, at DEBUG so a production query path stays quiet: query latency
    # is the risk this concurrency exists to manage, and a claim about it should come from
    # the code that did the work rather than from a wrapper's stopwatch.
    _log.debug(
        "preprocess.index.query.fan",
        leg=leg,
        parts=len(handle.parts),
        workers=workers,
        hits=len(out),
        seconds=round(time.perf_counter() - started, 6),
    )
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out[: int(top_k)]


# --------------------------------------------------------------------------- #
# Shard specs + the worker's resident context.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class IndexShardSpec:
    """One claimable unit: ``(variant, source_lang, part)``.

    ``variant is None`` names the shared English cell, the one built once and referenced by
    every variant's manifest (see property 4 in the module docstring).

    ``part``/``parts`` carry the cell's whole plan, not just this unit's index, so a shard is
    self-describing: the worker can write the census witness (:data:`_PART_FILENAME`) and
    assert the plan it was seeded under still matches the passage table it reads, without a
    second lookup. They default to the one-part cell so a hand-built spec is still legal and
    obvious, while :meth:`from_payload` requires both, a queue payload minted before the
    part axis existed must fail loudly, not resolve to "part 0 of 1" and index a prefix of
    its language.
    """

    variant: str | None
    source_lang: str
    part: int = 0
    parts: int = 1

    @property
    def name(self) -> str:
        """``index_<variant|_shared>_<lang>_p<NNNNN>``: the queue filename."""
        return (
            f"index_{self.variant or '_shared'}_{self.source_lang}"
            f"_p{int(self.part):05d}"
        )

    def payload(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "source_lang": self.source_lang,
            "part": int(self.part),
            "parts": int(self.parts),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IndexShardSpec:
        variant = payload.get("variant")
        missing = [key for key in ("part", "parts") if payload.get(key) is None]
        if missing:
            raise IndexIntegrityError(
                f"index shard payload {dict(payload)!r} carries no {missing}, the shard "
                "grain is (variant, source_lang, part), so a payload without them predates "
                "the part axis and would silently build one part of a multi-part cell under "
                "that cell's whole id set. Re-seed the queue."
            )
        return cls(
            variant=None if variant is None else str(variant),
            source_lang=str(payload["source_lang"]),
            part=int(payload["part"]),
            parts=int(payload["parts"]),
        )

    @property
    def text_rendering(self) -> str:
        """Which rendering's text this shard encodes.

        The shared English unit encodes the native composition: English is identity
        pass-through in both MT arms, so it is the same text in every rendering, and the
        native slice is the corpus's own bytes, which is the one composition that is not a
        join over re-assembled segments.
        """
        return self.variant or "original"


@dataclass(slots=True)
class IndexCtx:
    """The worker's resident context: Three encoders + the recipe + the input paths.

    All three models come from the ``serving`` singleton registry, this stage
    instantiates none of them. They are resident simultaneously (dense ~2 GB, MILCO ~1.3 GB,
    MTD ~1.1 GB), which is exactly why each leg's output is published independently.
    """

    dense: Any
    milco: Any
    mtd: Any
    opts: IndexBuildOptions
    legs: tuple[Leg, ...]
    layout: Layout
    recon_hash: str
    #: The packing this index is built from. Both it and ``recon_hash`` are needed to name
    #: the passages table (``recon12`` = which sentences, ``pack12`` = how they are grouped).
    pack_hash: str
    idx_hash: str
    tier: Tier
    #: The raw ``execution.query_part_workers`` value, forwarded from the config uninterpreted
    #: (``None`` = absent). It rides on the context rather than on a ``query_lang_leg``
    #: parameter because that signature is the frozen query contract, pinned from the
    #: consumer side by ``tests/retrieval/test_retrieval_small.py``; and it stays raw because
    #: :func:`resolve_query_workers` is the single owner of what a value means. Query-side
    #: only: the build half never reads it, which is why it is not on
    #: :class:`IndexBuildOptions` (that object is the hashed recipe; a width is not).
    query_workers: Any = None


@dataclass(slots=True)
class _Shard:
    """One claimable work unit, in ``saturate``'s ShardSpec shape."""

    name: str
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ShardItem:
    """One passage of a shard: its ids, its composed text, and its shared counts.

    ``token_count``/``is_oversized`` come from the one text-free passage table, so they are
    rendering-invariant, which is what lets ``token_count`` pin batch composition
    identically across the three renderings (property 3).
    """

    passage_id: str
    document_id: str
    text: str
    token_count: int
    is_oversized: bool


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_payload(shard: Path) -> dict[str, Any]:
    return json.loads(shard.read_text(encoding="utf-8").strip())


def _dir_bytes(path: Path) -> int:
    """Total bytes under ``path`` (the leg's measured on-disk size, for the manifest)."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _publish_dir(tmp: Path, final: Path) -> Path:
    """Atomically publish a built leg directory, then mark it done.

    ``common.io``'s temp->rename + ``_SUCCESS`` contract, applied to a directory (a leg's
    engine files + its id map) instead of a single file, the one level finer this stage
    needs so a leg is either fully published or absent. An unmarked leftover directory from a
    crashed attempt is removed first: it is by definition incomplete, and renaming onto it
    would mix two builds' files.
    """
    marker = success_marker(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if marker.exists():
            return final  # already published, resume is a no-op
        shutil.rmtree(final)
    os.replace(tmp, final)
    marker.write_bytes(b"")
    return final


def _assert_window_covers_packing(cfg: RunConfig, opts: IndexBuildOptions, mtd: Any) -> None:
    """The encode window must cover the packed passage plus the document path's own overhead.

    This is the invariant that replaced MaxP windowing, stated where it can be enforced. The
    packer guarantees every passage is ``<= packing.pack_budget`` tokens in every rendering;
    that guarantee is worth exactly nothing if the encoder's window is smaller, because then
    every passage above the window is truncated again, and truncated per-language, since
    only the long (Chinese) passages reach the limit.

    The comparison is ``pack_budget + DOCUMENT_MARKER_TOKENS``, not ``pack_budget``. A
    pack budget of ``B`` means ``B - 2`` content tokens (the packer reserves the two sequence
    specials), and PyLate's document path prepends one more position, the ``[unused1]``
    marker, so a budget-filling passage occupies ``B + 1`` positions: 509 content tokens give
    512 positions, and at 510 content tokens exactly one is dropped. A naive
    ``window >= budget`` check passes at ``512 >= 512`` and misses that.

    The client is the primary source, not the recipe.
    :meth:`~ragtime.serving.late_interaction.MtdEncoder.effective_document_length` already
    implements the real precedence (config override -> the checkpoint's own metadata ->
    default), and a checkpoint whose metadata sets the window is exactly the case a
    recipe-only check would miss. Asking the client makes this check independent of
    :func:`_assert_clients_match_recipe` rather than co-dependent on it, the two then fail
    for different reasons instead of one silently covering the other. ``opts.document_length``
    is used only when the client cannot answer at all (a test double), so the check degrades
    to the declared number rather than to no check.

    Checked at bring-up, before any GPU work, and without loading the model: every shipped
    config declares ``document_length``, so the client answers from config alone.
    """
    budget = int(cfg.blocks.get("packing", {}).get("pack_budget") or 0)
    if not budget:
        return  # no packing block: nothing has been packed to a budget to compare against
    getter = getattr(mtd, "effective_document_length", None)
    effective = int(getter() or 0) if callable(getter) else 0
    if not effective:
        effective = int(opts.document_length)
    required = budget + DOCUMENT_MARKER_TOKENS
    if effective and effective < required:
        raise ValueError(
            f"the late-interaction encode window is {effective} but a passage packed to "
            f"packing.pack_budget={budget} occupies {required} positions on the document "
            f"path ({budget} + {DOCUMENT_MARKER_TOKENS} for the ColBERT document marker, "
            "measured): the passages that fill the budget would be truncated, "
            "per-language, which is the loss rendering-invariant packing exists to remove. "
            f"Lower packing.pack_budget to {effective - DOCUMENT_MARKER_TOKENS} or raise "
            "index_build.config.document_length, both are fairness-shared config edits, "
            "not code changes (and 512 is XLM-R's architectural ceiling, so it is the "
            "budget that has to move)."
        )


def _assert_encode_device(opts: IndexBuildOptions) -> str:
    """A declared ``cuda`` encode must actually have cuda: the sibling of
    :func:`_assert_assemble_device`, and the more dangerous of the two.

    All three encoders resolve their device as ``"cuda" if torch.cuda.is_available() else
    "cpu"`` (``serving.encoders`` / ``serving.late_interaction`` / ``serving.sparse_milco``),
    so a vectorize worker that lands on a GPU-less node does not fail, it encodes on CPU and
    publishes the vectors, under a :func:`leg_encode_hash` whose body says
    ``encode_device: "cuda"``. Those blocks then pass the census, get assembled, and nothing
    downstream can tell: if some cells encoded on GPU and others on CPU, a rank delta between
    renderings is measuring the scheduler, which is precisely the confound the pinned batch
    composition exists to eliminate. Unlike the assemble knob there is no measurement saying
    the two devices agree, the presumption runs the other way.

    Checked at bring-up, before any client is built and any shard is claimed, so the failure
    is a scheduling error rather than a half-built, wrong-device cell. The message names the
    worker's observed provenance rather than a declared template, because that is what
    identifies the launch: the three sbatch templates leave three distinguishable signatures
    in ``RAGTIME_GPU_MODEL`` (a real card name = ``workqueue_worker.sbatch``; the literal
    ``none`` = ``workqueue_worker_cpu_sif.sbatch``, the assemble/CPU+SIF one; absent =
    ``workqueue_worker_cpu.sbatch``, the host CPU one). A declared ``cpu`` returns
    immediately: there is no such thing as a node without a CPU, and importing torch to prove
    it would make a torch-free environment fail for nothing.
    """
    device = opts.encode_device
    if device != "cuda":
        return device
    launched = _launch_provenance()
    try:
        import torch
    except ImportError as exc:
        raise ConfigError(
            "index_build.config.encode_device='cuda' but torch is not importable on this "
            f"worker ({launched}), so the encode cannot run on the declared device. A "
            "torch-free environment means this shard was launched under the host CPU "
            "template (workqueue_worker_cpu.sbatch), which has neither the card nor the "
            "index extra: route the vectorize substage to workqueue_worker.sbatch."
        ) from exc
    if not torch.cuda.is_available():
        raise ConfigError(
            "index_build.config.encode_device='cuda' but this worker reports no CUDA device "
            f"({launched}). The encoders would silently fall back to CPU and publish the "
            "vectors anyway, under a leg_encode_hash that records 'cuda', so some cells of "
            "one corpus would be GPU-encoded and others CPU-encoded, and the cross-rendering "
            "comparison would be measuring the scheduler. Schedule vectorize where the "
            "declared device exists (workqueue_worker.sbatch, the GPU template), not the "
            "CPU+SIF assemble template, or declare encode_device: cpu, which is a hashed, "
            "fairness-shared edit that re-keys every vector cell."
        )
    return device


def _launch_provenance() -> str:
    """The worker's observed job/hardware facts, for a scheduling error message.

    Read through ``saturate.worker_provenance``, the one reader of those envs, so the
    string in the error is exactly what the shard's ``worker`` record would have carried.
    """
    facts = saturate.worker_provenance()
    return ", ".join(f"{k}={v}" for k, v in sorted(facts.items())) or "no SLURM/GPU env"


def _assert_assemble_device(opts: IndexBuildOptions) -> str:
    """A declared ``cuda`` assemble must actually have cuda: never an opportunistic fallback.

    ``assemble_device`` reaches :func:`_open_plaid`, and fast-plaid resolving a missing GPU by
    quietly running on CPU is exactly the silent per-rendering divergence the key exists to
    close: an artifact built on "whichever device the worker happened to have" carries a
    difference with no cause in the text.
    Checked at bring-up, before any shard is claimed, so the failure is a scheduling error
    rather than a half-built leg.

    The CPU default costs nothing here (there is no such thing as a node without a CPU), which
    is deliberate: the check fires only for a run that has declared the non-default, re-keyed
    GPU alternative. An absent torch is a failure too on that path, not an unknown, a worker
    that cannot import torch certainly cannot assemble on cuda.
    """
    device = opts.assemble_device
    if device != "cuda":
        return device
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the real index worker always has torch
        raise ConfigError(
            "index_build.config.assemble_device='cuda' but torch is not importable on this "
            "worker, so the assemble cannot run on the declared device. CPU assembly is a "
            "different artifact, not a fallback, declare assemble_device: cpu "
            "(a hashed, fairness-shared edit) or schedule this stage on a GPU partition."
        ) from exc
    if not torch.cuda.is_available():
        raise ConfigError(
            "index_build.config.assemble_device='cuda' but this worker reports no CUDA "
            "device. Letting fast-plaid fall back to CPU would publish bytes that differ from "
            "the GPU-assembled siblings for no reason in the text (compute_kmeans: identical "
            "3/3 on cpu, 3 distinct on cuda). Declare assemble_device: cpu, the "
            "measured default, since both expensive assemble steps are CPU-bound, or "
            "schedule this stage where the declared device exists."
        )
    return device


def _assert_clients_match_recipe(clients: Any, opts: IndexBuildOptions) -> None:
    """The encoders that run must be the recipe this stage records. Abort if they differ.

    The manifest, ``leg_config_hash`` and the index path level are all derived from
    :class:`IndexBuildOptions`, while the encoders come from ``serving.registry``, two
    readers of one hashed block. When they disagree the artifact is a false record: that is
    not hypothetical, it is exactly how the dense leg shipped encoding with ``retrieval.dense``
    (an empty string, or Qwen3-Embedding) under a manifest naming BGE-M3, and how
    ``spine_maxp`` was hashed into every config while no code applied it (it has since been
    deleted from both). Checking at bring-up
    costs one comparison per worker and makes "declared but not wired" unreachable rather than
    merely reviewed for.

    Only declared identities are compared: a recipe leaf left empty means "the client's own
    documented default", so an absent ``sparse_model`` is not a mismatch, a wrong one is.
    """
    problems: list[str] = []
    dense_model = getattr(getattr(clients, "index_dense", None), "model", None)
    if dense_model != opts.dense_model:
        problems.append(
            f"dense: recipe says {opts.dense_model!r}, client is {dense_model!r}"
        )
    milco = getattr(clients, "milco", None)
    sparse_model = getattr(milco, "model", None)
    if opts.sparse_model and sparse_model != opts.sparse_model:
        problems.append(f"sparse: recipe says {opts.sparse_model!r}, client is {sparse_model!r}")
    # Compared unconditionally (unlike the identity leaves above): an encode window is never
    # "the client's own default" here, because this stage records a number for it. A recipe
    # saying 8192 over a client capped at 512 would publish a manifest describing an index in
    # which 2.36 % of Chinese passages are silently prefixes of themselves.
    sparse_window = int(getattr(milco, "max_length", 0))
    if sparse_window != opts.sparse_max_length:
        problems.append(
            f"sparse_max_length: recipe says {opts.sparse_max_length}, client is {sparse_window}"
        )
    mtd = getattr(clients, "mtd_colbert", None)
    checkpoint = getattr(mtd, "checkpoint", None)
    if opts.spine_model and checkpoint != opts.spine_model:
        problems.append(
            f"late_interaction: recipe says {opts.spine_model!r}, client is {checkpoint!r}"
        )
    window = int(getattr(mtd, "document_length", 0))
    if window != opts.document_length:
        problems.append(
            f"document_length: recipe says {opts.document_length}, client is {window}"
        )
    if problems:
        raise ValueError(
            "the index encoders do not match the hashed index_build recipe they will be "
            "recorded as (" + "; ".join(problems) + "), the manifest and leg_config_hash "
            "would describe an index that was never built"
        )


# --------------------------------------------------------------------------- #
# The one saturate.StageAdapter for the whole stage.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class IndexAdapter:
    """Indexation as one ``saturate.StageAdapter``, sharded per ``(variant, source_lang)``.

    One adapter for the whole stage, never one per leg, never a bespoke driver: the legs are
    built inside ``work``, which is what makes "the same method on all three renderings"
    a property of one code path instead of a convention across three.

    ``legs`` is injectable so a hermetic test can drive the adapter without the multi-GB
    checkpoints, but the set is checked: it must be exactly :data:`LEGS`, in order. That
    rejects both a 4th leg (no BM25 can be re-added quietly) and a 2-leg build (no leg can be
    present on one rendering and absent from another).
    """

    recon_hash: str
    pack_hash: str
    idx_hash: str
    base: str | None = None
    #: GPU: the dense/sparse/late-interaction encoders all run a forward pass.
    template: str = "workqueue_worker.sbatch"
    legs: tuple[Leg, ...] = field(default_factory=default_legs)
    stats: Statistics = field(default_factory=Statistics)
    stage: str = field(init=False)
    #: ``(source_lang, part, part_size) -> the id set that part must cover``. Keyed by the
    #: whole triple, not by language: a part's expected set is a slice of the pinned order,
    #: so a cache keyed by language alone would answer a question nobody asked.
    _expected: dict[tuple[str, int, int], frozenset[str]] = field(
        default_factory=dict, init=False
    )
    #: ``source_lang -> passage count`` from the one text-free passage table: the input to
    #: the shard plan. Cached per adapter because ``shards``/``merge`` both re-derive the
    #: plan and neither may get a different answer than the other.
    _counts: dict[str, dict[str, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        names = tuple(leg.name for leg in self.legs)
        if names != LEGS:
            raise ValueError(
                f"an index build has exactly these legs, in this order: {LEGS}; got {names}. "
                "A leg present on one rendering and absent from another is a failure, not an "
                "exception: the leg set is fixed, so no exception is admitted."
            )
        # Both hashes in the queue name: an edit to the corpus OR to the build recipe
        # resolves to a fresh queue subtree, so a stale done/ can never be reused.
        self.stage = f"index_{self.idx_hash[:_HASH_DIRLEN]}_{self.recon_hash[:_HASH_DIRLEN]}"

    @classmethod
    def for_config(
        cls,
        cfg: RunConfig,
        *,
        base: str | os.PathLike[str] | None = None,
        legs: tuple[Leg, ...] | None = None,
        stats: Statistics | None = None,
    ) -> IndexAdapter:
        """Build the adapter for ``cfg``: the only sanctioned construction path."""
        return cls(
            recon_hash=reconcile_hash(cfg),
            pack_hash=packing_hash(cfg),
            idx_hash=index_hash(cfg),
            base=None if base is None else str(base),
            legs=default_legs() if legs is None else legs,
            stats=stats if stats is not None else Statistics(),
        )

    # -- paths ------------------------------------------------------------- #
    def _layout(self, cfg: RunConfig) -> Layout:
        root = self.base if self.base is not None else _DEFAULT_BASE
        return Layout(
            run_dir=root,
            base=root,
            family=run_family(cfg),
            chunker_hash=all_hashes(cfg)["chunker"],
        )

    def index_dir(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).index_dir(self.recon_hash, self.idx_hash)

    def lang_dir(self, cfg: RunConfig, variant: str | None, source_lang: str) -> Path:
        """The ``(variant, source_lang)`` cell: the level a query fans over."""
        return self._layout(cfg).index_lang_dir(
            self.recon_hash, self.idx_hash, variant, source_lang
        )

    def shard_dir(self, cfg: RunConfig, spec: IndexShardSpec) -> Path:
        return self._layout(cfg).index_shard_dir(
            self.recon_hash, self.idx_hash, spec.variant, spec.source_lang, part=spec.part
        )

    def manifest_path(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).index_manifest_path(self.recon_hash, self.idx_hash)

    def passages_path(self, cfg: RunConfig) -> Path:
        """The packed passages this index is built from: ``recon12`` and ``pack12``.

        Both hashes are needed to name a passage table: ``recon12`` says which sentences
        exist, ``pack12`` says how they were grouped, and two packings of one inventory
        legitimately coexist under the same ``final/<recon12>/``. Passing only ``recon12``
        would name a directory, not a table.
        """
        return self._layout(cfg).final_passages_path(self.recon_hash, self.pack_hash)

    # -- seed --------------------------------------------------------------- #
    def shards(self, cfg: RunConfig) -> Iterator[_Shard]:
        """The claimable units: ``variants x non-English languages x parts`` + shared English.

        The languages come from ``cfg.languages`` (config-driven, never a hardcoded set);
        only the identity of the shared one is structural (:data:`SHARED_LANG`), and the
        number of PARTS per language comes from the corpus itself (see
        :meth:`shard_specs`).
        """
        for spec in self.shard_specs(cfg):
            yield _Shard(name=spec.name, payload=spec.payload())

    def lang_counts(self, cfg: RunConfig) -> dict[str, int]:
        """``source_lang -> passage count``, from the one text-free passage table.

        Column-pruned to ``lang`` alone (measured 82x for a single-column read) and cached:
        this is a count, so nothing is materialized per row. It is read from the corpus
        rather than configured because the shard plan must be a function of the corpus and
        the part size and nothing else, a declared count could go stale against the table
        and would then seed a plan that indexes a prefix of a language.
        """
        path = str(self.passages_path(cfg))
        if path not in self._counts:
            counts: dict[str, int] = {}
            for batch in iter_parquet_batches(path, columns=["lang"]):
                for row in batch:
                    lang = str(row["lang"])
                    counts[lang] = counts.get(lang, 0) + 1
            self._counts[path] = counts
        return self._counts[path]

    def parts_for(self, cfg: RunConfig, source_lang: str) -> int:
        """How many parts ``source_lang`` is cut into: ``ceil(passages / part size)``.

        Never zero: a language with no passages is one empty part, because an empty shard is
        a legal artifact (the dense/late-interaction writers both build one) and a language
        that silently contributed no shard at all would be invisible to ``merge``'s coverage
        assertion in exactly the way it must not be.
        """
        size = max(1, int(index_build_options(cfg).plaid_part_passages))
        return max(1, -(-int(self.lang_counts(cfg).get(source_lang, 0)) // size))

    def shard_specs(self, cfg: RunConfig) -> list[IndexShardSpec]:
        """The shard plan as specs (what ``shards`` yields, and what ``merge`` re-derives).

        ``len(RENDERINGS) * sum(parts of each non-English language) + parts(en)`` specs, 27
        for this corpus at the shipped part size, from 10 before the part axis. Deriving it
        here, from the table, rather than recording it anywhere means ``shards`` (seed) and
        ``merge`` (publish) cannot disagree about how many units the cell has.
        """
        langs = sorted(str(x) for x in cfg.languages if str(x) != SHARED_LANG)
        specs = [
            IndexShardSpec(variant=variant, source_lang=lang, part=part, parts=parts)
            for variant in RENDERINGS
            for lang in langs
            for parts in (self.parts_for(cfg, lang),)
            for part in range(parts)
        ]
        shared_parts = self.parts_for(cfg, SHARED_LANG)
        specs.extend(
            IndexShardSpec(
                variant=None, source_lang=SHARED_LANG, part=part, parts=shared_parts
            )
            for part in range(shared_parts)
        )
        return specs

    # -- bringup ------------------------------------------------------------ #
    def bringup(self, cfg: RunConfig) -> IndexCtx:
        """Stand up the three encoders once per worker, from the serving registry.

        The import is function-local so importing this module stays CUDA-free, and the
        clients are the registry's singletons, no stage instantiates a model.
        Batch shape comes from the hashed ``index_build.config`` block, not from ``execution``:
        which passages co-batch changes the vectors (60.97 % bitwise-identical when the same
        shard is batched differently), and the three per-rendering builds must therefore share
        one composition, a non-shared knob would let two family members build at different
        batch shapes while resolving to the same ``index_dir``.
        """
        from ragtime.serving.registry import build_clients

        clients = build_clients(cfg)
        opts = index_build_options(cfg)
        _assert_clients_match_recipe(clients, opts)
        _assert_assemble_device(opts)
        _assert_window_covers_packing(cfg, opts, getattr(clients, "mtd_colbert", None))
        return IndexCtx(
            # ``index_dense``, not ``embedder``: the dense leg's identity is the hashed
            # ``index_build.config.dense_model`` this stage records in the manifest and folds
            # into ``leg_config_hash``. ``embedder`` is built from the query-time
            # ``retrieval`` block, which no ``e2e-*`` config carries: taking it made the
            # manifest a false record of what encoded the vectors.
            dense=clients.index_dense,
            milco=clients.milco,
            mtd=clients.mtd_colbert,
            opts=opts,
            legs=self.legs,
            layout=self._layout(cfg),
            recon_hash=self.recon_hash,
            pack_hash=self.pack_hash,
            idx_hash=self.idx_hash,
            tier=Tier(
                token_budget=opts.encode_batch_token_budget,
                max_items=opts.encode_max_items,
            ),
        )

    def ctx_for(self, cfg: RunConfig, ctx: IndexCtx | None = None) -> IndexCtx:
        """``ctx`` or a freshly brought-up one: used by ``validate``/``merge``/queries."""
        return ctx if ctx is not None else self.bringup(cfg)

    # -- work --------------------------------------------------------------- #
    def part_bounds(
        self, passages_path: Path, spec: IndexShardSpec, part_size: int
    ) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
        """This part's half-open ``(token_count, passage_id)`` key range.

        The cut is ordinal arithmetic in the pinned order, part ``p`` is ordinals
        ``[p*N, (p+1)*N)``, but it is applied as a key range so nothing but two keys has to
        be held while the (expensive, joining) passage stream runs. The two are equivalent
        because the order is total (``passage_id`` is unique), and the key form is what makes
        part membership a function of the shared, text-free table alone: both keys come from
        ``final/<recon12>/passages/<pack12>/``, which has no ``variant`` column, so the three
        renderings cut at the identical places by construction.

        The ordering pass materializes one ``(token_count, passage_id)`` pair per passage of
        this language (~350 MB for English) and drops it before returning, deliberately
        transient, against a build whose steady state is tens of GiB.

        Raises if the table no longer implies the part count this shard was seeded under: a
        re-packed corpus under the same ``pack12`` is impossible, but a plan/table mismatch
        would silently index a prefix, which is the one failure mode this axis introduces.
        """
        keys = _ordered_lang_keys(passages_path, spec.source_lang, spec, part_size)
        low = spec.part * part_size
        high = low + part_size
        return (
            keys[low] if low < len(keys) else None,
            keys[high] if high < len(keys) else None,
        )

    def load_shard(self, ctx: IndexCtx, spec: IndexShardSpec) -> list[_ShardItem]:
        """This part's passages, text composed by the shared composer, in pinned order.

        Streams ``common.passage_store.iter_final_passages``, the same document-ordered
        co-walk (and therefore the same ``compose_passage_text``) the passage store itself
        uses, so the native-slice-vs-translated-join rule exists in exactly one place, and
        keeps only this ``source_lang``'s rows whose ``(token_count, passage_id)`` key falls
        in this part's range (:meth:`part_bounds`). The returned list is sorted by that same
        key: a total order over rendering-invariant keys, so the buckets built from it are
        identical across the three renderings and reproducible on a re-run (property 3), and
        the part is exactly the ordinal slice of the order the un-split shard would have had.

        For the shared English unit every rendering is still composed, not because they can
        differ (they cannot: English has no stored translation, so all three resolve to the
        same native slice) but so that :data:`STAT_EN_DIVERGENCE` stays a live corpus-wide
        canary for a stray identity row rather than a claim nobody re-checks.
        """
        rendering = spec.text_rendering
        shared = spec.variant is None
        renderings = ("original", *_TRANSLATED) if shared else ("original", rendering)
        part_size = max(1, int(ctx.opts.plaid_part_passages))
        low, high = self.part_bounds(
            ctx.layout.final_passages_path(ctx.recon_hash, ctx.pack_hash), spec, part_size
        )
        items: list[_ShardItem] = []
        for record in iter_final_passages(
            ctx.layout,
            ctx.recon_hash,
            pack_hash=ctx.pack_hash,
            renderings=tuple(dict.fromkeys(renderings)),
        ):
            if record["lang"] != spec.source_lang:
                continue
            key = (int(record["token_count"] or 0), str(record["passage_id"]))
            if low is None or key < low or (high is not None and key >= high):
                continue
            text = record[rendering]
            if shared:
                for other in _TRANSLATED:
                    if record.get(other) != text:
                        self.stats.emit(
                            STAT_EN_DIVERGENCE, lang=spec.source_lang, variant=other
                        )
            if record["is_oversized"]:
                self.stats.emit(STAT_OVERSIZED, lang=spec.source_lang, variant=rendering)
            items.append(
                _ShardItem(
                    passage_id=record["passage_id"],
                    document_id=record["document_id"],
                    text=text,
                    token_count=int(record["token_count"] or 0),
                    is_oversized=bool(record["is_oversized"]),
                )
            )
        items.sort(key=lambda it: (it.token_count, it.passage_id))
        self.stats.emit(
            STAT_PASSAGES, float(len(items)), lang=spec.source_lang, variant=rendering
        )
        return items

    def buckets(self, ctx: IndexCtx, items: Sequence[_ShardItem]) -> list[list[_ShardItem]]:
        """Length-bucket the shard through the shared ``serving.batching.Batcher``.

        ``items`` arrives in the pinned total order and ``Batcher``'s own sort is stable, so
        bucket membership is a deterministic function of the shard's passage set and the two
        ``execution`` bounds, never of arrival order, worker count or GPU type.
        """
        return Batcher(length_of=lambda it: max(1, it.token_count)).batch(list(items), ctx.tier)

    def work(self, ctx: IndexCtx, shard: Path) -> Path:
        """Build every not-yet-published leg of one shard; return the queue receipt.

        Order of operations is deliberate: legs already carrying a ``_SUCCESS`` marker are
        skipped before the passage text is read, so a retry after a leg-3 OOM re-streams
        nothing for legs 1-2. The remaining legs share one materialized item list (three
        encoders, one read), and each publishes independently.
        """
        spec = IndexShardSpec.from_payload(_read_payload(shard))
        shard_dir = self.shard_dir_from_ctx(ctx, spec)
        pending = [leg for leg in self.legs if not is_done(shard_dir / leg.name)]
        for leg in self.legs:
            if leg not in pending:
                self.stats.emit(
                    STAT_LEG_RESUMED, leg=leg.name, lang=spec.source_lang,
                    variant=spec.text_rendering,
                )
        report: dict[str, Any] = {
            "variant": spec.variant,
            "source_lang": spec.source_lang,
            "rendering": spec.text_rendering,
            "part": int(spec.part),
            "parts": int(spec.parts),
            "part_passages": int(ctx.opts.plaid_part_passages),
            "shard_dir": str(shard_dir),
            "lang_dir": str(shard_dir.parent),
            "index_hash": self.idx_hash,
            "reconcile_hash": self.recon_hash,
            "packing_hash": self.pack_hash,
            "created_at": _now(),
            # ``validate`` runs on a bare path, so the receipt names the table it must check
            # the id set against: it is then self-sufficient in any process, not dependent on
            # a same-process hand-off from ``work``.
            "passages_path": str(
                ctx.layout.final_passages_path(ctx.recon_hash, ctx.pack_hash)
            ),
            "legs": {},
            "worker": saturate.worker_provenance(),
        }
        items: list[_ShardItem] | None = None
        for leg in pending:
            if items is None:
                items = self.load_shard(ctx, spec)
            self._build_leg(ctx, spec, leg, items, shard_dir)
        for leg in self.legs:
            leg_dir = shard_dir / leg.name
            report["legs"][leg.name] = {
                "path": str(leg_dir),
                "config_hash": leg_config_hash(ctx.opts, leg.name),
                "passages": _idmap_rows(leg_dir),
                "bytes": _dir_bytes(leg_dir),
                # How many sub-indexes a reader must fan across (``None`` for the two legs
                # that have one). Read from the published leg, so a resumed leg reports the
                # parts it actually holds rather than the parts this process happened to
                # write. A manifest without this number would make a short fan undetectable.
                "plaid_parts": plaid_part_count(leg_dir) if is_done(leg_dir) else None,
            }
        # The census witness, written last: every leg above is published (or this line was
        # never reached), so a ``part-*`` directory carrying this file is a complete part and
        # one without it is an abandoned attempt. ``read_shard_parts`` reads exactly these.
        # ``passages`` is read back off the published id map rather than from ``items``, so a
        # resumed part (which re-streams nothing) records what it holds, not what this
        # process happened to encode.
        self._write_part_census(spec, shard_dir, int(ctx.opts.plaid_part_passages))
        out = saturate.shard_out_path(shard)
        write_jsonl(out, [report], skip_if_done=False)
        _log.info(
            "preprocess.index.shard.done",
            shard=shard.name,
            variant=spec.variant,
            lang=spec.source_lang,
            part=spec.part,
            parts=spec.parts,
            built=[leg.name for leg in pending],
            passages=0 if items is None else len(items),
        )
        return out

    def _write_part_census(
        self, spec: IndexShardSpec, shard_dir: Path, part_size: int
    ) -> None:
        """Record this part's whole plan inside its own directory (:data:`_PART_FILENAME`).

        Every part is an independent witness to the same plan, which is what lets
        :func:`read_shard_parts` establish the cell's part count without a single writer that
        knows the whole cell (there is none: parts are claimed by different workers on
        different machines) and without a glob (which cannot distinguish "9 parts" from "10
        parts, one lost").
        """
        write_jsonl(
            shard_dir / _PART_FILENAME,
            [
                {
                    "part": int(spec.part),
                    "parts": int(spec.parts),
                    "part_passages": int(part_size),
                    "passages": _idmap_rows(shard_dir / self.legs[0].name),
                    "source_lang": spec.source_lang,
                    "variant": spec.variant,
                    # No timestamp, deliberately (the PLAID ``parts.json`` carries none
                    # either): ``work`` re-writes this file on every re-entry, and a shard
                    # built twice must be byte-identical or the artifact tree stops being a
                    # checkpoint. The receipt beside it already dates the build.
                }
            ],
            skip_if_done=False,
        )

    def shard_dir_from_ctx(self, ctx: IndexCtx, spec: IndexShardSpec) -> Path:
        """The part's published directory, resolved through ``Layout`` (never hand-built)."""
        return ctx.layout.index_shard_dir(
            ctx.recon_hash, ctx.idx_hash, spec.variant, spec.source_lang, part=spec.part
        )

    def _build_leg(
        self,
        ctx: IndexCtx,
        spec: IndexShardSpec,
        leg: Leg,
        items: Sequence[_ShardItem],
        shard_dir: Path,
    ) -> Path:
        """Encode + write one leg of one shard, published atomically and independently."""
        final = shard_dir / leg.name
        tmp = shard_dir / f".{leg.name}.tmp-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        writer = leg.writer(tmp, ctx)
        # One bucketing pass: the ordinals written into the engine and the ordinals written
        # into the id map come from the same list, so the two can never disagree about which
        # vector belongs to which passage.
        batches = self.buckets(ctx, items)
        ordered = [it for bucket in batches for it in bucket]
        ordinal = 0
        for bucket in batches:
            texts = [it.text for it in bucket]
            self._emit_truncation(ctx, spec, leg.name, texts)
            reps = leg.encode_docs(ctx, texts)
            if len(reps) != len(bucket):
                raise IndexIntegrityError(
                    f"{leg.name}: encoder returned {len(reps)} representations for "
                    f"{len(bucket)} passages, a dropped passage would break the id-set "
                    "equality every rendering's index is asserted on"
                )
            writer.add(list(range(ordinal, ordinal + len(bucket))), reps)
            ordinal += len(bucket)
            self.stats.emit(
                STAT_ENCODED,
                float(len(bucket)),
                leg=leg.name,
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )
        writer.finish()
        empty = getattr(writer, "empty", 0)
        if empty:
            self.stats.emit(
                STAT_SPARSE_EMPTY, float(empty), leg=leg.name, lang=spec.source_lang
            )
        pruned = getattr(writer, "pruned", 0)
        if pruned:
            self.stats.emit(
                STAT_SPARSE_PRUNED, float(pruned), leg=leg.name, lang=spec.source_lang
            )
        # Same idiom as ``empty``: a writer that has parts reports them, one that has none
        # (dense/sparse) is not asked to pretend it does.
        parts = getattr(writer, "parts", 0)
        if parts:
            self.stats.emit(
                STAT_PLAID_PARTS,
                float(parts),
                leg=leg.name,
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )
            self.stats.emit(
                STAT_PLAID_PART_BYTES,
                float(getattr(writer, "buffered_bytes", 0)),
                leg=leg.name,
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )
        # The idmap is written inside the leg's temp dir, so it is published by the same
        # atomic rename: a leg can never be marked done with a missing or stale id map.
        _write_parquet_stream(
            tmp / IDMAP_FILENAME,
            (
                {
                    "ordinal": i,
                    "passage_id": it.passage_id,
                    "document_id": it.document_id,
                    "source_lang": spec.source_lang,
                }
                for i, it in enumerate(ordered)
            ),
            schema=idmap_arrow_schema(),
        )
        _publish_dir(tmp, final)
        self.stats.emit(
            STAT_LEG_BYTES,
            float(_dir_bytes(final)),
            leg=leg.name,
            lang=spec.source_lang,
            variant=spec.text_rendering,
        )
        return final

    def _emit_truncation(
        self, ctx: IndexCtx, spec: IndexShardSpec, leg: str, texts: Sequence[str]
    ) -> None:
        """Count the tokens this leg's encode drops, sliced by leg and language.

        Per-language truncation is a bias introduced by the retrieval method itself, and it
        has now bitten twice. On the late-interaction leg the checkpoint's 220-token window
        drops ~35 % of a 312-343-token Chinese passage, remedied upstream now
        (``preprocess.packing`` packs every passage to fit the window in every rendering),
        with the counter kept so the remedy is verified rather than assumed. On the
        sparse leg a plain module constant of 512, against MILCO's own 8192 ceiling, dropped
        the tail of 2.36 % of Chinese passage-renderings against 0.024 % of Spanish ones, and
        no gate could see it, because this method only ever looked at one leg. It now
        measures every leg whose client can
        measure itself; a client that cannot is skipped rather than reported as zero, which
        is why the counter's absence is never evidence of no loss.

        The dense leg is absent from :data:`_TRUNCATION_SOURCE` rather than
        silently skipped: ``serving.encoders.Encoder.embed`` passes no ``max_length`` at all,
        so its window is the checkpoint's own ``max_seq_length`` (8192 for BGE-M3) against a
        corpus maximum of 5,277 tokens, there is no cap of ours to measure.
        """
        source = _TRUNCATION_SOURCE.get(leg)
        if source is None:
            return
        client_attr, stat = source
        counter = getattr(getattr(ctx, client_attr, None), "truncated_tokens", None)
        if counter is None:
            return
        try:
            dropped = counter(list(texts))
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail a build
            _log.warning("preprocess.index.truncation_unmeasured", leg=leg, error=repr(exc))
            return
        total = float(sum(int(n) for n in dropped))
        if total:
            self.stats.emit(
                stat,
                total,
                leg=leg,
                lang=spec.source_lang,
                variant=spec.text_rendering,
            )

    # -- validate ----------------------------------------------------------- #
    def expected_ids(self, cfg: RunConfig, source_lang: str) -> frozenset[str]:
        """Every ``passage_id`` of ``source_lang``: the whole language, not one part.

        The one shared, text-free passage table is the authority, it carries no ``variant``
        column, so "the id set is identical across renderings" is not a three-way
        cross-derivation but a single equality against this set.
        ``merge``'s corpus-coverage assertion is about the whole language (the parts are
        unioned before it runs), so this takes no part argument; the per-part
        set is :meth:`expected_ids_at`'s ``part_size`` form, which ``validate`` uses.
        """
        return self.expected_ids_at(self.passages_path(cfg), source_lang)

    def expected_ids_at(
        self,
        passages_path: Path,
        source_lang: str,
        *,
        part: int = 0,
        parts: int = 1,
        part_size: int = 0,
    ) -> frozenset[str]:
        """:meth:`expected_ids` against an explicit table path (what ``validate`` has).

        ``part_size`` 0 means "the whole language is one part", which is what a caller
        asking about a language rather than a part means. The slice is taken from the same
        ``(token_count, passage_id)`` order the build cuts on (:meth:`part_bounds`), one
        rule, re-derived here rather than trusted from the artifact, which is the point
        of validating.
        """
        size = int(part_size) or 0
        key = (source_lang, int(part), size)
        if key not in self._expected:
            spec = IndexShardSpec(
                variant=None, source_lang=source_lang, part=int(part), parts=int(parts)
            )
            if not size:
                self._expected[key] = frozenset(
                    row["passage_id"]
                    for row in iter_parquet(passages_path, columns=["passage_id", "lang"])
                    if row["lang"] == source_lang
                )
            else:
                # The same ordinal slice of the same pinned order the build cut on: taken
                # directly rather than re-filtered by key range, so validation re-derives the
                # part from the table in one pass instead of trusting the artifact.
                keys = _ordered_lang_keys(Path(passages_path), source_lang, spec, size)
                low = int(part) * size
                self._expected[key] = frozenset(
                    passage for _, passage in keys[low : low + size]
                )
        return self._expected[key]

    def validate(self, path: Path) -> bool:
        """All three legs published and each leg's id set equal to this part's subset.

        Not a non-zero size: what is checked is that every leg covers exactly the passages
        this ``(variant, source_lang, part)`` unit should hold, nothing dropped by a
        truncated write, nothing duplicated, and (since the part axis) nothing shifted by a
        part boundary computed differently than the build computed it. A false return leaves
        the shard reclaimable.
        """
        if not (path.exists() and is_done(path)):
            return False
        report = json.loads(path.read_text(encoding="utf-8").strip())
        shard_dir = Path(report["shard_dir"])
        if not (shard_dir / _PART_FILENAME).exists():
            _log.warning(
                "preprocess.index.validate.part_census_missing", path=str(shard_dir)
            )
            return False
        passages_path = report.get("passages_path")
        expected = (
            self.expected_ids_at(
                Path(passages_path),
                str(report["source_lang"]),
                part=int(report.get("part") or 0),
                parts=int(report.get("parts") or 1),
                part_size=int(report.get("part_passages") or 0),
            )
            if passages_path
            else None
        )
        for leg in self.legs:
            leg_dir = shard_dir / leg.name
            if not is_done(leg_dir):
                _log.warning(
                    "preprocess.index.validate.leg_missing", leg=leg.name, path=str(leg_dir)
                )
                return False
            ids = _idmap_ids(leg_dir)
            if len(ids) != len(set(ids)):
                _log.warning("preprocess.index.validate.duplicate_ids", leg=leg.name)
                return False
            if expected is not None and frozenset(ids) != expected:
                _log.warning(
                    "preprocess.index.validate.id_set_mismatch",
                    leg=leg.name,
                    got=len(ids),
                    expected=len(expected),
                )
                return False
        return True

    # -- merge --------------------------------------------------------------- #
    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> Path:
        """Assert cross-variant id integrity, then publish one manifest. Or publish nothing.

        Four hard assertions, all aborting before any manifest is written:

        1. every ``(variant, source_lang, part)`` unit of the plan is present and carries all
           three legs, a leg on 2 of 3 renderings is a failure, not an exception;
        2. every cell's on-disk part census agrees with the plan (:func:`read_shard_parts`),
           so a cell that lost a part is a failed publication rather than a short fan;
        3. the parts of one cell are disjoint, their id sets sum to their union, so a part
           built under the wrong boundary (which would double-cover some ids and leave others
           uncovered) cannot hide inside a union that still looks complete;
        4. each variant's id set, unioned over its language cells (its own three plus the
           shared English one), equals the full id set of the packed passage table exactly,
           never fewer (a silent skip) and never a second, redundant English build.

        And one proof, not an assumption: the three renderings' part ``p`` of language ``l``
        must hold the same ids (compared by digest, :func:`_id_digest`). Part membership is
        identical across renderings by construction, the cut keys come from the one
        text-free table, and this is what makes that construction machine-checked, because
        it is the property the whole Task-2 comparison rests on.
        """
        reports = [json.loads(Path(p).read_text(encoding="utf-8").strip()) for p in shard_paths]
        by_unit = {
            (r["variant"], r["source_lang"], int(r.get("part") or 0)): r for r in reports
        }
        specs = self.shard_specs(cfg)
        for spec in specs:
            report = by_unit.get((spec.variant, spec.source_lang, spec.part))
            if report is None:
                raise IndexIntegrityError(
                    f"no shard output for ({spec.variant}, {spec.source_lang}, part "
                    f"{spec.part} of {spec.parts}), publishing a manifest now would name an "
                    "index that was never built"
                )
            if int(report.get("parts") or 0) != spec.parts:
                raise IndexIntegrityError(
                    f"({spec.variant}, {spec.source_lang}) part {spec.part} was built under a "
                    f"{report.get('parts')}-part plan but the corpus implies {spec.parts}, "
                    "the parts on disk are not one cell"
                )
            missing = [leg.name for leg in self.legs if leg.name not in report["legs"]]
            if missing:
                raise IndexIntegrityError(
                    f"({spec.variant}, {spec.source_lang}, part {spec.part}) is missing "
                    f"leg(s) {missing}: every rendering carries exactly {LEGS}, so a leg "
                    "present on one rendering and absent from another is a hard failure"
                )
        digests: dict[tuple[str, int], dict[str, str]] = {}
        variants: dict[str, Any] = {}
        for variant in RENDERINGS:
            langs = sorted({s.source_lang for s in specs if s.variant == variant})
            shards: dict[str, Any] = {}
            covered: set[str] = set()
            for source_lang in (*langs, SHARED_LANG):
                cell_variant = None if source_lang == SHARED_LANG else variant
                cell = [
                    by_unit[(cell_variant, source_lang, s.part)]
                    for s in specs
                    if s.variant == cell_variant and s.source_lang == source_lang
                ]
                entry, ids = self._merge_cell(source_lang, cell)
                for part, report in enumerate(cell):
                    digests.setdefault((source_lang, part), {})[variant] = entry[
                        "shard_parts"
                    ][part]["id_digest"]
                shards[source_lang] = entry
                covered |= ids
            self._assert_covers_corpus(cfg, variant, covered)
            variants[variant] = {"passages": len(covered), "shards": shards}
            self.stats.emit(STAT_ID_INTEGRITY, float(len(covered)), variant=variant)
        _assert_parts_identical_across_renderings(digests)
        shared = by_unit[(None, SHARED_LANG, 0)]
        manifest = {
            "index_hash": self.idx_hash,
            "reconcile_hash": self.recon_hash,
            # Which packing of that inventory these vectors encode. It is folded into
            # ``index_hash`` (so a re-pack cannot reuse this node), and recorded in full here
            # so the composite is auditable rather than opaque: the same discipline
            # ``final/manifest.json`` applies to ``recon12``'s three parents.
            "packing_hash": self.pack_hash,
            "legs": list(LEGS),
            "leg_config_hash": {
                leg: shared["legs"][leg]["config_hash"] for leg in (x.name for x in self.legs)
            },
            "provenance": _provenance(cfg, reports),
            "variants": variants,
            "created_at": _now(),
        }
        _log.info(
            "preprocess.index.shard_parts",
            stage=self.stage,
            parts={
                lang: variants[RENDERINGS[0]]["shards"][lang]["parts"]
                for lang in sorted(variants[RENDERINGS[0]]["shards"])
            },
            part_passages=index_build_options(cfg).plaid_part_passages,
        )
        out = self.manifest_path(cfg)
        write_jsonl(out, [manifest], skip_if_done=False)
        _log.info(
            "preprocess.index.done",
            stage=self.stage,
            path=str(out),
            variants=sorted(variants),
            passages={v: variants[v]["passages"] for v in sorted(variants)},
        )
        return out

    def _merge_cell(
        self, source_lang: str, cell: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], set[str]]:
        """One ``(variant, source_lang)`` cell's manifest entry + the ids it covers.

        Reads the cell's census off disk (:func:`read_shard_parts`) instead of trusting the
        reports it was handed, so a part directory that vanished between ``work`` and
        ``merge``, or was never published at all, fails here rather than being published as
        a complete cell; and asserts the parts are disjoint, because a part built under a
        wrong boundary produces a union that can still look complete while some ids are
        covered twice and others not at all.
        """
        lang_dir = Path(cell[0]["lang_dir"])
        census = read_shard_parts(lang_dir)
        if census["parts"] != len(cell):
            raise IndexIntegrityError(
                f"{lang_dir}: {census['parts']} part(s) on disk but {len(cell)} shard "
                "output(s), the cell is not the one the plan built"
            )
        parts: list[dict[str, Any]] = []
        covered: set[str] = set()
        total = 0
        for report in cell:
            ids = _shard_ids(report, self.legs)
            entry = _manifest_shard(report)
            entry["id_digest"] = _id_digest(ids)
            parts.append(entry)
            covered |= ids
            total += len(ids)
        if total != len(covered):
            raise IndexIntegrityError(
                f"{lang_dir}: its {len(cell)} parts cover {total} ids but only "
                f"{len(covered)} distinct ones, parts of one cell are disjoint slices of "
                "the pinned order, so an overlap means at least one was built under a "
                "different boundary and some passages are in no part"
            )
        return (
            {
                "path": str(lang_dir),
                "shared": cell[0]["variant"] is None,
                "parts": census["parts"],
                "part_passages": census["part_passages"],
                "part_dirs": census["dirs"],
                "passages": len(covered),
                "shard_parts": parts,
            },
            covered,
        )

    def _assert_covers_corpus(self, cfg: RunConfig, variant: str, covered: set[str]) -> None:
        """``covered`` must equal the corpus's whole ``passage_id`` set, exactly."""
        langs = {spec.source_lang for spec in self.shard_specs(cfg)}
        full: set[str] = set()
        for lang in sorted(langs):
            full |= self.expected_ids(cfg, lang)
        if covered != full:
            raise IndexIntegrityError(
                f"variant {variant!r} covers {len(covered)} passage ids but the corpus has "
                f"{len(full)} (missing {len(full - covered)}, extra {len(covered - full)}), "
                "nothing partial is published, so the manifest is not written"
            )


def _manifest_shard(report: Mapping[str, Any]) -> dict[str, Any]:
    """One part's manifest entry: path + per-leg counts/sizes/config hashes + its worker.

    ``worker`` is the observed hardware/job facts (:func:`saturate.worker_provenance`) of the
    process that built this part, carried through from its receipt. It is recorded per part,
    not only as the de-duplicated set ``_provenance``'s ``workers`` holds: that set says which
    architectures participated, which cannot answer "which shard ran where". The mapping does
    also survive in ``out/<shard>.jsonl`` and ``done/<shard>.result``, but those live in the
    queue directory, scratch, abandoned whenever the stage re-keys, while the manifest is
    the durable artifact. It matters because the build races a heterogeneous GPU
    OR-set (``workqueue_worker.sbatch``): the dense leg is measured bitwise-invariant across
    three architectures, and MILCO and PyLate are unmeasured on
    that axis, so a cross-architecture reproducibility question about them is answerable only
    from this field.

    An absent or empty ``worker`` becomes ``{}`` rather than a missing key: off SLURM
    ``worker_provenance`` legitimately returns nothing, and a key that is always present
    (sometimes empty) is readable, whereas a key present for some parts and absent for others
    looks complete while it is not.
    """
    return {
        "path": report["shard_dir"],
        "part": int(report.get("part") or 0),
        "shared": report["variant"] is None,
        "worker": dict(report.get("worker") or {}),
        "legs": {
            name: {
                "path": leg["path"],
                "config_hash": leg["config_hash"],
                "passages": leg["passages"],
                "bytes": leg["bytes"],
                # Carried through to the manifest: the count a reader fans across lives in
                # the published record, never in a reader's assumption. ``plaid_parts`` is
                # the leg's own sub-division (``parts.json``), distinct from the shard part
                # this entry describes: two axes, never one ambiguous "parts".
                "plaid_parts": leg.get("plaid_parts"),
            }
            for name, leg in sorted(report["legs"].items())
        },
    }


def _row_order_lang_ids(
    passages_path: Path,
    source_lang: str,
    unit_passages: int,
    units: Sequence[int],
) -> tuple[dict[int, frozenset[str]], int]:
    """The id sets row-order units ``units`` must hold + the unit count the table implies.

    The one definition of the cut the shipped build makes. A unit of ``S`` passages is
    rows ``[u*S, (u+1)*S)`` of ``source_lang`` in
    ``final/<recon12>/passages/<pack12>/passages.parquet`` row order (document order),
    which is exactly the half-open range :class:`~ragtime.preprocess.vectors.BlockPlan`
    computes on the ordinal side, for both of its axes: ``block_bounds`` (``u`` = an encode
    block, ``S`` = ``encode_block_passages``) and ``part_bounds`` (``u`` = an assemble part,
    ``S`` = ``plaid_part_passages``). Blocks divide parts evenly, so a part is the
    concatenation of its blocks and the two axes are this same arithmetic at two sizes.

    Both halves of the split call this function, ``preprocess.vectorize`` for blocks,
    ``preprocess.assemble`` for parts, because two implementations of one cut is precisely
    how the two halves drifted apart once: ``assemble.expected_ids_at`` validated a row-order
    artifact against :func:`_ordered_lang_keys`' token order, so every multi-part cell failed
    ``validate`` and returned to ``pending/`` (``token_count`` decreases 228,834 times in row
    order over the 460,326 real Spanish passages; the small fixtures happened to be monotone
    and could not see it).

    Only the asked-for units are materialized (~131 k ids for one assemble part, ~8 k for one
    block), never the whole language: this runs inside workers that hold three encoders or a
    whole PLAID buffer, so validation must not cost hundreds of MB. The scan is over two
    pruned columns of the one text-free passage table, which has no ``variant`` column, so
    the three renderings' unit ``u`` holds identical ids by construction.

    The second return value is what makes this more than a per-unit check: a shard seeded
    under a stale plan would cover a prefix of its language while every individual unit's id
    set still matched, and only the implied unit count reveals it.
    """
    size = max(1, int(unit_passages))
    wanted = {int(u) for u in units}
    if not wanted:
        raise IndexIntegrityError(
            f"{passages_path}: no row-order unit was asked for, a caller with nothing to "
            "validate must not reach the table at all"
        )
    lo = min(wanted) * size
    hi = (max(wanted) + 1) * size
    found: dict[int, set[str]] = {u: set() for u in wanted}
    seen = 0
    for row in iter_parquet(passages_path, columns=["lang", "passage_id"]):
        if row["lang"] != source_lang:
            continue
        position = seen
        seen += 1
        if lo <= position < hi:
            bucket = found.get(position // size)
            if bucket is not None:
                bucket.add(str(row["passage_id"]))
    return {u: frozenset(ids) for u, ids in found.items()}, max(1, -(-seen // size))


def _ordered_lang_keys(
    passages_path: Path, source_lang: str, spec: IndexShardSpec, part_size: int
) -> list[tuple[int, str]]:
    """One language's ``(token_count, passage_id)`` keys, in the legacy fused adapter's order.

    This is not the cut the shipped build makes, :func:`_row_order_lang_ids` above is.
    This order belongs to :class:`IndexAdapter` alone (:meth:`IndexAdapter.part_bounds` +
    :meth:`IndexAdapter.expected_ids_at`), the pre-split fused stage, whose two sides cut on
    it consistently; the ``vectorize``/``assemble`` pair cuts on row order, and uses this key
    only to order passages within an already-cut block (batch composition). Do not reach for
    it to describe a part or a block: the two orders differ on real data, which is exactly the
    defect this docstring's split records.

    Read from the one shared, text-free passage table with both columns pruned. Materializes
    one pair per passage of the language (~350 MB for English) and the callers drop it
    immediately; that is deliberate against a build whose steady state is tens of GiB, and it
    is what lets the expensive composing stream hold only two keys.

    It also re-derives the part count the table implies and refuses a plan that disagrees:
    a shard seeded under a stale plan would index a prefix of its language while every
    per-part check still passed, which is the one failure mode the part axis introduces.
    """
    keys = sorted(
        (int(row["token_count"] or 0), str(row["passage_id"]))
        for row in iter_parquet(passages_path, columns=["passage_id", "lang", "token_count"])
        if row["lang"] == source_lang
    )
    implied = max(1, -(-len(keys) // max(1, int(part_size))))
    if implied != int(spec.parts):
        raise IndexIntegrityError(
            f"{spec.name}: the shard plan says {spec.parts} part(s) of {part_size} passages, "
            f"but {passages_path} holds {len(keys)} {source_lang} passages, i.e. {implied} "
            "part(s), building or validating under a stale plan would leave part of the "
            "language unindexed while every per-part check still passed"
        )
    return keys


def _id_digest(ids: set[str]) -> str:
    """A stable fingerprint of an id set: order-free, so it compares memberships.

    Used to prove that part ``p`` of a language holds the identical passages in all three
    renderings without holding three copies of a million-id set in ``merge``'s memory at
    corpus scale.
    """
    digest = hashlib.sha256()
    for passage_id in sorted(ids):
        digest.update(passage_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _assert_parts_identical_across_renderings(
    digests: Mapping[tuple[str, int], Mapping[str, str]],
) -> None:
    """Part ``p`` of language ``l`` must hold the same ids in every rendering.

    A part is a slice of the one text-free passage table, which has no ``variant`` column,
    row order for the shipped ``vectorize``/``assemble`` split (:func:`_row_order_lang_ids`),
    the ``(token_count, passage_id)`` order for the legacy fused :class:`IndexAdapter`
    (:func:`_ordered_lang_keys`). Either way the cut is a function of that shared table alone,
    so this holds by construction, and it is asserted anyway because it is the property every
    cross-rendering comparison rests on: if ``omt``'s part 2 held different passages than
    ``original``'s, a rank difference between them would be measuring the cut, not the
    translation. Cheap (a digest per cell-part) and total (every part of every language).
    """
    disagreeing = {
        cell: dict(by_variant)
        for cell, by_variant in digests.items()
        if len(set(by_variant.values())) > 1
    }
    if disagreeing:
        raise IndexIntegrityError(
            "part membership differs between renderings for "
            f"{sorted(disagreeing)}: parts are cut on the shared, text-free passage "
            "table (row order under the vectorize/assemble split), so identical membership "
            "is structural, a difference means one rendering was built against a different "
            "table or a different part size, and every cross-rendering comparison over it "
            "would measure the cut instead of the translation"
        )


def _shard_ids(report: Mapping[str, Any], legs: Sequence[Leg]) -> set[str]:
    """The id set a shard actually covers: the intersection over its legs.

    Intersection, not union: an id present in two legs and missing from the
    third is not covered by that rendering's index, and a union would hide exactly the
    partial-leg failure the integrity assertion exists to catch.
    """
    shard_dir = Path(report["shard_dir"])
    sets = [frozenset(_idmap_ids(shard_dir / leg.name)) for leg in legs]
    return set(sets[0]).intersection(*sets[1:]) if sets else set()


def _idmap_ids(leg_dir: Path) -> list[str]:
    """A leg's ordinal-ordered ``passage_id``s (column-pruned; no text column exists)."""
    return [
        row["passage_id"]
        for row in iter_parquet(leg_dir / IDMAP_FILENAME, columns=["ordinal", "passage_id"])
    ]


def _idmap_rows(leg_dir: Path) -> int:
    """How many passages a published leg covers (0 for an unpublished one)."""
    path = leg_dir / IDMAP_FILENAME
    return len(_idmap_ids(leg_dir)) if path.exists() else 0


def _provenance(cfg: RunConfig, reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The manifest's provenance half: recipe identities + the pins that matter.

    ``gpu_arch_pin`` is recorded because it is still warranted for the sparse and
    late-interaction legs (their cross-architecture determinism is unmeasured); the dense leg
    measured bitwise-identical across three architectures and needs no constraint. The
    batch-composition pin, the floor that actually moves, is recorded as the order key and
    the two bounds, read from the hashed block so the manifest states the record rather than a
    code default, plus the observed hardware each shard ran on.
    """
    opts = index_build_options(cfg)
    hashes = all_hashes(cfg)
    return {
        "dense_model": opts.dense_model,
        "dense_revision": opts.dense_revision,
        "sparse_model": opts.sparse_model,
        "sparse_model_revision": opts.sparse_model_revision,
        # The sparse leg's build parameters, in full. Recorded as one sub-dict rather than
        # scattered leaves because the point is completeness: four of them were undeclared
        # vendor defaults, so a manifest written before they were named here could not tell
        # which pyseismic-lsr build had produced the index. ``top_k`` is here for the same
        # reason ``sparse_max_length`` is: it decides what the leg actually contains.
        "seismic": {
            "top_k": opts.seismic_top_k,
            "min_cluster_size": opts.seismic_min_cluster_size,
            "centroid_fraction": opts.seismic_centroid_fraction,
            "n_postings": opts.seismic_n_postings,
            "summary_energy": opts.seismic_summary_energy,
            "max_fraction": opts.seismic_max_fraction,
            "doc_cut": opts.seismic_doc_cut,
            "query_cut": opts.seismic_query_cut,
            "heap_factor": opts.seismic_heap_factor,
        },
        # Recorded for the same reason document_length is: it decides which of a passage's
        # tokens this leg ever saw, so a manifest that omitted it could not tell a 512-capped
        # build from an 8192 one.
        "sparse_max_length": opts.sparse_max_length,
        "spine_model": opts.spine_model,
        "spine_model_revision": opts.spine_model_revision,
        "spine_engine": opts.spine_engine,
        "ann_factory": opts.ann_factory,
        "plaid_nbits": opts.plaid_nbits,
        # Recorded for the same reason nbits is: it decides how this leg's vectors were
        # clustered and stored, so a manifest that omitted it could not tell a 1-part build
        # from a 12-part one at the same nbits.
        # One part size, not two: ``shard_part_passages`` was recorded here beside it and is
        # deleted with the key. After the vectorize/assemble split there is one part axis and
        # it is this one: an English cell is 23 parts of 131,072, not 3 of 524,288.
        "plaid_part_passages": opts.plaid_part_passages,
        "document_length": opts.document_length,
        "tokenizers": {
            DENSE_LEG: opts.dense_tokenizer_id,
            SPARSE_LEG: opts.sparse_pivot_tokenizer_id,
            LATE_INTERACTION_LEG: opts.late_interaction_tokenizer_id,
        },
        "gpu_arch_pin": opts.gpu_arch_pin,
        # The build's random state, stated per leg: including where it could not be pinned.
        # Recorded because "deterministic, no random state" is a spec claim, and an artifact
        # that carries the claim without naming its one exception is how a rendering-
        # correlated clustering difference stays invisible.
        "train_seed": {
            "seed": opts.train_seed,
            "seeded_legs": sorted(set(LEGS) - set(UNSEEDABLE_LEGS)),
            "unseeded_legs": dict(UNSEEDABLE_LEGS),
            # A seed is necessary and was not sufficient for the late-interaction leg: its
            # k-means ran a Triton assignment kernel that is not reproducible run-to-run
            # whatever the seed (measured). The pin that fixed it is recorded
            # here beside the seed, because a manifest claiming a seeded build without it
            # would have been true and still described a non-reproducible artifact.
            "plaid_use_triton_kmeans": _PLAID_USE_TRITON_KMEANS,
        },
        "batch_composition_pin": {
            "order": "(token_count, passage_id)",
            "token_budget": opts.encode_batch_token_budget,
            "max_items": opts.encode_max_items,
        },
        "chunker_hash": hashes["chunker"],
        # Which architectures participated: a de-duplicated set: it is the
        # one-glance answer to "was this build homogeneous?". It is not the per-shard
        # mapping, and must not be mistaken for it: "which shard ran where" is recorded per
        # part by ``_manifest_shard``'s ``worker`` field (see its docstring for why the
        # queue's own copy does not suffice).
        "workers": sorted(
            {json.dumps(r.get("worker") or {}, sort_keys=True) for r in reports}
        ),
    }


def shard_agreement(
    handle_a: LegHandle, handle_b: LegHandle, leg: str, query: str, top_k: int
) -> dict[str, float]:
    """Rank agreement between the same leg of two parts (e.g. ``omt`` vs ``omt_opus``).

    Part-grained rather than cell-grained: the two handles must hold the same
    passages for the comparison to be about translation, and part ``p`` of a language holds
    the identical ids in every rendering (asserted by
    :func:`_assert_parts_identical_across_renderings`). A whole-cell version composes this
    from :func:`query_lang_leg` the same way; nothing here needs re-deriving for it.

    The operational form of the primary, causally clean comparison: both renderings are
    monolingual-English retrieval over the same passages, so a low overlap here is
    attributable to MT quality alone. It reports numbers and asserts nothing, the threshold
    is read off the measured noise floor by the caller, never invented here, and this is a
    liveness/consistency signal, never a quality measurement (no qrels exist).
    """
    left = [pid for pid, _ in query_leg(handle_a, leg, query, top_k)]
    right = [pid for pid, _ in query_leg(handle_b, leg, query, top_k)]
    depth = max(1, min(len(left), len(right)))
    overlap = len(set(left[:depth]) & set(right[:depth])) / depth
    return {
        "overlap": overlap,
        "rbo": rbo(left, right),
        "tau_b": kendall_tau_b(left, right),
        "depth": float(depth),
    }
