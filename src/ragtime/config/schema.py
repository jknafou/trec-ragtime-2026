"""Config schema and validation: the config file is the reproducible record.

One-shot validation of a fixed set of named blocks whose leaves mix structured knobs
with free-form English prose. The checks are recursive unknown-key rejection, so the
file must stay the complete record with no hidden state; a required ``run`` block and
a required ``seeds`` value that is never silently defaulted; the presence of every
shared block; ``decomposition.model == llm.model``, since one served instance handles
both and there is no separate decomposer; the two knob enums, each defaulting to
``original``; and the ``kind`` and ``status`` enums.

Not pydantic or jsonschema: there is nothing to coerce and no open or
recursive shape, so a second modelling paradigm would only add a schema system.
``validate`` is pure, with no IO; ``loader.load`` supplies ``source_path`` and
``source_text`` afterwards via :func:`dataclasses.replace`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = [
    "PLACEHOLDER_TEAM_ID",
    "TEAM_ID_HELP",
    "ConfigError",
    "RunConfig",
    "team_id_problem",
    "validate",
]

# --------------------------------------------------------------------------- #
# Domain vocabulary: the enums live here rather than scattered across stages.
# --------------------------------------------------------------------------- #
KIND_E2E = "e2e_agentic"
# `run.kind` is a one-value enum: every shipped config, both families, runs the agentic
# pipeline, and Task 2 is that same pipeline with the search knob moved rather than a
# second code path. Kept as a table rather than folded into a constant because `seeds` is
# kind-derived: a config must state a value matching this table rather than inheriting a
# silent default, so the run record can never disagree with what ran. The value is 1:
# retrieval is deterministic, and the agentic arm was reduced to a single seed to buy topic
# coverage under the compute budget, which means the reported numbers are point estimates
# rather than multi-seed intervals.
_KIND_SEEDS: dict[str, int] = {KIND_E2E: 1}
_STATUS_VALUES = frozenset({"submitted", "optional"})
# The translation axis is MT tier: `omt` is NLLB-200-3.3B and `omt_opus` is OPUS-MT over
# the same final inventory, both compared against the untranslated `original`. Both
# translated renderings key into the one `final/<recon12>/` inventory, so the knob
# changes which text is shown, never which passages exist.
KNOB_VALUES = frozenset({"original", "omt", "omt_opus"})

# The blocks that stay byte-identical across a run family, which is what makes the
# translation delta attributable to translation rather than to infrastructure.
#
# Every block that decides what the corpus is belongs here: the whole family reads one
# merge map, one translated corpus, one final inventory, one packing and one index build
# method, so none of those can be a per-run choice. The same holds for `serialize`, since
# a different selection policy would turn a translation delta into a budget-fit artifact.
#
# This tuple doubles as the required-blocks list in `validate` below, so a config lacking
# any of them fails to load. Adding a name here therefore means adding the block to every
# config in the same change.
SHARED_BLOCKS: tuple[str, ...] = (
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
    "serialize",
    # The Task-1 and Task-3 contrast is original against translated over the same
    # requests, so two members reading different topics files are not comparable at all.
    # One honest limit: `family_guard` groups by family, so this is enforced within the
    # e2e arms and within the mlir arms but never across the two, and all three tasks
    # sharing one request set remains a project-level assumption.
    "topics",
)

# --------------------------------------------------------------------------- #
# The allowed schema: a flat nested Python literal beside the field list. A key
# mapping to a dict recurses into that sub-mapping; a key mapping to LEAF is a
# free-form leaf with any value and no recursion. Growing this into $ref and oneOf
# generality would be the signal to adopt jsonschema instead.
# --------------------------------------------------------------------------- #
LEAF = None


def _keys(*names: str) -> dict[str, Any]:
    """A closed set of free-value keys: the names are checked, the values are not.

    The middle ground the semantic ``.config`` leaves need. Fully free-form blocks
    contradict the rule that unknown keys are rejected so the file stays the complete
    record, and they have in practice accepted, hashed and ignored keys that were
    implemented nowhere, which makes the artifact advertise a recipe it does not have.
    Checking names only keeps the schema free of a modelling framework while making a
    typo or a phantom knob fail at load rather than never.
    """
    return dict.fromkeys(names, LEAF)


#: ``chunker.config``: the packer's semantics, byte-identical per family.
CHUNKER_CONFIG: dict[str, Any] = _keys(
    "token_budget", "overlap_frac", "segmenter_model", "tokenizer_id", "data_model",
    "strip_boilerplate", "boilerplate_rules",
    "prefer_paragraph_break", "prefer_paragraph_break_min_fill",
)

#: ``translation.config``: MT identity and decoding. ``compute_type`` and ``batch_type``
#: record what actually runs; machine facts such as paths and shape keys stay in
#: ``execution``.
TRANSLATION_CONFIG: dict[str, Any] = _keys(
    "engine", "omt_model", "target_lang", "source_langs",
    # `model_family` picks the source-sequence layout (`serving.mt`): the NLLB arm sends a forced
    # `eng_Latn` target prefix, the Marian arm must not, because its vocabulary has no such token.
    # `translate_units` picks the granularity of one MT call (`preprocess.translate_omt`): `merged`
    # translates the merge map's marker-joined units and splits them back, `sentences` is one
    # sentence in and one translation out. Both are read by the shipped code and both change the
    # artifact, so both belong in the hashed record rather than in an operator's invocation.
    "model_family", "translate_units",
    "beam_size", "no_repeat_ngram_size", "max_input_length", "max_decoding_cap",
    "compute_type", "batch_type", "bucket_token_budget", "max_sentences_per_bucket",
    "len_ratio_a", "len_ratio_b",
    "merge_marker", "join_sep",
    "qe_model", "qe_threshold",
)

#: ``index_build.config``: the one index recipe applied identically to all three
#: renderings. Several of these keys are additionally value-checked against the real
#: constants at ``preprocess.index`` bring-up, because naming a key here stops a typo but
#: only the bring-up check stops a well-spelled key from asserting a build that does not
#: happen. The rule for this block is that every key is either applied or validated, and
#: none is merely accepted.
INDEX_BUILD_CONFIG: dict[str, Any] = _keys(
    "variants", "shard_by", "english_once",
    "dense_engine", "dense_model", "dense_revision", "dense_tokenizer_id",
    "ann_factory", "nprobe", "quantizer_train_seed", "quantizer_train_sample",
    "sparse_engine", "sparse_model", "sparse_model_revision",
    "sparse_pivot_tokenizer_id", "sparse_max_length",
    "seismic_centroid_fraction", "seismic_min_cluster_size",
    "seismic_query_cut", "seismic_heap_factor",
    # The assemble-time cut on a passage's sparse components. Filed in
    # `preprocess.index.ASSEMBLE_KEYS` rather than `ENCODE_KEYS`, so re-tuning it
    # re-assembles on CPU and re-encodes no vector: the stored sparse vectors stay the
    # encoder's raw output.
    "seismic_top_k",
    # Four Seismic build parameters declared at their current effective values. Without a
    # declaration here, unknown-key rejection would make them unstateable and a library
    # upgrade could move them silently.
    "seismic_n_postings", "seismic_summary_energy",
    "seismic_max_fraction", "seismic_doc_cut",
    "spine_engine", "spine_model", "spine_model_revision",
    "late_interaction_tokenizer_id", "plaid_nbits", "plaid_part_passages",
    # Query-time keys, declared so a run record can state them and excluded from
    # `preprocess.index.index_hash` so that stating them re-keys no index: they decide how
    # a built index is read, never what was written.
    #
    # `query_plaid_device` is a scientific variable rather than a speed knob, because CPU
    # and CUDA late-interaction scoring disagree well inside the reranker's input pool.
    # `query_encode_device` is the dense encoder's device when it embeds a query-side
    # string rather than a corpus passage, today the decompose paraphrase-dedup
    # pre-filter. Both live in this fairness-shared block, and not in `execution`, because
    # a device changes a forward pass and therefore can change a result: a cosine near the
    # cutoff can merge on one device and not on the other. When absent,
    # `query_encode_device` inherits `encode_device`.
    "query_length", "document_length", "query_plaid_device", "query_encode_device",
    "encode_batch_token_budget", "encode_max_items", "encode_device",
    # The vectorize unit: how many passages of the pinned `(token_count, passage_id)`
    # order one encode block covers. It decides batch membership and therefore the vectors
    # bitwise, so it is hashed here rather than in `execution`, unlike
    # `execution.vectorize_blocks_per_task`, which only decides how many already-defined
    # blocks one worker claims and changes no byte.
    "encode_block_passages",
    # How a leg's vectors are stored between vectorize and assemble. Hashed because the
    # stored bytes are the artifact this split publishes: an fp16 store is a different,
    # lossy vector set to an fp32 one, and an assemble reading it builds a different index.
    # One key per leg, because the three legs store three different shapes.
    "dense_store_dtype", "late_interaction_store_dtype", "sparse_store_format",
    # The device the assemble step runs on, on the same precedent as `encode_device`:
    # clustering is reproducible on CPU and not on GPU, so the assembled bytes depend on
    # whether the worker had one. Left unstated, one rendering assembling on CPU and
    # another on GPU would be a per-rendering difference with no cause in the text. Never
    # "auto", which would hash identically while resolving to different numerics.
    "assemble_device",
)

_OUTPUT_ITEM: dict[str, Any] = {
    "task": LEAF,
    "track": LEAF,
    "path": LEAF,
    "format": LEAF,
    "run_id": LEAF,
    "run_desc": LEAF,
}

_ALLOWED: dict[str, Any] = {
    "run": {"id": LEAF, "kind": LEAF, "status": LEAF, "description": LEAF},
    "llm": {
        "model": LEAF,
        "role": LEAF,
        "decoding": {"temperature": LEAF, "top_p": LEAF, "top_k": LEAF},
        "pin": LEAF,
        # Wall-clock bound on one generation call, in seconds. Declared rather than left to
        # a client default because it changes results: a call that trips it raises instead
        # of returning, so the loop's budget backstop can fire. An HTTP read timeout does
        # not cover this, since it measures the gap between reads and is reset indefinitely
        # by a slowly trickling degenerate generation.
        "call_timeout_s": LEAF,
    },
    # `max_retries` bounds a same-turn re-decode of a malformed action, so a span-formatting
    # slip does not cost a turn of the search budget. Fairness-shared, because it changes
    # how much evidence a loop ends up holding.
    "claim_commit": {"rule": LEAF, "properties": LEAF, "max_retries": LEAF},
    # Enumerated rather than nested under a permissive LEAF: this block is fairness-shared
    # and its hash is the prompt hash recorded in provenance, so a silently accepted typo
    # would be a silent semantic change to the round-0 anchor, and `_reject_unknown_keys`
    # stops recursing at a LEAF. `seeds` is absent, since the top-level
    # `seeds` key is the single source of truth and `family_guard` checks it.
    "decomposition": {
        "policy": LEAF,
        "usage": LEAF,
        "model": LEAF,
        "output": LEAF,
        "decoding": {"temperature": LEAF, "top_p": LEAF, "top_k": LEAF},
        "few_shot": {"n": LEAF, "exemplar_set": LEAF},
        "k_band": LEAF,  # list of [limit_threshold, k_min, k_max] triples
        "R_max": LEAF,
        "min_new": LEAF,
        "low_streak": LEAF,
        "vital_cutoff": LEAF,
        "dedup": {"llm_paraphrase_merge": LEAF, "cosine_cutoff": LEAF},
        "retrieval_access": {"seed": LEAF, "audit": LEAF},
        "audit_evidence": LEAF,
        "grounding_gate": LEAF,
        "background": LEAF,
    },
    # The RAG-loop budget. Enumerated for the reason the `decomposition` note gives: the
    # block is fairness-shared and every knob in it changes how much evidence a loop
    # gathers before it answers, so a silently accepted `max_iter` would leave the real cap
    # at its default while the run record claimed otherwise.
    "rag_loop": {
        "budget": {
            "min_searches": LEAF,  # effort floor: abstaining below this many searches is illegal
            "max_iters": LEAF,  # turn cap, a backstop
            "max_searches": LEAF,  # search cap, a backstop
            "token_budget": LEAF,  # per-loop token ceiling, a backstop
            # Fraction of `max_iters` at which a loop holding committed claims is told once
            # that its turns are running out. Not a backstop and not a new terminal, since
            # `submit_answer` is already on the menu every turn; it only makes stopping
            # salient. `null` disables it.
            "warn_at_frac": LEAF,
        },
        "search_top_k": LEAF,  # passages returned per search action
        "fan_out": {"concurrency": LEAF},  # null derives it from the vLLM KV headroom
        "emission": LEAF,  # "action": one grammar-constrained {rationale, action} per turn
        "abstain": LEAF,  # "gated": legal only once the effort floor holds
    },
    "chunker": {"config": CHUNKER_CONFIG, "hash": LEAF},
    # Semantic translation knobs only: anything that changes the translated bytes, such as
    # model identity, beam, target and source languages, the repetition guard and bucket
    # composition. Declared with the same {config, hash} shape as `chunker`. Hardware and
    # throughput knobs live in `execution` so they never enter this hash.
    "translation": {"config": TRANSLATION_CONFIG, "hash": LEAF},
    # The post-chunk sentence merge, which is the translation-unit rule, keyed by its own
    # hash so it can never perturb `chunker`'s and therefore never invalidates the frozen
    # sentence spine. Unlike `chunker.config` this is a new top-level block, so it must be
    # enumerated here or `_reject_unknown_keys` fails every config.
    "merge": {
        "merge_short_sentences": LEAF,
        "min_sentence_tokens": LEAF,
        "merge_cap_frac": LEAF,
        "merge_cross_paragraph": LEAF,
        "merge_languages": LEAF,
    },
    # Sentence reconciliation: the fuse-versus-split-back policy applied to the raw,
    # marker-bearing MT output. Its own top-level block for the same reason `merge` is,
    # hashed separately and folded with the chunker, merge and translation hashes into the
    # composite `final/<recon12>/` path level, so a policy edit re-derives the final corpus
    # from stored text without touching the spine, the map or the translation.
    #
    # Fusion only: these keys decide which sentences exist and nothing else does, so
    # `recon12` names the sentence inventory. How those sentences are packed into passages
    # lives in the separate `packing` block, because a packing edit that moved `recon12`
    # would re-key `final/<recon12>/` and orphan everything under it by path.
    "reconcile": {
        "split_back": LEAF,
        "min_segment_chars": LEAF,
        # Whether `translations/<variant>.parquet` carries a row for an English sentence.
        # `false`, the shipped value, says no: English has no translation, and an identity
        # copy of `documents.text[start:end]` under an `omt` name is both a large duplicate
        # and an invitation to read it as MT output. Consumers resolve English from the
        # native span instead, so the rendering axis has exactly one English text.
        #
        # A hashed `reconcile` key rather than a code-side default, because it changes the
        # stored artifact and `recon12` must move with it. It does not change which
        # sentences exist; `preprocess.reconcile.inventory_hash` is the fusion-only subset
        # that keys artifacts derived from the inventory.
        "store_identity_translations": LEAF,
    },
    # Passage packing, the sentences-to-passages grouping, hashed on its own as
    # `pack12 = H(packing)` and folded into the passages artifact's path level under the
    # already `recon12`-keyed final node. Fairness-shared rather than an `execution` knob:
    # the three renderings must hold the same passages under the same ids, which is the
    # basis of the Task-1 and Task-2 comparison.
    "packing": {
        # "native" packs against the native span alone; "len_max" packs against the maximum
        # over the three renderings. Native packing lets a passage fit in Chinese and
        # overflow the retrieval window once read in English.
        "pack_length": LEAF,
        # The token budget packing targets; the content budget is this minus the special
        # tokens. Not `chunker.config.token_budget`, whose hash keys
        # `corpus_dir`, so moving it would orphan the documents, the sentence spine, the
        # merge map and the translations by path.
        "pack_budget": LEAF,
        # The tokenizer identity `len_max` was measured with, recorded and keyed into the
        # sidecar's path so the measurement is auditable.
        "len_max_tokenizer_id": LEAF,
    },
    # The build-time index recipe, in the same {config, hash} shape as `chunker` and
    # `translation` so the per-leg sub-keys need no enumeration here while the block stays
    # declared and separately hashed into the index path level. Named `index_build` rather
    # than `index` to avoid collision with `retrieval.index`, the search-time knob.
    # Query-time knobs stay in `retrieval` and machine facts stay in `execution`, but
    # encode composition is semantic and stays here, since a different batch composition
    # changes the vectors bitwise.
    "index_build": {"config": INDEX_BUILD_CONFIG, "hash": LEAF},
    # Select and serialize. Fairness-shared, because the selection policy must be identical
    # across a family or a translation delta becomes an artifact of a different budget fit.
    #
    # The tuned knobs have no code default: a missing `k_t3` must fail loudly
    # rather than silently pick a number nobody chose. The spec-pinned ones do default,
    # because those values come from the track specification.
    #
    # This set must equal `select_serialize.knobs.SerializeKnobs.__dataclass_fields__`. A
    # knob the stage can read but a config cannot write would take its value from a code
    # default, and the file would stop being the complete run record while still looking
    # like one. `test_serialize_allowed_keys_equal_the_knob_fields` fails on any drift; it
    # lives in tests/ because the dependency spine forbids `config` importing `pipeline`.
    "serialize": {
        "k_t3": LEAF,  # Task-3 nugget budget (tuned)
        "answers_cap": LEAF,  # answers kept per nugget in Task 3 (tuned)
        "dedup_a_cutoff": LEAF,  # terminal English-only answer dedup cosine (tuned)
        "rrf_k": LEAF,  # spec-pinned at 60
        "top_docs": LEAF,  # track rule: up to 1,000 results per query
        "dedup_q_cutoff": LEAF,  # terminal English-only question dedup
        "answer_score": LEAF,  # max_reference or as_written; orders tiers 1 and 2, cuts Task 3
        "validate": LEAF,  # the official-validator hard gate
        "empty_fallback": LEAF,  # emit the neutral framing sentence rather than nothing
        "collection_id": LEAF,  # track constant, representable so the record can be complete
        "run_mode": LEAF,  # submission metadata: automatic or manual
    },
    # Execution and scheduling knobs. Not a fairness-shared block and not chunking or
    # translation semantics, so a per-config tweak never trips `family_guard` nor changes a
    # semantic block hash. The family-shared corpus is built first-runner-wins, so members
    # need not agree on shard counts.
    "execution": {
        # The artifact root: the directory `common.Layout` resolves the family-shared
        # corpus, the work queues, the vector store and the container image under. A
        # machine fact, on the same precedent as `ct2_model_dir`: a path is not a semantic
        # identity, so hashing it would re-key every artifact the day the storage moves.
        #
        # Not enforced equal across a family, but it must be written equal: the corpus is
        # built first-runner-wins and shared byte-identically, so two members naming two
        # roots would build and search two corpora. Absent, it falls back to
        # $RAGTIME_ARTIFACT_ROOT and then to the repo-relative "runs" default.
        "artifact_root": LEAF,
        "corpus_shards": LEAF,
        "oversubscription": LEAF,
        "chunk_batch_size": LEAF,
        # Chunk-stage process-pool size inside one work-queue worker. Throughput only, with
        # byte-identical output; defaults to SLURM_CPUS_PER_TASK.
        "chunk_workers": LEAF,
        # Corpus substage widths: each substage after chunk reads its own
        # `<substage>_shards` and `_oversubscription` pair, while chunk keeps reading the
        # two keys above. Throughput only, since a tweak changes nothing about the artifact
        # and only how many workers race to produce it.
        "merge_shards": LEAF,
        "merge_oversubscription": LEAF,
        "translate_identity_shards": LEAF,
        "translate_identity_oversubscription": LEAF,
        "translate_shards": LEAF,
        "translate_oversubscription": LEAF,
        "reconcile_shards": LEAF,
        "reconcile_oversubscription": LEAF,
        # The len_max sidecar substage (one row per final sentence). Throughput only.
        "len_max_shards": LEAF,
        "len_max_oversubscription": LEAF,
        # The pack substage that derives passages from the frozen inventory. Throughput
        # only: the grouping is decided by the fairness-shared `packing` block, so a shard
        # count change re-fans the work without touching a stored byte.
        "pack_shards": LEAF,
        "pack_oversubscription": LEAF,
        # The shape-set key the translate replica is allocated against; the shapes
        # themselves live in config/serving/, since machine facts never join a run family.
        "translate_shape_key": LEAF,
        # Machine-specific path to the converted NLLB checkpoint. Not in
        # translation.config: a filesystem path is not a semantic identity, so it must not
        # enter the fairness hash. `translation.config.omt_model` is the identity that lands
        # on every row's model_id.
        "ct2_model_dir": LEAF,
        # Index substage widths and machine facts only. The recipe keys that decide a stored
        # byte are hashed in `index_build.config`; these only decide how many workers race
        # to produce the same bytes. `vectorize_blocks_per_task` is the one that looks
        # semantic and is not: a block's membership is fixed by the hashed
        # `encode_block_passages`, so claiming four blocks per task instead of one re-fans
        # the work without moving a boundary.
        "vectorize_shards": LEAF,
        "vectorize_oversubscription": LEAF,
        "vectorize_blocks_per_task": LEAF,
        "assemble_shards": LEAF,
        "assemble_oversubscription": LEAF,
        # The inner query-side width: how many of a language cell's index parts
        # `preprocess.index.query_lang_leg` searches at once. It belongs here for the same
        # reason `vectorize_blocks_per_task` does, in that it decides how the same work is
        # fanned rather than what the work produces: the fan collects in part order, so the
        # returned hits and scores are byte-identical at any width and a retune must not
        # re-key an index. Absent, it is derived from SLURM_CPUS_PER_TASK.
        "query_part_workers": LEAF,
        # The outer query-side width and the sibling of the leaf above: how many
        # `(query string, leg, language cell)` units `retrieval.service` searches at once.
        # Same reasoning and the same three-case contract (absent derives from
        # SLURM_CPUS_PER_TASK, 0 is sequential, N is N, a non-integer raises), and the same
        # byte-identity guarantee, since the fan collects in submission order.
        #
        # The two widths nest and are resolved as one decision by
        # `retrieval.concurrency.plan_query_concurrency`: an absent `query_part_workers`
        # becomes 1 whenever this fan is active, which stops the outer and inner widths from
        # multiplying out into far more threads than the node has. Naming both explicitly is
        # honoured verbatim.
        "query_leg_workers": LEAF,
        # The root a staged copy of the by-id passage store is served from, typically
        # node-local NVMe or /dev/shm. `retrieval.store_mirror` re-resolves the same
        # recon12- and pack12-keyed relative path beneath it, rather than accepting a
        # free-form path to a `data.mdb` that could address a differently-packed table,
        # verifies the copy against the origin's marker and size, and falls back to the
        # origin loudly.
        #
        # A machine fact that changes no returned byte, being the same LMDB with the same
        # keys and values at a different inode, so it must never enter `config_hash` and
        # `family_guard` must never compare it. What it changes is where the bytes are read
        # from, which dominates the tail of a depth-100 fetch. Absent, the canonical store
        # under `artifact_root` is used. Unlike `artifact_root` it need not be written equal
        # across a family, since it addresses a copy of one artifact.
        "passage_store_mirror_root": LEAF,
        # Per-substage reaper window in seconds, one key per link of the corpus chain: how
        # long a worker may be silent before another worker's `reap_stale` pass returns its
        # claimed shard to `pending/`. Absent, `saturate.MAX_AGE_S` applies. Non-shared for
        # the same reason as the widths above, since a liveness fact changes no byte. It is
        # per substage because the links' per-unit work spans three orders of magnitude, and
        # a single constant sized for a seconds-long merge shard would reclaim a
        # twenty-minute assemble part mid-flight forever. Rejected at load if it is at or
        # below the heartbeat interval.
        "chunk_max_age_s": LEAF,
        "merge_max_age_s": LEAF,
        "translate_identity_max_age_s": LEAF,
        "translate_max_age_s": LEAF,
        "reconcile_max_age_s": LEAF,
        "len_max_max_age_s": LEAF,
        "pack_max_age_s": LEAF,
        "vectorize_max_age_s": LEAF,
        "assemble_max_age_s": LEAF,
    },
    "seeds": LEAF,
    "citations": LEAF,
    "languages": LEAF,
    "passage_lang": LEAF,
    "retrieval": {
        "spine": LEAF,
        "dense": LEAF,
        "sparse": LEAF,
        "rrf": {"mode": LEAF, "k": LEAF, "quality_gated": LEAF},
        "reranker": {"model": LEAF, "depth": LEAF},
        "index": LEAF,
        # Declared but NOT consumed by any code path: every shipped run is `e2e_agentic`,
        # whose `retrieved[]` comes from the loop's own search actions at
        # `rag_loop.search_top_k`. The three `config/mlir-*.yml` still carry `top_k: 100`,
        # so the leaf must stay allowed here or those files would be rejected as unknown
        # keys. Drop the leaf and the three config lines together, never one without the
        # other.
        "top_k": LEAF,
    },
    "outputs": LEAF,  # a list of mappings, each item validated separately below
    # The request set this run reads. A mapping rather than a bare scalar, because that
    # keeps `cfg.blocks` homogeneous for `all_hashes` to iterate, because `{"path": LEAF}`
    # rejects a mistyped sub-key at load instead of at read time on a GPU node, and because
    # a second field such as a file checksum can join later without a shape change.
    "topics": {"path": LEAF},
    "team_id": LEAF,
}


class ConfigError(ValueError):
    """Raised on any validation failure: unknown key, missing or wrong seeds, bad enum."""


# --------------------------------------------------------------------------- #
# team_id: is this a registered team id, or is it still a stand-in?
# --------------------------------------------------------------------------- #
#: The stand-in a config carries before a registered team id is filled in.
PLACEHOLDER_TEAM_ID = "TBD (set before submission)"

#: Casefolded substrings that mark a value as a stand-in rather than an id. Deliberately a
#: short list of things nobody registers, since a longer one only adds false-positive
#: surface. A false positive is loud and one config edit away from fixed, whereas a false
#: negative is a placeholder uploaded to the track.
_TEAM_ID_STANDIN_MARKERS: tuple[str, ...] = (
    "tbd",
    "todo",
    "fixme",
    "placeholder",
    "changeme",
    "change me",
    "set before submission",
    "your team",
    "<",
    ">",
)

#: What to do about it, spelled once so every refusal says the same thing.
TEAM_ID_HELP = (
    "Set `team_id:` to the id registered with the track in all six config/*.yml in one "
    "pass (e2e-original, e2e-omt, e2e-omt-weak, mlir-original, mlir-omt, mlir-omt-weak). "
    "`team_id` is not a shared block, so `family_guard` will not "
    "notice if you update five and forget one. It keys nothing on disk, so changing it "
    "invalidates no artifact and nothing has to be recomputed."
)


def team_id_problem(team_id: Any) -> str | None:
    """Return an actionable message if ``team_id`` may not be submitted, else ``None``.

    A run record fails this in two ways, both of which reach ``metadata.team_id`` in every
    emitted submission line and both of which the official validator accepts, its schema
    being a bare string: the value is empty or absent, or it is present but a stand-in.

    Not called from :func:`validate`. ``config.load`` and ``config.check`` are
    the hard pre-launch gate on every ``run --config`` invocation, including pipeline cells
    that emit no submission at all, so raising there would stop the pipeline from computing
    over a field that keys nothing on disk, is read by nothing before serialization and is
    safe to set last. The earliest consequential point is when a submission file is about to
    be written, which is where ``select_serialize.submission.envelope`` calls this: its
    envelope construction runs before any topic is read, so a refusal lands with nothing
    published.
    """
    value = "" if team_id is None else str(team_id).strip()
    if not value:
        return (
            "config carries no team_id; every submission line requires one. " + TEAM_ID_HELP
        )
    folded = value.casefold()
    for marker in _TEAM_ID_STANDIN_MARKERS:
        if marker in folded:
            return (
                f"team_id {value!r} is a stand-in, not a registered team id (it contains "
                f"{marker!r}). The official validator's schema is a bare string, so this "
                f"would validate and would be uploaded. " + TEAM_ID_HELP
            )
    return None


# --------------------------------------------------------------------------- #
# The one frozen, fully-resolved run record.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RunConfig:
    """One frozen, fully-resolved config object: the launch record.

    ``blocks`` is the full validated and defaulted top-level map, nested read-only, and is
    the input to :func:`ragtime.config.all_hashes`. ``source_text`` is the exact UTF-8 file
    text and the input to ``fairness.family_guard``'s byte-identical shared-block check;
    only :func:`ragtime.config.load` populates it, so a synthesized ``RunConfig`` with an
    empty ``source_text`` fails the fairness gate loudly.
    """

    run_id: str
    kind: str
    status: str
    passage_lang: str  # the reading knob; resolved, default "original"
    retrieval_index: str  # the search knob; resolved, default "original"
    # `topics.path` lifted to a first-class field, verbatim as written in the config and
    # never resolved here. What it is relative to is the caller's decision, as with
    # `Layout.submission(track)`: inheriting the process working directory would make one
    # config read two different request sets depending on where the job started. Required,
    # since `topics` is a shared block, so `validate` always sets it.
    topics_path: str
    seeds: int
    languages: tuple[str, ...]
    team_id: str
    outputs: tuple[Mapping[str, Any], ...]
    blocks: Mapping[str, Any]
    source_path: Path = field(default=Path("."))
    source_text: str = ""


# --------------------------------------------------------------------------- #
# Immutability + unknown-key rejection helpers.
# --------------------------------------------------------------------------- #
def _freeze(obj: Any) -> Any:
    """Recursively wrap mappings as ``MappingProxyType`` and lists as tuples."""
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _reject_unknown_keys(d: Mapping[str, Any], allowed: Mapping[str, Any], path: str) -> None:
    """Recursively raise ``ConfigError`` for any key of ``d`` absent from ``allowed``."""
    for key, value in d.items():
        if key not in allowed:
            where = f"{path}.{key}" if path else key
            raise ConfigError(f"unknown config key: '{where}' (the file must stay the complete record)")
        sub = allowed[key]
        if isinstance(sub, Mapping) and isinstance(value, Mapping):
            _reject_unknown_keys(value, sub, f"{path}.{key}" if path else key)


def _check_knob(name: str, value: Any) -> None:
    if value not in KNOB_VALUES:
        raise ConfigError(f"{name} must be one of {sorted(KNOB_VALUES)}; got {value!r}")


# --------------------------------------------------------------------------- #
# The one validation entrypoint.
# --------------------------------------------------------------------------- #
def validate(raw: Mapping[str, Any]) -> RunConfig:
    """Validate an already-parsed config dict and return a frozen ``RunConfig``.

    Pure, with no IO. Rejects unknown keys, enforces ``run``, ``seeds`` and
    ``topics.path``, enum-checks ``kind``, ``status`` and the two knobs, and resolves the
    knob defaults to ``original``. ``source_path`` and ``source_text`` are left at their
    defaults for :func:`ragtime.config.load` to fill in.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError("config root must be a mapping")

    _reject_unknown_keys(raw, _ALLOWED, "")

    run = raw.get("run")
    if not isinstance(run, Mapping):
        raise ConfigError("missing required 'run' block")
    for req in ("id", "kind", "status"):
        if req not in run:
            raise ConfigError(f"missing required key: 'run.{req}'")

    # Every shared block must be present: a config missing one would otherwise hash it to
    # the empty string and pass family_guard vacuously.
    for name in SHARED_BLOCKS:
        if name not in raw:
            raise ConfigError(f"missing required block: '{name}'")

    # The decomposer is the same served instance as the loop's LLM: there is no separate
    # decomposer model, and one shared vLLM serves both. Two independently editable model
    # ids in one file would drift silently, and each sits inside a different
    # fairness-shared block, so a family could stay byte-identical while naming two models.
    decomp = raw.get("decomposition")
    llm_blk = raw.get("llm")
    if (
        isinstance(decomp, Mapping)
        and isinstance(llm_blk, Mapping)
        and "model" in decomp
        and decomp["model"] != llm_blk.get("model")
    ):
        raise ConfigError(
            f"decomposition.model ({decomp['model']!r}) must equal llm.model "
            f"({llm_blk.get('model')!r}); there is no separate decomposer model"
        )

    kind = run["kind"]
    if kind not in _KIND_SEEDS:
        raise ConfigError(f"run.kind must be one of {sorted(_KIND_SEEDS)}; got {kind!r}")
    status = run["status"]
    if status not in _STATUS_VALUES:
        raise ConfigError(f"run.status must be one of {sorted(_STATUS_VALUES)}; got {status!r}")

    # seeds: required and kind-derived, never silently defaulted.
    if "seeds" not in raw:
        raise ConfigError("'seeds' is required and is never defaulted")
    seeds = raw["seeds"]
    expected = _KIND_SEEDS[kind]
    if seeds != expected:
        raise ConfigError(f"seeds must be {expected} for kind {kind!r}; got {seeds!r}")

    # Two independent translation knobs, resolved to 'original' when absent.
    passage_lang = raw.get("passage_lang", "original")
    _check_knob("passage_lang", passage_lang)
    retrieval = raw.get("retrieval")
    if isinstance(retrieval, Mapping) and "index" in retrieval:
        retrieval_index = retrieval["index"]
    else:
        retrieval_index = "original"
    _check_knob("retrieval.index", retrieval_index)

    # The request set. `topics` is a shared block, so its presence is checked above; its
    # `path` is checked here for the same reason `outputs[i].path` is, since a block that
    # declares no path routes nowhere and `topics: {}` would otherwise hash fine and die at
    # read time on a compute node.
    topics_blk = raw["topics"]
    # `topics:` with an empty body parses to None, which is a missing `path` rather than a
    # wrong shape; reporting it as a type error would send the reader looking for a mistake
    # they did not make.
    if topics_blk is not None and not isinstance(topics_blk, Mapping):
        raise ConfigError("'topics' must be a mapping (e.g. `topics: {path: topics/....jsonl}`)")
    topics_path = topics_blk.get("path") if topics_blk is not None else None
    if not isinstance(topics_path, str) or not topics_path.strip():
        raise ConfigError("missing required key: 'topics.path' (a non-empty path to the topics file)")

    outputs_raw = raw.get("outputs")
    if not isinstance(outputs_raw, Sequence) or isinstance(outputs_raw, (str, bytes)):
        raise ConfigError("'outputs' must be a list of mappings")
    for i, item in enumerate(outputs_raw):
        if not isinstance(item, Mapping):
            raise ConfigError(f"outputs[{i}] must be a mapping")
        _reject_unknown_keys(item, _OUTPUT_ITEM, f"outputs[{i}]")
        # Load-bearing routing fields: a submission with no track, path or run_id would
        # route nowhere.
        for req in ("task", "track", "path", "run_id"):
            if req not in item:
                raise ConfigError(f"missing required key: 'outputs[{i}].{req}'")

    languages = tuple(raw.get("languages", ()))
    team_id = raw.get("team_id", "")

    # The full validated and defaulted block map, the input to all_hashes. Both knob
    # defaults are materialized symmetrically so that all_hashes has a uniform key set
    # across a family: a controlled e2e run without a retrieval block still gets a
    # "retrieval" key, mirroring how passage_lang is written back.
    defaulted = dict(raw)
    defaulted["passage_lang"] = passage_lang
    if not isinstance(retrieval, Mapping):
        defaulted["retrieval"] = {"index": retrieval_index}
    blocks = _freeze(defaulted)

    return RunConfig(
        run_id=run["id"],
        kind=kind,
        status=status,
        passage_lang=passage_lang,
        retrieval_index=retrieval_index,
        topics_path=topics_path,
        seeds=seeds,
        languages=languages,
        team_id=team_id,
        outputs=tuple(_freeze(o) for o in outputs_raw),
        blocks=blocks,
    )
