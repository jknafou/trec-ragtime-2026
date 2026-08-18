"""Run-directory layout: the single source of truth for artifact paths.

``Layout`` is the only translator between a run and its on-disk paths: the
per-round decompose files, the per-nugget RAG-loop files, the citation-score
directory, the corpus build, the index, the manifest and the success markers.
Submission paths are returned verbatim from the caller-supplied ``outputs``
mapping, so the reproducible config record and the emitted path cannot disagree.
This module imports nothing from the other packages; in particular it does not
import ``config``, and the caller passes ``outputs`` in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["INDEX_PART_GLOB", "MANIFEST_FRAGMENT_DIR", "Layout", "index_part_name"]

#: Per-run manifest fragment directory under the submission root. Defined here
#: rather than in `select_serialize.submission.manifest` so both sides share one
#: literal; `pipeline` may import `common` but not the reverse.
MANIFEST_FRAGMENT_DIR = "manifest.d"

# The family-shared corpus subtree, mirroring `orchestration.plan`'s convention:
# `plan.cell_artifact` writes a family-shared node under
# `<base>/corpus/<family>/<chunker_hash12>/<node>`, and the corpus node it plans is
# `plan.CORPUS`. The semantic chunker hash is a path level, so a chunker edit
# (token_budget, overlap_frac, segmenter_model, tokenizer_id) resolves to a fresh
# corpus directory and its `_SUCCESS` never masks a stale build. `common` cannot
# import `config`, so the hash is passed in by the caller. These constants and the
# truncation length must match `plan.cell_artifact`.
_CORPUS_ROOT = "corpus"
_CORPUS_NODE = "corpus-preprocess"
_HASH_DIRLEN = 12  # chunker-hash prefix length used as the corpus path level
# The merged passage spine is columnar (see `passages_path`); the per-shard chunk
# outputs it is built from stay JSONL.
_PASSAGES_SUFFIX = ".parquet"
# The one source language whose per-language cell is built once and referenced by
# every variant (see `index_lang_dir` and `vectors_cell_dir`). English is the target
# language of both MT arms, so its rendering is identity pass-through and the three
# renderings' English text is byte-identical. One constant serves both schemes, which
# express the same reuse.
_SHARED_LANG = "en"
#: The variant level a shared (English) cell resolves to, in both schemes.
_SHARED_VARIANT_DIR = "_shared"
#: The vector store's own level under one packing of the corpus (see `vectors_dir`).
_VECTORS_NODE = "vectors"

#: The stem every index shard part directory is named from: a ``(variant, source_lang)``
#: cell holds ``part-00000``, ``part-00001`` and so on, even when there is exactly one
#: part, so a missing part never looks like the normal case.
_INDEX_PART_PREFIX = "part"
#: What a reader globs to find the parts on disk, exported so the pattern and the name
#: come from the same literal.
INDEX_PART_GLOB = f"{_INDEX_PART_PREFIX}-*"


def index_part_name(part: int) -> str:
    """Return ``part-000NN``, one part of a ``(variant, source_lang)`` index cell.

    Zero-padded so the lexicographic on-disk order is the part order, which lets a
    census cross-check compare ``sorted(dirs)`` against the recorded names directly.
    """
    return f"{_INDEX_PART_PREFIX}-{int(part):05d}"


class Layout:
    """Path builder for one ``(run_id, variant, seed)`` run directory.

    Parameters
    ----------
    run_dir:
        Base directory for this run's artifacts.
    outputs:
        The ``config.outputs`` list, one entry per emitted deliverable, each a
        mapping with at least ``track`` and ``path``. Passed in by the caller rather
        than imported, and :meth:`submission` returns its ``path`` verbatim.
    """

    __slots__ = ("_chunker_hash", "_family", "_submissions", "base", "run_dir")

    def __init__(
        self,
        run_dir: str | Path,
        outputs: Sequence[Mapping[str, Any]] | None = None,
        *,
        base: str | Path | None = None,
        family: str | None = None,
        chunker_hash: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        # `base` is the artifact root (the `root` argument `plan.cell_artifact` takes)
        # under which the family-shared `corpus/` subtree lives. It is distinct from a
        # per-run `run_dir` (`<base>/<cell_key>`), and defaults to it only so existing
        # per-run callers are untouched.
        self.base = Path(base) if base is not None else self.run_dir
        # `family` and `chunker_hash` bind this Layout to one family's corpus build for
        # the outputs whose signature carries no family or hash argument.
        self._family = family
        self._chunker_hash = chunker_hash
        self._submissions: dict[str, str] = {}
        for entry in outputs or ():
            track = entry["track"]
            if track in self._submissions:
                raise ValueError(f"duplicate submission track in outputs: {track!r}")
            self._submissions[track] = entry["path"]

    # -- pipeline artifact paths -------------------------------------------- #
    def decompose_round(self, r: int) -> Path:
        """``<run>/decompose/round_{r}.jsonl``, one file per decompose round."""
        return self.run_dir / "decompose" / f"round_{r}.jsonl"

    def rag_loop(self, nugget_id: str) -> Path:
        """``<run>/rag_loop/{nugget_id}.jsonl``, one file per fanned-out nugget."""
        return self.run_dir / "rag_loop" / f"{nugget_id}.jsonl"

    def citation_scores(self) -> Path:
        """``<run>/citation_scores/``, the post-hoc citation scorer's output directory."""
        return self.run_dir / "citation_scores"

    def manifest(self) -> Path:
        """``<run>/manifest.json``, the run manifest path."""
        return self.run_dir / "manifest.json"

    def metrics(self) -> Path:
        """``<run>/metrics/counters.jsonl``, the cell's ``Statistics`` counters on disk.

        Per run directory, which for the online half is per ``(topic, variant, seed)``
        cell: that is the grain SLURM schedules and therefore the grain that can die
        independently. ``monitoring`` rolls these up across cells.

        ``Statistics.emit`` accumulates, so every cell is a sum and no distribution
        metric can come from this file; and the canonical slice vocabulary has no
        ``topic`` key, so per-topic quantities come from the artifact tree instead.
        """
        return self.run_dir / "metrics" / "counters.jsonl"

    def success(self) -> Path:
        """``<run>/_SUCCESS``, the run-level completion marker."""
        return self.run_dir / "_SUCCESS"

    # -- family-shared corpus and queue paths -------------------------------- #
    def corpus_dir(self, family: str, chunker_hash: str) -> Path:
        """``<base>/corpus/<family>/<chunker_hash12>/corpus-preprocess``.

        Byte-identical to ``orchestration.plan.cell_artifact`` for the family-shared
        corpus node. The semantic chunker hash is a path level, so a chunker edit
        resolves to a fresh directory and a completion marker never covers a stale
        corpus. Every other corpus helper hangs off this anchor.
        """
        return self.base / _CORPUS_ROOT / family / str(chunker_hash)[:_HASH_DIRLEN] / _CORPUS_NODE

    def corpus_raw_dir(self, family: str, chunker_hash: str) -> Path:
        """``<corpus_dir>/raw``, the byte-exact downloaded corpus store."""
        return self.corpus_dir(family, chunker_hash) / "raw"

    def corpus_slices_dir(self, family: str, chunker_hash: str) -> Path:
        """``<corpus_dir>/slices``, the per-shard document slices the corpus seed writes.

        One small compressed file per ``(file, doc-range)`` shard, materialized once by
        the seed role so a worker reads only its own slice instead of streaming a whole
        language file to its offset. Keyed by the same family and chunker hash as the
        rest of the corpus build, so a chunker edit re-shards into a fresh directory.
        """
        return self.corpus_dir(family, chunker_hash) / "slices"

    def _corpus_artifact(self, name: str) -> Path:
        """``<corpus_dir>/<name>`` for a family-bound Layout.

        Raises if the Layout was not bound to a family and chunker hash.
        """
        if self._family is None or self._chunker_hash is None:
            raise ValueError(
                f"{name} requires a family+chunker-hash-bound Layout "
                f"(construct with Layout(..., family=<run_family>, chunker_hash=<hash>))"
            )
        return self.corpus_dir(self._family, self._chunker_hash) / name

    def documents_path(self) -> Path:
        """``<corpus_dir>/documents.parquet``, the normalized document text.

        The single home of the corpus text (``schemas.Document`` /
        ``document_arrow_schema``): NFC, then the chunker's structural paragraph split,
        then the paragraphs joined by a space. Every sentence in :meth:`sentences_path`
        is a ``(start, end)`` span into this file's ``text`` column, so the two files
        are one artifact and must be read as a pair. Native text only; the translated
        renderings are per-sentence.
        """
        return self._corpus_artifact("documents.parquet")

    def sentences_path(self) -> Path:
        """``<corpus_dir>/sentences.parquet``, one row per segmented sentence, text-free.

        Under ``schemas.Sentence`` / ``sentence_arrow_schema``: the sentence's text is
        ``documents.text[start:end]`` rather than a second copy. Read both with
        ``io.iter_parquet_batches`` in constant memory, never with ``read_parquet``.
        """
        return self._corpus_artifact("sentences.parquet")

    def passages_path(self, rendering: str) -> Path:
        """``<corpus_dir>/passages/<rendering>.parquet``, the passage spine.

        Written by reconciliation, its sole producer; the chunk stage emits only
        :meth:`documents_path` and :meth:`sentences_path`, so two files named
        ``passages.parquet`` with different meanings cannot coexist. ``"native"`` is the
        id spine, while ``"omt"`` (NLLB-200-3.3B) and ``"omt_opus"`` (OPUS-MT) are the
        translated renderings. The run family and chunker hash are bound at construction,
        since this signature carries neither.

        Stored as Parquet with zstd rather than JSONL: the merged spine is several times
        smaller and scans far faster, particularly for a single id column. Per-shard chunk
        outputs stay JSONL; only this merged spine is columnar. Read it with
        ``io.iter_parquet``.
        """
        if self._family is None or self._chunker_hash is None:
            raise ValueError(
                "passages_path requires a family+chunker-hash-bound Layout "
                "(construct with Layout(..., family=<run_family>, chunker_hash=<hash>))"
            )
        return (
            self.corpus_dir(self._family, self._chunker_hash)
            / "passages"
            / f"{rendering}{_PASSAGES_SUFFIX}"
        )

    def merge_map_path(self, merge_hash: str, *, part: str = "data") -> Path:
        """``<corpus_dir>/merge_map/<merge_hash12>/<part>.parquet``.

        The translation-unit table: one row per non-English ``sentence_id``, under
        ``schemas.merge_map_arrow_schema``, saying which consecutive sentences MT must
        see as one string. Kept as a separate artifact rather than a column on
        :meth:`sentences_path`, because the sentence spine is the family's frozen chunk
        output and adding a column would change its bytes and invalidate the chunker hash.

        Hanging it off :meth:`corpus_dir` while keying its own level by the semantic
        ``merge`` block hash makes the path express dependence on both inputs: a new merge
        rule resolves to a fresh map beside the same spine, and a new spine resolves to a
        fresh corpus directory containing no map at all.
        """
        return (
            self._corpus_artifact("merge_map")
            / str(merge_hash)[:_HASH_DIRLEN]
            / f"{part}{_PASSAGES_SUFFIX}"
        )

    def translations_raw_path(
        self, variant: str, translation_hash: str, merge_hash: str, *, part: str = "data"
    ) -> Path:
        """``<corpus_dir>/translations_raw/<variant>/<t12>-<m12>/<part>.parquet``.

        The raw MT output table: one row per translation unit, with the section split
        markers left unscrubbed. Not ``translations/<variant>/...``, which
        belongs to reconciliation, whose output is the marker-free, sentence-keyed table
        everything downstream reads.

        Both semantic hashes appear in the path level. Keying by the ``translation`` block
        alone would let a re-run under a different merge map find a stale completion marker
        and reuse translations produced by the previous grouping; the unit boundaries are
        as much an input to the MT call as the beam size.

        The level is therefore ``<recipe12>-<input12>``: the ``translation`` hash names how
        the text was produced, and the second component names the work table it was produced
        from. For the ``omt`` arm that table is the merge map, hence the parameter name; for
        the ``omt_opus`` arm, which reads no merge map, it is the reconciled sentence
        inventory hash. The ``omt_opus`` raw table lives here under :meth:`corpus_dir`
        rather than inside :meth:`final_dir`, so that a ``reconcile`` edit changing nothing
        about which sentences exist does not orphan it by path.

        ``part`` separates the two producers of one variant's raw text, ``"data"`` for the
        GPU MT arm and ``"identity"`` for the English pass-through, so two substages never
        write one file while staying under the same node.
        """
        return (
            self._corpus_artifact("translations_raw")
            / variant
            / f"{str(translation_hash)[:_HASH_DIRLEN]}-{str(merge_hash)[:_HASH_DIRLEN]}"
            / f"{part}{_PASSAGES_SUFFIX}"
        )

    # -- reconciliation's final corpus node ---------------------------------- #
    def final_dir(self, reconcile_hash: str) -> Path:
        """``<corpus_dir>/final/<recon12>``, the node everything downstream reads.

        ``recon12`` is a composite key, ``H(chunker_hash, merge_hash, translation_hash,
        reconcile block)``. The final corpus is a join of three independently versioned
        inputs, so keying it by any one of them would leave the other two silently
        substitutable: a v1 sentence inventory beside a v2 translation would resolve to
        the same directory and be served as if coherent. Under a composite key that
        pairing is unaddressable rather than merely asserted against, and the manifest
        records the parent hashes in full so the composite stays auditable.
        """
        return self._corpus_artifact("final") / str(reconcile_hash)[:_HASH_DIRLEN]

    def final_sentences_path(self, reconcile_hash: str) -> Path:
        """``<final_dir>/sentences.parquet``, the final post-fusion sentence inventory.

        Uses the same pinned ``schemas.sentence_arrow_schema`` as the chunk stage's spine,
        because it is the same kind of thing: dense ids per document, each a verbatim span
        of ``documents.text``. Only which spans exist differs, since a merge unit whose
        split markers did not survive translation becomes one sentence spanning its
        constituents. This file, not ``sentences.parquet``, is the inventory the index and
        retrieval stages join on.
        """
        return self.final_dir(reconcile_hash) / "sentences.parquet"

    def final_passages_path(self, reconcile_hash: str, pack_hash: str | None) -> Path:
        """``<final_dir>/passages/<pack12>/passages.parquet``, the passage spine.

        Text-free (``schemas.final_passage_arrow_schema``): a passage is an ordered list
        of final sentence ids, so its native and translated renderings are joins rather
        than stored copies. Distinct from :meth:`passages_path`, which names the
        per-rendering text artifact of the earlier design.

        Two hashes, two questions, two levels. ``recon12``, the parent, answers which
        sentences these are, decided by fusion alone. ``pack12`` answers how they were
        grouped. The ``packing`` block was split out of ``reconcile`` precisely so that a
        packing edit cannot move ``final_dir``; if it could, the re-packed corpus would
        land in a fresh node and both renderings' translations already inside the old one
        would be orphaned by path.

        ``pack_hash`` is required and has no default. ``None`` is legal and resolves the
        legacy pre-``packing`` ``<final_dir>/passages.parquet``, but it has to be spelled
        out at the call site so that every legacy read is greppable.
        """
        return self._pack_node(reconcile_hash, pack_hash) / f"passages{_PASSAGES_SUFFIX}"

    def _pack_node(self, reconcile_hash: str, pack_hash: str | None) -> Path:
        """``<final_dir>/passages/<pack12>``, the node one packing of the corpus owns.

        Everything whose identity depends on how the sentences were grouped hangs off this
        anchor: the passage table, its rendering store, and the vector blocks encoded from
        it. A re-pack therefore resolves to a fresh node containing none of them, and no
        member of the set can move without the others.

        ``pack_hash=None`` is the legacy pre-``packing`` node, ``final_dir`` itself, where
        reconciliation wrote ``passages.parquet`` before packing became its own stage.
        """
        if pack_hash is None:
            return self.final_dir(reconcile_hash)
        return self.final_dir(reconcile_hash) / "passages" / str(pack_hash)[:_HASH_DIRLEN]

    def final_translations_path(self, reconcile_hash: str, variant: str) -> Path:
        """``<final_dir>/translations/<variant>.parquet``, marker-free and final-id-keyed.

        Named ``translations/`` against :meth:`translations_raw_path`'s
        ``translations_raw/``, and the raw table's ``text_raw`` column makes a mistaken
        read fail loudly rather than serve a marker-bearing string.
        """
        return self.final_dir(reconcile_hash) / "translations" / f"{variant}{_PASSAGES_SUFFIX}"

    def sentence_remap_path(self, reconcile_hash: str) -> Path:
        """``<final_dir>/remap.parquet``, the total map from old to final sentence id."""
        return self.final_dir(reconcile_hash) / "remap.parquet"

    def passage_store_path(self, reconcile_hash: str, pack_hash: str | None) -> Path:
        """``<final_dir>/passages/<pack12>/passages.lmdb``, the by-id rendering store.

        The destination environment of
        ``passage_store.LmdbPassageStore.build_from_final``: a derived copy of exactly the
        tables beside it, namely ``documents.parquet`` plus this node's sentences and
        translations and one packing of its passages. It sits under both keys, so a corpus
        rebuild resolves to a fresh ``final_dir`` with no store at all and a re-pack
        resolves to a fresh ``pack12`` directory with none either.

        ``pack_hash`` is required for the same reason as on :meth:`final_passages_path`;
        ``None`` resolves the legacy ``<final_dir>/passages.lmdb``.
        """
        return self._pack_node(reconcile_hash, pack_hash) / "passages.lmdb"

    # -- the vector store (the vectorize to assemble seam) ------------------- #
    def vectors_dir(
        self, reconcile_hash: str, pack_hash: str | None, leg: str, encode_hash: str
    ) -> Path:
        """``<pack_node>/vectors/<leg>/<encode12>``, one leg's vector node.

        The arguments are in path order, which is also the order of the questions they
        answer: which sentences, grouped how, encoded by which leg, under which encode
        recipe.

        The node sits under the packing node rather than beside it, because these blocks
        are the passages of exactly one packing, encoded. A re-pack therefore resolves to
        a fresh node with no vectors, while an ``index_build`` edit that moves no encode
        key leaves this path untouched and the whole encode is reused.

        Keyed by the per-leg encode hash, never by ``index_hash``, which also covers the
        assemble recipe and would re-encode the corpus for a change that touches no
        forward pass. The leg is its own level even though the encode hash folds it in:
        the hash makes collision impossible, the level makes the tree readable and lets
        one leg's blocks be listed, sized or deleted without a manifest join.

        The block level is absent. ``block-00000`` inside the cell belongs to
        ``preprocess.vectors``, exactly as ``plaid-00000`` inside an index part belongs to
        ``preprocess.index``: ``Layout`` owns every level down to the cell a stage fans
        over, and the stage owns the sub-division it alone can enumerate.
        """
        return (
            self._pack_node(reconcile_hash, pack_hash)
            / _VECTORS_NODE
            / str(leg)
            / str(encode_hash)[:_HASH_DIRLEN]
        )

    def vectors_cell_dir(
        self,
        reconcile_hash: str,
        pack_hash: str | None,
        leg: str,
        encode_hash: str,
        variant: str | None,
        source_lang: str,
    ) -> Path:
        """``<vectors_dir>/<variant|_shared>/<source_lang>``, one vector cell.

        The unit both halves of the split index build address: ``preprocess.vectorize``
        writes ``block-00000``, ``block-00001`` and so on into it, and
        ``preprocess.assemble`` streams a part's worth of those blocks out of it. It is
        the vector store's analogue of :meth:`index_lang_dir`, with the same argument
        order and the same English rule.

        ``source_lang == "en"`` always resolves to ``<vectors_dir>/_shared/en`` whatever
        ``variant`` says, for the same reason as in :meth:`index_lang_dir`: English is an
        identity pass-through in both MT arms, so all three renderings hold byte-identical
        English text under the same ``passage_id``. Collapsing the three paths to one
        means the corpus's largest language is encoded once, with no copy, symlink or
        hardlink step a later stage could forget. The block axis lives strictly inside the
        collapsed path. ``variant=None`` names the shared unit explicitly.
        """
        node = self.vectors_dir(reconcile_hash, pack_hash, leg, encode_hash)
        if source_lang == _SHARED_LANG:
            return node / _SHARED_VARIANT_DIR / _SHARED_LANG
        if variant is None:
            raise ValueError(
                f"vectors_cell_dir needs a variant for source_lang={source_lang!r}; "
                f"only {_SHARED_LANG!r} is variant-independent (encoded once, read by "
                "every variant's assemble)"
            )
        return node / variant / source_lang

    # -- the index node ------------------------------------------------------ #
    def index_dir(self, reconcile_hash: str, index_hash: str) -> Path:
        """``<final_dir>/index/<index_hash12>``, the three renderings' index node.

        Composite-keyed like :meth:`final_dir`, one level further in: the corpus identity
        is already the ``final_dir`` level, so this level carries only the build recipe,
        the hashed ``index_build`` block. A recipe edit resolves to a fresh index directory
        beside the same corpus with no re-translation, and a corpus edit resolves to a
        fresh ``final_dir`` that contains no index at all.
        """
        return self.final_dir(reconcile_hash) / "index" / str(index_hash)[:_HASH_DIRLEN]

    def index_lang_dir(
        self,
        reconcile_hash: str,
        index_hash: str,
        variant: str | None,
        source_lang: str,
    ) -> Path:
        """``<index_dir>/<variant>/<source_lang>``, one language cell of one rendering.

        This is the level a query fans over rather than the level a worker builds: it
        holds ``part-00000``, ``part-00001`` and so on (see :func:`index_part_name` and
        :meth:`index_shard_dir`). It is also the level the part census is read at, so a
        reader resolves which parts a cell has from one path rather than from a glob at an
        invented one.

        ``source_lang == "en"`` always resolves to ``<index_dir>/_shared/en`` whatever
        ``variant`` says, so the path scheme is itself the English-once reuse mechanism.
        English is an identity pass-through in both MT arms, so the three renderings hold
        byte-identical English text under the same ``passage_id``. Collapsing the paths
        means reuse needs no copy, symlink or hardlink step a later stage could forget:
        any variant's manifest naming its ``en`` cell names the one build, and that stays
        true part by part, because the part axis is a sub-division of this path rather
        than a sibling. ``variant=None`` names the shared unit explicitly.
        """
        if source_lang == _SHARED_LANG:
            return (
                self.index_dir(reconcile_hash, index_hash) / _SHARED_VARIANT_DIR / _SHARED_LANG
            )
        if variant is None:
            raise ValueError(
                f"index_lang_dir needs a variant for source_lang={source_lang!r}; "
                f"only {_SHARED_LANG!r} is variant-independent (built once, "
                "referenced by every variant's manifest)"
            )
        return self.index_dir(reconcile_hash, index_hash) / variant / source_lang

    def index_shard_dir(
        self,
        reconcile_hash: str,
        index_hash: str,
        variant: str | None,
        source_lang: str,
        *,
        part: int,
    ) -> Path:
        """``<index_lang_dir>/part-000NN``, one claimable shard.

        ``part`` is required and keyword-only: a defaulted part number would let a caller
        that has not thought about parts address part 0 of a cell that has ten, searching
        a tenth of a language and getting a full-looking answer. Callers that mean all of
        them use :meth:`index_lang_dir` and read the census instead.
        """
        return self.index_lang_dir(
            reconcile_hash, index_hash, variant, source_lang
        ) / index_part_name(part)

    def index_manifest_path(self, reconcile_hash: str, index_hash: str) -> Path:
        """``<index_dir>/manifest.json``, the published index's per-variant manifest.

        One file carrying one section per rendering rather than three sibling files,
        because the facts it records are cross-variant: the per-leg ``config_hash`` that
        must be equal on all three, and the ``en`` entry that must be the same shared path
        in all three. One artifact makes both readable from a single read.
        """
        return self.index_dir(reconcile_hash, index_hash) / "manifest.json"

    def sentence_len_max_path(self, reconcile_hash: str, len_max_hash: str) -> Path:
        """``<corpus_dir>/sentence_len_max/<recon12>-<lm12>/len_max.parquet``.

        One row per final sentence carrying its length in every rendering plus their
        maximum, which is what the packer targets so that "a passage fits the retrieval
        window" holds in every rendering rather than only the native one.

        Two hashes in one level, as in :meth:`translations_raw_path`, because the table is
        a join of two independent choices: ``recon12`` is which sentences these are, and
        ``lm12`` is how they were measured (tokenizer identity plus rendering set). It is
        not keyed by ``pack12``, since a length is a property of a sentence
        and no packing knob can change one, so a re-pack finds the sidecar it already paid
        for while a different inventory or tokenizer still resolves to a different file.

        Additive: nothing already built is re-keyed, re-read or invalidated by this
        file appearing beside it.
        """
        return (
            self._corpus_artifact("sentence_len_max")
            / f"{str(reconcile_hash)[:_HASH_DIRLEN]}-{str(len_max_hash)[:_HASH_DIRLEN]}"
            / f"len_max{_PASSAGES_SUFFIX}"
        )

    def final_manifest_path(self, reconcile_hash: str) -> Path:
        """``<final_dir>/manifest.json``: parent hashes, row counts and file checksums.

        The composite key's audit half: the directory name is a 12-hex digest, and this is
        where the parent hashes it was computed from are written down in full.
        """
        return self.final_dir(reconcile_hash) / "manifest.json"

    def wq_dir(self, family: str, chunker_hash: str, stage: str) -> Path:
        """``<corpus_dir>/queue/<stage>``, the self-claiming work-queue base for a substage.

        Lives under :meth:`corpus_dir`, the same anchor as everything else, so the
        ``WQ_DIR`` the sbatch templates require is Layout-derived rather than a second
        hardcoded path.
        """
        return self.corpus_dir(family, chunker_hash) / "queue" / stage

    def seed_bank(self, decompose_hash: str, topic_id: str, seed: int) -> Path:
        """``<base>/decompose_seeds/<dhash12>/<topic_id>/seed<N>.jsonl``, shared round 0.

        Round 0 belongs to the run family rather than to a run, and this path is that
        fact. The seed decomposition is retrieval-free by construction, so it reads the
        request and nothing else; its only inputs are the ``decomposition`` and ``llm``
        blocks, the topics file, the topic and the seed, and the first three are shared
        across every member of a family.

        Keyed by content rather than by family or run id: ``decompose_hash`` is
        ``H(decomposition, llm, topics)``, so two configs that agree on those three
        resolve to the same file and any config that changes one gets a fresh subtree.
        The reuse is therefore correct by construction rather than by naming convention,
        and the ``mlir`` arms can reuse the ``e2e`` seed banks.

        This is a correctness mechanism as well as a saving. The fairness invariant's
        runtime proof is that the round-0 bank is byte-identical across a family, and
        recomputing it once per arm would make that a hope about sampled decoding, which
        is not batch-invariant across instances. Computing it once and reading it three
        times makes the invariant hold by construction.
        """
        return (
            self.base
            / "decompose_seeds"
            / str(decompose_hash)[:_HASH_DIRLEN]
            / str(topic_id)
            / f"seed{int(seed)}.jsonl"
        )

    def vllm_registry_dir(self) -> Path:
        """``<base>/serving/vllm_endpoints``, the lease registry of live vLLM pairs.

        Not under :meth:`corpus_dir` and not per family. A vLLM instance is
        hardware serving a checkpoint, not an artifact of any family's corpus, and a run
        family does not own the cards it borrows. Keying it per family would partition one
        pool of pairs into several, leaving a free pair idle because it was published
        under a sibling's name. The filter that matters is ``(model, gpu_model)``, which
        ``serving.vllm_registry`` applies at claim time rather than by path.

        On the shared filesystem like every other queue base, because the exclusion is an
        atomic rename and every worker must be able to win its own independently.
        """
        return self.base / "serving" / "vllm_endpoints"

    def sif_path(self) -> Path:
        """``<base>/containers/ragtime-gpu.sif``, the Layout-derived ``RAGTIME_SIF`` path.

        Exported by ``cli._submit_dag`` from here rather than from a second hardcoded
        string. The chunk stage runs without Apptainer and never reads this.
        """
        return self.base / "containers" / "ragtime-gpu.sif"

    # -- submission paths (verbatim from config.outputs) --------------------- #
    def submission(self, track: str) -> Path:
        """Return the submission path for ``track`` verbatim from ``outputs``.

        Never invents a path. Raises ``KeyError`` if the track was not supplied, so a
        missing routing entry fails loudly rather than emitting to a guessed location.
        """
        try:
            return Path(self._submissions[track])
        except KeyError:
            raise KeyError(
                f"no submission output for track {track!r}; "
                f"known: {sorted(self._submissions)}"
            ) from None

    def submission_manifest(self) -> Path:
        """``<submission root>/manifest.json``, the upload record rather than a track file.

        Derived from ``outputs`` rather than invented: it is ``manifest.json`` under the
        one directory that is the common parent of every declared submission path. That
        keeps :meth:`submission`'s discipline, since the config record and the emitted
        path cannot disagree, while still giving the manifest a home that no ``outputs``
        entry declares because it is not a deliverable.

        Raises ``KeyError`` when no outputs were supplied and ``ValueError`` when the
        declared paths share no common parent, rather than guessing a root: a manifest
        written beside the wrong tree is a silently unfindable upload record.
        """
        if not self._submissions:
            raise KeyError(
                "no submission outputs were supplied, so there is no submission root to "
                "anchor the manifest to"
            )
        parents = {Path(p).parent.parent for p in self._submissions.values()}
        if len(parents) != 1:
            raise ValueError(
                "submission outputs do not share one root directory "
                f"({sorted(str(p) for p in parents)}); the manifest has no unambiguous home"
            )
        return parents.pop() / "manifest.json"

    def submission_manifest_fragment(self, run_id: str) -> Path:
        """``<submission root>/manifest.d/<run_id>.json``, one run's slice of the manifest.

        ``project()`` is a per-run call whose entries are only that run's files, whereas
        :meth:`submission_manifest` derives one path from the common parent of the declared
        outputs. Without a per-run fragment, every run would write a complete manifest to
        the same path and the last finisher would win.

        A fragment per run makes the write correct under concurrency: the runs are
        independent SLURM cells on different nodes over a shared filesystem, and each
        writes only a path keyed by its own ``run_id``, so there is no shared mutable file
        and no lost update. The family manifest is then a pure, idempotent reduction over
        this directory (``submission.manifest.assemble_family_manifest``), which is
        re-runnable and keeps the artifact tree as the checkpoint.
        """
        return self.submission_manifest().parent / MANIFEST_FRAGMENT_DIR / f"{run_id}.json"
