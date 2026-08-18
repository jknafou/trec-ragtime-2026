"""Corpus-scale acceptance pass over a built index: liveness and consistency, never quality.

There are no qrels, no dev set and no training data for this track, and there never will be.
Every check below asks whether the built artifact behaves the way a correctly assembled
index must behave, structurally. None of the numbers is an nDCG, a recall, a precision or
anything that may be quoted as "how good is the index"; where a number looks like a quality
metric (an overlap fraction, a rank, a margin) it is a liveness signal, and the surrounding
docstring says so again.

Readers are only the repo's own: :func:`~ragtime.preprocess.index.open_shard`,
:func:`~ragtime.preprocess.index.query_leg` / :func:`~ragtime.preprocess.index.search_with_rep`,
:func:`~ragtime.preprocess.index.shard_agreement`,
:func:`~ragtime.preprocess.index.read_shard_parts`,
:func:`~ragtime.preprocess.vectors.stream_part`,
:func:`~ragtime.common.passage_store.plan_final_ranges` /
:func:`~ragtime.common.passage_store.iter_final_passages`,
:func:`~ragtime.common.topics.load_topics`. A bare ``faiss``/``pyseismic_lsr``/``pylate``
call would validate something other than what production reads.

The checks run in phases, each its own SLURM job, each writing one JSON fragment:

* ``A``: id-set integrity and completeness. Parquet ids only, no container.
* ``B1``: self-retrieval on stored vectors, plus the no-text contract. No encoder is ever
  constructed: the :class:`~ragtime.preprocess.index.IndexCtx` this phase builds carries
  ``dense=milco=mtd=None``, so an accidental ``encode_query`` raises
  ``AttributeError`` instead of silently re-embedding text. One task per cell.
* ``B2``: cross-rendering agreement and cross-lingual sanity, with live encoders. One task
  per non-English language, since the agreement check excludes English (``omt`` and
  ``omt_opus`` are the same physical shard there) and the cross-lingual check is
  non-English by definition.

Step 0 runs in every phase and is not skippable: if the artifact under test is not
corpus-scale, this module raises :class:`AcceptanceError` before a single check runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragtime.common import Layout, get_logger
from ragtime.common.io import iter_parquet, read_jsonl
from ragtime.common.passage_store import iter_final_passages, plan_final_ranges
from ragtime.common.topics import CANONICAL_TOPICS_REL, load_topics
from ragtime.config import load as load_config
from ragtime.preprocess.assemble import AssembleAdapter, _reps_for
from ragtime.preprocess.index import (
    DENSE_LEG,
    IDMAP_FILENAME,
    LATE_INTERACTION_LEG,
    LEGS,
    SHARED_LANG,
    SPARSE_LEG,
    IndexCtx,
    _assert_parts_identical_across_renderings,
    _id_digest,
    _idmap_ids,
    _row_order_lang_ids,
    default_legs,
    index_build_options,
    leg_encode_hash,
    open_shard,
    read_shard_parts,
    search_with_rep,
    shard_agreement,
)
from ragtime.preprocess.vectors import read_cell_census, stream_part
from ragtime.serving.batching import Tier

__all__ = [
    "CORPUS_ACCEPTANCE_SEED",
    "MIN_CORPUS_PASSAGES",
    "AcceptanceError",
    "Probe",
    "cells",
    "main",
    "step0_precondition",
]

_log = get_logger(__name__)

#: One literal, never derived from the wall clock: so a re-run targets the identical
#: probes and a diff between two runs is attributable to the artifact, not to the sample.
#: (The date the pass was designed; any fixed literal would do.)
CORPUS_ACCEPTANCE_SEED = 20260805

#: Step 0's floor. The design's anchor is 9,941,840 passages; "within an order of magnitude"
#: is [~1e6, ~1e8]. The bounded-shard gate builds ~150-600 documents' worth (order 1e3), so
#: this floor cannot be met by the fixture: which is the point of the check.
MIN_CORPUS_PASSAGES = 1_000_000
MAX_CORPUS_PASSAGES = 100_000_000
#: The design's §1 anchor, recorded next to the measurement for the delta. Never asserted.
DESIGN_ANCHOR_PASSAGES = 9_941_840

#: §4's asymmetric depth: dense and sparse share one part sample; late-interaction (the
#: dominant I/O cost, ~9 GB per stored block) takes a strict subset of it.
PARTS_PER_CELL = 5
LATE_PARTS_PER_CELL = 3
PASSAGES_PER_PART = 10
#: FA03's part sample per language (same first/last-inclusion rule).
AGREEMENT_PARTS_PER_LANG = 5
#: FA04's probes per non-English language.
CROSS_LINGUAL_PROBES_PER_LANG = 3
TOP_K = 10

RENDERING_VARIANTS = ("original", "omt", "omt_opus")


class AcceptanceError(RuntimeError):
    """A precondition or a hard gate failed. Never raised for a recorded measurement."""


# --------------------------------------------------------------------------- #
# Resolution: config -> the artifact under test.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Resolved:
    """Everything every phase needs, resolved once from the config.

    ``base`` comes from ``orchestration.cli.artifact_root``, the one resolver, so this
    pass reads exactly the tree the build wrote, and a stray ``$RAGTIME_ARTIFACT_ROOT``
    disagreeing with the config is the same hard ``ConfigError`` it is everywhere else.
    """

    cfg: Any
    adapter: AssembleAdapter
    layout: Layout
    opts: Any
    recon_hash: str
    pack_hash: str | None
    idx_hash: str
    part_passages: int
    langs: tuple[str, ...]  # non-English, sorted
    encode_hashes: Mapping[str, str]

    @property
    def passages_path(self) -> Path:
        return self.layout.final_passages_path(self.recon_hash, self.pack_hash)

    def lang_dir(self, variant: str | None, source_lang: str) -> Path:
        return self.layout.index_lang_dir(
            self.recon_hash, self.idx_hash, variant, source_lang
        )

    def shard_dir(self, variant: str | None, source_lang: str, part: int) -> Path:
        return self.layout.index_shard_dir(
            self.recon_hash, self.idx_hash, variant, source_lang, part=part
        )

    def vectors_cell_dir(self, leg: str, variant: str | None, source_lang: str) -> Path:
        return self.layout.vectors_cell_dir(
            self.recon_hash,
            self.pack_hash,
            leg,
            self.encode_hashes[leg],
            variant,
            source_lang,
        )


def resolve(config_path: str | Path) -> _Resolved:
    """Load the config and resolve every path this pass will touch."""
    from ragtime.orchestration.cli import artifact_root

    cfg = load_config(config_path)
    root = artifact_root(cfg)
    adapter = AssembleAdapter.for_config(cfg, base=root)
    opts = index_build_options(cfg)
    return _Resolved(
        cfg=cfg,
        adapter=adapter,
        layout=adapter._layout(cfg),  # the adapter's own Layout, never a second path system
        opts=opts,
        recon_hash=adapter.recon_hash,
        pack_hash=adapter.pack_hash,
        idx_hash=adapter.idx_hash,
        part_passages=max(1, int(opts.plaid_part_passages)),
        langs=tuple(sorted(str(x) for x in cfg.languages if str(x) != SHARED_LANG)),
        encode_hashes={leg: leg_encode_hash(opts, leg) for leg in LEGS},
    )


def cells(res: _Resolved) -> list[tuple[str | None, str]]:
    """The 10 ``(variant, source_lang)`` cells, in a stable order (array-task index).

    English is one cell with ``variant=None``, ``Layout`` collapses it to ``_shared/en``,
    so comparing it across renderings would be comparing an index to itself.
    """
    return [(None, SHARED_LANG)] + [
        (variant, lang) for variant in RENDERING_VARIANTS for lang in res.langs
    ]


# --------------------------------------------------------------------------- #
# Step 0: the mandatory precondition. Fails loudly, never silently skips.
# --------------------------------------------------------------------------- #
def step0_precondition(res: _Resolved) -> dict[str, Any]:
    """Assert the artifact under test is the corpus-scale one. Not skippable.

    Reads the published manifest and every cell's part census (``read_shard_parts``, which
    cross-checks the census against the directories on disk rather than globbing), sums the
    witnessed passage counts, and refuses to run any check if the total is not within an
    order of magnitude of the design's ~9.9 M anchor. A green run against the bounded-shard
    fixture, or against a half-assembled tree, is worse than no run, because it produces a
    result nobody can trust.

    Also asserts the three renderings agree on each non-English language's witnessed count:
    they are cut from the same text-free table, so a disagreement means the tree is mixed.
    """
    manifest_path = res.layout.index_manifest_path(res.recon_hash, res.idx_hash)
    if not manifest_path.exists():
        raise AcceptanceError(
            f"{manifest_path}: no published index manifest. The assemble stage publishes it "
            "only after merge() has asserted cross-variant id integrity over every part, so "
            "its absence means the corpus-scale build has not finished, this pass must not "
            "validate a partial index."
        )
    manifest = read_jsonl(manifest_path)[0]
    # The manifest must describe the artifact the CONFIG resolves, not some other build that
    # happens to sit at this path. Key names are the ones assemble.merge() writes.
    stated = {
        "index_hash": str(manifest.get("index_hash") or ""),
        "reconcile_hash": str(manifest.get("reconcile_hash") or ""),
        "packing_hash": str(manifest.get("packing_hash") or ""),
    }
    resolved = {
        "index_hash": res.idx_hash,
        "reconcile_hash": res.recon_hash,
        "packing_hash": res.pack_hash or "",
    }
    drift = {k: (stated[k], resolved[k]) for k in stated if stated[k] != resolved[k]}
    if drift:
        raise AcceptanceError(
            f"{manifest_path}: the published manifest's identities disagree with the ones "
            f"{res.cfg.source_path} resolves ({drift}), this pass would be validating a "
            "different build than the config names. Fix path/config resolution first."
        )

    per_cell: dict[str, dict[str, Any]] = {}
    for variant, lang in cells(res):
        lang_dir = res.lang_dir(variant, lang)
        if not lang_dir.exists():
            raise AcceptanceError(f"{lang_dir}: cell directory is missing")
        census = read_shard_parts(lang_dir)
        per_cell[_cell_name(variant, lang)] = {
            "dir": str(lang_dir),
            "parts": int(census["parts"]),
            "part_passages": int(census["part_passages"]),
            "passages": int(sum(census["passages"])),
        }

    by_lang: dict[str, int] = {}
    for lang in (SHARED_LANG, *res.langs):
        if lang == SHARED_LANG:
            by_lang[lang] = per_cell[_cell_name(None, SHARED_LANG)]["passages"]
            continue
        counts = {v: per_cell[_cell_name(v, lang)]["passages"] for v in RENDERING_VARIANTS}
        if len(set(counts.values())) != 1:
            raise AcceptanceError(
                f"the three renderings of {lang!r} witness different passage counts "
                f"({counts}), parts are cut on the one text-free passage table, which has "
                "no variant column, so this is structurally impossible from one build. The "
                "tree is mixed (a stale artifact root, or two pack/recon hashes)."
            )
        by_lang[lang] = next(iter(counts.values()))

    total = sum(by_lang.values())
    if not (MIN_CORPUS_PASSAGES <= total <= MAX_CORPUS_PASSAGES):
        raise AcceptanceError(
            f"step 0 failed: the artifact under test covers {total:,} passages, outside "
            f"[{MIN_CORPUS_PASSAGES:,}, {MAX_CORPUS_PASSAGES:,}], i.e. it is not the "
            f"corpus-scale index (design anchor {DESIGN_ANCHOR_PASSAGES:,}). This pass "
            "exists to validate the real thing; it refuses to run against a bounded-shard "
            f"fixture or a partial build. Per language: {by_lang}."
        )
    _log.info(
        "acceptance.step0",
        passages=total,
        anchor=DESIGN_ANCHOR_PASSAGES,
        by_lang=by_lang,
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_identities": stated,
        "manifest_created_at": manifest.get("created_at"),
        # The manifest's own per-variant coverage count: a second, independent witness of
        # corpus scale beside the census sum above (they are computed from different things).
        "manifest_variant_passages": {
            v: int(d.get("passages") or 0)
            for v, d in sorted((manifest.get("variants") or {}).items())
        },
        "manifest_leg_encode_hash": manifest.get("leg_encode_hash"),
        "recon_hash": res.recon_hash,
        "pack_hash": res.pack_hash,
        "index_hash": res.idx_hash,
        "part_passages": res.part_passages,
        "witnessed_passages_total": total,
        "witnessed_passages_by_lang": by_lang,
        "design_anchor_passages": DESIGN_ANCHOR_PASSAGES,
        "delta_vs_anchor": total - DESIGN_ANCHOR_PASSAGES,
        "cells": per_cell,
        "passed": True,
    }


def _cell_name(variant: str | None, lang: str) -> str:
    return f"{variant or '_shared'}/{lang}"


# --------------------------------------------------------------------------- #
# Sampling: one seeded RNG, every probe recorded.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Probe:
    """One recorded self-retrieval probe. Written into the report so a re-run repeats it."""

    variant: str | None
    source_lang: str
    part: int
    ordinal: int
    passage_id: str

    def payload(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "source_lang": self.source_lang,
            "part": self.part,
            "ordinal": self.ordinal,
            "passage_id": self.passage_id,
        }


def _rng(*key: Any) -> random.Random:
    """A Random seeded by the module constant and the sampling key.

    Per-key rather than one shared stream: each array task samples its own
    cell independently, so the sample must not depend on which tasks ran, or in what order.
    """
    return random.Random(f"{CORPUS_ACCEPTANCE_SEED}:" + ":".join(str(k) for k in key))


def sample_parts(rng: random.Random, parts: int, want: int) -> list[int]:
    """``want`` parts, always including part 0 and part ``parts-1``.

    Boundary-adjacent by construction, because the defects this check exists for are
    part-boundary defects (a token-sorted cut validated against a row-order artifact) and a
    flat random sample over ids would not stress that class nearly as directly.
    """
    if parts <= want:
        return list(range(parts))
    forced = sorted({0, parts - 1})
    rest = [p for p in range(parts) if p not in forced]
    chosen = forced + rng.sample(rest, max(0, want - len(forced)))
    return sorted(chosen)


def sample_ordinals(rng: random.Random, part_passages: int, want: int) -> list[int]:
    """``want`` part-local ordinals, shared by every leg probing this part."""
    if part_passages <= want:
        return list(range(part_passages))
    return sorted(rng.sample(range(part_passages), want))


# --------------------------------------------------------------------------- #
# Phase A: FA02 (id-set integrity) + FA06 (completeness). Ids only; no SIF.
# --------------------------------------------------------------------------- #
def _leg_id_sets(shard_dir: Path) -> dict[str, list[str]]:
    """One part's ordinal-ordered ids, per leg, from each leg's own ``idmap.parquet``."""
    return {leg: _idmap_ids(shard_dir / leg) for leg in LEGS}


def phase_a(res: _Resolved) -> dict[str, Any]:
    """FA02 + FA06: both exhaustive (no sampling) because both are id-only and cheap.

    FA02 re-derives, post-hoc and against the bytes on disk, the cross-rendering
    part-membership identity ``assemble.merge()`` asserts at build time, deliberate
    defense-in-depth against a stale tree, which a build-time assertion cannot catch. It
    compares what each rendering's idmaps actually hold (not the shared table to itself,
    which would be circular) and also compares each against the row-order cut
    ``_row_order_lang_ids`` implies.

    FA06 asserts every passage of a language sits in exactly one part: the witnessed census
    sum equals the table's own per-language row count, and ``Σ|ids| == |∪ids|`` so a
    duplicate with a compensating drop cannot hide behind a matching sum.

    Both are hard gates. A failure indicts tooling/paths, not the corpus and not the
    retrieval method, investigate path resolution before considering any re-build.
    """
    started = time.time()
    lang_counts = res.adapter.lang_counts(res.cfg)  # one column-pruned scan of `lang`
    out: dict[str, Any] = {
        "fa02": {"gate": "hard", "cells": {}, "failures": []},
        "fa06": {"gate": "hard", "languages": {}, "failures": []},
        "lang_counts_from_table": dict(sorted(lang_counts.items())),
    }
    digests: dict[tuple[str, int], dict[str, str]] = {}

    for lang in (SHARED_LANG, *res.langs):
        variants: list[str | None] = (
            [None] if lang == SHARED_LANG else list(RENDERING_VARIANTS)
        )
        census0 = read_shard_parts(res.lang_dir(variants[0], lang))
        parts = int(census0["parts"])
        # The table's own row-order cut, materialized once per language (never per part:
        # that would re-scan passages.parquet `parts` times).
        table_ids, implied_parts = _row_order_lang_ids(
            res.passages_path, lang, res.part_passages, list(range(parts))
        )
        if implied_parts != parts:
            out["fa06"]["failures"].append(
                f"{lang}: the cells claim {parts} part(s) but the table implies "
                f"{implied_parts}, the index was built under a stale plan"
            )

        per_variant_union: dict[str, set[str]] = {}
        per_variant_sum: dict[str, int] = {}
        for variant in variants:
            name = _cell_name(variant, lang)
            census = read_shard_parts(res.lang_dir(variant, lang))
            union: set[str] = set()
            total = 0
            cell_report: dict[str, Any] = {"parts": int(census["parts"]), "part_ids": {}}
            for part in range(int(census["parts"])):
                shard = res.shard_dir(variant, lang, part)
                by_leg = _leg_id_sets(shard)
                sets = {leg: set(ids) for leg, ids in by_leg.items()}
                if len({_id_digest(s) for s in sets.values()}) != 1:
                    out["fa02"]["failures"].append(
                        f"{name} part {part}: the three legs' idmaps hold different id sets "
                        f"({ {leg: len(s) for leg, s in sets.items()} }), this part is not "
                        "covered by all three legs"
                    )
                ids = sets[DENSE_LEG]
                digest = _id_digest(ids)
                digests.setdefault((lang, part), {})[variant or "_shared"] = digest
                cell_report["part_ids"][str(part)] = {
                    "n": len(ids),
                    "digest": digest,
                    "matches_table_row_order_cut": digest == _id_digest(set(table_ids[part])),
                }
                if not cell_report["part_ids"][str(part)]["matches_table_row_order_cut"]:
                    out["fa02"]["failures"].append(
                        f"{name} part {part}: the idmap's id set is not the row-order slice "
                        f"[{part * res.part_passages}, {(part + 1) * res.part_passages}) of "
                        f"{lang} in {res.passages_path}"
                    )
                union |= ids
                total += len(ids)
            per_variant_union[variant or "_shared"] = union
            per_variant_sum[variant or "_shared"] = total
            cell_report["census_passages"] = int(sum(census["passages"]))
            out["fa02"]["cells"][name] = cell_report

        truth = int(lang_counts.get(lang, 0))
        full_table_ids = set().union(*(set(s) for s in table_ids.values()))
        lang_report: dict[str, Any] = {
            "table_rows": truth,
            "parts": parts,
            "by_variant": {},
            "table_id_set_size": len(full_table_ids),
        }
        for variant in variants:
            key = variant or "_shared"
            census = read_shard_parts(res.lang_dir(variant, lang))
            witnessed = int(sum(census["passages"]))
            disjoint = per_variant_sum[key] == len(per_variant_union[key])
            covers = per_variant_union[key] == full_table_ids
            lang_report["by_variant"][key] = {
                "witnessed_census_passages": witnessed,
                "sum_of_part_id_counts": per_variant_sum[key],
                "union_of_part_ids": len(per_variant_union[key]),
                "parts_are_disjoint": disjoint,
                "union_equals_table_id_set": covers,
                "census_equals_table_rows": witnessed == truth,
            }
            if witnessed != truth:
                out["fa06"]["failures"].append(
                    f"{_cell_name(variant, lang)}: census witnesses {witnessed} passages, "
                    f"the table holds {truth} rows for {lang}"
                )
            if not disjoint:
                out["fa06"]["failures"].append(
                    f"{_cell_name(variant, lang)}: Σ|part ids| = {per_variant_sum[key]} but "
                    f"|∪ ids| = {len(per_variant_union[key])}, a passage is in two parts"
                )
            if not covers:
                out["fa06"]["failures"].append(
                    f"{_cell_name(variant, lang)}: the union of its parts' ids is not the "
                    f"{lang} id set of {res.passages_path}"
                )
        out["fa06"]["languages"][lang] = lang_report
        del table_ids, full_table_ids, per_variant_union

    # The existing build-time assertion, re-run post-hoc over what the disk holds.
    try:
        _assert_parts_identical_across_renderings(digests)
        out["fa02"]["parts_identical_across_renderings"] = True
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        out["fa02"]["parts_identical_across_renderings"] = False
        out["fa02"]["failures"].append(str(exc))

    out["fa02"]["passed"] = not out["fa02"]["failures"]
    out["fa06"]["passed"] = not out["fa06"]["failures"]
    out["wall_clock_s"] = round(time.time() - started, 1)
    return out


# --------------------------------------------------------------------------- #
# Phase B1: FA01 (self-retrieval on stored vectors) + FA05 (no-text contract).
# --------------------------------------------------------------------------- #
def reader_ctx(res: _Resolved) -> IndexCtx:
    """An :class:`IndexCtx` that can open and search but cannot encode.

    ``dense``/``milco``/``mtd`` are ``None`` deliberately. Every leg's ``open`` and
    ``search`` read only ``ctx.opts``; only ``encode_query``/``encode_docs`` touch a client.
    So this context makes "B1 re-encodes nothing" a property of the object (an accidental
    encode raises ``AttributeError`` on ``None``) rather than a discipline at the call site,
    and it keeps three multi-GB models off a job that has no use for them.
    """
    return IndexCtx(
        dense=None,
        milco=None,
        mtd=None,
        opts=res.opts,
        legs=default_legs(),
        layout=res.layout,
        recon_hash=res.recon_hash,
        pack_hash=res.pack_hash or "",
        idx_hash=res.idx_hash,
        tier=Tier(
            token_budget=res.opts.encode_batch_token_budget,
            max_items=res.opts.encode_max_items,
        ),
    )


def _idmap_columns(shard_dir: Path, leg: str) -> list[str]:
    """A leg's published ``idmap.parquet`` column set: FA05's structural half."""
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(shard_dir / leg / IDMAP_FILENAME).schema_arrow.names)


#: Column names that would carry passage text: the thing FA05 exists to prove is absent.
#: The three renderings are named because they are what a "helpful" writer would add.
_TEXT_BEARING_COLUMNS = frozenset({"text", "text_raw", *RENDERING_VARIANTS, "passage_text"})


def _idmap_schema_failures(
    cell: str, part: int, leg: str, cols: Sequence[str]
) -> list[str]:
    """FA05's structural half, pinned against the code's schema, not a hardcoded triple.

    The written design named a three-column set, ``{ordinal, passage_id, document_id}``, but
    the shipped pin (:func:`~ragtime.preprocess.index.idmap_arrow_schema`) has four: the
    fourth is ``source_lang``, which carries no text and is there so a shard is
    self-describing. Asserting the design's triple would fail on a correct artifact.

    So this asserts the two things the check is actually for, and both are strictly stronger
    than the triple: (1) the column list is exactly the pinned schema's, in order, so any
    drift from the contract, added column or dropped column alike, is caught; (2) no
    text-bearing column name appears, named explicitly so the intent survives a future
    schema change that legitimately adds a fifth non-text column.
    """
    from ragtime.preprocess.index import idmap_arrow_schema

    expected = [f.name for f in idmap_arrow_schema()]
    out: list[str] = []
    if list(cols) != expected:
        out.append(
            f"{cell} part {part} {leg}: idmap columns are {list(cols)}, not the pinned "
            f"idmap_arrow_schema() order {expected}, the id map is not the published contract"
        )
    leaked = sorted(_TEXT_BEARING_COLUMNS.intersection(cols))
    if leaked:
        out.append(
            f"{cell} part {part} {leg}: idmap carries text-bearing column(s) {leaked}, the "
            "search/display decoupling is broken: a search could return passage text"
        )
    return out


def _stored_reps(
    res: _Resolved,
    leg: str,
    variant: str | None,
    lang: str,
    part: int,
    ordinals: Sequence[int],
) -> Iterator[tuple[int, str, str, Any]]:
    """``(ordinal, passage_id, document_id, rep)`` for ``ordinals`` of one part.

    Through ``vectors.stream_part``, the census-validated, sequential reader assemble
    itself used, so the representation searched with is byte-for-byte the one that was
    indexed. Never a re-encode: a re-encode would test the encoder (and, on a CPU node, a
    different device than the one that wrote the vectors), not the index.
    """
    cell = res.vectors_cell_dir(leg, variant, lang)
    census = read_cell_census(cell)
    wanted = sorted(int(o) for o in ordinals)
    for block in stream_part(
        cell, part=part, part_passages=res.part_passages, census=census
    ):
        lo, hi = block.ordinals.start, block.ordinals.stop
        for ordinal in wanted:
            if lo <= ordinal < hi:
                i = ordinal - lo
                ref = block.refs[i]
                yield ordinal, ref.passage_id, ref.document_id, block.reps[i]
        del block  # a late-interaction block is ~9 GB; do not hold two


def _query_form(leg: str, rep: Any) -> Any:
    """A stored document representation in the shape that leg's ``search`` takes as a query.

    Two adaptations, both exact, neither of them a change to the vector:

    * sparse, the stored form is ``seismic_components``' own ``(components, values)`` pair,
      and ``SeismicSparseLeg.search`` re-derives that pair from a mapping. The conversion is
      ``assemble._reps_for``'s, reused rather than re-written, and it is the identity (the
      components are already the stringified dim ids, distinct and sorted by that string).
    * late-interaction, the stored dtype is ``float16`` (``late_interaction_store_dtype``)
      while the query side of fast-plaid runs in float32. Widening fp16 -> fp32 is lossless
      (every fp16 is exactly representable in fp32), so this changes no value; it only avoids
      a dtype mismatch on the query path, which the document path never sees.
    """
    if leg == SPARSE_LEG:
        return _reps_for(leg, [rep])[0]
    if leg == LATE_INTERACTION_LEG:
        import numpy as np

        return np.asarray(rep, dtype="float32")
    return rep


#: Tolerance for "the same score" when deciding a duplicate tie. Scores come back as float32
#: inner products of unit-normalized vectors, so a self-match lands at 1.0 +/- one float32 ulp
#: (observed: 1.0000001192092896). 1e-6 is ~8x that and far below any real score gap.
_TIE_EPS = 1e-6


def duplicate_tie_explains_miss(row: Mapping[str, Any]) -> bool:
    """Is this FA01-dense rank-0 miss explained by a byte-identical duplicate, or is it a defect?

    Why a tie is the only benign explanation. BGE-M3 vectors are unit-normalized and searched
    under exact ``IndexFlatIP``, so ``<v, d> <= 1`` for every indexed ``d`` with equality only at
    ``d == v``. A probe queried with its own stored vector therefore cannot be beaten, only tied,
    and a tie means another row holds the same vector. So: every competitor sitting at the top score
    means the top-k window is saturated with maximum-scoring rows and FAISS's arbitrary tie-break
    pushed the probe out. Any other shape (competitors at varied scores below the maximum) means the
    probe's vector is not where the id map says it is, a mis-keyed idmap, which is the defect class
    this gate exists to catch, and which must still fail.

    Measured on the built corpus, three of ten cells failed this gate purely on ties, while an
    independent census of every dense vector found no NaN and no zero-norm rows. Exact
    duplicate vectors are well under one percent of rows and are enriched at part boundaries,
    so a saturated window is expected rather than exotic.

    Trap this function exists next to: a miss records ``own_score = float("nan")`` as a sentinel
    for "not in the returned top-k" (``own = ... if rank >= 0 else float("nan")``). That NaN is not
    a computed score and not evidence of a corrupt vector, it was read as one before this was
    written. Judge on ``top_score`` and ``competitors`` only.
    """
    comps = row.get("competitors") or []
    top = row.get("top_score")
    if top is None or not comps:
        # No evidence recorded => cannot be explained away. Fail closed.
        return False
    try:
        top_f = float(top)
    except (TypeError, ValueError):
        return False
    if math.isnan(top_f):  # NaN top_score: no usable evidence
        return False
    for c in comps:
        s = c.get("score")
        if s is None:
            return False
        try:
            sf = float(s)
        except (TypeError, ValueError):
            return False
        if math.isnan(sf) or abs(sf - top_f) > _TIE_EPS:
            return False
    return True


def phase_b1(res: _Resolved, cell_index: int) -> dict[str, Any]:
    """FA01 + FA05 for one ``(variant, source_lang)`` cell.

    FA01-dense is a hard gate at rank 0. BGE-M3 vectors are unit-normalized and searched
    under exact ``IndexFlatIP``, so ``⟨v, d⟩ ≤ 1`` for every indexed ``d`` with equality only
    at ``d = v``: a stored vector queried against its own index cannot be beaten, only tied.
    The one named exception is a byte-identical duplicate passage in the same part, which
    FAISS's arbitrary tie-break can put first; a rank-0 miss is therefore recorded with the
    tied competitors' scores so the duplicate explanation is checkable before the miss is
    called a defect.

    FA01-sparse and FA01-late-interaction are soft. Only presence in the top-k is
    asserted; the rank and the score gap to the top hit are recorded measurements with no
    threshold. Demanding rank 0 of MILCO's unnormalized SPLADE scores or of Seismic's pruned
    postings would assert something neither method ever promised.

    FA05 is a hard gate and free: every return is checked to be a 2-tuple
    ``(str, float)``, and each leg's ``idmap.parquet`` column set is read and required to be
    exactly ``{ordinal, passage_id, document_id}``, no ``text`` column anywhere reachable.
    """
    started = time.time()
    variant, lang = cells(res)[cell_index]
    name = _cell_name(variant, lang)
    census = read_shard_parts(res.lang_dir(variant, lang))
    parts = int(census["parts"])

    rng = _rng("b1", name)
    chosen_parts = sample_parts(rng, parts, PARTS_PER_CELL)
    late_parts = sample_parts(_rng("b1-late", name), len(chosen_parts), LATE_PARTS_PER_CELL)
    late_parts = [chosen_parts[i] for i in late_parts]

    ctx = reader_ctx(res)
    probes: list[dict[str, Any]] = []
    results: dict[str, list[dict[str, Any]]] = {leg: [] for leg in LEGS}
    fa05: dict[str, Any] = {"gate": "hard", "idmap_columns": {}, "failures": []}
    shape_violations: list[str] = []
    bytes_read = {leg: 0 for leg in LEGS}

    for part in chosen_parts:
        shard_dir = res.shard_dir(variant, lang, part)
        handle = open_shard(shard_dir, ctx)
        n = int(census["passages"][part])
        ordinals = sample_ordinals(_rng("b1-ord", name, part), n, PASSAGES_PER_PART)
        legs_here = list(LEGS) if part in late_parts else [DENSE_LEG, SPARSE_LEG]
        for leg in legs_here:
            cols = _idmap_columns(shard_dir, leg)
            fa05["idmap_columns"][f"{name}/part-{part:05d}/{leg}"] = cols
            fa05["failures"].extend(_idmap_schema_failures(name, part, leg, cols))
            cell_dir = res.vectors_cell_dir(leg, variant, lang)
            bytes_read[leg] += _part_bytes(cell_dir, part, res.part_passages)
            for ordinal, pid, _doc, rep in _stored_reps(
                res, leg, variant, lang, part, ordinals
            ):
                if leg == DENSE_LEG:
                    probes.append(
                        Probe(variant, lang, part, ordinal, pid).payload()
                    )
                hits = search_with_rep(handle, leg, _query_form(leg, rep), TOP_K)
                for item in hits:
                    if not (
                        type(item) is tuple
                        and len(item) == 2
                        and isinstance(item[0], str)
                        and isinstance(item[1], float)
                    ):
                        shape_violations.append(f"{name} part {part} {leg}: {item!r}")
                ids = [h[0] for h in hits]
                rank = ids.index(pid) if pid in ids else -1
                top_score = float(hits[0][1]) if hits else float("nan")
                own = float(hits[rank][1]) if rank >= 0 else float("nan")
                results[leg].append(
                    {
                        "part": part,
                        "ordinal": ordinal,
                        "passage_id": pid,
                        "rank": rank,
                        "own_score": own,
                        "top_score": top_score,
                        "gap_to_top": (top_score - own) if rank >= 0 else None,
                        # Recorded only for a dense rank-0 miss: the ids/scores that tied or
                        # beat the probe are what makes the duplicate explanation checkable.
                        "competitors": (
                            [
                                {"passage_id": i, "score": s}
                                for i, s in (hits[: rank + 1] if rank >= 0 else hits)
                            ]
                            if leg == DENSE_LEG and rank != 0
                            else None
                        ),
                    }
                )
        del handle

    if shape_violations:
        fa05["failures"].extend(shape_violations)
    fa05["passed"] = not fa05["failures"]
    fa05["probes_checked"] = sum(len(v) for v in results.values())

    fa01: dict[str, Any] = {"cell": name, "by_leg": {}}
    for leg in LEGS:
        rows = results[leg]
        ranks = [r["rank"] for r in rows]
        found = [r for r in ranks if r >= 0]
        fa01["by_leg"][leg] = {
            "gate": "hard-rank0" if leg == DENSE_LEG else "soft-retrieval-only",
            "probes": len(rows),
            "retrieved_in_top_k": len(found),
            "rank0": sum(1 for r in ranks if r == 0),
            "misses": [r for r in rows if r["rank"] != 0]
            if leg == DENSE_LEG
            else [r for r in rows if r["rank"] < 0],
            "rank_histogram": {
                str(r): ranks.count(r) for r in sorted(set(ranks))
            },
            "mean_gap_to_top": _mean(
                [r["gap_to_top"] for r in rows if r["gap_to_top"] is not None]
            ),
        }
    # The gate accepts rank 0, or a tie at the maximum explained by a byte-identical
    # duplicate (see duplicate_tie_explains_miss for why a tie can only be a duplicate).
    # A census of every dense vector found no NaN and no zero-norm rows; exact-duplicate
    # vectors are 0.78% of rows, enriched 12-18x at part boundaries, which is why three of
    # ten cells on the built corpus failed this gate on ties alone.
    dense_rows = results[DENSE_LEG]
    unexplained = [
        r for r in dense_rows
        if r.get("rank") != 0 and not duplicate_tie_explains_miss(r)
    ]
    fa01["dense_misses_explained_by_duplicate_tie"] = sum(
        1 for r in dense_rows if r.get("rank") != 0 and duplicate_tie_explains_miss(r)
    )
    fa01["dense_unexplained_misses"] = [
        {k: r.get(k) for k in ("part", "ordinal", "passage_id", "rank", "top_score")}
        for r in unexplained
    ]
    fa01["dense_hard_gate_passed"] = not unexplained
    fa01["note"] = (
        "sparse / late-interaction ranks are recorded measurements with no pass/fail "
        "threshold, no qrels exist to derive one. Only the dense leg's rank-0 is a gate."
    )

    return {
        "cell": name,
        "variant": variant,
        "source_lang": lang,
        "parts_in_cell": parts,
        "sampled_parts": chosen_parts,
        "sampled_parts_late_interaction": late_parts,
        "seed": CORPUS_ACCEPTANCE_SEED,
        "probes": probes,
        "fa01": fa01,
        "fa05": fa05,
        "bytes_read_by_leg": bytes_read,
        "wall_clock_s": round(time.time() - started, 1),
    }


def _part_bytes(cell_dir: Path, part: int, part_passages: int) -> int:
    """Bytes on disk of the vector blocks this part's read touches (for the I/O report)."""
    try:
        census = read_cell_census(cell_dir)
        plan = census.plan(part_passages)
        from ragtime.preprocess.vectors import block_dir_name

        total = 0
        for block in plan.blocks_of_part(part):
            d = cell_dir / block_dir_name(block)
            total += sum(f.stat().st_size for f in d.iterdir() if f.is_file())
        return total
    except Exception:  # noqa: BLE001 - a byte count is reporting, never a gate
        return -1


def _mean(values: Sequence[float]) -> float | None:
    import math

    vals = [float(v) for v in values if not math.isnan(float(v))]
    return round(sum(vals) / len(vals), 6) if vals else None


# --------------------------------------------------------------------------- #
# Phase B2: FA03 (cross-rendering agreement) + FA04 (cross-lingual sanity).
# --------------------------------------------------------------------------- #
def _encoder_ctx(res: _Resolved) -> IndexCtx:
    """The three live encoders, from the serving singleton registry (invariant 3).

    ``IndexAdapter.bringup`` is the one sanctioned construction path (it also re-asserts the
    encoders match the hashed recipe they will be recorded as). Note honestly, and in the
    report: on a CPU node the clients resolve to CPU while the corpus vectors were encoded
    on GPU. For FA03 that is harmless, the query representation is a shared constant across
    the two arms being compared, and for FA04 it is a caveat on a number that is already a
    recorded measurement, not a gate.
    """
    from ragtime.orchestration.cli import artifact_root
    from ragtime.preprocess.index import IndexAdapter

    adapter = IndexAdapter(
        recon_hash=res.recon_hash,
        pack_hash=res.pack_hash,
        idx_hash=res.idx_hash,
        base=artifact_root(res.cfg),
    )
    return adapter.bringup(res.cfg)


def phase_b2(
    res: _Resolved, lang_index: int, topics_path: str | Path, out_path: Path
) -> dict[str, Any]:
    """FA03 + FA04 for one non-English language. Both are recorded measurements.

    FA03, the primary, causally clean comparison: ``omt`` vs ``omt_opus`` is monolingual
    English retrieval over the same passages (FA02 proves the id sets are identical), so a
    measured gap is attributable to MT quality alone. It reuses
    :func:`~ragtime.preprocess.index.shard_agreement` verbatim. There is no threshold;
    read the numbers against the already-measured re-encode noise floor, and note that a
    near-100 % overlap is as worth a manual spot-check as a near-zero one (the first could
    be path aliasing, the second genuinely divergent MT).

    FA04, an English query (the passage's own ``omt`` rendering) retrieving its native
    counterpart from the ``original`` index. N = 3 per language: small-N, not a
    statistically powered corpus-wide claim. Sliced per language, never pooled, so a
    language-specific weakness stays visible.

    The fragment is rewritten after every topic, so a walltime kill still leaves an honest,
    explicitly partial result rather than nothing.
    """
    started = time.time()
    lang = res.langs[lang_index]
    ctx = _encoder_ctx(res)
    topics = load_topics(topics_path)

    census = read_shard_parts(res.lang_dir("omt", lang))
    parts = int(census["parts"])
    agreement_parts = sample_parts(
        _rng("b2-parts", lang), parts, AGREEMENT_PARTS_PER_LANG
    )

    report: dict[str, Any] = {
        "source_lang": lang,
        "seed": CORPUS_ACCEPTANCE_SEED,
        "topics": len(topics),
        "fa03": {
            "gate": "measurement-only",
            "comparison": "omt vs omt_opus, same passages, English-vs-English",
            "sampled_parts": agreement_parts,
            "topics_completed": 0,
            "by_leg": {leg: [] for leg in LEGS},
            "note": (
                "no pass/fail threshold exists for these numbers, there are no qrels. "
                "They are a liveness/consistency signal, never a retrieval-quality metric."
            ),
        },
        "fa04": {"gate": "measurement-only", "probes": []},
        "partial": True,
    }

    handles = {
        variant: {
            part: open_shard(res.shard_dir(variant, lang, part), ctx)
            for part in agreement_parts
        }
        for variant in ("omt", "omt_opus")
    }
    for i, topic in enumerate(topics):
        for leg in LEGS:
            for part in agreement_parts:
                stats = shard_agreement(
                    handles["omt"][part],
                    handles["omt_opus"][part],
                    leg,
                    topic.problem_statement,
                    TOP_K,
                )
                report["fa03"]["by_leg"][leg].append(
                    {"topic_id": topic.topic_id, "part": part, **stats}
                )
        report["fa03"]["topics_completed"] = i + 1
        # A sibling file, never `out_path` itself: main() owns that one (it carries Step 0's
        # record), and a walltime kill must leave an honest partial beside it, not clobber it.
        _write_json(out_path.with_suffix(".partial.json"), report)
    del handles

    report["fa03"]["summary"] = {
        leg: _distribution(report["fa03"]["by_leg"][leg]) for leg in LEGS
    }
    report["fa04"] = _cross_lingual(res, ctx, lang)
    report["partial"] = False
    report["wall_clock_s"] = round(time.time() - started, 1)
    _write_json(out_path.with_suffix(".partial.json"), report)
    return report


def _distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean/median/min/max of overlap, RBO and τ_b. Descriptive only."""
    out: dict[str, Any] = {"n": len(rows)}
    for key in ("overlap", "rbo", "tau_b"):
        vals = sorted(float(r[key]) for r in rows)
        if not vals:
            out[key] = None
            continue
        out[key] = {
            "mean": round(sum(vals) / len(vals), 6),
            "median": round(vals[len(vals) // 2], 6),
            "min": round(vals[0], 6),
            "max": round(vals[-1], 6),
        }
    return out


def _cross_lingual(res: _Resolved, ctx: IndexCtx, lang: str) -> dict[str, Any]:
    """FA04 for one language: 3 English queries against the whole ``original`` cell.

    The fan over the cell's parts is ``query_lang_leg``'s own documented merge, each part
    asked for its own top-k, merged by the total order ``(-score, passage_id)``, applied
    one part at a time so the reader state is released between parts. That is a memory
    bound, not a different algorithm: a resident 20-part Spanish Seismic fan is ~42 GB.
    """
    census = read_shard_parts(res.lang_dir("original", lang))
    parts = int(census["parts"])
    rng = _rng("b2-xling", lang)
    picks = [
        (part, rng.randrange(int(census["passages"][part])))
        for part in sample_parts(rng, parts, CROSS_LINGUAL_PROBES_PER_LANG)
    ][:CROSS_LINGUAL_PROBES_PER_LANG]

    # The probes' ids + owning documents, read from the original cell's own idmaps.
    wanted: list[dict[str, Any]] = []
    for part, ordinal in picks:
        shard = res.shard_dir("original", lang, part)
        ids = _idmap_ids(shard / DENSE_LEG)
        docs = [
            row["document_id"]
            for row in iter_parquet(
                shard / DENSE_LEG / IDMAP_FILENAME,
                columns=["ordinal", "document_id"],
            )
        ]
        wanted.append(
            {
                "part": part,
                "ordinal": ordinal,
                "passage_id": ids[ordinal],
                "document_id": docs[ordinal],
            }
        )

    english = _english_renderings(res, [w["document_id"] for w in wanted])
    out: dict[str, Any] = {
        "gate": "measurement-only",
        "source_lang": lang,
        "n": len(wanted),
        "probes": [],
        "note": (
            "N=3 per language, small-N, not a statistically powered corpus-wide claim. "
            "Sliced per language and never pooled: a language-specific weakness (the "
            "Chinese cross-lingual margin is the known tight one) must stay visible."
        ),
    }
    # Encode once per (probe, leg): 9 forward passes, then fan part-outer so each part's
    # ~2.7 GB of reader state is loaded once and serves all 9 queries before it is released.
    # Part-inner would re-load it 9 times: ~57 GB of BeeGFS reads become ~510 GB.
    impls = {impl.name: impl for impl in ctx.legs}
    live = [p for p in wanted if english.get(p["passage_id"])]
    for probe in wanted:
        if not english.get(probe["passage_id"]):
            out["probes"].append({**probe, "error": "no omt rendering composed"})
    reps = {
        (i, leg): impls[leg].encode_query(ctx, english[p["passage_id"]])
        for i, p in enumerate(live)
        for leg in LEGS
    }
    merged: dict[tuple[int, str], list[tuple[str, float]]] = {k: [] for k in reps}
    for part in range(parts):
        handle = open_shard(res.shard_dir("original", lang, part), ctx)
        for key, rep in reps.items():
            merged[key].extend(search_with_rep(handle, key[1], rep, TOP_K))
        del handle
    for i, probe in enumerate(live):
        ranks: dict[str, Any] = {}
        for leg in LEGS:
            hits = sorted(merged[(i, leg)], key=lambda kv: (-kv[1], kv[0]))[:TOP_K]
            ids = [pid for pid, _ in hits]
            ranks[leg] = ids.index(probe["passage_id"]) if probe["passage_id"] in ids else -1
        out["probes"].append(
            {
                **probe,
                "english_query_chars": len(english[probe["passage_id"]]),
                "rank": ranks,
            }
        )
    return out


def _english_renderings(res: _Resolved, document_ids: Sequence[str]) -> dict[str, str]:
    """``passage_id -> omt (English) text`` for a handful of documents.

    One column-pruned scan of ``documents.parquet`` to find the documents' row ordinals,
    then :func:`~ragtime.common.passage_store.plan_final_ranges` +
    :func:`~ragtime.common.passage_store.iter_final_passages` with a ``document_range``, the
    pushdown, so this reads a few documents' row groups, not the corpus.
    """
    wanted = set(document_ids)
    rows: dict[str, int] = {}
    for i, row in enumerate(
        iter_parquet(res.layout.documents_path(), columns=["document_id"])
    ):
        if row["document_id"] in wanted:
            rows[str(row["document_id"])] = i
            if len(rows) == len(wanted):
                break
    # deduplicated: plan_final_ranges requires non-decreasing, non-overlapping ranges, and
    # two probes legitimately landing in one document would otherwise raise.
    ordered = sorted(set(rows.values()))
    ranges = plan_final_ranges(
        res.layout,
        res.recon_hash,
        pack_hash=res.pack_hash,
        document_ranges=[(i, i + 1) for i in ordered],
        renderings=("original", "omt"),
    )
    out: dict[str, str] = {}
    for rng_ in ranges:
        for record in iter_final_passages(
            res.layout,
            res.recon_hash,
            pack_hash=res.pack_hash,
            renderings=("original", "omt"),
            document_range=rng_,
        ):
            out[str(record["passage_id"])] = str(record.get("omt") or "")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragtime-acceptance", description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--phase", required=True, choices=["A", "B1", "B2", "step0"])
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="array-task index: the cell (B1) or the non-English language (B2)",
    )
    p.add_argument("--out", required=True, help="where to write this task's JSON fragment")
    p.add_argument("--topics", default=CANONICAL_TOPICS_REL)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    res = resolve(args.config)

    # Step 0 is not skippable. It runs in every phase, before every check.
    fragment: dict[str, Any] = {
        "phase": args.phase,
        "index": args.index,
        "config": str(args.config),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
        "seed": CORPUS_ACCEPTANCE_SEED,
    }
    fragment["step0"] = step0_precondition(res)
    _write_json(out, fragment)

    if args.phase == "step0":
        pass
    elif args.phase == "A":
        fragment["result"] = phase_a(res)
    elif args.phase == "B1":
        if args.index is None:
            raise SystemExit("--index is required for phase B1 (the cell)")
        fragment["result"] = phase_b1(res, int(args.index))
    else:
        if args.index is None:
            raise SystemExit("--index is required for phase B2 (the language)")
        fragment["result"] = phase_b2(res, int(args.index), args.topics, out)
    _write_json(out, fragment)

    failures = _hard_failures(fragment)
    if failures:
        for line in failures:
            _log.error("acceptance.hard_gate_failed", detail=line)
        return 1
    return 0


def _hard_failures(fragment: Mapping[str, Any]) -> list[str]:
    """Only hard gates set the exit code: FA01-dense, FA02, FA05, FA06 (and Step 0).

    FA01-sparse/late, FA03 and FA04 are recorded measurements, they can never turn this
    process red, because there is no bar for them to fall below.
    """
    result = fragment.get("result") or {}
    out: list[str] = []
    for key in ("fa02", "fa06", "fa05"):
        section = result.get(key) or {}
        out.extend(str(f) for f in section.get("failures", ()))
    fa01 = result.get("fa01") or {}
    if fa01 and not fa01.get("dense_hard_gate_passed", True):
        misses = (fa01.get("by_leg", {}).get(DENSE_LEG, {}) or {}).get("misses", [])
        out.append(f"FA01-dense: {len(misses)} probe(s) not at rank 0: {misses[:5]}")
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
